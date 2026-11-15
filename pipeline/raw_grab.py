"""
Pull the upstream raw data of every fire, one channel at a time, into a Hugging Face dataset repo.
Built for Kaggle (notebooks/kaggle_data_grab.ipynb) but runs anywhere with internet access.

  python pipeline/raw_grab.py --channel firms --hf-repo you/wildfire-raw-2014-2025 --work /tmp/raw
  python pipeline/raw_grab.py --channel goes  --hf-repo you/wildfire-raw-2014-2025 --max-hours 11.5
  python pipeline/raw_grab.py --status --hf-repo you/wildfire-raw-2014-2025
  python pipeline/raw_grab.py --channel history --ids F2003403265 --dry-run --work /tmp/raw   # local test

How a run works
  1. The fire list comes from the built dataset (--source-repo): its catalog rows, and by default only fires
     that have a built folder (1,841). Each fire's box and windows are the build's own (make_grid, days_of).
  2. Resume: fires already in manifest/<channel>/*.csv of the target repo (ok / empty / n/a) are skipped;
     errors are retried.
  3. Each fire is fetched into a staging folder, then moved into the current batch. A batch is uploaded as one
     commit when it reaches --batch-gb or --batch-fires, when disk runs low, or before the deadline, together
     with its manifest file; the local copy is deleted right after. A failed upload keeps the batch on disk.
  4. One-time shared downloads (archive geodatabases, ICS-209-PLUS, OSM snapshots) are deleted when the
     channel ends (OSM: after each region-year).

Target repo layout
  README.md                          dataset card (docs/RAW_DATA_CARD.md)
  catalog/fires.csv, qc_fires.csv    copied from the built dataset
  fires/<fire_id>/<channel>/...      raw files
  manifest/<channel>/<batch>.csv     fire_id, status, files, bytes, sources, fetched_utc, error
  shared/ics209/...                  source files used by many fires, stored once

Needs HF_TOKEN (write access) unless --dry-run; FIRMS_MAP_KEY for firms; LANDFIRE_EMAIL for landfire.
"""
import argparse
import io
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import raw_channels as RC  # noqa: E402

SOURCE_REPO = "Javyn123/wildfire-spread-2014-2025"
DONE = {"ok", "empty", "n/a"}
MANIFEST_COLS = ["fire_id", "name", "year", "channel", "status", "n_files", "bytes", "files", "sources", "fetched_utc", "error"]


# ------------------------------------------------------------------------------------- fire list

def load_fires(source_repo, which="built", ids=None, cache=None):
    """Fires as dicts: fire_id, incident, name, year, first, last, bounds (build grid box), rating."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    import build_multiyear as BM
    kw = dict(repo_type="dataset", local_dir=str(cache) if cache else None)
    cat = pd.read_csv(hf_hub_download(source_repo, "catalog/fires.csv", **kw))
    qc = pd.read_csv(hf_hub_download(source_repo, "catalog/qc_fires.csv", **kw), usecols=["fire_id", "rating"])
    cat = cat.merge(qc, on="fire_id", how="left")
    if which == "built":
        built = {p.path.split("/")[-1] for p in HfApi().list_repo_tree(source_repo, path_in_repo="fires", repo_type="dataset")}
        cat = cat[cat.fire_id.isin(built)]
    elif which == "keep":
        cat = cat[cat.rating == "keep"]
    if ids:
        cat = cat[cat.fire_id.isin(set(ids))]
    fires = []
    for _, row in cat.iterrows():
        g = BM.make_grid(row)
        fires.append(dict(fire_id=row.fire_id, incident=str(row.incident), name=str(row["name"]), year=int(row.year),
                          first=str(row["first"]), last=str(row["last"]), bounds=list(g["bounds_lonlat"]),
                          rating=str(row.rating)))
    return fires


# ------------------------------------------------------------------------------------- manifests

def read_manifests(api, repo, channel=None):
    """All manifest rows of the target repo (one channel, or all), as a DataFrame."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    path = f"manifest/{channel}" if channel else "manifest"
    try:
        files = [p.path for p in api.list_repo_tree(repo, path_in_repo=path, recursive=True, repo_type="dataset")
                 if p.path.endswith(".csv")]
    except (EntryNotFoundError, RepositoryNotFoundError):
        return pd.DataFrame(columns=MANIFEST_COLS)
    except Exception as ex:  # noqa: BLE001  (folder not created yet)
        if "404" in str(ex):
            return pd.DataFrame(columns=MANIFEST_COLS)
        raise
    frames = [pd.read_csv(hf_hub_download(repo, f, repo_type="dataset", token=api.token), dtype=str, keep_default_na=False)
              for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MANIFEST_COLS)


def done_ids(manifest):
    """Fires finished in a channel: the latest row per fire has a final status (errors are retried)."""
    if manifest is None or not len(manifest):
        return set()
    last = manifest.sort_values("fetched_utc").groupby("fire_id").tail(1)
    return set(last.loc[last.status.isin(DONE), "fire_id"])


# ------------------------------------------------------------------------------------- batches

class Batcher:
    """Collects finished fires into a folder laid out like the repo; upload() commits it and deletes it."""

    def __init__(self, api, repo, channel, work, batch_gb=2.0, batch_fires=100, min_free_gb=5.0, dry_run=False,
                 keep=False, uploaded_shared=()):
        self.api, self.repo, self.channel = api, repo, channel
        self.root = Path(work) / "batch"
        self.batch_bytes, self.batch_fires, self.min_free = batch_gb * 1e9, batch_fires, min_free_gb * 1e9
        self.dry_run, self.keep = dry_run, keep or dry_run
        self.uploaded_shared = set(uploaded_shared)
        self.rows, self.bytes, self.dir, self.k = [], 0, None, 0
        self.totals = {"ok": 0, "empty": 0, "n/a": 0, "error": 0, "bytes": 0, "batches": 0}

    def _open(self):
        if self.dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.name = f"{self.channel}-{stamp}-{self.k:03d}"
            self.dir = self.root / self.name
            self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def add_shared(self, src, repo_path):
        if repo_path in self.uploaded_shared or not Path(src).exists():
            return
        dst = self._open() / repo_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.bytes += dst.stat().st_size
        self.uploaded_shared.add(repo_path)

    def add(self, fire, status, stage_dir=None, sources=(), error=""):
        files, nbytes = [], 0
        if stage_dir is not None and Path(stage_dir).exists():
            dst = self._open() / "fires" / fire["fire_id"] / self.channel
            if status in ("ok", "empty") and any(Path(stage_dir).iterdir()):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(dst, ignore_errors=True)
                shutil.move(str(stage_dir), str(dst))
                files = sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file())
                nbytes = sum((dst / f).stat().st_size for f in files)
            else:
                shutil.rmtree(stage_dir, ignore_errors=True)
        self._open()
        self.rows.append(dict(fire_id=fire["fire_id"], name=fire["name"], year=fire["year"], channel=self.channel,
                              status=status, n_files=len(files), bytes=nbytes, files=";".join(files),
                              sources=";".join(RC.redact(s) for s in sources),
                              fetched_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              error=RC.redact(error)[:500]))       # manifests are published: no keys, no email
        self.bytes += nbytes
        self.totals[status if status in self.totals else "error"] += 1
        self.totals["bytes"] += nbytes
        if len(self.rows) >= self.batch_fires or self.bytes >= self.batch_bytes or self.low_disk():
            self.upload()

    def low_disk(self):
        return shutil.disk_usage(self.root if self.root.exists() else self.root.parent).free < self.min_free

    def upload(self):
        """Commit the batch (files + its manifest) and delete it locally. No-op when empty."""
        import pandas as pd
        if self.dir is None or not self.rows:
            return
        mdir = self.dir / "manifest" / self.channel
        mdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows, columns=MANIFEST_COLS).to_csv(mdir / f"{self.name}.csv", index=False)
        n_ok = sum(r["status"] == "ok" for r in self.rows)
        msg = f"raw {self.channel}: {len(self.rows)} fires ({n_ok} ok), {self.bytes / 1e9:.2f} GB"
        if not self.dry_run:
            for attempt in range(6):
                try:
                    self.api.upload_folder(repo_id=self.repo, folder_path=str(self.dir), repo_type="dataset", commit_message=msg)
                    break
                except Exception as ex:  # noqa: BLE001  (rate limits, transient 5xx)
                    if attempt == 5:
                        raise RuntimeError(f"upload failed, batch kept at {self.dir}: {ex}") from ex
                    wait = 30 * 2 ** attempt
                    print(f"  upload retry {attempt + 1} in {wait}s: {str(ex)[:150]}", flush=True)
                    time.sleep(wait)
        print(f"  {'[dry-run] ' if self.dry_run else ''}{msg}" + (f" -> kept {self.dir}" if self.keep else " -> uploaded, local copy deleted"), flush=True)
        self.totals["batches"] += 1
        if not self.keep:
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir, self.rows, self.bytes = None, [], 0
        self.k += 1


# ------------------------------------------------------------------------------------- repo setup

def ensure_repo(api, repo, private, source_repo, cache):
    """Create the target repo; on first use add the dataset card and the catalog it was built from."""
    from huggingface_hub import CommitOperationAdd, hf_hub_download
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    have = {p.path for p in api.list_repo_tree(repo, repo_type="dataset")}
    if "README.md" in have:
        return
    card = (HERE.parent / "docs" / "RAW_DATA_CARD.md").read_text().replace("{SOURCE_REPO}", source_repo)
    ops = [CommitOperationAdd("README.md", io.BytesIO(card.encode()))]
    for f in ("catalog/fires.csv", "catalog/qc_fires.csv"):
        ops.append(CommitOperationAdd(f, hf_hub_download(source_repo, f, repo_type="dataset", local_dir=str(cache))))
    api.create_commit(repo, ops, repo_type="dataset", commit_message="Dataset card and catalog")


def shared_in_repo(api, repo):
    try:
        return {p.path for p in api.list_repo_tree(repo, path_in_repo="shared", recursive=True, repo_type="dataset")}
    except Exception:  # noqa: BLE001
        return set()


# ------------------------------------------------------------------------------------- run

def run(a):
    from huggingface_hub import HfApi
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    deadline = time.time() + a.max_hours * 3600 if a.max_hours else None
    ch = a.channel
    RC.requirements(ch)

    fires = load_fires(a.source_repo, a.fires, a.ids, work / "source")
    print(f"{len(fires)} fires ({a.fires}{', ' + str(len(a.ids)) + ' ids' if a.ids else ''})", flush=True)
    if not a.dry_run:
        ensure_repo(api, a.hf_repo, a.private, a.source_repo, work / "source")
    manifest = read_manifests(api, a.hf_repo, ch) if (a.hf_repo and not a.redo) else None
    done = done_ids(manifest)
    todo = [f for f in fires if f["fire_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"channel {ch}: {len(done)} already done, {len(todo)} to do", flush=True)

    ctx = RC.Ctx(ch, work, deadline=deadline, threads=a.threads)
    b = Batcher(api, a.hf_repo, ch, work, a.batch_gb, a.batch_fires, a.min_free_gb, a.dry_run, a.keep,
                shared_in_repo(api, a.hf_repo) if (a.hf_repo and not a.dry_run) else ())
    t0 = time.time()
    try:
        runnable = []
        for f in todo:
            why = RC.applies(ch, f)
            if why:
                b.add(f, "n/a", error=why)
            else:
                runnable.append(f)
        if runnable:
            for src, repo_path in RC.shared_files(ch, ctx):
                b.add_shared(src, repo_path)
        if ch in RC.GROUP:
            results = RC.GROUP_FUNCS[ch](runnable, ctx)
            for k, (f, status, sources, err) in enumerate(results, 1):
                err = RC.redact(err)
                stage = ctx.stage_root / f["fire_id"] / ch
                b.add(f, status, stage, sources, err)
                ctx.drop_stage(f["fire_id"])
                print(f"[{k}/{len(runnable)}] {f['fire_id']} {f['name']} ({f['year']}): {status}"
                      + (f" {err[:120]}" if err else "") + f"  ({(time.time() - t0) / 60:.0f} min)", flush=True)
        else:
            fn = RC.PER_FIRE[ch]
            for k, f in enumerate(runnable, 1):
                if ctx.expired():
                    print("deadline reached: stopping; rerun to continue", flush=True)
                    break
                stage = ctx.stage(f["fire_id"])
                try:
                    status, sources = fn(f, stage, ctx)
                    err = ""
                except Exception as ex:  # noqa: BLE001
                    status, sources, err = "error", [], RC.redact(f"{type(ex).__name__}: {ex}")
                    if a.verbose:
                        print(RC.redact(traceback.format_exc()), flush=True)
                b.add(f, status, stage, sources, err)
                ctx.drop_stage(f["fire_id"])
                print(f"[{k}/{len(runnable)}] {f['fire_id']} {f['name']} ({f['year']}): {status}"
                      + (f" {err[:120]}" if err else "") + f"  ({(time.time() - t0) / 60:.0f} min)", flush=True)
        b.upload()
    finally:
        if not a.keep_cache:
            ctx.cleanup()
    t = b.totals
    print(f"done: {t['ok']} ok, {t['empty']} empty, {t['n/a']} n/a, {t['error']} error; {t['bytes'] / 1e9:.2f} GB "
          f"in {t['batches']} batch(es), {(time.time() - t0) / 60:.0f} min", flush=True)
    return 0


def status(a):
    import pandas as pd
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    m = read_manifests(api, a.hf_repo)
    if not len(m):
        print("no manifests yet")
        return 0
    m["bytes"] = pd.to_numeric(m["bytes"], errors="coerce").fillna(0)
    last = m.sort_values("fetched_utc").groupby(["channel", "fire_id"]).tail(1)
    tab = last.pivot_table(index="channel", columns="status", values="fire_id", aggfunc="count", fill_value=0)
    tab["GB"] = last.groupby("channel")["bytes"].sum() / 1e9
    print(tab.reindex([c for c in RC.CHANNELS if c in tab.index]).round(2).to_string())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", choices=RC.CHANNELS)
    ap.add_argument("--status", action="store_true", help="per-channel progress of --hf-repo")
    ap.add_argument("--hf-repo", help="target dataset repo, e.g. you/wildfire-raw-2014-2025")
    ap.add_argument("--source-repo", default=SOURCE_REPO, help="built dataset the fire list comes from")
    ap.add_argument("--work", default="raw_work", help="local scratch (staging, batches, one-time downloads)")
    ap.add_argument("--fires", choices=["built", "keep", "all"], default="built")
    ap.add_argument("--ids", nargs="*", help="only these fire ids")
    ap.add_argument("--limit", type=int, help="at most N fires this run")
    ap.add_argument("--batch-gb", type=float, default=2.0)
    ap.add_argument("--batch-fires", type=int, default=100)
    ap.add_argument("--min-free-gb", type=float, default=5.0, help="upload early when free disk falls below this")
    ap.add_argument("--max-hours", type=float, help="stop cleanly before this many hours (Kaggle: 11.5)")
    ap.add_argument("--threads", type=int, default=16, help="parallel downloads for goes / hrrr")
    ap.add_argument("--private", action="store_true", help="create the target repo private")
    ap.add_argument("--redo", action="store_true", help="ignore manifests: fetch every fire again")
    ap.add_argument("--dry-run", action="store_true", help="no upload; batches are kept under WORK/batch")
    ap.add_argument("--keep", action="store_true", help="keep local batches after upload")
    ap.add_argument("--keep-cache", action="store_true", help="keep one-time downloads at the end")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.status:
        if not a.hf_repo:
            ap.error("--status needs --hf-repo")
        return status(a)
    if not a.channel:
        ap.error("--channel is required")
    if not a.hf_repo and not a.dry_run:
        ap.error("--hf-repo is required (or --dry-run)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
