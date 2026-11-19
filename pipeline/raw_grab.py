"""
Pull the upstream raw data of every fire, one channel at a time, into a Hugging Face dataset repo.
Built for Kaggle (notebooks/kaggle_data_grab.ipynb) but runs anywhere with internet access.

  python pipeline/raw_grab.py --channel firms --hf-repo you/wildfire-raw-2014-2025 --work /tmp/raw
  python pipeline/raw_grab.py --channel goes  --hf-repo you/wildfire-raw-2014-2025 --max-hours 11.5
  python pipeline/raw_grab.py --status --hf-repo you/wildfire-raw-2014-2025
  python pipeline/raw_grab.py --links  --hf-repo you/wildfire-raw-2014-2025          # per-fire links files
  python pipeline/raw_grab.py --extents --hf-repo you/wildfire-raw-2014-2025         # after firms: windows, ignition
  python pipeline/raw_grab.py --repack --channel perimeters --hf-repo you/wildfire-raw-2014-2025  # gzip old GeoJSON
  python pipeline/raw_grab.py --channel history --ids F2003403265 --dry-run --work /tmp/raw   # local test

How a run works
  1. The fire list comes from the built dataset (--source-repo): its catalog rows, and by default only fires
     that have a built folder (1,841). Each fire's region is the box around all its perimeters pushed out
     32 km on every side (raw_channels.area), so any 64 km tile centred anywhere in the fire can be cut from
     the files. Its time window is the perimeter days stretched over the FIRMS activity that leads into and
     out of them, plus a day on each side: run firms first, then --extents, which writes
     catalog/fire_extents.csv (window, ignition point, region); the time-dependent channels (gridmet, goes,
     hrrr, hrrr_next, gfs, vegetation, viirs) read it.
  2. Resume: fires already in manifest/<channel>/*.csv of the target repo (ok / empty / n/a) are skipped;
     errors are retried. A row also has to match the fire's current window and region (columns window,
     region), so widening either sends the fire back to the queue.
  3. Each fire is fetched into a staging folder, then moved into the current batch. A batch is uploaded as one
     commit when it reaches --batch-gb or --batch-fires, when disk runs low, or before the deadline, together
     with its manifest file; the local copy is deleted right after. A failed upload keeps the batch on disk.
     A fire fetched again replaces its folder: the same commit deletes the channel's older files of that fire.
  4. Files: GeoJSON is gzipped with a fixed header (.geojson.gz); viirs is one viirs.nc per fire. Rows that
     list the older names (plain .geojson) count as not done; --repack gzips such files in place instead of
     fetching again (perimeters: it keeps the build box, so the region change does not refetch it).
  5. One-time shared downloads (archive geodatabases, ICS-209-PLUS, OSM snapshots) are deleted when the
     channel ends (OSM: after each region-year).

Target repo layout
  README.md                          dataset card (docs/RAW_DATA_CARD.md)
  catalog/fires.csv, qc_fires.csv    copied from the built dataset
  catalog/raw_index.csv              --links: one row per fire and channel (status, files, bytes)
  catalog/fire_extents.csv           --extents: each fire's time window, ignition point and region
  fires/<fire_id>/<channel>/...      raw files
  fires/<fire_id>/links.json, LINKS.md   --links: every channel of the fire, its files here and its upstream URLs
  manifest/<channel>/<batch>.csv     fire_id, status, files, bytes, sources, fetched_utc, error, window, region
  shared/ics209/...                  source files used by many fires, stored once

Needs HF_TOKEN (write access) unless --dry-run, FIRMS_MAP_KEY for firms, EARTHDATA_TOKEN (free Earthdata
Login token) for viirs, vegetation and landcover, and Google Earth Engine access for gfs (GEE_SERVICE_ACCOUNT_KEY,
the service account's JSON key, or `earthengine authenticate` on this machine; GEE_PROJECT, the Cloud project
registered for Earth Engine). No other keys.
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

import raw_channels as RC  # noqa: E402

SOURCE_REPO = "Javyn123/wildfire-spread-2014-2025"
DONE = {"ok", "empty", "n/a"}
SEP = " | "                      # between source URLs in a manifest cell (URLs may hold ';' and ',')
MANIFEST_COLS = ["fire_id", "name", "year", "channel", "status", "n_files", "bytes", "files", "sources", "fetched_utc", "error",
                 "window", "region"]  # the days / box the channel covered (raw_channels.window_label / region_label)


# ------------------------------------------------------------------------------------- fire list

GRID_RES_M = 150.0               # the built dataset's cell size and margin: its grid box is each fire's "bounds"
GRID_PAD_KM = 9.6


def build_grid(row):
    """The built dataset's grid for a catalog row: the perimeters' box plus GRID_PAD_KM, on GRID_RES_M cells
    (north-up lon/lat). -> {"transform": [dx, 0, x0, 0, -dy, y0], "bounds_lonlat": [w, s, e, n]}"""
    import math
    pad = int(round(GRID_PAD_KM * 1000 / GRID_RES_M))
    lat0 = (row.south + row.north) / 2
    dy = GRID_RES_M / 110570.0
    dx = GRID_RES_M / (111320.0 * math.cos(math.radians(lat0)))
    w = int(math.ceil((row.east - row.west) / dx)) + 2 * pad
    h = int(math.ceil((row.north - row.south) / dy)) + 2 * pad
    x0, y0 = row.west - pad * dx, row.north + pad * dy
    return {"transform": [dx, 0.0, x0, 0.0, -dy, y0], "bounds_lonlat": [x0, y0 - h * dy, x0 + w * dx, y0]}


def load_fires(source_repo, which="built", ids=None, cache=None):
    """Fires as dicts: fire_id, incident, name, year, first, last, bounds (build grid box), grid_transform,
    perim_bounds (the perimeters' own box), rating."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
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
        g = build_grid(row)
        fires.append(dict(fire_id=row.fire_id, incident=str(row.incident), name=str(row["name"]), year=int(row.year),
                          first=str(row["first"]), last=str(row["last"]), bounds=list(g["bounds_lonlat"]),
                          grid_transform=list(g["transform"]),
                          perim_bounds=[float(row[k]) for k in ("west", "south", "east", "north")]
                          if all(k in row and pd.notna(row[k]) for k in ("west", "south", "east", "north")) else None,
                          rating=str(row.rating)))
    return fires


def load_extents(api, repo, cache=None, path=None):
    """{fire_id: row} of catalog/fire_extents.csv (from a local file, or the target repo), or None if absent."""
    import pandas as pd
    if path is None:
        if not repo:
            return None
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(repo, RC.EXTENTS_FILE, repo_type="dataset", token=api.token,
                                   local_dir=str(cache) if cache else None)
        except Exception as ex:  # noqa: BLE001  (not written yet)
            if "404" in str(ex) or "EntryNotFound" in type(ex).__name__ or "RepositoryNotFound" in type(ex).__name__:
                return None
            raise
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {r["fire_id"]: r for r in t.to_dict("records")}


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


def done_ids(manifest, channel=None, fires=None):
    """Fires finished in a channel: the latest row per fire has a final status (errors are retried). With a
    channel that lists expected files (raw_channels.expected_files), an 'ok' row must name all of them. Given
    the fires, the row must also have been fetched for the fire's current window (time-dependent channels) and
    region (channels cut to the region): rows from before these rules, or for a window or region since
    widened, count as not done."""
    import pandas as pd
    if manifest is None or not len(manifest):
        return set()
    last = manifest.sort_values("fetched_utc").groupby("fire_id").tail(1)
    done = last.status.isin(DONE)
    need = RC.expected_files(channel) if channel else None
    if need:
        col = last["files"] if "files" in last else pd.Series("", index=last.index)
        listed = col.fillna("").astype(str).map(lambda s: set(s.split(";")))
        done &= (last.status != "ok") | listed.map(need.issubset)
    for col, chans, label in (("window", RC.WINDOW_CHANNELS, RC.window_label), ("region", RC.REGION_CHANNELS, RC.region_label)):
        if channel in chans and fires is not None:
            want = {f["fire_id"]: label(channel, f) for f in fires}
            have = last[col].fillna("").astype(str) if col in last else pd.Series("", index=last.index)
            done &= have == last["fire_id"].map(want)
    return set(last.loc[done, "fire_id"])


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
        self.replace = []           # fires/<id>/<channel>/* of the fires in this batch: older files are deleted
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

    def add(self, fire, status, stage_dir=None, sources=(), error="", labels=None):
        """labels: (window, region) to record instead of the fire's current ones (--repack keeps the old)."""
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
                              sources=SEP.join(RC.redact(s) for s in sources),
                              fetched_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              error=RC.redact(error)[:500],        # manifests are published: no keys, no email
                              **dict(zip(("window", "region"), labels or (RC.window_label(self.channel, fire),
                                                                          RC.region_label(self.channel, fire))))))
        if status in ("ok", "empty"):            # a new result: whatever this fire had in the channel before goes
            self.replace.append(f"fires/{fire['fire_id']}/{self.channel}/*")
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
                    self.api.upload_folder(repo_id=self.repo, folder_path=str(self.dir), repo_type="dataset",
                                           commit_message=msg, delete_patterns=self.replace or None)
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
        self.dir, self.rows, self.bytes, self.replace = None, [], 0, []
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


SETUP_ERROR = 3          # exit code for a setup problem (token, repo name): retrying other channels cannot help
CHANNEL_BROKEN = 4       # exit code when a channel fails for many fires in a row: other channels may still work


class ErrorStreak:
    """Counts errors in a row; trips at `limit` (0 = never) so a broken source does not burn through
    every fire. Any fire that is not an error resets it."""

    def __init__(self, limit):
        self.limit, self.n, self.last, self.tripped = limit, 0, "", False

    def hit(self, status, err):
        if status != "error":
            self.n = 0
            return False
        self.n, self.last = self.n + 1, err
        self.tripped = bool(self.limit) and self.n >= self.limit
        return self.tripped


def setup_error(msg):
    print(f"SETUP ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(SETUP_ERROR)


def check_hf(api, repo):
    """Fail fast, with one clear line, when this token cannot write to `repo`. Returns the account name."""
    if not repo or "/" not in repo:
        setup_error(f"--hf-repo must look like 'account/repo-name', got {repo!r}")
    try:
        me = api.whoami()                      # HF_TOKEN, else a cached `hf auth login`
    except Exception as ex:  # noqa: BLE001
        if not api.token:
            setup_error("HF_TOKEN is not set. On Kaggle: Add-ons > Secrets > add a WRITE token from "
                        "https://huggingface.co/settings/tokens under the label SECRET_LABELS names (default HF_TOKEN_ALL), "
                        "switch it on for this notebook, then rerun the secrets cell.")
        setup_error(f"Hugging Face rejected HF_TOKEN ({type(ex).__name__}: {RC.redact(str(ex))[:120]}). "
                    "Make a new WRITE token at https://huggingface.co/settings/tokens and update the Kaggle secret.")
    names = {me.get("name")} | {o.get("name") for o in me.get("orgs") or []}
    names.discard(None)
    ns = repo.split("/")[0]
    if ns not in names:
        setup_error(f"HF_REPO {repo!r}: '{ns}' is not your account or one of your organisations ({', '.join(sorted(names))}). "
                    f"Use e.g. HF_REPO = \"{me.get('name')}/wildfire-raw-2014-2025\".")
    role = ((me.get("auth") or {}).get("accessToken") or {}).get("role")
    if role == "read":
        setup_error("HF_TOKEN is read-only. Make a token with write access at https://huggingface.co/settings/tokens.")
    return me.get("name")


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
    if not a.dry_run:
        check_hf(api, a.hf_repo)
    RC.requirements(ch)

    fires = load_fires(a.source_repo, a.fires, a.ids, work / "source")
    print(f"{len(fires)} fires ({a.fires}{', ' + str(len(a.ids)) + ' ids' if a.ids else ''})", flush=True)
    if ch in RC.WINDOW_CHANNELS and ch != "firms":
        extents = load_extents(api, a.hf_repo, work / "target", a.extents_file)
        if extents is None:
            msg = (f"{RC.EXTENTS_FILE} not found: run the firms channel, then raw_grab.py --extents, so {ch} "
                   "covers the days the fire was active (not only its perimeter days)")
            if not a.dry_run:
                raise SystemExit(msg)
            print(f"warning: {msg}; using the perimeter days +- 1", flush=True)
        RC.attach_extents(fires, extents)
        print(f"time window: {sum('window' in f for f in fires)} of {len(fires)} fires from {RC.EXTENTS_FILE}", flush=True)
    if not a.dry_run:
        ensure_repo(api, a.hf_repo, a.private, a.source_repo, work / "source")
    manifest = read_manifests(api, a.hf_repo, ch) if (a.hf_repo and not a.redo) else None
    done = done_ids(manifest, ch, fires)
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
        streak = ErrorStreak(a.max_errors_in_row)
        if ch in RC.GROUP:
            results = RC.GROUP_FUNCS[ch](runnable, ctx)
            for k, (f, status, sources, err) in enumerate(results, 1):
                err = RC.redact(err)
                stage = ctx.stage_root / f["fire_id"] / ch
                b.add(f, status, stage, sources, err)
                ctx.drop_stage(f["fire_id"])
                print(f"[{k}/{len(runnable)}] {f['fire_id']} {f['name']} ({f['year']}): {status}"
                      + (f" {err[:120]}" if err else "") + f"  ({(time.time() - t0) / 60:.0f} min)", flush=True)
                if streak.hit(status, err):
                    break
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
                if streak.hit(status, err):
                    break
        b.upload()
    finally:
        if not a.keep_cache:
            ctx.cleanup()
    t = b.totals
    if streak.tripped:
        print(f"CHANNEL STOPPED: {ch} failed for {streak.n} fires in a row, so the source or this machine's access to it "
              f"is probably broken, not single fires. Last error: {streak.last[:300]}\n"
              f"Those fires are recorded as errors and retried on the next run.", file=sys.stderr, flush=True)
        return CHANNEL_BROKEN
    print(f"done: {t['ok']} ok, {t['empty']} empty, {t['n/a']} n/a, {t['error']} error; {t['bytes'] / 1e9:.2f} GB "
          f"in {t['batches']} batch(es), {(time.time() - t0) / 60:.0f} min", flush=True)
    return 0


# ------------------------------------------------------------------------------------- per-fire links

def fire_links(fire, rows, repo, source_repo=SOURCE_REPO):
    """links.json for one fire: for every channel its status, its files in this repo and the upstream
    URLs they came from (latest manifest row per channel; channels not run yet are 'pending')."""
    fid = fire["fire_id"]
    base = f"https://huggingface.co/datasets/{repo}" if repo else "."
    latest = {}
    for r in sorted(rows, key=lambda r: str(r.get("fetched_utc", ""))):
        latest[r["channel"]] = r
    chans = {}
    for ch in RC.CHANNELS:
        r = latest.get(ch)
        if r is None:
            chans[ch] = {"status": "pending"}
            continue
        e = {"status": r["status"], "fetched_utc": r.get("fetched_utc", "")}
        files = [x for x in str(r.get("files") or "").split(";") if x]
        if files:
            e["bytes"] = int(float(r.get("bytes") or 0))
            e["files"] = [{"name": x, "url": f"{base}/resolve/main/fires/{fid}/{ch}/{x}"} for x in files]
        src = [s.strip() for s in str(r.get("sources") or "").split(SEP) if s.strip()]
        if src:
            e["sources"] = src
        if r.get("error"):
            e["note"] = r["error"]
        chans[ch] = e
    return {"fire_id": fid, "name": fire["name"], "year": fire["year"], "irwin_incident": RC.norm_irwin(fire["incident"]),
            "first_perimeter_utc": fire["first"], "last_perimeter_utc": fire["last"],
            "time_window_utc": [t.isoformat() for t in RC.span(fire)],
            "region_lonlat": [round(v, 5) for v in RC.area(fire)], "build_box_lonlat": fire["bounds"],
            "ignition": dict(zip(("lat", "lon", "time_utc", "method"), fire["ignition"])) if fire.get("ignition") else None,
            "folder": f"{base}/tree/main/fires/{fid}",
            "built_dataset": f"https://huggingface.co/datasets/{source_repo}/tree/main/fires/{fid}",
            "channels": chans}


def links_markdown(L):
    """LINKS.md: the same as links.json, readable on the Hugging Face file page."""
    w, s, e, n = L["region_lonlat"]
    t0, t1 = L["time_window_utc"]
    ig = L.get("ignition")
    out = [f"# {L['name']} ({L['year']}) · `{L['fire_id']}`", "",
           f"IRWIN `{L['irwin_incident']}` · perimeters {str(L['first_perimeter_utc'])[:16]} → {str(L['last_perimeter_utc'])[:16]} UTC · "
           f"files cover {t0[:16]} → {t1[:16]} UTC over region W {w:.4f} S {s:.4f} E {e:.4f} N {n:.4f}"
           + (f" · ignition {ig['lat']:.4f}, {ig['lon']:.4f} at {ig['time_utc']} ({ig['method']})" if ig else ""), "",
           f"[This fire's folder]({L['folder']}) · [Built dataset layers]({L['built_dataset']})", "",
           "| channel | status | files | upstream sources |", "|---|---|---|---|"]
    for ch, c in L["channels"].items():
        files = ", ".join(f"[{x['name']}]({x['url']})" for x in c.get("files", [])) or "—"
        src = c.get("sources", [])
        srct = f"[link]({src[0]})" if len(src) == 1 else (f"{len(src)} links (below)" if src else "—")
        status = c["status"] + (f" ({c['note'][:80]})" if c.get("note") and c["status"] != "ok" else "")
        out.append(f"| {ch} | {status} | {files} | {srct} |")
    many = [(ch, c["sources"]) for ch, c in L["channels"].items() if len(c.get("sources", [])) > 1]
    if many:
        out += ["", "## Upstream sources"]
        for ch, src in many:
            out += ["", f"<details><summary>{ch}: {len(src)} links</summary>", ""] + [f"- {u}" for u in src] + ["", "</details>"]
    if any("{MAP_KEY}" in u for _, c in L["channels"].items() for u in c.get("sources", [])):
        out += ["", "FIRMS links need your own map key in place of `{MAP_KEY}` (free: https://firms.modaps.eosdis.nasa.gov/api/map_key/)."]
    return "\n".join(out) + "\n"


def build_links(a):
    """Write fires/<id>/links.json + LINKS.md for every fire, and catalog/raw_index.csv (one row per fire
    and channel), from the manifests; upload them in commits of --links-chunk fires."""
    import json
    import glob
    import pandas as pd
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    work = Path(a.work)
    if not a.dry_run:
        check_hf(api, a.hf_repo)
    fires = load_fires(a.source_repo, a.fires, a.ids, work / "source")
    RC.attach_extents(fires, load_extents(api, a.hf_repo, work / "target", a.extents_file))
    if a.local_manifests:
        files = glob.glob(str(Path(a.local_manifests) / "**" / "manifest" / "*" / "*.csv"), recursive=True)
        m = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True) if files \
            else pd.DataFrame(columns=MANIFEST_COLS)
    else:
        m = read_manifests(api, a.hf_repo)
    by_fire = {fid: g.to_dict("records") for fid, g in m.groupby("fire_id")} if len(m) else {}
    print(f"links for {len(fires)} fires from {len(m)} manifest rows", flush=True)
    root = work / "links"
    shutil.rmtree(root, ignore_errors=True)
    index = []
    for k in range(0, len(fires), a.links_chunk):
        d = root / f"chunk{k // a.links_chunk:03d}"
        for f in fires[k:k + a.links_chunk]:
            L = fire_links(f, by_fire.get(f["fire_id"], []), a.hf_repo, a.source_repo)
            fd = d / "fires" / f["fire_id"]
            fd.mkdir(parents=True, exist_ok=True)
            (fd / "links.json").write_text(json.dumps(L, indent=1))
            (fd / "LINKS.md").write_text(links_markdown(L))
            index += [dict(fire_id=f["fire_id"], name=f["name"], year=f["year"], channel=ch, status=c["status"],
                           n_files=len(c.get("files", [])), bytes=c.get("bytes", 0)) for ch, c in L["channels"].items()]
        if k + a.links_chunk >= len(fires):
            (d / "catalog").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(index).to_csv(d / "catalog" / "raw_index.csv", index=False)
        if not a.dry_run:
            api.upload_folder(repo_id=a.hf_repo, folder_path=str(d), repo_type="dataset",
                              commit_message=f"Per-fire links ({min(k + a.links_chunk, len(fires))}/{len(fires)} fires)")
            shutil.rmtree(d, ignore_errors=True)
    print(("[dry-run] kept " + str(root)) if a.dry_run else "links uploaded", flush=True)
    return 0


def first_perimeters(source_repo, fires, cache):
    """{fire_id: (time, lat, lon)} of each fire's first mapped perimeter, from the built dataset's labels.npz
    (arrival index per cell) on the fire's build grid. Fires without labels are left out."""
    import numpy as np
    from huggingface_hub import snapshot_download
    snapshot_download(source_repo, repo_type="dataset", local_dir=str(cache),
                      allow_patterns=[f"fires/{f['fire_id']}/labels.npz" for f in fires])
    out = {}
    for f in fires:
        p = Path(cache) / "fires" / f["fire_id"] / "labels.npz"
        if p.exists() and f.get("grid_transform"):
            with np.load(p, allow_pickle=False) as z:
                pt = RC.first_perimeter_point(z["arrival"], f["grid_transform"], z["times"])
            if pt is not None:
                out[f["fire_id"]] = pt
    return out


def build_extents(a):
    """catalog/fire_extents.csv: each fire's time window, ignition point and region (raw_channels.fire_extent)
    from its firms files, read from the target repo (or local --dry-run batches with --local-manifests), and
    its first perimeter from the built dataset. Rows of fires not in this run (--ids) are kept."""
    import glob
    import pandas as pd
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    work = Path(a.work)
    if not a.dry_run:
        check_hf(api, a.hf_repo)
    fires = load_fires(a.source_repo, a.fires, a.ids, work / "source")
    if not fires:
        print("no fires matched (--fires / --ids): nothing to do", flush=True)
        return 1
    if a.local_manifests:
        files = glob.glob(str(Path(a.local_manifests) / "**" / "manifest" / "firms" / "*.csv"), recursive=True)
        m = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True) if files \
            else pd.DataFrame(columns=MANIFEST_COLS)
        roots = [Path(p) for p in glob.glob(str(Path(a.local_manifests) / "**" / "fires"), recursive=True)]
    else:
        m = read_manifests(api, a.hf_repo, "firms")
        dl = work / "firms_files"
        snapshot_download(a.hf_repo, repo_type="dataset", local_dir=str(dl), token=api.token,
                          allow_patterns=[f"fires/{f['fire_id']}/firms/*" for f in fires] if a.ids else ["fires/*/firms/*"])
        roots = [dl / "fires"]
    current = done_ids(m, "firms", fires)
    stale = [f["fire_id"] for f in fires if f["fire_id"] not in current]
    firsts = first_perimeters(a.source_repo, fires, work / "source")
    rows = []
    for f in fires:
        paths = [p for r in roots for p in (r / f["fire_id"] / "firms").glob("*.csv.gz")]
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else None
        rows.append(RC.fire_extent(f, df, firsts.get(f["fire_id"])))
    new = pd.DataFrame(rows)
    old = load_extents(api, None if a.dry_run else a.hf_repo, work / "target")
    if old and a.ids:
        keep = pd.DataFrame([r for fid, r in old.items() if fid not in set(new.fire_id)])
        new = pd.concat([keep, new], ignore_index=True)
    new = new.sort_values("fire_id")
    out = work / "extents" / RC.EXTENTS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(out, index=False)
    before = (pd.to_numeric(new.active_days_before) > 0).sum()
    after = (pd.to_numeric(new.active_days_after) > 0).sum()
    print(f"{len(new)} fires: {before} windows start before the first perimeter, {after} end after the last "
          f"(activity), {(new.window_rule == 'perimeters').sum()} without FIRMS data; ignition from "
          + ", ".join(f"{k} {v}" for k, v in new.ignition_method.value_counts().items()), flush=True)
    if stale:
        print(f"warning: {len(stale)} fires have no firms row for the current FIRMS days "
              f"({RC.FIRMS_BEFORE} before / {RC.FIRMS_AFTER} after the perimeters), e.g. {stale[:3]}: rerun the firms "
              "channel, then --extents again, or their windows miss activity outside the old span", flush=True)
    if a.dry_run:
        print(f"[dry-run] kept {out}", flush=True)
    else:
        api.upload_file(path_or_fileobj=str(out), path_in_repo=RC.EXTENTS_FILE, repo_id=a.hf_repo, repo_type="dataset",
                        commit_message=f"Fire extents: window, ignition, region ({len(new)} fires)")
        print(f"uploaded {RC.EXTENTS_FILE}", flush=True)
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


# ------------------------------------------------------------------------------------- repack in place

REPACK_CHANNELS = ("perimeters", "history", "suppression")


def repack(a):
    """Gzip a channel's plain .geojson files already in the repo (fires whose latest row is 'ok' and lists only
    .geojson files): download each, write <name>.geojson.gz with the fixed header, commit it in place of the old
    file with a new manifest row (same sources; window / region of the old row; note: when it was fetched).
    For channels the next run does not fetch again anyway: perimeters keeps the build box."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    ch = a.channel or "perimeters"
    if ch not in REPACK_CHANNELS:
        raise SystemExit(f"--repack works on {', '.join(REPACK_CHANNELS)}, not {ch}")
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    if not a.dry_run:
        check_hf(api, a.hf_repo)
    work = Path(a.work)
    m = read_manifests(api, a.hf_repo, ch)
    if not len(m):
        print(f"repack {ch}: no manifest rows", flush=True)
        return 0
    last = m.sort_values("fetched_utc").groupby("fire_id").tail(1)
    if a.ids:
        last = last[last.fire_id.isin(set(a.ids))]
    names = last["files"].fillna("").astype(str).map(lambda x: [f for f in x.split(";") if f])
    todo = last[(last.status == "ok") & names.map(lambda fs: bool(fs) and all(f.endswith(".geojson") for f in fs))]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"repack {ch}: {len(todo)} of {len(last)} fires have plain .geojson", flush=True)
    b = Batcher(api, a.hf_repo, ch, work, a.batch_gb, a.batch_fires, a.min_free_gb, a.dry_run, a.keep)
    dl, t0 = work / "repack_download", time.time()
    for k, r in enumerate(todo.to_dict("records"), 1):
        stage = work / "repack_stage" / r["fire_id"] / ch
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        try:
            for name in r["files"].split(";"):
                src = hf_hub_download(a.hf_repo, f"fires/{r['fire_id']}/{ch}/{name}", repo_type="dataset",
                                      token=api.token, local_dir=str(dl))
                RC.gzip_copy(src, stage / f"{name}.gz")
                Path(src).unlink()
        except Exception as ex:  # noqa: BLE001  (left as it is; the next --repack tries again)
            shutil.rmtree(stage, ignore_errors=True)
            print(f"[{k}/{len(todo)}] {r['fire_id']}: skipped, {RC.redact(str(ex))[:150]}", flush=True)
            continue
        fire = {"fire_id": r["fire_id"], "name": r.get("name", ""), "year": r.get("year", "")}
        b.add(fire, "ok", stage, [x for x in str(r.get("sources") or "").split(SEP) if x],
              f"gzip of the files fetched {r['fetched_utc']}", labels=(r.get("window", ""), r.get("region", "")))
        print(f"[{k}/{len(todo)}] {r['fire_id']} {r.get('name', '')}: repacked  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    b.upload()
    shutil.rmtree(dl, ignore_errors=True)
    t = b.totals
    print(f"repack {ch}: {t['ok']} fires, {t['bytes'] / 1e9:.2f} GB in {t['batches']} batch(es)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", choices=RC.CHANNELS)
    ap.add_argument("--status", action="store_true", help="per-channel progress of --hf-repo")
    ap.add_argument("--check", action="store_true", help="check HF_TOKEN can write to --hf-repo, and which keys are set")
    ap.add_argument("--links", action="store_true",
                    help="write fires/<id>/links.json + LINKS.md (every channel's files and sources) from the manifests")
    ap.add_argument("--links-chunk", type=int, default=500, help="fires per commit for --links")
    ap.add_argument("--extents", action="store_true",
                    help=f"write {RC.EXTENTS_FILE} (each fire's time window, ignition point, region) from the firms files; run after firms")
    ap.add_argument("--repack", action="store_true",
                    help=f"gzip the plain .geojson files already in --hf-repo for --channel ({', '.join(REPACK_CHANNELS)}; default perimeters)")
    ap.add_argument("--extents-file", help=f"read the extents from this local file instead of {RC.EXTENTS_FILE} in --hf-repo")
    ap.add_argument("--local-manifests", help="--links / --extents: read manifests (and firms files) from local --dry-run batches under this folder")
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
    ap.add_argument("--threads", type=int, default=16, help="parallel downloads for goes / hrrr / hrrr_next / gfs; viirs uses up to 8 worker processes")
    ap.add_argument("--private", action="store_true", help="create the target repo private")
    ap.add_argument("--redo", action="store_true", help="ignore manifests: fetch every fire again")
    ap.add_argument("--dry-run", action="store_true", help="no upload; batches are kept under WORK/batch")
    ap.add_argument("--keep", action="store_true", help="keep local batches after upload")
    ap.add_argument("--keep-cache", action="store_true", help="keep one-time downloads at the end")
    ap.add_argument("--max-errors-in-row", type=int, default=10,
                    help="stop the channel (exit 4) after this many fires fail in a row; 0 = never")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        from huggingface_hub import HfApi
        who = check_hf(HfApi(token=os.environ.get("HF_TOKEN") or None), a.hf_repo)
        keys = {k: ("set" if os.environ.get(k) else "missing (channel skipped)")
                for k in ("FIRMS_MAP_KEY", "EARTHDATA_TOKEN", "GEE_SERVICE_ACCOUNT_KEY", "GEE_PROJECT")}
        print(f"OK: HF_TOKEN can write as '{who}' to {a.hf_repo}; {keys}", flush=True)
        return 0
    if a.status:
        if not a.hf_repo:
            ap.error("--status needs --hf-repo")
        return status(a)
    if a.links:
        if not a.hf_repo and not a.dry_run:
            ap.error("--links needs --hf-repo (or --dry-run)")
        return build_links(a)
    if a.extents:
        if not a.hf_repo and not a.local_manifests:
            ap.error("--extents needs --hf-repo (or --local-manifests with --dry-run)")
        return build_extents(a)
    if a.repack:
        if not a.hf_repo:
            ap.error("--repack needs --hf-repo")
        return repack(a)
    if not a.channel:
        ap.error("--channel is required")
    if not a.hf_repo and not a.dry_run:
        ap.error("--hf-repo is required (or --dry-run)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
