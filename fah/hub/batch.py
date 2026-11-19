"""Upload batches: finished fires collected in a folder laid out like the repo, committed together with their
manifest, then deleted locally."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fah.core.fire import FireRef
from fah.core.result import Status
from fah.core.secrets import Secrets
from fah.hub.manifest import ManifestBook
from fah.hub.repo import HubRepo

log = logging.getLogger(__name__)


class BatchUploader:
    """A batch is committed when it reaches batch_gb or batch_fires, when free disk falls below min_free_gb, or on
    upload(). A fire fetched again replaces its folder: the commit deletes the channel's older files of that fire. A
    failed upload keeps the batch on disk."""

    def __init__(
        self,
        hub: HubRepo | None,
        channel: str,
        work: Path,
        batch_gb: float = 2.0,
        batch_fires: int = 100,
        min_free_gb: float = 5.0,
        dry_run: bool = False,
        keep: bool = False,
        uploaded_shared: Iterable[str] = (),
    ) -> None:
        self.hub, self.channel = hub, channel
        self.root = Path(work) / "batch"
        self.batch_bytes, self.batch_fires, self.min_free = batch_gb * 1e9, batch_fires, min_free_gb * 1e9
        self.dry_run, self.keep = dry_run, keep or dry_run
        self.uploaded_shared = set(uploaded_shared)
        self.rows: list[dict] = []
        self.bytes, self.k = 0, 0
        self.dir: Path | None = None
        self.name = ""
        self.replace: list[str] = []  # fires/<id>/<channel>/* of this batch's fires: their older files go
        self.totals = {"ok": 0, "empty": 0, "n/a": 0, "error": 0, "bytes": 0, "batches": 0}

    def _open(self) -> Path:
        if self.dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.name = f"{self.channel}-{stamp}-{self.k:03d}"
            self.dir = self.root / self.name
            self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def add_shared(self, src: Path, repo_path: str) -> None:
        """A file stored once per repo (shared/...), with this batch unless the repo has it already."""
        if repo_path in self.uploaded_shared or not Path(src).exists():
            return
        dst = self._open() / repo_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.bytes += dst.stat().st_size
        self.uploaded_shared.add(repo_path)

    def add(
        self,
        fire: FireRef,
        status: str,
        stage_dir: Path | None = None,
        sources: Sequence[str] = (),
        error: str = "",
        labels: tuple[str, str] = ("", ""),
    ) -> None:
        """One fire's result: its staged files (kept for 'ok' / 'empty') and its manifest row. labels: the (window,
        region) the files cover."""
        files, nbytes = [], 0
        if stage_dir is not None and Path(stage_dir).exists():
            dst = self._open() / "fires" / fire.fire_id / self.channel
            if status in Status.STORED and any(Path(stage_dir).iterdir()):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(dst, ignore_errors=True)
                shutil.move(str(stage_dir), str(dst))
                files = sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file())
                nbytes = sum((dst / f).stat().st_size for f in files)
            else:
                shutil.rmtree(stage_dir, ignore_errors=True)
        self._open()
        window, region = labels
        self.rows.append(
            dict(
                fire_id=fire.fire_id,
                name=fire.name,
                year=fire.year,
                channel=self.channel,
                status=status,
                n_files=len(files),
                bytes=nbytes,
                files=";".join(files),
                sources=ManifestBook.SEP.join(Secrets.redact(s) for s in sources),
                fetched_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                error=Secrets.redact(error)[:500],
                window=window,
                region=region,
            )
        )
        if status in Status.STORED:  # a new result: whatever this fire had in the channel before goes
            self.replace.append(f"fires/{fire.fire_id}/{self.channel}/*")
        self.bytes += nbytes
        self.totals[status if status in self.totals else "error"] += 1
        self.totals["bytes"] += nbytes
        if len(self.rows) >= self.batch_fires or self.bytes >= self.batch_bytes or self.low_disk():
            self.upload()

    def low_disk(self) -> bool:
        return shutil.disk_usage(self.root if self.root.exists() else self.root.parent).free < self.min_free

    def upload(self) -> None:
        """Commit the batch (files + its manifest) and delete it locally. No-op when empty."""
        if self.dir is None or not self.rows:
            return
        mdir = self.dir / "manifest" / self.channel
        mdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows, columns=ManifestBook.COLUMNS).to_csv(mdir / f"{self.name}.csv", index=False)
        n_ok = sum(r["status"] == Status.OK for r in self.rows)
        msg = f"raw {self.channel}: {len(self.rows)} fires ({n_ok} ok), {self.bytes / 1e9:.2f} GB"
        if not self.dry_run:
            for attempt in range(6):
                try:
                    self.hub.upload_folder(self.dir, msg, self.replace or None)
                    break
                except Exception as ex:  # noqa: BLE001  (rate limits, transient 5xx)
                    if attempt == 5:
                        raise RuntimeError(f"upload failed, batch kept at {self.dir}: {ex}") from ex
                    wait = 30 * 2**attempt
                    log.info(f"  upload retry {attempt + 1} in {wait}s: {str(ex)[:150]}")
                    time.sleep(wait)
        log.info(
            f"  {'[dry-run] ' if self.dry_run else ''}{msg}"
            + (f" -> kept {self.dir}" if self.keep else " -> uploaded, local copy deleted")
        )
        self.totals["batches"] += 1
        if not self.keep:
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir, self.rows, self.bytes, self.replace = None, [], 0, []
        self.k += 1
