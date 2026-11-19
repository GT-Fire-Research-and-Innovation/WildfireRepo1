"""Gzip a channel's plain .geojson files already in the repo, in place, instead of fetching them again."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from fah.commands.settings import RunSettings
from fah.core.errors import ChannelUnavailable, ExitCode
from fah.core.fire import FireRef
from fah.core.result import Status
from fah.core.secrets import Secrets
from fah.hub.batch import BatchUploader
from fah.hub.manifest import ManifestBook
from fah.hub.repo import HubRepo
from fah.storage.geojson import GzipGeoJson

log = logging.getLogger(__name__)


class Repacker:
    """Fires whose latest row is 'ok' and lists only .geojson files: each file is downloaded, written as
    <name>.geojson.gz with the fixed header and committed in place of the old one, with a new manifest row (same
    sources; window / region of the old row; note: when it was fetched). For channels the next run does not fetch
    again anyway (perimeters keeps the build box)."""

    CHANNELS = ("perimeters", "history", "suppression")

    def __init__(self, settings: RunSettings) -> None:
        self.s = settings
        self.channel = settings.channel or "perimeters"
        if self.channel not in self.CHANNELS:
            raise ChannelUnavailable(f"--repack works on {', '.join(self.CHANNELS)}, not {self.channel}")

    def execute(self) -> int:
        s, ch = self.s, self.channel
        hub = HubRepo.connect(s.hf_repo)
        if not s.dry_run:
            hub.check_write()
        work = Path(s.work)
        m = ManifestBook.read(hub, ch)
        if not len(m):
            log.info(f"repack {ch}: no manifest rows")
            return ExitCode.OK
        last = ManifestBook.latest(m)
        if s.ids:
            last = last[last.fire_id.isin(set(s.ids))]
        names = last["files"].fillna("").astype(str).map(lambda x: [f for f in x.split(";") if f])
        todo = last[
            (last.status == Status.OK) & names.map(lambda fs: bool(fs) and all(f.endswith(".geojson") for f in fs))
        ]
        if s.limit:
            todo = todo.head(s.limit)
        log.info(f"repack {ch}: {len(todo)} of {len(last)} fires have plain .geojson")
        batch = BatchUploader(hub, ch, work, s.batch_gb, s.batch_fires, s.min_free_gb, s.dry_run, s.keep)
        dl, t0 = work / "repack_download", time.time()
        for k, r in enumerate(todo.to_dict("records"), 1):
            stage = work / "repack_stage" / r["fire_id"] / ch
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True)
            try:
                for name in r["files"].split(";"):
                    src = hub.download(f"fires/{r['fire_id']}/{ch}/{name}", dl)
                    GzipGeoJson.compress(src, stage / f"{name}.gz")
                    Path(src).unlink()
            except Exception as ex:  # noqa: BLE001  (left as it is; the next --repack tries again)
                shutil.rmtree(stage, ignore_errors=True)
                log.info(f"[{k}/{len(todo)}] {r['fire_id']}: skipped, {Secrets.redact(str(ex))[:150]}")
                continue
            batch.add(
                FireRef(r["fire_id"], r.get("name", ""), r.get("year", "")),
                Status.OK,
                stage,
                [x for x in str(r.get("sources") or "").split(ManifestBook.SEP) if x],
                f"gzip of the files fetched {r['fetched_utc']}",
                labels=(r.get("window", ""), r.get("region", "")),
            )
            log.info(
                f"[{k}/{len(todo)}] {r['fire_id']} {r.get('name', '')}: repacked  ({(time.time() - t0) / 60:.0f} min)"
            )
        batch.upload()
        shutil.rmtree(dl, ignore_errors=True)
        t = batch.totals
        log.info(f"repack {ch}: {t['ok']} fires, {t['bytes'] / 1e9:.2f} GB in {t['batches']} batch(es)")
        return ExitCode.OK
