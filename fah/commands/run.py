"""Fetch one channel for every fire not done yet, and upload the results in batches."""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

from fah.channels.base import Channel
from fah.channels.registry import ChannelRegistry
from fah.commands.progress import ChannelProgress
from fah.commands.settings import RunSettings
from fah.core.context import RunContext
from fah.core.errors import ChannelUnavailable, ExitCode
from fah.core.fire import Fire
from fah.core.result import Status
from fah.core.secrets import Secrets
from fah.hub.batch import BatchUploader
from fah.hub.catalog import FireCatalog
from fah.hub.manifest import ManifestBook
from fah.hub.repo import HubRepo

log = logging.getLogger(__name__)


class ErrorStreak:
    """Counts errors in a row; trips at `limit` (0 = never) so a broken source does not burn through every fire. Any
    fire that is not an error resets it."""

    def __init__(self, limit: int) -> None:
        self.limit, self.n, self.last, self.tripped = limit, 0, "", False

    def hit(self, status: str, err: str) -> bool:
        if status != Status.ERROR:
            self.n = 0
            return False
        self.n, self.last = self.n + 1, err
        self.tripped = bool(self.limit) and self.n >= self.limit
        return self.tripped


class ChannelRun:
    """One channel over the fire list: resume from the manifests, skip fires the source does not cover, fetch the
    rest, record every fire in a batch. -> exit code 0, or CHANNEL_BROKEN when the error streak trips."""

    CARD = Path(__file__).resolve().parents[2] / "docs" / "RAW_DATA_CARD.md"

    def __init__(self, settings: RunSettings, registry: ChannelRegistry | None = None) -> None:
        self.s = settings
        self.channel: Channel = (registry or ChannelRegistry()).get(settings.channel)

    def execute(self) -> int:
        s, channel = self.s, self.channel
        work = Path(s.work)
        work.mkdir(parents=True, exist_ok=True)
        hub = HubRepo.connect(s.hf_repo)
        deadline = time.time() + s.max_hours * 3600 if s.max_hours else None
        if not s.dry_run:
            hub.check_write()
        channel.check_ready()

        catalog = FireCatalog(s.source_repo, work / "source")
        fires = catalog.fires(s.fires, list(s.ids) if s.ids else None)
        log.info(f"{len(fires)} fires ({s.fires}{', ' + str(len(s.ids)) + ' ids' if s.ids else ''})")
        if channel.time_dependent and channel.name != "firms":
            fires = self._with_extents(hub, fires, work)
        if not s.dry_run:
            card = self.CARD.read_text().replace("{SOURCE_REPO}", s.source_repo)
            hub.ensure(s.private, card, catalog.catalog_files())
        manifest = ManifestBook.read(hub, channel.name) if (s.hf_repo and not s.redo) else None
        done = ChannelProgress.done_ids(manifest, channel, fires)
        todo = [f for f in fires if f.fire_id not in done]
        if s.limit:
            todo = todo[: s.limit]
        log.info(f"channel {channel.name}: {len(done)} already done, {len(todo)} to do")

        ctx = RunContext(channel.name, work, deadline=deadline, threads=s.threads)
        batch = BatchUploader(
            hub,
            channel.name,
            work,
            s.batch_gb,
            s.batch_fires,
            s.min_free_gb,
            s.dry_run,
            s.keep,
            hub.shared_paths() if (s.hf_repo and not s.dry_run) else (),
        )
        streak = ErrorStreak(s.max_errors_in_row)
        t0 = time.time()
        try:
            runnable = self._record_skipped(todo, batch)
            if runnable:
                for src, repo_path in channel.shared_files(ctx):
                    batch.add_shared(src, repo_path)
            with contextlib.closing(channel.fetch(runnable, ctx)) as results:
                for k, r in enumerate(results, 1):
                    err = Secrets.redact(r.error)
                    fid = r.fire.fire_id
                    batch.add(r.fire.ref(), r.status, ctx.stage_path(fid), r.sources, err, channel.labels(r.fire))
                    ctx.drop_stage(fid)
                    if r.trace and s.verbose:
                        log.info(Secrets.redact(r.trace))
                    log.info(
                        f"[{k}/{len(runnable)}] {fid} {r.fire.name} ({r.fire.year}): {r.status}"
                        + (f" {err[:120]}" if err else "")
                        + f"  ({(time.time() - t0) / 60:.0f} min)"
                    )
                    if streak.hit(r.status, err):
                        break
            batch.upload()
        finally:
            if not s.keep_cache:
                ctx.cleanup()
        t = batch.totals
        if streak.tripped:
            log.error(
                f"CHANNEL STOPPED: {channel.name} failed for {streak.n} fires in a row, so the source or this "
                f"machine's access to it is probably broken, not single fires. Last error: {streak.last[:300]}\n"
                "Those fires are recorded as errors and retried on the next run."
            )
            return ExitCode.CHANNEL_BROKEN
        log.info(
            f"done: {t['ok']} ok, {t['empty']} empty, {t['n/a']} n/a, {t['error']} error; {t['bytes'] / 1e9:.2f} GB "
            f"in {t['batches']} batch(es), {(time.time() - t0) / 60:.0f} min"
        )
        return ExitCode.OK

    def _with_extents(self, hub: HubRepo, fires: list[Fire], work: Path) -> list[Fire]:
        """The fires with their windows from catalog/fire_extents.csv (needed by every time-dependent channel but
        firms, which is its input)."""
        extents = FireCatalog.load_extents(hub, work / "target", self.s.extents_file)
        if extents is None:
            msg = (
                f"{FireCatalog.EXTENTS_FILE} not found: run the firms channel, then --extents, so "
                f"{self.channel.name} covers the days the fire was active (not only its perimeter days)"
            )
            if not self.s.dry_run:
                raise ChannelUnavailable(msg)
            log.warning(f"warning: {msg}; using the perimeter days +- 1")
        fires = FireCatalog.with_extents(fires, extents)
        log.info(
            f"time window: {sum(f.window_days is not None for f in fires)} of {len(fires)} fires from "
            f"{FireCatalog.EXTENTS_FILE}"
        )
        return fires

    def _record_skipped(self, todo: list[Fire], batch: BatchUploader) -> list[Fire]:
        """Record the fires the source does not cover (status n/a); -> the others."""
        runnable = []
        for f in todo:
            why = self.channel.skip_reason(f)
            if why:
                batch.add(f.ref(), Status.NOT_APPLICABLE, error=why, labels=self.channel.labels(f))
            else:
                runnable.append(f)
        return runnable
