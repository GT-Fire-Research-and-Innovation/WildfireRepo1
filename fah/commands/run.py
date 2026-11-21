"""Fetch one channel for every fire not done yet, and upload the results in batches."""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fah.channels.base import Channel
from fah.channels.registry import ChannelRegistry
from fah.commands.progress import ChannelProgress
from fah.commands.settings import RunSettings
from fah.core.context import RunContext
from fah.core.errors import ChannelUnavailable, ExitCode
from fah.core.extent import FireExtentEstimator
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
    rest, record every fire in a batch. With --max-hours, a pass that ends before the deadline is followed by a
    check of the channel's whole manifest in the repo, and the fires still not done (errors such as rate-limited
    calls, fires a pass never reached) get another pass after RETRY_PAUSE_S, until every fire is done or the
    deadline nears. -> exit code 0, or CHANNEL_BROKEN when the error streak trips in the first pass."""

    RETRY_PAUSE_S = 600.0  # between passes: rate limits reset (FIRMS: per 10 min), and no commit every few seconds

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

        fires = FireCatalog(hub, work / "target", s.catalog_file).fires(s.archive, list(s.ids) if s.ids else None)
        log.info(f"{len(fires)} fires (archive {s.archive}{', ' + str(len(s.ids)) + ' ids' if s.ids else ''})")
        if channel.time_dependent and channel.name != "firms":
            fires = self._with_extents(hub, fires, work)
        if not s.dry_run:
            hub.ensure(s.private)
        # --redo: only rows written by this run count as done when the repo is checked again
        since = datetime.now(timezone.utc).isoformat(timespec="seconds") if s.redo else None
        manifest = ManifestBook.read(hub, channel.name) if (s.hf_repo and not s.redo) else None
        done = ChannelProgress.done_ids(manifest, channel, fires)
        todo = [f for f in fires if f.fire_id not in done]
        if s.limit:
            todo = todo[: s.limit]
        scope = todo if s.limit else fires  # the fires this run answers for: later passes look only here
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
        t0 = time.time()
        sweep, shared_sent, broken = 0, False, None
        try:
            while todo:
                sweep += 1
                # the streak guards the first pass only: later passes hold just the fires that failed already
                streak = ErrorStreak(s.max_errors_in_row if sweep == 1 else 0)
                runnable = self._record_skipped(todo, batch)
                if runnable and not shared_sent:
                    for src, repo_path in channel.shared_files(ctx):
                        batch.add_shared(src, repo_path)
                    shared_sent = True
                self._fetch(runnable, ctx, batch, streak, sweep, t0)
                batch.upload()
                if streak.tripped:
                    broken = streak
                    break
                if s.dry_run or not s.hf_repo or ctx.deadline is None or ctx.expired():
                    break  # without --max-hours nothing would end the passes: one pass, as before
                todo = self._missed(hub, scope, since)
                if not todo:
                    log.info(f"pass {sweep}: every fire of the run is done in the repo")
                    break
                log.info(
                    f"pass {sweep}: {len(todo)} fires not done in the repo (errors, rate limits, not reached), e.g. "
                    f"{', '.join(f.fire_id for f in todo[:3])}: going back to them in "
                    f"{self.RETRY_PAUSE_S / 60:.0f} min"
                )
                time.sleep(max(0.0, min(self.RETRY_PAUSE_S, ctx.deadline - ctx.reserve_s - time.time())))
                if ctx.expired():
                    break
        finally:
            if not s.keep_cache:
                ctx.cleanup()
        t = batch.totals
        if broken is not None:
            log.error(
                f"CHANNEL STOPPED: {channel.name} failed for {broken.n} fires in a row, so the source or this "
                f"machine's access to it is probably broken, not single fires. Last error: {broken.last[:300]}\n"
                "Those fires are recorded as errors and retried on the next run."
            )
            return ExitCode.CHANNEL_BROKEN
        log.info(
            f"done: {t['ok']} ok, {t['empty']} empty, {t['n/a']} n/a, {t['error']} error; {t['bytes'] / 1e9:.2f} GB "
            f"in {t['batches']} batch(es), {sweep} pass(es), {(time.time() - t0) / 60:.0f} min"
        )
        return ExitCode.OK

    def _fetch(
        self, runnable: list[Fire], ctx: RunContext, batch: BatchUploader, streak: ErrorStreak, sweep: int, t0: float
    ) -> None:
        """One pass: fetch the fires and add each result to the batch, until they are all done, the deadline nears
        or the error streak trips."""
        tag = f"pass {sweep} " if sweep > 1 else ""
        with contextlib.closing(self.channel.fetch(runnable, ctx)) as results:
            for k, r in enumerate(results, 1):
                err = Secrets.redact(r.error)
                fid = r.fire.fire_id
                batch.add(r.fire.ref(), r.status, ctx.stage_path(fid), r.sources, err, self.channel.labels(r.fire))
                ctx.drop_stage(fid)
                if r.trace and self.s.verbose:
                    log.info(Secrets.redact(r.trace))
                log.info(
                    f"[{tag}{k}/{len(runnable)}] {fid} {r.fire.name} ({r.fire.year}): {r.status}"
                    + (f" {err[:120]}" if err else "")
                    + f"  ({(time.time() - t0) / 60:.0f} min)"
                )
                if streak.hit(r.status, err):
                    return

    def _missed(self, hub: HubRepo, scope: list[Fire], since: str | None) -> list[Fire]:
        """The fires of scope still not done, by the channel's whole manifest read again from the repo after a pass.
        `since`: only rows written from then on count (--redo)."""
        manifest = ManifestBook.read(hub, self.channel.name)
        if since and len(manifest):
            manifest = manifest[manifest["fetched_utc"].astype(str) >= since]
        done = ChannelProgress.done_ids(manifest, self.channel, scope)
        return [f for f in scope if f.fire_id not in done]

    def _with_extents(self, hub: HubRepo, fires: list[Fire], work: Path) -> list[Fire]:
        """The fires with their windows from catalog/fire_extents.csv (needed by every time-dependent channel but
        firms, which is its input). Refuses while some windows can still grow (firms not finished for those fires,
        or the catalog changed since --extents), since those fires would be fetched again; --partial-windows (or
        --dry-run) goes ahead with the perimeter days +- 1 for them."""
        extents = FireCatalog.load_extents(hub, work / "target", self.s.extents_file) or {}
        waiting = [f.fire_id for f in fires if not FireExtentEstimator.is_final(f, extents.get(f.fire_id))]
        if waiting:
            msg = (
                f"{len(waiting)} of {len(fires)} fires have no final window in {FireCatalog.EXTENTS_FILE} yet (e.g. "
                f"{', '.join(waiting[:3])}): finish the firms channel, run --extents, then {self.channel.name}; or "
                "pass --partial-windows to fetch them now (they are fetched again once their windows are final)"
            )
            if not (self.s.partial_windows or self.s.dry_run):
                raise ChannelUnavailable(msg)
            log.warning(f"warning: {msg}")
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
