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
    deadline nears. The error streak ends a pass: when fires worked earlier in that pass, the source is taken to be
    down for a while (an outage, throttling) and the run pauses and goes back; when none did, it is broken.
    -> exit code 0, or CHANNEL_BROKEN when the streak trips in a pass where no fire worked."""

    RETRY_PAUSE_S = 600.0  # between passes: rate limits reset (FIRMS: per 10 min), outages pass, few commits

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
        sweep, shared_sent, broken, tried = 0, False, None, set()
        try:
            while todo:
                sweep += 1
                streak = ErrorStreak(s.max_errors_in_row)
                runnable = self._record_skipped(todo, batch)
                if runnable and not shared_sent:
                    for src, repo_path in channel.shared_files(ctx):
                        batch.add_shared(src, repo_path)
                    shared_sent = True
                worked = self._fetch(runnable, ctx, batch, streak, sweep, t0, tried)
                batch.upload()
                if streak.tripped:
                    if not worked or s.dry_run or not s.hf_repo or ctx.deadline is None:
                        broken = streak  # nothing worked this pass (or no later pass to go to): the source is down
                        break
                    log.warning(
                        f"warning: pass {sweep}: {streak.n} fires failed in a row after {worked} worked "
                        f"({streak.last[:200]}): the source is probably down for a while; pausing, then going back"
                    )
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
        self._report_gaps(hub, scope, since)
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
        self,
        runnable: list[Fire],
        ctx: RunContext,
        batch: BatchUploader,
        streak: ErrorStreak,
        sweep: int,
        t0: float,
        tried: set[str],
    ) -> int:
        """One pass: fetch the fires and add each result to the batch, until they are all done, the deadline nears
        or the error streak trips. Errors of fires this run tried before (in `tried`) do not count toward the streak:
        they failed already, so failing again says nothing new about the source. -> fires that did not fail."""
        tag = f"pass {sweep} " if sweep > 1 else ""
        worked = 0
        with contextlib.closing(self.channel.fetch(runnable, ctx)) as results:
            for k, r in enumerate(results, 1):
                err = Secrets.redact(r.error)
                fid = r.fire.fire_id
                retried = fid in tried
                tried.add(fid)
                worked += r.status != Status.ERROR
                batch.add(r.fire.ref(), r.status, ctx.stage_path(fid), r.sources, err, self.channel.labels(r.fire))
                ctx.drop_stage(fid)
                if r.trace and self.s.verbose:
                    log.info(Secrets.redact(r.trace))
                log.info(
                    f"[{tag}{k}/{len(runnable)}] {fid} {r.fire.name} ({r.fire.year}): {r.status}"
                    + (f" {err[:120]}" if err else "")
                    + f"  ({(time.time() - t0) / 60:.0f} min)"
                )
                if not (retried and r.status == Status.ERROR) and streak.hit(r.status, err):
                    return worked
        return worked

    def _missed(self, hub: HubRepo, scope: list[Fire], since: str | None) -> list[Fire]:
        """The fires of scope still not done, by the channel's whole manifest read again from the repo after a pass.
        `since`: only rows written from then on count (--redo)."""
        manifest = ManifestBook.read(hub, self.channel.name)
        if since and len(manifest):
            manifest = manifest[manifest["fetched_utc"].astype(str) >= since]
        done = ChannelProgress.done_ids(manifest, self.channel, scope)
        return [f for f in scope if f.fire_id not in done]

    def _report_gaps(self, hub: HubRepo, scope: list[Fire], since: str | None) -> None:
        """Log every fire of the run the channel missed (not done, or empty), with its name and its folder in the
        repo (LINKS.md there links every channel's files and sources once --links has run)."""
        s, name = self.s, self.channel.name
        try:
            if s.dry_run:
                manifest = ManifestBook.read_local(Path(s.work) / "batch", name)
            else:
                manifest = ManifestBook.read(hub, name)
        except Exception as ex:  # noqa: BLE001  (the report is a convenience: never fail the run over it)
            log.warning(f"warning: could not read the {name} manifests for the missed-fire list: {ex}")
            return
        if since and len(manifest):
            manifest = manifest[manifest["fetched_utc"].astype(str) >= since]
        gaps = ChannelProgress.gaps(manifest, self.channel, scope)
        n_empty = sum(st == Status.EMPTY for _, st, _ in gaps)
        log.info(
            f"missed in {name}: {len(gaps) - n_empty} not done, {n_empty} empty (of {len(scope)} fires)"
            + (":" if gaps else "")
        )
        base = f"https://huggingface.co/datasets/{s.hf_repo}/tree/main/fires" if s.hf_repo else None
        for f, status, note in gaps:
            link = f"  {base}/{f.fire_id}" if base else ""
            log.info(f"  {f.fire_id}  {f.name} ({f.year})  {status}{link}" + (f"  {note[:150]}" if note else ""))

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
