"""catalog/fire_extents.csv: each fire's time window, ignition point and region, from its firms files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fah.channels.registry import ChannelRegistry
from fah.commands.progress import ChannelProgress
from fah.commands.settings import RunSettings
from fah.core.errors import ExitCode
from fah.core.extent import FireExtentEstimator
from fah.core.fire import Fire
from fah.hub.catalog import FireCatalog
from fah.hub.manifest import ManifestBook
from fah.hub.repo import HubRepo

log = logging.getLogger(__name__)


class ExtentsBuilder:
    """Reads the firms files from the target repo (or local --dry-run batches with --local-manifests) and each fire's
    first perimeter record and reported origin from the catalog. Rows of fires not in this run (--ids) are kept."""

    def __init__(self, settings: RunSettings, registry: ChannelRegistry | None = None) -> None:
        self.s = settings
        self.firms = (registry or ChannelRegistry()).get("firms")

    def execute(self) -> int:
        s = self.s
        hub = HubRepo.connect(s.hf_repo)
        work = Path(s.work)
        if not s.dry_run:
            hub.check_write()
        fires = FireCatalog(hub, work / "target", s.catalog_file).fires(s.archive, list(s.ids) if s.ids else None)
        if not fires:
            log.info("no fires matched (--archive / --ids): nothing to do")
            return ExitCode.FAILED
        if s.local_manifests:
            m = ManifestBook.read_local(Path(s.local_manifests), "firms")
            roots = [p for p in Path(s.local_manifests).glob("**/fires") if p.is_dir()]
        else:
            m = ManifestBook.read(hub, "firms")
            dl = work / "firms_files"
            hub.snapshot([f"fires/{f.fire_id}/firms/*" for f in fires] if s.ids else ["fires/*/firms/*"], dl)
            roots = [dl / "fires"]
        current = ChannelProgress.done_ids(m, self.firms, fires)
        stale = [f.fire_id for f in fires if f.fire_id not in current]
        rows = [FireExtentEstimator.row(f, self._detections(f, roots)) for f in fires]
        new = pd.DataFrame([r | {"firms_current": r["fire_id"] in current} for r in rows])
        old = FireCatalog.load_extents(None if s.dry_run else hub, work / "target")
        if old and s.ids:
            keep = pd.DataFrame([r for fid, r in old.items() if fid not in set(new.fire_id)])
            new = pd.concat([keep, new], ignore_index=True)
        new = new.sort_values("fire_id")
        out = work / "extents" / FireCatalog.EXTENTS_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(out, index=False)
        self._summary(new, stale)
        if s.dry_run:
            log.info(f"[dry-run] kept {out}")
        else:
            hub.upload_file(out, FireCatalog.EXTENTS_FILE, f"Fire extents: window, ignition, region ({len(new)} fires)")
            log.info(f"uploaded {FireCatalog.EXTENTS_FILE}")
        return ExitCode.OK

    @staticmethod
    def _detections(fire: Fire, roots: list[Path]) -> pd.DataFrame | None:
        paths = [p for r in roots for p in (r / fire.fire_id / "firms").glob("*.csv.gz")]
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else None

    @staticmethod
    def _summary(table: pd.DataFrame, stale: list[str]) -> None:
        before = (pd.to_numeric(table.active_days_before) > 0).sum()
        after = (pd.to_numeric(table.active_days_after) > 0).sum()
        log.info(
            f"{len(table)} fires: {before} windows start before the first perimeter, {after} end after the last "
            f"(activity), {(table.window_rule == 'perimeters').sum()} without FIRMS data; ignition from "
            + ", ".join(f"{k} {v}" for k, v in table.ignition_method.value_counts().items())
        )
        if stale:
            log.warning(
                f"warning: {len(stale)} fires have no firms row for the current FIRMS days "
                f"({Fire.FIRMS_BEFORE_DAYS} before / {Fire.FIRMS_AFTER_DAYS} after the perimeters), e.g. "
                f"{stale[:3]}: rerun the firms channel, then --extents again, or their windows miss activity "
                "outside the old span"
            )
