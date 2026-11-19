"""osm: the OpenStreetMap extract of the region, from the Geofabrik snapshot of 1 January of the fire's year."""

from __future__ import annotations

from collections.abc import Iterator

from fah.channels.base import SharedDownloadChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.result import FetchResult
from fah.sources.http import HttpClient
from fah.sources.osm import Geofabrik


class OsmChannel(SharedDownloadChannel):
    """osm_<YYYY>-01-01.osm.pbf: the full (unfiltered) extract. Fires are grouped by (region file, year) so each
    ~2-3 GB snapshot downloads once; it is deleted before the next one."""

    name = "osm"

    def skip_reason(self, fire: Fire) -> str | None:
        try:
            Geofabrik.region(fire.region())
        except RuntimeError as ex:
            return str(ex)
        return None

    def check_ready(self) -> None:
        Geofabrik.check_ready()

    def fetch(self, fires: list[Fire], ctx: RunContext) -> Iterator[FetchResult]:
        geofabrik = Geofabrik(HttpClient(ctx.session), ctx.cache)
        groups: dict[tuple[str, int], list[Fire]] = {}
        for f in fires:
            groups.setdefault((Geofabrik.region(f.region()), int(f.year)), []).append(f)
        for (region, year), group in sorted(groups.items()):
            if ctx.expired():
                return
            url = Geofabrik.snapshot_url(region, year)
            try:
                snap = geofabrik.snapshot(region, year)
            except Exception as ex:  # noqa: BLE001  (every fire of this snapshot gets the error; others go on)
                for f in group:
                    yield FetchResult(f, "error", (), f"snapshot {url}: {str(ex)[:200]}")
                continue
            for f in group:
                if ctx.expired():
                    break
                r = Geofabrik.extract(snap, f.region(), ctx.stage(f.fire_id) / f"osm_{year}-01-01.osm.pbf")
                yield FetchResult(f, "error", (), r.stderr[-300:]) if r.returncode else FetchResult(f, "ok", (url,))
            snap.unlink(missing_ok=True)
