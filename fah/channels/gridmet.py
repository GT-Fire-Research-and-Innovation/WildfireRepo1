"""gridmet: GRIDMET daily weather and fire danger, 13 variables, one netCDF each."""

from __future__ import annotations

from pathlib import Path

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.sources.gridmet import GridmetService
from fah.sources.http import HttpClient


class GridmetChannel(PerFireChannel):
    """gridmet_<var>.nc exactly as the server returns it, over the region and Fire.days() (14 days before the
    window to 2 after; PDSI from 10 days earlier still)."""

    name = "gridmet"
    time_dependent = True

    @property
    def expected_files(self) -> frozenset[str]:
        return frozenset(f"gridmet_{v}.nc" for v in GridmetService.VARIABLES)

    def window_label(self, fire: Fire) -> str:
        d = fire.days()
        return f"{d[0].date().isoformat()}/{d[-1].date().isoformat()}"

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        service = GridmetService(HttpClient(ctx.session))
        ds = fire.days()
        sources = []
        for short in GridmetService.VARIABLES:
            url, data = service.subset(short, fire.region(), ds[0], ds[-1])
            (out / f"gridmet_{short}.nc").write_bytes(data)
            sources.append(url)
        return "ok", sources
