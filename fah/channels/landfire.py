"""landfire: LANDFIRE fuel model and canopy layers of the newest version before the fire."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.sources.http import HttpClient
from fah.sources.landfire import LandfireImageService


class LandfireChannel(PerFireChannel):
    """landfire_<version>_<layer>.tif for FBFM40, CC, CH, CBH, CBD: native 30 m pixels in EPSG:5070, the LZW
    GeoTIFF exactly as the service returns it."""

    name = "landfire"
    LAYERS = ("FBFM40", "CC", "CH", "CBH", "CBD")

    @staticmethod
    def version(year: int) -> str | None:
        """The newest LANDFIRE version that predates the fire (the built dataset's rule)."""
        if year <= 2016:
            return None
        if year <= 2022:
            return "LF2016"
        return f"LF{min(year - 1, 2025)}"

    def skip_reason(self, fire: Fire) -> str | None:
        if self.version(int(fire.year)) is None:
            return "no LANDFIRE version predates fires before 2017"
        return None

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        v = self.version(int(fire.year))
        service = ctx.memo("landfire", lambda: LandfireImageService(ctx.session, HttpClient(ctx.session)))
        box = fire.region()

        def one(layer: str) -> str:
            url, data = service.export(v, layer, box)
            (out / f"landfire_{v}_{layer}.tif").write_bytes(data)
            return url

        try:
            with ThreadPoolExecutor(len(self.LAYERS)) as ex:
                return "ok", list(ex.map(one, self.LAYERS))
        except requests.RequestException as ex:
            raise RuntimeError(f"LANDFIRE {v}: {ex}") from ex
