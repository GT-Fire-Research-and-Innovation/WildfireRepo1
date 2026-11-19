"""landcover: MODIS MCD12Q1 v061 annual land cover for the year before the fire."""

from __future__ import annotations

from pathlib import Path

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.geometry import SinusoidalGrid
from fah.sources.earthdata import EarthdataClient
from fah.sources.http import DownloadCache
from fah.storage.sinusoidal import SinusoidalMosaic


class LandcoverChannel(PerFireChannel):
    """mcd12q1_<year>.nc: every layer of the product (LC_Type1, TS-SatFire's land-cover band, LC_Type2-5,
    LC_Prop1-3 and their assessments, QC, LW) on the sinusoidal 463 m grid, the region cut from each tile and
    joined pixel for pixel. Tiles are LP DAAC HDF4 files (13 MB), downloaded whole once per run and shared by
    every fire on them."""

    name = "landcover"
    PRODUCT = "MCD12Q1"

    def check_ready(self) -> None:
        EarthdataClient.token()
        self.require_module("pyhdf.SD", "landcover reads MCD12Q1 HDF4 tiles: pip install pyhdf")

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        from pyhdf.SD import SD, SDC

        earthdata = EarthdataClient(ctx.session)
        yr = int(fire.year) - 1
        box = fire.region()
        win = SinusoidalGrid.window(box)
        grans = [
            g
            for g in earthdata.granules(self.PRODUCT, f"{yr}-01-01", f"{yr}-12-31T23:59:59", box)
            if EarthdataClient.granule_parts(g["url"])[1] == f"A{yr}001"
            and SinusoidalGrid.tile_part(win, *SinusoidalGrid.tile_hv(g["name"]))
        ]
        token = EarthdataClient.token()
        cache = DownloadCache(ctx.cache / "files")
        mosaic, sources = SinusoidalMosaic(win), []
        for g in grans:
            h, v = SinusoidalGrid.tile_hv(g["name"])
            path = cache.get(g["url"], lambda p, u=g["url"]: earthdata.download(u, p, token))
            hdf = SD(str(path), SDC.READ)
            try:
                for name in hdf.datasets():
                    sds = hdf.select(name)
                    try:
                        if list(sds.info()[2]) == [SinusoidalGrid.N, SinusoidalGrid.N]:
                            mosaic.put(name, sds, sds.attributes(), h, v)
                    finally:
                        sds.endaccess()
            finally:
                hdf.end()
            sources.append(g["url"])
        if not mosaic.layers:
            return "empty", sources
        mosaic.write(
            out / f"mcd12q1_{yr}.nc",
            {
                "title": f"MODIS MCD12Q1 v061 land cover {yr} cropped to {fire.fire_id}",
                "fire_id": fire.fire_id,
                "map_year": yr,
                "sources": " | ".join(sources),
            },
        )
        return "ok", sources
