"""terrain: Copernicus GLO-30 elevation of the region, at its native 1 arc-second grid."""

from __future__ import annotations

from pathlib import Path

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.sources.dem import CopernicusDem
from fah.sources.http import HttpClient


class TerrainChannel(PerFireChannel):
    """dem_glo30.tif: the region cut from the 1-degree tiles and merged (deflate, float predictor)."""

    name = "terrain"

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        import rasterio
        from rasterio.merge import merge

        box = fire.region()
        dem = CopernicusDem(ctx.session, HttpClient(ctx.session), ctx.cache)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            srcs, sources = dem.open_tiles(box)
            if not srcs:
                return "empty", []
            arr, tr = merge(srcs, bounds=tuple(box))
            prof = srcs[0].profile.copy()
            for d in srcs:
                d.close()
        prof.update(
            driver="GTiff",
            height=arr.shape[1],
            width=arr.shape[2],
            transform=tr,
            count=arr.shape[0],
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(out / "dem_glo30.tif", "w", **prof) as dst:
            dst.write(arr)
            dst.update_tags(source=CopernicusDem.CREDIT, tiles=";".join(sources))
        return "ok", sources
