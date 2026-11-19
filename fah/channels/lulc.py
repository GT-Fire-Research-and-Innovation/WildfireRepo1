"""lulc: Esri / Impact Observatory 10 m annual land use / land cover (the map behind TS-SatFire's ESRI_LULC)."""

from __future__ import annotations

import json
from pathlib import Path

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.geometry import PixelWindow
from fah.sources.planetary import PlanetaryComputer


class LulcChannel(PerFireChannel):
    """io_lulc_<year>_<tile>.tif per source tile (one UTM zone each): the region's pixels read from the
    cloud-optimized GeoTIFF on its own grid, no resampling. Class legend in the file's `classes` tag."""

    name = "lulc"
    COLLECTION = "io-lulc-annual-v02"  # on Microsoft Planetary Computer
    FIRST_YEAR, LAST_YEAR = 2017, 2023  # maps that exist

    @classmethod
    def map_year(cls, year: int) -> int:
        """The year before the fire, within the maps that exist (2017 fires get the first map, as TS-SatFire's do)."""
        return min(max(int(year) - 1, cls.FIRST_YEAR), cls.LAST_YEAR)

    def skip_reason(self, fire: Fire) -> str | None:
        if int(fire.year) < self.FIRST_YEAR:
            return "Esri 10 m land cover starts 2017"
        return None

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import Window

        pc = ctx.memo("planetary", lambda: PlanetaryComputer(ctx.session))
        yr = self.map_year(fire.year)
        box = fire.region()
        sources = []
        for it in pc.search(self.COLLECTION, box, yr):
            a = it["assets"]["data"]
            href = a["href"]
            legend = {str(x["values"][0]): x["summary"] for x in a.get("file:values", [])}
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="5"
            ):
                with rasterio.open(f"/vsicurl/{href}?{pc.token(self.COLLECTION)}") as src:
                    win = PixelWindow.covering(
                        src.transform, src.width, src.height, transform_bounds(4326, src.crs, *box, densify_pts=21)
                    )
                    if win is None:
                        continue
                    r0, r1, c0, c1 = win
                    w = Window(c0, r0, c1 - c0, r1 - r0)
                    arr = src.read(window=w)
                    prof = {
                        k: v
                        for k, v in src.profile.items()
                        if k not in ("blockxsize", "blockysize", "tiled", "interleave")
                    }
                    prof.update(
                        driver="GTiff",
                        width=c1 - c0,
                        height=r1 - r0,
                        transform=src.window_transform(w),
                        compress="deflate",
                        tiled=True,
                        blockxsize=256,
                        blockysize=256,
                        BIGTIFF="IF_SAFER",
                    )
            tile = it["id"].rsplit("-", 1)[0]
            with rasterio.open(out / f"io_lulc_{yr}_{tile}.tif", "w", **prof) as dst:
                dst.write(arr)
                dst.update_tags(
                    source=href,
                    stac_item=it["id"],
                    collection=self.COLLECTION,
                    map_year=str(yr),
                    fire_year=str(fire.year),
                    classes=json.dumps(legend),
                    window_rows_cols=win.label(),
                    licence="CC BY 4.0: Impact Observatory, Microsoft and Esri",
                )
            sources.append(href)
        return ("ok" if sources else "empty"), sources
