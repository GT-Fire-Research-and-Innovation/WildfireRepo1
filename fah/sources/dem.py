"""Copernicus GLO-30 DEM: 1-degree cloud-optimized GeoTIFF tiles on AWS, native 1 arc-second grid."""

from __future__ import annotations

import math
from pathlib import Path

import requests

from fah.core.geometry import BoundingBox
from fah.sources.http import HttpClient


class CopernicusDem:
    """Tiles over a box, opened for windowed reads (a whole tile is downloaded only when remote reads fail)."""

    URL = (
        "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
        "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
    )
    CREDIT = "Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus 2014-2018, ESA licence"

    def __init__(self, session: requests.Session, http: HttpClient, cache_dir: Path) -> None:
        self.session, self.http, self.cache_dir = session, http, Path(cache_dir)

    @classmethod
    def tile_url(cls, lat: int, lon: int) -> str:
        return cls.URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="W" if lon < 0 else "E", lon=abs(lon))

    @classmethod
    def tile_urls(cls, box: BoundingBox) -> list[str]:
        """Every tile the box touches, south to north, west to east."""
        return [
            cls.tile_url(lat, lon)
            for lat in range(math.floor(box.south), math.floor(box.north) + 1)
            for lon in range(math.floor(box.west), math.floor(box.east) + 1)
        ]

    def open_tiles(self, box: BoundingBox) -> tuple[list, list[str]]:
        """(open rasterio datasets, their URLs) of the tiles over the box; ocean tiles (no file) are skipped. Call
        inside the rasterio.Env of the caller; the caller closes the datasets."""
        import rasterio

        srcs, urls = [], []
        for url in self.tile_urls(box):
            code = self.session.head(url, timeout=60).status_code
            if code in (403, 404):
                continue  # ocean: the archive has no tile there
            if code != 200:
                raise RuntimeError(f"DEM tile HEAD {code}: {url}")
            try:
                srcs.append(rasterio.open("/vsicurl/" + url))  # COG: reads only the window it needs
            except rasterio.errors.RasterioIOError:
                tile = self.cache_dir / "dem" / url.rsplit("/", 1)[1]  # no remote reads here: fetch the whole tile
                if not tile.exists():
                    tile.parent.mkdir(parents=True, exist_ok=True)
                    tile.write_bytes(self.http.get_bytes(url, timeout=600))
                srcs.append(rasterio.open(tile))
            urls.append(url)
        return srcs, urls
