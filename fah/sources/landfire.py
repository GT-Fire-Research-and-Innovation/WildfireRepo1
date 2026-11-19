"""LANDFIRE image services: stored 30 m pixels of a layer version, exported in its own projection (EPSG:5070)."""

from __future__ import annotations

import math
from urllib.parse import urlencode

import requests

from fah.core.geometry import BoundingBox
from fah.sources.http import HttpClient


class LandfireImageService:
    """Exports snapped to the service's pixel grid with nearest neighbour, so the GeoTIFF holds the stored pixels
    with no resampling. No email or job queue (the LFPS route needed both)."""

    URL = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_{v}/{v}_{layer}_CONUS/ImageServer"
    TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00")

    def __init__(self, session: requests.Session, http: HttpClient) -> None:
        self.session, self.http = session, http
        self._info: dict[tuple[str, str], dict] = {}

    @staticmethod
    def native_window(
        box: BoundingBox, origin: tuple[float, float], res: float, crs: int = 5070
    ) -> tuple[float, float, float, float, int, int]:
        """The box in the service's projection, snapped outward to its pixel edges (origin = top-left corner of the
        service extent). -> (x0, y0, x1, y1, width, height)"""
        from pyproj import Transformer

        xs0, ys0, xs1, ys1 = Transformer.from_crs(4326, crs, always_xy=True).transform_bounds(*box, densify_pts=21)
        X0, Y1 = origin
        x0 = X0 + math.floor((xs0 - X0) / res) * res
        x1 = X0 + math.ceil((xs1 - X0) / res) * res
        y1 = Y1 - math.floor((Y1 - ys1) / res) * res
        y0 = Y1 - math.ceil((Y1 - ys0) / res) * res
        return x0, y0, x1, y1, int(round((x1 - x0) / res)), int(round((y1 - y0) / res))

    @classmethod
    def export_url(cls, version: str, layer: str, box: BoundingBox, info: dict) -> str:
        ext, res = info["extent"], info["pixelSizeX"]
        x0, y0, x1, y1, w, h = cls.native_window(box, (ext["xmin"], ext["ymax"]), res, info["spatialReference"]["wkid"])
        q = {
            "bbox": f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}",
            "bboxSR": 5070,
            "imageSR": 5070,
            "size": f"{w},{h}",
            "format": "tiff",
            "compression": "LZW",
            "pixelType": info["pixelType"],
            "noData": info.get("noDataValue", -9999),
            "interpolation": "RSP_NearestNeighbor",
            "f": "image",
        }
        return f"{cls.URL.format(v=version, layer=layer)}/exportImage?{urlencode(q)}"

    def info(self, version: str, layer: str) -> dict:
        """The image service's description (extent, pixel size, type), fetched once per run."""
        key = (version, layer)
        if key not in self._info:
            self._info[key] = self.session.get(
                self.URL.format(v=version, layer=layer), params={"f": "json"}, timeout=60
            ).json()
        info = self._info[key]
        if "extent" not in info:
            raise RuntimeError(f"LANDFIRE {version}_{layer}: service info unavailable: {str(info)[:150]}")
        return info

    def export(self, version: str, layer: str, box: BoundingBox) -> tuple[str, bytes]:
        """(url, GeoTIFF bytes) of the box, exactly as the server returns them (LZW)."""
        url = self.export_url(version, layer, box, self.info(version, layer))
        data = self.http.get_bytes(url, timeout=600)
        if data[:4] not in self.TIFF_MAGIC:
            raise RuntimeError(f"LANDFIRE {version}_{layer}: reply is not a GeoTIFF: {data[:150]!r}")
        return url, data
