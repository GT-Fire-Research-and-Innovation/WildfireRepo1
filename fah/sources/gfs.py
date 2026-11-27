"""NOAA GFS 0.25 deg forecasts from Google Earth Engine (NOAA/GFS0P25), read on the collection's native grid."""

from __future__ import annotations

import io
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fah.core.clock import Utc
from fah.core.errors import ChannelUnavailable
from fah.core.geometry import BoundingBox, PixelWindow


class EarthEngineGfs:
    """One computePixels request per (run, window, chunk of leads): every band each image has, values as Earth Engine
    stores them. Fires of one run share requests (clusters)."""

    COLLECTION = "NOAA/GFS0P25"
    START = "2015-07-01"  # first image in the collection
    # Leads the collection holds (catalog page): 3-hourly to 36 h until 2017-07-09; from then hourly to 120 h,
    # 3-hourly to 240 h (and on to 384 h, 12-hourly until 2021-06-11, 3-hourly after).
    MAX_LEAD = (("2015-07-01", 36), ("2017-07-09", 240))
    MAX_BANDS = 990  # bands per request (22 per image since 2025-01-15)
    # The .npy header of a request lists each band as ('f<lead>__<band>', '<f4'), ~50 bytes: a full request's header
    # is far past numpy's 10000-byte default (27158 bytes for 2017 runs, 48822 with the 22 bands of 2025 on).
    MAX_NPY_HEADER = MAX_BANDS * 128
    MAX_CELLS = 2500  # 0.25 deg cells per request
    THREADS = 10  # earthengine-api's requests session pools 10 connections per host: more threads only churn them
    UNITS = {
        "temperature_2m_above_ground": "degC",
        "dew_point_temperature_2m_above_ground": "degC",
        "maximum_temperature_2m_above_ground": "degC",
        "minimum_temperature_2m_above_ground": "degC",
        "specific_humidity_2m_above_ground": "kg kg-1",
        "relative_humidity_2m_above_ground": "%",
        "u_component_of_wind_10m_above_ground": "m s-1",
        "v_component_of_wind_10m_above_ground": "m s-1",
        "u_component_of_wind_planetary_boundary_layer": "m s-1",
        "v_component_of_wind_planetary_boundary_layer": "m s-1",
        "gust": "m s-1",
        "total_precipitation_surface": "kg m-2",
        "precipitation_rate": "kg m-2 s-1",
        "precipitable_water_entire_atmosphere": "kg m-2",
        "total_cloud_cover_entire_atmosphere": "%",
        "downward_shortwave_radiation_flux": "W m-2",
        "downward_longwave_radiation_flux": "W m-2",
        "upward_shortwave_radiation_flux": "W m-2",
        "upward_longwave_radiation_flux": "W m-2",
        "planetary_boundary_layer_height": "m",
        "haines_index": "1",
        "ventilation_rate": "m2 s-1",
    }
    HELP = """gfs reads NOAA/GFS0P25 from Google Earth Engine (free for research and non-commercial use):
  1. Register a Google Cloud project for Earth Engine: https://code.earthengine.google.com/register
  2. Either create a service account in that project, give it the 'Earth Engine Resource Viewer' role and put
     its JSON key in GEE_SERVICE_ACCOUNT_KEY (a Kaggle secret of that name), or run `earthengine authenticate`
     on this machine.
  3. Set GEE_PROJECT to the project id."""
    HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"

    def __init__(self, ee) -> None:
        self.ee = ee

    @classmethod
    def check_ready(cls) -> None:
        try:
            import ee  # noqa: F401
        except ImportError as ex:
            raise ChannelUnavailable(
                "gfs reads NOAA/GFS0P25 from Google Earth Engine: pip install earthengine-api"
            ) from ex
        if not (
            os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
            or (Path.home() / ".config" / "earthengine" / "credentials").exists()
        ):
            raise ChannelUnavailable(cls.HELP)

    @classmethod
    def connect(cls) -> EarthEngineGfs:
        """Earth Engine on its high-volume endpoint: a service account's JSON key (GEE_SERVICE_ACCOUNT_KEY) or this
        machine's `earthengine authenticate` credentials, in GEE_PROJECT (default: the key's own project)."""
        import ee

        key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
        project = os.environ.get("GEE_PROJECT") or None
        if key:
            info = json.loads(key)
            cred, project = (
                ee.ServiceAccountCredentials(info["client_email"], key_data=key),
                project or info.get("project_id"),
            )
        else:
            cred = "persistent"
        try:
            ee.Initialize(cred, project=project, url=cls.HIGH_VOLUME)  # earthengine-api 1.x
        except TypeError:
            ee.Initialize(cred, project=project, opt_url=cls.HIGH_VOLUME)  # 0.1.x
        return cls(ee)

    @classmethod
    def max_lead(cls, run: pd.Timestamp) -> int:
        return [h for d, h in cls.MAX_LEAD if Utc.stamp(run) >= pd.Timestamp(d, tz="UTC")][-1]

    def grid(self) -> tuple[str, tuple[float, ...], int, int]:
        """(crs, crs_transform (a, b, c, d, e, f), width, height) of the collection's native 0.25 deg grid."""
        band = self.ee.ImageCollection(self.COLLECTION).first().getInfo()["bands"][0]
        w, h = band["dimensions"]
        return band["crs"], tuple(float(v) for v in band["crs_transform"]), int(w), int(h)

    def fetch(self, run: pd.Timestamp, leads: Sequence[int], win: PixelWindow, grid: tuple) -> dict[int, dict]:
        """{lead: {band: 2-D array}} of one run over win; leads the collection lacks are absent, a run it lacks
        gives {}."""
        ee = self.ee
        crs, (a, b, c, d, e, f), _, _ = grid
        r0, r1, c0, c1 = win
        coll = (
            ee.ImageCollection(self.COLLECTION)
            .filter(ee.Filter.eq("creation_time", int(run.value // 10**6)))
            .filter(ee.Filter.inList("forecast_hours", list(leads)))
        )

        def tag(im):  # band -> f<lead>__band, so toBands' own prefix (the image id) does not matter
            pre = ee.String("f").cat(ee.Number(im.get("forecast_hours")).int().format("%03d")).cat("__")
            return im.rename(im.bandNames().map(lambda bn: pre.cat(ee.String(bn))))

        req = {
            "expression": coll.map(tag).toBands(),
            "fileFormat": "NPY",  # the bytes, read by decode(): earthengine-api's NUMPY_NDARRAY keeps numpy's limit
            "grid": {
                "dimensions": {"width": c1 - c0, "height": r1 - r0},
                "crsCode": crs,
                "affineTransform": {
                    "scaleX": a,
                    "shearX": b,
                    "translateX": c + c0 * a,
                    "shearY": d,
                    "scaleY": e,
                    "translateY": f + r0 * e,
                },
            },
        }
        for attempt in range(6):
            try:
                npy = ee.data.computePixels(req)
                break
            except Exception as ex:  # noqa: BLE001  (Earth Engine raises its own exception types)
                msg = str(ex).lower()
                if "no bands" in msg or "zero bands" in msg:
                    return {}
                if attempt == 5 or not any(
                    k in msg for k in ("429", "too many", "quota", "503", "500", "timed out", "deadline")
                ):
                    raise
                time.sleep(5 * 2**attempt)
        return self.decode(npy)

    @classmethod
    def decode(cls, npy: bytes) -> dict[int, dict]:
        """{lead: {band: 2-D array}} of computePixels' NPY bytes: one structured array, a field per band, its header
        allowed up to MAX_NPY_HEADER bytes (never unpickled)."""
        arr = np.load(io.BytesIO(npy), allow_pickle=False, max_header_size=cls.MAX_NPY_HEADER)
        out: dict[int, dict] = {}
        for name in arr.dtype.names or ():
            m = re.search(r"f(\d{3})__(\w+)$", name)
            if m:
                out.setdefault(int(m.group(1)), {}).setdefault(m.group(2), arr[name])
        return out

    # ------------------------------------------------------------------------------------- grid arithmetic

    @staticmethod
    def _east_lon(transform: Sequence[float], width: int) -> bool:
        """True when the grid's longitudes run 0..360 (then fire longitudes are shifted to match)."""
        a, _, c, *_ = transform
        return c > -90 and c + a * width > 270

    @classmethod
    def window(cls, transform: Sequence[float], width: int, height: int, box: BoundingBox) -> PixelWindow | None:
        """The cells covering the box on the regular lon/lat grid, snapped outward."""
        from affine import Affine

        w, s, e, n = box
        if cls._east_lon(transform, width):
            w, e = w % 360, e % 360
        return PixelWindow.covering(Affine(*transform), width, height, (w, s, e, n))

    @staticmethod
    def latlon(transform: Sequence[float], width: int, win: PixelWindow) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre lat / lon (2-D, longitudes in -180..180) of a window."""
        a, _, c, _, e, f = transform
        lon = c + a * (np.arange(win.c0, win.c1) + 0.5)
        lon = np.where(lon > 180, lon - 360, lon)
        lat = f + e * (np.arange(win.r0, win.r1) + 0.5)
        return np.meshgrid(lat, lon, indexing="ij")

    @classmethod
    def clusters(
        cls, wins: dict[str, PixelWindow], max_cells: int | None = None
    ) -> list[tuple[PixelWindow, list[str]]]:
        """[(union window, [fire_id])]: fires grouped greedily, west to east, so one request covers each group and
        no group's window exceeds max_cells."""
        max_cells = cls.MAX_CELLS if max_cells is None else max_cells
        groups: list[list] = []
        for fid, w in sorted(wins.items(), key=lambda kv: (kv[1][2], kv[1][0], kv[0])):
            for g in groups:
                u = g[0].union(w)
                if u.cells() <= max_cells:
                    g[0] = u
                    g[1].append(fid)
                    break
            else:
                groups.append([w, [fid]])
        return [(g[0], g[1]) for g in groups]

    @classmethod
    def lead_chunks(cls, leads: Sequence[int]) -> list[tuple[int, ...]]:
        """Leads in chunks that keep a request under MAX_BANDS."""
        k = max(1, cls.MAX_BANDS // len(cls.UNITS))
        return [tuple(leads[i : i + k]) for i in range(0, len(leads), k)]
