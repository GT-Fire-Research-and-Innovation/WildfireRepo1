"""GOES-16/17/18/19 ABI L2 fire product (FDCC) in NOAA's public S3 buckets: the scans of an hour, read whole."""

from __future__ import annotations

import logging
import math
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fah.sources.http import HttpClient

log = logging.getLogger(__name__)


class GoesArchive:
    """Scans are read in full and the file closed again, so a file with a damaged compressed chunk (it opens fine
    and fails only when that chunk is read: 'RuntimeError: NetCDF: HDF error') fails as a whole, before any fire's
    series gets part of it; the next nearest scan of the hour is tried instead."""

    # (satellite, bucket, first day, last day) of the fire product
    BUCKETS = [
        ("G16", "noaa-goes16", "2017-06-01", "2025-04-06"),
        ("G19", "noaa-goes19", "2025-04-07", "2030-01-01"),
        ("G17", "noaa-goes17", "2018-11-15", "2023-01-09"),
        ("G18", "noaa-goes18", "2023-01-10", "2030-01-01"),
    ]
    START = "2017-06-01"
    VARS = ("Mask", "Power", "Area", "Temp", "DQF")
    PRODUCT = "ABI-L2-FDCC"

    def __init__(self, http: HttpClient, tmp_dir: Path) -> None:
        self.http, self.tmp_dir = http, Path(tmp_dir)
        self._grids: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def satellites(cls, t: pd.Timestamp) -> list[tuple[str, str]]:
        """[(satellite, bucket)] operating at t."""
        return [(s, b) for s, b, a, z in cls.BUCKETS if pd.Timestamp(a, tz="UTC") <= t <= pd.Timestamp(z, tz="UTC")]

    @classmethod
    def day_prefix(cls, t: pd.Timestamp) -> str:
        return f"{cls.PRODUCT}/{t.year}/{t.day_of_year:03d}/"

    @staticmethod
    def scan_time(key: str) -> int:
        """Scan start (ns, UTC) from an object key's _s<YYYYJJJHHMMSSs> part."""
        return pd.to_datetime(key.split("_s")[1][:13], format="%Y%j%H%M%S", utc=True).value

    def hour(self, t: pd.Timestamp, sat: str, bucket: str) -> tuple[list[str], bytes | None]:
        """The hour's scan keys, nearest to t first, and the nearest one's bytes; ([], None) when nothing came."""
        prefix = f"{self.day_prefix(t)}{t.hour:02d}/"
        for attempt in range(3):
            try:
                r = self.http.session.get(
                    f"https://{bucket}.s3.amazonaws.com/", params={"list-type": 2, "prefix": prefix}, timeout=60
                )
                r.raise_for_status()
                keys = sorted(
                    re.findall(r"<Key>([^<]+)</Key>", r.text), key=lambda k: abs(int(k.split("_s")[1][9:11]) - t.minute)
                )
                if not keys:
                    return [], None
                return keys, self.http.s3_object(bucket, keys[0])
            except Exception as ex:  # noqa: BLE001  (an hour that never lists is left out)
                if attempt == 2:
                    log.info(f"    GOES {sat} {t}: {str(ex)[:120]}")
                    return [], None
                time.sleep(3 * (attempt + 1))
        return [], None

    def nearest_readable_scan(
        self, sat: str, bucket: str, keys: list[str], data: bytes | None, tries: int = 3
    ) -> tuple[dict | None, str | None]:
        """(scan, key) of the nearest scan that reads, trying up to `tries` files; (None, None) when none does.
        A scan is {"vars": {name: (array, dtype, attrs)} packed as in the file, "x", "y", "proj": the projection's
        attributes, "latlon": (lat, lon) of its grid}."""
        for i, key in enumerate(keys[:tries]):
            try:
                if i:
                    data = self.http.s3_object(bucket, key)
                if data is None:
                    break
                return self._read(data, sat), key
            except (OSError, RuntimeError, KeyError, ValueError, IndexError) as ex:
                then = "trying the next scan of the hour" if i + 1 < min(tries, len(keys)) else "hour left out"
                name = key.rsplit("/", 1)[-1]
                log.info(f"    GOES {sat} {name}: unreadable ({type(ex).__name__}: {str(ex)[:100]}), {then}")
        return None, None

    def open_bytes(self, data: bytes, netcdf=None):
        """A downloaded netCDF opened from memory, or, when this netCDF4 build cannot (some preinstalled builds),
        from a temporary file. -> (dataset, temporary path or None)"""
        if netcdf is None:
            import netCDF4 as netcdf
        try:
            return netcdf.Dataset("mem", memory=data), None
        except (OSError, ValueError, RuntimeError):
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            fd, p = tempfile.mkstemp(suffix=".nc", dir=self.tmp_dir)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            return netcdf.Dataset(p), Path(p)

    def _read(self, data: bytes, sat: str) -> dict:
        ds, tmp = self.open_bytes(data)
        try:
            shape = ds.variables["Mask"].shape
            if (sat, shape) not in self._grids:
                self._grids[(sat, shape)] = self.latlon(ds)
            proj = ds.variables["goes_imager_projection"]
            scan = {
                "vars": {},
                "latlon": self._grids[(sat, shape)],
                "x": ds.variables["x"][:],
                "y": ds.variables["y"][:],
                "proj": {a: proj.getncattr(a) for a in proj.ncattrs()},
            }
            for nm in self.VARS:
                if nm in ds.variables:
                    v = ds.variables[nm]
                    v.set_auto_maskandscale(False)
                    scan["vars"][nm] = (v[:], v.dtype, {a: v.getncattr(a) for a in v.ncattrs()})
            return scan
        finally:
            ds.close()
            if tmp:
                tmp.unlink(missing_ok=True)

    @staticmethod
    def latlon(ds) -> tuple[np.ndarray, np.ndarray]:
        """Latitude / longitude of the ABI fixed grid (GOES-R PUG formulas)."""
        p = ds.variables["goes_imager_projection"]
        H = p.perspective_point_height + p.semi_major_axis
        req, rpol = p.semi_major_axis, p.semi_minor_axis
        lon0 = math.radians(p.longitude_of_projection_origin)
        x = ds.variables["x"][:].astype(np.float64)
        y = ds.variables["y"][:].astype(np.float64)
        X, Y = np.meshgrid(x, y)
        a = np.sin(X) ** 2 + np.cos(X) ** 2 * (np.cos(Y) ** 2 + (req / rpol) ** 2 * np.sin(Y) ** 2)
        b = -2 * H * np.cos(X) * np.cos(Y)
        c = H**2 - req**2
        with np.errstate(invalid="ignore"):
            rs = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            sx = rs * np.cos(X) * np.cos(Y)
            sy = -rs * np.sin(X)
            sz = rs * np.cos(X) * np.sin(Y)
            lat = np.degrees(np.arctan((req / rpol) ** 2 * sz / np.sqrt((H - sx) ** 2 + sy**2)))
            lon = np.degrees(lon0 - np.arctan(sy / (H - sx)))
        return lat.astype(np.float32), lon.astype(np.float32)
