"""NOAA HRRR on AWS: chosen GRIB2 messages of a file, fetched by the byte ranges in its .idx, on the native 3 km
Lambert grid."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

from fah.core.clock import Utc
from fah.core.geometry import BoundingBox, PixelWindow

log = logging.getLogger(__name__)


class HrrrArchive:
    """Fields are [(name, .idx key)]: the first message whose .idx line holds the key."""

    URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d}/conus/hrrr.t{h:02d}z.wrfsfcf{f:02d}.grib2"
    BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
    START = "2014-08-01"  # first HRRR files on AWS used by the built dataset
    FIELDS = [
        ("gust", ":GUST:surface:"),
        ("u10", ":UGRD:10 m above ground:"),
        ("v10", ":VGRD:10 m above ground:"),
        ("rh2", ":RH:2 m above ground:"),
    ]
    # APCP is the message accumulated from the run start ('0-9 hour acc', '0-1 day acc'), not the last hour's.
    FORECAST_FIELDS = FIELDS + [
        ("t2", ":TMP:2 m above ground:"),
        ("q2", ":SPFH:2 m above ground:"),
        ("apcp", ":APCP:surface:0-"),
    ]
    UNITS = {
        "gust": "m s-1",
        "u10": "m s-1",
        "v10": "m s-1",
        "rh2": "%",
        "t2": "K",
        "q2": "kg kg-1",
        "apcp": "kg m-2 (accumulated from the run start to the valid time)",
    }
    # How far the 12z run reached: 15 h in HRRRv1, 18 h from HRRRv2 (2016-08-23), 36 h from HRRRv3 (2018-07-12).
    MAX_LEAD = (("2014-01-01", 15), ("2016-08-23", 18), ("2018-07-12", 36))
    PAD_DEG = 0.05

    def __init__(self, session: requests.Session) -> None:
        self.session = session

    @classmethod
    def url(cls, run: pd.Timestamp, lead_h: int) -> str:
        return cls.URL.format(d=run.strftime("%Y%m%d"), h=run.hour, f=lead_h)

    @classmethod
    def max_lead(cls, run: pd.Timestamp) -> int:
        return [h for d, h in cls.MAX_LEAD if Utc.stamp(run) >= pd.Timestamp(d, tz="UTC")][-1]

    @classmethod
    def window(cls, lat: np.ndarray, lon: np.ndarray, box: BoundingBox) -> PixelWindow | None:
        """The grid cells covering the box plus PAD_DEG."""
        return PixelWindow.of_points(lat, lon, box, cls.PAD_DEG)

    def byte_ranges(self, url: str, fields: list[tuple[str, str]] | None = None) -> dict[str, tuple] | None:
        """{name: (first byte, last byte or None)} from the file's .idx, or None when the file is missing."""
        r = self.session.get(url + ".idx", timeout=60)
        if r.status_code != 200:
            return None
        lines = r.text.strip().splitlines()
        starts = [int(line.split(":")[1]) for line in lines]
        rng = {}
        for i, line in enumerate(lines):
            for name, key in fields or self.FIELDS:
                if key in line and name not in rng:
                    rng[name] = (starts[i], starts[i + 1] - 1 if i + 1 < len(lines) else None)
        return rng

    def fetch(self, url: str, fields: list[tuple[str, str]] | None = None) -> dict[str, np.ndarray] | None:
        """{name: 2-D float32 array} of one file (CONUS grid), or None when unavailable."""
        import pygrib

        for tries in range(4):
            try:
                rng = self.byte_ranges(url, fields)
                if not rng:
                    return None
                out = {}
                for name, (a, b) in rng.items():
                    r = self.session.get(url, headers={"Range": f"bytes={a}-{b if b is not None else ''}"}, timeout=120)
                    r.raise_for_status()
                    out[name] = pygrib.fromstring(r.content).values.astype(np.float32)
                return out
            except Exception as ex:  # noqa: BLE001  (retried; a file that never reads is left out)
                if tries == 3:
                    log.info(f"    HRRR failed {url}: {ex}")
                    return None
                time.sleep(5 * (tries + 1))
        return None

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon) of the CONUS grid, from one message of a reference file."""
        import pygrib

        url = self.URL.format(d="20200820", h=12, f=0)
        a, b = self.byte_ranges(url)["u10"]
        r = self.session.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
        return pygrib.fromstring(r.content).latlons()
