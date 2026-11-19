"""NASA FIRMS area API: VIIRS 375 m active-fire detections. Limit: 5000 transactions per 10 minutes per key."""

from __future__ import annotations

import io
import time

import pandas as pd
import requests

from fah.core.errors import ChannelUnavailable
from fah.core.geometry import BoundingBox
from fah.core.secrets import Secrets


class FirmsClient:
    """Area queries of up to 5 days. The map key comes from FIRMS_MAP_KEY (environment or .env)."""

    API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/{days}"
    STANDARD = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")
    NEAR_REAL_TIME = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")

    def __init__(self, session: requests.Session) -> None:
        self.session = session

    @staticmethod
    def map_key() -> str:
        key = Secrets.lookup("FIRMS_MAP_KEY")
        if not key:
            raise ChannelUnavailable("Set FIRMS_MAP_KEY (environment or .env).")
        return key

    @classmethod
    def url(cls, src: str, box: BoundingBox, days: int, day: str | None = None, key: str | None = None) -> str:
        """The area call; with key None, the call as a manifest cites it (key replaced by {MAP_KEY} on redaction)."""
        u = cls.API.format(key=key or "KEY", src=src, bbox=",".join(f"{v:.4f}" for v in box), days=days)
        return u + f"/{day}" if day else u

    def area(self, src: str, box: BoundingBox, days: int, day: str | None = None) -> pd.DataFrame:
        """The detections of one call, as FIRMS returns them, plus 'source'."""
        url = self.url(src, box, days, day, key=self.map_key())
        for t in range(4):
            r = self.session.get(url, timeout=120)
            if r.status_code == 200 and not r.text.startswith("Invalid"):
                df = pd.read_csv(io.StringIO(r.text)) if r.text.strip() else pd.DataFrame()
                if len(df):
                    df["source"] = src
                return df
            time.sleep(10 * (t + 1))
        raise RuntimeError(f"FIRMS {src}: HTTP {r.status_code} {r.text[:200]}")
