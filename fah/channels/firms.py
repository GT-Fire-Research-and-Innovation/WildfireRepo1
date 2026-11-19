"""firms: NASA FIRMS VIIRS 375 m active-fire detections around the fire's perimeter days."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.secrets import Secrets
from fah.sources.firms import FirmsClient


class FirmsChannel(PerFireChannel):
    """<SOURCE>.csv.gz per satellite product: the standard product in 5-day calls over Fire.firms_days(), and for
    2024+ fires the near-real-time product for the days the standard one has not reached yet."""

    name = "firms"
    time_dependent = True
    CALL_GAP_S = 0.15  # 5000 transactions / 10 min per key

    def check_ready(self) -> None:
        FirmsClient.map_key()

    def window_label(self, fire: Fire) -> str:
        """FIRMS covers fixed days around the perimeters, not the window (it is the window's input)."""
        d = fire.firms_days()
        return f"{d[0].date().isoformat()}/{d[-1].date().isoformat()}"

    @staticmethod
    def calls(fire: Fire) -> list[tuple[str, str, int]]:
        """(source, first day, n days) of the standard-product requests: 5-day calls over firms_days()."""
        ds = fire.firms_days()
        srcs = [FirmsClient.STANDARD[0]] + ([FirmsClient.STANDARD[1]] if int(fire.year) >= 2018 else [])
        out = []
        for src in srcs:
            i = 0
            while i < len(ds):
                k = min(5, len(ds) - i)
                out.append((src, ds[i].date().isoformat(), k))
                i += k
        return out

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        client = FirmsClient(ctx.session)
        region = fire.region()
        frames: dict[str, list[pd.DataFrame]] = {}
        sources: list[str] = []

        def call(src: str, day: str, k: int) -> None:
            ctx.throttle(self.CALL_GAP_S)
            df = client.area(src, region, k, day)
            sources.append(Secrets.redact(client.url(src, region, k, day)))
            if len(df):
                frames.setdefault(src, []).append(df)

        for src, day, k in self.calls(fire):
            call(src, day, k)
        ds = fire.firms_days()
        if int(fire.year) >= 2024:
            sp = [pd.Timestamp(d["acq_date"].max()) for fs in frames.values() for d in fs]
            start = (max(sp) + pd.Timedelta(days=1)) if sp else ds[0]
            while start <= ds[-1]:
                k = min(5, (ds[-1] - start).days + 1)
                for src in FirmsClient.NEAR_REAL_TIME:
                    call(src, start.date().isoformat(), k)
                start += pd.Timedelta(days=k)
        n = 0
        for src, fs in frames.items():
            df = pd.concat(fs, ignore_index=True)
            n += len(df)
            with gzip.open(out / f"{src}.csv.gz", "wt") as fh:
                df.to_csv(fh, index=False)
        return ("ok" if n else "empty"), sources
