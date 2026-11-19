"""hrrr_next: the HRRR 12z run's forecasts every 3 h out to 36 h, for each day of the window."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fah.channels.base import SharedDownloadChannel
from fah.core.clock import Utc
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.result import FetchResult
from fah.sources.hrrr import HrrrArchive
from fah.sources.http import ParallelMap
from fah.storage.ncseries import NcSeriesWriter


class HrrrNextChannel(SharedDownloadChannel):
    """hrrr_next.nc with time = valid time, lead_h and run_time per step: gust, u10, v10, rh2, t2, q2 and apcp decoded
    from each file's own GRIB2 messages and cropped to the region on the native 3 km grid. The next-day forecast
    TS-SatFire takes from GFS, at HRRR's resolution."""

    name = "hrrr_next"
    time_dependent = True
    GROUP_SIZE = 25
    RUN_HOUR = 12
    LEADS = tuple(range(3, 37, 3))

    def skip_reason(self, fire: Fire) -> str | None:
        if fire.span().end < pd.Timestamp(HrrrArchive.START, tz="UTC"):
            return "HRRR archive starts 2014-08"
        return None

    def check_ready(self) -> None:
        self.require_module("pygrib", "hrrr_next decodes GRIB2 messages: pip install pygrib")

    @classmethod
    def times(cls, fire: Fire) -> list[tuple[pd.Timestamp, int]]:
        """[(12z run, lead hours)]: each day of the window the 12z run's forecasts every 3 h out to 36 h (the next
        day's afternoon), as far as that HRRR version ran."""
        return [
            (run, h) for run in fire.span().daily(cls.RUN_HOUR) for h in cls.LEADS if h <= HrrrArchive.max_lead(run)
        ]

    def fetch(self, fires: list[Fire], ctx: RunContext) -> Iterator[FetchResult]:
        archive = HrrrArchive(ctx.session)
        lat, lon = ctx.memo("hrrr_grid", archive.grid)
        fields = HrrrArchive.FORECAST_FIELDS
        names = [n for n, _ in fields]

        def fetch_file(key: tuple[pd.Timestamp, int]) -> tuple:
            run, h = key
            url = HrrrArchive.url(run, h)
            return key, url, archive.fetch(url, fields)

        for group in self.groups(fires, self.GROUP_SIZE):
            if ctx.expired():
                return
            need: dict[tuple, list[Fire]] = {}
            left, wins, series, got = {}, {}, {}, {}
            for f in group:
                fid = f.fire_id
                keys = [(run, h) for run, h in self.times(f) if run >= Utc.stamp(HrrrArchive.START)]
                wins[fid] = HrrrArchive.window(lat, lon, f.region())
                left[fid], got[fid] = len(keys), []
                if not keys:
                    yield FetchResult(f, "empty")
                for k in keys:
                    need.setdefault(k, []).append(f)
            stopped = False
            with ThreadPoolExecutor(ctx.threads) as ex:
                for (run, h), url, values in ParallelMap.windowed(ex, fetch_file, sorted(need), 2 * ctx.threads):
                    for f in need[(run, h)]:
                        fid = f.fire_id
                        win = wins[fid]
                        if values and win is not None:
                            rows, cols = win.slices()
                            if fid not in series:
                                series[fid] = NcSeriesWriter(
                                    ctx.stage(fid) / "hrrr_next.nc",
                                    lat[rows, cols],
                                    lon[rows, cols],
                                    {
                                        "title": f"HRRR 12z run, forecasts every 3 h out to 36 h, cropped to {fid}",
                                        "fire_id": fid,
                                        "source": HrrrArchive.BUCKET,
                                        "fields": "GUST surface; UGRD/VGRD 10 m; RH 2 m; TMP 2 m; SPFH 2 m; "
                                        "APCP surface accumulated from the run start",
                                        "window_rows_cols": win.label(),
                                    },
                                )
                            series[fid].append(
                                (run + pd.Timedelta(hours=h)).value,
                                url,
                                {
                                    n: (values[n][rows, cols], "f4", {"units": HrrrArchive.UNITS[n]})
                                    for n in names
                                    if n in values
                                },
                                extra={
                                    "lead_h": (h, "i2", {"long_name": "forecast lead", "units": "hours"}),
                                    "run_time": (
                                        run.value,
                                        "i8",
                                        {
                                            "long_name": "model run (initialisation) time",
                                            "units": "nanoseconds since 1970-01-01 00:00:00 UTC",
                                        },
                                    ),
                                },
                            )
                            got[fid].append(url)
                        left[fid] -= 1
                        if left[fid] == 0:
                            if fid in series:
                                series.pop(fid).close()
                            yield FetchResult(f, "ok" if got[fid] else "empty", tuple(sorted(got[fid])))
                    if ctx.expired():
                        stopped = True
                        break
            if stopped:
                for fid, s in series.items():
                    s.close()
                    ctx.drop_stage(fid)
                return
