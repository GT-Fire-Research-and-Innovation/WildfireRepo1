"""hrrr: HRRR analyses (f00) every 3 hours of the window, on the native 3 km grid."""

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


class HrrrChannel(SharedDownloadChannel):
    """hrrr_ana.nc: gust, u10, v10, rh2 decoded from each file's own GRIB2 messages and cropped to the region. The 12z
    run's forecasts are hrrr_next's (its lead_h = 9 step is the file, fields and crop an older hrrr_fc.nc held)."""

    name = "hrrr"
    time_dependent = True
    GROUP_SIZE = 25
    STEP_H = 3

    def skip_reason(self, fire: Fire) -> str | None:
        if fire.span().end < pd.Timestamp(HrrrArchive.START, tz="UTC"):
            return "HRRR archive starts 2014-08"
        return None

    def check_ready(self) -> None:
        self.require_module("pygrib", "hrrr decodes GRIB2 messages: pip install pygrib")

    def fetch(self, fires: list[Fire], ctx: RunContext) -> Iterator[FetchResult]:
        archive = HrrrArchive(ctx.session)
        lat, lon = ctx.memo("hrrr_grid", archive.grid)
        names = [n for n, _ in HrrrArchive.FIELDS]

        def fetch_file(t: pd.Timestamp) -> tuple:
            url = HrrrArchive.url(t, 0)
            return t, url, archive.fetch(url)

        for group in self.groups(fires, self.GROUP_SIZE):
            if ctx.expired():
                return
            need: dict[pd.Timestamp, list[Fire]] = {}
            left, wins, series, got = {}, {}, {}, {}
            for f in group:
                fid = f.fire_id
                times = [t for t in f.span().hours(self.STEP_H) if t >= Utc.stamp(HrrrArchive.START)]
                wins[fid] = HrrrArchive.window(lat, lon, f.region())
                left[fid], got[fid] = len(times), []
                if not times:
                    yield FetchResult(f, "empty")
                for t in times:
                    need.setdefault(t, []).append(f)
            stopped = False
            with ThreadPoolExecutor(ctx.threads) as ex:
                for t, url, fields in ParallelMap.windowed(ex, fetch_file, sorted(need), 2 * ctx.threads):
                    for f in need[t]:
                        fid = f.fire_id
                        win = wins[fid]
                        if fields and win is not None:
                            rows, cols = win.slices()
                            if fid not in series:
                                series[fid] = NcSeriesWriter(
                                    ctx.stage(fid) / "hrrr_ana.nc",
                                    lat[rows, cols],
                                    lon[rows, cols],
                                    {
                                        "title": f"HRRR analyses (f00) cropped to {fid}",
                                        "fire_id": fid,
                                        "source": HrrrArchive.BUCKET,
                                        "fields": "GUST surface; UGRD/VGRD 10 m; RH 2 m",
                                        "window_rows_cols": win.label(),
                                    },
                                )
                            series[fid].append(
                                t.value, url, {n: (fields[n][rows, cols], "f4", {}) for n in names if n in fields}
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
