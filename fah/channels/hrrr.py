"""hrrr: HRRR analyses (f00) every 3 hours of the window and the 3 days before it, on the native 3 km grid."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fah.channels.base import SharedDownloadChannel
from fah.core.clock import TimeSpan, Utc
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.result import FetchResult
from fah.sources.hrrr import HrrrArchive
from fah.sources.http import ParallelMap
from fah.storage.ncseries import NcSeriesWriter


class HrrrChannel(SharedDownloadChannel):
    """Two files, the same valid times (every 3 h from SPINUP_H before the window to its end), each field decoded from
    the file's own GRIB2 message and cropped to the region:
      hrrr_ana.nc  the analysis (f00): gust, u10, v10, rh2, t2, dpt, dswrf (dswrf where the analysis holds it)
      hrrr_f03.nc  the 3 h forecast (f03) of the run 3 h earlier: apcp (rain of the 3 h before), dswrf (every version)
    The spin-up lets an hourly fine-fuel moisture model settle before the window's first day. The 12z run's longer
    forecasts are hrrr_next's."""

    name = "hrrr"
    time_dependent = True
    required_files = frozenset({"hrrr_ana.nc", "hrrr_f03.nc"})
    GROUP_SIZE = 25
    STEP_H = 3
    SPINUP_H = 72
    # file: (hours back to the run, lead hours, fields, title, the fields' GRIB2 names)
    FILES = {
        "hrrr_ana.nc": (0, 0, HrrrArchive.ANALYSIS_FIELDS, "HRRR analyses (f00)",
                        "GUST surface; UGRD/VGRD 10 m; RH/TMP/DPT 2 m; DSWRF surface"),
        "hrrr_f03.nc": (3, 3, HrrrArchive.PREVIOUS_F03_FIELDS, "HRRR 3 h forecasts (f03) of the run 3 h earlier",
                        "APCP surface 0-3 h accumulation; DSWRF surface"),
    }  # fmt: skip

    def coverage(self, fire: Fire) -> TimeSpan:
        span = fire.span()
        return TimeSpan(span.start - pd.Timedelta(hours=self.SPINUP_H), span.end)

    def window_label(self, fire: Fire) -> str:
        return self.coverage(fire).label()

    def times(self, fire: Fire) -> list[pd.Timestamp]:
        return [t for t in self.coverage(fire).hours(self.STEP_H) if t >= Utc.stamp(HrrrArchive.START)]

    def skip_reason(self, fire: Fire) -> str | None:
        if fire.span().end < pd.Timestamp(HrrrArchive.START, tz="UTC"):
            return "HRRR archive starts 2014-08"
        return None

    def check_ready(self) -> None:
        self.require_module("pygrib", "hrrr decodes GRIB2 messages: pip install pygrib")

    def fetch(self, fires: list[Fire], ctx: RunContext) -> Iterator[FetchResult]:
        archive = HrrrArchive(ctx.session)
        lat, lon = ctx.memo("hrrr_grid", archive.grid)

        def fetch_file(key: tuple[pd.Timestamp, str]) -> tuple:
            t, file = key
            back, lead, fields, _, _ = self.FILES[file]
            url = HrrrArchive.url(t - pd.Timedelta(hours=back), lead)
            return key, url, archive.fetch(url, fields)

        for group in self.groups(fires, self.GROUP_SIZE):
            if ctx.expired():
                return
            need: dict[tuple[pd.Timestamp, str], list[Fire]] = {}
            left, wins, series, got = {}, {}, {}, {}
            for f in group:
                fid = f.fire_id
                keys = [(t, file) for t in self.times(f) for file in self.FILES]
                wins[fid] = HrrrArchive.window(lat, lon, f.region())
                left[fid], got[fid] = len(keys), []
                if not keys:
                    yield FetchResult(f, "empty")
                for k in keys:
                    need.setdefault(k, []).append(f)
            stopped = False
            with ThreadPoolExecutor(ctx.threads) as ex:
                for (t, file), url, values in ParallelMap.windowed(ex, fetch_file, sorted(need), 2 * ctx.threads):
                    for f in need[(t, file)]:
                        fid = f.fire_id
                        win = wins[fid]
                        if values and win is not None:
                            rows, cols = win.slices()
                            if (fid, file) not in series:
                                _, _, _, title, names = self.FILES[file]
                                series[(fid, file)] = NcSeriesWriter(
                                    ctx.stage(fid) / file,
                                    lat[rows, cols],
                                    lon[rows, cols],
                                    {
                                        "title": f"{title} cropped to {fid}",
                                        "fire_id": fid,
                                        "source": HrrrArchive.BUCKET,
                                        "fields": names,
                                        "time": "valid time",
                                        "window_rows_cols": win.label(),
                                    },
                                )
                            series[(fid, file)].append(
                                t.value,
                                url,
                                {n: (v[rows, cols], "f4", {"units": HrrrArchive.UNITS[n]}) for n, v in values.items()},
                            )
                            got[fid].append(url)
                        left[fid] -= 1
                        if left[fid] == 0:
                            for key in [k for k in series if k[0] == fid]:
                                series.pop(key).close()
                            yield FetchResult(f, "ok" if got[fid] else "empty", tuple(sorted(got[fid])))
                    if ctx.expired():
                        stopped = True
                        break
            if stopped:
                for w in series.values():
                    w.close()
                for fid in {fid for fid, _ in series}:
                    ctx.drop_stage(fid)
                return
