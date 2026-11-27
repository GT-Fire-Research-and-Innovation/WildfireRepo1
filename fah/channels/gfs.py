"""gfs: NOAA GFS 0.25 deg (TS-SatFire's forecast source), the 12z run's forecasts for each day of the window."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from fah.channels.base import SharedDownloadChannel
from fah.core.clock import Utc
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.result import FetchResult
from fah.core.secrets import Secrets
from fah.sources.gfs import EarthEngineGfs
from fah.sources.http import ParallelMap
from fah.storage.ncseries import NcSeriesWriter


class GfsChannel(SharedDownloadChannel):
    """gfs.nc with time = valid time, lead_h and run_time per step: every 3 h out to 10 days where the collection has
    them, every band each image has, on the native grid (no resampling; values as Earth Engine serves them,
    temperatures in degC). Fires of the same run share requests."""

    name = "gfs"
    time_dependent = True
    GROUP_SIZE = 200
    RUN_HOUR = 12
    LEADS = tuple(range(3, 241, 3))  # every 3 h out to 10 days

    def skip_reason(self, fire: Fire) -> str | None:
        if fire.span().end < pd.Timestamp(EarthEngineGfs.START, tz="UTC"):
            return "GFS 0.25 deg in Earth Engine starts 2015-07-01"
        return None

    def check_ready(self) -> None:
        EarthEngineGfs.check_ready()

    @classmethod
    def times(cls, fire: Fire) -> list[tuple[pd.Timestamp, int]]:
        """[(12z run, lead hours)]: each day of the window the 12z run's forecasts every 3 h, out to 10 days where the
        collection has them (36 h before 2017-07-09)."""
        start = Utc.stamp(EarthEngineGfs.START)
        return [
            (run, h)
            for run in fire.span().daily(cls.RUN_HOUR)
            if run >= start
            for h in cls.LEADS
            if h <= EarthEngineGfs.max_lead(run)
        ]

    def fetch(self, fires: list[Fire], ctx: RunContext) -> Iterator[FetchResult]:
        gfs = ctx.memo("gee", EarthEngineGfs.connect)
        grid = ctx.memo("gfs_grid", gfs.grid)
        crs, tr, width, height = grid

        for group in self.groups(fires, self.GROUP_SIZE):
            if ctx.expired():
                return
            need: dict[pd.Timestamp, list[str]] = {}
            left, wins, series, got, by_id, errs = {}, {}, {}, {}, {}, {}
            for f in group:
                fid = f.fire_id
                runs = sorted({r for r, _ in self.times(f)})
                wins[fid] = EarthEngineGfs.window(tr, width, height, f.region())
                if not runs or wins[fid] is None:
                    yield FetchResult(f, "empty")
                    continue
                by_id[fid], left[fid], got[fid], errs[fid] = f, len(runs), [], []
                for r in runs:
                    need.setdefault(r, []).append(fid)

            def fetch_run(run: pd.Timestamp, wins: dict = wins, need: dict = need) -> tuple:
                leads = [h for h in self.LEADS if h <= EarthEngineGfs.max_lead(run)]
                out = []
                for cwin, ids in EarthEngineGfs.clusters({fid: wins[fid] for fid in need[run]}):
                    data, err = {}, None
                    try:
                        for chunk in EarthEngineGfs.lead_chunks(leads):
                            for h, bands in gfs.fetch(run, chunk, cwin, grid).items():
                                data.setdefault(h, {}).update(bands)
                    except Exception as ex:  # noqa: BLE001  (reported for the fires of this run; the others go on)
                        data, err = {}, Secrets.redact(f"{run:%Y-%m-%dT%HZ}: {type(ex).__name__}: {ex}")[:300]
                    out.append((cwin, ids, data, err))
                return run, out

            stopped = False
            threads = min(ctx.threads, EarthEngineGfs.THREADS)
            with ThreadPoolExecutor(threads) as ex:
                for run, out in ParallelMap.windowed(ex, fetch_run, sorted(need), 2 * threads):
                    src = f"ee:{EarthEngineGfs.COLLECTION} creation_time={run.strftime('%Y-%m-%dT%H:%MZ')}"
                    for cwin, ids, data, err in out:
                        for fid in ids:
                            if err:
                                errs[fid].append(err)
                            if data and fid not in series:
                                series[fid] = self._open_series(ctx, fid, wins[fid], grid)
                            sub = wins[fid].slices_within(cwin)
                            for h in sorted(data):
                                series[fid].append(
                                    (run + pd.Timedelta(hours=h)).value,
                                    f"{src} forecast_hours={h}",
                                    {
                                        b: (
                                            np.asarray(v)[sub],
                                            np.asarray(v).dtype,
                                            {"units": EarthEngineGfs.UNITS.get(b, "")},
                                        )
                                        for b, v in sorted(data[h].items())
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
                            if data:
                                got[fid].append(src)
                            left[fid] -= 1
                            if left[fid] == 0:
                                if fid in series:
                                    series.pop(fid).close()
                                e = errs[fid]
                                yield FetchResult(
                                    by_id[fid],
                                    "ok" if got[fid] else "error" if e else "empty",
                                    tuple(sorted(got[fid])),
                                    f"{len(e)} run(s) failed: {e[0]}" if e else "",
                                )
                    if ctx.expired():
                        stopped = True
                        break
            if stopped:
                for fid, s in series.items():
                    s.close()
                    ctx.drop_stage(fid)
                return

    @staticmethod
    def _open_series(ctx: RunContext, fid: str, win, grid: tuple) -> NcSeriesWriter:
        crs, tr, width, _ = grid
        lat, lon = EarthEngineGfs.latlon(tr, width, win)
        return NcSeriesWriter(
            ctx.stage(fid) / "gfs.nc",
            lat,
            lon,
            {
                "title": f"NOAA GFS 0.25 deg, 12z run, forecasts every 3 h out to 10 days, cropped to {fid}",
                "fire_id": fid,
                "source": f"Google Earth Engine {EarthEngineGfs.COLLECTION} (NOAA NCEP GFS)",
                "crs": crs,
                "crs_transform": " ".join(f"{v:g}" for v in tr),
                "window_rows_cols": win.label(),
                "total_precipitation_surface": "accumulated over the last ((lead_h - 1) % 6) + 1 hours "
                "(Earth Engine catalog); sum the 6-hourly leads for totals",
                "note": "values as Earth Engine serves them (temperatures in degC); bands added to the "
                "collection on 2025-01-15 appear from then on",
            },
        )
