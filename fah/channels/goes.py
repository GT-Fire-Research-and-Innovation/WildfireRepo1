"""goes: GOES ABI L2 fire product (FDCC), the scan nearest each hour of the window, per satellite."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fah.channels.base import SharedDownloadChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.geometry import PixelWindow
from fah.core.result import FetchResult
from fah.sources.goes import GoesArchive
from fah.sources.http import HttpClient, ParallelMap
from fah.storage.ncseries import NcSeriesWriter


class GoesChannel(SharedDownloadChannel):
    """goes_<sat>.nc: per fire and satellite, the scan nearest each hour cropped to the region on the ABI fixed grid,
    with the fire product's own variables (Mask, Power, Area, Temp, DQF) left packed as in the source. A scan that is
    corrupt in the bucket is replaced by the next nearest of the same hour."""

    name = "goes"
    time_dependent = True
    GROUP_SIZE = 25

    def skip_reason(self, fire: Fire) -> str | None:
        if fire.span().end < pd.Timestamp(GoesArchive.START, tz="UTC"):
            return "GOES-16 fire product starts 2017-06-01"
        return None

    @staticmethod
    def hours(fire: Fire) -> list[tuple[pd.Timestamp, list[tuple[str, str]]]]:
        """[(hour, [(satellite, bucket)])] hourly across the window, with the satellites operating then."""
        out = []
        for t in fire.span().hours(1):
            sats = GoesArchive.satellites(t)
            if sats:
                out.append((t, sats))
        return out

    def fetch(self, fires: list[Fire], ctx: RunContext) -> Iterator[FetchResult]:
        archive = GoesArchive(HttpClient(ctx.session), ctx.tmp())
        for group in self.groups(fires, self.GROUP_SIZE):
            if ctx.expired():
                return
            need: dict[tuple, list[Fire]] = {}
            left: dict[str, int] = {}
            for f in group:
                hrs = self.hours(f)
                left[f.fire_id] = sum(len(s) for _, s in hrs)
                for t, sats in hrs:
                    for sat, bucket in sats:
                        need.setdefault((t, sat, bucket), []).append(f)
            series: dict[tuple[str, str], tuple[NcSeriesWriter, PixelWindow] | None] = {}
            scans: dict[str, dict[tuple[str, str], int]] = {f.fire_id: {} for f in group}

            def fetch_hour(item: tuple) -> tuple:
                t, sat, bucket = item
                return (item, *archive.hour(t, sat, bucket))

            stopped = False
            with ThreadPoolExecutor(ctx.threads) as ex:
                for (t, sat, bucket), keys, data in ParallelMap.windowed(ex, fetch_hour, sorted(need), 4 * ctx.threads):
                    scan, key = archive.nearest_readable_scan(sat, bucket, keys, data)
                    if scan is not None:
                        for f in need[(t, sat, bucket)]:
                            self._append(f, sat, bucket, t, scan, key, series, scans, ctx)
                    for f in need[(t, sat, bucket)]:
                        fid = f.fire_id
                        left[fid] -= 1
                        if left[fid] == 0:
                            for (ff, _), v in list(series.items()):
                                if ff == fid and v is not None:
                                    v[0].close()
                            # one listing link per satellite-day (the exact scan of each hour is in the file's `source`)
                            yield FetchResult(
                                f,
                                "ok" if scans[fid] else "empty",
                                tuple(
                                    f"https://{b}.s3.amazonaws.com/?list-type=2&prefix={d}"
                                    for b, d in sorted(scans[fid])
                                ),
                            )
                    if ctx.expired():
                        stopped = True
                        break
            if stopped:
                for (fid, _), v in series.items():
                    if v is not None and left.get(fid, 0) > 0:
                        v[0].close()
                        ctx.drop_stage(fid)
                return

    @staticmethod
    def _append(
        f: Fire,
        sat: str,
        bucket: str,
        t: pd.Timestamp,
        scan: dict,
        key: str,
        series: dict,
        scans: dict,
        ctx: RunContext,
    ) -> None:
        """Add one scan to the fire's series of that satellite, opening the series (and its window) on first use."""
        fid = f.fire_id
        k = (fid, sat)
        if k not in series:
            lat, lon = scan["latlon"]
            win = PixelWindow.of_points(lat, lon, f.region())
            if win is None:
                series[k] = None
            else:
                r0, r1, c0, c1 = win
                static = {
                    "x": (("x",), scan["x"][c0:c1].astype("f8"), {"units": "rad", "long_name": "ABI fixed grid x"}),
                    "y": (("y",), scan["y"][r0:r1].astype("f8"), {"units": "rad", "long_name": "ABI fixed grid y"}),
                    "goes_imager_projection": ((), 0, scan["proj"]),
                }
                writer = NcSeriesWriter(
                    ctx.stage(fid) / f"goes_{sat}.nc",
                    lat[r0:r1, c0:c1],
                    lon[r0:r1, c0:c1],
                    {
                        "title": f"GOES ABI L2 FDCC ({sat}) cropped to {fid}",
                        "fire_id": fid,
                        "source": f"s3://{bucket}/{GoesArchive.PRODUCT}/",
                        "window_rows_cols": win.label(),
                    },
                    static,
                )
                series[k] = (writer, win)
        if series[k] is not None:
            writer, win = series[k]
            rows, cols = win.slices()
            writer.append(
                GoesArchive.scan_time(key),
                f"s3://{bucket}/{key}",
                {nm: (a[rows, cols], dt, at) for nm, (a, dt, at) in scan["vars"].items()},
            )
            day = GoesArchive.day_prefix(t)
            scans[fid][(bucket, day)] = scans[fid].get((bucket, day), 0) + 1
