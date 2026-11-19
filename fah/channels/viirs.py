"""viirs: VIIRS L1B swath granules (TS-SatFire's VIIRS_Day / VIIRS_Night source), packed in one viirs.nc per fire."""

from __future__ import annotations

import logging
import multiprocessing
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pandas as pd
import requests

from fah.channels.base import SharedDownloadChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.geometry import BoundingBox, PixelWindow
from fah.core.result import FetchResult
from fah.core.secrets import Secrets
from fah.sources.earthdata import EarthdataClient
from fah.sources.http import ParallelMap
from fah.storage.attrs import NetcdfAttrs
from fah.storage.viirs_file import ViirsFile

log = logging.getLogger(__name__)


class ViirsChannel(SharedDownloadChannel):
    """Every S-NPP, NOAA-20 and NOAA-21 granule over the region during the window, day and night, cropped on the swath
    itself (no regridding): I1-I5 at 375 m and M11 at 750 m as stored (scaled integers, quality flags, the I4 / I5
    brightness-temperature tables), with each pixel's latitude, longitude and sensor / solar zenith. Remote files are
    read by byte range (a fire's part of a 250 MB granule is a few MB); each granule is read once for every fire on
    it. Pixels beyond the region + MASK_PAD_KM are blanked. Granules that fail are named in the manifest note; a fire
    is an error only when all of them fail."""

    name = "viirs"
    time_dependent = True
    required_files = frozenset({ViirsFile.NAME})
    GROUP_SIZE = 25
    # (satellite, LAADS product prefix, first day of data). Products <prefix>02IMG (I1-I5, 375 m), 03IMG (its
    # geolocation), 02MOD (M-bands, 750 m), 03MOD (its geolocation); the newest collection version of each.
    SATELLITES = (("SNPP", "VNP", "2012-01-19"), ("NOAA20", "VJ1", "2018-01-05"), ("NOAA21", "VJ2", "2023-02-10"))
    PRODUCTS = ("02IMG", "03IMG", "02MOD", "03MOD")
    I_BANDS = ("I01", "I02", "I03", "I04", "I05")  # with their _quality_flags, _uncert_index, LUTs
    M_BANDS = ("M11",)
    GEOLOCATION = ("latitude", "longitude", "sensor_zenith", "solar_zenith")
    MASK_PAD_KM = 2.0  # pixels up to this far outside the region are kept (edge resampling); beyond: fill
    POOL = "process"  # h5py serialises reads within a process: granules go to worker processes
    WORKERS = 8

    def check_ready(self) -> None:
        EarthdataClient.token()
        self.require_module("h5py", "viirs reads HDF5 granules: pip install h5py")

    @classmethod
    def granules(cls, session: requests.Session, fire: Fire) -> dict[tuple[str, str, str], dict]:
        """Swath granules over the region during the window: {(sat, 'A2018314', '2048'): {'sat', 'daynight', 'start',
        '02IMG': url, '03IMG': url, '02MOD': url, '03MOD': url}}, only those with I-bands and their geolocation."""
        earthdata = EarthdataClient(session)
        t0, t1 = fire.span()
        out: dict[tuple, dict] = {}
        for sat, prefix, first in cls.SATELLITES:
            if t1 < pd.Timestamp(first, tz="UTC"):
                continue
            for prod in cls.PRODUCTS:
                for g in earthdata.granules(prefix + prod, t0, t1, fire.region()):
                    _, a, hhmm, _ = EarthdataClient.granule_parts(g["url"])
                    e = out.setdefault((sat, a, hhmm), {"sat": sat})
                    e[prod] = g["url"]
                    if prod == "02IMG":
                        e["daynight"], e["start"] = g["daynight"], g["start"]
        return {k: e for k, e in out.items() if "02IMG" in e and "03IMG" in e}

    def fetch(self, fires: list[Fire], ctx: RunContext) -> Iterator[FetchResult]:
        workers = max(1, min(ctx.threads, self.WORKERS))
        t_start = time.time()

        def pool(n: int):
            if self.POOL == "process":  # spawn: forking a process that runs threads can deadlock
                return ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"))
            return ThreadPoolExecutor(n)

        def look(f: Fire) -> tuple:
            try:
                return f, self.granules(ctx.session, f), None
            except Exception as ex:  # noqa: BLE001  (this fire is an error; the others go on)
                return f, None, Secrets.redact(f"CMR: {type(ex).__name__}: {ex}")

        for group in self.groups(fires, self.GROUP_SIZE):
            if ctx.expired():
                return
            need: dict[tuple, tuple[dict, list[Fire]]] = {}
            left, files, srcs, failed = {}, {}, {}, {}
            with ThreadPoolExecutor(8) as ex:
                for f, grans, err in ex.map(look, group):
                    if err or not grans:
                        yield FetchResult(f, "error" if err else "empty", (), err or "")
                        continue
                    fid = f.fire_id
                    left[fid], files[fid], srcs[fid], failed[fid] = (
                        len(grans),
                        ViirsFile(ctx.stage_path(fid), fid),
                        [],
                        [],
                    )
                    for k, e in grans.items():
                        need.setdefault(k, (e, []))[1].append(f)
            jobs = [
                (k, e, [(f.fire_id, f.region()) for f in fs])
                for k, (e, fs) in sorted(need.items(), key=lambda kv: (str(kv[1][0].get("start")), kv[0]))
            ]
            if jobs:
                log.info(f"  viirs: {len(jobs)} granules for {len(left)} fires")
            stopped, got = False, 0
            with pool(workers) as ex:
                for key, crops, nbytes, err in ParallelMap.windowed(ex, self.read_granule, jobs, 2 * workers):
                    got += nbytes
                    e, fs = need[key]
                    by_id = {f.fire_id: f for f in fs}
                    for fid, crop in crops.items():
                        ctx.stage(fid)
                        files[fid].add_granule(
                            key,
                            e,
                            crop,
                            by_id[fid].region().buffered(self.MASK_PAD_KM),
                            " | ".join(e[p] for p in self.PRODUCTS if p in e),
                        )
                        srcs[fid].append(e["02IMG"])
                    for f in fs:
                        fid = f.fire_id
                        if err:
                            failed[fid].append(err)
                        left[fid] -= 1
                        if left[fid] == 0:
                            note = f"{len(failed[fid])} granule(s) failed: {failed[fid][0]}" if failed[fid] else ""
                            if files[fid].groups:
                                files[fid].finish(f.region(), self.MASK_PAD_KM)
                            status = "ok" if files[fid].groups else ("error" if failed[fid] else "empty")
                            yield FetchResult(f, status, tuple(srcs[fid]), note)
                    if ctx.expired():
                        stopped = True
                        break
            if jobs:
                log.info(f"  viirs: {got / 1e9:.2f} GB read in {(time.time() - t_start) / 60:.0f} min so far")
            if stopped:
                for fid, n in left.items():
                    if n > 0:
                        ctx.drop_stage(fid)
                return

    @classmethod
    def read_granule(cls, job: tuple) -> tuple:
        """One granule for every fire on it (runs in a worker process). job = (key, entry, [(fire_id, region)]).
        -> (key, {fire_id: crop}, bytes downloaded, error or None). The fire windows come from the 750 m geolocation
        (a quarter of the 375 m one); I-band windows follow from them (PixelWindow.m_to_i)."""
        import h5py

        key, e, fires = job
        session = requests.Session()
        session.headers["User-Agent"] = RunContext.USER_AGENT
        earthdata = EarthdataClient(session)
        opened = []

        def open_h5(prod: str):
            fo = earthdata.open_remote(e[prod], EarthdataClient.token())
            opened.append(fo)
            return h5py.File(fo, "r")

        try:
            has_m = "02MOD" in e and "03MOD" in e
            with open_h5("03MOD" if has_m else "03IMG") as g:
                gl = g["geolocation_data"]
                lat, lon = gl["latitude"][:], gl["longitude"][:]
            wins = {
                fid: w for fid, box in fires if (w := PixelWindow.of_points(lat, lon, BoundingBox(*box))) is not None
            }
            crops: dict[str, dict] = {}
            if wins:
                with open_h5("02IMG") as fi, open_h5("03IMG") as gi:
                    obs_i, geo_i = fi["observation_data"], gi["geolocation_data"]
                    i_shape = geo_i["latitude"].shape
                    glob = NetcdfAttrs.clean(fi.attrs)
                    luts: dict = {}
                    for fid, w in wins.items():
                        m_win = w.grown(1, lat.shape) if has_m else None
                        i_win = m_win.m_to_i(i_shape) if has_m else w.grown(2, i_shape)
                        crops[fid] = {
                            "i_win": i_win,
                            "m_win": m_win,
                            "glob": glob,
                            "m": {},
                            "i": {
                                **cls._window_of(geo_i, cls.GEOLOCATION, i_shape, i_win, {}),
                                **cls._window_of(obs_i, cls.I_BANDS, i_shape, i_win, luts),
                            },
                        }
                if has_m:
                    with open_h5("02MOD") as fm, open_h5("03MOD") as gm:
                        obs_m, geo_m = fm["observation_data"], gm["geolocation_data"]
                        luts = {}
                        for c in crops.values():
                            geo = cls._window_of(geo_m, cls.GEOLOCATION, lat.shape, c["m_win"], {})
                            c["m"] = {
                                **{f"m_{k}": v for k, v in geo.items()},
                                **cls._window_of(obs_m, cls.M_BANDS, lat.shape, c["m_win"], luts),
                            }
            return key, crops, sum(getattr(f, "fetched", 0) for f in opened), None
        except Exception as ex:  # noqa: BLE001  (reported per granule; the pool must not stop)
            return (
                key,
                {},
                sum(getattr(f, "fetched", 0) for f in opened),
                Secrets.redact(f"{e.get('02IMG')}: {type(ex).__name__}: {ex}")[:400],
            )
        finally:
            for f in opened:
                f.close()

    @staticmethod
    def _window_of(group, prefixes: tuple[str, ...], shape: tuple[int, int], win: PixelWindow, luts: dict) -> dict:
        """{name: (array, attrs)}: the window of every 2-D dataset of `shape` whose name starts with one of prefixes,
        and every 1-D numeric dataset with such a name (look-up tables) whole, read once per granule."""
        import h5py

        rows, cols = win.slices()
        out = {}
        for name, ds in group.items():
            if not isinstance(ds, h5py.Dataset) or not name.startswith(prefixes) or ds.dtype.kind not in "iuf":
                continue
            if ds.shape == tuple(shape):
                out[name] = (ds[rows, cols], dict(ds.attrs))
            elif ds.ndim == 1:
                if name not in luts:
                    luts[name] = (ds[:], dict(ds.attrs))
                out[name] = luts[name]
        return out
