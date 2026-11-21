"""Shared-download channels end to end, offline (fake buckets, grids and files): GOES skips a damaged scan, HRRR
writes analyses and the previous run's rain, GFS fires of one run share requests, and a VIIRS granule is cropped on the swath for each
fire and packed in one file."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fah.channels.gfs import GfsChannel
from fah.channels.goes import GoesChannel
from fah.channels.hrrr import HrrrChannel
from fah.channels.viirs import ViirsChannel
from fah.core.context import RunContext
from fah.core.geometry import BoundingBox, PixelWindow
from fah.sources.earthdata import EarthdataClient
from fah.sources.gfs import EarthEngineGfs
from fah.sources.goes import GoesArchive
from fah.sources.hrrr import HrrrArchive

# ------------------------------------------------------------------------------------- goes


def fdcc(path: Path, n: int = 40, seed: int = 0) -> bytes:
    """A small FDCC-like file: the fire variables packed in zlib chunks, the ABI fixed-grid x / y, the projection."""
    import netCDF4

    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("y", n)
        ds.createDimension("x", n)
        for nm, vals in (("x", np.linspace(-0.1, 0.1, n)), ("y", np.linspace(0.1, -0.1, n))):
            v = ds.createVariable(nm, "i2", (nm,))
            v.scale_factor, v.add_offset = 1e-5, 0.0
            v[:] = vals
        ds.createVariable("goes_imager_projection", "i4").longitude_of_projection_origin = -75.0
        for nm, dt in (("Mask", "i2"), ("Power", "f4"), ("Area", "u2"), ("Temp", "u2"), ("DQF", "u1")):
            v = ds.createVariable(nm, dt, ("y", "x"), zlib=True, chunksizes=(20, 20))
            v[:] = rng.integers(0, 100, (n, n)).astype(dt)
    return Path(path).read_bytes()


def break_chunks(data: bytes, tmp_path: Path) -> bytes:
    """The same file with Mask's compressed chunks overwritten: it still opens, reading Mask fails."""
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "probe.nc"
    p.write_bytes(data)
    with h5py.File(p, "r") as f:
        m = f["Mask"].id
        chunks = [m.get_chunk_info(i) for i in range(m.get_num_chunks())]
    b = bytearray(data)
    for c in chunks:
        b[c.byte_offset : c.byte_offset + c.size] = b"\xff" * c.size
    return bytes(b)


class Resp:
    def __init__(self, content: bytes, headers: dict | None = None) -> None:
        self.content, self.headers, self.text = content, headers or {}, content.decode("latin-1")

    def raise_for_status(self) -> None:
        pass


class S3:
    """Bucket listings, and objects with their ETag (MD5); garble[key] = how many times it arrives cut short."""

    def __init__(self, objs: dict, garble: dict | None = None) -> None:
        self.objs, self.garble, self.got = objs, dict(garble or {}), []

    def get(self, url, params=None, timeout=None):
        if params:
            return Resp(
                "".join(f"<Key>{k}</Key>" for k in sorted(self.objs) if k.startswith(params["prefix"])).encode()
            )
        key = url.split(".amazonaws.com/")[1]
        self.got.append(key)
        data = self.objs[key]
        etag = hashlib.md5(data).hexdigest()
        if self.garble.get(key):
            self.garble[key] -= 1
            data = data[: len(data) // 2]
        return Resp(data, {"ETag": f'"{etag}"'})


def test_goes_skips_a_damaged_scan_instead_of_stopping_the_channel(tmp_path, monkeypatch, camp):
    netCDF4 = pytest.importorskip("netCDF4")
    good, good2 = fdcc(tmp_path / "a.nc", seed=1), fdcc(tmp_path / "b.nc", seed=2)
    bad = break_chunks(good, tmp_path)
    with netCDF4.Dataset("mem", memory=bad) as ds:  # the Kaggle failure: opens, then the read fails
        with pytest.raises(RuntimeError, match="HDF error"):
            ds.variables["Mask"][:]
    hours = [pd.Timestamp("2018-11-09 00:00", tz="UTC") + pd.Timedelta(hours=h) for h in range(3)]

    def key(t, minute):
        return (f"ABI-L2-FDCC/{t.year}/{t.day_of_year:03d}/{t.hour:02d}/"
                f"OR_ABI-L2-FDCC-M6_G16_s{t.year}{t.day_of_year:03d}{t.hour:02d}{minute:02d}171_e0_c0.nc")  # fmt: skip

    objs = {key(hours[0], 1): bad, key(hours[0], 6): good2, key(hours[0], 11): good,  # nearest corrupt in bucket
            key(hours[1], 1): good, key(hours[1], 6): good2,  # nearest garbled once in transit
            key(hours[2], 1): bad, key(hours[2], 6): bad}  # nothing readable that hour  # fmt: skip
    s3 = S3(objs, garble={key(hours[1], 1): 1})
    lat, lon = np.meshgrid(np.linspace(44, 36, 40), np.linspace(-125, -117, 40), indexing="ij")
    monkeypatch.setattr(GoesArchive, "latlon", staticmethod(lambda ds: (lat.astype("f4"), lon.astype("f4"))))
    monkeypatch.setattr(GoesChannel, "hours", staticmethod(lambda f: [(t, [("G16", "bkt")]) for t in hours]))
    monkeypatch.setattr("time.sleep", lambda s: None)

    out = list(GoesChannel().fetch([camp], RunContext("goes", tmp_path / "w", threads=2, session=s3)))

    assert [(r.fire.fire_id, r.status) for r in out] == [(camp.fire_id, "ok")]
    assert s3.got.count(key(hours[1], 1)) == 2  # downloaded again after the MD5 check
    with netCDF4.Dataset(tmp_path / "w" / "stage" / camp.fire_id / "goes" / "goes_G16.nc") as ds:
        assert list(ds.variables["source"][:]) == [f"s3://bkt/{key(hours[0], 6)}", f"s3://bkt/{key(hours[1], 1)}"]
        assert ds.variables["Mask"].shape[0] == 2 and ds.variables["Mask"][:].mask.sum() == 0


# ------------------------------------------------------------------------------------- hrrr


def test_hrrr_writes_analyses_and_the_previous_runs_rain_from_three_days_before(tmp_path, monkeypatch, camp):
    """Every 3 h valid time: f00 of that run (hrrr_ana.nc) and f03 of the run 3 h earlier (hrrr_f03.nc), from 72 h
    before the window; the 12z run's longer forecasts are hrrr_next's."""
    netCDF4 = pytest.importorskip("netCDF4")
    lat, lon = np.meshgrid(np.linspace(41, 38, 40), np.linspace(-123, -120, 40), indexing="ij")
    urls = []

    def fetch(self, url, fields=None):
        urls.append(url)
        names = [n for n, _ in fields]
        if url.endswith("f00.grib2") and url.split("/")[-3] == "hrrr.20181102":
            names.remove("dswrf")  # an analysis without DSWRF (as in 2014)
        return {n: np.full(lat.shape, float(len(urls)), np.float32) for n in names}

    monkeypatch.setattr(HrrrArchive, "grid", lambda self: (lat, lon))
    monkeypatch.setattr(HrrrArchive, "fetch", fetch)
    channel = HrrrChannel()
    [r] = list(channel.fetch([camp], RunContext("hrrr", tmp_path, threads=2)))
    valid = channel.times(camp)
    assert valid[0] == camp.span().start - pd.Timedelta(hours=72) and valid[-1] == camp.span().end - pd.Timedelta(
        hours=2
    )
    assert channel.window_label(camp) == "2018-11-05/2018-11-26" and camp.span().label() == "2018-11-08/2018-11-26"
    assert r.status == "ok" and len(r.sources) == len(urls) == 2 * len(valid)
    assert HrrrArchive.url(valid[0], 0) in urls and HrrrArchive.url(valid[0] - pd.Timedelta(hours=3), 3) in urls
    stage = tmp_path / "stage" / camp.fire_id / "hrrr"
    assert sorted(p.name for p in stage.iterdir()) == ["hrrr_ana.nc", "hrrr_f03.nc"] == sorted(channel.required_files)
    with netCDF4.Dataset(stage / "hrrr_ana.nc") as ds:
        assert len(ds["time"][:]) == len(valid) and ds["time"][0] == valid[0].value
        assert {"gust", "u10", "v10", "rh2", "t2", "dpt", "dswrf"} <= set(ds.variables)
        assert ds["dswrf"].units == "W m-2" and ds["dpt"].units == "K"
    with netCDF4.Dataset(stage / "hrrr_f03.nc") as ds:
        assert {"apcp", "dswrf"} <= set(ds.variables) and "gust" not in ds.variables
        assert all(str(s).endswith("wrfsfcf03.grib2") for s in ds["source"][:])


# ------------------------------------------------------------------------------------- gfs

GRID = ("EPSG:4326", (0.25, 0.0, -180.125, 0.0, -0.25, 90.125), 1440, 721)


def test_gfs_series_per_fire_from_shared_requests(tmp_path, monkeypatch, camp_perimeters):
    netCDF4 = pytest.importorskip("netCDF4")
    pytest.importorskip("affine")
    calls = []

    class FakeGfs:
        def grid(self):
            return GRID

        def fetch(self, run, leads, win, grid):
            calls.append((run, tuple(leads), win))
            r0, r1, c0, c1 = win
            rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
            if run.day == 10:
                raise RuntimeError("Too many concurrent aggregations")
            return {h: {"temperature_2m_above_ground": (rr * 10000 + cc + h / 1000).astype("f4"),
                        "total_precipitation_surface": np.full(rr.shape, h, "f4")} for h in leads if h % 6 == 0}  # fmt: skip

    monkeypatch.setattr(EarthEngineGfs, "connect", classmethod(lambda cls: FakeGfs()))
    short = {"first": "2018-11-09 00:00+00:00", "last": "2018-11-09 06:00+00:00"}
    near = camp_perimeters.replace(**short)  # runs 11-08, 11-09, 11-10
    next_door = camp_perimeters.replace(fire_id="next", box=BoundingBox(-121.2, 39.6, -121.0, 39.9), **short)
    old = camp_perimeters.replace(
        fire_id="old", year=2014, first="2014-08-01 00:00+00:00", last="2014-08-02 00:00+00:00"
    )
    res = {
        r.fire.fire_id: r for r in GfsChannel().fetch([near, next_door, old], RunContext("gfs", tmp_path, threads=2))
    }
    assert res["old"].status == "empty"
    assert {k for k, r in res.items() if r.status == "ok"} == {near.fire_id, "next"}
    assert "1 run(s) failed" in res["next"].error and "Too many concurrent" in res["next"].error
    assert len(res["next"].sources) == 2
    assert res["next"].sources[0].startswith("ee:NOAA/GFS0P25 creation_time=2018-11-08T12:00Z")
    assert len({c[2] for c in calls}) == 1 and len(calls) == 2 + 2 + 1  # both fires share each request;
    #                                                                     the failing run stops at its first chunk
    w = EarthEngineGfs.window(GRID[1], GRID[2], GRID[3], near.region())
    with netCDF4.Dataset(tmp_path / "stage" / near.fire_id / "gfs" / "gfs.nc") as ds:
        assert ds["time"].shape[0] == 2 * 40  # 2 runs that worked x leads 6..240 by 6
        lead, run = ds["lead_h"][:], ds["run_time"][:]
        assert lead[0] == 6 and lead[-1] == 240 and run[0] == pd.Timestamp("2018-11-08 12:00", tz="UTC").value
        assert ds["time"][0] == pd.Timestamp("2018-11-08 18:00", tz="UTC").value
        t = ds["temperature_2m_above_ground"][0]
        rr, cc = np.meshgrid(np.arange(w[0], w[1]), np.arange(w[2], w[3]), indexing="ij")
        assert np.allclose(t, rr * 10000 + cc + 0.006) and ds["temperature_2m_above_ground"].units == "degC"
        assert ds["lat"].shape == (w[1] - w[0], w[3] - w[2]) and ds.crs == "EPSG:4326"
        assert "forecast_hours=6" in ds["source"][0]


# ------------------------------------------------------------------------------------- viirs


def granule(tmp: Path, prod: str, shape: tuple, lat: np.ndarray, lon: np.ndarray) -> Path:
    h5py = pytest.importorskip("h5py")
    p = tmp / f"{prod}.nc"
    with h5py.File(p, "w") as f:
        f.attrs["time_coverage_start"] = b"2018-11-10T20:48:00Z"
        f.attrs["DayNightFlag"] = b"Day"
        if prod.startswith("03"):
            g = f.create_group("geolocation_data")
            g["latitude"], g["longitude"] = lat.astype("f4"), lon.astype("f4")
            g["sensor_zenith"] = np.full(shape, 1200, "i2")
            g["sensor_zenith"].attrs["scale_factor"] = np.float32(0.01)
            g["solar_zenith"] = np.full(shape, 5000, "i2")
        else:
            g = f.create_group("observation_data")
            bands = ("I01", "I02", "I03", "I04", "I05") if prod == "02IMG" else ("M10", "M11")
            for k, b in enumerate(bands):
                d = g.create_dataset(b, data=(np.arange(np.prod(shape)).reshape(shape) + 1000 * k).astype("u2"))
                d.attrs["_FillValue"] = np.array([65535], "u2")
                d.attrs["scale_factor"] = np.float32(2e-5)
                g.create_dataset(f"{b}_quality_flags", data=np.zeros(shape, "u2"))
            if prod == "02IMG":
                g["I04_brightness_temperature_lut"] = np.linspace(200, 400, 65536).astype("f4")
    return p


def test_viirs_granule_is_cropped_on_the_swath_for_each_fire(tmp_path, monkeypatch, camp):
    netCDF4 = pytest.importorskip("netCDF4")
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    mlat, mlon = np.meshgrid(np.linspace(41, 39, 32), np.linspace(-122.5, -120.5, 40), indexing="ij")
    ilat, ilon = np.meshgrid(np.linspace(41, 39, 64), np.linspace(-122.5, -120.5, 80), indexing="ij")
    files = {p: granule(tmp_path, p, s, la, lo) for p, s, la, lo in
             (("02IMG", (64, 80), ilat, ilon), ("03IMG", (64, 80), ilat, ilon),
              ("02MOD", (32, 40), mlat, mlon), ("03MOD", (32, 40), mlat, mlon))}  # fmt: skip
    base = "https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads"
    entry = {"sat": "SNPP", "daynight": "Day", "start": "2018-11-10T20:48:00.000Z",
             **{p: f"{base}/VNP{p}/VNP{p}.A2018314.2048.002.x.nc" for p in files}}  # fmt: skip
    key, key2 = ("SNPP", "A2018314", "2048"), ("SNPP", "A2018314", "0930")
    entry2 = {**entry, "daynight": "Night", "start": "2018-11-10T09:30:00.000Z",
              **{p: u.replace(".2048.", ".0930.") for p, u in entry.items() if p in files}}  # fmt: skip
    monkeypatch.setattr(
        EarthdataClient, "open_remote", lambda self, url, token=None: open(files[url.split("/VNP")[1][:5]], "rb")
    )
    monkeypatch.setattr(ViirsChannel, "granules",
                        classmethod(lambda cls, s, f: {key: entry, key2: entry2} if f.fire_id != "far" else {}))  # fmt: skip
    monkeypatch.setattr(ViirsChannel, "POOL", "thread")
    near, far = camp, camp.replace(fire_id="far", box=BoundingBox(0, 0, 1, 1))
    res = {r.fire.fire_id: r for r in ViirsChannel().fetch([near, far], RunContext("viirs", tmp_path, threads=2))}
    assert res["far"].status == "empty" and res[near.fire_id].status == "ok"
    assert sorted(res[near.fire_id].sources) == sorted([entry["02IMG"], entry2["02IMG"]])

    stage = tmp_path / "stage" / near.fire_id / "viirs"
    assert sorted(p.name for p in stage.iterdir()) == ["viirs.nc"]  # one file per fire, no .part left
    mr0, mr1, mc0, mc1 = PixelWindow.of_points(mlat, mlon, near.region()).grown(1, (32, 40))
    ir0, ir1, ic0, ic1 = PixelWindow(mr0, mr1, mc0, mc1).m_to_i((64, 80))
    w, s_, e, n = near.region().buffered(ViirsChannel.MASK_PAD_KM)
    la, lo = ilat[ir0:ir1, ic0:ic1], ilon[ir0:ir1, ic0:ic1]
    out_i = ~((la >= s_) & (la <= n) & (lo >= w) & (lo <= e))
    mla, mlo = mlat[mr0:mr1, mc0:mc1], mlon[mr0:mr1, mc0:mc1]
    out_m = ~((mla >= s_) & (mla <= n) & (mlo >= w) & (mlo <= e))
    assert out_i.any() and (~out_i).any()  # the window reaches past the region
    with netCDF4.Dataset(stage / "viirs.nc") as ds:
        assert list(ds["granule"][:]) == ["SNPP_2018-11-10T0930_night", "SNPP_2018-11-10T2048_day"]  # in time order
        assert list(ds["day_night"][:]) == ["Night", "Day"] and ds.n_granules == 2 and ds.fire_id == near.fire_id
        luts = ds.groups["luts"].variables
        assert len(luts) == 1 and next(iter(luts.values())).shape == (65536,)  # one table for both granules
        g = ds.groups["SNPP_2018-11-10T2048_day"]
        assert ds[g.I04_brightness_temperature_lut].source_name == "I04_brightness_temperature_lut"
        assert (
            ds.groups["SNPP_2018-11-10T0930_night"].I04_brightness_temperature_lut == g.I04_brightness_temperature_lut
        )
        for v in g.variables.values():
            v.set_auto_maskandscale(False)
        src = np.arange(64 * 80).reshape(64, 80)[ir0:ir1, ic0:ic1]
        i01, i04 = g["I01"][:], g["I04"][:]
        assert (i01[~out_i] == src[~out_i]).all() and (i01[out_i] == 65535).all()  # outside the region: fill
        assert (i04[~out_i] == src[~out_i] + 3000).all() and g.blanked_pixels_i == int(out_i.sum())
        assert (
            g["I04"]._FillValue == 65535
            and g["I04"].scale_factor == np.float32(2e-5)
            and "I05_quality_flags" in g.variables
        )
        assert (g["latitude"][:] == la.astype("f4")).all() and g["sensor_zenith"].scale_factor == np.float32(0.01)
        m11 = g["M11"][:]
        msrc = np.arange(32 * 40).reshape(32, 40)[mr0:mr1, mc0:mc1] + 1000
        assert (m11[~out_m] == msrc[~out_m]).all() and (m11[out_m] == 65535).all() and "M10" not in g.variables
        assert (g["m_latitude"][:] == mla.astype("f4")).all()
        lat = g["latitude"][:]
        reg = near.region()  # cut to the fire's region, not the build's box
        assert lat.min() <= reg[1] + 0.07 and lat.max() >= reg[3] - 0.07
        assert (
            g.day_night == "Day"
            and g.source_DayNightFlag == "Day"
            and g.i_window_rows_cols == f"{ir0}:{ir1},{ic0}:{ic1}"
        )


def test_viirs_granule_failures_are_notes_unless_all_fail(tmp_path, monkeypatch, camp):
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    keys = [("SNPP", "A2018314", "2048"), ("SNPP", "A2018315", "2030")]
    grans = {
        k: {"sat": "SNPP", "daynight": "Day", "start": k[1], "02IMG": f"https://x/{k[1]}", "03IMG": "y"} for k in keys
    }

    def bad_open(self, url, token=None):
        raise PermissionError("Earthdata Login refused the request")

    monkeypatch.setattr(EarthdataClient, "open_remote", bad_open)
    monkeypatch.setattr(ViirsChannel, "granules", classmethod(lambda cls, s, f: grans))
    monkeypatch.setattr(ViirsChannel, "POOL", "thread")
    [r] = list(ViirsChannel().fetch([camp], RunContext("viirs", tmp_path, threads=2)))
    assert r.status == "error" and "2 granule(s) failed" in r.error and "Earthdata Login refused" in r.error
