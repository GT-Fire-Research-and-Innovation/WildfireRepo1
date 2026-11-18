"""Raw-data grab, the channels that bring TS-SatFire's inputs in (gridmet extras, hrrr_next, lulc, landcover,
vegetation, viirs), offline: grids are cut on their own pixels and joined without resampling, swath windows
follow the M-to-I geometry, byte-range reads keep the Earthdata token off S3, a grown channel refetches only
fires that miss its new files, and a synthetic VIIRS granule comes out cropped exactly.
Run: python -m pytest tests/test_raw_satfire.py -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import earthdata as ED  # noqa: E402
import raw_channels as RC  # noqa: E402
import raw_grab as RG  # noqa: E402

np = pytest.importorskip("numpy")

CAMP = dict(fire_id="F2003403265", incident="75E64DB8-9B75-4A68-BEDD-67CC62658E38", name="CAMP", year=2018,
            first="2018-11-09 00:00:00.000+00:00", last="2018-11-25 18:22:00.000+00:00", rating="keep",
            bounds=[-121.88999101802999, 39.51118309776612, -121.23979602288742, 39.98463882716831])


def fire(**kw):
    return dict(CAMP, **kw)


# ------------------------------------------------------------------------------------- gridmet

def test_gridmet_gains_tssatfire_variables():
    v = RC.gridmet_vars()
    assert len(v) == 13 and {"pr", "tmmn", "sph", "pdsi"} <= set(v)
    assert v["pdsi"] == "daily_mean_palmer_drought_severity_index" and v["sph"] == "daily_mean_specific_humidity"
    d = RC.days(CAMP)
    u = RC.gridmet_url("pdsi", v["pdsi"], CAMP["bounds"], d[0] - pd.Timedelta(days=RC.GRIDMET_LEAD_DAYS["pdsi"]), d[-1], 0.05)
    assert "/ncss/grid/agg_met_pdsi_1979_CurrentYear_CONUS.nc?var=daily_mean_palmer_drought_severity_index" in u
    assert "time_start=2018-10-16T00:00:00Z" in u


def test_grown_channel_refetches_only_fires_missing_new_files():
    old = ";".join(f"gridmet_{k}.nc" for k in ("vpd", "fm100", "fm1000", "bi", "erc", "rmin", "vs", "th", "tmmx"))
    new = ";".join(f"gridmet_{k}.nc" for k in RC.gridmet_vars())
    m = pd.DataFrame([dict(fire_id="A", status="ok", files=old, fetched_utc="2026-11-01"),
                      dict(fire_id="B", status="ok", files=new, fetched_utc="2026-11-01"),
                      dict(fire_id="C", status="n/a", files="", fetched_utc="2026-11-01"),
                      dict(fire_id="D", status="error", files="", fetched_utc="2026-11-01")])
    assert RG.done_ids(m, "gridmet") == {"B", "C"}
    assert RG.done_ids(m, "terrain") == {"A", "B", "C"}           # no expected files: any ok row counts
    assert RG.done_ids(m) == {"A", "B", "C"}


# ------------------------------------------------------------------------------------- hrrr_next

def test_hrrr_next_runs_reach_the_next_afternoon_as_far_as_hrrr_ran():
    t = RC.hrrr_next_times(CAMP)
    runs = sorted({r for r, _ in t})
    assert len(runs) == 19 and len(t) == 19 * 12                 # HRRRv3: 3..36 h every 3 h
    assert runs[0] == pd.Timestamp("2018-11-08 12:00", tz="UTC") and max(h for _, h in t) == 36
    v1 = RC.hrrr_next_times(fire(year=2016, first="2016-07-07 00:00+00:00", last="2016-07-08 00:00+00:00"))
    v2 = RC.hrrr_next_times(fire(year=2017, first="2017-07-07 00:00+00:00", last="2017-07-08 00:00+00:00"))
    assert {h for _, h in v1} == {3, 6, 9, 12, 15} and {h for _, h in v2} == {3, 6, 9, 12, 15, 18}
    assert RC.applies("hrrr_next", fire(year=2014, first="2014-06-01 00:00+00:00", last="2014-06-20 00:00+00:00"))


def test_hrrr_next_takes_precipitation_accumulated_from_the_run_start():
    import weather_features as WF
    idx = ("1:0:d=2020090712:REFC:entire atmosphere:9 hour fcst:\n"
           "2:100:d=2020090712:TMP:2 m above ground:9 hour fcst:\n"
           "3:200:d=2020090712:SPFH:2 m above ground:9 hour fcst:\n"
           "4:300:d=2020090712:APCP:surface:8-9 hour acc fcst:\n"
           "5:400:d=2020090712:APCP:surface:0-9 hour acc fcst:\n"
           "6:500:d=2020090712:WEASD:surface:0-9 hour acc fcst:\n")

    class S:
        def get(self, url, timeout):
            return type("R", (), {"status_code": 200, "text": idx})()

    rng = WF._ranges(S(), "u", RC.hrrr_next_fields())
    assert rng["t2"] == (100, 199) and rng["q2"] == (200, 299) and rng["apcp"] == (400, 499)
    assert WF._ranges(S(), "u").keys() <= {"gust", "u10", "v10", "rh2"}   # the hrrr channel's fields unchanged


# ------------------------------------------------------------------------------------- secrets

def test_redact_removes_earthdata_token_and_signed_url_parts(monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "eyJhbGciOiJSUzI1NiJ9.payload.sig")
    t = RC.redact("401 for https://data.lpdaac.earthdatacloud.nasa.gov/x.h5 with Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig")
    assert "payload" not in t and "{EARTHDATA_TOKEN}" in t
    s = RC.redact("https://acct.blob.core.windows.net/c/x.tif?st=2026-11-17&se=2026-11-18&sp=rl&sv=2024&sr=c&sig=AbC%2Bd")
    assert s == "https://acct.blob.core.windows.net/c/x.tif"
    a = RC.redact("https://bucket.s3.us-west-2.amazonaws.com/f.nc?A-userid=me&X-Amz-Credential=K&X-Amz-Signature=S")
    assert "X-Amz" not in a and "A-userid=me" in a


# ------------------------------------------------------------------------------------- sinusoidal tiles

def test_sinusoidal_window_holds_the_box_and_tiles_join_pixel_for_pixel():
    pyproj = pytest.importorskip("pyproj")
    box = [-121.0, 39.9, -120.8, 40.1]                           # straddles 40 N: tiles h08v04 and h08v05
    win = ED.sinu_window(box)
    r0, r1, c0, c1 = win
    assert r0 < 5 * 2400 <= r1 and ED.tile_part(win, 8, 4) and ED.tile_part(win, 8, 5) and not ED.tile_part(win, 9, 5)
    x, y = ED.sinu_coords(win)
    to_ll = pyproj.Transformer.from_crs(ED.SINU_PROJ, 4326, always_xy=True)
    lon, lat = to_ll.transform(*np.meshgrid(x, y))
    assert lat.max() > 40.1 - 0.005 and lat.min() < 39.9 + 0.005 and lon.min() < -121.0 + 0.01 and lon.max() > -120.8 - 0.01

    m = RC.SinuMosaic(win)
    for h, v in ((8, 4), (8, 5)):
        rows = np.add.outer(v * 2400 + np.arange(2400), np.zeros(2400, int)).astype("u2")
        cols = np.add.outer(np.zeros(2400, int), h * 2400 + np.arange(2400)).astype("u2")
        m.put("row", rows, {"_FillValue": np.array([65535], "u2"), "units": b"1"}, h, v)
        m.put("col", cols, {}, h, v)
    got_r, fill, at = m.layers["row"]
    assert (got_r == np.arange(r0, r1)[:, None]).all() and fill == 65535 and at["units"] == "1"
    assert (m.layers["col"][0] == np.arange(c0, c1)[None, :]).all()


def test_sinusoidal_netcdf_keeps_values_fill_and_grid(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    win = (11990, 11994, 19500, 19503)
    m = RC.SinuMosaic(win, nt=2)
    tile = np.arange(2400 * 2400, dtype="i2").reshape(2400, 2400) % 1000
    m.put("NDVI", tile, {"_FillValue": np.int16(-15000), "scale_factor": np.float32(1e-4)}, 8, 4, t=1, source_name="500 m 16 days NDVI")
    RC.write_sinu_nc(tmp_path / "v.nc", win, m.layers, {"title": "t"}, times_ns=[1, 2])
    with netCDF4.Dataset(tmp_path / "v.nc") as ds:
        v = ds.variables["NDVI"]
        v.set_auto_maskandscale(False)
        assert v.shape == (2, 4, 3) and (v[0] == -15000).all() and v.source_name == "500 m 16 days NDVI"
        assert (v[1] == tile[11990 - 9600:11994 - 9600, 19500 - 19200:19503 - 19200]).all()
        assert ds.variables["crs"].grid_mapping_name == "sinusoidal" and ds.sinusoidal_rows_cols == "11990:11994,19500:19503"
        assert ds.variables["x"][0] == pytest.approx(ED.SINU_X0 + 19500.5 * ED.SINU_TILE_M / 2400)
    assert RC._var_name("HDFEOS/GRIDS/VIIRS_Grid_16Day_VI_500m/Data Fields/500 m 16 days pixel reliability") == "pixel_reliability"


def test_tiles_and_versions_from_granule_names():
    a = "https://x/VNP13A1.A2018305.h08v05.002.2022307200550.h5"
    assert ED.tile_hv(a.rsplit("/", 1)[1]) == (8, 5) and ED.granule_parts(a) == ("VNP13A1", "A2018305", "h08v05", "002")
    assert ED.granule_day(a) == pd.Timestamp("2018-11-01", tz="UTC")
    g = [{"url": "https://x/VJ102IMG.A2018314.2000.021.1.nc", "start": "b", "name": "1"},
         {"url": "https://x/VJ102IMG.A2018314.2000.002.1.nc", "start": "b", "name": "2"},
         {"url": "https://x/VJ102IMG.A2018314.1018.021.1.nc", "start": "a", "name": "3"}]
    assert [x["name"] for x in ED.newest_versions(g)] == ["3", "1"]


def test_lulc_year_and_pixel_window():
    assert [RC.lulc_year(y) for y in (2017, 2018, 2021, 2025)] == [2017, 2017, 2020, 2023]
    from affine import Affine
    t = Affine(10, 0, 500000, 0, -10, 4400000)
    assert RC.pixel_window(t, 1000, 1000, (500015, 4399000, 500101, 4399985)) == (1, 100, 1, 11)
    assert RC.pixel_window(t, 1000, 1000, (400000, 0, 400100, 10)) is None


# ------------------------------------------------------------------------------------- swaths

def test_swath_window_and_m_to_i_geometry():
    lat, lon = np.meshgrid(np.linspace(41, 39, 20), np.linspace(-122, -121, 30), indexing="ij")
    w = ED.swath_window(lat, lon, [-121.6, 39.8, -121.4, 40.2])
    rows, cols = np.where((lat >= 39.8) & (lat <= 40.2) & (lon >= -121.6) & (lon <= -121.4))
    assert w == (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)
    assert ED.swath_window(lat, lon, [0, 0, 1, 1]) is None
    assert ED.m_to_i((5, 9, 10, 12), (40, 60)) == (8, 20, 18, 26) and ED.m_to_i((0, 20, 0, 30), (40, 60)) == (0, 40, 0, 60)


# ------------------------------------------------------------------------------------- byte ranges

class RangeSess:
    """Earthdata Cloud in miniature: the NASA URL redirects (token required) to a signed S3 URL that serves
    byte ranges of `data` and refuses any Authorization header."""

    def __init__(self, data, token="tok", login=False):
        self.data, self.token, self.login, self.calls = data, token, login, []

    def get(self, url, headers=None, allow_redirects=True, stream=False, timeout=None):
        h = dict(headers or {})
        self.calls.append((url, h.get("Authorization")))
        R = type("R", (), {"close": lambda s: None})
        r = R()
        r.headers, r.text = {}, ""
        if "nasa.gov" in url:
            if self.login or h.get("Authorization") != f"Bearer {self.token}":
                r.status_code, r.headers = 302, {"Location": "https://urs.earthdata.nasa.gov/oauth/authorize?x=1"}
            else:
                r.status_code, r.headers = 307, {"Location": "https://bucket.s3.us-west-2.amazonaws.com/f.nc?X-Amz-Signature=s"}
            return r
        assert "Authorization" not in h, "token sent to S3"
        a, b = (int(x) for x in h["Range"].split("=")[1].split("-"))
        r.status_code, r.content = 206, self.data[a:b + 1]
        r.headers = {"Content-Range": f"bytes {a}-{b}/{len(self.data)}"}
        return r


def test_range_file_reads_hdf5_through_a_signed_redirect(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "x.nc"
    with h5py.File(p, "w") as f:
        f.create_dataset("observation_data/I04", data=np.arange(300 * 200, dtype="u2").reshape(300, 200), chunks=(50, 200),
                         compression="gzip")
    s = RangeSess(p.read_bytes())
    fo = ED.RangeFile(s, "https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads/VNP02IMG/x.nc", "tok", block=4096)
    with h5py.File(fo, "r") as f:
        assert (f["observation_data/I04"][120:130, 7:9] == np.arange(300 * 200).reshape(300, 200)[120:130, 7:9]).all()
    assert fo.fetched < p.stat().st_size and all(a is None for u, a in s.calls if "amazonaws" in u)
    with pytest.raises(PermissionError, match="EARTHDATA_TOKEN"):
        ED.RangeFile(RangeSess(b"", login=True), "https://data.lpdaac.earthdatacloud.nasa.gov/x.h5", "tok")


# ------------------------------------------------------------------------------------- viirs, end to end

def _granule(tmp, prod, shape, lat, lon):
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


class Ctx:
    def __init__(self, tmp):
        self.tmp, self.threads, self.sess = tmp, 2, None

    def stage(self, fid):
        p = self.tmp / "stage" / fid / "viirs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def drop_stage(self, fid):
        pass

    def expired(self):
        return False


def test_viirs_granule_is_cropped_on_the_swath_for_each_fire(tmp_path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4")
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    mlat, mlon = np.meshgrid(np.linspace(41, 39, 32), np.linspace(-122.5, -120.5, 40), indexing="ij")
    ilat, ilon = np.meshgrid(np.linspace(41, 39, 64), np.linspace(-122.5, -120.5, 80), indexing="ij")
    files = {p: _granule(tmp_path, p, s, la, lo) for p, s, la, lo in
             (("02IMG", (64, 80), ilat, ilon), ("03IMG", (64, 80), ilat, ilon),
              ("02MOD", (32, 40), mlat, mlon), ("03MOD", (32, 40), mlat, mlon))}
    entry = {"sat": "SNPP", "daynight": "Day", "start": "2018-11-10T20:48:00.000Z",
             **{p: f"https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads/VNP{p}/VNP{p}.A2018314.2048.002.x.nc" for p in files}}
    key = ("SNPP", "A2018314", "2048")
    monkeypatch.setattr(ED, "open_remote", lambda sess, url, token=None: open(files[url.split("/VNP")[1][:5]], "rb"))
    monkeypatch.setattr(RC, "viirs_granules", lambda sess, f: {key: entry} if f["fire_id"] != "far" else {})
    monkeypatch.setattr(RC, "VIIRS_POOL", "thread")
    near, far = fire(), fire(fire_id="far", bounds=[0, 0, 1, 1])
    res = {f["fire_id"]: (s, src, err) for f, s, src, err in RC.grab_viirs_many([near, far], Ctx(tmp_path))}
    assert res["far"][0] == "empty" and res[near["fire_id"]][0] == "ok" and res[near["fire_id"]][1] == [entry["02IMG"]]

    out = tmp_path / "stage" / near["fire_id"] / "viirs" / "SNPP_2018-11-10T2048_day.nc"
    mw = ED.swath_window(mlat, mlon, RC.area(near))
    mr0, mr1, mc0, mc1 = RC._grow(mw, 1, (32, 40))
    ir0, ir1, ic0, ic1 = ED.m_to_i((mr0, mr1, mc0, mc1), (64, 80))
    with netCDF4.Dataset(out) as ds:
        for v in ds.variables.values():
            v.set_auto_maskandscale(False)
        src = np.arange(64 * 80).reshape(64, 80)
        assert (ds["I01"][:] == src[ir0:ir1, ic0:ic1]).all() and (ds["I04"][:] == src[ir0:ir1, ic0:ic1] + 3000).all()
        assert ds["I04"]._FillValue == 65535 and ds["I04"].scale_factor == np.float32(2e-5)
        assert ds["I04_brightness_temperature_lut"].shape == (65536,) and "I05_quality_flags" in ds.variables
        assert (ds["latitude"][:] == ilat[ir0:ir1, ic0:ic1].astype("f4")).all() and ds["sensor_zenith"].scale_factor == np.float32(0.01)
        assert (ds["M11"][:] == np.arange(32 * 40).reshape(32, 40)[mr0:mr1, mc0:mc1] + 1000).all() and "M10" not in ds.variables
        assert (ds["m_latitude"][:] == mlat[mr0:mr1, mc0:mc1].astype("f4")).all()
        lat = ds["latitude"][:]
        reg = RC.area(near)                                        # cut to the fire's region, not the build's box
        assert lat.min() <= reg[1] + 0.07 and lat.max() >= reg[3] - 0.07
        assert ds.day_night == "Day" and ds.source_DayNightFlag == "Day" and ds.i_window_rows_cols == f"{ir0}:{ir1},{ic0}:{ic1}"


def test_viirs_granule_failures_are_notes_unless_all_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok")
    keys = [("SNPP", "A2018314", "2048"), ("SNPP", "A2018315", "2030")]
    grans = {k: {"sat": "SNPP", "daynight": "Day", "start": k[1], "02IMG": f"https://x/{k[1]}", "03IMG": "y"} for k in keys}

    def bad_open(sess, url, token=None):
        raise PermissionError("Earthdata Login refused the request")

    monkeypatch.setattr(ED, "open_remote", bad_open)
    monkeypatch.setattr(RC, "viirs_granules", lambda sess, f: grans)
    monkeypatch.setattr(RC, "VIIRS_POOL", "thread")
    [(f, status, src, err)] = list(RC.grab_viirs_many([fire()], Ctx(tmp_path)))
    assert status == "error" and "2 granule(s) failed" in err and "Earthdata Login refused" in err
