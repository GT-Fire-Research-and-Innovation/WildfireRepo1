"""Raw-data grab (pipeline/raw_grab.py, raw_channels.py), offline: time windows match the build, channels
skip fires they have no data for, secrets never reach the manifest, dirty IRWIN keys still match, resume
reads manifests right, and a batch is deleted only after its upload worked.
Run: python -m pytest tests/test_raw_grab.py -q"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import raw_channels as RC  # noqa: E402
import raw_grab as RG  # noqa: E402

# Camp Fire 2018 as the catalog and grid.json describe it
CAMP = dict(fire_id="F2003403265", incident="75E64DB8-9B75-4A68-BEDD-67CC62658E38", name="CAMP", year=2018,
            first="2018-11-09 00:00:00.000+00:00", last="2018-11-25 18:22:00.000+00:00", rating="keep",
            bounds=[-121.88999101802999, 39.51118309776612, -121.23979602288742, 39.98463882716831])


def fire(**kw):
    return dict(CAMP, **kw)


# ------------------------------------------------------------------------------------- windows

def test_firms_calls_cover_build_window_in_5_day_steps():
    calls = RC.firms_calls(CAMP)
    days = RC.days(CAMP)
    assert (days[0].date().isoformat(), days[-1].date().isoformat()) == ("2018-10-26", "2018-11-27")
    for src in ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"):
        mine = [(d, k) for s, d, k in calls if s == src]
        assert mine[0] == ("2018-10-26", 5) and mine[-1] == ("2018-11-25", 3)
        assert sum(k for _, k in mine) == len(days)
    assert {s for s, _, _ in RC.firms_calls(fire(year=2016))} == {"VIIRS_SNPP_SP"}   # NOAA-20 from 2018


def test_hrrr_and_goes_times_match_build_stages():
    ana, fc = RC.hrrr_times(CAMP)
    assert (len(ana), len(fc)) == (143, 18)
    assert ana[0] == pd.Timestamp("2018-11-08 00:00", tz="UTC") and ana[-1] == pd.Timestamp("2018-11-25 18:00", tz="UTC")
    hours = RC.goes_hours(CAMP)
    assert len(hours) == 428
    g17 = [t for t, sats in hours if any(s == "G17" for s, _ in sats)]
    assert g17[0] == pd.Timestamp("2018-11-15 00:00", tz="UTC") and len(g17) == 260


def test_channels_skip_fires_without_data():
    old, new = fire(year=2014, first="2014-08-01 00:00+00:00", last="2014-08-20 00:00+00:00"), \
        fire(year=2021, first="2021-07-14 00:00+00:00", last="2021-08-20 00:00+00:00")
    assert all(RC.applies(c, CAMP) is None for c in RC.CHANNELS)
    assert {c for c in RC.CHANNELS if RC.applies(c, old)} == {"suppression", "landfire", "tessera", "goes", "lulc"}
    assert {c for c in RC.CHANNELS if RC.applies(c, new)} == {"ics209"}
    assert RC.applies("hrrr", fire(year=2014, first="2014-06-01 00:00+00:00", last="2014-06-20 00:00+00:00"))
    assert RC.landfire_version(2017) == "LF2016" and RC.landfire_version(2024) == "LF2023" and RC.landfire_version(2016) is None


# ------------------------------------------------------------------------------------- keys and secrets

def test_dirty_irwin_keys_still_match_records():
    assert RC.norm_irwin("D6024AFC-A330-404F-ACF1-7A21986EEFEA}\r\n") == "D6024AFC-A330-404F-ACF1-7A21986EEFEA"
    assert RC.norm_irwin("{75e64db8-9b75-4a68-bedd-67cc62658e38}") == CAMP["incident"]
    assert RC.fire_key(CAMP) == (CAMP["incident"], None)
    assert RC.fire_key(fire(incident="name:dixie|CA-PNF-001")) == (None, ("dixie", "CA-PNF-001"))


def test_redact_removes_map_key_and_email():
    u = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/abc123SECRET/VIIRS_SNPP_SP/-121.9,39.5,-121.2,40.0/5/2018-11-02"
    assert "abc123SECRET" not in RC.redact(u) and "{MAP_KEY}" in RC.redact(u)
    v = RC.redact("https://lfps.usgs.gov/api/job/submit?Email=me@x.org&Layer_List=LF2016_CC&Area_of_Interest=1 2 3 4")
    assert "me@x.org" not in v and "Layer_List=LF2016_CC" in v


def test_errors_in_manifest_never_hold_secrets(tmp_path, monkeypatch):
    """A connection error quotes the request URL; the manifest is public, so the key and email must go."""
    monkeypatch.setenv("FIRMS_MAP_KEY", "k3yK3yK3y")
    monkeypatch.setenv("LANDFIRE_EMAIL", "crew@example.org")
    api = FakeApi()
    b = RG.Batcher(api, "me/raw", "firms", tmp_path, batch_fires=2, min_free_gb=0, keep=True)
    b.add(fire(fire_id="A"), "error", error="ConnectionError: HTTPSConnectionPool(host='firms.modaps.eosdis.nasa.gov', "
                                            "port=443): Max retries exceeded with url: /api/area/csv/k3yK3yK3y/VIIRS_SNPP_SP/x/5/2018-11-02")
    b.add(fire(fire_id="B"), "error", error="ReadTimeout for https://lfps.usgs.gov/api/job/submit?Email=crew%40example.org&Layer_List=x "
                                            "(contact crew@example.org)")
    text = "".join(p.read_text() for p in (tmp_path / "batch").rglob("*.csv"))
    assert "k3yK3yK3y" not in text and "crew@example.org" not in text and "crew%40example.org" not in text


# ------------------------------------------------------------------------------------- resume

def test_latest_manifest_row_decides_done():
    m = pd.DataFrame([dict(fire_id="A", status="error", fetched_utc="2026-11-01"), dict(fire_id="A", status="ok", fetched_utc="2026-11-02"),
                      dict(fire_id="B", status="ok", fetched_utc="2026-11-01"), dict(fire_id="B", status="error", fetched_utc="2026-11-02"),
                      dict(fire_id="C", status="n/a", fetched_utc="2026-11-01"), dict(fire_id="D", status="empty", fetched_utc="2026-11-01")])
    assert RG.done_ids(m) == {"A", "C", "D"}
    assert RG.done_ids(None) == set()


# ------------------------------------------------------------------------------------- batches

class FakeApi:
    def __init__(self, fail=False):
        self.fail, self.commits = fail, []

    def upload_folder(self, repo_id, folder_path, repo_type, commit_message):
        if self.fail:
            raise RuntimeError("503")
        files = sorted(str(p.relative_to(folder_path)) for p in Path(folder_path).rglob("*") if p.is_file())
        self.commits.append((commit_message, files))


def staged(tmp, fid, name="x.csv", size=10):
    d = tmp / "stage" / fid / "firms"
    d.mkdir(parents=True)
    (d / name).write_bytes(b"0" * size)
    return d


def test_batch_uploads_with_manifest_then_deletes(tmp_path):
    api = FakeApi()
    b = RG.Batcher(api, "me/raw", "firms", tmp_path, batch_fires=2, min_free_gb=0)
    b.add(fire(fire_id="A"), "ok", staged(tmp_path, "A"), ["https://firms/x"])
    assert not api.commits
    b.add(fire(fire_id="B"), "n/a", error="no data")
    assert len(api.commits) == 1
    msg, files = api.commits[0]
    assert "fires/A/firms/x.csv" in files and any(f.startswith("manifest/firms/") for f in files)
    assert not list((tmp_path / "batch").iterdir())                      # deleted after upload
    assert b.totals["ok"] == 1 and b.totals["n/a"] == 1


def test_failed_upload_keeps_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(RG.time, "sleep", lambda s: None)
    b = RG.Batcher(FakeApi(fail=True), "me/raw", "firms", tmp_path, batch_fires=1, min_free_gb=0)
    with pytest.raises(RuntimeError, match="batch kept"):
        b.add(fire(fire_id="A"), "ok", staged(tmp_path, "A"))
    assert list((tmp_path / "batch").rglob("x.csv"))


def test_error_status_uploads_no_files(tmp_path):
    api = FakeApi()
    b = RG.Batcher(api, "me/raw", "firms", tmp_path, batch_fires=1, min_free_gb=0)
    b.add(fire(fire_id="A"), "error", staged(tmp_path, "A"), error="HTTP 500")
    assert not any(f.startswith("fires/") for f in api.commits[0][1])


def test_shared_file_added_once(tmp_path):
    src = tmp_path / "ics.zip"
    src.write_bytes(b"zip")
    api = FakeApi()
    b = RG.Batcher(api, "me/raw", "ics209", tmp_path, batch_fires=1, min_free_gb=0, uploaded_shared={"shared/old.zip"})
    b.add_shared(src, "shared/ics209/ics209plus-wildfire.zip")
    b.add_shared(src, "shared/old.zip")
    b.add(fire(fire_id="A"), "empty")
    files = api.commits[0][1]
    assert "shared/ics209/ics209plus-wildfire.zip" in files and "shared/old.zip" not in files


def test_gridmet_url_is_plain_https_subset():
    d = RC.days(CAMP)
    u = RC.gridmet_url("vpd", "daily_mean_vapor_pressure_deficit", CAMP["bounds"], d[0], d[-1], 0.05)
    assert u.startswith("https://thredds.northwestknowledge.net/thredds/ncss/grid/agg_met_vpd_1979_CurrentYear_CONUS.nc?")
    assert "time_start=2018-10-26T00:00:00Z" in u and "time_end=2018-11-27T00:00:00Z" in u and "accept=netcdf" in u
    assert "north=40.0346" in u and "west=-121.9400" in u and "/dodsC/" not in u


class Reply:
    def __init__(self, code, body=b""):
        self.status_code, self.content, self.text = code, body, body.decode(errors="replace")

    def json(self):
        return json.loads(self.content)


class Sess:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def get(self, url, timeout):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_http_get_retries_server_errors_not_client_errors(monkeypatch):
    import requests
    monkeypatch.setattr(RC.time, "sleep", lambda s: None)
    ctx = type("C", (), {})()
    ctx.sess = Sess([Reply(503), requests.ConnectionError("reset"), Reply(200, b"CDF\x01")])
    assert RC.http_get(ctx, "u") == b"CDF\x01" and ctx.sess.calls == 3
    ctx.sess = Sess([Reply(400, b"Requested time range is outside the dataset")])
    with pytest.raises(RuntimeError, match="HTTP 400: Requested time range"):
        RC.http_get(ctx, "u")
    assert ctx.sess.calls == 1


def test_landfire_window_snaps_to_native_30m_grid():
    pytest.importorskip("pyproj")
    x0, y0, x1, y1, w, h = RC.native_window(CAMP["bounds"], (-2362425, 3177435), 30)
    assert (x0 + 2362425) % 30 == 0 and (3177435 - y1) % 30 == 0          # pixel edges of the stored grid
    assert (w, h) == ((x1 - x0) / 30, (y1 - y0) / 30) == (2246, 2195)      # Camp, as served
    info = {"extent": {"xmin": -2362425, "ymax": 3177435}, "pixelSizeX": 30, "spatialReference": {"wkid": 5070},
            "pixelType": "S16", "noDataValue": -9999}
    u = RC.landfire_export_url("LF2016", "FBFM40", CAMP["bounds"], info)
    assert "/Landfire_LF2016/LF2016_FBFM40_CONUS/ImageServer/exportImage?" in u
    assert "size=2246%2C2195" in u and "interpolation=RSP_NearestNeighbor" in u and "compression=LZW" in u
    assert "Email" not in u


class IdSession:
    """ArcGIS layer with 450 features: lists ids, then returns the ones asked for (optionally dropping one)."""

    def __init__(self, drop=None):
        self.drop, self.posts = drop, []

    def get(self, url, params, timeout):
        assert params["returnIdsOnly"] == "true" and "resultOffset" not in params
        return Reply(200, json.dumps({"objectIds": list(range(450, 0, -1))}).encode())

    def post(self, url, data, timeout):
        ids = [int(i) for i in data["objectIds"].split(",") if int(i) != self.drop]
        self.posts.append(len(ids))
        return Reply(200, json.dumps({"type": "FeatureCollection", "features": [{"id": i} for i in ids]}).encode())


def test_box_features_page_by_id_and_check_completeness():
    s = IdSession()
    feats = RC.arcgis_box_features(s, "https://x/FeatureServer/1", CAMP["bounds"])
    assert [f["id"] for f in feats] == list(range(1, 451)) and s.posts == [200, 200, 50]
    with pytest.raises(RuntimeError, match="listed 450 features, received 449"):
        RC.arcgis_box_features(IdSession(drop=7), "https://x/FeatureServer/1", CAMP["bounds"])


def test_netcdf_bytes_fall_back_to_a_file(tmp_path):
    class NoMem:
        def Dataset(self, path, memory=None):
            if memory is not None:
                raise OSError("NetCDF: HDF error")
            return ("opened", Path(path).read_bytes())
    ctx = type("C", (), {"tmp": lambda self: tmp_path})()
    ds, tmp = RC._open_nc_bytes(NoMem(), b"CDF\x01data", ctx)
    assert ds == ("opened", b"CDF\x01data") and tmp.exists()


def test_error_streak_stops_a_broken_channel_only():
    s = RG.ErrorStreak(3)
    assert not any(s.hit(st, "e") for st in ["error", "error", "ok", "error", "error", "n/a"])
    assert [s.hit("error", f"e{i}") for i in range(3)] == [False, False, True] and s.tripped and s.last == "e2"
    never = RG.ErrorStreak(0)
    assert not any(never.hit("error", "e") for _ in range(50))


# ------------------------------------------------------------------------------------- setup check

class WhoAmI:
    def __init__(self, token="hf_x", me=None, fail=False):
        self.token, self.me, self.fail = token, me, fail

    def whoami(self):
        if self.fail:
            raise RuntimeError("401 Client Error: Unauthorized")
        return self.me


ME = {"name": "sam", "orgs": [{"name": "gt-fire"}], "auth": {"accessToken": {"role": "write"}}}


@pytest.mark.parametrize("api, repo, says", [
    (WhoAmI(token=None, fail=True), "sam/raw", "HF_TOKEN is not set"),
    (WhoAmI(fail=True), "sam/raw", "rejected HF_TOKEN"),
    (WhoAmI(me=ME), "your-name/wildfire-raw-2014-2025", "'your-name' is not your account"),
    (WhoAmI(me=dict(ME, auth={"accessToken": {"role": "read"}})), "sam/raw", "read-only"),
    (WhoAmI(me=ME), "no-slash", "account/repo-name"),
])
def test_setup_problems_stop_with_one_clear_line(api, repo, says, capsys):
    with pytest.raises(SystemExit) as e:
        RG.check_hf(api, repo)
    assert e.value.code == RG.SETUP_ERROR and says in capsys.readouterr().err


def test_setup_ok_for_own_account_and_org():
    assert RG.check_hf(WhoAmI(me=ME), "sam/raw") == "sam"
    assert RG.check_hf(WhoAmI(me=ME), "gt-fire/raw") == "sam"


# ------------------------------------------------------------------------------------- per-fire links

def test_fire_links_cover_every_channel_with_files_and_sources():
    lf = "https://lfps.usgs.gov/api/job/submit?Layer_List=LF2016_FBFM40%3BLF2016_CC&Area_of_Interest=-121.89+39.51+-121.24+39.98"
    rows = [dict(fire_id="F2003403265", channel="terrain", status="error", files="", sources="", bytes="0",
                 fetched_utc="2026-11-15T10:00:00+00:00", error="timeout"),
            dict(fire_id="F2003403265", channel="terrain", status="ok", files="dem_glo30.tif", bytes="10106268",
                 sources="https://copernicus-dem-30m.s3.amazonaws.com/a.tif", fetched_utc="2026-11-15T11:00:00+00:00", error=""),
            dict(fire_id="F2003403265", channel="landfire", status="ok", files="landfire_LF2016.zip", bytes="5",
                 sources=lf, fetched_utc="2026-11-15T11:00:00+00:00", error=""),
            dict(fire_id="F2003403265", channel="hrrr", status="ok", files="hrrr_ana.nc;hrrr_fc.nc", bytes="9",
                 sources=RG.SEP.join(["https://h/a.grib2", "https://h/b.grib2"]), fetched_utc="2026-11-15T11:00:00+00:00", error=""),
            dict(fire_id="F2003403265", channel="ics209", status="n/a", files="", bytes="0", sources="",
                 fetched_utc="2026-11-15T11:00:00+00:00", error="ICS-209-PLUS covers 1999-2020")]
    L = RG.fire_links(CAMP, rows, "me/raw")
    assert list(L["channels"]) == RC.CHANNELS
    t = L["channels"]["terrain"]
    assert t["status"] == "ok" and t["files"][0]["url"] == \
        "https://huggingface.co/datasets/me/raw/resolve/main/fires/F2003403265/terrain/dem_glo30.tif"
    assert L["channels"]["landfire"]["sources"] == [lf]                     # ';' inside a URL is not a separator
    assert len(L["channels"]["hrrr"]["sources"]) == 2 and len(L["channels"]["hrrr"]["files"]) == 2
    assert L["channels"]["ics209"]["note"].startswith("ICS-209") and L["channels"]["goes"] == {"status": "pending"}
    assert L["folder"].endswith("/tree/main/fires/F2003403265")
    md = RG.links_markdown(L)
    assert "[dem_glo30.tif](https://huggingface.co/datasets/me/raw/resolve/main/fires/F2003403265/terrain/dem_glo30.tif)" in md
    assert "| goes | pending | — | — |" in md and "hrrr: 2 links" in md


def test_query_links_are_complete_urls():
    u = RC.arcgis_query_url("https://x/FeatureServer/0", CAMP["bounds"], where="FIRE_YEAR_INT < 2018")
    assert "f=geojson" in u and "where=FIRE_YEAR_INT+%3C+2018" in u and "geometry=-121.88999%2C39.51118" in u


# ------------------------------------------------------------------------------------- helpers

def test_windowed_map_keeps_order():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        assert list(RC._windowed_map(ex, lambda k: k * k, range(50), 3)) == [k * k for k in range(50)]


def test_ncseries_keeps_packed_values(tmp_path):
    np = pytest.importorskip("numpy")
    netCDF4 = pytest.importorskip("netCDF4")
    lat, lon = np.zeros((2, 3)), np.ones((2, 3))
    s = RC.NcSeries(tmp_path / "t.nc", lat, lon, {"title": "t"})
    raw = np.array([[1, 2, 3], [4, -99, 6]], np.int16)
    s.append(10, "s3://b/k1", {"Power": (raw, np.int16, {"_FillValue": np.int16(-99), "scale_factor": 0.5})})
    s.append(20, "s3://b/k2", {"Power": (raw + 1, np.int16, {"_FillValue": np.int16(-99), "scale_factor": 0.5})})
    s.close()
    with netCDF4.Dataset(tmp_path / "t.nc") as ds:
        v = ds.variables["Power"]
        v.set_auto_maskandscale(False)
        assert (v[0] == raw).all() and v.scale_factor == 0.5
        v.set_auto_maskandscale(True)
        assert v[0][0, 1] == 1.0 and np.ma.is_masked(v[0][1, 1])
        assert list(ds.variables["time"][:]) == [10, 20] and ds.variables["source"][1] == "s3://b/k2"
