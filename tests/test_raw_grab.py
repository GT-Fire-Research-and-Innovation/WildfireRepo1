"""Raw-data grab (pipeline/raw_grab.py, raw_channels.py), offline: time windows match the build, channels
skip fires they have no data for, secrets never reach the manifest, dirty IRWIN keys still match, resume
reads manifests right, and a batch is deleted only after its upload worked.
Run: python -m pytest tests/test_raw_grab.py -q"""
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
    assert {c for c in RC.CHANNELS if RC.applies(c, old)} == {"suppression", "landfire", "tessera", "goes"}
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
