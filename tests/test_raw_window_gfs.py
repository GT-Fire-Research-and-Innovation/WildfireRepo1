"""Raw-data grab, the fire's extent and the gfs channel, offline: activity before the first perimeter and after
the last widens the time window only when it chains onto the fire; the region reaches 32 km past the perimeters'
box on every side at any latitude; the ignition point is the earliest observation (first VIIRS overpass or first
mapped perimeter); a changed window or region sends fires back to the queue; --extents writes the catalog file
from firms files; and GFS from Earth Engine is cut on its own 0.25 deg grid (requests shared between fires,
values untouched, leads as far as the collection holds them).
Run: python -m pytest tests/test_raw_window_gfs.py -q"""
import gzip
import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import raw_channels as RC  # noqa: E402
import raw_grab as RG  # noqa: E402

np = pytest.importorskip("numpy")

CAMP = dict(fire_id="F2003403265", incident="75E64DB8-9B75-4A68-BEDD-67CC62658E38", name="CAMP", year=2018,
            first="2018-11-09 00:00:00.000+00:00", last="2018-11-25 18:22:00.000+00:00", rating="keep",
            bounds=[-121.88999101802999, 39.51118309776612, -121.23979602288742, 39.98463882716831],
            perim_bounds=[-121.777828, 39.598585, -121.352605, 39.897816])


def fire(**kw):
    return dict(CAMP, **kw)


def day(s):
    return pd.Timestamp(s)


def detections(rows):
    """FIRMS-like frame from (date, n, lat, lon, confidence, type[, acq_time HHMM])."""
    out = []
    for d, n, lat, lon, conf, typ, *hhmm in rows:
        out += [dict(latitude=lat, longitude=lon, acq_date=d, acq_time=hhmm[0] if hhmm else 2100, confidence=conf,
                     type=typ, bright_ti4=330.0)] * n
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------- window rule

def test_activity_chains_onto_the_perimeter_days_across_short_gaps():
    p0, p1 = day("2018-11-10"), day("2018-11-15")
    act = [day(d) for d in ("2018-11-01", "2018-11-05", "2018-11-07", "2018-11-08", "2018-11-12",
                            "2018-11-16", "2018-11-20", "2018-11-30")]
    s, e = RC.activity_window(p0, p1, act)
    # back: 11-08 (2 days before), 11-07, 11-05 (2 quiet days) join; 11-01 is 4 days off 11-05: another fire
    # forward: 11-16 joins; 11-20 is 4 days past it
    assert (s, e) == (day("2018-11-05"), day("2018-11-16"))
    assert RC.activity_window(p0, p1, []) == (p0, p1)


def test_active_days_ignore_low_confidence_static_sources_and_neighbours():
    lat, lon = 39.7, -121.6
    df = detections([("2018-11-05", 3, lat, lon, "n", 0),          # active
                     ("2018-11-06", 1, lat, lon, "h", 0),          # one detection: not enough
                     ("2018-11-07", 5, lat, lon, "l", 0),          # low confidence
                     ("2018-11-08", 5, lat, lon, "h", 2),          # static land source
                     ("2018-11-09", 5, 39.0, lon, "h", 0),         # outside the perimeters' box
                     ("2018-11-10", 2, 39.9, -121.35, "nominal", 0)])   # on the box edge (pad)
    assert RC.active_days(df, CAMP["perim_bounds"]) == [day("2018-11-05"), day("2018-11-10")]
    assert RC.active_days(None, CAMP["perim_bounds"]) == []
    nrt = df.drop(columns=["type"])                                 # NRT files have no type column
    assert day("2018-11-08") in RC.active_days(nrt, CAMP["perim_bounds"])


def test_window_row_and_span_move_every_time_dependent_channel():
    lat, lon = 39.7, -121.6
    df = detections([("2018-11-06", 4, lat, lon, "n", 0), ("2018-11-07", 9, lat, lon, "h", 0),
                     ("2018-11-27", 3, lat, lon, "n", 0)])
    row = RC.fire_extent(CAMP, df)
    assert (row["start_day"], row["end_day"]) == ("2018-11-05", "2018-11-28")
    assert (row["active_days_before"], row["active_days_after"], row["window_rule"]) == (3, 2, "perimeters+firms")
    assert RC.fire_extent(CAMP)["start_day"] == "2018-11-08" and RC.fire_extent(CAMP)["window_rule"] == "perimeters"

    f = RC.attach_extents([fire()], {CAMP["fire_id"]: row})[0]
    t0, t1 = RC.span(f)
    assert (t0, t1) == (pd.Timestamp("2018-11-05", tz="UTC"), pd.Timestamp("2018-11-28 23:00", tz="UTC"))
    d = RC.days(f)                                    # gridmet: 14 / 2 days around, moved out with the window
    assert (d[0], d[-1]) == (day("2018-10-23"), day("2018-11-29"))
    ana, fc = RC.hrrr_times(f)
    assert ana[0] == t0 and fc[-1] == pd.Timestamp("2018-11-28 12:00", tz="UTC")
    assert RC.goes_hours(f)[0][0] == t0 and RC.gfs_times(f)[0][0] == pd.Timestamp("2018-11-05 12:00", tz="UTC")
    fd = RC.firms_days(f)                             # firms ignores the window: it is the window's input
    assert (fd[0], fd[-1]) == (day("2018-10-26"), day("2018-12-09"))
    assert RC.window_label("hrrr", f) == "2018-11-05/2018-11-28" and RC.window_label("terrain", f) == ""


def test_fires_fetched_for_another_window_or_region_are_fetched_again():
    a, b, c = fire(fire_id="A"), fire(fire_id="B"), fire(fire_id="C")
    RC.attach_extents([b], {"B": {"start_day": "2018-11-01", "end_day": "2018-11-30"}})
    reg = RC.region_label("hrrr", a)
    m = pd.DataFrame([dict(fire_id="A", status="ok", fetched_utc="2026-11-01", window=RC.window_label("hrrr", a), region=reg),
                      dict(fire_id="B", status="ok", fetched_utc="2026-11-01", window=RC.window_label("hrrr", a), region=reg),
                      dict(fire_id="C", status="ok", fetched_utc="2026-11-01", window="", region="")])   # before the rules
    assert RG.done_ids(m, "hrrr", [a, b, c]) == {"A"}                         # B's window grew, C has neither
    assert RG.done_ids(m, "terrain", [a, b, c]) == {"A", "B"}                 # region only: C was cut to the old box
    gz = m.assign(files="records.geojson.gz")
    assert RG.done_ids(gz, "perimeters", [a, b, c]) == {"A", "B", "C"}        # on the build's box, no window
    plain = gz.assign(files=["records.geojson", "records.geojson.gz", "records.geojson"])
    assert RG.done_ids(plain, "perimeters", [a, b, c]) == {"B"}               # plain .geojson: repack or refetch
    old = m.drop(columns=["window", "region"])                                 # manifests written before the columns
    assert RG.done_ids(old, "goes", [a, b, c]) == set() and RG.done_ids(old, "goes") == {"A", "B", "C"}


def test_batch_rows_record_the_window(tmp_path):
    b = RG.Batcher(None, "me/raw", "gfs", tmp_path, dry_run=True)
    f = RC.attach_extents([fire()], {CAMP["fire_id"]: {"start_day": "2018-11-05", "end_day": "2018-11-28"}})[0]
    b.add(f, "n/a", error="x")
    assert b.rows[-1]["window"] == "2018-11-05/2018-11-28" and "window" in RG.MANIFEST_COLS
    assert b.rows[-1]["region"] == RC.region_label("gfs", f) != "" and "region" in RG.MANIFEST_COLS


def km_between(lat1, lon1, lat2, lon2):
    """Great-circle distance, km."""
    import math
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


@pytest.mark.parametrize("lat", [25.0, 39.7, 48.5, 64.0])
def test_region_reaches_32_km_past_the_perimeters_on_every_side(lat):
    pb = [-121.7, lat - 0.05, -121.6, lat + 0.05]                   # a 9 x 11 km fire
    f = fire(perim_bounds=pb, bounds=RC.buffer_box(pb, 9.6))
    w, s, e, n = RC.area(f)
    assert km_between(s, -121.65, pb[1], -121.65) >= 32 and km_between(pb[3], -121.65, n, -121.65) >= 32
    for y in (s, pb[1], pb[3], n):                                  # east-west margin holds at every latitude of the box
        assert km_between(y, w, y, pb[0]) >= 32 and km_between(y, pb[2], y, e) >= 32
    assert km_between(s, w, s, e) >= 64 and km_between(s, w, n, w) >= 64
    assert w <= f["bounds"][0] and n >= f["bounds"][3]               # never smaller than the build's box
    tiny = fire(perim_bounds=[-121.6, lat, -121.6, lat], bounds=[-121.6, lat, -121.6, lat])
    tw, ts, te, tn = RC.area(tiny)                                   # a point fire: a 64 km square (+ the 1 km margin)
    assert 64 <= km_between(ts, tw, tn, tw) <= 67 and 64 <= km_between(ts, tw, ts, te) <= 67


def test_ignition_is_the_earliest_observation_of_the_fire():
    lat, lon = 39.7, -121.6
    df = detections([("2018-11-07", 3, lat, lon, "h", 0, 954),       # first overpass that day
                     ("2018-11-07", 2, lat + 0.02, lon, "n", 0, 1003),   # same overpass (9 min later)
                     ("2018-11-07", 4, lat + 0.1, lon + 0.1, "h", 0, 2112),  # next overpass: grown
                     ("2018-11-08", 9, lat + 0.1, lon + 0.1, "h", 0, 2050),
                     ("2018-10-20", 9, lat, lon, "h", 0, 2000)])     # weeks earlier: not chained to this fire
    act = RC.active_days(df, CAMP["perim_bounds"])
    y, x, t, how = RC.ignition_point(CAMP, df, act)
    assert how == "firms" and t == pd.Timestamp("2018-11-07 09:54", tz="UTC")
    assert y == pytest.approx(lat + 0.008) and x == pytest.approx(lon)
    perim_first = (pd.Timestamp("2018-11-07 06:00", tz="UTC"), 39.71, -121.61)   # mapped before that overpass
    assert RC.ignition_point(CAMP, df, act, perim_first)[3] == "first_perimeter"
    single = detections([("2018-11-10", 1, lat, lon, "h", 0, 2030)])          # never an active day
    assert RC.ignition_point(CAMP, single, [])[3] == "firms_single"
    y, x, t, how = RC.ignition_point(CAMP)
    assert how == "perimeter_box_centre" and y == pytest.approx(39.7482, abs=1e-3)


def test_first_perimeter_point_is_the_centre_of_the_earliest_arrival():
    arrival = np.full((10, 10), -1, "i2")
    arrival[2:4, 6:8] = 0
    arrival[2:9, 1:9] = np.where(arrival[2:9, 1:9] == 0, 0, 1)
    times = np.array([pd.Timestamp("2018-11-09", tz="UTC").value, pd.Timestamp("2018-11-10", tz="UTC").value])
    t, y, x = RC.first_perimeter_point(arrival, [0.01, 0.0, -122.0, 0.0, -0.01, 40.0], times)
    assert t == pd.Timestamp("2018-11-09", tz="UTC") and y == pytest.approx(40.0 - 0.03) and x == pytest.approx(-122.0 + 0.07)   # cells 2-3 x 6-7: centre 3.0, 7.0
    assert RC.first_perimeter_point(np.full((3, 3), -1), [0.01, 0, -122, 0, -0.01, 40], times) is None


def test_extents_step_writes_catalog_from_local_firms_files(tmp_path, monkeypatch):
    batch = tmp_path / "batch" / "firms-x-000"
    fd = batch / "fires" / CAMP["fire_id"] / "firms"
    fd.mkdir(parents=True)
    with gzip.open(fd / "VIIRS_SNPP_SP.csv.gz", "wt") as fh:
        detections([("2018-11-07", 6, 39.7, -121.6, "h", 0)]).to_csv(fh, index=False)
    (batch / "manifest" / "firms").mkdir(parents=True)
    other = fire(fire_id="F0000000000")
    pd.DataFrame([dict(fire_id=CAMP["fire_id"], status="ok", fetched_utc="2026-11-18", window=RC.window_label("firms", CAMP))],
                 columns=RG.MANIFEST_COLS).to_csv(batch / "manifest" / "firms" / "firms-x-000.csv", index=False)
    monkeypatch.setattr(RG, "load_fires", lambda *a, **k: [fire(), other])
    first = (pd.Timestamp("2018-11-09 02:00", tz="UTC"), 39.75, -121.55)          # the other fire: no FIRMS, a perimeter
    monkeypatch.setattr(RG, "first_perimeters", lambda repo, fires, cache: {"F0000000000": first})
    a = Namespace(work=str(tmp_path / "w"), dry_run=True, hf_repo=None, source_repo="x", fires="built", ids=None,
                  local_manifests=str(tmp_path / "batch"))
    assert RG.build_extents(a) == 0
    t = pd.read_csv(tmp_path / "w" / "extents" / RC.EXTENTS_FILE).set_index("fire_id")
    assert t.loc[CAMP["fire_id"], "start_day"] == "2018-11-06" and t.loc[CAMP["fire_id"], "window_rule"] == "perimeters+firms"
    assert t.loc["F0000000000", "start_day"] == "2018-11-08" and t.loc["F0000000000", "window_rule"] == "perimeters"
    assert t.loc[CAMP["fire_id"], "ignition_method"] == "firms" and t.loc[CAMP["fire_id"], "ignition_time_utc"] == "2018-11-07T21:00Z"
    assert t.loc["F0000000000", "ignition_method"] == "first_perimeter" and t.loc["F0000000000", "ignition_lat"] == 39.75
    assert t.loc[CAMP["fire_id"], "region_west"] == round(RC.area(CAMP)[0], 5)
    w = RG.load_extents(None, None, path=tmp_path / "w" / "extents" / RC.EXTENTS_FILE)
    assert RC.span(RC.attach_extents([fire()], w)[0])[0] == pd.Timestamp("2018-11-06", tz="UTC")


# ------------------------------------------------------------------------------------- gfs

GRID = ("EPSG:4326", (0.25, 0.0, -180.125, 0.0, -0.25, 90.125), 1440, 721)        # cell centres on 0.25 deg
GRID360 = ("EPSG:4326", (0.25, 0.0, -0.125, 0.0, -0.25, 90.125), 1440, 721)


def test_gfs_leads_follow_what_the_collection_holds():
    t = RC.gfs_times(CAMP)
    runs = sorted({r for r, _ in t})
    assert len(runs) == 19 and len(t) == 19 * 80 and max(h for _, h in t) == 240
    early = RC.gfs_times(fire(year=2016, first="2016-07-07 00:00+00:00", last="2016-07-08 00:00+00:00"))
    assert {h for _, h in early} == set(range(3, 37, 3))
    assert RC.gfs_times(fire(year=2015, first="2015-06-01 00:00+00:00", last="2015-06-10 00:00+00:00")) == []
    assert RC.applies("gfs", fire(year=2015, first="2015-06-01 00:00+00:00", last="2015-06-10 00:00+00:00"))
    assert RC.applies("gfs", CAMP) is None


def test_gfs_window_is_cut_on_the_native_grid_either_longitude_convention():
    w = RC.grid_window(GRID[1], GRID[2], GRID[3], CAMP["bounds"])
    lat, lon = RC.grid_latlon(GRID[1], GRID[2], w)
    h = 0.125                                                                   # cells (not centres) cover the box
    assert lat[0, 0] + h >= CAMP["bounds"][3] and lat[-1, 0] - h <= CAMP["bounds"][1]
    assert lon[0, 0] - h <= CAMP["bounds"][0] and lon[0, -1] + h >= CAMP["bounds"][2]
    assert lon[0, 1] - h > CAMP["bounds"][0] and lon[0, -2] + h < CAMP["bounds"][2]  # and no more than that
    assert np.allclose((lat * 4) % 1, 0) and np.allclose((lon * 4) % 1, 0)       # whole 0.25 deg cells, no shift
    w360 = RC.grid_window(GRID360[1], GRID360[2], GRID360[3], CAMP["bounds"])
    lat3, lon3 = RC.grid_latlon(GRID360[1], GRID360[2], w360)
    assert np.allclose(lat3, lat) and np.allclose(lon3, lon) and w360[2] > 720


def test_gfs_fires_of_one_run_share_requests_within_a_size_limit():
    wins = {"a": (10, 13, 100, 103), "b": (11, 14, 102, 105), "c": (300, 303, 900, 903)}
    cl = RC.gfs_clusters(wins, max_cells=100)
    assert sorted(sorted(ids) for _, ids in cl) == [["a", "b"], ["c"]]
    assert dict((tuple(ids), w) for w, ids in cl)[("a", "b")] == (10, 14, 100, 105)
    assert len(RC.gfs_clusters(wins, max_cells=9)) == 3
    assert [len(c) for c in RC._lead_chunks(list(RC.GFS_LEADS))] == [45, 35]


class Ctx:
    def __init__(self, tmp):
        self.tmp, self.threads, self._m = tmp, 2, {}

    def stage(self, fid):
        p = self.tmp / "stage" / fid / "gfs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def drop_stage(self, fid):
        pass

    def expired(self):
        return False

    def memo(self, key, fn):
        if key not in self._m:
            self._m[key] = fn()
        return self._m[key]


def test_gfs_series_per_fire_from_shared_requests(tmp_path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4")
    calls = []

    def fake_fetch(ee, run, leads, win, grid):
        calls.append((run, tuple(leads), win))
        r0, r1, c0, c1 = win
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        if run.day == 10:
            raise RuntimeError("Too many concurrent aggregations")
        return {h: {"temperature_2m_above_ground": (rr * 10000 + cc + h / 1000).astype("f4"),
                    "total_precipitation_surface": np.full(rr.shape, h, "f4")} for h in leads if h % 6 == 0}

    monkeypatch.setattr(RC, "gee_init", lambda: "ee")
    monkeypatch.setattr(RC, "gfs_grid", lambda ee: GRID)
    monkeypatch.setattr(RC, "gfs_fetch", fake_fetch)
    short = dict(first="2018-11-09 00:00+00:00", last="2018-11-09 06:00+00:00")
    near = fire(**short)                                             # runs 11-08, 11-09, 11-10
    next_door = fire(fire_id="next", bounds=[-121.2, 39.6, -121.0, 39.9], **short)
    old = fire(fire_id="old", year=2014, first="2014-08-01 00:00+00:00", last="2014-08-02 00:00+00:00")
    res = {f["fire_id"]: (s, src, err) for f, s, src, err in RC.grab_gfs_many([near, next_door, old], Ctx(tmp_path))}
    assert res["old"][0] == "empty"
    assert {k for k, (s, _, _) in res.items() if s == "ok"} == {near["fire_id"], "next"}
    assert "1 run(s) failed" in res["next"][2] and "Too many concurrent" in res["next"][2]
    assert len(res["next"][1]) == 2 and res["next"][1][0].startswith("ee:NOAA/GFS0P25 creation_time=2018-11-08T12:00Z")
    assert len({c[2] for c in calls}) == 1 and len(calls) == 2 + 2 + 1           # both fires share each request;
    #                                                                           the failing run stops at its first chunk

    w = RC.grid_window(GRID[1], GRID[2], GRID[3], RC.area(near))
    with netCDF4.Dataset(tmp_path / "stage" / near["fire_id"] / "gfs" / "gfs.nc") as ds:
        assert ds["time"].shape[0] == 2 * 40                                      # 2 runs that worked x leads 6..240 by 6
        lead, run = ds["lead_h"][:], ds["run_time"][:]
        assert lead[0] == 6 and lead[-1] == 240 and run[0] == pd.Timestamp("2018-11-08 12:00", tz="UTC").value
        assert ds["time"][0] == pd.Timestamp("2018-11-08 18:00", tz="UTC").value
        t = ds["temperature_2m_above_ground"][0]
        rr, cc = np.meshgrid(np.arange(w[0], w[1]), np.arange(w[2], w[3]), indexing="ij")
        assert np.allclose(t, rr * 10000 + cc + 0.006) and ds["temperature_2m_above_ground"].units == "degC"
        assert ds["lat"].shape == (w[1] - w[0], w[3] - w[2]) and ds.crs == "EPSG:4326"
        assert "forecast_hours=6" in ds["source"][0]


def test_earth_engine_key_never_reaches_the_manifest(monkeypatch):
    key = json.dumps({"type": "service_account", "project_id": "p", "private_key_id": "abc123def456",
                      "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----\n",
                      "client_email": "grab@p.iam.gserviceaccount.com"})
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT_KEY", key)
    out = RC.redact(f"failed with {key}")
    assert "MIIEvQ" not in out and "abc123def456" not in out
    loose = RC.redact('{"private_key_id": "zz99zz", "private_key": "-----BEGIN PRIVATE KEY-----\\nQQQ\\n-----END PRIVATE KEY-----"}')
    assert "zz99zz" not in loose and "QQQ" not in loose
