"""Terrain-adjusted wind (WindNinja response applied to HRRR): the rule that combines the two stored runs,
the loader option, and, when WindNinja is installed (WINDNINJA_CLI=/path/to/WindNinja_cli), the real thing
on a small ridge."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))
import data as D  # noqa: E402
import windninja_fields as WF  # noqa: E402

H = W = 64
FLAT = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)[:, :, None, None] * np.ones((1, 1, H, W), np.float32)


def test_flat_ground_changes_nothing():
    u, v = np.full((3, H, W), 4.0, np.float32), np.full((3, H, W), -2.0, np.float32)
    u2, v2 = D.terrain_wind(FLAT, u, v)
    assert np.allclose(u2, u) and np.allclose(v2, v)


def test_response_is_applied_per_cell_and_adds_up():
    r = FLAT.copy()
    r[0, :, 10, 10] = (1.5, 0.5)           # at this cell an east-blowing wind speeds up and turns north
    r[1, :, 10, 10] = (0.0, 0.6)           # and a north-blowing wind is sheltered
    u, v = np.full((1, H, W), 2.0, np.float32), np.full((1, H, W), 10.0, np.float32)
    u2, v2 = D.terrain_wind(r, u, v)
    assert np.isclose(u2[0, 10, 10], 2 * 1.5 + 10 * 0.0) and np.isclose(v2[0, 10, 10], 2 * 0.5 + 10 * 0.6)
    assert np.isclose(u2[0, 0, 0], 2.0) and np.isclose(v2[0, 0, 0], 10.0)


def test_unit_winds_match_the_directions_that_are_run():
    assert np.allclose(WF.unit(270.0), (1, 0), atol=1e-9)        # from the west blows east
    assert np.allclose(WF.unit(180.0), (0, 1), atol=1e-9)        # from the south blows north
    a, b = WF.unit(WF.CHECK_DIR)
    assert np.allclose(a * WF.unit(WF.DIRS[0]) + b * WF.unit(WF.DIRS[1]), WF.unit(WF.CHECK_DIR))


def test_fill_nearest_and_utm_zone():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    a[0, 0] = np.nan
    assert np.isfinite(WF.fill_nearest(a)).all() and WF.fill_nearest(a)[0, 0] in (1.0, 4.0)
    assert WF.utm_epsg(-121.5, 39.8) == 32610 and WF.utm_epsg(-105.0, 40.0) == 32613


def make_fire(tmp, resp=None):
    """A fire with a steady 6 m/s wind blowing east in HRRR, and optionally a terrain response file."""
    d = tmp / "F1"
    d.mkdir(exist_ok=True)
    (d / "grid.json").write_text(json.dumps(dict(fire_id="F1", year=2019, height=H, width=W, res_m=150.0, name="t",
                                                 transform=[0.0017, 0, -120.0, 0, -0.00135, 40.0])))
    arrival = -np.ones((H, W), np.int16)
    arrival[28:36, 28:36] = 0
    arrival[24:28, 28:36] = 1
    times = pd.to_datetime(["2019-08-10 18:00", "2019-08-11 18:00"], utc=True)
    np.savez(d / "labels.npz", arrival=arrival, other=np.zeros((H, W), bool), times=times.as_unit("ns").asi8)
    z = np.zeros((H, W), np.float32)
    np.savez(d / "terrain.npz", elev=z + 1000, slope=z, aspect=z, tpi_500m=z, tpi_2km=z, rough=z)
    days = pd.date_range("2019-08-01", "2019-08-13")
    gm = np.full((len(days), len(D.GM), 2, 2), 10.0, np.float32)
    np.savez(d / "weather.npz", dates=np.array([str(x.date()) for x in days]), gm=gm, gm_vars=np.array(D.GM),
             gm_row=np.zeros(H, np.int16), gm_col=np.zeros(W, np.int16))
    hours = pd.date_range("2019-08-10 12:00", "2019-08-11 21:00", freq="3h", tz="UTC")
    fields = np.zeros((len(hours), 4, 2, 2), np.float32)
    fields[:, 0], fields[:, 1], fields[:, 3] = 9.0, 6.0, 30.0            # gust, u (east), rh; v stays 0
    np.savez(d / "hrrr.npz", times=np.array([str(t) for t in hours]), fields=fields, idx=np.zeros((H, W), np.int32))
    if resp is not None:
        np.savez(d / "terrain_wind.npz", resp=resp.astype(np.float16), valid=np.ones((H, W), bool))
    return D.Fire(d)


@pytest.fixture
def settings():
    keep = (D.WIND_TERRAIN, D.HRRR_ROTATE)
    D.HRRR_ROTATE = False                       # keep the test wind exactly east
    D.TW_MISSING.clear()
    D._CACHE.clear()                            # other test files also call their fire "F1"
    yield
    D.WIND_TERRAIN, D.HRRR_ROTATE = keep
    D._CACHE.clear()


def hrrr(f):
    return f.sample(0, 24.0, (0, H, 0, W))[0]["hrrr"]


def test_loader_applies_the_response_only_when_asked(tmp_path, settings):
    r = FLAT.copy()
    r[0, 0, :, 40:] = 1.5                       # east half: east-blowing wind 50 % faster
    r[0, 1, :, :20] = 1.0                       # west strip: turned 45 degrees to the north
    f = make_fire(tmp_path, r)
    D.WIND_TERRAIN = False
    off = hrrr(f)
    assert np.allclose(off[2] * 8, 6.0, atol=1e-3) and np.allclose(off[1], 0.0, atol=1e-6)
    D.WIND_TERRAIN = D.wind_file(1)
    D._CACHE.clear()
    on = hrrr(f)
    assert np.allclose(on[2][:, 40:] * 8, 9.0, atol=1e-2)            # speed-up where the response says so
    assert np.allclose(on[2][:, 20:40] * 8, 6.0, atol=1e-2)          # untouched where the ground is flat
    assert np.allclose(on[1][:, :20] * 8, 6.0, atol=1e-2)            # northward part appears in the west strip
    assert np.allclose(on[4][:, 40:] * 20, 13.5, atol=2e-2)          # gusts follow the local speed (9 x 1.5)
    assert np.allclose(f.broad_spd, 6.0, atol=1e-3)                  # evaluation bins cells by the weather model's wind
    assert not D.TW_MISSING


def test_missing_file_is_recorded_and_wind_left_alone(tmp_path, settings):
    f = make_fire(tmp_path)
    D.WIND_TERRAIN = D.wind_file(1)
    assert np.allclose(hrrr(f)[2] * 8, 6.0, atol=1e-3)
    assert D.TW_MISSING == {"F1"}


def test_the_finer_version_is_a_separate_file(tmp_path, settings):
    assert D.wind_file(0) is False and D.wind_file(1) == "terrain_wind" and D.wind_file(75) == "terrain_wind75"
    f = make_fire(tmp_path, FLAT)                                    # the grid-terrain file says: no change
    r = FLAT.copy()
    r[0, 0] = 2.0
    np.savez(f.d / "terrain_wind75.npz", resp=r.astype(np.float16), valid=np.ones((H, W), bool))
    D.WIND_TERRAIN = D.wind_file(75)
    assert np.allclose(hrrr(f)[2] * 8, 12.0, atol=1e-2)
    D.WIND_TERRAIN = D.wind_file(1)
    D._CACHE.clear()
    assert np.allclose(hrrr(f)[2] * 8, 6.0, atol=1e-2)


def test_mesh_is_coarsened_to_fit_memory():
    assert WF.mesh_for(100_000, 150.0) == 150.0
    assert WF.mesh_for(1_600_000, 75.0) == 150.0                     # four times too many cells: twice the mesh
    assert 75.0 < WF.mesh_for(600_000, 75.0) <= 95.0


@pytest.mark.skipif(not os.environ.get("WINDNINJA_CLI"), reason="set WINDNINJA_CLI to run WindNinja itself")
def test_windninja_on_a_ridge(tmp_path):
    """A north-south ridge: a west wind speeds up over the crest, and the stored response reproduces a
    direct north-west run (the linearity the whole scheme rests on)."""
    n = 80
    x = np.arange(n) - n / 2
    elev = (1000 + 400 * np.exp(-(x / 8.0) ** 2))[None, :].repeat(n, 0).astype(np.float32)
    g = dict(height=n, width=n, res_m=150.0, transform=[150 / (111320 * np.cos(np.radians(40))), 0, -120.0, 0, -150 / 110570, 40.0])
    out = WF.response(elev, g, os.environ["WINDNINJA_CLI"], str(tmp_path))
    r = out["resp"].astype(np.float32)
    assert out["valid"].all() and float(out["check_p95"]) < 0.02
    crest, plain = r[0, 0, 20:60, n // 2], r[0, 0, 20:60, 5]
    assert crest.mean() > 1.15 and abs(plain.mean() - 1.0) < 0.15     # faster on the crest, about unchanged far upwind
    assert abs(r[1, 1, 20:60, 20:60].mean() - 1.0) < 0.1              # wind along the ridge is barely changed
