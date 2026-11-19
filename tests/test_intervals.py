"""Stage 0 (interval-censored labels): synthetic fires with KNOWN burn times. Every label must contain the true burn
time, use every later perimeter within the horizon, censor correctly at the horizon and at the last perimeter, widen
for uncertain perimeter times, and leave the old next-perimeter target and mask untouched."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
import data as D  # noqa: E402
import intervals as IV  # noqa: E402

TIMES = np.array([0.0, 5.0, 26.0, 30.0, 55.0, 80.0, 120.0])     # perimeter times (h), irregular like the real ones


def known_fire(n=81, speed_kmh=0.12, cell_km=0.15):
    """Circular fire at constant speed: true burn time T (h) per cell, and the arrival index the perimeters give."""
    yy, xx = np.mgrid[:n, :n]
    T = np.hypot(yy - n // 2, xx - n // 2) * cell_km / speed_kmh
    arrival = np.full(T.shape, -1, np.int64)
    for k in range(len(TIMES) - 1, -1, -1):
        arrival[T <= TIMES[k]] = k
    return T, arrival


@pytest.mark.parametrize("i", range(len(TIMES) - 1))
def test_true_burn_time_is_inside_every_label(i):
    T, arr = known_fire()
    lo, hi = IV.perimeter_bounds(TIMES - TIMES[i])
    t_lo, t_hi, valid = IV.interval_labels(arr, i, lo, hi)
    rel = T - TIMES[i]
    ev = valid & np.isfinite(t_hi)
    cen = valid & np.isinf(t_hi)
    assert ev.any() or i >= 4
    assert (t_lo[ev] < rel[ev]).all() and (rel[ev] <= t_hi[ev]).all()       # burned inside (t_lo, t_hi]
    assert (rel[cen] > t_lo[cen]).all()                                       # censored: had not burned by t_lo
    assert (t_hi[ev] <= IV.HORIZON_H).all()                                   # no event interval past the horizon
    assert not valid[(arr >= 0) & (arr <= i)].any()                           # burned at the start: left out
    assert valid[arr < 0].all()


def test_every_later_perimeter_supervises_the_sample():
    T, arr = known_fire()
    lo, hi = IV.perimeter_bounds(TIMES)
    t_lo, t_hi, valid = IV.interval_labels(arr, 0, lo, hi)
    for k in (1, 2, 3, 4):                                    # perimeters at 5, 26, 30 and 55 h: all inside 72 h
        sel = arr == k
        assert sel.any() and valid[sel].all()
        assert np.allclose(t_lo[sel], TIMES[k - 1]) and np.allclose(t_hi[sel], TIMES[k]), k


def test_horizon_and_last_perimeter_censoring():
    T, arr = known_fire()
    lo, hi = IV.perimeter_bounds(TIMES)
    t_lo, t_hi, valid = IV.interval_labels(arr, 0, lo, hi)
    s5, s6, never = arr == 5, arr == 6, arr < 0
    # the last perimeter inside 72 h is at 55 h: (55, 80] crosses 72 h and is kept as "not burned by 55 h", not
    # dropped; later burns and unburned cells are censored at the SAME time (non-informative censoring)
    for sel in (s5, s6, never):
        assert valid[sel].all() and np.isinf(t_hi[sel]).all() and np.allclose(t_lo[sel], 55)
    # next perimeter already past the horizon: the sample carries no information
    lo4, hi4 = IV.perimeter_bounds(TIMES[:5].tolist() + [130.0])
    assert not IV.interval_labels(np.where(arr > 5, -1, arr), 4, lo4 - 55, hi4 - 55)[2].any()
    # from the 80 h perimeter the series ends 40 h later: unburned cells are censored there, not at 72 h
    lo5, hi5 = IV.perimeter_bounds(TIMES - 80)
    a, b, v = IV.interval_labels(arr, 5, lo5, hi5)
    assert np.allclose(a[never], 40) and np.isinf(b[never]).all()
    assert v[s6].all() and np.allclose(a[s6], 0) and np.allclose(b[s6], 40)          # (80, 120] is (0, 40] from 80 h


@pytest.fixture
def slack12():
    keep = IV.EDIT_SLACK_H, IV.MAPPED_SLACK_H
    IV.EDIT_SLACK_H, IV.MAPPED_SLACK_H = 12.0, 0.0
    yield
    IV.EDIT_SLACK_H, IV.MAPPED_SLACK_H = keep


def test_uncertain_times_widen_only_the_uncertain_side(slack12):
    T, arr = known_fire()
    src = ["mapped", "date_only", "edit", "mapped", "mapped", "mapped", "mapped"]
    lo, hi = IV.perimeter_bounds(TIMES, src)
    t_lo, t_hi, valid = IV.interval_labels(arr, 0, lo, hi)
    assert np.allclose(t_hi[arr == 1], 5 + 24)        # bare date stamped at local midnight: any time that day ...
    assert np.allclose(t_lo[arr == 2], 5)             # ... but never before the day began
    assert np.allclose(t_hi[arr == 2], 26)            # edit time: the map existed by the stamp, so "burned by" holds
    assert np.allclose(t_lo[arr == 3], 26 - 12)       # ... and "not burned at that map" moves earlier
    lo2, hi2 = IV.perimeter_bounds(TIMES, src, day_extra_h=[0, 8, 0, 0, 0, 0, 0])
    assert np.isclose(hi2[1], 5 + 32) and np.isclose(lo2[1], 5)      # date stored as 00:00 UTC at lon -120
    IV.MAPPED_SLACK_H = 1.0
    lo3, hi3 = IV.perimeter_bounds(TIMES, ["mapped"] * len(TIMES))
    assert np.allclose(hi3 - TIMES, 1) and np.allclose(TIMES - lo3, 1)


@pytest.mark.parametrize("seed", range(5))
def test_true_burn_time_stays_inside_when_stamps_are_wrong(seed, slack12):
    """True mapping times differ from the stamps the way each source allows: an edit stamp comes 0-12 h AFTER the
    real map, a bare date is any hour of its day. Arrival comes from the TRUE times, labels from the stamps."""
    rng = np.random.default_rng(seed)
    T, _ = known_fire()
    true = np.concatenate([[0.0], np.cumsum(rng.uniform(3, 30, 6))])
    src = ["mapped"] + list(rng.choice(["mapped", "edit", "date_only"], 6))
    stamp = true.copy()
    for k, s_ in enumerate(src):
        if s_ == "edit":
            stamp[k] = true[k] + rng.uniform(0, 12)
        elif s_ == "date_only":
            stamp[k] = true[k] - rng.uniform(0, 24)        # the stamp is the day's midnight, before the real map
    stamp[0], src[0] = 0.0, "mapped"
    arr = np.full(T.shape, -1, np.int64)
    for k in range(len(true) - 1, -1, -1):
        arr[T <= true[k]] = k
    for i in range(len(true) - 1):
        lo, hi = IV.perimeter_bounds(stamp - stamp[i], src)
        t_lo, t_hi, valid = IV.interval_labels(arr, i, lo, hi)
        # hours from the start as the model sees it. An edit-stamped START map is older than its stamp, so some cells
        # truly burned before t_i; the hazard starts at 0, so for them the right label is (0, t_hi]
        rel = np.maximum(T - stamp[i], 0.0)
        ev, cen = valid & np.isfinite(t_hi), valid & np.isinf(t_hi)
        assert (t_lo[ev] <= rel[ev] + 1e-6).all() and (rel[ev] <= t_hi[ev] + 1e-6).all(), (seed, i, src)
        assert (rel[cen] >= t_lo[cen] - 1e-6).all(), (seed, i, src)


def test_untrusted_burns_keep_only_not_burned_before():
    T, arr = known_fire()
    lo, hi = IV.perimeter_bounds(TIMES)
    bad = (arr == 2) | (arr == 1)
    t_lo, t_hi, valid = IV.interval_labels(arr, 0, lo, hi, bad=bad)
    assert np.isinf(t_hi[arr == 2]).all() and np.allclose(t_lo[arr == 2], 5) and valid[arr == 2].all()
    assert not valid[arr == 1].any()                  # "not burned by 0 h" says nothing: dropped
    ex = np.zeros_like(bad)
    ex[:10] = True
    assert not IV.interval_labels(arr, 0, lo, hi, exclude=ex)[2][:10].any()


# ---------------------------------------------------------------------------- on a Fire built from files
H = W = 64
STAMPS = ["2024-08-10 18:00", "2024-08-11 00:00", "2024-08-12 00:00", "2024-08-12 20:00", "2024-08-14 22:00"]   # 0, 6, 30, 50, 100 h


def make_fire(tmp, sources=None):
    d = tmp / "F1"
    d.mkdir()
    (d / "grid.json").write_text(json.dumps(dict(fire_id="F1", year=2024, height=H, width=W, res_m=150.0, name="t",
                                                 transform=[0.0017, 0, -120.0, 0, -0.00135, 40.0])))
    arrival = -np.ones((H, W), np.int16)
    for k, a in enumerate([28, 24, 20, 16, 12]):               # rings: perimeter k covers the square a..H-a
        ring = np.zeros((H, W), bool)
        ring[a:H - a, a:W - a] = True
        arrival[ring & ((arrival < 0) | (arrival > k))] = k
    arrival[56:62, 0:6] = 2                                    # a patch far from the fire with no satellite fire: untrusted
    times = pd.to_datetime(STAMPS, utc=True)
    extra = {"time_sources": np.array(sources)} if sources else {}
    np.savez(d / "labels.npz", arrival=arrival, other=np.zeros((H, W), bool), times=times.as_unit("ns").asi8,
             times_utc=np.array(True), **extra)
    z = np.zeros((H, W), np.float32)
    np.savez(d / "terrain.npz", elev=z + 1000, slope=z + 8, aspect=z + 180, tpi_500m=z, tpi_2km=z, rough=z)
    days = pd.date_range("2024-08-01", "2024-08-20")
    np.savez(d / "weather.npz", dates=np.array([str(x.date()) for x in days]), gm=np.full((len(days), len(D.GM), 2, 2), 10.0, np.float32),
             gm_vars=np.array(D.GM), gm_row=np.zeros(H, np.int16), gm_col=np.zeros(W, np.int16))
    af = np.zeros((len(days), H, W), np.uint8)
    af[9:15, 8:56, 8:56] = 1                                   # VIIRS fire over the main fire on 08-10..08-15
    np.savez(d / "viirs.npz", dates=np.array([str(x.date()) for x in days]), af_day=af,
             af_night=np.zeros((len(days), H, W), np.uint8), frp_max=np.zeros((len(days), H, W), np.float16))
    hours = pd.date_range("2024-08-06 00:00", "2024-08-18 21:00", freq="3h", tz="UTC")
    fields = np.zeros((len(hours), 4, 2, 2), np.float32)
    fields[:, 0], fields[:, 2], fields[:, 3] = 9.0, 2.0, 25.0
    fields[:, 1] = np.arange(len(hours))[:, None, None] % 8    # u changes every 3 h: each bin must get its own
    np.savez(d / "hrrr.npz", times=hours.as_unit("ns").asi8, fields=fields, idx=np.zeros((H, W), np.int32))
    return D.Fire(d)


@pytest.fixture
def clean():
    keep = {k: getattr(D, k) for k in ("FM_ON", "FUEL_V2", "TEMPORAL", "PHYSICS_V2", "WIND_TERRAIN", "HRRR_ROTATE")}
    D._CACHE.clear()
    D.HRRR_ROTATE = False
    yield
    for k, v in keep.items():
        setattr(D, k, v)
    D._CACHE.clear()


def test_fire_intervals_match_the_rule_and_the_old_label_check(tmp_path, clean):
    f = make_fire(tmp_path, ["mapped"] * len(STAMPS))
    win = (0, H, 0, W)
    t_lo, t_hi, valid = f.intervals(0, win, 72.0)
    arr = f.arrival
    assert np.allclose(t_lo[arr == 1], 0) and np.allclose(t_hi[arr == 1], 6)
    main2 = (arr == 2) & (np.arange(H)[:, None] < 50)
    assert np.allclose(t_lo[main2], 6) and np.allclose(t_hi[main2], 30)
    assert np.allclose(t_lo[arr == 3], 30) and np.allclose(t_hi[arr == 3], 50)
    assert np.allclose(t_lo[arr == 4], 50) and np.isinf(t_hi[arr == 4]).all()        # (50, 100] crosses 72 h
    assert np.allclose(t_lo[arr < 0], 50) and np.isinf(t_hi[arr < 0]).all()          # censored at the same 50 h
    far = np.zeros((H, W), bool)
    far[56:62, 0:6] = True
    assert np.isinf(t_hi[far]).all() and np.allclose(t_lo[far], 6) and valid[far].all()   # untrusted: censored at 6 h
    # the label-side evidence is the one sample() uses
    for j in range(len(STAMPS) - 1):
        f.sample(j, 24.0, win)
        assert (f.label_evidence(j) == f._evidence).all(), j


def test_weather_window_does_not_change_the_old_target(tmp_path, clean):
    f = make_fire(tmp_path)
    win = (0, H, 0, W)
    _, _, y0, m0, b0 = f.sample(1, 24.0, win)
    feats, _, y1, m1, b1 = f.sample(1, 72.0, win, wx_h=72.0)
    assert (y0 == y1).all() and (m0 == m1).all() and (b0 == b1).all()
    # HRRR mean u over [t_i - 1 h, t_i + 72 h], not over the 24 h to the next perimeter
    ts = pd.date_range("2024-08-06 00:00", "2024-08-18 21:00", freq="3h", tz="UTC")
    t_i = pd.Timestamp(STAMPS[1], tz="UTC")
    sel = (ts >= t_i - pd.Timedelta(hours=1)) & (ts <= t_i + pd.Timedelta(hours=72))
    assert np.isclose(feats["hrrr"][0].mean() * 8, (np.arange(len(ts)) % 8)[sel].mean(), atol=1e-4)


def test_bin_weather_gives_each_bin_its_own_hours(tmp_path, clean):
    f = make_fire(tmp_path)
    wx = f.bin_weather(0, (0, 8, 0, 8))
    ts = pd.date_range("2024-08-06 00:00", "2024-08-18 21:00", freq="3h", tz="UTC")
    t0 = pd.Timestamp(STAMPS[0], tz="UTC")
    assert wx.shape == (24, D.Fire.BIN_CH, 8, 8) and wx[:, 7].all()
    for b in range(24):
        k = ts.get_loc(t0 + pd.Timedelta(hours=3 * b))
        assert np.isclose(wx[b, 0, 0, 0] * 8, k % 8), b
        hour = (18 + 3 * b + 1.5 - 8) % 24                            # local solar hour at lon -120 (UTC - 8 h)
        assert np.isclose(wx[b, 5, 0, 0], np.sin(2 * np.pi * hour / 24), atol=1e-5)


def test_fire_without_time_sources_is_widened_as_unknown(tmp_path, clean):
    f = make_fire(tmp_path)                       # no time_sources in labels.npz, no catalog: every stamp "unknown"
    t_lo, t_hi, _ = f.intervals(0, (0, H, 0, W), 72.0)
    assert set(f.time_src) == {"unknown"}
    assert np.allclose(t_hi[f.arrival == 1], 6 + IV.UNKNOWN_AFTER_H)
    assert np.allclose(t_lo[f.arrival == 3], max(30 - IV.EDIT_SLACK_H, 0))
