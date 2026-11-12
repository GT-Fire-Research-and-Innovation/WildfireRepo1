"""The live forecast's starting fire: reported size for fires without a perimeter, perimeter plus only
newer detections for fires with one."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import forecast_now as FN  # noqa: E402

G = dict(height=100, width=100, res_m=150.0, transform=[0.0017, 0, -120.0, 0, -0.00135, 40.0],
         bounds_lonlat=[-120.0, 39.865, -119.83, 40.0])
CELL_AC = 150 ** 2 / 4046.856
NOW = pd.Timestamp("2026-08-01 20:00", tz="UTC")


def pts(rows):
    return pd.DataFrame([dict(latitude=la, longitude=lo, acq_date=t.strftime("%Y-%m-%d"), acq_time=int(t.strftime("%H%M")),
                              daynight="D", t=t) for lo, la, t in rows])


def lab(core=None):
    a = -np.ones((100, 100), np.int16)
    if core is not None:
        a[core] = 0
    return dict(arrival=a, other=np.zeros((100, 100), bool))


def test_incident_uses_reported_size(monkeypatch):
    monkeypatch.setattr(FN, "_points", lambda d, g, now: pts([(-119.915, 39.932, NOW - pd.Timedelta(hours=5))]))
    fire = dict(source="incident", acres=70, lon=-119.915, lat=39.932)
    arr, how, nd = FN.start_fire(".", G, fire, lab(), NOW, NOW)
    ac = (arr >= 0).sum() * CELL_AC
    assert abs(ac - 70) < CELL_AC and nd == 1


def test_incident_detections_beat_stale_report(monkeypatch):
    rows = [(-119.915 + 0.0017 * k, 39.932, NOW - pd.Timedelta(hours=3)) for k in range(12)]
    monkeypatch.setattr(FN, "_points", lambda d, g, now: pts(rows))
    fire = dict(source="incident", acres=1, lon=-119.915, lat=39.932)
    arr, how, nd = FN.start_fire(".", G, fire, lab(), NOW, NOW)
    assert (arr >= 0).sum() >= 12 and "more than the report" in how


def test_perimeter_adds_only_newer_nearby_detections(monkeypatch):
    t_per = NOW - pd.Timedelta(hours=10)
    core = (slice(40, 50), slice(40, 50))
    x = lambda c: -120.0 + 0.0017 * (c + 0.5)
    y = lambda r: 40.0 - 0.00135 * (r + 0.5)
    rows = [(x(52), y(45), NOW - pd.Timedelta(hours=2)),     # newer, next to the fire: added
            (x(53), y(45), t_per - pd.Timedelta(hours=1)),   # older than the perimeter: ignored
            (x(95), y(95), NOW - pd.Timedelta(hours=2))]     # newer but ~7 km away: ignored
    monkeypatch.setattr(FN, "_points", lambda d, g, now: pts(rows))
    arr, how, nd = FN.start_fire(".", G, dict(source="wfigs"), lab(core), NOW, t_per)
    assert nd == 1 and (arr >= 0).sum() == 101 and arr[45, 52] == 0
