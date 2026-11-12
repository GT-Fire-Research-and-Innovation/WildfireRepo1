"""Hourly, slope-aware fine fuel moisture (model/fuel_moisture.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import fuel_moisture as FM  # noqa: E402

LAT, LON = 40.0, -120.0
T0 = pd.Timestamp("2021-08-10 00:00", tz="UTC")      # 16:00 local the day before


def day(rain_at=None, hours=96):
    """Hot dry afternoons (35 C, 12 %), cool damp nights (12 C, 70 %), every 3 h."""
    steps = []
    for k in range(0, hours + 1, 3):
        t = T0 - pd.Timedelta(hours=72) + pd.Timedelta(hours=k)
        local = (t.hour + LON / 15) % 24
        warm = 0.5 * (1 + np.cos((local - 15) / 24 * 2 * np.pi))
        steps.append(dict(t=t, temp_c=12 + 23 * warm, rh=70 - 58 * warm, wind_ms=3.0,
                          rain_mm=8.0 if rain_at is not None and k == rain_at else 0.0, ghi=None))
    return steps


def flat(n=1):
    z = np.zeros((n, n), np.float32)
    return dict(elev_dz=z, slope_deg=z, aspect_deg=z, canopy_cover=z)


def test_emc_matches_published_values():
    assert abs(float(FM.emc(70.0, 5.0)) - 1.24) < 0.05      # hot desert air: about 1 %
    assert abs(float(FM.emc(70.0, 30.0)) - 6.0) < 0.2
    assert abs(float(FM.emc(70.0, 90.0)) - 20.4) < 0.5


def test_night_recovery_and_afternoon_drying():
    out, fm = FM.integrate(day(), T0, T0 + pd.Timedelta(hours=24), LAT, LON, **flat())
    m_min, m_max = out[1, 0, 0] * 30, out[3, 0, 0] * 30
    assert m_min < 5 and m_max > 10                          # dry afternoon, damp night
    assert out[14].all() and out[11, 0, 0] == 0              # mask on, no rain


def test_south_slope_dries_more_than_north_and_shade_helps():
    g = dict(elev_dz=np.zeros((1, 3), np.float32), slope_deg=np.array([[30, 30, 30]], np.float32),
             aspect_deg=np.array([[180, 0, 180]], np.float32), canopy_cover=np.array([[0, 0, 90]], np.float32))
    out, _ = FM.integrate(day(), T0, T0 + pd.Timedelta(hours=24), LAT, LON, **g)
    south, north, shaded = out[1, 0] * 30
    assert south < north and south < shaded


def test_higher_ground_is_cooler_and_moister():
    g = dict(elev_dz=np.array([[-500, 0, 500]], np.float32), slope_deg=np.zeros((1, 3), np.float32),
             aspect_deg=np.zeros((1, 3), np.float32), canopy_cover=np.zeros((1, 3), np.float32))
    out, _ = FM.integrate(day(), T0, T0 + pd.Timedelta(hours=24), LAT, LON, **g)
    assert out[2, 0, 0] < out[2, 0, 1] < out[2, 0, 2]


def test_rain_wets_fuels_and_is_recorded():
    dry, _ = FM.integrate(day(), T0, T0 + pd.Timedelta(hours=24), LAT, LON, **flat())
    wet, _ = FM.integrate(day(rain_at=69), T0, T0 + pd.Timedelta(hours=24), LAT, LON, **flat())   # 3 h before the start
    assert wet[0, 0, 0] > dry[0, 0, 0] and wet[4, 0, 0] > dry[4, 0, 0]       # 1-h and 10-h at the start
    assert wet[10, 0, 0] > 0 and wet[12, 0, 0] < dry[12, 0, 0]                # rain before, hours since rain
    late, _ = FM.integrate(day(rain_at=84), T0, T0 + pd.Timedelta(hours=24), LAT, LON, **flat())  # during the window
    assert late[11, 0, 0] > 0 and late[0, 0, 0] == dry[0, 0, 0]               # the start state does not see it


def test_nothing_after_the_window_is_used():
    a, _ = FM.integrate(day(hours=96), T0, T0 + pd.Timedelta(hours=24), LAT, LON, **flat())
    b, _ = FM.integrate(day(rain_at=120, hours=144), T0, T0 + pd.Timedelta(hours=24), LAT, LON, **flat())
    assert np.allclose(a, b)
