"""Inputs must never use satellite passes made after the forecast start (the 2026-11-11 time-leak fix)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import data as D  # noqa: E402


def fire(pass_ends, lon=-120.0):
    f = object.__new__(D.Fire)
    f.g = {"transform": [0.0016, 0.0, lon, 0.0, -0.0013, 40.0]}
    f._pass_ends = pass_ends
    return f


def test_same_day_afternoon_pass_after_start_is_excluded():
    # local solar day 2025-08-10 at lon -120 (UTC-8): night pass 01:30 local = 09:30 UTC, day pass 13:30 local = 21:30 UTC
    ends = {("2025-08-10", "N"): pd.Timestamp("2025-08-10 09:40"), ("2025-08-10", "D"): pd.Timestamp("2025-08-10 21:40")}
    t_i = pd.Timestamp("2025-08-10 10:00", tz="UTC")          # 02:00 local, after the night pass, before the day pass
    use_day, use_night = fire(ends).passes_before(pd.Timestamp("2025-08-10"), t_i)
    assert use_night and not use_day


def test_previous_day_is_fully_available():
    ends = {("2025-08-09", "N"): pd.Timestamp("2025-08-09 09:40"), ("2025-08-09", "D"): pd.Timestamp("2025-08-09 21:40")}
    t_i = pd.Timestamp("2025-08-10 10:00", tz="UTC")
    assert fire(ends).passes_before(pd.Timestamp("2025-08-09"), t_i) == (True, True)


def test_pass_ending_exactly_at_start_is_excluded():
    ends = {("2025-08-10", "D"): pd.Timestamp("2025-08-10 21:40")}
    t_i = pd.Timestamp("2025-08-10 21:40", tz="UTC")
    assert fire(ends).passes_before(pd.Timestamp("2025-08-10"), t_i)[0] is False


def test_without_detection_times_the_start_day_is_left_out():
    t_i = pd.Timestamp("2025-08-10 23:00", tz="UTC")          # 15:00 local on 2025-08-10
    f = fire(None)
    f.pass_ends = lambda: None
    assert f.passes_before(pd.Timestamp("2025-08-10"), t_i) == (False, False)
    assert f.passes_before(pd.Timestamp("2025-08-09"), t_i) == (True, True)


def test_goes_pixel_waits_for_its_companion():
    import numpy as np
    t = pd.to_datetime(["2025-08-10 09:00", "2025-08-10 11:00", "2025-08-10 09:00"], utc=True)
    gz = {"t": np.array(t.as_unit("ns").asi8), "lon": np.array([-120.0, -120.01, -121.0]), "lat": np.array([40.0, 40.0, 40.0])}
    f = object.__new__(D.Fire)
    comp = f.goes_companion_t(gz)
    t_i = pd.Timestamp("2025-08-10 10:00", tz="UTC").value
    assert comp[0] > t_i                 # its only companion (1 km away) appears at 11:00, after the forecast start
    assert comp[0] == t[1].value and comp[1] == t[1].value
    assert comp[2] == np.iinfo(np.int64).max     # isolated forever


def _supp_line(saved_utc, created_utc=None):
    ms = lambda s: int(pd.Timestamp(s, tz="UTC").value // 10 ** 6)  # noqa: E731
    props = {"FeatureCategory": "Completed Dozer Line", "IncidentName": "Camp", "GDB_FROM_DATE": ms(saved_utc),
             "CreateDate": ms(created_utc or saved_utc), "DateCurrent": ms(saved_utc), "GDB_TO_DATE": None}
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "LineString", "coordinates": [[-120.0, 39.985], [-119.96, 39.985]]}}


def test_suppression_is_shown_from_the_burn_day_after_it_was_saved():
    """Day d's suppression map holds only versions saved by 06:00 local on d+1, and the loader gives a forecast
    the previous burn day's map (tests/test_loader.py), so a line drawn on a burn day first appears the day after."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    import suppression_layers as SL
    fire = {"height": 4, "width": 4, "transform": [0.01, 0, -120.0, 0, -0.01, 40.0],
            "bounds_lonlat": [-120.0, 39.96, -119.96, 40.0]}
    days = ["2019-08-09", "2019-08-10"]
    dozer = SL.CHANNEL_NAMES.index("dozer_line")
    saved = SL.archive_features([_supp_line("2019-08-10 23:00")])      # 15:00 local: burn day 08-10
    lay = SL.rasterize_layers(saved, fire, days)
    assert lay[0].sum() == 0 and lay[1, dozer].any()
    edited = SL.archive_features([_supp_line("2019-08-10 23:00", created_utc="2019-08-01 00:00")])
    assert edited.t_from.iloc[0] == pd.Timestamp("2019-08-10 23:00", tz="UTC")   # an edit counts from the edit
    stamped_late = SL.archive_features([_supp_line("2019-08-09 12:00", created_utc="2019-08-10 23:00")])
    assert stamped_late.t_from.iloc[0] == pd.Timestamp("2019-08-10 23:00", tz="UTC")  # the latest stamp wins
    assert SL.rasterize_layers(stamped_late, fire, days)[0].sum() == 0
