"""The fire's extent from FIRMS: activity widens the window only when it chains onto the fire, low-confidence and
static detections do not count, and the ignition point is the earliest observation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fah.core.extent import FireExtentEstimator as X


def day(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def detections(rows: list[tuple]) -> pd.DataFrame:
    """FIRMS-like frame from (date, n, lat, lon, confidence, type[, acq_time HHMM])."""
    out = []
    for d, n, lat, lon, conf, typ, *hhmm in rows:
        out += [
            dict(
                latitude=lat,
                longitude=lon,
                acq_date=d,
                acq_time=hhmm[0] if hhmm else 2100,
                confidence=conf,
                type=typ,
                bright_ti4=330.0,
            )
        ] * n
    return pd.DataFrame(out)


def test_activity_chains_onto_the_perimeter_days_across_short_gaps():
    p0, p1 = day("2018-11-10"), day("2018-11-15")
    act = [day(d) for d in ("2018-11-01", "2018-11-05", "2018-11-07", "2018-11-08", "2018-11-12", "2018-11-16",
                            "2018-11-20", "2018-11-30")]  # fmt: skip
    # back: 11-08 (2 days before), 11-07, 11-05 (2 quiet days) join; 11-01 is 4 days off 11-05: another fire
    # forward: 11-16 joins; 11-20 is 4 days past it
    assert X.activity_window(p0, p1, act) == (day("2018-11-05"), day("2018-11-16"))
    assert X.activity_window(p0, p1, []) == (p0, p1)


def test_active_days_ignore_low_confidence_static_sources_and_neighbours(camp_perimeters):
    lat, lon = 39.7, -121.6
    df = detections([
        ("2018-11-05", 3, lat, lon, "n", 0),  # active
        ("2018-11-06", 1, lat, lon, "h", 0),  # one detection: not enough
        ("2018-11-07", 5, lat, lon, "l", 0),  # low confidence
        ("2018-11-08", 5, lat, lon, "h", 2),  # static land source
        ("2018-11-09", 5, 39.0, lon, "h", 0),  # outside the perimeters' box
        ("2018-11-10", 2, 39.9, -121.35, "nominal", 0),  # on the box edge (pad)
    ])  # fmt: skip
    box = camp_perimeters.perimeter_box
    assert X.active_days(df, box) == [day("2018-11-05"), day("2018-11-10")]
    assert X.active_days(None, box) == []
    nrt = df.drop(columns=["type"])  # NRT files have no type column
    assert day("2018-11-08") in X.active_days(nrt, box)


def test_extent_row_records_the_window_and_how_it_was_found(camp_perimeters):
    lat, lon = 39.7, -121.6
    df = detections([("2018-11-06", 4, lat, lon, "n", 0), ("2018-11-07", 9, lat, lon, "h", 0),
                     ("2018-11-27", 3, lat, lon, "n", 0)])  # fmt: skip
    row = X.row(camp_perimeters, df)
    assert (row["start_day"], row["end_day"]) == ("2018-11-05", "2018-11-28")
    assert (row["active_days_before"], row["active_days_after"], row["window_rule"]) == (3, 2, "perimeters+firms")
    plain = X.row(camp_perimeters)
    assert plain["start_day"] == "2018-11-08" and plain["window_rule"] == "perimeters"
    assert camp_perimeters.with_extent(row).span().start == pd.Timestamp("2018-11-05", tz="UTC")


def test_ignition_is_the_earliest_observation_of_the_fire(camp_perimeters):
    lat, lon = 39.7, -121.6
    df = detections([
        ("2018-11-07", 3, lat, lon, "h", 0, 954),  # first overpass that day
        ("2018-11-07", 2, lat + 0.02, lon, "n", 0, 1003),  # same overpass (9 min later)
        ("2018-11-07", 4, lat + 0.1, lon + 0.1, "h", 0, 2112),  # next overpass: grown
        ("2018-11-08", 9, lat + 0.1, lon + 0.1, "h", 0, 2050),
        ("2018-10-20", 9, lat, lon, "h", 0, 2000),  # weeks earlier: not chained to this fire
    ])  # fmt: skip
    act = X.active_days(df, camp_perimeters.perimeter_box)
    y, x, t, how = X.ignition(camp_perimeters, df, act)
    assert how == "firms" and t == pd.Timestamp("2018-11-07 09:54", tz="UTC")
    assert y == pytest.approx(lat + 0.008) and x == pytest.approx(lon)
    perim_first = (pd.Timestamp("2018-11-07 06:00", tz="UTC"), 39.71, -121.61)  # mapped before that overpass
    assert X.ignition(camp_perimeters, df, act, perim_first).method == "first_perimeter"
    single = detections([("2018-11-10", 1, lat, lon, "h", 0, 2030)])  # never an active day
    assert X.ignition(camp_perimeters, single, []).method == "firms_single"
    y, x, t, how = X.ignition(camp_perimeters)
    assert how == "perimeter_box_centre" and y == pytest.approx(39.7482, abs=1e-3)


def test_first_perimeter_point_is_the_centre_of_the_earliest_arrival():
    arrival = np.full((10, 10), -1, "i2")
    arrival[2:4, 6:8] = 0
    arrival[2:9, 1:9] = np.where(arrival[2:9, 1:9] == 0, 0, 1)
    times = np.array([pd.Timestamp("2018-11-09", tz="UTC").value, pd.Timestamp("2018-11-10", tz="UTC").value])
    t, y, x = X.first_perimeter_point(arrival, [0.01, 0.0, -122.0, 0.0, -0.01, 40.0], times)
    assert t == pd.Timestamp("2018-11-09", tz="UTC")  # cells 2-3 x 6-7: centre 3.0, 7.0
    assert y == pytest.approx(40.0 - 0.03) and x == pytest.approx(-122.0 + 0.07)
    assert X.first_perimeter_point(np.full((3, 3), -1), [0.01, 0, -122, 0, -0.01, 40], times) is None
