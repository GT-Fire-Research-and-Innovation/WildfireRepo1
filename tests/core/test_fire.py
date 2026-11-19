"""A fire's identity, region and days: the region reaches 32 km past the perimeters' box at any latitude, the
window moves every time-dependent day list, and dirty IRWIN keys still match."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fah.core.fire import Fire
from fah.core.geometry import BoundingBox


def km_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance, km."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def test_days_cover_the_window_and_the_firms_span(camp):
    days = camp.firms_days()
    assert (days[0].date().isoformat(), days[-1].date().isoformat()) == ("2018-10-26", "2018-12-09")
    g = camp.days()  # gridmet keeps the build's days without activity
    assert (g[0].date().isoformat(), g[-1].date().isoformat()) == ("2018-10-26", "2018-11-27")
    ana, runs = camp.span().hours(3), camp.span().daily(12)  # perimeters 11-09 .. 11-25 -> 11-08 00:00 .. 11-26 23:00
    assert (len(ana), len(runs)) == (152, 19)
    assert ana[0] == pd.Timestamp("2018-11-08 00:00", tz="UTC") and ana[-1] == pd.Timestamp(
        "2018-11-26 21:00", tz="UTC"
    )


def test_dirty_irwin_keys_still_match_records(camp):
    assert Fire.normalise_irwin("D6024AFC-A330-404F-ACF1-7A21986EEFEA}\r\n") == "D6024AFC-A330-404F-ACF1-7A21986EEFEA"
    assert Fire.normalise_irwin("{75e64db8-9b75-4a68-bedd-67cc62658e38}") == camp.incident
    assert camp.incident_key() == (camp.incident, None)
    assert camp.replace(incident="name:dixie|CA-PNF-001").incident_key() == (None, ("dixie", "CA-PNF-001"))


def test_extent_row_moves_the_window(camp_perimeters):
    f = camp_perimeters.with_extent({"start_day": "2018-11-05", "end_day": "2018-11-28"})
    t0, t1 = f.span()
    assert (t0, t1) == (pd.Timestamp("2018-11-05", tz="UTC"), pd.Timestamp("2018-11-28 23:00", tz="UTC"))
    d = f.days()  # gridmet: 14 / 2 days around, moved out with the window
    assert (d[0], d[-1]) == (pd.Timestamp("2018-10-23"), pd.Timestamp("2018-11-29"))
    fd = f.firms_days()  # firms ignores the window: it is the window's input
    assert (fd[0], fd[-1]) == (pd.Timestamp("2018-10-26"), pd.Timestamp("2018-12-09"))
    assert f.with_extent(None) is f and f.ignition is None


@pytest.mark.parametrize("lat", [25.0, 39.7, 48.5, 64.0])
def test_region_reaches_32_km_past_the_perimeters_on_every_side(camp_perimeters, lat):
    pb = BoundingBox(-121.7, lat - 0.05, -121.6, lat + 0.05)  # a 9 x 11 km fire
    f = camp_perimeters.replace(perimeter_box=pb, build_box=pb.buffered(9.6))
    w, s, e, n = f.region()
    assert km_between(s, -121.65, pb[1], -121.65) >= 32 and km_between(pb[3], -121.65, n, -121.65) >= 32
    for y in (s, pb[1], pb[3], n):  # east-west margin holds at every latitude of the box
        assert km_between(y, w, y, pb[0]) >= 32 and km_between(y, pb[2], y, e) >= 32
    assert km_between(s, w, s, e) >= 64 and km_between(s, w, n, w) >= 64
    assert w <= f.build_box[0] and n >= f.build_box[3]  # never smaller than the build's box
    point = BoundingBox(-121.6, lat, -121.6, lat)
    tw, ts, te, tn = f.replace(perimeter_box=point, build_box=point).region()  # a 64 km square (+ the 1 km margin)
    assert 64 <= km_between(ts, tw, tn, tw) <= 67 and 64 <= km_between(ts, tw, ts, te) <= 67
