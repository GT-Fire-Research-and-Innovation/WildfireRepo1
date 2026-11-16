"""Catalog rules v2: same-day perimeters are kept only when they were mapped separately and far enough apart,
auto-generated circles are left out, and the daily perimeters the first rules kept are still kept."""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import nifc_perimeters as N  # noqa: E402


def rec(t, size, method="IR Image Interpretation", tsrc="mapped", inc="A", gid=None):
    """One perimeter record: a square `size` degrees wide near lon -120 (local time = UTC - 8 h)."""
    return dict(source="nifs", incident_name=inc, irwin_id=inc, unique_fire_id=inc, category="Wildfire Daily Fire Perimeter",
                map_method=method, comments="", perim_time=pd.Timestamp(t, tz="UTC"), time_source=tsrc, version_time=pd.Timestamp(t, tz="UTC"),
                geometry_id=gid or f"{inc}-{t}", deleted=False, geometry=box(-120, 40, -120 + size, 40 + size))


def frame(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


# burn day = local date after a 6 h shift: at lon -120, 14:00 UTC is 06:00 local, the start of a burn day
ROWS = [
    rec("2024-08-10 18:00", 0.010),                                   # day 1, 10:00 local
    rec("2024-08-11 02:00", 0.020),                                   # day 1, 18:00 local: the day's pick
    rec("2024-08-11 20:00", 0.022),                                   # day 2, noon
    rec("2024-08-11 21:30", 0.024),                                   # day 2, 1.5 h later: too close to its neighbours
    rec("2024-08-12 03:00", 0.030),                                   # day 2, evening: the day's pick
    rec("2024-08-12 20:00", 0.031, tsrc="edit"),                      # day 3, an edit-time copy: not a separate mapping
    rec("2024-08-13 04:00", 0.040),                                   # day 3: the day's pick
    rec("2024-08-10 16:00", 0.050, method="Auto-generated"),          # a circle of the reported size
]


def times(g):
    return [t.strftime("%m-%d %H:%M") for t in g.sort_values("perim_time")["perim_time"]]


def test_daily_rule_is_unchanged_by_default():
    g, _ = N.clean(frame(ROWS))
    assert times(g) == ["08-11 02:00", "08-12 03:00", "08-13 04:00"]      # one per burn day: the latest


def test_same_day_perimeters_kept_when_mapped_separately_and_three_hours_apart():
    g, drops = N.clean(frame(ROWS), drop_methods=N.DROP_MAP_METHODS | N.AUTO_MAP_METHODS, sameday_h=3.0)
    assert times(g) == ["08-10 18:00", "08-11 02:00", "08-11 20:00", "08-12 03:00", "08-13 04:00"]
    assert drops["map method left out"] == 1                          # the circle
    daily, _ = N.clean(frame(ROWS), drop_methods=N.DROP_MAP_METHODS | N.AUTO_MAP_METHODS)
    assert set(times(daily)) <= set(times(g))                         # every daily perimeter is still there


def test_auto_generated_fire_disappears():
    rows = [rec(f"2024-08-{d} 02:00", 0.01 * k, method="Auto-generated for InFORM", inc="C") for k, d in enumerate((11, 12, 13), 1)]
    g, _ = N.clean(frame(rows), drop_methods=N.DROP_MAP_METHODS | N.AUTO_MAP_METHODS, sameday_h=3.0)
    assert g.empty
    g, _ = N.clean(frame(rows))
    assert len(g) == 3                                                # the first rules kept it


def test_shrinking_same_day_copy_is_dropped():
    rows = [rec("2024-08-10 18:00", 0.030), rec("2024-08-10 23:00", 0.010), rec("2024-08-11 03:00", 0.032)]
    g, _ = N.clean(frame(rows), sameday_h=3.0)
    assert times(g) == ["08-10 18:00", "08-11 03:00"]
