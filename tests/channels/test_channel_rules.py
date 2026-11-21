"""Channel facts, offline: which fires each source covers, the times each channel fetches, the window and region a
manifest records, and the files a fire needs to count as done."""

from __future__ import annotations

import pandas as pd

from fah.channels.firms import FirmsChannel
from fah.channels.gfs import GfsChannel
from fah.channels.goes import GoesChannel
from fah.channels.hrrr_next import HrrrNextChannel
from fah.channels.landfire import LandfireChannel
from fah.channels.lulc import LulcChannel


def at(fire, first: str, last: str, year: int):
    return fire.replace(year=year, first=f"{first} 00:00+00:00", last=f"{last} 00:00+00:00")


def test_channels_skip_fires_without_data(camp, channels):
    old, new = at(camp, "2014-08-01", "2014-08-20", 2014), at(camp, "2021-07-14", "2021-08-20", 2021)
    assert all(c.skip_reason(camp) is None for c in channels)
    assert {c.name for c in channels if c.skip_reason(old)} == {
        "suppression",
        "landfire",
        "tessera",
        "goes",
        "lulc",
        "gfs",
    }
    assert {c.name for c in channels if c.skip_reason(new)} == {"ics209"}
    assert channels.get("hrrr").skip_reason(at(camp, "2014-06-01", "2014-06-20", 2014))
    assert channels.get("hrrr_next").skip_reason(at(camp, "2014-06-01", "2014-06-20", 2014))
    assert LandfireChannel.version(2017) == "LF2016" and LandfireChannel.version(2024) == "LF2023"
    assert LandfireChannel.version(2016) is None
    assert [LulcChannel.map_year(y) for y in (2017, 2018, 2021, 2025)] == [2017, 2017, 2020, 2023]


def test_firms_calls_cover_two_weeks_around_the_perimeters_in_5_day_steps(camp):
    calls, days = FirmsChannel.calls(camp), camp.firms_days()
    for src in ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"):
        mine = [(d, k) for s, d, k in calls if s == src]
        assert mine[0] == ("2018-10-26", 5) and mine[-1] == ("2018-12-05", 5)
        assert sum(k for _, k in mine) == len(days)
    assert {s for s, _, _ in FirmsChannel.calls(camp.replace(year=2016))} == {"VIIRS_SNPP_SP"}  # NOAA-20 from 2018


def test_goes_hours_follow_the_satellites_in_service(camp):
    hours = GoesChannel.hours(camp)
    assert len(hours) == 456
    g17 = [t for t, sats in hours if any(s == "G17" for s, _ in sats)]
    assert g17[0] == pd.Timestamp("2018-11-15 00:00", tz="UTC") and len(g17) == 288


def test_hrrr_next_runs_reach_the_next_afternoon_as_far_as_hrrr_ran(camp):
    t = HrrrNextChannel.times(camp)
    runs = sorted({r for r, _ in t})
    assert len(runs) == 19 and len(t) == 19 * 12  # HRRRv3: 3..36 h every 3 h
    assert runs[0] == pd.Timestamp("2018-11-08 12:00", tz="UTC") and max(h for _, h in t) == 36
    v1 = HrrrNextChannel.times(at(camp, "2016-07-07", "2016-07-08", 2016))
    v2 = HrrrNextChannel.times(at(camp, "2017-07-07", "2017-07-08", 2017))
    assert {h for _, h in v1} == {3, 6, 9, 12, 15} and {h for _, h in v2} == {3, 6, 9, 12, 15, 18}


def test_gfs_leads_follow_what_the_collection_holds(camp, channels):
    t = GfsChannel.times(camp)
    runs = sorted({r for r, _ in t})
    assert len(runs) == 19 and len(t) == 19 * 80 and max(h for _, h in t) == 240
    early = GfsChannel.times(at(camp, "2016-07-07", "2016-07-08", 2016))
    assert {h for _, h in early} == set(range(3, 37, 3))
    before = at(camp, "2015-06-01", "2015-06-10", 2015)
    assert GfsChannel.times(before) == [] and channels.get("gfs").skip_reason(before)
    assert channels.get("gfs").skip_reason(camp) is None


def test_the_window_moves_every_time_dependent_channel(camp_perimeters, channels):
    f = camp_perimeters.with_extent({"start_day": "2018-11-05", "end_day": "2018-11-28"})
    t0 = f.span().start
    ana, runs = f.span().hours(3), f.span().daily(12)
    assert ana[0] == t0 and runs[-1] == pd.Timestamp("2018-11-28 12:00", tz="UTC")
    assert GoesChannel.hours(f)[0][0] == t0 and GfsChannel.times(f)[0][0] == pd.Timestamp("2018-11-05 12:00", tz="UTC")
    assert (
        channels.get("goes").window_label(f) == "2018-11-05/2018-11-28"
        and channels.get("hrrr").window_label(f) == "2018-11-02/2018-11-28"  # 3 days of spin-up
        and channels.get("terrain").window_label(f) == ""
    )
    assert channels.get("firms").window_label(f) == "2018-10-26/2018-12-09"  # the window's input: fixed days
    assert channels.get("gridmet").window_label(f) == "2018-10-23/2018-11-29"
    gfs = channels.get("gfs")
    assert gfs.labels(f) == ("2018-11-05/2018-11-28", f.region().label()) and gfs.region_label(f) != ""
    assert channels.get("perimeters").region_label(f) == channels.get("ics209").region_label(f) == ""


def test_new_layout_names_are_required_for_a_fire_to_count_as_done(channels):
    assert channels.get("perimeters").expected_files == {"records.geojson.gz"}
    assert channels.get("viirs").expected_files == {"viirs.nc"}
    assert channels.get("suppression").expected_files == {"event_lines.geojson.gz", "event_polygons.geojson.gz"}
    assert channels.get("terrain").expected_files is None
    assert len(channels.get("gridmet").expected_files) == 13
