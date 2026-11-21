"""Incidents: the old fire ids come back verbatim, each archive's clock is read by its own rule, and every gate gives
its reason."""

from __future__ import annotations

import pandas as pd
import pytest

from fah.core.geometry import BoundingBox
from fah.core.incidents import ArchiveRules, CatalogGates, GeomacRules, IncidentKey, NifsRules


def ms(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").value // 1_000_000)


@pytest.mark.parametrize(
    "irwin, fire_id",
    [
        ("{75e64db8-9b75-4a68-bedd-67cc62658e38}", "F2003403265"),  # Camp
        ("03DB9419-11F5-47AC-B7A1-A00C1E53C708", "F3dafe466b9"),  # Lynx Meadow
        ("{2E19CEF9-FDD2-49BB-A3DA-6F3969388F9C}\xa0\xa0\xa0", "Fad75fdf1d7"),  # La Bonte: junk kept in the key
        ("{AD9B0ACE-5F92-485B-88E4-E4C7134836BB}\n", "F5f374cc29c"),  # Quartz Ridge
    ],
)
def test_incident_key_reproduces_the_old_fire_ids(irwin, fire_id):
    key = IncidentKey.of(irwin, "any name", "CA-XXX-1")
    assert IncidentKey.fire_id(key) == fire_id
    assert IncidentKey.irwin_uuid(key) == irwin.strip("{}\n\xa0 ").upper()  # junk-free for joins


def test_incident_without_irwin_is_keyed_by_name_and_unique_id():
    key = IncidentKey.of(None, "  Big   Knife ", "2016-MTLNF-000123")
    assert key == "name:big knife|2016-MTLNF-000123" and IncidentKey.irwin_uuid(key) is None
    assert IncidentKey.of(float("nan"), "X", None) == "name:x|"


def test_record_time_takes_the_first_plausible_field():
    rec = pd.DataFrame({"polygondatetime": [ms("2022-08-01 10:00"), 0, None],
                        "datecurrent": [None, ms("2022-08-03 00:00"), "2022-08-04T05:00:00Z"]})  # fmt: skip
    t = NifsRules().record_time(rec)
    assert list(t) == [pd.Timestamp(x, tz="UTC") for x in ("2022-08-01 10:00", "2022-08-03", "2022-08-04 05:00")]


def test_geomac_times_are_local_clock_stored_as_utc():
    stored = ms("2018-08-01 14:00")  # written as 14:00 on the fire's own clock
    rec = pd.DataFrame({"perimeterdatetime": [stored, stored, stored], "_cx": [-120.0, -105.0, -120.0],
                        "_cy": [38.0, 40.0, 38.0], "_year": [2018, 2018, 2019]})  # fmt: skip
    t = GeomacRules().record_time(rec)
    assert t[0] == pd.Timestamp("2018-08-01 21:00", tz="UTC")  # California, PDT
    assert t[1] == pd.Timestamp("2018-08-01 20:00", tz="UTC")  # Colorado, MDT
    assert t[2] == pd.Timestamp("2018-08-01 20:00", tz="UTC")  # 2019: Denver time for every fire
    assert GeomacRules.zone(-112.0, 33.5, 2017) == "America/Phoenix" and GeomacRules.zone(-80, 27, 2016).endswith(
        "York"
    )


def test_nifs_category_delete_flag_and_uncounted_methods():
    rec = pd.DataFrame({"featurecategory": ["Wildfire Daily Fire Perimeter", "Prescribed Fire", None],
                        "deletethis": ["No", "Yes", None],
                        "mapmethod": ["Infrared Image", "Hand Sketch", "Modeled"]})  # fmt: skip
    rules = NifsRules()
    assert list(rules.is_wildfire(rec)) == [True, False, False]
    assert list(rules.deleted(rec)) == [False, True, False]
    assert list(rules.counted(rec)) == [True, False, False]


def test_night_flights_count_for_the_day_that_burned():
    t = pd.Series(pd.to_datetime(["2018-08-02 03:00", "2018-08-02 12:00", "2018-08-02 16:00"], utc=True))
    days = ArchiveRules.burn_day(t, pd.Series([-120.0] * 3))
    assert list(days.dt.day) == [1, 1, 2]  # 19:00 and 04:00 local belong to Aug 1; 08:00 local to Aug 2


def perimeters(irwin: str | None, name: str, days: list[str], acres: list[float], **extra) -> list[dict]:
    """NIFS-like raw rows, one per day, 18:00 UTC (late morning in California)."""
    base = dict(_archive="nifs", _cx=-120.5, _cy=38.5, irwinid=irwin, incidentname=name,
                featurecategory="Wildfire Daily Fire Perimeter", mapmethod="Infrared Image", _year=2022) | extra  # fmt: skip
    return [base | {"polygondatetime": ms(f"{d} 18:00"), "gisacres": a} for d, a in zip(days, acres, strict=True)]


def irwin(n: int) -> str:
    return f"{{{n:08d}-0000-0000-0000-000000000000}}"


def key(n: int) -> str:
    return IncidentKey.of(irwin(n), "", "")


def test_each_gate_gives_its_reason():
    d3 = ["2022-08-01", "2022-08-02", "2022-08-03"]
    rows = (
        perimeters(irwin(1), "Good", d3, [200, 900, 1500])
        + perimeters(irwin(2), "Burn Plan", d3, [1500] * 3)
        + perimeters(irwin(3), "Small", d3, [100, 300, 900])
        + perimeters(irwin(4), "Short", d3[:2], [2000, 3000])
        + perimeters(irwin(5), "Alaska", d3, [5000] * 3, _cx=-150.0, _cy=64.0)
        + perimeters(irwin(6), "Sketch", d3, [5000] * 3, mapmethod="Hand Sketch")
        + perimeters(irwin(7), "Old", ["2012-08-01", "2012-08-02", "2012-08-03"], [5000] * 3)
        + perimeters(irwin(8), "Rx", d3, [5000] * 3, featurecategory="Prescribed Fire")
        + perimeters(irwin(9), "Gone", d3, [5000] * 3, deletethis="Yes")
        # two days in August, two in October: four burn days, but no run of three
        + perimeters(irwin(10), "Split", ["2022-08-01", "2022-08-02", "2022-10-01", "2022-10-02"], [1500] * 4)
    )
    gates = CatalogGates()
    out = gates.evaluate(gates.records(pd.DataFrame(rows)), {IncidentKey.irwin_uuid(key(2)): "RX"})
    why = dict(zip(out["incident"], out["reason"], strict=True))
    assert why == {
        key(1): "",
        key(2): "prescribed fire (WFIGS incident type)",
        key(3): "under 1000 acres",
        key(4): "under 3 burn days",
        key(5): "outside the continental US",
        key(6): "only hand-sketched or modeled perimeters",
        key(7): "outside 2014-2025",
        key(8): "not a wildfire perimeter",
        key(10): "under 3 burn days in its main run of perimeters",
    }  # key(9): every record flagged DeleteThis, so not an incident at all
    good = out.set_index("incident").loc[key(1)]
    assert (good["days"], good["year"], good["archive"]) == (3, 2022, "nifs") and len(good["_run_index"]) == 3
    assert good["fire_id"] == IncidentKey.fire_id(key(1))


def test_main_run_is_the_longest_stretch_without_a_long_gap():
    rows = perimeters(irwin(1), "Two Starts", ["2022-07-01", "2022-07-02", "2022-07-03", "2022-07-04",
                                               "2022-08-20", "2022-08-21", "2022-08-22"], [1500] * 7)  # fmt: skip
    gates = CatalogGates()
    rec = gates.records(pd.DataFrame(rows))
    row = gates.evaluate(rec).iloc[0]
    run = rec.loc[row["_run_index"]]
    assert (row["reason"], row["days"], row["all_days"]) == ("", 4, 7) and run["t"].max().month == 7


def test_gate_values_are_settings():
    rows = perimeters(irwin(1), "Mid", ["2022-08-01", "2022-08-02", "2022-08-03"], [600] * 3)
    loose = CatalogGates(min_acres=500, min_days=2, region=BoundingBox(-125, 24, -66, 50))
    rec = loose.records(pd.DataFrame(rows))
    assert loose.evaluate(rec)["reason"].tolist() == [""]
    assert CatalogGates(min_days=4).evaluate(rec)["reason"].tolist() == ["under 1000 acres"]


def test_a_far_off_record_carrying_the_same_irwin_id_is_not_part_of_the_fire():
    """As in the 2018 NIFS archive: 'Bacon Rind' records (Yellowstone) carry the IRWIN id of Natchez (Oregon)."""
    natchez = perimeters(irwin(1), "Natchez", ["2018-07-20", "2018-07-25", "2018-08-01", "2018-08-10"],
                         [2000, 8000, 12000, 14000], _cx=-123.5, _cy=41.9)  # fmt: skip
    natchez[1] |= {"_cx": -123.35, "_cy": 42.0}  # the fire grows north-east, under 25 km a step: still one group
    natchez[2] |= {"_cx": -123.2, "_cy": 42.1}
    bacon_rind = perimeters(irwin(1), "Bacon Rind", ["2018-08-26", "2018-09-07"], [2060, 980], _cx=-111.1, _cy=44.97)
    gates = CatalogGates()
    rec = gates.records(pd.DataFrame(natchez + bacon_rind))
    row = gates.evaluate(rec).iloc[0]
    run = rec.loc[row["_run_index"]]
    assert row["reason"] == "" and row["all_days"] == 6 and row["days"] == 4
    assert set(run["incidentname"]) == {"Natchez"} and run["_cx"].max() < -120
    two = gates.records(pd.DataFrame(natchez[:2] + bacon_rind))  # 2 days each: the far group cannot add its days
    assert gates.evaluate(two)["reason"].tolist() == ["under 3 burn days in its main run of perimeters"]
