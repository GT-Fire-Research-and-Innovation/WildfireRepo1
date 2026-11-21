"""--catalog build, offline: raw crawl tables in, fires.csv / incidents.csv / gates.json out, and FireCatalog reads
the result back as fires with a box, a first record and a reported origin."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fah.commands.catalog import CatalogBuilder
from fah.commands.settings import RunSettings
from fah.hub.catalog import FireCatalog

pytest.importorskip("pyarrow")

CAMP = "{75E64DB8-9B75-4A68-BEDD-67CC62658E38}"
RX = "{11111111-2222-3333-4444-555555555555}"


def ms(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").value // 1_000_000)


def nifs_rows(irwin: str, name: str, days: list[str], acres: list[float], x0: float) -> list[dict]:
    """Raw rows as the scan saves a geodatabase layer: the layer's own field names, record box and centroid."""
    return [
        {"IncidentName": name, "IRWINID": irwin, "FeatureCategory": "Wildfire Daily Fire Perimeter",
         "MapMethod": "Infrared Image", "GISAcres": a, "PolygonDateTime": ms(f"{d} 18:00"), "DeleteThis": "No",
         "Comments": "kept raw", "_cx": x0 + 0.1 * k, "_cy": 39.7, "_xmin": x0 + 0.1 * k - 0.05, "_ymin": 39.6 - 0.01 * k,
         "_xmax": x0 + 0.1 * k + 0.05, "_ymax": 39.8, "_archive": "nifs", "_year": 2018, "_layer": "ODA 2018",
         "_url": "fgdb://oda2018.gdb|perimeters"}
        for k, (d, a) in enumerate(zip(days, acres, strict=True))
    ]  # fmt: skip


def test_build_writes_the_gated_fire_list_with_box_and_incident_fields(tmp_path):
    raw = tmp_path / "w" / "catalog_repo" / "catalog" / "raw"
    (raw / "perimeters").mkdir(parents=True)
    (raw / "incidents").mkdir(parents=True)
    days = ["2018-11-08", "2018-11-09", "2018-11-10", "2018-11-12"]
    rows = nifs_rows(CAMP, "Camp", days, [20000, 70000, 100000, 140000], -121.7)
    rows += nifs_rows(RX, "Burn Unit 4", days[:3], [1500] * 3, -110.0)
    pd.DataFrame(rows).to_parquet(raw / "perimeters" / "2018_nifs_ODA_2018.parquet", index=False)
    wfigs = pd.DataFrame([
        {"IrwinID": CAMP, "IncidentTypeCategory": "WF", "FireDiscoveryDateTime": ms("2018-11-08 14:33"),
         "FireCauseGeneral": "Power generation/transmission/distribution", "FinalAcres": 153336.0,
         "InitialLatitude": None, "InitialLongitude": None, "_lat": 39.8136, "_lon": -121.4347, "POOState": "US-CA"},
        {"IrwinID": RX, "IncidentTypeCategory": "RX", "FireDiscoveryDateTime": None, "_lat": 33.0, "_lon": -110.0},
    ])  # fmt: skip
    wfigs.to_parquet(raw / "incidents" / "wfigs_2018.parquet", index=False)

    s = RunSettings(work=tmp_path / "w", dry_run=True, catalog="build", first_year=2018, last_year=2018, threads=2)
    assert CatalogBuilder(s).execute() == 0
    out = tmp_path / "w" / "catalog_repo" / "catalog"
    fires = pd.read_csv(out / "fires.csv")
    assert fires["fire_id"].tolist() == ["F2003403265"]  # the old id; the prescribed burn is gated out
    camp = fires.iloc[0]
    assert (camp["archive"], camp["days"], camp["records"], camp["max_gis_acres"]) == ("nifs", 4, 4, 140000.0)
    assert (camp["west"], camp["east"], camp["south"], camp["north"]) == pytest.approx((-121.75, -121.35, 39.57, 39.8))
    assert camp["first_record_time"] == "2018-11-08T18:00:00+00:00" and camp["first_lon"] == pytest.approx(-121.7)
    assert camp["discovery_utc"] == "2018-11-08T14:33:00+00:00" and camp["incident_lat"] == pytest.approx(39.8136)
    assert camp["irwin_id"] == CAMP.strip("{}") and camp["final_acres"] == 153336.0

    report = pd.read_csv(out / "incidents.csv").set_index("fire_id")
    assert len(report) == 2 and (report["reason"].fillna("") != "").sum() == 1
    assert "prescribed fire (WFIGS incident type)" in report["reason"].tolist()
    gates = json.loads((out / "gates.json").read_text())
    assert gates["fires"] == 1 and gates["gates"]["min_days"] == 3 and gates["gates"]["first_year"] == 2018

    [fire] = FireCatalog(None, tmp_path / "cache", out / "fires.csv").fires()
    assert fire.fire_id == "F2003403265" and fire.archive == "nifs" and fire.box.west == pytest.approx(-121.75)
    assert fire.origin[1:] == pytest.approx((39.8136, -121.4347)) and fire.first_record[1] == pytest.approx(39.7)
    assert fire.region().west < fire.box.west - 0.3  # 32 km out


def test_build_without_a_scan_says_so(tmp_path):
    s = RunSettings(work=tmp_path, dry_run=True, catalog="build")
    with pytest.raises(RuntimeError, match="run --catalog scan first"):
        CatalogBuilder(s).execute()
