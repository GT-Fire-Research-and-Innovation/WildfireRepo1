"""The perimeter archive: every changed outline is saved once, a late poll catches up, nothing is
re-downloaded for no reason, and failures are logged, not hidden. Runs against a fake NIFC service."""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live"))
import wfigs_poller as WP  # noqa: E402

NOW_MS = int(time.time() * 1000)
DAY = 86400_000


def feat(oid, fid, edited_days_ago, size=1.0, mapped_days_ago=None, geometry=True):
    """One perimeter as NIFC serves it: a square whose side is `size`."""
    mapped = edited_days_ago if mapped_days_ago is None else mapped_days_ago
    return {"OBJECTID": oid, "attr_UniqueFireIdentifier": fid, "attr_IrwinID": "{%s}" % fid,
            "attr_IncidentName": fid, "poly_DateCurrent": NOW_MS - int(edited_days_ago * DAY),
            "poly_PolygonDateTime": None if mapped is None else NOW_MS - int(mapped * DAY),
            "Shape__Area": size * size, "Shape__Length": 4 * size, "poly_GISAcres": size,
            "_geom": {"type": "Polygon", "coordinates": [[[0, 0], [size, 0], [size, size], [0, size], [0, 0]]]} if geometry else None}


class Service:
    """Stands in for the two NIFC services; counts outline downloads."""

    def __init__(self, current=(), ytd=()):
        self.data = {WP.URL: list(current), WP.YTD: list(ytd)}
        self.outlines = 0
        self.page_cap = None

    def get(self, url, params, tries=4):
        rows = self.data[url]
        m = re.match(r"OBJECTID IN \((.*)\)", params["where"])
        if m:
            ids = {int(x) for x in m.group(1).split(",")}
            rows = [r for r in rows if r["OBJECTID"] in ids]
        rows = sorted(rows, key=lambda r: r["OBJECTID"])[params["resultOffset"]:]
        cap = self.page_cap or params["resultRecordCount"]
        page, more = rows[:cap], len(rows) > cap
        if params["f"] == "json":
            keep = WP.SIG_FIELDS.split(",")
            return {"features": [{"attributes": {k: r.get(k) for k in keep}} for r in page], "exceededTransferLimit": more}
        self.outlines += len(page)
        return {"features": [{"type": "Feature", "geometry": r["_geom"],
                              "properties": {k: v for k, v in r.items() if k != "_geom"}} for r in page],
                "properties": {"exceededTransferLimit": True} if more else {}}


@pytest.fixture
def svc(monkeypatch):
    s = Service()
    monkeypatch.setattr(WP, "_get", s.get)
    return s


def index(out):
    return pd.read_csv(out / "perimeters" / "index.csv")


def test_first_poll_saves_current_and_recent_and_only_notes_old_ones(svc, tmp_path):
    svc.data[WP.URL] = [feat(1, "A", 200)]                                   # still in Current, however old
    svc.data[WP.YTD] = [feat(1, "A", 200), feat(2, "B", 3), feat(3, "C", 200), feat(4, "D", 200, mapped_days_ago=2)]
    st = WP.poll(tmp_path)
    assert st["saved"] == 3 and set(index(tmp_path)["fire_id"]) == {"A", "B", "D"}
    assert svc.outlines == 3                                                 # A once (not from both services), B, D
    assert len(pd.read_csv(tmp_path / "perimeters" / "seen.csv")) == 4       # C is noted, to be saved when it changes


def test_nothing_changed_downloads_nothing(svc, tmp_path):
    svc.data[WP.YTD] = [feat(1, "A", 1), feat(2, "B", 1)]
    WP.poll(tmp_path)
    svc.outlines = 0
    st = WP.poll(tmp_path)
    assert st["changed"] == 0 and st["saved"] == 0 and svc.outlines == 0


def test_an_edited_outline_becomes_a_new_version_whenever_the_poll_comes(svc, tmp_path):
    svc.data[WP.YTD] = [feat(1, "A", 1), feat(3, "C", 200)]
    WP.poll(tmp_path)
    # both are redrawn; C's edit is 12 days old by the time we look (a long outage): still caught
    svc.data[WP.YTD] = [feat(1, "A", 0, size=2.0), feat(3, "C", 12, size=3.0)]
    st = WP.poll(tmp_path)
    idx = index(tmp_path)
    assert st["saved"] == 2 and len(idx[idx.fire_id == "A"]) == 2 and len(idx[idx.fire_id == "C"]) == 1
    files = [tmp_path / "perimeters" / f for f in idx["file"]]
    assert all(f.exists() for f in files) and len(set(files)) == 3


def test_outline_change_with_no_change_in_dates_is_still_caught(svc, tmp_path):
    a = feat(1, "A", 1)
    svc.data[WP.YTD] = [a]
    WP.poll(tmp_path)
    b = feat(1, "A", 1, size=1.5)
    b["poly_DateCurrent"], b["poly_PolygonDateTime"] = a["poly_DateCurrent"], a["poly_PolygonDateTime"]
    svc.data[WP.YTD] = [b]
    assert WP.poll(tmp_path)["saved"] == 1


def test_edit_that_leaves_the_outline_alone_is_not_a_new_version_and_not_refetched(svc, tmp_path):
    a = feat(1, "A", 1)
    svc.data[WP.YTD] = [a]
    WP.poll(tmp_path)
    svc.data[WP.YTD] = [dict(a, poly_DateCurrent=NOW_MS)]                    # e.g. containment updated
    st = WP.poll(tmp_path)
    assert st["changed"] == 1 and st["saved"] == 0 and len(index(tmp_path)) == 1
    svc.outlines = 0
    assert WP.poll(tmp_path)["changed"] == 0 and svc.outlines == 0


def test_existing_archive_is_not_duplicated(svc, tmp_path):
    """Switching an archive made by the old poller (index.csv, no seen.csv) to this one adds no copies."""
    svc.data[WP.URL] = [feat(1, "A", 1)]
    WP.poll(tmp_path)
    (tmp_path / "perimeters" / "seen.csv").unlink()
    assert WP.poll(tmp_path)["saved"] == 0 and len(index(tmp_path)) == 1


def test_fire_with_several_polygons_and_no_fire_id(svc, tmp_path):
    x = feat(3, None, 1)
    x["attr_IrwinID"] = "{ABC-1}"
    svc.data[WP.YTD] = [feat(1, "A", 1), feat(2, "A", 1, size=2.0), x]
    assert WP.poll(tmp_path)["saved"] == 3
    assert sorted(index(tmp_path)["fire_id"]) == ["A", "A", "ABC-1"]


def test_pages_cut_short_by_the_service_are_followed(svc, tmp_path):
    svc.data[WP.YTD] = [feat(i, f"F{i}", 1, size=1 + i / 10) for i in range(1, 8)]
    svc.page_cap = 3                                                         # fewer than asked for, with the flag set
    assert WP.poll(tmp_path)["saved"] == 7


def test_missing_outline_is_skipped_once(svc, tmp_path):
    svc.data[WP.YTD] = [feat(1, "A", 1, geometry=False)]
    assert WP.poll(tmp_path)["saved"] == 0
    assert WP.poll(tmp_path)["changed"] == 0


def test_failed_poll_is_logged_and_the_next_one_recovers(svc, tmp_path, monkeypatch):
    svc.data[WP.YTD] = [feat(1, "A", 1)]

    def down(url, params, tries=4):
        raise RuntimeError("ArcGIS error: service unavailable")
    monkeypatch.setattr(WP, "_get", down)
    assert WP.poll_logged(tmp_path) is False
    monkeypatch.setattr(WP, "_get", svc.get)
    assert WP.poll_logged(tmp_path) is True
    log = pd.concat(pd.read_csv(f) for f in (tmp_path / "perimeters" / "polls").glob("*.csv"))
    assert list(log["ok"]) == [0, 1] and "service unavailable" in log["error"].iloc[0] and log["saved"].iloc[1] == 1
    WP.report(tmp_path)                                                      # runs on a small archive


def test_error_body_with_http_200_is_an_error(monkeypatch):
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": {"code": 400, "message": "Cannot perform query"}}
    monkeypatch.setattr(WP.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(WP.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        WP._get(WP.URL, {})
