"""Commands, offline: a grown channel or a changed window / region sends fires back to the queue, the error streak
stops only a broken channel, links cover every channel, --extents writes the catalog file from firms files, and
--repack gzips plain GeoJSON in place."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from fah.__main__ import main
from fah.commands.extents import ExtentsBuilder
from fah.commands.links import LinksBuilder
from fah.commands.progress import ChannelProgress
from fah.commands.run import ChannelRun, ErrorStreak
from fah.commands.settings import RunSettings
from fah.core.errors import ChannelUnavailable
from fah.core.extent import FireExtentEstimator
from fah.hub.catalog import FireCatalog
from fah.hub.manifest import ManifestBook
from fah.hub.repo import HubRepo


def test_grown_channel_refetches_only_fires_missing_new_files(channels):
    old = ";".join(f"gridmet_{k}.nc" for k in ("vpd", "fm100", "fm1000", "bi", "erc", "rmin", "vs", "th", "tmmx"))
    new = ";".join(sorted(channels.get("gridmet").expected_files))
    m = pd.DataFrame([dict(fire_id="A", status="ok", files=old, fetched_utc="2026-11-01"),
                      dict(fire_id="B", status="ok", files=new, fetched_utc="2026-11-01"),
                      dict(fire_id="C", status="n/a", files="", fetched_utc="2026-11-01"),
                      dict(fire_id="D", status="error", files="", fetched_utc="2026-11-01")])  # fmt: skip
    assert ChannelProgress.done_ids(m, channels.get("gridmet")) == {"B", "C"}
    assert ChannelProgress.done_ids(m, channels.get("terrain")) == {"A", "B", "C"}  # no expected files: any ok row
    assert ChannelProgress.done_ids(m) == {"A", "B", "C"}


def test_fires_fetched_for_another_window_or_region_are_fetched_again(camp_perimeters, channels):
    a, c = camp_perimeters.replace(fire_id="A"), camp_perimeters.replace(fire_id="C")
    b = camp_perimeters.replace(fire_id="B").with_extent({"start_day": "2018-11-01", "end_day": "2018-11-30"})
    hrrr = channels.get("hrrr")
    reg = hrrr.region_label(a)
    m = pd.DataFrame([dict(fire_id="A", status="ok", fetched_utc="2026-11-01", window=hrrr.window_label(a), region=reg),
                      dict(fire_id="B", status="ok", fetched_utc="2026-11-01", window=hrrr.window_label(a), region=reg),
                      dict(fire_id="C", status="ok", fetched_utc="2026-11-01", window="", region="")])  # fmt: skip
    m["files"] = "hrrr_ana.nc;hrrr_f03.nc"
    fires = [a, b, c]
    assert ChannelProgress.done_ids(m, hrrr, fires) == {"A"}  # B's window grew, C has neither
    old = m.assign(files="hrrr_ana.nc")  # fetched before hrrr_f03.nc existed
    assert ChannelProgress.done_ids(old, hrrr, fires) == set()
    assert ChannelProgress.done_ids(m, channels.get("terrain"), fires) == {"A", "B"}  # C was cut to the old box
    gz = m.assign(files="records.geojson.gz")
    assert ChannelProgress.done_ids(gz, channels.get("perimeters"), fires) == {
        "A",
        "B",
        "C",
    }  # perimeters' box, no window
    plain = gz.assign(files=["records.geojson", "records.geojson.gz", "records.geojson"])
    assert ChannelProgress.done_ids(plain, channels.get("perimeters"), fires) == {"B"}  # plain .geojson: repack
    old = m.drop(columns=["window", "region"])  # manifests written before the columns
    assert ChannelProgress.done_ids(old, channels.get("goes"), fires) == set()
    assert ChannelProgress.done_ids(old, channels.get("goes")) == {"A", "B", "C"}


def test_error_streak_stops_a_broken_channel_only():
    s = ErrorStreak(3)
    assert not any(s.hit(st, "e") for st in ["error", "error", "ok", "error", "error", "n/a"])
    assert [s.hit("error", f"e{i}") for i in range(3)] == [False, False, True] and s.tripped and s.last == "e2"
    never = ErrorStreak(0)
    assert not any(never.hit("error", "e") for _ in range(50))


def test_fire_links_cover_every_channel_with_files_and_sources(camp, channels):
    lf = "https://lfps.usgs.gov/api/job/submit?Layer_List=LF2016_FBFM40%3BLF2016_CC&Area_of_Interest=-121.89+39.51"
    rows = [dict(fire_id=camp.fire_id, channel="terrain", status="error", files="", sources="", bytes="0",
                 fetched_utc="2026-11-15T10:00:00+00:00", error="timeout"),
            dict(fire_id=camp.fire_id, channel="terrain", status="ok", files="dem_glo30.tif", bytes="10106268",
                 sources="https://copernicus-dem-30m.s3.amazonaws.com/a.tif", fetched_utc="2026-11-15T11:00:00+00:00",
                 error=""),
            dict(fire_id=camp.fire_id, channel="landfire", status="ok", files="landfire_LF2016.zip", bytes="5",
                 sources=lf, fetched_utc="2026-11-15T11:00:00+00:00", error=""),
            dict(fire_id=camp.fire_id, channel="hrrr", status="ok", files="hrrr_ana.nc;hrrr_f03.nc", bytes="9",
                 sources=ManifestBook.SEP.join(["https://h/a.grib2", "https://h/b.grib2"]),
                 fetched_utc="2026-11-15T11:00:00+00:00", error=""),
            dict(fire_id=camp.fire_id, channel="ics209", status="n/a", files="", bytes="0", sources="",
                 fetched_utc="2026-11-15T11:00:00+00:00", error="ICS-209-PLUS covers 1999-2020")]  # fmt: skip
    links = LinksBuilder.fire_links(camp, rows, "me/raw", channels.names())
    assert list(links["channels"]) == channels.names()
    t = links["channels"]["terrain"]
    assert t["status"] == "ok"
    assert (
        t["files"][0]["url"]
        == "https://huggingface.co/datasets/me/raw/resolve/main/fires/F2003403265/terrain/dem_glo30.tif"
    )
    assert links["channels"]["landfire"]["sources"] == [lf]  # ';' inside a URL is not a separator
    assert len(links["channels"]["hrrr"]["sources"]) == 2 and len(links["channels"]["hrrr"]["files"]) == 2
    assert links["channels"]["ics209"]["note"].startswith("ICS-209") and links["channels"]["goes"] == {
        "status": "pending"
    }
    assert links["folder"].endswith("/tree/main/fires/F2003403265")
    assert links["perimeters_box_lonlat"] == [round(v, 5) for v in camp.box] and links["archive"] == "both"
    md = LinksBuilder.markdown(links)
    assert (
        "[dem_glo30.tif](https://huggingface.co/datasets/me/raw/resolve/main/fires/F2003403265/terrain/dem_glo30.tif)"
        in md
    )
    assert "| goes | pending | — | — |" in md and "hrrr: 2 links" in md
    json.dumps(links)  # links.json is plain JSON


def test_extents_step_writes_catalog_from_local_firms_files(tmp_path, monkeypatch, camp_perimeters, channels):
    batch = tmp_path / "batch" / "firms-x-000"
    fd = batch / "fires" / camp_perimeters.fire_id / "firms"
    fd.mkdir(parents=True)
    rows = [dict(latitude=39.7, longitude=-121.6, acq_date="2018-11-07", acq_time=2100, confidence="h", type=0,
                 bright_ti4=330.0)] * 6  # fmt: skip
    with gzip.open(fd / "VIIRS_SNPP_SP.csv.gz", "wt") as fh:
        pd.DataFrame(rows).to_csv(fh, index=False)
    (batch / "manifest" / "firms").mkdir(parents=True)
    window = channels.get("firms").window_label(camp_perimeters)
    pd.DataFrame([dict(fire_id=camp_perimeters.fire_id, status="ok", fetched_utc="2026-11-18", window=window,
                       region=channels.get("firms").region_label(camp_perimeters), files="VIIRS_SNPP_SP.csv.gz")],
                 columns=ManifestBook.COLUMNS).to_csv(batch / "manifest" / "firms" / "firms-x-000.csv", index=False)  # fmt: skip
    first = ("2018-11-09T02:00:00+00:00", 39.75, -121.55)  # the other fire: no FIRMS, its first perimeter record
    other = camp_perimeters.replace(fire_id="F0000000000", first_record=first)
    monkeypatch.setattr(FireCatalog, "fires", lambda self, archive="all", ids=None: [camp_perimeters, other])
    s = RunSettings(work=tmp_path / "w", dry_run=True, local_manifests=str(tmp_path / "batch"))
    assert ExtentsBuilder(s).execute() == 0
    path = tmp_path / "w" / "extents" / FireCatalog.EXTENTS_FILE
    t = pd.read_csv(path).set_index("fire_id")
    camp_row = t.loc[camp_perimeters.fire_id]
    assert camp_row["start_day"] == "2018-11-06" and camp_row["window_rule"] == "perimeters+firms"
    assert t.loc["F0000000000", "start_day"] == "2018-11-08" and t.loc["F0000000000", "window_rule"] == "perimeters"
    assert camp_row["ignition_method"] == "firms" and camp_row["ignition_time_utc"] == "2018-11-07T21:00Z"
    assert (
        t.loc["F0000000000", "ignition_method"] == "first_perimeter" and t.loc["F0000000000", "ignition_lat"] == 39.75
    )
    assert camp_row["region_west"] == round(camp_perimeters.region()[0], 5)
    table = FireCatalog.load_extents(None, path=path)
    assert camp_perimeters.with_extent(table[camp_perimeters.fire_id]).span()[0] == pd.Timestamp("2018-11-06", tz="UTC")
    assert bool(camp_row["firms_current"]) and not bool(t.loc["F0000000000", "firms_current"])

    # a time-dependent channel waits until every window is final, unless told to go ahead
    run = ChannelRun(RunSettings(channel="gridmet", work=tmp_path / "w", extents_file=str(path)))
    with pytest.raises(ChannelUnavailable, match="1 of 2 fires have no final window"):
        run._with_extents(None, [camp_perimeters, other], tmp_path / "w")
    ahead = ChannelRun(
        RunSettings(channel="gridmet", work=tmp_path / "w", extents_file=str(path), partial_windows=True)
    )
    got = ahead._with_extents(None, [camp_perimeters, other], tmp_path / "w")
    assert got[0].window_days == ("2018-11-06", "2018-11-26") and got[1].window_days is not None
    moved = camp_perimeters.replace(first="2018-11-07 20:00:00+00:00")  # the catalog changed since --extents
    assert not FireExtentEstimator.is_final(moved, table[camp_perimeters.fire_id])
    assert FireExtentEstimator.is_final(camp_perimeters, table[camp_perimeters.fire_id])


def test_repack_gzips_plain_geojson_in_place_and_keeps_the_old_rows_labels(tmp_path, monkeypatch, channels):
    feats = [{"type": "Feature", "properties": {"name": "Camp", "acres": 153336}, "geometry": None}]
    hub = tmp_path / "hub"
    for fid in ("A", "B"):
        p = hub / "fires" / fid / "perimeters" / "records.geojson"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"type": "FeatureCollection", "features": feats, "fid": fid}))
    rows = [dict(fire_id="A", name="Camp", year="2018", channel="perimeters", status="ok", files="records.geojson",
                 sources="https://nifc/a | https://nifc/b", fetched_utc="2026-11-16T10:00:00+00:00", window="", region=""),
            dict(fire_id="B", name="Old", year="2018", channel="perimeters", status="ok", files="records.geojson.gz",
                 sources="", fetched_utc="2026-11-16T10:00:00+00:00", window="", region=""),
            dict(fire_id="C", name="Err", year="2018", channel="perimeters", status="error", files="",
                 sources="", fetched_utc="2026-11-16T10:00:00+00:00", window="", region="")]  # fmt: skip
    monkeypatch.setattr(ManifestBook, "read", classmethod(lambda cls, hub, channel=None: pd.DataFrame(rows)))
    got = []

    def download(self, path, local_dir=None):
        got.append(path)
        dst = Path(local_dir) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((hub / path).read_bytes())
        return str(dst)

    monkeypatch.setattr(HubRepo, "download", download)
    work = tmp_path / "w"
    assert main(["--repack", "--channel", "perimeters", "--hf-repo", "me/raw", "--dry-run", "--work", str(work)]) == 0
    assert got == ["fires/A/perimeters/records.geojson"]  # B is already gzipped, C has nothing
    [batch] = (work / "batch").iterdir()
    gz = batch / "fires" / "A" / "perimeters" / "records.geojson.gz"
    assert gzip.open(gz).read() == (hub / "fires/A/perimeters/records.geojson").read_bytes()
    [m] = (batch / "manifest" / "perimeters").glob("*.csv")
    row = pd.read_csv(m, dtype=str, keep_default_na=False).iloc[0].to_dict()
    assert row["fire_id"] == "A" and row["files"] == "records.geojson.gz" and row["status"] == "ok"
    assert row["sources"] == "https://nifc/a | https://nifc/b" and "2026-11-16T10:00:00" in row["error"]
    assert ChannelProgress.done_ids(pd.DataFrame([row]), channels.get("perimeters")) == {"A"}
    assert not (work / "repack_download").exists()
