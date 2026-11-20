"""Upstream clients, offline: retries and their limits, id paging that checks completeness, URLs exactly as each
service wants them, HRRR byte ranges, Earthdata names and signed redirects, and GFS grid arithmetic."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fah.core.geometry import PixelWindow, SinusoidalGrid
from fah.sources.arcgis import ArcGisClient
from fah.sources.earthdata import EarthdataClient, RangeFile
from fah.sources.gfs import EarthEngineGfs
from fah.sources.goes import GoesArchive
from fah.sources.gridmet import GridmetService
from fah.sources.hrrr import HrrrArchive
from fah.sources.http import HttpClient, ParallelMap
from fah.sources.landfire import LandfireImageService
from fah.sources.nifc import NifcArchive


class Reply:
    def __init__(self, code: int, body: bytes = b"") -> None:
        self.status_code, self.content, self.text = code, body, body.decode(errors="replace")

    def json(self):
        return json.loads(self.content)


class Sess:
    def __init__(self, replies: list) -> None:
        self.replies, self.calls = list(replies), 0

    def get(self, url, timeout):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_http_get_retries_server_errors_not_client_errors(monkeypatch):
    import requests

    monkeypatch.setattr("fah.sources.http.time.sleep", lambda s: None)
    s = Sess([Reply(503), requests.ConnectionError("reset"), Reply(200, b"CDF\x01")])
    assert HttpClient(s).get_bytes("u") == b"CDF\x01" and s.calls == 3
    s = Sess([Reply(400, b"Requested time range is outside the dataset")])
    with pytest.raises(RuntimeError, match="HTTP 400: Requested time range"):
        HttpClient(s).get_bytes("u")
    assert s.calls == 1


def test_windowed_map_keeps_order():
    with ThreadPoolExecutor(4) as ex:
        assert list(ParallelMap.windowed(ex, lambda k: k * k, range(50), 3)) == [k * k for k in range(50)]


class IdSession:
    """ArcGIS layer with 450 features: lists ids, then returns the ones asked for (optionally dropping one)."""

    def __init__(self, drop: int | None = None) -> None:
        self.drop, self.posts = drop, []

    def get(self, url, params, timeout):
        assert params["returnIdsOnly"] == "true" and "resultOffset" not in params
        return Reply(200, json.dumps({"objectIds": list(range(450, 0, -1))}).encode())

    def post(self, url, data, timeout):
        ids = [int(i) for i in data["objectIds"].split(",") if int(i) != self.drop]
        self.posts.append(len(ids))
        return Reply(200, json.dumps({"type": "FeatureCollection", "features": [{"id": i} for i in ids]}).encode())


def test_box_features_page_by_id_and_check_completeness(camp):
    s = IdSession()
    feats = ArcGisClient(s).box_features("https://x/FeatureServer/1", camp.build_box)
    assert [f["id"] for f in feats] == list(range(1, 451)) and s.posts == [200, 200, 50]
    with pytest.raises(RuntimeError, match="listed 450 features, received 449"):
        ArcGisClient(IdSession(drop=7)).box_features("https://x/FeatureServer/1", camp.build_box)


def test_query_links_are_complete_urls(camp):
    u = ArcGisClient.query_url("https://x/FeatureServer/0", camp.build_box, where="FIRE_YEAR_INT < 2018")
    assert "f=geojson" in u and "where=FIRE_YEAR_INT+%3C+2018" in u and "geometry=-121.88999%2C39.51118" in u


def test_gridmet_url_is_plain_https_subset(camp):
    d = camp.days()
    u = GridmetService.subset_url("vpd", "daily_mean_vapor_pressure_deficit", camp.build_box, d[0], d[-1], 0.05)
    assert u.startswith(
        "https://thredds.northwestknowledge.net/thredds/ncss/grid/agg_met_vpd_1979_CurrentYear_CONUS.nc?"
    )
    assert "time_start=2018-10-26T00:00:00Z" in u and "time_end=2018-11-27T00:00:00Z" in u and "accept=netcdf" in u
    assert "north=40.0346" in u and "west=-121.9400" in u and "/dodsC/" not in u


def test_gridmet_gains_tssatfire_variables(camp):
    v = GridmetService.VARIABLES
    assert len(v) == 13 and {"pr", "tmmn", "sph", "pdsi"} <= set(v)
    assert v["pdsi"] == "daily_mean_palmer_drought_severity_index" and v["sph"] == "daily_mean_specific_humidity"
    d = camp.days()
    start = d[0] - pd.Timedelta(days=GridmetService.LEAD_DAYS["pdsi"])
    u = GridmetService.subset_url("pdsi", v["pdsi"], camp.build_box, start, d[-1], 0.05)
    assert "/ncss/grid/agg_met_pdsi_1979_CurrentYear_CONUS.nc?var=daily_mean_palmer_drought_severity_index" in u
    assert "time_start=2018-10-16T00:00:00Z" in u


def test_landfire_window_snaps_to_native_30m_grid(camp):
    pytest.importorskip("pyproj")
    x0, y0, x1, y1, w, h = LandfireImageService.native_window(camp.build_box, (-2362425, 3177435), 30)
    assert (x0 + 2362425) % 30 == 0 and (3177435 - y1) % 30 == 0  # pixel edges of the stored grid
    assert (w, h) == ((x1 - x0) / 30, (y1 - y0) / 30) == (2246, 2195)  # Camp, as served
    info = {"extent": {"xmin": -2362425, "ymax": 3177435}, "pixelSizeX": 30, "spatialReference": {"wkid": 5070},
            "pixelType": "S16", "noDataValue": -9999}  # fmt: skip
    u = LandfireImageService.export_url("LF2016", "FBFM40", camp.build_box, info)
    assert "/Landfire_LF2016/LF2016_FBFM40_CONUS/ImageServer/exportImage?" in u
    assert "size=2246%2C2195" in u and "interpolation=RSP_NearestNeighbor" in u and "compression=LZW" in u
    assert "Email" not in u


def test_hrrr_forecast_takes_precipitation_accumulated_from_the_run_start():
    idx = ("1:0:d=2020090712:REFC:entire atmosphere:9 hour fcst:\n"
           "2:100:d=2020090712:TMP:2 m above ground:9 hour fcst:\n"
           "3:200:d=2020090712:SPFH:2 m above ground:9 hour fcst:\n"
           "4:300:d=2020090712:APCP:surface:8-9 hour acc fcst:\n"
           "5:400:d=2020090712:APCP:surface:0-9 hour acc fcst:\n"
           "6:500:d=2020090712:WEASD:surface:0-9 hour acc fcst:\n")  # fmt: skip

    class S:
        def get(self, url, timeout):
            return type("R", (), {"status_code": 200, "text": idx})()

    rng = HrrrArchive(S()).byte_ranges("u", HrrrArchive.FORECAST_FIELDS)
    assert rng["t2"] == (100, 199) and rng["q2"] == (200, 299) and rng["apcp"] == (400, 499)
    assert HrrrArchive(S()).byte_ranges("u").keys() <= {"gust", "u10", "v10", "rh2"}  # the analyses' fields


def test_netcdf_bytes_fall_back_to_a_file(tmp_path):
    class NoMem:
        def Dataset(self, path, memory=None):
            if memory is not None:
                raise OSError("NetCDF: HDF error")
            return ("opened", Path(path).read_bytes())

    ds, tmp = GoesArchive(HttpClient(None), tmp_path).open_bytes(b"CDF\x01data", NoMem())
    assert ds == ("opened", b"CDF\x01data") and tmp.exists()


def test_tiles_and_versions_from_granule_names():
    a = "https://x/VNP13A1.A2018305.h08v05.002.2022307200550.h5"
    assert SinusoidalGrid.tile_hv(a.rsplit("/", 1)[1]) == (8, 5)
    assert EarthdataClient.granule_parts(a) == ("VNP13A1", "A2018305", "h08v05", "002")
    assert EarthdataClient.granule_day(a) == pd.Timestamp("2018-11-01", tz="UTC")
    g = [{"url": "https://x/VJ102IMG.A2018314.2000.021.1.nc", "start": "b", "name": "1"},
         {"url": "https://x/VJ102IMG.A2018314.2000.002.1.nc", "start": "b", "name": "2"},
         {"url": "https://x/VJ102IMG.A2018314.1018.021.1.nc", "start": "a", "name": "3"}]  # fmt: skip
    assert [x["name"] for x in EarthdataClient.newest_versions(g)] == ["3", "1"]


class RangeSess:
    """Earthdata Cloud in miniature: the NASA URL redirects (token required) to a signed S3 URL that serves byte
    ranges of `data` and refuses any Authorization header."""

    def __init__(self, data: bytes, token: str = "tok", login: bool = False) -> None:
        self.data, self.token, self.login, self.calls = data, token, login, []

    def get(self, url, headers=None, allow_redirects=True, stream=False, timeout=None):
        h = dict(headers or {})
        self.calls.append((url, h.get("Authorization")))
        r = type("R", (), {"close": lambda s: None})()
        r.headers, r.text = {}, ""
        if "nasa.gov" in url:
            if self.login or h.get("Authorization") != f"Bearer {self.token}":
                r.status_code, r.headers = 302, {"Location": "https://urs.earthdata.nasa.gov/oauth/authorize?x=1"}
            else:
                r.status_code = 307
                r.headers = {"Location": "https://bucket.s3.us-west-2.amazonaws.com/f.nc?X-Amz-Signature=s"}
            return r
        assert "Authorization" not in h, "token sent to S3"
        a, b = (int(x) for x in h["Range"].split("=")[1].split("-"))
        r.status_code, r.content = 206, self.data[a : b + 1]
        r.headers = {"Content-Range": f"bytes {a}-{b}/{len(self.data)}"}
        return r


def test_range_file_reads_hdf5_through_a_signed_redirect(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "x.nc"
    with h5py.File(p, "w") as f:
        f.create_dataset("observation_data/I04", data=np.arange(300 * 200, dtype="u2").reshape(300, 200),
                         chunks=(50, 200), compression="gzip")  # fmt: skip
    s = RangeSess(p.read_bytes())
    fo = RangeFile(s, "https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads/VNP02IMG/x.nc", "tok", block=4096)
    with h5py.File(fo, "r") as f:
        assert (f["observation_data/I04"][120:130, 7:9] == np.arange(300 * 200).reshape(300, 200)[120:130, 7:9]).all()
    assert fo.fetched < p.stat().st_size and all(a is None for u, a in s.calls if "amazonaws" in u)
    with pytest.raises(PermissionError, match="EARTHDATA_TOKEN"):
        RangeFile(RangeSess(b"", login=True), "https://data.lpdaac.earthdatacloud.nasa.gov/x.h5", "tok")


GRID = ("EPSG:4326", (0.25, 0.0, -180.125, 0.0, -0.25, 90.125), 1440, 721)  # cell centres on 0.25 deg
GRID360 = ("EPSG:4326", (0.25, 0.0, -0.125, 0.0, -0.25, 90.125), 1440, 721)


def test_gfs_window_is_cut_on_the_native_grid_either_longitude_convention(camp):
    pytest.importorskip("affine")
    box = camp.build_box
    w = EarthEngineGfs.window(GRID[1], GRID[2], GRID[3], box)
    lat, lon = EarthEngineGfs.latlon(GRID[1], GRID[2], w)
    h = 0.125  # cells (not centres) cover the box
    assert lat[0, 0] + h >= box.north and lat[-1, 0] - h <= box.south
    assert lon[0, 0] - h <= box.west and lon[0, -1] + h >= box.east
    assert lon[0, 1] - h > box.west and lon[0, -2] + h < box.east  # and no more than that
    assert np.allclose((lat * 4) % 1, 0) and np.allclose((lon * 4) % 1, 0)  # whole 0.25 deg cells, no shift
    w360 = EarthEngineGfs.window(GRID360[1], GRID360[2], GRID360[3], box)
    lat3, lon3 = EarthEngineGfs.latlon(GRID360[1], GRID360[2], w360)
    assert np.allclose(lat3, lat) and np.allclose(lon3, lon) and w360[2] > 720


def test_gfs_fires_of_one_run_share_requests_within_a_size_limit():
    wins = {
        "a": PixelWindow(10, 13, 100, 103),
        "b": PixelWindow(11, 14, 102, 105),
        "c": PixelWindow(300, 303, 900, 903),
    }
    cl = EarthEngineGfs.clusters(wins, max_cells=100)
    assert sorted(sorted(ids) for _, ids in cl) == [["a", "b"], ["c"]]
    assert dict((tuple(ids), w) for w, ids in cl)[("a", "b")] == (10, 14, 100, 105)
    assert len(EarthEngineGfs.clusters(wins, max_cells=9)) == 3
    from fah.channels.gfs import GfsChannel

    assert [len(c) for c in EarthEngineGfs.lead_chunks(list(GfsChannel.LEADS))] == [45, 35]


class Listing:
    """The NIFC service listing: empty for the first `empty` calls (as ArcGIS Online now and then answers), then
    one 2018 archive service whose layer list is `layers`."""

    def __init__(self, empty: int, layers: list | None = None) -> None:
        self.empty, self.calls = empty, 0
        self.layers = (
            [{"id": 1, "name": "EventLine2018"}, {"id": 2, "name": "EventPolygon2018"}] if layers is None else layers
        )

    def get_json(self, url, params=None):
        if url == NifcArchive.ROOT:
            self.calls += 1
            if self.calls <= self.empty:
                return {"currentVersion": 11, "services": []}
            return {"services": [{"name": "National_Incident_Feature_Service_2018", "type": "FeatureServer"}]}
        return {"layers": self.layers}


def test_an_empty_nifc_listing_is_retried_then_an_error_never_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    flaky = Listing(empty=2)
    nifc = NifcArchive(flaky, HttpClient(None), tmp_path)
    assert [kind for _, kind in nifc.event_layers(2018)] == ["line", "polygon"] and flaky.calls == 3
    nifc.perimeter_layers(2018)
    assert flaky.calls == 3  # listed once per run
    with pytest.raises(RuntimeError, match="came back empty"):
        NifcArchive(Listing(empty=99), HttpClient(None), tmp_path).event_layers(2018)
    with pytest.raises(RuntimeError, match="listed no layers"):
        NifcArchive(Listing(empty=0, layers=[]), HttpClient(None), tmp_path).event_layers(2018)
