"""File formats: gzipped GeoJSON is byte-stable, time series keep packed values, and sinusoidal tiles join pixel for
pixel with their fill, grid and attributes."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from fah.core.geometry import BoundingBox, PixelWindow, SinusoidalGrid
from fah.storage.attrs import NetcdfAttrs
from fah.storage.geojson import GzipGeoJson
from fah.storage.ncseries import NcSeriesWriter
from fah.storage.sinusoidal import SinusoidalMosaic

FEATS = [{"type": "Feature", "properties": {"name": "Camp", "acres": 153336}, "geometry": None}]


def test_geojson_is_gzipped_with_the_same_bytes_every_time(tmp_path):
    a, b = tmp_path / "a.geojson.gz", tmp_path / "b.geojson.gz"
    GzipGeoJson.write(a, FEATS)
    GzipGeoJson.write(b, FEATS)
    assert a.read_bytes() == b.read_bytes() and a.read_bytes()[:2] == b"\x1f\x8b"
    assert json.load(gzip.open(a)) == {"type": "FeatureCollection", "features": FEATS}
    plain = tmp_path / "c.geojson"
    plain.write_text(json.dumps({"type": "FeatureCollection", "features": FEATS}))
    GzipGeoJson.compress(plain, tmp_path / "c.geojson.gz")
    assert gzip.open(tmp_path / "c.geojson.gz").read() == plain.read_bytes()


def test_ncseries_keeps_packed_values(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    lat, lon = np.zeros((2, 3)), np.ones((2, 3))
    raw = np.array([[1, 2, 3], [4, -99, 6]], np.int16)
    with NcSeriesWriter(tmp_path / "t.nc", lat, lon, {"title": "t"}) as s:
        s.append(10, "s3://b/k1", {"Power": (raw, np.int16, {"_FillValue": np.int16(-99), "scale_factor": 0.5})})
        s.append(20, "s3://b/k2", {"Power": (raw + 1, np.int16, {"_FillValue": np.int16(-99), "scale_factor": 0.5})})
    with netCDF4.Dataset(tmp_path / "t.nc") as ds:
        v = ds.variables["Power"]
        v.set_auto_maskandscale(False)
        assert (v[0] == raw).all() and v.scale_factor == 0.5
        v.set_auto_maskandscale(True)
        assert v[0][0, 1] == 1.0 and np.ma.is_masked(v[0][1, 1])
        assert list(ds.variables["time"][:]) == [10, 20] and ds.variables["source"][1] == "s3://b/k2"


def test_sinusoidal_window_holds_the_box_and_tiles_join_pixel_for_pixel():
    pyproj = pytest.importorskip("pyproj")
    pytest.importorskip("netCDF4")
    box = BoundingBox(-121.0, 39.9, -120.8, 40.1)  # straddles 40 N: tiles h08v04 and h08v05
    win = SinusoidalGrid.window(box)
    r0, r1, c0, c1 = win
    assert r0 < 5 * 2400 <= r1 and SinusoidalGrid.tile_part(win, 8, 4) and SinusoidalGrid.tile_part(win, 8, 5)
    assert not SinusoidalGrid.tile_part(win, 9, 5)
    x, y = SinusoidalGrid.coords(win)
    to_ll = pyproj.Transformer.from_crs(SinusoidalGrid.PROJ, 4326, always_xy=True)
    lon, lat = to_ll.transform(*np.meshgrid(x, y))
    assert (
        lat.max() > 40.1 - 0.005
        and lat.min() < 39.9 + 0.005
        and lon.min() < -121.0 + 0.01
        and lon.max() > -120.8 - 0.01
    )

    m = SinusoidalMosaic(win)
    for h, v in ((8, 4), (8, 5)):
        rows = np.add.outer(v * 2400 + np.arange(2400), np.zeros(2400, int)).astype("u2")
        cols = np.add.outer(np.zeros(2400, int), h * 2400 + np.arange(2400)).astype("u2")
        m.put("row", rows, {"_FillValue": np.array([65535], "u2"), "units": b"1"}, h, v)
        m.put("col", cols, {}, h, v)
    got_r, fill, at = m.layers["row"]
    assert (got_r == np.arange(r0, r1)[:, None]).all() and fill == 65535 and at["units"] == "1"
    assert (m.layers["col"][0] == np.arange(c0, c1)[None, :]).all()


def test_sinusoidal_netcdf_keeps_values_fill_and_grid(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    win = PixelWindow(11990, 11994, 19500, 19503)
    m = SinusoidalMosaic(win, nt=2)
    tile = np.arange(2400 * 2400, dtype="i2").reshape(2400, 2400) % 1000
    m.put("NDVI", tile, {"_FillValue": np.int16(-15000), "scale_factor": np.float32(1e-4)}, 8, 4, t=1,
          source_name="500 m 16 days NDVI")  # fmt: skip
    m.write(tmp_path / "v.nc", {"title": "t"}, times_ns=[1, 2])
    with netCDF4.Dataset(tmp_path / "v.nc") as ds:
        v = ds.variables["NDVI"]
        v.set_auto_maskandscale(False)
        assert v.shape == (2, 4, 3) and (v[0] == -15000).all() and v.source_name == "500 m 16 days NDVI"
        assert (v[1] == tile[11990 - 9600 : 11994 - 9600, 19500 - 19200 : 19503 - 19200]).all()
        assert (
            ds.variables["crs"].grid_mapping_name == "sinusoidal"
            and ds.sinusoidal_rows_cols == "11990:11994,19500:19503"
        )
        assert ds.variables["x"][0] == pytest.approx(SinusoidalGrid.X0 + 19500.5 * SinusoidalGrid.TILE_M / 2400)
    name = "HDFEOS/GRIDS/VIIRS_Grid_16Day_VI_500m/Data Fields/500 m 16 days pixel reliability"
    assert NetcdfAttrs.variable_name(name) == "pixel_reliability"


def test_source_attribute_names_netcdf_refuses_are_made_legal(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    attrs = {"Cropland/Natural Vegetation Mosaics": np.uint8(14), "Water Bodies": np.uint8(17), "trailing ": "x",
             "-dash": 1, "Cropland_Natural Vegetation Mosaics": np.uint8(99), "units": "class number"}  # fmt: skip
    got = NetcdfAttrs.clean(attrs)
    assert list(got) == ["Cropland_Natural Vegetation Mosaics", "Water Bodies", "trailing", "_-dash", "units"]
    assert got["Cropland_Natural Vegetation Mosaics"] == 14  # the first of two names that become one stays
    with netCDF4.Dataset(tmp_path / "a.nc", "w") as ds:
        ds.setncatts(got)  # MCD12Q1's class names stopped every landcover fire here
