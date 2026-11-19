"""Secrets never reach published text; pixel windows on rasters and swaths, and the M-to-I swath geometry."""

from __future__ import annotations

import json

import numpy as np

from fah.core.geometry import BoundingBox, PixelWindow
from fah.core.secrets import Secrets


def test_redact_removes_map_key_and_email():
    u = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/abc123SECRET/VIIRS_SNPP_SP/-121.9,39.5,-121.2,40.0/5/2018-11-02"
    assert "abc123SECRET" not in Secrets.redact(u) and "{MAP_KEY}" in Secrets.redact(u)
    v = Secrets.redact(
        "https://lfps.usgs.gov/api/job/submit?Email=me@x.org&Layer_List=LF2016_CC&Area_of_Interest=1 2 3 4"
    )
    assert "me@x.org" not in v and "Layer_List=LF2016_CC" in v


def test_redact_removes_earthdata_token_and_signed_url_parts(monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "eyJhbGciOiJSUzI1NiJ9.payload.sig")
    t = Secrets.redact(
        "401 for https://data.lpdaac.earthdatacloud.nasa.gov/x.h5 with Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig"
    )
    assert "payload" not in t and "{EARTHDATA_TOKEN}" in t
    s = Secrets.redact(
        "https://acct.blob.core.windows.net/c/x.tif?st=2026-11-17&se=2026-11-18&sp=rl&sv=2024&sr=c&sig=AbC%2Bd"
    )
    assert s == "https://acct.blob.core.windows.net/c/x.tif"
    a = Secrets.redact(
        "https://bucket.s3.us-west-2.amazonaws.com/f.nc?A-userid=me&X-Amz-Credential=K&X-Amz-Signature=S"
    )
    assert "X-Amz" not in a and "A-userid=me" in a


def test_earth_engine_key_never_reaches_the_manifest(monkeypatch):
    key = json.dumps(
        {
            "type": "service_account",
            "project_id": "p",
            "private_key_id": "abc123def456",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----\n",
            "client_email": "grab@p.iam.gserviceaccount.com",
        }
    )
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT_KEY", key)
    out = Secrets.redact(f"failed with {key}")
    assert "MIIEvQ" not in out and "abc123def456" not in out
    loose = Secrets.redact(
        '{"private_key_id": "zz99zz", "private_key": "-----BEGIN PRIVATE KEY-----\\nQQQ\\n-----END PRIVATE KEY-----"}'
    )
    assert "zz99zz" not in loose and "QQQ" not in loose


def test_pixel_window_of_a_raster():
    from affine import Affine

    t = Affine(10, 0, 500000, 0, -10, 4400000)
    assert PixelWindow.covering(t, 1000, 1000, (500015, 4399000, 500101, 4399985)) == (1, 100, 1, 11)
    assert PixelWindow.covering(t, 1000, 1000, (400000, 0, 400100, 10)) is None


def test_swath_window_and_m_to_i_geometry():
    lat, lon = np.meshgrid(np.linspace(41, 39, 20), np.linspace(-122, -121, 30), indexing="ij")
    w = PixelWindow.of_points(lat, lon, BoundingBox(-121.6, 39.8, -121.4, 40.2))
    rows, cols = np.where((lat >= 39.8) & (lat <= 40.2) & (lon >= -121.6) & (lon <= -121.4))
    assert w == (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)
    assert PixelWindow.of_points(lat, lon, BoundingBox(0, 0, 1, 1)) is None
    assert PixelWindow(5, 9, 10, 12).m_to_i((40, 60)) == (8, 20, 18, 26)
    assert PixelWindow(0, 20, 0, 30).m_to_i((40, 60)) == (0, 40, 0, 60)
