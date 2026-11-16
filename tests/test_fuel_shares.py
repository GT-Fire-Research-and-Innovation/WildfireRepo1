"""Fuel shares per cell: the 30 m summary (pipeline) and the loader option that uses it."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))
import build_multiyear as BM  # noqa: E402
import data as D  # noqa: E402

H = W = 64
DX, DY = 0.0017, -0.00135
G = dict(fire_id="F1", year=2024, height=H, width=W, res_m=150.0, name="t", transform=[DX, 0, -120.0, 0, DY, 40.0])


def fine(k=5):
    """A 30 m-like raster, k x k pixels per cell: timber everywhere, except cell (10, 10) is 60 % grass / 40 % timber
    and a one-pixel-wide road (non-burnable) runs through row 20."""
    a = np.zeros((5, H * k, W * k), np.int16)
    a[0] = 183                                   # TL3
    a[1], a[2], a[3], a[4] = 60, 200, 30, 10     # canopy cover, height, base height, bulk density
    a[0, 10 * k:10 * k + 3, 10 * k:11 * k] = 102     # GR2 on 3 of the 5 fine rows of that cell: 60 %
    a[1, 10 * k:10 * k + 3, 10 * k:11 * k] = 0
    a[0, 20 * k + 2, :] = 91                     # road: one fine pixel wide
    return a, Affine(DX / k, 0, -120.0, 0, DY / k, 40.0)


def test_shares_majority_and_means():
    a, tr = fine()
    out = BM.fuel_shares([(a, tr)], G)
    fam = [f[0] for f in BM.FUEL_FAMILIES]
    frac = out["frac"].astype(float) / 255
    assert out["n"].min() == 25
    assert np.isclose(frac[fam.index("GR"), 10, 10], 0.6, atol=0.01) and np.isclose(frac[fam.index("TL"), 10, 10], 0.4, atol=0.01)
    assert out["fbfm40"][10, 10] == 102 and out["fbfm40"][11, 11] == 183          # most common model
    assert np.isclose(out["cc"][10, 10], 60 * 0.4, atol=1)                         # mean canopy cover, not a picked pixel
    assert np.allclose(frac[fam.index("NB"), 20], 0.2, atol=0.01)                  # the road is a fifth of its cells, not all or none
    assert out["fbfm40"][20, 5] == 183
    assert np.allclose(frac.sum(0), 1.0, atol=0.02)


def test_pieces_add_up_and_empty_cells_fall_back():
    a, tr = fine()
    k = a.shape[1] // H
    left, right = a[:, :, : 30 * k], a[:, :, 30 * k: 60 * k]                       # two tiles; columns 60-63 not covered
    tr_r = Affine(tr.a, 0, -120.0 + 30 * DX, 0, tr.e, 40.0)
    old = dict(fbfm40=np.full((H, W), 122, np.int16), cc=np.full((H, W), 5, np.int16), ch=np.zeros((H, W), np.int16),
               cbh=np.zeros((H, W), np.int16), cbd=np.zeros((H, W), np.int16))
    out = BM.fuel_shares([(left, tr), (right, tr_r)], G, old)
    whole = BM.fuel_shares([(a, tr)], G)
    assert np.array_equal(out["frac"][:, :, :60], whole["frac"][:, :, :60])
    assert (out["fbfm40"][:, 60:] == 122).all() and (out["cc"][:, 60:] == 5).all() and (out["n"][:, 60:] == 0).all()
    fam = [f[0] for f in BM.FUEL_FAMILIES]
    assert (out["frac"][fam.index("GS"), :, 60:] == 255).all()


def test_families_match_the_model():
    assert BM.FUEL_FAMILIES == D.FUEL_FAMILIES


def make_fire(tmp):
    d = tmp / "F1"
    d.mkdir()
    (d / "grid.json").write_text(json.dumps(G))
    arrival = -np.ones((H, W), np.int16)
    arrival[28:36, 28:36] = 0
    arrival[24:28, 28:36] = 1
    times = pd.to_datetime(["2024-08-10 18:00", "2024-08-11 18:00"], utc=True)
    np.savez(d / "labels.npz", arrival=arrival, other=np.zeros((H, W), bool), times=times.as_unit("ns").asi8)
    z = np.zeros((H, W), np.float32)
    np.savez(d / "terrain.npz", elev=z + 1000, slope=z, aspect=z, tpi_500m=z, tpi_2km=z, rough=z)
    days = pd.date_range("2024-08-01", "2024-08-13")
    np.savez(d / "weather.npz", dates=np.array([str(x.date()) for x in days]), gm=np.full((len(days), len(D.GM), 2, 2), 10.0, np.float32),
             gm_vars=np.array(D.GM), gm_row=np.zeros(H, np.int16), gm_col=np.zeros(W, np.int16))
    i16 = lambda v: np.full((H, W), v, np.int16)  # noqa: E731
    np.savez(d / "landfire.npz", fbfm40=i16(102), cc=i16(0), ch=i16(0), cbh=i16(0), cbd=i16(0), missing=np.array(False))  # picked pixel: grass
    a, tr = fine()
    np.savez(d / "fuelfrac.npz", **BM.fuel_shares([(a, tr)], G), missing=np.array(False))
    return D.Fire(d)


@pytest.fixture
def settings():
    keep = D.FUEL_FRAC
    D.FF_MISSING.clear()
    D._CACHE.clear()
    yield
    D.FUEL_FRAC = keep
    D._CACHE.clear()


def test_loader_uses_shares_only_when_asked(tmp_path, settings):
    f = make_fire(tmp_path)
    fam = [x[0] for x in D.FUEL_FAMILIES]
    D.FUEL_FRAC = False
    off = f.sample(0, 24.0, (0, H, 0, W))[0]["fuels"]
    assert (off[fam.index("GR")] == 1).all() and (off[fam.index("TL")] == 0).all()        # the picked pixel said grass everywhere
    D.FUEL_FRAC = True
    D._CACHE.clear()
    on = f.sample(0, 24.0, (0, H, 0, W))[0]["fuels"]
    assert np.isclose(on[fam.index("TL"), 5, 5], 1.0) and np.isclose(on[fam.index("GR"), 10, 10], 0.6, atol=0.01)
    assert np.isclose(on[7, 5, 5], 0.60, atol=0.01)                                        # mean canopy cover from the fine pixels
    assert on.shape == off.shape and not D.FF_MISSING


def test_fire_without_shares_keeps_the_picked_pixel_and_is_recorded(tmp_path, settings):
    f = make_fire(tmp_path)
    (f.d / "fuelfrac.npz").unlink()
    D.FUEL_FRAC = True
    fam = [x[0] for x in D.FUEL_FAMILIES]
    assert (f.sample(0, 24.0, (0, H, 0, W))[0]["fuels"][fam.index("GR")] == 1).all()
    assert D.FF_MISSING == {"F1"}
