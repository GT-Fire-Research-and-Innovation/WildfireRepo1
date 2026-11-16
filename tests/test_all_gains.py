"""Every confirmed gain switched on together (fuel moisture, late start, terrain wind, fuel shares) on one small
synthetic fire: the loader builds the full input stack and the network runs a training step on it."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
import data as D  # noqa: E402
import train_hybrid as TH  # noqa: E402

H = W = 64
FLAGS = ("FM_ON", "FUEL_V2", "TEMPORAL", "PHYSICS_V2", "WIND_TERRAIN", "FUEL_FRAC", "HRRR_ROTATE")


def make_fire(tmp):
    d = tmp / "F1"
    d.mkdir()
    (d / "grid.json").write_text(json.dumps(dict(fire_id="F1", year=2024, height=H, width=W, res_m=150.0, name="t",
                                                 transform=[0.0017, 0, -120.0, 0, -0.00135, 40.0])))
    arrival = -np.ones((H, W), np.int16)
    arrival[28:36, 28:36] = 0
    arrival[22:28, 28:40] = 1
    times = pd.to_datetime(["2024-08-10 18:00", "2024-08-11 18:00"], utc=True)
    np.savez(d / "labels.npz", arrival=arrival, other=np.zeros((H, W), bool), times=times.as_unit("ns").asi8)
    z = np.zeros((H, W), np.float32)
    np.savez(d / "terrain.npz", elev=z + 1000, slope=z + 8, aspect=z + 180, tpi_500m=z, tpi_2km=z, rough=z)
    days = pd.date_range("2024-08-01", "2024-08-13")
    np.savez(d / "weather.npz", dates=np.array([str(x.date()) for x in days]), gm=np.full((len(days), len(D.GM), 2, 2), 10.0, np.float32),
             gm_vars=np.array(D.GM), gm_row=np.zeros(H, np.int16), gm_col=np.zeros(W, np.int16))
    np.savez(d / "viirs.npz", dates=np.array([str(x.date()) for x in days]), af_day=np.zeros((len(days), H, W), np.uint8),
             af_night=np.zeros((len(days), H, W), np.uint8), frp_max=np.zeros((len(days), H, W), np.float16))
    hours = pd.date_range("2024-08-06 00:00", "2024-08-11 21:00", freq="3h", tz="UTC")
    t_ns = hours.as_unit("ns").asi8
    fields = np.zeros((len(hours), 4, 2, 2), np.float32)
    fields[:, 0], fields[:, 1], fields[:, 2], fields[:, 3] = 9.0, 5.0, 2.0, 25.0
    idx = np.zeros((H, W), np.int32)
    np.savez(d / "hrrr.npz", times=t_ns, fields=fields, idx=idx)
    np.savez(d / "hrrr2.npz", times=t_ns, t2=np.full((len(hours), 2, 2), 30.0, np.float32), dpt=np.full((len(hours), 2, 2), 5.0, np.float32),
             dswrf=np.full((len(hours), 2, 2), 400.0, np.float32), apcp3=np.zeros((len(hours), 2, 2), np.float32), idx=idx)
    ends = pd.to_datetime(["2024-06-20", "2024-07-10", "2024-08-01"], utc=True).as_unit("ns").asi8
    np.savez(d / "green.npz", ends=ends, ndvi=np.full((3, H, W), 4000, np.int16), evi=np.full((3, H, W), 2500, np.int16), missing=np.array(False))
    i16 = lambda v: np.full((H, W), v, np.int16)  # noqa: E731
    np.savez(d / "landfire.npz", fbfm40=i16(102), cc=i16(0), ch=i16(0), cbh=i16(0), cbd=i16(0), missing=np.array(False))
    frac = np.zeros((7, H, W), np.uint8)
    frac[5], frac[1] = 179, 76                                            # 70 % timber litter, 30 % grass
    np.savez(d / "fuelfrac.npz", frac=frac, fbfm40=i16(183), cc=i16(55), ch=i16(180), cbh=i16(20), cbd=i16(9), missing=np.array(False))
    r = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)[:, :, None, None] * np.ones((1, 1, H, W), np.float32)
    r[0, 0, :, 32:] = 1.4                                                 # east half: faster
    np.savez(d / "terrain_wind.npz", resp=r.astype(np.float16), valid=np.ones((H, W), bool))
    return D.Fire(d)


@pytest.fixture
def all_on():
    keep = {k: getattr(D, k) for k in FLAGS}
    D._CACHE.clear()
    D.FM_ON = D.FUEL_V2 = D.TEMPORAL = D.PHYSICS_V2 = D.FUEL_FRAC = True
    D.WIND_TERRAIN, D.HRRR_ROTATE = D.wind_file(1), False
    yield
    for k, v in keep.items():
        setattr(D, k, v)
    D._CACHE.clear()


def test_all_inputs_together_and_one_training_step(tmp_path, all_on):
    f = make_fire(tmp_path)
    TH.set_grid(f.res_m / 1000)
    feats, phys, y, m, b0 = f.sample(0, 24.0, (0, H, 0, W), lag_h=6.0)
    groups = TH.groups_used(set(), v2=True, t3=True, origin=True, fm=True)
    for g in groups:
        assert feats[g].shape[0] == D.GROUPS[g], g
        assert np.isfinite(feats[g]).all(), g
    assert np.isclose(feats["origin"].max(), 6.0 / 48)                    # the perimeter is 6 h old at the start
    assert np.isclose(feats["fuels"][5, 5, 5], 179 / 255, atol=1e-3)      # fuel shares, not the picked pixel (grass)
    spd = feats["hrrr"][2] * 8
    assert spd[:, 32:].mean() > 1.25 * spd[:, :32].mean()                 # terrain wind applied
    assert feats["fmoist"].any() and feats["green"][3].all()              # fuel moisture and greenness present
    assert phys is not None                                               # the Rothermel fields were built too
    for k, v in (phys.items() if isinstance(phys, dict) else []):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            assert np.isfinite(v).all(), k

    x = torch.from_numpy(TH.pad8(D.stack(feats, groups))).float()[None]
    model = TH.build_model("unet3", x.shape[1], 16, "cpu", dx=f.res_m, arch={"time": True, "attn": False})
    out = model(x)
    logit = out[0] if isinstance(out, (tuple, list)) else out
    assert logit.shape[-2:] == x.shape[-2:] and torch.isfinite(logit).all()
    target = torch.from_numpy(TH.pad8(y[None].astype(np.float32)))[None]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.reshape(target.shape), target)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
