"""Loader checks on a tiny synthetic fire: inputs must not depend on the crop window, and the suppression
layers must be the ones that existed at the forecast start."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402

H = W = 96


def make_fire(tmp):
    d = tmp / "F1"
    d.mkdir()
    (d / "grid.json").write_text(json.dumps(dict(fire_id="F1", year=2019, height=H, width=W, res_m=150.0, name="t",
                                                 transform=[0.0017, 0, -120.0, 0, -0.00135, 40.0])))
    arrival = -np.ones((H, W), np.int16)
    arrival[40:50, 40:50] = 0
    arrival[36:40, 40:50] = 1
    # 2019-08-10 18:00 UTC = 10:00 local at lon -120; second perimeter one day later
    times = pd.to_datetime(["2019-08-10 18:00", "2019-08-11 18:00"], utc=True)
    np.savez(d / "labels.npz", arrival=arrival, other=np.zeros((H, W), bool), times=times.as_unit("ns").asi8)
    rng = np.random.default_rng(0)
    elev = np.add.outer(np.arange(H) * 10.0, np.arange(W) * 3.0).astype(np.float32) + 1000
    z = np.zeros((H, W), np.float32)
    np.savez(d / "terrain.npz", elev=elev, slope=z + 10, aspect=z + 180, tpi_500m=z, tpi_2km=z, rough=z)
    days = pd.date_range("2019-08-01", "2019-08-13")
    gm = rng.normal(10, 1, (len(days), len(D.GM), 2, 2)).astype(np.float32)
    np.savez(d / "weather.npz", dates=np.array([str(x.date()) for x in days]), gm=gm, gm_vars=np.array(D.GM),
             gm_row=np.zeros(H, np.int16), gm_col=np.zeros(W, np.int16))
    np.savez(d / "viirs.npz", dates=np.array([str(x.date()) for x in days]), af_day=np.zeros((len(days), H, W), np.uint8),
             af_night=np.zeros((len(days), H, W), np.uint8), frp_max=np.zeros((len(days), H, W), np.float16))
    sdates = ["2019-08-09", "2019-08-10"]
    lay = np.zeros((2, 7, H, W), np.uint8)
    lay[0, 0, 30, :] = 1            # hand line that existed on the morning of the 10th (burn day 08-09's map)
    lay[1, 6, 36, 40:50] = 1        # uncontrolled edge as of 06:00 on the 11th: the fire's NEXT position
    np.savez(d / "suppression.npz", dates=np.array(sdates), layers=np.packbits(lay, axis=1), n_channels=np.array(7),
             has_spatial=np.array(True), has_ics=np.array(False))
    return D.Fire(d)


def test_inputs_do_not_depend_on_the_window(tmp_path):
    f = make_fire(tmp_path)
    full = f.sample(0, 24.0, (0, H, 0, W))[0]
    crop = f.sample(0, 24.0, (24, 56, 44, 76))[0]             # holds only part of the fire
    for g, ch in (("fire", 6), ("fire", 7), ("fire", 9), ("terrain", 0)):
        assert np.allclose(full[g][ch][24:56, 44:76], crop[g][ch]), (g, ch)


def test_suppression_is_the_map_at_the_forecast_start(tmp_path):
    f = make_fire(tmp_path)
    sup = f.sample(0, 24.0, (0, H, 0, W))[0]["suppression"]
    assert sup[0, 30].all()                    # yesterday's line is visible
    assert sup[6].sum() == 0                   # the edge mapped the next morning is not
