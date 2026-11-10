"""Crown fire and spotting (physics/crown_spot.py) and the hybrid2 probability combination."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from physics import crown_spot as CS  # noqa: E402
from hybrid import combine_logits  # noqa: E402


def test_active_crown_rate_matches_cruz_2005_example():
    # Cruz et al. (2005) worked example: EFFM 4 %, CBD 0.2 kg/m3, U10 50 km/h -> ~137-139 m/min (active)
    one = np.ones((1, 1))
    p, cros, r = CS.crown_fire(one * 1.0, one * 2.0, one * 0.2, one * 50, one * 50, one * 4, one * 3, one.astype(bool))
    assert 130 < cros[0, 0] < 145
    assert p[0, 0] > 0.99                     # dry, windy, low canopy base: crowning is near certain
    assert abs(r[0, 0] - cros[0, 0]) < 2      # expected spread ~ crown spread


def test_no_crowning_without_canopy_or_with_high_base():
    one = np.ones((1, 1))
    p, _, r = CS.crown_fire(one * 2.0, one * 2.0, one * 0.2, one * 5, one * 20, one * 8, one * 3, one.astype(bool))
    assert p[0, 0] == 0 and r[0, 0] == 2.0    # canopy cover 5 % -> surface fire only
    p, _, _ = CS.crown_fire(one * 2.0, one * 12.0, one * 0.2, one * 50, one * 5, one * 15, one * 0.5, one.astype(bool))
    assert p[0, 0] < 0.01                     # calm, moist, tall canopy base, little surface fuel


def test_embers_land_downwind():
    b = np.zeros((101, 101), bool)
    b[45:56, 45:56] = True
    e = CS.ember_field(b, np.full(b.shape, 8000.0), np.full(b.shape, 0.8), np.full(b.shape, 10.0), 10.0, 0.0, 150.0)
    assert e[50, 60:].sum() > 20 * e[50, :40].sum()        # wind toward the east: embers east of the fire
    assert e[b].sum() == 0


def test_combine_logits_is_probability_union():
    a, b = torch.tensor([-2.0, 0.0, 3.0]), torch.tensor([-1.0, -5.0, 1.0])
    pa, pb = torch.sigmoid(a), torch.sigmoid(b)
    want = 1 - (1 - pa) * (1 - pb)
    assert torch.allclose(torch.sigmoid(combine_logits(a, b)), want, atol=1e-5)
    assert torch.isfinite(combine_logits(torch.tensor([-60.0]), torch.tensor([-60.0]))).all()
