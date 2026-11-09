"""Hybrid model learns a known spread-rate correction through the solver (synthetic)."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.hybrid import Hybrid, loss_fn  # noqa: E402
from physics import rothermel as R  # noqa: E402
from physics import solver as S  # noqa: E402


def test_learns_correction(steps=150):
    torch.manual_seed(0)
    N = 48
    f = lambda v: torch.full((N, N), float(v))  # noqa: E731
    phys = R.spread(R.fuel_index(torch.full((N, N), 102)), f(.06), f(.07), f(.08), f(.6), f(.9),
                    f(3), f(270), f(0), f(0))
    # truth: east half spreads 3x faster than Rothermel says
    zone = torch.zeros(N, N)
    zone[:, N // 2:] = 1
    true_R = phys["r_head"] * (1 + 2 * zone)
    burning = torch.zeros(N, N, dtype=torch.bool)
    burning[N // 2 - 2:N // 2 + 2, N // 2 - 8:N // 2 - 4] = True
    T_true = S.arrival_time(S.step_costs(true_R, phys["ecc"], phys["head_x"], phys["head_y"], 150.0), burning)
    horizon = 480.0
    y = (T_true <= horizon) & ~burning
    mask = (~burning).float()

    net = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 2, 3, padding=1))
    model = Hybrid(net, h_eff_init=horizon, tau_init=30.0)
    model.log_h.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)
    x = torch.stack([zone, torch.ones(N, N)])[None]
    P = {k: v[None] for k, v in phys.items() if k in ("r_head", "ecc", "head_x", "head_y")}
    for i in range(steps):
        logits, T, c = model(x, P, burning[None], torch.tensor([1.0]))
        loss = loss_fn(logits, y[None], mask[None], pos_weight=1.0, corr=c)
        opt.zero_grad()
        loss.backward()
        opt.step()
    c = c.detach()[0]
    east, west = c[:, N // 2 + 2:].mean().item(), c[:, :N // 2 - 2].mean().item()
    print(f"learned log-correction east {east:.2f} (true {torch.log(torch.tensor(3.)):.2f}), west {west:.2f} (true 0)")
    assert east - west > 0.6


if __name__ == "__main__":
    test_learns_correction()
    print("ok")
