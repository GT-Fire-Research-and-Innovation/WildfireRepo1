"""Physics checks: flat/no wind gives a circle, wind gives the Rothermel ellipse, slope speeds
upslope spread, non-burnable cells block, and gradients reach the spread-rate field.
Run: python -m pytest tests/test_physics.py -q   (or python tests/test_physics.py)"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from physics import rothermel as R  # noqa: E402
from physics import solver as S  # noqa: E402

N, DX = 121, 30.0
C = N // 2


def grid(code=102, wind=0.0, wind_from=270.0, slope=0.0, aspect=0.0, m1=0.06):
    f = lambda v: torch.full((N, N), float(v), dtype=torch.float64)  # noqa: E731
    idx = R.fuel_index(torch.full((N, N), code))
    return R.spread(idx, f(m1), f(m1 + .01), f(m1 + .02), f(0.6), f(0.9), f(wind), f(wind_from), f(slope), f(aspect))


def solve(out, burning=None):
    costs = S.step_costs(out["r_head"], out["ecc"], out["head_x"], out["head_y"], DX)
    if burning is None:
        burning = torch.zeros((N, N), dtype=torch.bool)
        burning[C, C] = True
    return S.arrival_time(costs, burning)


def extent(T, t):
    """Distance (m) the fire reached by time t along +x (east), -x, +y (north) and -y."""
    b = T <= t
    row, col = b[C], b[:, C]
    east = (row[C:].nonzero().max().item()) * DX
    west = (row[:C + 1].flip(0).nonzero().max().item()) * DX
    north = (col[:C + 1].flip(0).nonzero().max().item()) * DX
    south = (col[C:].nonzero().max().item()) * DX
    return east, west, north, south


def test_flat_circle():
    out = grid()
    r0 = out["r0"][0, 0].item()
    T = solve(out)
    t = 50 * DX / r0                       # time to travel 50 cells
    e, w, n, s = extent(T, t)
    for d in (e, w, n, s):
        assert abs(d - 50 * DX) <= DX, (d, 50 * DX)
    # diagonal: arrival at (C+30, C+30) ~ distance / r0 (16-neighbour error < 3 %)
    d = math.hypot(30, 30) * DX
    assert abs(T[C + 30, C + 30].item() - d / r0) / (d / r0) < 0.03


def test_wind_ellipse():
    out = grid(wind=6.0, wind_from=270.0)   # wind from the west -> head fire runs east
    rh, ecc, lb = out["r_head"][0, 0].item(), out["ecc"][0, 0].item(), out["lb"][0, 0].item()
    T = solve(out)
    t = 50 * DX / rh
    e, w, n, s = extent(T, t)
    assert abs(e - 50 * DX) <= DX
    back = rh * (1 - ecc) / (1 + ecc) * t
    assert abs(w - back) <= DX + 0.05 * back
    assert abs(n - s) <= DX
    # total length / width ~ L/B (width measured at the ellipse centre)
    length = e + w
    centre = C + round((e - w) / 2 / DX)
    col = (T[:, centre] <= t).nonzero()
    width = (col.max() - col.min()).item() * DX
    assert abs(length / width - lb) / lb < 0.12, (length / width, lb)


def test_slope_upslope_faster():
    out = grid(slope=0.4, aspect=270.0)     # faces west -> upslope is east
    T = solve(out)
    t = 40 * DX / out["r_head"][0, 0].item()
    e, w, n, s = extent(T, t)
    assert e > 1.5 * w and abs(e - 40 * DX) <= DX


def test_barrier_blocks():
    codes = torch.full((N, N), 102)
    codes[:, C + 10] = 98                   # a column of water, non-burnable
    f = lambda v: torch.full((N, N), float(v), dtype=torch.float64)  # noqa: E731
    out = R.spread(R.fuel_index(codes), f(.06), f(.07), f(.08), f(.6), f(.9), f(0), f(0), f(0), f(0))
    T = solve(out)
    t = 40 * DX / out["r0"][0, 0].item()
    assert (T[:, C + 11:] <= t).sum() == 0


def test_gradients_flow():
    out = grid(wind=3.0)
    rh = out["r_head"].clone().requires_grad_(True)
    costs = S.step_costs(rh, out["ecc"], out["head_x"], out["head_y"], DX)
    burning = torch.zeros((N, N), dtype=torch.bool)
    burning[C, C] = True
    T = S.arrival_time(costs, burning)
    loss = T[C, C + 30]
    loss.backward()
    g = rh.grad
    assert torch.isfinite(g).all() and g.sum() < 0          # faster spread -> earlier arrival
    # finite-difference check: scale all rates by 1.01 -> arrival time scales by 1/1.01
    T2 = S.arrival_time(S.step_costs(rh.detach() * 1.01, out["ecc"], out["head_x"], out["head_y"], DX), burning)
    fd = (T2[C, C + 30] - T[C, C + 30]).item()
    an = (g * rh.detach() * 0.01).sum().item()
    assert abs(fd - an) / abs(fd) < 0.05, (fd, an)


def test_moisture_gradient():
    m1 = torch.full((8, 8), 0.06, dtype=torch.float64, requires_grad=True)
    f = lambda v: torch.full((8, 8), float(v), dtype=torch.float64)  # noqa: E731
    out = R.spread(R.fuel_index(torch.full((8, 8), 122)), m1, f(.07), f(.08), f(.6), f(.9), f(4), f(0), f(.1), f(0))
    out["r_head"].sum().backward()
    assert torch.isfinite(m1.grad).all() and (m1.grad < 0).all()   # wetter fuel -> slower


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
