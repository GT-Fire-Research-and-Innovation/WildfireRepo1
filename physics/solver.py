"""
Differentiable fire-front solver: arrival time of the fire at every cell, from the cells already
burning and a directional spread-rate field.

Method: shortest travel time over a 16-neighbour grid (the minimum-travel-time idea behind
FlamMap MTT). Each step from cell y to neighbour x costs
    distance / ray speed, averaged over the cells the step crosses,
where the ray speed is Rothermel's elliptical spread (ignition at the rear focus). The fixed point
T(x) = min_k [T(x - offset_k) + cost_k(x)] is found by repeated relaxation.

Gradients: a custom autograd function. Forward stores which neighbour each cell's minimum came
from (the fire's "parent" tree). Backward pushes dL/dT down that tree (an adjoint sweep), so
dL/dcost is exact for the chosen paths and memory does not grow with the number of iterations.
"""
import math

import torch

from .rothermel import ray_speed

# 16-neighbour stencil (row, col) offsets: 4 rook, 4 bishop, 8 knight moves
OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1),
           (-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)]
BIG = 1e9


def _shift(t, dr, dc, fill):
    """out[r, c] = t[r - dr, c - dc] (value at the neighbour the step comes FROM)."""
    H, W = t.shape[-2:]
    out = torch.full_like(t, fill)
    rs, re_ = max(dr, 0), H + min(dr, 0)
    cs, ce = max(dc, 0), W + min(dc, 0)
    out[..., rs:re_, cs:ce] = t[..., rs - dr:re_ - dr, cs - dc:ce - dc]
    return out


def step_costs(r_head, ecc, head_x, head_y, dx, min_rate=1e-3):
    """Travel time (minutes) for each of the 16 steps into every cell. Returns (16, H, W).
    r_head in m/min, dx in metres. Step k enters cell x from x - offset_k."""
    costs = []
    for dr, dc in OFFSETS:
        dist = dx * math.hypot(dr, dc)
        ux, uy = dc / math.hypot(dr, dc), -dr / math.hypot(dr, dc)      # east, north (rows run south)
        r_here = ray_speed(r_head, ecc, head_x, head_y, ux, uy)
        slow_here = 1.0 / r_here.clamp_min(min_rate)
        slow_from = _shift(slow_here, dr, dc, 1.0 / min_rate)
        if abs(dr) + abs(dc) == 3:                                       # knight move crosses 2 more cells
            # cells straddled: the rook and bishop neighbours along the step
            sr, sc = int(math.copysign(1, dr)), int(math.copysign(1, dc))
            if abs(dr) == 2:
                m1, m2 = (sr, 0), (sr, sc)
            else:
                m1, m2 = (0, sc), (sr, sc)
            a = _shift(slow_here, dr - m1[0], dc - m1[1], 1.0 / min_rate)
            b = _shift(slow_here, dr - m2[0], dc - m2[1], 1.0 / min_rate)
            slow = (slow_from + a + b + slow_here) / 4
        else:
            slow = (slow_from + slow_here) / 2
        costs.append(dist * slow)
    return torch.stack(costs)


class _MinTravelTime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, costs, t0, max_iter, t_max):
        T = t0.clone()
        parent = torch.full(T.shape, -1, dtype=torch.long, device=T.device)
        for _ in range(max_iter):
            cand = torch.stack([_shift(T, dr, dc, BIG) for dr, dc in OFFSETS]) + costs
            best, arg = cand.min(0)
            better = best < T - 1e-7
            if not better.any():
                break
            T = torch.where(better, best, T)
            parent = torch.where(better, arg, parent)
            if t_max is not None:          # cells beyond the horizon never feed earlier cells
                far = T > t_max * 4
                T = torch.where(far, torch.full_like(T, BIG), T)
                parent = torch.where(far, torch.full_like(parent, -1), parent)
        ctx.save_for_backward(parent)
        ctx.max_iter = max_iter
        return T

    @staticmethod
    def backward(ctx, grad_T):
        (parent,) = ctx.saved_tensors
        # total adjoint: lam_tot(y) = g(y) + sum over children x of lam_tot(x)
        tot = grad_T.clone()
        for _ in range(ctx.max_iter):
            new = grad_T.clone()
            for k, (dr, dc) in enumerate(OFFSETS):
                child = torch.where(parent == k, tot, torch.zeros_like(tot))
                new = new + _shift(child, -dr, -dc, 0.0)          # send child's adjoint back to its parent
            if torch.allclose(new, tot):
                tot = new
                break
            tot = new
        g_cost = torch.stack([torch.where(parent == k, tot, torch.zeros_like(tot)) for k in range(len(OFFSETS))])
        return g_cost, None, None, None


def arrival_time(costs, burning, max_iter=400, t_max=None):
    """costs (16, H, W) from step_costs; burning (H, W) bool, cells on fire at t = 0.
    Returns arrival time in minutes (BIG where unreached)."""
    t0 = torch.where(burning, torch.zeros_like(costs[0]), torch.full_like(costs[0], BIG))
    return _MinTravelTime.apply(costs, t0, max_iter, t_max)


def burn_probability(T, horizon_min, softness_min=60.0):
    """Smooth 'burned by the horizon' map from arrival times, for a BCE loss against labels."""
    return torch.sigmoid((horizon_min - T.clamp(max=horizon_min * 10)) / softness_min)
