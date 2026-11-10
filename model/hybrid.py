"""
Hybrid physics + learning model (the plan's core design).

    R(x)     = R_rothermel(x) * exp(c(x)) + r_floor * sigmoid(f(x))
    T(x)     = arrival_time(R, heading, eccentricity, burning at day k)      (physics/solver.py)
    P(x)     = sigmoid((H_eff * gap - T(x)) / tau)                            burned by day k+1

The network (any 2-D encoder-decoder; the U-Net in pipeline/train.py works) sees the same inputs as
the plain U-Net plus the physics fields, and outputs two channels per cell:
  c  log-correction of the Rothermel spread rate (bounded by tanh to +/- MAX_LOG_CORR)
  f  logit of an additive spread floor, so spread is possible where LANDFIRE says non-burnable
     but fire still moved (urban fuels, mapping errors, spotting)
H_eff (effective burning minutes per day) and tau (softness) are learned scalars.

Gradients flow from the BCE loss through P, through the solver's exact adjoint, into R, and into
the network. Head direction and eccentricity come from Rothermel (wind + slope) and are not
corrected in v1.
"""
import math

import torch
import torch.nn as nn

from physics import solver as S

MAX_LOG_CORR = 3.0          # correction limited to a factor of e^3 = 20 either way


class Hybrid(nn.Module):
    def __init__(self, net, dx=150.0, r_floor=2.0, h_eff_init=480.0, tau_init=60.0, max_iter=400):
        """net: module mapping (B, C, H, W) -> (B, 2, H, W). r_floor in m/min."""
        super().__init__()
        self.net = net
        self.dx, self.r_floor, self.max_iter = dx, r_floor, max_iter
        self.log_h = nn.Parameter(torch.tensor(math.log(h_eff_init)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    def rate(self, x, r_roth):
        out = self.net(x)
        c = MAX_LOG_CORR * torch.tanh(out[:, 0] / MAX_LOG_CORR)
        floor = self.r_floor * torch.sigmoid(out[:, 1] - 4.0)          # starts near zero
        return r_roth * torch.exp(c) + floor, c

    def forward(self, x, phys, burning, gap):
        """x (B, C, H, W) inputs; phys dict of (B, H, W): r_head, ecc, head_x, head_y (Rothermel);
        burning (B, H, W) bool; gap (B,) days. Returns burn logits (B, H, W), arrival T, correction c."""
        R, c = self.rate(x, phys["r_head"])
        logits, Ts = [], []
        h_eff, tau = self.log_h.exp(), self.log_tau.exp()
        for b in range(x.shape[0]):
            costs = S.step_costs(R[b], phys["ecc"][b], phys["head_x"][b], phys["head_y"][b], self.dx)
            horizon = h_eff * gap[b]
            T = S.arrival_time(costs, burning[b], max_iter=self.max_iter, t_max=float(horizon.detach()) * 1.5)
            Ts.append(T)
            logits.append((horizon - T.clamp(max=1e5)) / tau)
        return torch.stack(logits), torch.stack(Ts), c


def loss_fn(logits, y, mask, pos_weight=8.0, corr=None, corr_weight=1e-3):
    """Masked BCE (+ a small penalty keeping the correction near zero, i.e. near physics)."""
    pw = torch.tensor(pos_weight, device=logits.device)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pw, reduction="none")
    loss = (bce * mask).sum() / mask.sum().clamp_min(1)
    if corr is not None:
        loss = loss + corr_weight * (corr ** 2).mean()
    return loss


def combine_logits(a, b):
    """Logit of P = 1 - (1 - sigmoid(a)) (1 - sigmoid(b)): burned by spread OR by embers."""
    l1p = -nn.functional.softplus(a) - nn.functional.softplus(b)          # log(1 - P)
    lp = torch.log(-torch.expm1(l1p.clamp(max=-1e-7)))                    # log(P)
    return lp - l1p


class Hybrid2(Hybrid):
    """Physics v2: crown fire and spotting (physics/crown_spot.py).

        R      = (R_surface + w (R_crown - R_surface)) * exp(c) + floor       w learned, starts at 0.5
        spread = sigmoid((H_eff * gap - T) / tau)                             as in Hybrid
        spot   = sigmoid(alpha * log(embers * gap + 1e-4) + beta + g)         alpha, beta learned
        P      = 1 - (1 - spread) (1 - spot)

    The network outputs three channels: c (spread correction), f (floor) and g (spotting correction).
    w, alpha and beta are scalars, so the physics keeps its shape and the data sets the strengths."""

    def __init__(self, net, dx=150.0, r_floor=2.0, h_eff_init=480.0, tau_init=60.0, max_iter=400):
        super().__init__(net, dx, r_floor, h_eff_init, tau_init, max_iter)
        self.crown_logit = nn.Parameter(torch.tensor(0.0))
        self.log_spot_a = nn.Parameter(torch.tensor(0.0))
        self.spot_b = nn.Parameter(torch.tensor(-4.0))

    def rate2(self, x, r_surf, r_crown):
        out = self.net(x)
        c = MAX_LOG_CORR * torch.tanh(out[:, 0] / MAX_LOG_CORR)
        floor = self.r_floor * torch.sigmoid(out[:, 1] - 4.0)
        w = torch.sigmoid(self.crown_logit)
        return (r_surf + w * (r_crown - r_surf).clamp_min(0)) * torch.exp(c) + floor, c, out[:, 2]

    def spot_logit(self, ember, gap, g):
        return self.log_spot_a.exp() * torch.log(ember * gap.view(-1, 1, 1) + 1e-4) + self.spot_b + g

    def forward(self, x, phys, burning, gap):
        R, c, g = self.rate2(x, phys["r_head"], phys["r_crown"])
        logits, Ts = [], []
        h_eff, tau = self.log_h.exp(), self.log_tau.exp()
        for b in range(x.shape[0]):
            costs = S.step_costs(R[b], phys["ecc"][b], phys["head_x"][b], phys["head_y"][b], self.dx)
            horizon = h_eff * gap[b]
            T = S.arrival_time(costs, burning[b], max_iter=self.max_iter, t_max=float(horizon.detach()) * 1.5)
            Ts.append(T)
            logits.append((horizon - T.clamp(max=1e5)) / tau)
        spread = torch.stack(logits)
        return combine_logits(spread, self.spot_logit(phys["ember"], gap, g)), torch.stack(Ts), c
