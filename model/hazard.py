"""
Stage 1 of the hazard model: a conditional-hazard head on the current network (UNet3), trained on interval-censored
perimeters (intervals.py) with a survival likelihood.

  hazard     lambda_b(x) >= 0 per hour, constant inside each of N_BINS bins of BIN_H hours after the start. A small
             head SHARED by all bins takes the network's map features plus that bin's weather and hour of day, so a
             bin knows a windy afternoon from a calm night.
  cumulative Lambda(t) = integral of lambda from the start to t (exact for piecewise-constant hazard);
             survival S(t) = exp(-Lambda(t)); P(burned by t) = 1 - S(t), which never decreases with t.
  loss       burned in (t_lo, t_hi]:  -log(S(t_lo) - S(t_hi)) = Lambda(t_lo) - log(1 - exp(-(Lambda(t_hi) - Lambda(t_lo))))
             right-censored at t_c:   -log S(t_c) = Lambda(t_c)
             all in log space; no class weighting.
  old score  the logit of P(burned by the next perimeter), so evaluate.py / compare.py score it like every earlier
             model. Past the horizon the last bin's hazard is carried on (an extrapolation; flagged in the results).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BINS, BIN_H = 24, 3.0
LAMBDA0 = 0.002          # initial hazard per hour everywhere: P(burn within a 26 h gap) about 5 %


def log1mexp(d):
    """log(1 - exp(-d)) for d > 0, accurate for small and large d."""
    c = math.log(2)          # each branch sees only inputs it is accurate for, so the unused one has no inf / NaN gradient
    return torch.where(d < c, torch.log(-torch.expm1(-d.clamp(max=c))), torch.log1p(-torch.exp(-d.clamp(min=c))))


def cumhaz_at(lam, t, bin_h=BIN_H):
    """Lambda(t) from per-bin hazards lam (B, K, H, W) at times t (B, H, W) in hours. t below 0 counts as 0;
    beyond the last bin the last bin's hazard is carried on."""
    K = lam.shape[1]
    cum = torch.cat([torch.zeros_like(lam[:, :1]), torch.cumsum(lam, 1) * bin_h], 1)      # (B, K+1, H, W)
    t = t.clamp(min=0)
    k = torch.clamp((t / bin_h).floor().long(), max=K - 1)
    base = torch.gather(cum, 1, k[:, None])[:, 0]
    rate = torch.gather(lam, 1, k[:, None])[:, 0]
    return base + rate * (t - k.to(t.dtype) * bin_h)


def survival_nll(lam, t_lo, t_hi, valid, bin_h=BIN_H, reduce=True):
    """Interval-censored negative log-likelihood per valid cell. t_hi = inf marks right-censored at t_lo."""
    cens = torch.isinf(t_hi)
    L_lo = cumhaz_at(lam, t_lo, bin_h)
    L_hi = cumhaz_at(lam, torch.where(cens, t_lo, t_hi), bin_h)
    d = (L_hi - L_lo).clamp(min=1e-8)
    nll = torch.where(cens, L_lo, L_lo - log1mexp(d))
    v = valid.to(nll.dtype)
    if not reduce:
        return nll * v
    return (nll * v).sum() / v.sum().clamp(min=1.0)


def logit_burned_by(lam, t, bin_h=BIN_H):
    """logit of P(burned by t) = log(exp(Lambda) - 1), stable for tiny and large Lambda."""
    L = cumhaz_at(lam, t, bin_h).clamp(min=1e-30)
    return torch.where(L > 20, L, torch.log(torch.expm1(L.clamp(max=20))))


class HazardNet(nn.Module):
    """trunk: a UNet3 whose 1x1 output head is replaced by Identity (it returns its last feature map)."""

    def __init__(self, trunk, n_wx, width, hidden=32, n_bins=N_BINS, bin_h=BIN_H, lam0=LAMBDA0):
        super().__init__()
        trunk.head = nn.Identity()
        self.trunk, self.n_bins, self.bin_h = trunk, n_bins, bin_h
        self.f_map = nn.Conv2d(width, hidden, 1)
        self.f_wx = nn.Linear(n_wx, hidden)
        self.f2 = nn.Conv2d(hidden, hidden, 1)
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.constant_(self.out.bias, math.log(math.expm1(lam0)))       # softplus(bias) = lam0

    def forward(self, x, wx):
        """x (B, C, H, W) map inputs; wx (B, K, n_wx, H, W) per-bin weather. Returns hazard per hour (B, K, H, W)."""
        feat = self.trunk(x)
        B, K = wx.shape[:2]
        H, W = feat.shape[-2:]
        a = self.f_map(feat)[:, None] + torch.einsum("bkchw,dc->bkdhw", wx.to(feat.dtype), self.f_wx.weight) \
            + self.f_wx.bias[None, None, :, None, None]
        a = F.gelu(a).flatten(0, 1)
        a = F.gelu(self.f2(a))
        return F.softplus(self.out(a)).view(B, K, H, W)
