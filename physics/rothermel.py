"""
Differentiable Rothermel (1972) surface fire spread, vectorised over a grid in PyTorch.

Follows Rothermel 1972 with Albini 1976 corrections, Scott & Burgan 2005 dynamic fuel loading,
BehavePlus wind adjustment (Andrews 2012), wind/slope vector combination on the slope plane,
and the BehavePlus length-to-width ratio. Checked against the pyretechnics reference
implementation in tests/test_physics.py.

Every operation is a torch op, so gradients flow from spread rate back to fuel moisture, wind
and any learned correction applied to the inputs or outputs.

Units: inputs are SI or fractions (documented per argument). Rothermel's equations run in
English units internally, and outputs are SI.
"""
import torch

from .sb40_table import SB40

FT = 0.3048                      # m per ft
EPS = 1e-9

# Parameter table as tensors, indexed by "fuel index"; CODES maps fuel model number -> index
CODES = sorted(SB40)
CODE_TO_IDX = {c: i for i, c in enumerate(CODES)}
_T = torch.tensor([SB40[c] for c in CODES], dtype=torch.float64)


def fuel_index(codes):
    """Map a tensor of LANDFIRE FBFM40 codes to table indices. Unknown codes -> NB1 (non-burnable)."""
    lut = torch.full((256,), CODE_TO_IDX[91], dtype=torch.long)
    for c, i in CODE_TO_IDX.items():
        lut[c] = i
    codes = torch.as_tensor(codes).long().clamp(0, 255)
    return lut[codes]


def _params(idx, dtype):
    p = _T.to(dtype=dtype, device=idx.device)[idx]          # (..., 13)
    p = torch.movedim(p, -1, 0)
    depth, mx_dead, heat = p[0], p[1] * 0.01, p[2] * 1000.0
    load = p[3:8]                                              # 1h, 10h, 100h, live herb, live woody
    sav = p[8:13]
    burnable = (depth > 0)
    dynamic = (idx >= 0) & (torch.as_tensor(CODES, device=idx.device)[idx] > 100) & (load[3] > 0)
    return depth, mx_dead, heat, load, sav, burnable, dynamic


def _size_class(sav):
    """Albini (1976) FIREMOD size classes used for the net-load weights g_ij."""
    return torch.bucketize(-sav, torch.tensor([-1200.0, -192.0, -96.0, -48.0, -16.0], dtype=sav.dtype,
                                               device=sav.device), right=True).to(sav.dtype)


def spread(fuel_idx, m1, m10, m100, m_herb, m_woody, wind10_ms, wind_from_deg, slope, aspect_deg,
           canopy_cover=None, canopy_height_m=None, wind_limit=True, waf=None):
    """
    Head fire spread on every cell.

    fuel_idx       (H, W) long   index into the fuel table (use fuel_index on FBFM40 codes)
    m1..m_woody    (H, W)        fuel moisture fractions (0.06 = 6 %)
    wind10_ms      (H, W)        10 m wind speed, m/s
    wind_from_deg  (H, W)        direction the wind blows FROM, degrees clockwise from north
    slope          (H, W)        rise / run
    aspect_deg     (H, W)        downslope direction, degrees clockwise from north
    canopy_cover   (H, W)        fraction 0-1, optional (for the wind adjustment factor)
    canopy_height_m(H, W)        metres, optional
    waf            (H, W)        optional wind adjustment factor overriding the BehavePlus one

    Returns dict of (H, W) tensors:
      r0        no-wind no-slope spread rate, m/min
      r_head    head fire spread rate, m/min
      head_x/y  unit vector (east, north) of the head fire direction on the horizontal plane
      lb        length-to-breadth ratio
      ecc       eccentricity
      u_eff     effective wind speed, m/min
      i_r       reaction intensity, kW/m2
    """
    dtype = m1.dtype
    depth, mx_dead, heat, load, sav, burnable, dynamic = _params(fuel_idx, dtype)
    dyn = dynamic.to(dtype)

    # Dynamic fuel models: cure part of the herbaceous load into a dead herb class (Scott & Burgan 2005)
    green = torch.clamp(m_herb / 0.9 - 1.0 / 3.0, 0.0, 1.0)
    green = dyn * green + (1 - dyn)
    w = torch.stack([load[0], load[1], load[2], load[3] * (1 - green), load[3] * green, load[4]])
    s = torch.stack([sav[0], sav[1], sav[2], sav[3] * dyn, sav[3], sav[4]])
    M = torch.stack([m1, m10, m100, m1, m_herb, m_woody])
    dead = torch.tensor([1, 1, 1, 1, 0, 0], dtype=dtype, device=m1.device).view(6, *([1] * m1.dim()))
    live = 1 - dead

    # Weighting factors (surface-area based)
    a = s * w / 32.0
    a_d, a_l = (a * dead).sum(0), (a * live).sum(0)
    f = a * dead / (a_d + EPS) + a * live / (a_l + EPS)
    f_d, f_l = a_d / (a_d + a_l + EPS), a_l / (a_d + a_l + EPS)
    fcat = torch.stack([f_d, f_l])

    # g_ij: share of the category's area in the same FIREMOD size class (Albini 1976)
    sc = _size_class(s) * (s > 0).to(dtype) - (s <= 0).to(dtype)   # -1 for absent classes
    same = (sc[:, None] == sc[None, :]).to(dtype) * (dead[:, None] == dead[None, :]).to(dtype)
    g = (same * f[None, :]).sum(1)

    def cat_sum(x):
        return torch.stack([(x * dead).sum(0), (x * live).sum(0)])

    # Live moisture of extinction (Albini 1976)
    expA = torch.where(s > 0, torch.exp(torch.where(dead > 0, -138.0, -500.0) / s.clamp_min(EPS)),
                       torch.zeros_like(s))
    lf = w * expA
    dead_lf, live_lf = (lf * dead).sum(0), (lf * live).sum(0)
    md_f = (lf * M * dead).sum(0) / (dead_lf + EPS)
    mx_live = torch.maximum(mx_dead, 2.9 * dead_lf / (live_lf + EPS) * (1 - md_f / (mx_dead + EPS)) - 0.226)
    has_both = (dead_lf > 0) & (live_lf > 0)
    mx_live = torch.where(has_both, mx_live, mx_dead)
    Mx = torch.stack([mx_dead, mx_dead, mx_dead, mx_dead * dyn, mx_live, mx_live])

    # Category moisture damping
    Mf_i, Mx_i = cat_sum(f * M), cat_sum(f * Mx)
    r = torch.clamp(Mf_i / (Mx_i + EPS), max=1.0)
    eta_m = torch.where(Mx_i > 0, 1 - 2.59 * r + 5.11 * r ** 2 - 3.52 * r ** 3, torch.zeros_like(r))
    eta_s = 0.174 * 0.01 ** -0.19
    wn = cat_sum(g * w * (1 - 0.0555))

    sigma = (fcat * cat_sum(f * s)).sum(0)                      # characteristic SAV (1/ft)
    sig = sigma.clamp_min(1.0)
    depth_s = depth.clamp_min(EPS)
    beta = w.sum(0) / 32.0 / depth_s
    beta_op = 3.348 * sig ** -0.8189
    A = 133.0 * sig ** -0.7913
    gmax = sig ** 1.5 / (495.0 + 0.0594 * sig ** 1.5)
    ratio = (beta / beta_op).clamp_min(EPS)
    gamma = gmax * ratio ** A * torch.exp(A * (1 - ratio))
    i_r = gamma * (wn * heat * eta_m * eta_s).sum(0)            # Btu/ft2/min
    xi = torch.exp((0.792 + 0.681 * sig ** 0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sig)
    rho_b = w.sum(0) / depth_s
    eps_h = torch.where(s > 0, torch.exp(-138.0 / s.clamp_min(EPS)), torch.zeros_like(s))
    q_ig = (250.0 + 1116.0 * M) * (M > 0).to(dtype)
    sink = rho_b * (fcat * cat_sum(f * eps_h * q_ig)).sum(0)
    r0 = i_r * xi / (sink + EPS)                                 # ft/min
    r0 = torch.where(burnable, r0, torch.zeros_like(r0))

    # Wind factor coefficients
    B = 0.02526 * sig ** 0.54
    C = 7.47 * torch.exp(-0.133 * sig ** 0.55)
    E = 0.715 * torch.exp(-3.59e-4 * sig)
    F = ratio ** E

    # Midflame wind: 10 m -> 20 ft (/1.15), then wind adjustment factor
    if waf is None:
        cc = torch.zeros_like(m1) if canopy_cover is None else canopy_cover
        ch = torch.zeros_like(m1) if canopy_height_m is None else canopy_height_m / FT
        fill = cc / 3.0
        sheltered = (fill >= 0.05) & (ch >= 6.0)
        waf_sh = 0.555 / (torch.sqrt((fill * ch).clamp_min(EPS)) *
                          torch.log(((20 + 0.36 * ch) / (0.13 * ch.clamp_min(EPS))).clamp_min(1 + 1e-6)))
        waf_un = 1.83 / torch.log((20 + 0.36 * depth_s) / (0.13 * depth_s))
        waf = torch.where(sheltered, waf_sh, waf_un)
        waf = torch.where(burnable, waf, torch.zeros_like(waf))
    u_mid = wind10_ms / 1.15 * waf / FT * 60.0                  # ft/min

    # Vectors (x east, y north). Wind blows toward wind_from + 180; upslope is aspect + 180.
    wr = torch.deg2rad(wind_from_deg + 180.0)
    ar = torch.deg2rad(aspect_deg + 180.0)
    wx, wy = torch.sin(wr), torch.cos(wr)
    gx, gy = slope * torch.sin(ar), slope * torch.cos(ar)       # elevation gradient
    # project onto the slope plane: z = v . grad
    w3 = torch.stack([u_mid * wx, u_mid * wy, u_mid * (wx * gx + wy * gy)])
    s3 = torch.stack([gx, gy, gx * gx + gy * gy])
    u_proj = torch.sqrt((w3 ** 2).sum(0) + EPS)
    phi_w = C * u_proj ** B / F
    phi_s = 5.275 * beta ** -0.3 * slope ** 2
    phi_w = torch.where(u_mid > 0, phi_w, torch.zeros_like(phi_w))
    w_unit = w3 / u_proj
    s_unit = s3 / torch.sqrt((s3 ** 2).sum(0) + EPS)
    pe = phi_w * w_unit + phi_s * s_unit
    phi_e = torch.sqrt((pe ** 2).sum(0) + EPS ** 2)
    u_eff = (phi_e * F / C).clamp_min(EPS) ** (1.0 / B)          # ft/min
    if wind_limit:
        u_max = 0.9 * i_r
        over = u_eff > u_max
        phi_e = torch.where(over, C * u_max.clamp_min(EPS) ** B / F, phi_e)
        u_eff = torch.where(over, u_max, u_eff)
    r_head = r0 * (1 + phi_e)

    # Heading on the horizontal plane (fallback: north)
    hx, hy = pe[0], pe[1]
    hn = torch.sqrt(hx ** 2 + hy ** 2)
    ok = hn > 1e-6
    head_x = torch.where(ok, hx / hn.clamp_min(1e-6), torch.zeros_like(hx))
    head_y = torch.where(ok, hy / hn.clamp_min(1e-6), torch.ones_like(hy))

    u_mph = u_eff * 60.0 / 5280.0
    lb = torch.clamp(0.936 * torch.exp(0.1147 * u_mph) + 0.461 * torch.exp(-0.0692 * u_mph) - 0.397, 1.0, 8.0)
    ecc = torch.sqrt((lb ** 2 - 1).clamp_min(0) + EPS ** 2) / lb

    # non-burnable cells (water, rock, urban, ag): no spread, neutral shape (avoids 0 * inf = NaN)
    zero = torch.zeros_like(r0)
    r_head = torch.where(burnable, r_head, zero)
    head_x = torch.where(burnable, head_x, zero)
    head_y = torch.where(burnable, head_y, torch.ones_like(r0))
    lb = torch.where(burnable, lb, torch.ones_like(r0))
    ecc = torch.where(burnable, ecc, zero)
    u_eff = torch.where(burnable, u_eff, zero)
    i_r = torch.where(burnable, i_r, zero)
    return dict(r0=r0 * FT, r_head=r_head * FT, head_x=head_x, head_y=head_y, lb=lb, ecc=ecc,
                u_eff=u_eff * FT, i_r=i_r * 1.055056 / FT ** 2 / 60.0, burnable=burnable,
                sigma=torch.where(burnable, sig, torch.ones_like(r0)))          # characteristic SAV, 1/ft


def ray_speed(r_head, ecc, head_x, head_y, dir_x, dir_y):
    """Spread rate along a ray from the ignition point in direction (dir_x, dir_y): an ellipse with
    the ignition at its rear focus, R(psi) = R_head (1 - e) / (1 - e cos psi)."""
    cos = head_x * dir_x + head_y * dir_y
    return r_head * (1 - ecc) / (1 - ecc * cos).clamp_min(1e-6)


__all__ = ["spread", "ray_speed", "fuel_index", "CODES", "CODE_TO_IDX"]
