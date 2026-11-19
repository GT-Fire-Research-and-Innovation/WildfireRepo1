"""Stage 1 (hazard head + survival loss): exact cumulative hazard, the likelihood against its closed form, recovery of
a known constant hazard and of a known day/night hazard from interval-censored perimeters, probabilities that never
decrease with horizon, and one end-to-end training step + old-style scoring on a synthetic fire."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
import data as D  # noqa: E402
import hazard as HZ  # noqa: E402
import intervals as IV  # noqa: E402
import train_hybrid as TH  # noqa: E402

K = HZ.N_BINS


def test_cumulative_hazard_is_exact_for_piecewise_constant_rates():
    lam = torch.rand(2, K, 3, 4) * 0.1
    for t in (0.0, 1.3, 3.0, 17.9, 71.99, 72.0, 90.0):
        got = HZ.cumhaz_at(lam, torch.full((2, 3, 4), t))
        edges = np.arange(K + 1) * HZ.BIN_H
        want = np.zeros((2, 3, 4))
        for b in range(K):
            want += lam[:, b].numpy() * max(0.0, min(t, edges[b + 1]) - edges[b])
        if t > edges[-1]:
            want += lam[:, -1].numpy() * (t - edges[-1])          # carried on past the horizon
        assert np.allclose(got.numpy(), want, atol=1e-5), t


def test_likelihood_matches_the_closed_form():
    c = 0.04
    lam = torch.full((1, K, 1, 3), c)
    t_lo = torch.tensor([[[2.0, 10.0, 30.0]]])
    t_hi = torch.tensor([[[7.0, math.inf, 31.0]]])
    nll = HZ.survival_nll(lam, t_lo, t_hi, torch.ones(1, 1, 3, dtype=torch.bool), reduce=False)[0, 0]
    want = [-math.log(math.exp(-c * 2) - math.exp(-c * 7)), c * 10, -math.log(math.exp(-c * 30) - math.exp(-c * 31))]
    assert np.allclose(nll.numpy(), want, rtol=1e-5)
    lg = HZ.logit_burned_by(lam, torch.full((1, 1, 3), 26.0))
    p = 1 - math.exp(-c * 26)
    assert np.allclose(lg.numpy(), math.log(p / (1 - p)), rtol=1e-5)


def simulate(rate_fn, n_fires=300, cells=300, seed=0):
    """Burn times drawn from a known hazard; each 'fire' has its own irregular perimeter schedule (3-30 h gaps) and
    start hour. Labels come from the Stage 0 rule, so the two stages are tested together."""
    rng = np.random.default_rng(seed)
    T_LO, T_HI, V, START = [], [], [], []
    grid = np.arange(0, 400, 0.05)
    for _ in range(n_fires):
        h0 = rng.uniform(0, 24)
        rate = np.array([rate_fn(h0, t) for t in grid])
        cum = np.concatenate([[0], np.cumsum(rate[:-1]) * 0.05])
        T = np.interp(rng.exponential(1.0, cells), cum, grid)         # inverse transform: P(T > t) = exp(-Lambda(t))
        times = np.concatenate([[0], np.cumsum(rng.uniform(3, 30, 12))])
        arr = np.full(cells, -1)
        for k in range(len(times) - 1, 0, -1):
            arr[T <= times[k]] = k
        lo, hi = IV.perimeter_bounds(times)
        a, b, v = IV.interval_labels(arr[None], 0, lo, hi)
        T_LO.append(a[0]); T_HI.append(b[0]); V.append(v[0]); START.append(h0)
    t = lambda x: torch.from_numpy(np.stack(x))[:, None]  # noqa: E731   (fires, 1, cells)
    return t(T_LO), t(T_HI), t(V), np.array(START)


def test_recovers_a_known_constant_hazard():
    true = 0.03
    t_lo, t_hi, v, _ = simulate(lambda h0, t: true)
    raw = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(400):
        lam = torch.nn.functional.softplus(raw - 3.5)[None, :, None, None].expand(t_lo.shape[0], K, 1, t_lo.shape[-1])
        loss = HZ.survival_nll(lam, t_lo, t_hi, v)
        opt.zero_grad(); loss.backward(); opt.step()
    got = torch.nn.functional.softplus(raw - 3.5).detach().numpy()
    r = got / true
    assert abs(r.mean() - 1) < 0.03, got                          # pooled: 1.003 / 0.999 / 1.003 over seeds 0-2
    for part in (r[:6], r[6:12], r[12:18], r[18:]):               # no drift across the horizon (late bins were biased
        assert abs(part.mean() - 1) < 0.10, got                   # toward 0 by informative censoring before the fix)
    assert np.all(np.abs(r - 1) < 0.6), got                       # single 3 h bins are noisy: perimeter gaps are 3-30 h


def test_shared_head_learns_a_day_night_hazard_from_the_bin_weather():
    """Hazard 0.08/h in the local afternoon (12-18 h), 0.005/h otherwise. The head only sees the hour of day through
    the per-bin channels, so recovering the contrast means the bins really use their own weather."""
    torch.manual_seed(0)
    day = lambda hr: 12 <= hr % 24 < 18  # noqa: E731
    rate_fn = lambda h0, t: 0.08 if day(h0 + t) else 0.005  # noqa: E731
    n_f, side = 64, 16
    t_lo, t_hi, v, h0 = simulate(rate_fn, n_fires=n_f, cells=side * side, seed=1)
    t_lo, t_hi, v = (a.view(n_f, side, side) for a in (t_lo, t_hi, v))
    wx = torch.zeros(n_f, K, D.Fire.BIN_CH, side, side)
    hrs = h0[:, None] + (np.arange(K) + 0.5) * HZ.BIN_H
    wx[:, :, 5] = torch.from_numpy(np.sin(2 * np.pi * hrs / 24)).float()[:, :, None, None]
    wx[:, :, 6] = torch.from_numpy(np.cos(2 * np.pi * hrs / 24)).float()[:, :, None, None]
    x = torch.zeros(n_f, 8, side, side)
    trunk = TH.build_model("unet3", 8, 8, "cpu", arch={"time": False, "attn": False})
    net = HZ.HazardNet(trunk, D.Fire.BIN_CH, 8, hidden=16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(300):
        loss = HZ.survival_nll(net(x, wx), t_lo, t_hi, v)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lam = net(x, wx)[:, :, 0, 0].numpy()
    is_day = np.vectorize(day)(hrs)
    d, n = lam[is_day].mean(), lam[~is_day].mean()
    assert d / n > 5, (d, n)                                      # true contrast 16x; a flat fit gives 1x
    assert 0.04 < d < 0.12 and n < 0.02, (d, n)


def test_probabilities_never_decrease_with_horizon():
    torch.manual_seed(1)
    trunk = TH.build_model("unet3", 5, 8, "cpu", arch={"time": False, "attn": False})
    net = HZ.HazardNet(trunk, D.Fire.BIN_CH, 8, hidden=8)
    for p in net.parameters():                                    # far from the initial near-constant hazard
        torch.nn.init.normal_(p, std=0.5)
    with torch.no_grad():
        lam = net(torch.randn(2, 5, 16, 16), torch.randn(2, K, D.Fire.BIN_CH, 16, 16))
        ts = torch.linspace(0, 100, 401)
        P = torch.stack([1 - torch.exp(-HZ.cumhaz_at(lam, torch.full((2, 16, 16), float(t)))) for t in ts])
        L = torch.stack([HZ.logit_burned_by(lam, torch.full((2, 16, 16), float(t))) for t in ts[1:]])
    assert (lam >= 0).all()
    assert (P[1:] - P[:-1] >= -1e-6).all()
    assert (L[1:] - L[:-1] >= -1e-4).all()


# ---------------------------------------------------------------------------- end to end on a synthetic fire
@pytest.fixture
def fire(tmp_path):
    import test_intervals as TI
    keep = {k: getattr(D, k) for k in ("FM_ON", "FUEL_V2", "TEMPORAL", "PHYSICS_V2", "WIND_TERRAIN", "HRRR_ROTATE")}
    D._CACHE.clear()
    D.HRRR_ROTATE, D.TEMPORAL, D.PHYSICS_V2 = False, True, True
    f = TI.make_fire(tmp_path, ["mapped", "edit", "mapped", "date_only", "mapped"])
    TH.set_grid(f.res_m / 1000)
    yield f
    for k, v in keep.items():
        setattr(D, k, v)
    D._CACHE.clear()
    TH.set_grid(0.15)


def test_training_step_and_old_style_scoring(fire, tmp_path):
    import train_hazard as THZ
    import evaluate as EV
    TH.CROP, TH.EDGE = 48, 4
    groups = TH.groups_used(set(), v2=True, t3=True, origin=False, fm=False)
    n_in = sum(D.GROUPS[g] for g in groups)
    ds = THZ.HazardCrops([fire], groups, 2)
    x, wx, t_lo, t_hi, valid = ds[0]
    assert x.shape == (n_in, 48, 48) and wx.shape == (K, D.Fire.BIN_CH, 48, 48) and wx.dtype == torch.float16
    assert valid.any() and torch.isinf(t_hi[valid]).any() and torch.isfinite(t_hi[valid]).any()
    net = THZ.build(n_in, 8, False, True, "cpu", hidden=8)
    b = [torch.stack([t] * 2) for t in (x, wx, t_lo, t_hi, valid)]
    loss = HZ.survival_nll(net(b[0], b[1].float()), b[2], b[3], b[4])
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)

    m, rows = THZ.evaluate(net, [fire], groups, "cpu")
    assert np.isfinite(m["ap"]) and np.isfinite(m["interval_nll"]) and len(rows) == len(fire.pairs)
    assert m["by_horizon"] and m["by_gap"]

    # evaluate.py scores a saved hazard model with the protocol of every earlier model
    runs = tmp_path / "runs"
    runs.mkdir()
    torch.save(net.state_dict(), runs / "hz.pt")
    (runs / "hz.json").write_text(json.dumps({"mode": "hazard", "arch": {"time": True, "attn": False}, "width": 8, "hidden": 8,
                                               "n_bins": K, "bin_h": HZ.BIN_H, "fuel_moisture": False}))
    mb = EV.Member(runs, "hz", None, groups, "cpu", fire.res_m, 8)
    arrays, cells, rows2 = EV.run_split([mb], [fire], calibrate=True)
    assert len(rows2) == len(fire.pairs) and np.isfinite(arrays["raw"]).all()
    # same cells and target as the old protocol, and the same scores as train_hazard's own pass
    assert cells["y"].sum() == sum(r["new"] for r in rows)
    assert np.isclose(TH.metrics(arrays["raw"], cells["y"])["ap"], m["ap"], atol=1e-6)
