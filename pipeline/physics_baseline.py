"""
Physics-only baseline: Rothermel spread on LANDFIRE fuels + weather, run through the arrival-time
solver from the day-k fire, scored with the same strict protocol as the neural models
(new burn only, unknown cells masked, threshold chosen on validation fires).

Score per cell = -(arrival time), so AP measures how well physics ranks which cells burn next;
the validation threshold is equivalent to calibrating "effective burning minutes per day".

  python physics_baseline.py --out data [--ignite perim|perim_af] [--split val test]

Inputs per fire: data/landfire/<fid>.tif (FBFM40, CC, CH, CBH, CBD), data/train/<fid>.npz
(GRIDMET + HRRR raw), data/cache/<fid>/ (labels, slope, aspect, VIIRS).
Fuel moisture: 1 h and 10 h from the afternoon equilibrium moisture content (Simard 1968, Tmax and
RHmin), 100 h from GRIDMET; live herbaceous 60 %, live woody 90 % (late-season defaults).
Wind: HRRR 21z forecast for day k+1 when available, else GRIDMET daily mean of day k.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import build_dataset as B
import train as T

_here = Path(__file__).resolve().parent
for _p in (_here.parent, _here.parent / "WildfireRepo1"):       # repo layout, or the working copy
    if (_p / "physics" / "rothermel.py").exists():
        sys.path.insert(0, str(_p))
        break
from physics import rothermel as R  # noqa: E402
from physics import solver as S  # noqa: E402

DX = 150.0
HERB, WOODY = 0.6, 0.9


def emc(t_f, rh):
    """Equilibrium moisture content (%) from temperature (F) and relative humidity (%), Simard 1968."""
    return np.where(rh < 10, 0.03229 + 0.281073 * rh - 0.000578 * rh * t_f,
                    np.where(rh < 50, 2.22749 + 0.160107 * rh - 0.01478 * t_f,
                             21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * t_f - 0.483199 * rh))


class Inputs:
    """Static fuels/terrain and raw weather for one fire on its 150 m grid."""

    def __init__(self, out, fire):
        import rasterio
        z = np.load(out / "train" / f"{fire.fid}.npz")
        r0, r1, c0, c1 = map(int, z["region"])
        lon, lat = B.grid_centres(list(z["transform150"]), r0, r1, c0, c1)
        with rasterio.open(out / "landfire" / f"{fire.fid}.tif") as d:
            a = d.read()
            tr = d.transform
            rr, cc = B.regular_index(lon, lat, tr.c, tr.f, tr.a, -tr.e, d.width, d.height)
        fb = a[0][rr, cc].astype(np.int64)
        fb[fb < 0] = 91
        self.fuel = R.fuel_index(torch.from_numpy(fb))
        self.cc = np.clip(a[1][rr, cc], 0, 100).astype(np.float32) / 100
        self.ch = np.clip(a[2][rr, cc], 0, None).astype(np.float32) / 10
        self.slope = np.tan(np.radians(np.asarray(fire.slope, np.float32)))
        self.aspect = np.asarray(fire.aspect, np.float32)
        self.gm = z["gm"]
        self.gm_vars = {str(v): i for i, v in enumerate(z["gm_vars"])}
        self.w_dates = {str(d): i for i, d in enumerate(z["w_dates"])}
        self.gr, self.gc = z["gm_row"].astype(np.int64), z["gm_col"].astype(np.int64)
        self.h_fc = z["hrrr_fc"] if "hrrr_fc" in z.files else None
        self.h_idx = z["hrrr_idx"].astype(np.int64) if "hrrr_idx" in z.files else None
        self.h_vars = [str(v) for v in z["hrrr_vars"]] if "hrrr_vars" in z.files else []

    def weather(self, d0, d1, win):
        r0, r1, c0, c1 = win
        sl = (slice(r0, r1), slice(c0, c1))
        i0 = self.w_dates.get(d0)
        if i0 is None:
            return None
        g = self.gm[i0][:, self.gr[sl], self.gc[sl]].astype(np.float32)
        v = self.gm_vars
        t_f = (np.nan_to_num(g[v["tmmx"]], nan=300.0) - 273.15) * 9 / 5 + 32
        rh = np.clip(np.nan_to_num(g[v["rmin"]], nan=20.0), 1, 100)
        e = emc(t_f, rh)
        m1, m10 = 1.03 * e / 100, 1.28 * e / 100
        m100 = np.nan_to_num(g[v["fm100"]], nan=10.0) / 100
        ws, wd = np.nan_to_num(g[v["vs"]], nan=3.0), np.nan_to_num(g[v["th"]], nan=270.0)
        i1 = self.w_dates.get(d1)
        if self.h_fc is not None and i1 is not None:
            fc = self.h_fc[i1][:, self.h_idx[sl]]
            u, vv = fc[self.h_vars.index("u10")], fc[self.h_vars.index("v10")]
            ok = np.isfinite(u) & np.isfinite(vv)
            ws = np.where(ok, np.hypot(u, vv), ws)
            wd = np.where(ok, (np.degrees(np.arctan2(-u, -vv)) + 360) % 360, wd)
        return m1, m10, m100, ws, wd


def arrival(inp, fire, k, gap, win, ignite):
    r0, r1, c0, c1 = win
    sl = (slice(r0, r1), slice(c0, c1))
    w = inp.weather(fire.dates[k], fire.dates[k + 1], win)
    if w is None:
        return None
    m1, m10, m100, ws, wd = w
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))  # noqa: E731
    h, wd_ = r1 - r0, c1 - c0
    out = R.spread(inp.fuel[sl], t(m1), t(m10), t(m100), torch.full((h, wd_), HERB), torch.full((h, wd_), WOODY),
                   t(ws), t(wd), t(inp.slope[sl]), t(inp.aspect[sl]), t(inp.cc[sl]), t(inp.ch[sl]))
    burning = torch.from_numpy((fire.arrival[sl] >= 0) & (fire.arrival[sl] <= k))
    if ignite == "perim_af" and fire.viirs_have[k]:
        burning |= torch.from_numpy(np.asarray(fire.viirs[k, 0][sl]) > 0)
    if not burning.any():
        return None
    costs = S.step_costs(out["r_head"], out["ecc"], out["head_x"], out["head_y"], DX)
    with torch.no_grad():
        Tm = S.arrival_time(costs, burning, max_iter=600, t_max=1440.0 * gap)
    return Tm.numpy(), out


def score_split(out, fires, ignite):
    Sc, Y, rows = [], [], []
    for fire in fires:
        if not (out / "landfire" / f"{fire.fid}.tif").exists():
            print(f"  skip {fire.fid}: no LANDFIRE", flush=True)
            continue
        t0 = time.time()
        inp = Inputs(out, fire)
        n = 0
        for k, gap in fire.pairs:
            reg = T.eval_region(fire, k)
            if reg is None:
                continue
            res = arrival(inp, fire, k, gap, reg, ignite)
            if res is None:
                continue
            Tm, phys = res
            _, y, m, _ = fire.features(k, gap, reg)
            s = -np.minimum(Tm, 1e7)
            Sc.append(s[m])
            Y.append(y[m])
            rows.append(dict(fire=fire.fid, k=k, gap=gap, new=int(y[m].sum()),
                             r_head_med=float(phys["r_head"][phys["burnable"]].median()) if phys["burnable"].any() else 0.0))
            n += 1
        print(f"  {fire.fid}: {n} pairs ({time.time() - t0:.0f}s)", flush=True)
    return np.concatenate(Sc), np.concatenate(Y), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--ignite", default="perim", choices=["perim", "perim_af"])
    a = ap.parse_args()
    out = Path(a.out)
    torch.set_num_threads(1)
    va, te = T.load(out, ["val"]), T.load(out, ["test"])
    sv, yv, rv = score_split(out, va, a.ignite)
    mv = T.metrics(sv, yv)
    st, yt, rt = score_split(out, te, a.ignite)
    mt = T.metrics(st, yt, mv["thr"])
    name = f"base_physics_{a.ignite}"
    mv["effective_minutes"] = -mv["thr"]
    T.save_run(out, name, {"name": name, "val": mv, "test": mt, "pairs_val": rv, "pairs_test": rt})
    print(f"[physics, {a.ignite}] effective burning minutes per day (val-calibrated): {-mv['thr']:.0f}")
    T.show(name, mt)


if __name__ == "__main__":
    sys.exit(main())
