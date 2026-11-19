"""
Stage 1 of the hazard model ("FireStep"): UNet3 trunk + shared per-bin hazard head (hazard.py), trained on
interval-censored perimeters (intervals.py) with a survival likelihood, no class weighting.

Same fires, split, crops and input groups as train_hybrid.py (unet3), with three differences:
  - every later perimeter within the 72 h horizon supervises a sample, not only the next one;
  - the weather inputs cover the 72 h after the start, not the time to the next perimeter (which is not known at
    forecast time), and each 3 h bin also gets its own HRRR weather and hour of day;
  - the gap-to-next-perimeter input channel is a constant (the horizon).
The checkpoint is chosen by validation AP of P(burned by the next perimeter), the score every earlier model has.

  python train_hazard.py --data ~/scratch/wildfire/my --name hazard_v1 --width 64 --u3-time 1 --u3-attn 0 \
         --fuel-moisture 1 --seed 0
Then score it with evaluate.py like any other model (it reads runs/<name>.json).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import hazard as HZ  # noqa: E402
import intervals as IV  # noqa: E402
import train_hybrid as TH  # noqa: E402

HORIZON = HZ.N_BINS * HZ.BIN_H
H_EDGES = [12, 24, 48]                     # calibration by horizon: <12, 12-24, 24-48, 48-72 h
H_NAMES = ["<12 h", "12-24 h", "24-48 h", "48-72 h"]


def build(n_in, width, attn, temporal, device, hidden=32):
    arch = {"time": temporal, "attn": attn}
    trunk = TH.build_model("unet3", n_in, width, "cpu", arch=arch)
    return HZ.HazardNet(trunk, D.Fire.BIN_CH, width, hidden).to(device)


def inputs(f, i, win, groups, hide=()):
    """Everything one forecast from perimeter i needs: map inputs, per-bin weather, interval labels, and the old
    next-perimeter target and mask (unchanged, for old-style scoring)."""
    feats, phys, y, m, b0 = f.sample(i, HORIZON, win, hide, wx_h=HORIZON)
    x = D.stack(feats, groups)
    wx = f.bin_weather(i, win, HZ.N_BINS, HZ.BIN_H)
    t_lo, t_hi, valid = f.intervals(i, win, HORIZON)
    return x, wx, t_lo, t_hi, valid, y, m, feats


class HazardCrops(torch.utils.data.Dataset):
    """Random training crops centred on the fire front of a random start perimeter (as train_hybrid.Crops)."""

    def __init__(self, fires, groups, n, seed=0):
        self.items = [(fi, i) for fi, f in enumerate(fires) for i, _ in f.pairs]
        self.fires, self.groups, self.n, self.seed = fires, groups, n, seed
        self.fx, self.fy = TH.flip_x_channels(groups), TH.flip_y_channels(groups)

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, k, int(time.time() * 1e3) % 10 ** 9))
        fi, i = self.items[rng.integers(len(self.items))]
        f = self.fires[fi]
        C, E = TH.CROP, TH.EDGE
        fc = f.front_cells(i)
        r, c = fc[rng.integers(len(fc))] if len(fc) else (f.H // 2, f.W // 2)
        jit = C // 7
        r0 = int(np.clip(r - C // 2 + rng.integers(-jit, jit + 1), 0, max(f.H - C, 0)))
        c0 = int(np.clip(c - C // 2 + rng.integers(-jit, jit + 1), 0, max(f.W - C, 0)))
        win = (r0, min(r0 + C, f.H), c0, min(c0 + C, f.W))
        hide = tuple(g for g in D.OPTIONAL if rng.random() < TH.HIDE_P)
        x, wx, t_lo, t_hi, valid, _, _, _ = inputs(f, i, win, self.groups, hide)
        if "hrrr" in hide:                                         # hidden like a missing hrrr.npz: hour of day only
            wx[:, :5], wx[:, 7] = 0.0, 0.0
        valid = valid.copy()
        valid[:E] = valid[-E:] = False
        valid[:, :E] = valid[:, -E:] = False
        if rng.random() < 0.5:                                     # east-west mirror
            x, wx, t_lo, t_hi, valid = x[..., ::-1].copy(), wx[..., ::-1].copy(), t_lo[:, ::-1], t_hi[:, ::-1], valid[:, ::-1]
            for chans in self.fx.values():
                x[chans] *= -1
            wx[:, 0] *= -1
        if rng.random() < 0.5:                                     # north-south mirror
            x, wx, t_lo, t_hi, valid = x[..., ::-1, :].copy(), wx[..., ::-1, :].copy(), t_lo[::-1], t_hi[::-1], valid[::-1]
            for chans in self.fy.values():
                x[chans] *= -1
            wx[:, 1] *= -1
        p = lambda a: TH.pad8(np.ascontiguousarray(a), C)  # noqa: E731
        return (torch.from_numpy(p(x).astype(np.float32)), torch.from_numpy(p(wx).astype(np.float16)),
                torch.from_numpy(p(t_lo[None])[0]), torch.from_numpy(p(t_hi[None])[0]), torch.from_numpy(p(valid[None])[0]))


def _cal(p, y):
    """Brier, ECE (10 bins, bins with >= 50 cells) and mean predicted vs observed."""
    if not len(p):
        return {}
    bins = np.minimum((p * 10).astype(int), 9)
    rel = [(p[bins == b].mean(), y[bins == b].mean(), (bins == b).sum()) for b in range(10) if (bins == b).sum() >= 50]
    n = max(sum(r[2] for r in rel), 1)
    return {"cells": int(len(p)), "brier": float(np.mean((p - y) ** 2)), "ece": float(sum(abs(a - b) * c for a, b, c in rel) / n),
            "mean_pred": float(p.mean()), "observed": float(y.mean())}


@torch.no_grad()
def evaluate(model, fires, groups, device, max_pairs=None, thr=None, cal_cells=20000, seed=0):
    """Old-style scores (logit of P(burned by the next perimeter), same windows and cells as train_hybrid.evaluate),
    the interval NLL, and calibration by gap (old-style) and by horizon (every later perimeter within 72 h)."""
    import evaluate as EV
    model.eval()
    rng = np.random.default_rng(seed)
    S, Y, G, rows = [], [], [], []
    nll, nll_n = 0.0, 0
    hp = {k: [] for k in range(len(H_NAMES))}
    n = 0
    for f in fires:
        for i, gap in f.pairs:
            win = EV.window(f, i)
            x, wx, t_lo, t_hi, valid, y, m, _ = inputs(f, i, win, groups)
            h, w = y.shape
            lam = model(torch.from_numpy(TH.pad8(x))[None].float().to(device),
                        torch.from_numpy(TH.pad8(wx))[None].float().to(device))[:, :, :h, :w]
            s = HZ.logit_burned_by(lam, torch.full((1, h, w), float(gap), device=device))[0].cpu().numpy()
            S.append(s[m]); Y.append(y[m]); G.append(np.full(int(m.sum()), np.digitize(gap, EV.GAP_EDGES), np.int8))
            tl, th, vv = (torch.from_numpy(a)[None].to(device) for a in (t_lo, t_hi, valid))
            nll += float(HZ.survival_nll(lam, tl, th, vv, reduce=False).sum())
            nll_n += int(valid.sum())
            rel = (f.times - f.times[i]).total_seconds().values / 3600
            arr = f.arrival[win[0]:win[1], win[2]:win[3]]
            for a in range(i + 1, len(f.times)):
                if rel[a] > HORIZON:
                    break
                L = HZ.cumhaz_at(lam, torch.full((1, h, w), float(rel[a]), device=device))[0].cpu().numpy()
                by = (arr > i) & (arr <= a)
                pop = valid & ~(by & np.isinf(t_hi))                # untrusted or past-horizon burns left out
                idx = np.flatnonzero(pop)
                if len(idx) > cal_cells:
                    idx = rng.choice(idx, cal_cells, replace=False)
                hp[int(np.digitize(rel[a], H_EDGES))].append((1 - np.exp(-L.ravel()[idx]), by.ravel()[idx]))
            rows.append(dict(fire=f.fid, year=f.year, i=i, gap_h=gap, new=int(y[m].sum()), cells=int(m.sum())))
            n += 1
            if max_pairs and n >= max_pairs:
                break
        if max_pairs and n >= max_pairs:
            break
    s, y, g = np.concatenate(S), np.concatenate(Y), np.concatenate(G)
    out = TH.metrics(s, y, thr)                              # thr from validation; None picks the best F1 here
    prob = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
    out["interval_nll"] = nll / max(nll_n, 1)
    out["by_gap"] = [dict(group=nm, **_cal(prob[g == j], y[g == j])) for j, nm in enumerate(EV.GAP_NAMES) if (g == j).sum()]
    out["by_horizon"] = [dict(group=H_NAMES[k], **_cal(np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v]).astype(float)))
                         for k, v in hp.items() if v]
    out["gap_over_horizon_cells"] = int((g == 3).sum())      # scored by carrying the last bin's hazard on
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="hazard")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=32, help="hazard head width")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--max-fires", type=int)
    ap.add_argument("--max-pairs", type=int, help="quick runs: score only this many pairs per split")
    ap.add_argument("--catalog")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop-km", type=float, default=24.0)
    ap.add_argument("--u3-time", type=int, default=1)
    ap.add_argument("--u3-attn", type=int, default=0)
    ap.add_argument("--fuel-moisture", type=int, default=1)
    ap.add_argument("--fuel-frac", type=int, default=0)
    ap.add_argument("--wind-terrain", type=int, default=0)
    ap.add_argument("--edit-slack-h", type=float, default=IV.EDIT_SLACK_H,
                    help="an edit-time stamp may be this many hours after the real mapping (only the earlier bound moves); "
                         "use about the 90th percentile from pipeline/measure_edit_delay.py")
    ap.add_argument("--mapped-slack-h", type=float, default=IV.MAPPED_SLACK_H, help="rounding of real mapping times (+-)")
    a = ap.parse_args()
    IV.EDIT_SLACK_H, IV.MAPPED_SLACK_H = a.edit_slack_h, a.mapped_slack_h      # set before the loader workers start
    runs = Path(a.data) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    ck = runs / f"{a.name}.ckpt"
    torch.manual_seed(a.seed)
    pca = TH.fit_pca(a.data, runs)
    fires = TH.load_fires(a.data, pca, None, a.catalog)
    split = {s: [f for f in fires if D.year_split(f.year) == s] for s in ("train", "val", "test")}
    if a.max_fires:
        k = a.max_fires
        split = {"train": split["train"][:max(1, k * 3 // 5)], "val": split["val"][:max(1, k // 5)], "test": split["test"][:max(1, k // 5)]}
    from collections import Counter
    src = Counter(t for f in fires for t in getattr(f, "time_src", []))
    print(f"  slack: edit stamps up to {IV.EDIT_SLACK_H:g} h after mapping, mapped +-{IV.MAPPED_SLACK_H:g} h, bare dates the whole local day; "
          "perimeter time sources: "
          + ", ".join(f"{k} {v}" for k, v in src.most_common()), flush=True)
    D.PHYSICS_V2 = True                                  # unet3 uses the crown-fire / ember group
    D.WIND_TERRAIN = D.wind_file(a.wind_terrain)
    D.FUEL_FRAC = bool(a.fuel_frac)
    D.TEMPORAL = bool(a.u3_time)
    D.FM_ON = D.FUEL_V2 = bool(a.fuel_moisture)
    groups = TH.groups_used(set(), v2=True, t3=D.TEMPORAL, origin=False, fm=bool(a.fuel_moisture))
    n_in = sum(D.GROUPS[g] for g in groups)
    res = {f.res_m for f in fires}
    if len(res) != 1:
        raise SystemExit(f"mixed grid resolutions in {a.data}: {sorted(res)}")
    res_m = res.pop()
    TH.CROP_KM = a.crop_km
    TH.set_grid(res_m / 1000)
    print(f"[{a.name}] hazard (unet3 trunk, {HZ.N_BINS} bins x {HZ.BIN_H:g} h) inputs={n_in} ch + {D.Fire.BIN_CH} per bin  "
          f"fires train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}  crops {TH.CROP} cells", flush=True)
    model = build(n_in, a.width, bool(a.u3_attn), D.TEMPORAL, a.device, a.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
    start, best, hist = 0, (-1.0, None, None), []
    if ck.exists():
        c = torch.load(ck, weights_only=False, map_location=a.device)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        start, best, hist = c["epoch"], c["best"], c["hist"]
        print(f"  resumed after epoch {start}", flush=True)
    ds = HazardCrops(split["train"], groups, a.steps * a.batch, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, persistent_workers=a.workers > 0)
    for ep in range(start, a.epochs):
        model.train()
        t0, tl = time.time(), 0.0
        for x, wx, t_lo, t_hi, valid in dl:
            x, wx, t_lo, t_hi, valid = (t.to(a.device, non_blocking=True) for t in (x, wx, t_lo, t_hi, valid))
            lam = model(x, wx.float())
            loss = HZ.survival_nll(lam, t_lo, t_hi, valid)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
        rec = {"epoch": ep + 1, "loss": tl / a.steps, "min": (time.time() - t0) / 60}
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            mv, _ = evaluate(model, split["val"], groups, a.device, a.max_pairs)
            rec.update(val_ap=mv["ap"], val_nll=mv["interval_nll"])
            if mv["ap"] > best[0]:
                best = (mv["ap"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, mv)
        hist.append(rec)
        print(f"  epoch {ep + 1}/{a.epochs} loss {rec['loss']:.4f}" +
              (f"  val AP {rec['val_ap']:.3f} interval NLL {rec['val_nll']:.4f}" if "val_ap" in rec else "") +
              f"  ({rec['min']:.1f} min)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best,
                    "hist": hist, "epoch": ep + 1, "data_version": D.DATA_VERSION}, ck)
    model.load_state_dict(best[1])
    mt, rows = evaluate(model, split["test"], groups, a.device, a.max_pairs, thr=best[2]["thr"])
    torch.save(model.state_dict(), runs / f"{a.name}.pt")
    meta = {"name": a.name, "mode": "hazard", "drop": [], "origin": False, "wind_terrain": int(a.wind_terrain),
            "fuel_frac": bool(a.fuel_frac), "fuel_moisture": bool(a.fuel_moisture), "fm_physics": False,
            "data_version": D.DATA_VERSION, "input_version": D.INPUT_VERSION, "weather": D.WEATHER_MODE,
            "arch": {"time": D.TEMPORAL, "attn": bool(a.u3_attn)}, "width": a.width, "hidden": a.hidden,
            "n_bins": HZ.N_BINS, "bin_h": HZ.BIN_H,
            "edit_slack_h": IV.EDIT_SLACK_H, "mapped_slack_h": IV.MAPPED_SLACK_H, "val": best[2], "test": mt, "history": hist, "pairs": rows,
            "params": sum(p.numel() for p in model.parameters())}
    json.dump(meta, open(runs / f"{a.name}.json", "w"), indent=1, default=float)
    print(f"[{a.name}] TEST AP {mt['ap']:.3f} F1 {mt['f1']:.3f} area {mt['area_ratio']:.2f} (old-style, next perimeter)  interval NLL {mt['interval_nll']:.4f}  "
          f"Brier {mt.get('brier', float('nan')):.4f} ECE {mt.get('ece', float('nan')):.4f}", flush=True)
    for key in ("by_gap", "by_horizon"):
        print(f"  {key[3:]}: " + "  ".join(f"{r['group']} ECE {r.get('ece', float('nan')):.3f} pred {r.get('mean_pred', float('nan')):.3f} "
                                           f"obs {r.get('observed', float('nan')):.3f}" for r in mt[key]), flush=True)
    print("  official scores (same protocol as 0.261): run evaluate.py on this model", flush=True)


if __name__ == "__main__":
    main()
