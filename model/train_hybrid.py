"""
Train and evaluate on the multi-year dataset.

  --mode hybrid   network corrects the Rothermel spread rate; solver -> arrival time -> burn map (the plan)
  --mode unet     same inputs, the network predicts the burn map directly (plain U-Net baseline)
  --mode physics  no learning: Rothermel + solver, effective burning time calibrated on validation

Split by year (train 2014-2022, val 2023, test 2024-2025). Checkpoints every epoch and resumes.
Every model is scored on exactly the same perimeter pairs and cells.

  python train_hybrid.py --data ~/scratch/wildfire/my --name hybrid_v1 --mode hybrid --device cuda
  python train_hybrid.py --data ... --name ablate_no_tessera --drop tessera     # ablation
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
from hybrid import Hybrid, Hybrid2, loss_fn  # noqa: E402
from nets import UNet  # noqa: E402

CROP_KM = 24.0                   # training crops cover 24 km (160 cells at 150 m, 320 at 75 m)
EDGE_KM = 2.4                    # crop edge excluded from the loss (fire may enter from outside)
EVAL_PAD_KM = 9.6                # evaluation window: current fire + this margin
EVAL_MAX_KM = 115.0              # very large fires: window this size centred on the fire
CROP, EDGE, EVAL_PAD, EVAL_MAX = 160, 16, 64, 768


def set_grid(cell_km):
    """Turn the km constants into cells for this dataset's resolution (multiples of 8 for the U-Net)."""
    global CROP, EDGE, EVAL_PAD, EVAL_MAX
    CROP = int(round(CROP_KM / cell_km / 8)) * 8
    EDGE = int(round(EDGE_KM / cell_km))
    EVAL_PAD = int(round(EVAL_PAD_KM / cell_km))
    EVAL_MAX = int(round(EVAL_MAX_KM / cell_km / 8)) * 8
HIDE_P = 0.15                    # chance of hiding each optional group during training


def excluded_fires(catalog):
    """Fires rated bad by qc_fires.py (incl. the post-build satellite check) or dropped on the QC page."""
    import pandas as pd
    cat = Path(catalog)
    bad = set()
    if (cat / "qc_fires.csv").exists():
        q = pd.read_csv(cat / "qc_fires.csv")
        bad = set(q.loc[q["rating"] == "bad", "fire_id"])
    if (cat / "overrides.csv").exists():
        o = pd.read_csv(cat / "overrides.csv")
        bad = (bad - set(o.loc[o.verdict == "keep", "fire_id"])) | set(o.loc[o.verdict == "drop", "fire_id"])
    return bad


def load_fires(root, pca, max_fires=None, catalog=None):
    dirs = sorted(p.parent for p in Path(root, "fires").glob("*/labels.npz"))
    need = ["labels", "terrain", "weather", "viirs"]
    dirs = [d for d in dirs if all((d / f"{n}.npz").exists() for n in need)]
    bad = excluded_fires(catalog or Path(root).parent / "catalog")
    if bad:
        n0 = len(dirs)
        dirs = [d for d in dirs if d.name not in bad]
        print(f"  QC: left out {n0 - len(dirs)} fires rated bad or dropped by the team", flush=True)
    if max_fires:
        dirs = dirs[:max_fires]
    fires = [D.Fire(d, pca) for d in dirs]
    return [f for f in fires if f.pairs]


def fit_pca(root, out, n=16, per_fire=300, seed=0):
    p = Path(out) / "tessera_pca.npz"
    if p.exists():
        z = np.load(p)
        return z["comp"], z["mean"]
    rng = np.random.default_rng(seed)
    X = []
    for d in sorted(Path(root, "fires").glob("*/tessera.npz")):
        g = json.loads((d.parent / "grid.json").read_text())
        if D.year_split(g["year"]) != "train":
            continue
        z = np.load(d)
        if bool(z["missing"]):
            continue
        e, cov = z["embedding"], z["cover"]
        rr, cc = np.nonzero(cov)
        if not len(rr):
            continue
        k = rng.choice(len(rr), min(per_fire, len(rr)), replace=False)
        X.append(e[:, rr[k], cc[k]].T.astype(np.float32))
    if not X:
        return None
    X = np.concatenate(X)
    mean = X.mean(0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    comp = vt[:n]
    np.savez(p, comp=comp, mean=mean, explained=(s[:n] ** 2).sum() / (s ** 2).sum())
    print(f"Tessera PCA: {len(X)} pixels, {n} components explain {(s[:n] ** 2).sum() / (s ** 2).sum():.1%}")
    return comp, mean


V2_MODES = ("hybrid2", "physics2")


def groups_used(drop, v2=False):
    """Input groups in channel order; the physics-v2 group only for the v2 modes."""
    return [g for g in D.GROUPS if g not in drop and (v2 or g not in D.V2_GROUPS)]


def phys_stack(phys):
    keys = ["r_head", "ecc", "head_x", "head_y"] + (["r_crown", "ember"] if "r_crown" in phys else [])
    return np.stack([phys[k] for k in keys])


def pad8(a, size=None):
    h, w = a.shape[-2:]
    H, W = (size, size) if size else (-(-h // 8) * 8, -(-w // 8) * 8)
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, H - h), (0, W - w)])


class Crops(torch.utils.data.Dataset):
    """Random training crops centred on the fire front of a random perimeter pair."""

    def __init__(self, fires, groups, n_per_epoch, seed=0):
        self.items = [(fi, i, gap) for fi, f in enumerate(fires) for i, gap in f.pairs]
        self.fires, self.groups, self.n = fires, groups, n_per_epoch
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, k, int(time.time() * 1e3) % 10 ** 9))
        fi, i, gap = self.items[rng.integers(len(self.items))]
        f = self.fires[fi]
        fc = f.front_cells(i)
        r, c = fc[rng.integers(len(fc))] if len(fc) else (f.H // 2, f.W // 2)
        jit = CROP // 7
        r0 = int(np.clip(r - CROP // 2 + rng.integers(-jit, jit + 1), 0, max(f.H - CROP, 0)))
        c0 = int(np.clip(c - CROP // 2 + rng.integers(-jit, jit + 1), 0, max(f.W - CROP, 0)))
        win = (r0, min(r0 + CROP, f.H), c0, min(c0 + CROP, f.W))
        hide = tuple(g for g in D.OPTIONAL if rng.random() < HIDE_P)
        feats, phys, y, m, b0 = f.sample(i, gap, win, hide)
        m = m.copy()
        m[:EDGE] = m[-EDGE:] = False
        m[:, :EDGE] = m[:, -EDGE:] = False
        x = D.stack(feats, self.groups)
        if rng.random() < 0.5:   # flips (east-west), fixing direction channels
            x, y, m, b0 = x[..., ::-1], y[..., ::-1], m[..., ::-1], b0[..., ::-1]
            phys = {k: v[..., ::-1] for k, v in phys.items()}
            phys["head_x"] = -phys["head_x"]
            x = x.copy()
            for g, chans in flip_x_channels(self.groups).items():
                for ch in chans:
                    x[ch] *= -1
        P = phys_stack(phys)
        return (torch.from_numpy(pad8(x, CROP).astype(np.float32)), torch.from_numpy(pad8(P, CROP).astype(np.float32)),
                torch.from_numpy(pad8(y[None], CROP)[0].copy()), torch.from_numpy(pad8(m[None], CROP)[0].copy()),
                torch.from_numpy(pad8(b0[None], CROP)[0].copy()), torch.tensor(gap / 24, dtype=torch.float32))


def flip_x_channels(groups):
    """Channel indices whose sign flips under an east-west mirror."""
    out, base = {}, 0
    xs = {"terrain": [2], "hrrr": [0], "physics": [2], "weather": []}
    for g in groups:
        if g in xs:
            out[g] = [base + j for j in xs[g]]
        base += D.GROUPS[g]
    return out


def build_model(mode, n_in, width, device, dx=150.0):
    if mode == "unet":
        return UNet(n_in, 1, width).to(device)
    if mode == "hybrid":
        return Hybrid(UNet(n_in, 2, width), dx=dx, max_iter=int(400 * 150 / dx)).to(device)
    if mode == "hybrid2":
        return Hybrid2(UNet(n_in, 3, width), dx=dx, max_iter=int(400 * 150 / dx)).to(device)
    if mode == "physics2":
        return Hybrid2(ZeroNet(3), dx=dx, max_iter=int(400 * 150 / dx)).to(device)
    return Hybrid(ZeroNet(), dx=dx, max_iter=int(400 * 150 / dx)).to(device)


class ZeroNet(torch.nn.Module):
    """Physics only: zero correction, zero floor (and zero spotting correction for v2)."""

    def __init__(self, channels=2):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        z = torch.zeros(x.shape[0], self.channels, *x.shape[2:], device=x.device)
        z[:, 1] = -20
        return z


def forward(model, mode, x, P, b0, gap_days):
    if mode == "unet":
        return model(x)[:, 0], None
    phys = {"r_head": P[:, 0], "ecc": P[:, 1], "head_x": P[:, 2], "head_y": P[:, 3]}
    if P.shape[1] >= 6:
        phys["r_crown"], phys["ember"] = P[:, 4], P[:, 5]
    logits, T, c = model(x, phys, b0.bool(), gap_days)
    return logits, c


def edge_error_km(b0, pred_new, true_new, cell_km=0.15):
    """Symmetric mean distance (km) between the predicted and the actual NEW fire edge at the next
    perimeter: only edge cells that moved off the current perimeter count (the shared old edge and
    unburned islands are ignored). If one side did not grow, its edge is the current perimeter."""
    from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, distance_transform_edt
    B = binary_fill_holes(b0)
    old_edge = binary_dilation(B & ~binary_erosion(B), iterations=1)
    P, T = binary_fill_holes(B | pred_new), binary_fill_holes(B | true_new)
    eP, eT = P & ~binary_erosion(P), T & ~binary_erosion(T)
    nP, nT = eP & ~old_edge, eT & ~old_edge          # edge that moved
    if not nP.any() and not nT.any():
        return float("nan")                           # neither grew: nothing to measure
    if not nP.any():
        nP = eP                                       # predicted no growth: its edge is the current one
    if not nT.any():
        nT = eT
    dT = distance_transform_edt(~eT) * cell_km
    dP = distance_transform_edt(~eP) * cell_km
    return float((dT[nP].mean() + dP[nT].mean()) / 2)


@torch.no_grad()
def evaluate(model, mode, fires, groups, device, max_pairs=None, thr=None):
    """thr (from validation) also computes each pair's fire-edge error in km."""
    """Whole-region scoring of every pair; returns pooled scores, labels and per-pair records."""
    model.eval()
    S, Y, rows = [], [], []
    n = 0
    for f in fires:
        for i, gap in f.pairs:
            b = f.burned(i)
            rr, cc = np.nonzero(b)
            r0, r1 = max(rr.min() - EVAL_PAD, 0), min(rr.max() + EVAL_PAD + 1, f.H)
            c0, c1 = max(cc.min() - EVAL_PAD, 0), min(cc.max() + EVAL_PAD + 1, f.W)
            if (r1 - r0) > EVAL_MAX or (c1 - c0) > EVAL_MAX:   # very large fire: centre window on the front
                rm, cm = int(rr.mean()), int(cc.mean())
                r0, r1 = max(rm - EVAL_MAX // 2, 0), min(rm + EVAL_MAX // 2, f.H)
                c0, c1 = max(cm - EVAL_MAX // 2, 0), min(cm + EVAL_MAX // 2, f.W)
            feats, phys, y, m, b0 = f.sample(i, gap, (r0, r1, c0, c1))
            x = torch.from_numpy(pad8(D.stack(feats, groups)))[None].to(device)
            P = torch.from_numpy(pad8(phys_stack(phys)))[None].to(device)
            bb = torch.from_numpy(pad8(b0[None]))[None, 0].to(device)
            logit, _ = forward(model, mode, x.float(), P.float(), bb, torch.tensor([gap / 24], device=device))
            s = logit[0, :y.shape[0], :y.shape[1]].float().cpu().numpy()
            S.append(s[m])
            Y.append(y[m])
            rec = dict(fire=f.fid, year=f.year, i=i, gap_h=gap, new=int(y[m].sum()), cells=int(m.sum()))
            if thr is not None:
                rec["edge_km"] = edge_error_km(b0, (s >= thr) & m, y & m, f.cell_km)
                rec["edge_km_persist"] = edge_error_km(b0, np.zeros_like(b0), y & m, f.cell_km)   # "no growth" reference
            rows.append(rec)
            n += 1
            if max_pairs and n >= max_pairs:
                return np.concatenate(S), np.concatenate(Y), rows
    return np.concatenate(S), np.concatenate(Y), rows


def mode_probs(s):
    """Scores are logits for the U-Net and hybrid; the physics-only model's are also logits of the
    same burn-by-horizon form, so calibration is reported for all of them."""
    return True


def edge_summary(rows):
    e = np.array([r["edge_km"] for r in rows if "edge_km" in r and np.isfinite(r["edge_km"])])
    e0 = np.array([r["edge_km_persist"] for r in rows if "edge_km_persist" in r and np.isfinite(r["edge_km_persist"])])
    if not len(e):
        return {}
    return {"edge_km_mean": float(e.mean()), "edge_km_median": float(np.median(e)),
            "edge_km_no_growth_mean": float(e0.mean()) if len(e0) else float("nan")}


def metrics(s, y, thr=None):
    from sklearn.metrics import average_precision_score
    y = y.astype(bool)
    out = {"n": int(len(y)), "pos_rate": float(y.mean()), "ap": float(average_precision_score(y, s)) if y.any() else float("nan")}
    if thr is None:
        qs = np.unique(np.quantile(s, np.linspace(0.5, 0.9995, 400)))
        f1s = [2 * ((s >= q) & y).sum() / ((s >= q).sum() + y.sum() + 1e-9) for q in qs]
        thr = float(qs[int(np.argmax(f1s))])
    if mode_probs(s):
        prob = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
        out["brier"] = float(np.mean((prob - y) ** 2))
        bins = np.minimum((prob * 10).astype(int), 9)
        rel = []
        for b in range(10):
            sel = bins == b
            if sel.sum() >= 50:
                rel.append([round(b / 10 + 0.05, 2), round(float(prob[sel].mean()), 4), round(float(y[sel].mean()), 4), int(sel.sum())])
        out["reliability"] = rel           # [bin centre, mean predicted, observed burn rate, cells]
        out["ece"] = float(sum(abs(r[1] - r[2]) * r[3] for r in rel) / max(sum(r[3] for r in rel), 1))
    p = s >= thr
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    out.update(thr=thr, f1=float(2 * tp / (2 * tp + fp + fn + 1e-9)), iou=float(tp / (tp + fp + fn + 1e-9)),
               precision=float(tp / (tp + fp + 1e-9)), recall=float(tp / (tp + fn + 1e-9)),
               area_ratio=float(p.sum() / max(y.sum(), 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="hybrid")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "unet", "physics", "hybrid2", "physics2"],
                    help="hybrid2 / physics2 add crown fire and spotting (physics v2)")
    ap.add_argument("--drop", nargs="*", default=[], help="input groups to leave out (ablations)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=600, help="training batches per epoch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--max-fires", type=int)
    ap.add_argument("--catalog", help="catalog folder with qc_fires.csv (default: <data>/../catalog)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-years", default="2023", help="comma list (default 2023)")
    ap.add_argument("--test-years", default="2024,2025", help="comma list (default 2024,2025)")
    ap.add_argument("--seed", type=int, default=0, help="different seeds = ensemble members")
    a = ap.parse_args()
    D.VAL_YEARS = {int(y) for y in a.val_years.split(",")}
    D.TEST_YEARS = {int(y) for y in a.test_years.split(",")}
    runs = Path(a.data) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    ck = runs / f"{a.name}.ckpt"
    if ck.exists() and "data_version" not in torch.load(ck, weights_only=False, map_location="cpu"):
        D.DATA_VERSION = 1      # started before the label/GOES clean-up: finish it on the data it began with
    print(f"  data version {D.DATA_VERSION}", flush=True)
    torch.manual_seed(a.seed)
    pca = fit_pca(a.data, runs)
    fires = load_fires(a.data, pca, None, a.catalog)
    split = {s: [f for f in fires if D.year_split(f.year) == s] for s in ("train", "val", "test")}
    if a.max_fires:   # quick runs: a few fires from every split, so validation and test are never empty
        n = a.max_fires
        split = {"train": split["train"][:max(1, n * 3 // 5)], "val": split["val"][:max(1, n // 5)],
                 "test": split["test"][:max(1, n // 5)]}
        fires = split["train"] + split["val"] + split["test"]
    D.PHYSICS_V2 = a.mode in V2_MODES
    groups = groups_used(set(a.drop), v2=D.PHYSICS_V2)
    n_in = sum(D.GROUPS[g] for g in groups)
    print(f"[{a.name}] mode={a.mode} inputs={n_in} ch  fires train/val/test = "
          f"{len(split['train'])}/{len(split['val'])}/{len(split['test'])}  pairs = "
          f"{sum(len(f.pairs) for f in split['train'])}/{sum(len(f.pairs) for f in split['val'])}/"
          f"{sum(len(f.pairs) for f in split['test'])}", flush=True)
    res = {f.res_m for f in fires}
    if len(res) != 1:
        raise SystemExit(f"mixed grid resolutions in {a.data}: {sorted(res)}")
    res_m = res.pop()
    set_grid(res_m / 1000)
    print(f"  grid {res_m:.0f} m: crops {CROP} cells, eval margin {EVAL_PAD} cells", flush=True)
    model = build_model(a.mode, n_in, a.width, a.device, dx=res_m)

    if a.mode == "physics":
        sv, yv, _ = evaluate(model, a.mode, split["val"], groups, a.device)
        mv = metrics(sv, yv)
        st, yt, rows = evaluate(model, a.mode, split["test"], groups, a.device, thr=mv["thr"])
        mt = metrics(st, yt, mv["thr"])
        mt.update(edge_summary(rows))
        json.dump({"name": a.name, "mode": a.mode, "val": mv, "test": mt, "pairs": rows}, open(runs / f"{a.name}.json", "w"), indent=1)
        print(f"[{a.name}] TEST AP {mt['ap']:.3f} F1 {mt['f1']:.3f} IoU {mt['iou']:.3f} area {mt['area_ratio']:.2f}  "
              f"edge {mt.get('edge_km_mean', float('nan')):.2f} km (no-growth {mt.get('edge_km_no_growth_mean', float('nan')):.2f})  "
              f"Brier {mt.get('brier', float('nan')):.4f} ECE {mt.get('ece', float('nan')):.4f}")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
    start, best, hist = 0, (-1.0, None, None), []
    if ck.exists():
        c = torch.load(ck, weights_only=False, map_location=a.device)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        start, best, hist = c["epoch"], c["best"], c["hist"]
        print(f"  resumed after epoch {start}", flush=True)
    ds = Crops(split["train"], groups, a.steps * a.batch, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, persistent_workers=a.workers > 0)
    for ep in range(start, a.epochs):
        model.train()
        t0, tl = time.time(), 0.0
        for x, P, y, m, b0, gap in dl:
            x, P, y, m, b0, gap = (t.to(a.device, non_blocking=True) for t in (x, P, y, m, b0, gap))
            logits, c = forward(model, a.mode, x, P, b0, gap)
            loss = loss_fn(logits, y, m.float(), corr=c)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
        rec = {"epoch": ep + 1, "loss": tl / a.steps, "min": (time.time() - t0) / 60}
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            sv, yv, _ = evaluate(model, a.mode, split["val"], groups, a.device)
            mv = metrics(sv, yv)
            rec.update(val_ap=mv["ap"], val_f1=mv["f1"])
            if mv["ap"] > best[0]:
                best = (mv["ap"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, mv)
        hist.append(rec)
        print(f"  epoch {ep + 1}/{a.epochs} loss {rec['loss']:.4f}" +
              (f"  val AP {rec['val_ap']:.3f} F1 {rec['val_f1']:.3f}" if "val_ap" in rec else "") +
              f"  ({rec['min']:.1f} min)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best,
                    "hist": hist, "epoch": ep + 1, "data_version": D.DATA_VERSION}, ck)
    model.load_state_dict(best[1])
    st, yt, rows = evaluate(model, a.mode, split["test"], groups, a.device, thr=best[2]["thr"])
    mt = metrics(st, yt, best[2]["thr"])
    mt.update(edge_summary(rows))
    torch.save(model.state_dict(), runs / f"{a.name}.pt")
    extra = {}
    if a.mode in ("hybrid",) + V2_MODES:
        extra = {"effective_minutes_per_day": model.log_h.exp().item(), "softness_min": model.log_tau.exp().item()}
    if a.mode in V2_MODES:
        extra.update(crown_weight=torch.sigmoid(model.crown_logit).item(), spot_alpha=model.log_spot_a.exp().item(),
                     spot_beta=model.spot_b.item())
        print(f"  learned: crown weight {extra['crown_weight']:.2f}, spotting alpha {extra['spot_alpha']:.2f} "
              f"beta {extra['spot_beta']:.2f}, burning {extra['effective_minutes_per_day']:.0f} min/day", flush=True)
    json.dump({"name": a.name, "mode": a.mode, "drop": a.drop, "data_version": D.DATA_VERSION, "val": best[2], "test": mt, "history": hist,
               "pairs": rows, **extra}, open(runs / f"{a.name}.json", "w"), indent=1)
    print(f"[{a.name}] TEST AP {mt['ap']:.3f} F1 {mt['f1']:.3f} IoU {mt['iou']:.3f} P {mt['precision']:.3f} "
          f"R {mt['recall']:.3f} area {mt['area_ratio']:.2f}  edge {mt.get('edge_km_mean', float('nan')):.2f} km "
          f"(no-growth {mt.get('edge_km_no_growth_mean', float('nan')):.2f})  Brier {mt.get('brier', float('nan')):.4f} "
          f"ECE {mt.get('ece', float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
