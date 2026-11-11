"""
Score trained models on the test years, with everything the literature review asked for.

For every model (or an ensemble of models) and every perimeter pair it reports:
  raw          the model as trained
  calibrated   per-fire calibration at forecast time: two scalars are fitted to the fire's most
               recent observed growth (the previous perimeter pair, all known at forecast time),
               then used for the next forecast.
                 hybrid / physics: a = log spread-rate multiplier, s = log softness multiplier
                                   (arrival times scale exactly with 1/rate, so no re-solve)
                 unet:             logit bias b and log temperature s
  distance     baseline: score = -(distance from the current perimeter) / gap days
  no-growth    baseline: predicts nothing new

Metrics, on the same cells as training's evaluation:
  pooled AP / F1 / IoU / precision / recall / area ratio at the validation-chosen threshold,
  Brier and ECE, recall at matched area (flag exactly as many cells as actually burned),
  the TS-SatFire per-window F1 / IoU (each pair scored alone, empty-vs-empty = 1, mean per fire,
  then mean over fires), the fire-edge error in km, and breakdowns by fuel family, HRRR wind
  speed band and gap length.

Checkpoints: runs/<name>.pt (finished) or the best weights inside runs/<name>.ckpt (still training).

  python evaluate.py --data ~/scratch/wildfire/my --model physics_v1:physics --out physics_eval
  python evaluate.py --data ... --model hybrid_v1 --out hybrid_v1_eval
  python evaluate.py --data ... --model hybrid_v1 hybrid_s1 hybrid_s2 --out hybrid_ens3   # ensemble
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import train_hybrid as TH  # noqa: E402
from physics import solver as S  # noqa: E402
from hybrid import spread_or_spot  # noqa: E402

POS_WEIGHT = 8.0                 # same weighting as the training loss, so calibrated logits keep their scale
CAL_A = 1.0                      # calibration limit: spread rate x/÷ e (U-Net: logit bias +-1)
CAL_HISTORY = 3                  # fit on up to the 3 previous pairs of the fire
CAL_MIN_BURNED = 30              # skip calibration when they burned fewer cells than this
CAL_SHRINK_BURNED = 300          # a is scaled by n / (n + 300): weak evidence -> small adjustment
CAL_STEPS, CAL_MAX_CELLS = 60, 200_000
FUEL_NAMES = [n for n, _, _ in D.FUEL_FAMILIES] + ["unknown"]
WIND_EDGES = [3, 6, 9]           # m/s: <3, 3-6, 6-9, >=9  (~<11, 11-22, 22-32, >32 km/h)
WIND_NAMES = ["<3 m/s", "3-6 m/s", "6-9 m/s", ">=9 m/s"]
GAP_EDGES = [12, 36, 72]         # hours
GAP_NAMES = ["<=12 h", "12-36 h", "36-72 h", ">72 h"]
GROW_EDGES = [0.5, 5, 50]        # km² of new burn in the pair
GROW_NAMES = ["growth <0.5 km²", "0.5-5 km²", "5-50 km²", ">50 km²"]
SIZE_EDGES = [40, 400]           # km² burned by the end of the fire (~10k and ~100k acres)
SIZE_NAMES = ["fire <40 km²", "40-400 km²", ">400 km²"]


# ---------------------------------------------------------------------------- models
class Member:
    """One trained model. Produces, per pair, a cacheable 'state' from which raw and calibrated
    logits are cheap to compute."""

    def __init__(self, runs, spec, n_in, groups_all, device, dx, width):
        name, _, rest = spec.partition(":")
        mode = rest.split(":")[0] if rest else None
        meta = runs / f"{name}.json"
        drop = []
        if meta.exists():
            j = json.loads(meta.read_text())
            mode = mode or j.get("mode")
            drop = j.get("drop", []) or []
        if ":" in rest:
            drop = [g for g in rest.split(":")[1].split("+") if g]
        if mode is None:
            mode = next((m for m in ("physics2", "hybrid2", "unet3", "unet2", "physics", "unet", "hybrid") if name.startswith(m)), "hybrid")
        self.name, self.mode, self.device = name, mode, device
        if mode == "unet3":                               # architecture read from the weights
            pt, ck = runs / f"{name}.pt", runs / f"{name}.ckpt"
            if pt.exists():
                sd, src = torch.load(pt, map_location=device, weights_only=False), "final weights"
            elif ck.exists():
                c = torch.load(ck, map_location=device, weights_only=False)
                best = c.get("best")
                sd, src = (best[1], f"best so far (val AP {best[0]:.3f})") if best and best[1] is not None else (c["model"], "latest")
            else:
                raise SystemExit(f"no checkpoint for {name} in {runs}")
            arch = {"time": any(k.startswith("temporal.") for k in sd), "attn": any(k.startswith("bottleneck.") for k in sd)}
            if arch["time"]:
                D.TEMPORAL = True                         # the loader now also builds the day-by-day history
            self.groups = TH.groups_used(set(drop), v2=True, t3=arch["time"])
            n_in = sum(D.GROUPS[g] for g in self.groups)
            self.model = TH.build_model(mode, n_in, int(sd["head.weight"].shape[1]), device, dx=dx, arch=arch)
            self.model.load_state_dict(sd)
            self.model.eval()
            print(f"  {name}: unet3 (time {'on' if arch['time'] else 'off'}, attention {'on' if arch['attn'] else 'off'}), {src}", flush=True)
            return
        v2 = mode in TH.V2_MODES
        self.groups = [g for g in TH.groups_used(set(drop), v2=v2)]
        n_in = sum(D.GROUPS[g] for g in self.groups)
        if mode == "physics":
            self.model = TH.build_model(mode, n_in, width, device, dx=dx)
            print(f"  {name}: physics only (no weights)", flush=True)
        else:
            pt, ck = runs / f"{name}.pt", runs / f"{name}.ckpt"
            if pt.exists():
                sd, src = torch.load(pt, map_location=device, weights_only=False), "final weights"
            elif ck.exists():
                c = torch.load(ck, map_location=device, weights_only=False)
                best = c.get("best")
                if best is not None and best[1] is not None:
                    sd, src = best[1], f"best so far (val AP {best[0]:.3f}, after epoch {c['epoch']})"
                else:
                    sd, src = c["model"], f"latest (epoch {c['epoch']}, no validation yet)"
            elif mode == "physics2":
                sd, src = TH.build_model(mode, n_in, width, device, dx=dx).state_dict(), "default settings (not fitted)"
            else:
                raise SystemExit(f"no checkpoint for {name} in {runs}")
            if mode == "physics2":                        # learned scalars only, no network
                self.model = TH.build_model(mode, n_in, width, device, dx=dx)
            else:
                head = next(v for k, v in sd.items() if k.endswith("head.weight"))
                first = next(v for k, v in sd.items() if v.ndim == 4)
                if first.shape[1] != n_in:
                    raise SystemExit(f"{name} was trained on {first.shape[1]} input channels, the data gives {n_in} "
                                     f"(add its dropped groups: {name}:{mode}:group1+group2)")
                self.model = TH.build_model(mode, n_in, int(head.shape[1]), device, dx=dx)   # width read from the weights
            self.model.load_state_dict(sd)
            print(f"  {name}: {mode}, {src}", flush=True)
        self.model.eval()

    @torch.no_grad()
    def state(self, feats, phys, b0, gap_h):
        x = torch.from_numpy(TH.pad8(D.stack(feats, self.groups))).float()[None].to(self.device)
        h, w = b0.shape
        if self.mode in TH.UNET_MODES:
            return {"logit": self.model(x)[0, 0, :h, :w].float()}
        P = torch.from_numpy(TH.pad8(TH.phys_stack(phys))).float().to(self.device)
        bb = torch.from_numpy(TH.pad8(b0[None]))[0].to(self.device).bool()
        m = self.model
        z = None
        if self.mode in TH.HYBRID_V2:
            R, _, g = m.rate2(x, P[None, 0], P[None, 4])
            z = m.ignitions(P[None, 5], torch.tensor([gap_h / 24], device=self.device), g)[0, :h, :w].float()
        else:
            R, _ = m.rate(x, P[None, 0])
        costs = S.step_costs(R[0], P[1], P[2], P[3], m.dx)
        horizon = float(m.log_h.exp()) * gap_h / 24
        # wide horizon so arrival times stay finite for any rate multiplier calibration may pick
        T = S.arrival_time(costs, bb, max_iter=m.max_iter, t_max=horizon * 1.5 * math.exp(1.5))
        st = {"T": T[:h, :w].clamp(max=1e5).float(), "horizon": horizon, "tau": float(m.log_tau.exp())}
        if z is not None:
            st["z"] = z                                   # expected ember ignitions (calibration leaves them as is)
        return st

    def logits(self, st, a=0.0, s=0.0):
        if "logit" in st:
            return st["logit"] * math.exp(-s) + a
        lg = (st["horizon"] - st["T"] * math.exp(-a)) / (st["tau"] * math.exp(s))
        return spread_or_spot(lg, st["z"]) if "z" in st else lg

    def fit(self, history):
        """One calibration scalar from the fire's recent observed growth (up to CAL_HISTORY earlier pairs,
        all known at forecast time): a = log spread-rate multiplier (hybrid / physics) or logit bias
        (U-Net). It is shrunk toward 0 by how much growth was seen, and skipped when that was tiny,
        because one small pair says little about the next day's weather-driven growth."""
        parts, n_pos = [], 0
        for st, y, mask in history:
            sel = torch.from_numpy(mask).to(self.device)
            yy = torch.from_numpy(y).to(self.device)[sel].float()
            if yy.numel() == 0:
                continue
            base = st["logit"][sel] if "logit" in st else st["T"][sel]
            zz = st["z"][sel] if "z" in st else None
            cap = CAL_MAX_CELLS // max(len(history), 1)
            if base.numel() > cap:                         # keep every burned cell, sample the rest
                pos = torch.nonzero(yy > 0)[:, 0]
                neg = torch.nonzero(yy == 0)[:, 0]
                keep = torch.cat([pos, neg[torch.randperm(len(neg), device=neg.device)[:max(cap - len(pos), 0)]]])
                base, yy = base[keep], yy[keep]
                zz = zz[keep] if zz is not None else None
            n_pos += int(yy.sum())
            parts.append((st, base, yy, zz))
        if n_pos < CAL_MIN_BURNED:
            return 0.0, 0.0
        a = torch.zeros((), device=self.device, requires_grad=True)
        opt = torch.optim.Adam([a], lr=0.1)
        pw = torch.tensor(1.0, device=self.device)        # unweighted: expected burned area matches what was seen
        for _ in range(CAL_STEPS):
            losses = []
            for st, base, yy, zz in parts:
                if "logit" in st:
                    lg = base + a
                else:
                    lg = (st["horizon"] - base * torch.exp(-a)) / st["tau"]
                    if zz is not None:
                        lg = spread_or_spot(lg, zz)
                losses.append(torch.nn.functional.binary_cross_entropy_with_logits(lg, yy, pos_weight=pw, reduction="sum"))
            loss = torch.stack(losses).sum() / sum(len(p[2]) for p in parts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(-CAL_A, CAL_A)
        shrink = n_pos / (n_pos + CAL_SHRINK_BURNED)       # little evidence -> stay close to the model
        return float(a.detach()) * shrink, 0.0


def ens_logit(logits):
    """Ensemble = mean probability, returned as a logit."""
    if len(logits) == 1:
        return logits[0]
    p = torch.stack([torch.sigmoid(l) for l in logits]).mean(0).clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


# ---------------------------------------------------------------------------- scoring pass
def window(f, i):
    b = f.burned(i)
    rr, cc = np.nonzero(b)
    r0, r1 = max(rr.min() - TH.EVAL_PAD, 0), min(rr.max() + TH.EVAL_PAD + 1, f.H)
    c0, c1 = max(cc.min() - TH.EVAL_PAD, 0), min(cc.max() + TH.EVAL_PAD + 1, f.W)
    if (r1 - r0) > TH.EVAL_MAX or (c1 - c0) > TH.EVAL_MAX:
        rm, cm = int(rr.mean()), int(cc.mean())
        r0, r1 = max(rm - TH.EVAL_MAX // 2, 0), min(rm + TH.EVAL_MAX // 2, f.H)
        c0, c1 = max(cm - TH.EVAL_MAX // 2, 0), min(cm + TH.EVAL_MAX // 2, f.W)
    return r0, r1, c0, c1


def run_split(members, fires, calibrate, max_pairs=None, log_every=200):
    """One pass over all pairs. Returns per-variant score arrays (masked cells), labels, cell
    attributes for breakdowns, and per-pair records (with index ranges into the arrays)."""
    from scipy.ndimage import distance_transform_edt
    variants = ["raw"] + (["calibrated"] if calibrate else []) + ["distance"]
    out = {v: [] for v in variants}
    Y, FUEL, WIND, GAP, GROW, SIZE, rows = [], [], [], [], [], [], []
    cursor, n, t0 = 0, 0, time.time()
    for f in fires:
        final_km2 = float((f.arrival >= 0).sum() * f.cell_km ** 2)
        hist = []                         # earlier pairs of this fire: (per-member states, y, mask)
        for i, gap in f.pairs:
            feats, phys, y, m, b0 = f.sample(i, gap, window(f, i))
            states = [mb.state(feats, phys, b0, gap) for mb in members]
            raw = ens_logit([mb.logits(st) for mb, st in zip(members, states)])
            out["raw"].append(raw.cpu().numpy()[m])
            rec = dict(fire=f.fid, year=f.year, i=i, gap_h=gap, new=int(y[m].sum()), cells=int(m.sum()),
                       start=cursor, stop=cursor + int(m.sum()))
            if calibrate:
                if hist:
                    params = [mb.fit([(h[0][k], h[1], h[2]) for h in hist[-CAL_HISTORY:]]) for k, mb in enumerate(members)]
                    rec["cal"] = [[round(a, 3), round(s, 3)] for a, s in params]
                    if not any(a for a, _ in params):
                        rec["cal"] = None                        # too little recent growth: left as the model
                else:
                    params = [(0.0, 0.0)] * len(members)     # first pair of a fire: nothing to calibrate on
                    rec["cal"] = None
                cal = ens_logit([mb.logits(st, a, s) for mb, st, (a, s) in zip(members, states, params)])
                out["calibrated"].append(cal.cpu().numpy()[m])
            dist = distance_transform_edt(~b0) * f.cell_km if b0.any() else np.full(b0.shape, 30.0)
            out["distance"].append((-dist / max(gap / 24, 1e-3))[m].astype(np.float32))
            Y.append(y[m])
            fu = feats["fuels"]
            fuel = np.where(fu[11] > 0, np.argmax(fu[:7], 0), 7).astype(np.int8)
            hr = feats["hrrr"]
            spd = hr[2] * 8 if hr[6].max() > 0 else feats["weather"][6] * 2 + 4
            FUEL.append(fuel[m])
            WIND.append(np.digitize(spd, WIND_EDGES).astype(np.int8)[m])
            GAP.append(np.full(int(m.sum()), np.digitize(gap, GAP_EDGES), np.int8))
            GROW.append(np.full(int(m.sum()), np.digitize(y[m].sum() * f.cell_km ** 2, GROW_EDGES), np.int8))
            SIZE.append(np.full(int(m.sum()), np.digitize(final_km2, SIZE_EDGES), np.int8))
            rec["final_km2"] = round(final_km2, 2)
            rec["_b0"] = b0                               # kept only until edge errors are computed
            rec["_y"], rec["_m"] = y, m
            rows.append(rec)
            cursor = rec["stop"]
            hist = (hist + [(states, y, m)])[-CAL_HISTORY:]
            n += 1
            if n % log_every == 0:
                print(f"    {n} pairs ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if max_pairs and n >= max_pairs:
                break
        if max_pairs and n >= max_pairs:
            break
    arrays = {v: np.concatenate(out[v]) for v in variants}
    cells = dict(y=np.concatenate(Y).astype(bool), fuel=np.concatenate(FUEL), wind=np.concatenate(WIND),
                 gap=np.concatenate(GAP), grow=np.concatenate(GROW), size=np.concatenate(SIZE))
    return arrays, cells, rows


def save_preds(path, sv, cv, rows_v, st, ct, rows_t):
    """Per-cell forecasts of every variant on validation and test, for training a model router later.
    Cell order is the same for every model (same fires, pairs, windows and masks), so files from
    different evaluations line up cell by cell; per-pair cell counts are stored to check that."""
    out = {}
    for tag, sc, cl, rw in (("val", sv, cv, rows_v), ("test", st, ct, rows_t)):
        for v, arr in sc.items():
            out[f"{tag}_{v}"] = arr.astype(np.float16)
        for k, arr in cl.items():
            out[f"{tag}_{k}"] = arr
        out[f"{tag}_pair_fire"] = np.array([str(r["fire"]) for r in rw])
        out[f"{tag}_pair_i"] = np.array([r["i"] for r in rw], np.int32)
        out[f"{tag}_pair_gap_h"] = np.array([r["gap_h"] for r in rw], np.float32)
        out[f"{tag}_pair_cells"] = np.array([r["cells"] for r in rw], np.int64)
    np.savez_compressed(path, **out)
    print(f"  per-cell predictions -> {path.name} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)


# ---------------------------------------------------------------------------- metrics
def window_scores(s, y, rows, thr):
    """TS-SatFire style: every pair scored alone (empty prediction on an empty target = 1),
    mean over a fire's pairs, then mean over fires."""
    per_fire = {}
    for r in rows:
        ss, yy = s[r["start"]:r["stop"]] >= thr, y[r["start"]:r["stop"]]
        tp, ps, ts = (ss & yy).sum(), ss.sum(), yy.sum()
        u = (ss | yy).sum()
        f1 = 1.0 if ps + ts == 0 else 2 * tp / (ps + ts)
        iou = 1.0 if u == 0 else tp / u
        per_fire.setdefault(r["fire"], []).append((f1, iou))
    f = np.array([np.mean(v, 0) for v in per_fire.values()])
    return {"f1_window": float(f[:, 0].mean()), "iou_window": float(f[:, 1].mean()), "fires": len(per_fire)}


def matched_area_recall(s, y):
    k = int(y.sum())
    if k == 0:
        return float("nan")
    top = np.argpartition(-s, k - 1)[:k]
    return float(y[top].mean())


def edge_errors(s, rows, thr, cell_km):
    e = []
    for r in rows:
        m = r["_m"]
        pred = np.zeros_like(m)
        pred[m] = s[r["start"]:r["stop"]] >= thr
        e.append(TH.edge_error_km(r["_b0"], pred, r["_y"] & m, cell_km))
    e = np.array(e, dtype=float)
    return float(np.nanmean(e)) if np.isfinite(e).any() else float("nan")


def breakdown(s, cells, thr, key, names):
    from sklearn.metrics import average_precision_score
    y, k = cells["y"], cells[key]
    out = []
    for j, nm in enumerate(names):
        sel = k == j
        if sel.sum() < 1000:
            continue
        yy, ss = y[sel], s[sel]
        p = ss >= thr
        tp, fp, fn = (p & yy).sum(), (p & ~yy).sum(), (~p & yy).sum()
        out.append(dict(group=nm, cells=int(sel.sum()), burned=int(yy.sum()),
                        ap=float(average_precision_score(yy, ss)) if yy.any() and (~yy).any() else float("nan"),
                        f1=float(2 * tp / max(2 * tp + fp + fn, 1)), recall=float(tp / max(tp + fn, 1)),
                        area_ratio=float(p.sum() / max(yy.sum(), 1))))
    return out


def no_growth(rows, y):
    per_fire = {}
    for r in rows:
        empty = not y[r["start"]:r["stop"]].any()
        per_fire.setdefault(r["fire"], []).append(1.0 if empty else 0.0)
    v = float(np.mean([np.mean(x) for x in per_fire.values()]))
    return {"ap": float(y.mean()), "f1": 0.0, "iou": 0.0, "f1_window": v, "iou_window": v}


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", nargs="+", required=True,
                    help="NAME[:MODE[:drop1+drop2]] ... ; several = ensemble (mean probability)")
    ap.add_argument("--out", required=True, help="results go to runs/eval_<out>.json")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--max-pairs", type=int, help="quick check: stop after this many pairs per split")
    ap.add_argument("--catalog")
    ap.add_argument("--fires", help="score only these test fires: a CSV with a fire_id column, or ids separated by commas "
                                    "(e.g. ../benchmarks/tssatfire_2021_test_match.csv for the group's TS-SatFire test fires)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-years", default="2023")
    ap.add_argument("--test-years", default="2024,2025")
    ap.add_argument("--weather", default="observed", choices=["observed", "forecast"],
                    help="weather inputs; use what the model was trained with (forecast = only weather known at t_i)")
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("  !! no usable GPU on this node (driver problem): running on the CPU instead, slower", flush=True)
        a.device = "cpu"
    D.VAL_YEARS = {int(y) for y in a.val_years.split(",")}
    D.TEST_YEARS = {int(y) for y in a.test_years.split(",")}
    D.WEATHER_MODE = a.weather
    torch.manual_seed(0)
    runs = Path(a.data) / "runs"
    pca = TH.fit_pca(a.data, runs)
    fires = TH.load_fires(a.data, pca, None, a.catalog)
    split = {s: [f for f in fires if D.year_split(f.year) == s] for s in ("val", "test")}
    if a.fires:
        p = Path(a.fires)
        keep = set(pd.read_csv(p)["fire_id"].dropna()) if p.suffix == ".csv" else set(a.fires.split(","))
        split["test"] = [f for f in split["test"] if f.fid in keep]
        print(f"  test limited to {len(split['test'])} of {len(keep)} listed fires", flush=True)
    res_m = fires[0].res_m
    TH.set_grid(res_m / 1000)
    groups_all = TH.groups_used(set())
    D.PHYSICS_V2 = any(sp.partition(":")[2].split(":")[0] in TH.V2_MODES or sp.split(":")[0].startswith(("hybrid2", "physics2", "unet2", "unet3"))
                       for sp in a.model) or any(json.loads((runs / f"{sp.split(':')[0]}.json").read_text()).get("mode") in TH.V2_MODES
                                                 for sp in a.model if (runs / f"{sp.split(':')[0]}.json").exists())
    members = [Member(runs, sp, None, groups_all, a.device, res_m, a.width) for sp in a.model]
    calibrate = not a.no_calibration
    print(f"[{a.out}] {len(members)} model(s); val/test fires {len(split['val'])}/{len(split['test'])}; "
          f"calibration {'on' if calibrate else 'off'}", flush=True)

    res = {"name": a.out, "models": a.model, "resolution_m": res_m, "data_version": D.DATA_VERSION, "weather": D.WEATHER_MODE}
    print("  validation pass (thresholds)", flush=True)
    sv, cv, rows_v = run_split(members, split["val"], calibrate, a.max_pairs)
    thr = {v: TH.metrics(sv[v], cv["y"])["thr"] for v in sv}
    print("  test pass", flush=True)
    st, ct, rows = run_split(members, split["test"], calibrate, a.max_pairs)
    y = ct["y"]
    for v in st:
        mt = TH.metrics(st[v], y, thr[v])
        if v == "distance":
            mt.pop("brier", None), mt.pop("ece", None), mt.pop("reliability", None)   # not probabilities
        mt.update(window_scores(st[v], y, rows, thr[v]))
        mt["recall_matched_area"] = matched_area_recall(st[v], y)
        mt["edge_km_mean"] = edge_errors(st[v], rows, thr[v], fires[0].cell_km)
        mt["by_fuel"] = breakdown(st[v], ct, thr[v], "fuel", FUEL_NAMES)
        mt["by_wind"] = breakdown(st[v], ct, thr[v], "wind", WIND_NAMES)
        mt["by_gap"] = breakdown(st[v], ct, thr[v], "gap", GAP_NAMES)
        mt["by_growth"] = breakdown(st[v], ct, thr[v], "grow", GROW_NAMES)
        mt["by_fire_size"] = breakdown(st[v], ct, thr[v], "size", SIZE_NAMES)
        res[v] = mt
    res["no_growth"] = no_growth(rows, y)
    cal = [r["cal"] for r in rows if r.get("cal")]
    if cal:
        arr = np.array(cal, dtype=float)              # (pairs, members, 2)
        res["calibration_params"] = {"pairs_calibrated": len(cal), "pairs_total": len(rows)}
        for j, mb in enumerate(members):                # a = log rate multiplier (hybrid/physics) or logit bias (U-Net)
            a_ = arr[:, j, 0]
            res["calibration_params"][mb.name] = (
                {"logit_bias_median": float(np.median(a_)), "logit_bias_p10_p90": [float(np.percentile(a_, q)) for q in (10, 90)]}
                if mb.mode in TH.UNET_MODES else
                {"rate_multiplier_median": float(np.exp(np.median(a_))),
                 "rate_multiplier_p10_p90": [float(np.exp(np.percentile(a_, q))) for q in (10, 90)]})
    for r in rows:
        for k in ("_b0", "_y", "_m"):
            r.pop(k, None)
    res["pairs"] = rows
    path = runs / f"eval_{a.out}.json"
    json.dump(res, open(path, "w"), indent=1)
    try:
        save_preds(runs / f"preds_{a.out}.npz", sv, cv, rows_v, st, ct, rows)
    except Exception as e:                              # the dump is for the model router; never lose the scores over it
        print(f"  (could not save per-cell predictions: {e})", flush=True)

    print(f"\n[{a.out}] TEST ({len(rows)} pairs, {res['raw']['fires']} fires)")
    print(f"  {'variant':<12} {'AP':>6} {'F1':>6} {'IoU':>6} {'area':>6} {'R@area':>7} {'F1win':>6} {'IoUwin':>7} {'edge km':>8}")
    for v in list(st) + ["no_growth"]:
        m = res[v]
        print(f"  {v:<12} {m['ap']:6.3f} {m['f1']:6.3f} {m['iou']:6.3f} {m.get('area_ratio', float('nan')):6.2f} "
              f"{m.get('recall_matched_area', float('nan')):7.3f} {m['f1_window']:6.3f} {m['iou_window']:7.3f} "
              f"{m.get('edge_km_mean', float('nan')):8.2f}")
    if "calibration_params" in res:
        c = res["calibration_params"]
        print(f"  calibration: {c['pairs_calibrated']}/{c['pairs_total']} pairs calibrated on up to {CAL_HISTORY} earlier pairs (the rest had too little recent growth)")
        for mb in members:
            q = c[mb.name]
            if "rate_multiplier_median" in q:
                print(f"    {mb.name}: spread-rate multiplier median {q['rate_multiplier_median']:.2f} "
                      f"(p10-p90 {q['rate_multiplier_p10_p90'][0]:.2f}-{q['rate_multiplier_p10_p90'][1]:.2f})")
            else:
                print(f"    {mb.name}: logit bias median {q['logit_bias_median']:+.2f}")
    best = "calibrated" if "calibrated" in res else "raw"
    for key in ("by_growth", "by_fire_size", "by_fuel", "by_wind", "by_gap"):
        print(f"  {best} {key[3:]}: " + "  ".join(f"{b['group']} AP {b['ap']:.2f} F1 {b['f1']:.2f} (area {b['area_ratio']:.1f})" for b in res[best][key]))
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
