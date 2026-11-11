"""
Combine models from their saved per-cell forecasts (runs/preds_<eval>.npz, written by evaluate.py).
No GPU and no retraining: everything is fitted on the validation year and scored on the test years.

  1. global calibration: one temperature + bias per model, fitted on validation (fixes the U-Net's
     inflated percentages; AP is unchanged because the ranking is unchanged)
  2. averages: mean of the calibrated probabilities of two or more models
  3. router: a logistic regression on validation that weights each model by time gap, wind and fuel
     (it learns which model to trust in which conditions), scored on the test years
  4. --h2h: per-window F1 of a head-to-head evaluation split by gap (one-day windows ~ TS-SatFire)

  python combine.py --data ~/scratch/wildfire/my --members unet2_eval hybrid_v1_eval hybrid2_eval \
                    --h2h unet_h21_vs_tssatfire
Writes runs/combine.md and runs/combine.json.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import evaluate as E
import train_hybrid as TH

FIT_CELLS = 3_000_000          # validation cells used to fit calibration / router (random subset)


def load(runs, name):
    z = np.load(runs / f"preds_{name}.npz")
    rows = json.loads((runs / f"eval_{name}.json").read_text())
    return z, rows


def logits(z, split):
    return np.clip(z[f"{split}_raw"].astype(np.float32), -30, 30)


def fit_temp(s, y, idx):
    lr = LogisticRegression(C=1e4, max_iter=200).fit(s[idx, None], y[idx])
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def best_f1_threshold(s, y):
    """Same rule as train_hybrid.metrics (best F1 over 400 quantiles), with one sort instead of 400 passes."""
    qs = np.unique(np.quantile(s, np.linspace(0.5, 0.9995, 400)))
    o = np.argsort(s, kind="stable")
    ss, cy = s[o], np.concatenate([[0], np.cumsum(y[o])])
    k = np.searchsorted(ss, qs, side="left")           # cells below each threshold
    n_above, tp = len(s) - k, cy[-1] - cy[k]
    f1 = 2 * tp / (n_above + cy[-1] + 1e-9)
    return float(qs[int(np.argmax(f1))])


def onehot(k, n):
    return np.eye(n, dtype=np.float32)[np.clip(k, 0, n - 1)]


def router_features(zs, cells):
    """Each model's calibrated logit, plus that logit times gap and wind indicators: the router can
    trust a model more in some conditions. No per-condition intercepts: they improve log loss but
    reshuffle the ranking across conditions (pooled AP drops), seen on the sample data."""
    gap, wind = onehot(cells["gap"], 4), onehot(cells["wind"], 4)
    Z = np.clip(np.stack(zs, 1), -12, 12)
    inter = [Z[:, j:j + 1] * g for j in range(Z.shape[1]) for g in (gap, wind)]
    return np.concatenate([Z] + inter, 1)


def score(name, s, y, cells, rows, thr):
    m = TH.metrics(s, y, thr)
    m.update(E.window_scores(s, y, rows, m["thr"]))
    m["edge_km_mean"] = None
    m["by_gap"] = E.breakdown(s, cells, m["thr"], "gap", E.GAP_NAMES)
    m["by_wind"] = E.breakdown(s, cells, m["thr"], "wind", E.WIND_NAMES)
    m["by_growth"] = E.breakdown(s, cells, m["thr"], "grow", E.GROW_NAMES)
    print(f"  {name:<34} AP {m['ap']:.3f}  F1 {m['f1']:.3f}  F1win {m['f1_window']:.3f}  "
          f"ECE {m.get('ece', float('nan')):.3f}  area {m['area_ratio']:.2f}", flush=True)
    return m


def h2h(runs, name):
    """Per-window F1 (TS-SatFire style) of one evaluation, split by the time gap."""
    z, ev = load(runs, name)
    s, y = logits(z, "test"), z["test_y"]
    thr = ev["raw"]["thr"]
    out = {}
    for label, lo, hi in (("all windows", 0, 1e9), ("one-day gaps (12-36 h)", 12, 36), ("up to 36 h", 0, 36),
                          ("over 36 h", 36, 1e9)):
        rows = [r for r in ev["pairs"] if lo <= r["gap_h"] <= hi]
        if not rows:
            continue
        w = E.window_scores(s, y, rows, thr)
        ng = E.window_scores(np.full_like(s, -30), y, rows, 0.0)        # predict no growth anywhere
        skill = (w["f1_window"] - ng["f1_window"]) / max(1 - ng["f1_window"], 1e-9)
        out[label] = dict(windows=len(rows), fires=w["fires"], f1_window=w["f1_window"], iou_window=w["iou_window"],
                          no_growth_f1_window=ng["f1_window"], skill_over_no_growth=skill)
        print(f"  {label:<24} {len(rows):>4} windows  F1win {w['f1_window']:.3f}  (no growth {ng['f1_window']:.3f}, "
              f"skill {skill:.3f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--members", nargs="+", default=["unet2_eval", "hybrid_v1_eval", "hybrid2_eval"])
    ap.add_argument("--h2h", help="evaluation name to split by gap, e.g. unet_h21_vs_tssatfire")
    a = ap.parse_args()
    runs = Path(a.data) / "runs"
    rng = np.random.default_rng(0)
    res = {"members": a.members}

    if a.h2h:
        print(f"\n[{a.h2h}] per-window F1 by gap (TS-SatFire metric)")
        res["h2h"] = h2h(runs, a.h2h)

    print("\nloading per-cell forecasts", flush=True)
    data = {n: load(runs, n) for n in a.members}
    z0 = data[a.members[0]][0]
    for n in a.members[1:]:          # same fires, pairs, windows and masks -> same cells in the same order
        z = data[n][0]
        for k in ("val_pair_cells", "test_pair_cells", "val_y", "test_y"):
            if not np.array_equal(z[k], z0[k]):
                raise SystemExit(f"{n} and {a.members[0]} were not scored on the same cells ({k} differs)")
    yv, yt = z0["val_y"], z0["test_y"]
    cv = {k: z0[f"val_{k}"] for k in ("y", "fuel", "wind", "gap", "grow", "size")}
    ct = {k: z0[f"test_{k}"] for k in ("y", "fuel", "wind", "gap", "grow", "size")}
    rows = data[a.members[0]][1]["pairs"]
    idx = rng.choice(len(yv), min(FIT_CELLS, len(yv)), replace=False)

    def thr_on_val(sv):
        return best_f1_threshold(sv, yv)

    print(f"\ntest: {len(yt):,} cells, {int(yt.sum()):,} burned, {len(rows)} windows")
    res["single"], cal_v, cal_t, val_ap = {}, {}, {}, {}
    print("\nsingle models (raw, then with global calibration)")
    for n in a.members:
        z = data[n][0]
        sv, st = logits(z, "val"), logits(z, "test")
        res["single"][n + " raw"] = score(n + " raw", st, yt, ct, rows, thr_on_val(sv))
        A, B = fit_temp(sv, yv, idx)
        cal_v[n], cal_t[n] = A * sv + B, A * st + B
        m = score(n + " calibrated", cal_t[n], yt, ct, rows, thr_on_val(cal_v[n]))
        m["temperature"], m["bias"] = A, B
        res["single"][n + " calibrated"] = m
        val_ap[n + " calibrated"] = float(average_precision_score(yv, cal_v[n]))

    print("\naverages of calibrated probabilities")
    res["average"] = {}
    for r in range(2, len(a.members) + 1):
        for combo in itertools.combinations(a.members, r):
            pv = np.mean([1 / (1 + np.exp(-cal_v[n])) for n in combo], 0)
            pt = np.mean([1 / (1 + np.exp(-cal_t[n])) for n in combo], 0)
            sv, st = np.log(pv + 1e-7) - np.log1p(-pv + 1e-7), np.log(pt + 1e-7) - np.log1p(-pt + 1e-7)
            res["average"][" + ".join(combo)] = score(" + ".join(combo), st, yt, ct, rows, thr_on_val(sv))
            val_ap[" + ".join(combo)] = float(average_precision_score(yv, sv))

    print("\nrouter (weights each model by gap, wind and fuel; fitted on validation)")
    sub = {k: v[idx] for k, v in cv.items()}
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(router_features([cal_v[n][idx] for n in a.members], sub), yv[idx])

    def route(cal, cells, chunk=4_000_000):               # in chunks: the full feature matrix would need ~10 GB
        out = np.empty(len(cells["gap"]), np.float32)
        for s0 in range(0, len(out), chunk):
            sl = slice(s0, s0 + chunk)
            out[sl] = lr.decision_function(router_features([cal[n][sl] for n in a.members], {k: v[sl] for k, v in cells.items()}))
        return out
    sv, st = route(cal_v, cv), route(cal_t, ct)
    res["router"] = score("router(" + ", ".join(a.members) + ")", st, yt, ct, rows, thr_on_val(sv))
    nm = len(a.members)
    w = lr.coef_[0]
    res["router"]["weights"] = {
        "base": dict(zip(a.members, map(float, w[:nm]))),
        "by_gap": {n: dict(zip(E.GAP_NAMES, map(float, w[nm + 8 * j: nm + 8 * j + 4] + w[j]))) for j, n in enumerate(a.members)},
        "by_wind": {n: dict(zip(E.WIND_NAMES, map(float, w[nm + 8 * j + 4: nm + 8 * j + 8] + w[j]))) for j, n in enumerate(a.members)},
    }
    val_ap["router"] = float(average_precision_score(yv, sv))
    best = max(val_ap, key=val_ap.get)
    res["best_on_validation"] = {"name": best, "val_ap": val_ap[best]}
    print(f"\nbest on validation (the honest pick): {best} (val AP {val_ap[best]:.3f})")

    json.dump(res, open(runs / "combine.json", "w"), indent=1, default=float)
    write_md(res, runs / "combine.md")
    print(f"\nwrote {runs / 'combine.md'}")


def write_md(res, path):
    L = ["# Model combination (fitted on validation, scored on the test years)", ""]
    if "h2h" in res:
        L += ["## Head-to-head windows by gap (TS-SatFire per-window metric)", "",
              "| windows | n | fires | F1 win | no-growth F1 win | skill over no growth |", "|---|---|---|---|---|---|"]
        for k, v in res["h2h"].items():
            L.append(f"| {k} | {v['windows']} | {v['fires']} | {v['f1_window']:.3f} | {v['no_growth_f1_window']:.3f} | "
                     f"{v['skill_over_no_growth']:.3f} |")
        L.append("")
    L += ["## Test scores", "", "| model | AP | F1 | IoU | F1 win | area | Brier | ECE |", "|---|---|---|---|---|---|---|---|"]
    allm = {**res["single"], **res["average"], "router": res["router"]}
    for k, m in allm.items():
        L.append(f"| {k} | {m['ap']:.3f} | {m['f1']:.3f} | {m['iou']:.3f} | {m['f1_window']:.3f} | {m['area_ratio']:.2f} | "
                 f"{m.get('brier', float('nan')):.4f} | {m.get('ece', float('nan')):.3f} |")
    for key, names in (("by_gap", E.GAP_NAMES), ("by_wind", E.WIND_NAMES), ("by_growth", E.GROW_NAMES)):
        L += ["", f"## {key.replace('_', ' ')} (AP / F1)", "", "| model | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
        for k, m in allm.items():
            d = {b["group"]: b for b in m[key]}
            L.append(f"| {k} | " + " | ".join(f"{d[n]['ap']:.2f} / {d[n]['f1']:.2f}" if n in d else "-" for n in names) + " |")
    b = res["best_on_validation"]
    L += ["", f"Best on validation (pick this one, its test score is honest): **{b['name']}** (val AP {b['val_ap']:.3f})"]
    w = res["router"]["weights"]
    L += ["", "## Router weights (how much each model's logit counts, by condition)", "",
          "| model | " + " | ".join(E.GAP_NAMES + E.WIND_NAMES) + " |", "|---|" + "---|" * 8]
    for n in w["by_gap"]:
        L.append(f"| {n} | " + " | ".join(f"{w['by_gap'][n][g]:.2f}" for g in E.GAP_NAMES) + " | "
                 + " | ".join(f"{w['by_wind'][n][g]:.2f}" for g in E.WIND_NAMES) + " |")
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
