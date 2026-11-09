"""Error analysis of a trained run on the test fires: per fire, by gap, by distance to front,
over/under prediction, and example maps.   python analyze.py --out data --name full"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--name", default="full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--raw-labels", action="store_true", help="score against uncleaned labels")
    a = ap.parse_args()
    T.CLEAN_LABELS = not a.raw_labels
    out = Path(a.out)
    run = json.load(open(out / "runs" / f"{a.name}.json"))
    ga, gb = run["groups_a"], run["groups_b"]
    model = T.make_model(sum(T.GROUP_SIZES[g] for g in ga), sum(T.GROUP_SIZES[g] for g in gb), run["width"])
    model.load_state_dict(torch.load(out / "runs" / f"{a.name}.pt"))
    model.eval()
    torch.set_num_threads(2)
    thr = run["val"]["thr"]
    fires = T.load(out, [a.split])
    summ = pd.read_csv(out / "summary.csv", dtype={"fire_id": str}).set_index("fire_id")

    rows, pix, examples = [], [], []
    for fire in fires:
        for k, gap in fire.pairs:
            reg = T.eval_region(fire, k)
            if reg is None:
                continue
            r0, _, c0, _ = reg
            wins = list(T.tiles(reg, fire.H, fire.W))
            r1, c1 = max(w[1] for w in wins), max(w[3] for w in wins)
            S = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
            Y = np.zeros_like(S, bool)
            M = np.zeros_like(S, bool)
            D = np.zeros_like(S)
            AF = np.zeros_like(S)
            for (wr0, wr1, wc0, wc1) in wins:
                e0, e1 = max(wr0 - T.TILE_MARGIN, 0), min(wr1 + T.TILE_MARGIN, fire.H)
                e2, e3 = max(wc0 - T.TILE_MARGIN, 0), min(wc1 + T.TILE_MARGIN, fire.W)
                f, y, m, dist = fire.features(k, gap, (e0, e1, e2, e3))
                xa = torch.from_numpy(np.concatenate([T.pad8(f[g]) for g in ga]))[None]
                xb = torch.from_numpy(np.concatenate([T.pad8(f[g]) for g in gb]))[None] if gb else None
                with torch.no_grad():
                    s = model(xa, xb)[0].numpy()
                inner = (slice(wr0 - e0, wr1 - e0), slice(wc0 - e2, wc1 - e2))
                sl = (slice(wr0 - r0, wr1 - r0), slice(wc0 - c0, wc1 - c0))
                S[sl], Y[sl], M[sl], D[sl], AF[sl] = (s[inner], y[inner], m[inner], dist[inner],
                                                      f["fire"][4][inner])
            s, y, d, af = S[M], Y[M], D[M], AF[M]
            p = s >= thr
            tp = int((p & y).sum())
            rows.append(dict(fire=fire.fid, year=int(summ.loc[fire.fid, "fire_year"]),
                             name=summ.loc[fire.fid, "primary_incident"], k=k, gap=gap,
                             new_cells=int(y.sum()), pred_cells=int(p.sum()), tp=tp,
                             af_today=int((AF > 0).sum()), burned_km2=float((fire.arrival[r0:r1, c0:c1] <= k).sum()
                                                                            * 0.0225)))
            pix.append(pd.DataFrame(dict(s=s, y=y, d=d, af=af > 0, gap=gap, fire=fire.fid)))
            if y.sum() > 200:
                examples.append((y.sum(), fire.fid, k, S.copy(), Y.copy(), (fire.arrival[r0:r1, c0:c1] >= 0)
                                 & (fire.arrival[r0:r1, c0:c1] <= k), M.copy(), fire.dates[k], fire.dates[k + 1]))
    df = pd.DataFrame(rows)
    P = pd.concat(pix, ignore_index=True)
    from sklearn.metrics import average_precision_score as aps

    def m(g):
        p = g.s >= thr
        tp = (p & g.y).sum()
        return pd.Series(dict(pairs=g.groupby(["fire"]).ngroups if "fire" in g else 0, pos_rate=g.y.mean(),
                              AP=aps(g.y, g.s) if g.y.any() else np.nan,
                              F1=2 * tp / (p.sum() + g.y.sum() + 1e-9),
                              precision=tp / (p.sum() + 1e-9), recall=tp / (g.y.sum() + 1e-9),
                              area_ratio=p.sum() / max(g.y.sum(), 1)))
    print("OVERALL", m(P).round(3).to_dict())
    print("\nBy gap (days between perimeters):")
    print(P.groupby("gap").apply(m).round(3).to_string())
    P["dist_band"] = pd.cut(P.d, [-0.1, 0.3, 1, 3, 100], labels=["<0.3 km", "0.3-1 km", "1-3 km", ">3 km"])
    print("\nBy distance from current fire:")
    print(P.groupby("dist_band", observed=True).apply(m).round(3).to_string())
    print("\nShare of true new burn by distance:", (P[P.y].dist_band.value_counts(normalize=True).round(3).to_dict()))
    print("\nActive fire detected in cell today vs not:")
    print(P.groupby("af").apply(m).round(3).to_string())
    pf = P.groupby("fire").apply(m)
    pf = pf.join(df.groupby("fire").agg(name=("name", "first"), year=("year", "first"), new=("new_cells", "sum"),
                                        pairs=("k", "count")), rsuffix="_")
    print("\nPer test fire:")
    print(pf[["name", "year", "pairs_", "new", "AP", "F1", "area_ratio"]].sort_values("new", ascending=False)
          .round(3).to_string())
    # daily growth size vs skill
    df["f1"] = 2 * df.tp / (df.pred_cells + df.new_cells + 1e-9)
    df["growth_band"] = pd.cut(df.new_cells * 0.0225, [-1, 0.5, 5, 50, 1e6], labels=["<0.5 km2", "0.5-5", "5-50", ">50 km2"])
    print("\nPer pair F1 by size of next-day growth:")
    print(df.groupby("growth_band", observed=True).agg(pairs=("f1", "size"), mean_f1=("f1", "mean"),
                                                       pred_over_true=("pred_cells", "sum")).round(3).to_string())
    df.to_csv(out / "runs" / f"{a.name}_pairs.csv", index=False)

    # example maps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    examples.sort(key=lambda e: -e[0])
    pick = [examples[i] for i in np.linspace(0, len(examples) - 1, min(4, len(examples))).astype(int)]
    fig, axes = plt.subplots(len(pick), 2, figsize=(10, 4.6 * len(pick)))
    for row, (n, fid, k, S, Y, B, M, d0, d1) in zip(np.atleast_2d(axes), pick):
        prob = 1 / (1 + np.exp(-S))
        prob[B] = np.nan
        row[0].imshow(prob, cmap="magma", vmin=0, vmax=1)
        row[0].contour(B, levels=[0.5], colors="cyan", linewidths=0.8)
        row[0].set_title(f"{summ.loc[fid, 'primary_incident']} {d0}: predicted chance of burning by {d1}", fontsize=9)
        lab = np.zeros(S.shape)
        lab[B] = 1
        lab[Y] = 2
        pred = (S >= thr) & M
        lab[pred & ~Y] = 3
        lab[pred & Y] = 4
        cm = ListedColormap(["#f4f1ea", "#555555", "#2b6cb0", "#e8a33d", "#c0392b"])
        row[1].imshow(lab, cmap=cm, vmin=0, vmax=4, interpolation="nearest")
        row[1].set_title("grey burned already · blue missed · orange false alarm · red hit", fontsize=9)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "runs" / f"{a.name}_examples{'_raw' if a.raw_labels else ''}.png", dpi=110)
    print("\nmaps:", out / "runs" / f"{a.name}_examples.png")


if __name__ == "__main__":
    sys.exit(main())
