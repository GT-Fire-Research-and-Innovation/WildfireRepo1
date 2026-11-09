"""
Next-day fire spread model on the TS-SatFire / NIFC / suppression / weather / OSM dataset.

  python train.py prep  --out data                  # splits, Tessera PCA (training fires only), disk cache
  python train.py base  --out data                  # baselines (distance growth, logistic regression)
  python train.py train --out data --name full      # train a model; --drop <groups> for ablations
  python train.py report --out data                 # table of all runs

Task: given everything known at the end of label day t, predict the cells first burned by the
next label day t+1. Cells already burned at t, and cells burned by other fires, are excluded
from the loss and from scoring. Evaluation tiles cover the current fire (+ VIIRS active fire)
and 9.6 km around it, chosen without looking at the target. Splits are by split_group.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
CROP = 128               # 128 x 150 m = 19.2 km
EDT_PAD = 100            # cells of context for distance-to-fire (15 km)
MAX_GAP = 3              # skip label pairs more than 3 days apart
N_PCA = 16
CLEAN_LABELS = True
TILE_MARGIN = 16        # context cells around each evaluation tile (removes tile seams)
SUP_KEEP = ["hand_line", "dozer_line", "other_line", "burnout", "retardant", "uncontrolled_edge"]
GROUP_SIZES = {"fire": 11, "viirs": 4, "terrain": 14, "tessera": N_PCA, "human": 7, "weather_base": 6,
               "weather_extra": 12, "physics": 3, "suppression": len(SUP_KEEP) + 4}
GROUPS = list(GROUP_SIZES)
BRANCH_A = ["fire", "viirs", "terrain", "tessera", "human"]
BRANCH_B = ["weather_base", "weather_extra", "physics", "suppression"]
STATIC = ["terrain", "tessera", "human"]   # stored together in static.npy, in this order


# ============================================================================ prep

def load_splits(out):
    sg = pd.read_csv(out / "split_groups.csv", dtype={"fire_id": str, "split_group": str})
    have = {p.stem for p in (out / "train").glob("*.npz")}
    sg = sg[sg["fire_id"].isin(have)]
    groups = sorted(sg["split_group"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)
    n = len(groups)
    test = set(groups[: round(0.15 * n)])
    val = set(groups[round(0.15 * n): round(0.30 * n)])
    return {f: ("test" if g in test else "val" if g in val else "train") for f, g in zip(sg["fire_id"], sg["split_group"])}


def prep(out):
    split = load_splits(out)
    rng = np.random.default_rng(SEED)
    # Tessera PCA and weather normalisation from TRAINING fires only
    X, gm, ana, fc = [], [], [], []
    for f, s in split.items():
        if s != "train":
            continue
        z = np.load(out / "train" / f"{f}.npz")
        e = z["tessera"].reshape(128, -1).T.astype(np.float32)[z["tessera_cov"].ravel() > 0.5]
        if len(e):
            X.append(e[rng.choice(len(e), min(4000, len(e)), replace=False)])
        if "gm" in z.files:
            gm.append(np.nanmean(z["gm"], axis=(2, 3)))
            if z["hrrr_ana"].ndim == 3 and z["hrrr_ana"].shape[2]:
                ana.append(np.nanmean(z["hrrr_ana"], axis=2))
                fc.append(np.nanmean(z["hrrr_fc"], axis=2))
    X = np.concatenate(X)
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    comps = Vt[:N_PCA]
    sd = ((X - mu) @ comps.T).std(0)
    print(f"PCA: {N_PCA} components explain {(S[:N_PCA] ** 2).sum() / (S ** 2).sum():.1%} of Tessera variance")
    st = {}
    for k, v in (("gm", gm), ("ana", ana), ("fc", fc)):
        a = np.concatenate(v)
        st[f"{k}_mean"], st[f"{k}_std"] = np.nanmean(a, 0).astype(np.float32), (np.nanstd(a, 0) + 1e-6).astype(np.float32)
    json.dump(split, open(out / "splits.json", "w"), indent=1)
    print("fires per split:", pd.Series(split).value_counts().to_dict())

    cache = out / "cache"
    for i, f in enumerate(sorted(split)):
        d = cache / f
        d.mkdir(parents=True, exist_ok=True)
        z = np.load(out / "train" / f"{f}.npz")
        H, W = z["arrival"].shape
        np.save(d / "arrival.npy", z["arrival"].astype(np.int16))
        np.save(d / "other.npy", z["other"].astype(bool))
        e = z["tessera"].reshape(128, -1).T.astype(np.float32)
        tess = (((e - mu) @ comps.T) / sd).T.reshape(N_PCA, H, W) * (z["tessera_cov"] > 0.5)
        elev, slope = z["elev"].astype(np.float32), z["slope"].astype(np.float32)
        asp = np.radians(z["aspect"].astype(np.float32))
        terrain = np.concatenate([((elev - elev.mean()) / 500.0)[None], (elev / 3000.0)[None], (slope / 30.0)[None],
                                  (np.sin(asp) * np.sin(np.radians(slope)))[None],
                                  (np.cos(asp) * np.sin(np.radians(slope)))[None], z["static"].astype(np.float32)])
        human = z["human"].astype(np.float32) if "human" in z.files else np.zeros((7, H, W), np.float32)
        np.save(d / "static.npy", np.concatenate([terrain, tess, human]).astype(np.float16))
        np.save(d / "slope.npy", slope.astype(np.float16))
        np.save(d / "aspect.npy", asp.astype(np.float16))
        v = z["viirs"].astype(np.float32)        # af, i4d, i5d, m11, i4n, i5n
        q = np.stack([v[:, 0] > 0, np.clip((v[:, 1] - 250) * 2, 0, 255) * (v[:, 1] > 0), np.clip(v[:, 3] * 4, 0, 255),
                      np.clip((v[:, 4] - 250) * 2, 0, 255) * (v[:, 4] > 0)], 1).astype(np.uint8)
        np.save(d / "viirs.npy", q)
        meta = dict(dates=z["dates"], viirs_have=z["viirs_have"])
        if "sup" in z.files:
            ch = [str(c) for c in z["sup_channels"]]
            s6 = z["sup"][:, [ch.index(c) for c in SUP_KEEP]].astype(np.uint8)
            bits = np.zeros(s6.shape[:1] + s6.shape[2:], np.uint8)
            for j in range(len(SUP_KEEP)):
                bits |= (s6[:, j] > 0).astype(np.uint8) << j
            np.save(d / "sup.npy", bits)
            meta.update(sup_dates=z["sup_dates"], ics=z["ics"], ics_fields=z["ics_fields"])
        if "gm" in z.files:
            gmn = (z["gm"] - st["gm_mean"][None, :, None, None]) / st["gm_std"][None, :, None, None]
            meta.update(w_dates=z["w_dates"], gm=gmn.astype(np.float32), gm_vars=z["gm_vars"],
                        gm_th=z["gm"][:, list(map(str, z["gm_vars"])).index("th")].astype(np.float32))
            np.save(d / "gm_row.npy", z["gm_row"].astype(np.int16))
            np.save(d / "gm_col.npy", z["gm_col"].astype(np.int16))
            if z["hrrr_ana"].ndim == 3 and z["hrrr_ana"].shape[2]:
                meta.update(ana_n=((z["hrrr_ana"] - st["ana_mean"][None, :, None]) / st["ana_std"][None, :, None]).astype(np.float32),
                            fc_n=((z["hrrr_fc"] - st["fc_mean"][None, :, None]) / st["fc_std"][None, :, None]).astype(np.float32),
                            fc_raw=z["hrrr_fc"].astype(np.float32))
                np.save(d / "h_idx.npy", z["hrrr_idx"].astype(np.int32))
        np.savez(d / "meta.npz", **meta)
        if (i + 1) % 20 == 0:
            print(f"  cached {i + 1}/{len(split)}")
    print("cache ready:", cache)


# ============================================================================ data

class Fire:
    def __init__(self, out, fid):
        d = out / "cache" / fid
        self.fid = fid
        mm = dict(mmap_mode="r")
        self.arrival = np.load(d / "arrival.npy")
        self.other = np.load(d / "other.npy")
        self.static = np.load(d / "static.npy", **mm)
        self.slope = np.load(d / "slope.npy", **mm)
        self.aspect = np.load(d / "aspect.npy", **mm)
        self.viirs = np.load(d / "viirs.npy", **mm)
        m = np.load(d / "meta.npz")
        self.dates = [str(x) for x in m["dates"]]
        self.viirs_have = m["viirs_have"]
        self.has_sup = (d / "sup.npy").exists()
        if self.has_sup:
            self.sup = np.load(d / "sup.npy", **mm)
            self.sup_dates = {str(x): i for i, x in enumerate(m["sup_dates"])}
            self.ics, self.ics_fields = m["ics"], {str(n): i for i, n in enumerate(m["ics_fields"])}
        self.has_w = "gm" in m.files
        if self.has_w:
            self.w_dates = {str(x): i for i, x in enumerate(m["w_dates"])}
            self.gm, self.gm_th = m["gm"], m["gm_th"]
            self.gm_vars = [str(v) for v in m["gm_vars"]]
            self.gm_row, self.gm_col = np.load(d / "gm_row.npy", **mm), np.load(d / "gm_col.npy", **mm)
            self.has_h = "ana_n" in m.files
            if self.has_h:
                self.ana_n, self.fc_n, self.fc_raw = m["ana_n"], m["fc_n"], m["fc_raw"]
                self.h_idx = np.load(d / "h_idx.npy", **mm)
        self.H, self.W = self.arrival.shape
        self.pairs = []
        for k in range(len(self.dates) - 1):
            gap = (pd.Timestamp(self.dates[k + 1]) - pd.Timestamp(self.dates[k])).days
            if 1 <= gap <= MAX_GAP and ((self.arrival >= 0) & (self.arrival <= k)).any():
                self.pairs.append((k, gap))

    def current(self, k):
        """Current fire for choosing crops/tiles: burned by day k, or VIIRS active fire on day k."""
        cur = (self.arrival >= 0) & (self.arrival <= k)
        if self.viirs_have[k]:
            cur = cur | (self.viirs[k, 0] > 0)
        return cur

    def features(self, k, gap, win):
        """Feature groups, target and loss mask for pair (k -> k+1) on window (r0, r1, c0, c1)."""
        from scipy.ndimage import distance_transform_edt
        r0, r1, c0, c1 = win
        R0, R1, C0, C1 = max(r0 - EDT_PAD, 0), min(r1 + EDT_PAD, self.H), max(c0 - EDT_PAD, 0), min(c1 + EDT_PAD, self.W)
        arr = self.arrival[R0:R1, C0:C1]
        b0 = (arr >= 0) & (arr <= k)
        if b0.any():
            dist, (iy, ix) = distance_transform_edt(~b0, return_indices=True)
        else:
            dist = np.full(b0.shape, 1e3)
            iy, ix = np.indices(b0.shape)
        sl = (slice(r0 - R0, r1 - R0), slice(c0 - C0, c1 - C0))
        win_sl = (slice(r0, r1), slice(c0, c1))
        a = arr[sl]
        h, w = a.shape
        B0 = b0[sl]
        B1 = ((a >= 0) & (a <= k - 1)) if k >= 1 else B0
        B2 = ((a >= 0) & (a <= k - 2)) if k >= 2 else B1
        dist_km = dist[sl] * 0.15
        af0 = self.viirs[k, 0][win_sl].astype(np.float32) if self.viirs_have[k] else np.zeros((h, w), np.float32)
        af1 = self.viirs[k - 1, 0][win_sl].astype(np.float32) if k >= 1 and self.viirs_have[k - 1] else af0
        oth = self.other[win_sl]
        af_recent = np.zeros((h, w), bool)
        for j in range(max(0, k - 3), k + 1):
            if self.viirs_have[j]:
                af_recent |= self.viirs[j, 0][win_sl] > 0
        f = {}
        f["fire"] = np.stack([B0, B1, B2, a == k, af0, af1, np.clip(dist_km, 0, 15) / 15, np.exp(-dist_km / 2),
                              oth, np.full((h, w), gap / 3), af_recent]).astype(np.float32)
        v = self.viirs[k][:, r0:r1, c0:c1].astype(np.float32)
        f["viirs"] = np.stack([v[0], v[1] / 128 - 1, v[2] / 128, v[3] / 128 - 1]).astype(np.float32) * float(self.viirs_have[k])
        st = self.static[:, r0:r1, c0:c1].astype(np.float32)
        f["terrain"], f["tessera"], f["human"] = st[:14], st[14:14 + N_PCA], st[14 + N_PCA:]

        d, d1 = self.dates[k], self.dates[k + 1]
        u = vv = np.zeros((h, w), np.float32)
        f["weather_base"] = np.zeros((6, h, w), np.float32)
        f["weather_extra"] = np.zeros((12, h, w), np.float32)
        if self.has_w and d in self.w_dates:
            i0, i1 = self.w_dates[d], self.w_dates.get(d1, self.w_dates[d])
            gr, gc = self.gm_row[win_sl].astype(np.int64), self.gm_col[win_sl].astype(np.int64)
            g = np.nan_to_num(self.gm[i0][:, gr, gc])
            vi = {n: j for j, n in enumerate(self.gm_vars)}
            th = np.radians(np.nan_to_num(self.gm_th[i0][gr, gc]))
            f["weather_base"] = np.stack([g[vi["vs"]], g[vi["erc"]], g[vi["rmin"]], g[vi["tmmx"]], np.sin(th), np.cos(th)])
            extra = [g[vi[n]] for n in ("vpd", "fm100", "fm1000", "bi")]
            if self.has_h:
                hi = self.h_idx[win_sl].astype(np.int64)
                an = np.nan_to_num(self.ana_n[i0][:, hi])
                fn = np.nan_to_num(self.fc_n[i1][:, hi])
                u, vv = np.nan_to_num(self.fc_raw[i1][1][hi]), np.nan_to_num(self.fc_raw[i1][2][hi])
                extra += list(an) + list(fn)
            else:
                extra += [np.zeros((h, w), np.float32)] * 8
            f["weather_extra"] = np.stack(extra).astype(np.float32)

        # physics: wind pushing fire uphill, and cells downwind of the current fire
        wspd = np.hypot(u, vv) + 1e-6
        ux, uy = u / wspd, vv / wspd                       # wind blowing toward (east, north)
        asp = np.radians(self.aspect[win_sl].astype(np.float32))   # aspect is stored in degrees
        slp = np.radians(self.slope[win_sl].astype(np.float32))
        upslope_wind = (ux * -np.sin(asp) + uy * -np.cos(asp)) * np.sin(slp) * np.minimum(wspd / 10, 2)
        gy = (np.arange(R1 - R0)[:, None] - iy)[sl] * -1.0  # + = north of nearest burned cell
        gx = (np.arange(C1 - C0)[None, :] - ix)[sl] * 1.0
        r = np.hypot(gx, gy) + 1e-6
        downwind = ((gx * ux + gy * uy) / r) * np.exp(-dist_km / 3) * np.minimum(wspd / 10, 2)
        downwind[B0] = 0
        f["physics"] = np.stack([upslope_wind, downwind, np.minimum(wspd / 10, 3)]).astype(np.float32)

        sup = np.zeros((len(SUP_KEEP) + 4, h, w), np.float32)
        if self.has_sup and d in self.sup_dates:
            j = self.sup_dates[d]
            bits = self.sup[j][win_sl]
            for c in range(len(SUP_KEEP)):
                sup[c] = (bits >> c) & 1
            ics, fi = self.ics[j], self.ics_fields
            if np.isfinite(ics[fi["total_personnel"]]):
                sup[-4] = math.log1p(ics[fi["total_personnel"]]) / 8
                sup[-3] = math.log1p(np.nan_to_num(ics[fi["total_aerial"]])) / 4
                sup[-2] = np.nan_to_num(ics[fi["pct_contained"]]) / 100
                sup[-1] = 1.0
        f["suppression"] = sup
        target = a == k + 1
        mask = ~B0 & ~oth
        if CLEAN_LABELS:
            from scipy.ndimage import binary_dilation
            ev = np.zeros(b0.shape, bool)
            for j in range(max(0, k - 3), min(k + 2, len(self.dates))):
                if self.viirs_have[j]:
                    ev |= self.viirs[j, 0][R0:R1, C0:C1] > 0
            ev = binary_dilation(ev, iterations=3)[sl]
            mask &= ~(target & (dist_km > 3) & ~ev)
        return f, target, mask, dist_km


def eval_region(fire, k):
    cur = fire.current(k)
    if not cur.any():
        return None
    rows, cols = np.where(cur)
    return (max(rows.min() - 64, 0), min(rows.max() + 65, fire.H), max(cols.min() - 64, 0), min(cols.max() + 65, fire.W))


def tiles(reg, H, W):
    r0, r1, c0, c1 = reg
    for r in range(r0, r1, CROP):
        for c in range(c0, c1, CROP):
            yield (r, min(r + CROP, H), c, min(c + CROP, W))


def pad_to(x, size=CROP):
    h, w = x.shape[-2:]
    if h == size and w == size:
        return x
    return np.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, size - h), (0, size - w)])


def pad8(x):
    """Pad the last two axes up to a multiple of 8 (the U-Net halves the size three times)."""
    h, w = x.shape[-2:]
    H, W = -(-h // 8) * 8, -(-w // 8) * 8
    return x if (H, W) == (h, w) else np.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, H - h), (0, W - w)])


def flip_inplace(f, hflip, vflip):
    """Flip all groups and fix the sign of east/west and north/south channels."""
    ax = tuple(a for a, on in ((-1, hflip), (-2, vflip)) if on)
    if not ax:
        return f
    g = {k: np.flip(v, ax).copy() for k, v in f.items()}
    if hflip:
        g["terrain"][3] *= -1        # east component of slope
        g["weather_base"][4] *= -1   # sin(wind dir)
        g["physics"][1] *= 1         # downwind alignment is invariant
    if vflip:
        g["terrain"][4] *= -1        # north component of slope
        g["weather_base"][5] *= -1   # cos(wind dir)
    # HRRR u10/v10 (analysis at 5,6 and forecast at 9,10 in weather_extra; order gust,u,v,rh)
    if hflip:
        g["weather_extra"][[5, 9]] *= -1
    if vflip:
        g["weather_extra"][[6, 10]] *= -1
    return g


# ============================================================================ model

def make_model(n_a, n_b, width=16):
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())

    class Gate(nn.Module):
        """The context branch (weather / physics / suppression) gates the fire+landscape branch
        per pixel and channel: learned fusion instead of plain concatenation."""

        def __init__(self, ca, cb):
            super().__init__()
            self.g = nn.Sequential(nn.Conv2d(ca + cb, ca, 1), nn.Sigmoid())
            self.m = nn.Conv2d(cb, ca, 1)

        def forward(self, a, b):
            return a * (1 + self.g(torch.cat([a, b], 1))) + self.m(b)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = [width, width * 2, width * 4, width * 8]
            wb = [max(8, x // 2) for x in w]
            self.n_b = n_b
            self.ea = nn.ModuleList([block(n_a, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])])
            if n_b:
                self.eb = nn.ModuleList([block(n_b, wb[0]), block(wb[0], wb[1]), block(wb[1], wb[2]), block(wb[2], wb[3])])
                self.gates = nn.ModuleList([Gate(w[i], wb[i]) for i in range(4)])
            self.pool = nn.MaxPool2d(2)
            self.up = nn.ModuleList([nn.ConvTranspose2d(w[i], w[i - 1], 2, stride=2) for i in (3, 2, 1)])
            self.dec = nn.ModuleList([block(w[i - 1] * 2, w[i - 1]) for i in (3, 2, 1)])
            self.head = nn.Conv2d(w[0], 1, 1)

        def forward(self, xa, xb=None):
            skips, a, b = [], xa, xb
            for i in range(4):
                a = self.ea[i](a)
                if self.n_b:
                    b = self.eb[i](b)
                    a = self.gates[i](a, b)
                skips.append(a)
                if i < 3:
                    a = self.pool(a)
                    if self.n_b:
                        b = self.pool(b)
            x = skips[3]
            for j, i in enumerate((2, 1, 0)):
                x = self.dec[j](torch.cat([self.up[j](x), skips[i]], 1))
            return self.head(x)[:, 0]

    return Net()


# ============================================================================ evaluation

def metrics(scores, y, thr=None):
    from sklearn.metrics import average_precision_score
    y = y.astype(bool)
    out = {"n_pos": int(y.sum()), "n": int(len(y)), "pos_rate": float(y.mean())}
    out["ap"] = float(average_precision_score(y, scores)) if y.any() else float("nan")
    if thr is None:  # pick the best-F1 threshold (validation only; reused on test)
        qs = np.unique(np.quantile(scores, np.linspace(0.5, 0.9995, 400)))
        f1s = [2 * ((scores >= q) & y).sum() / ((scores >= q).sum() + y.sum() + 1e-9) for q in qs]
        thr = float(qs[int(np.argmax(f1s))])
    p = scores >= thr
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    out.update(thr=thr, f1=float(2 * tp / (2 * tp + fp + fn + 1e-9)), iou=float(tp / (tp + fp + fn + 1e-9)),
               precision=float(tp / (tp + fp + 1e-9)), recall=float(tp / (tp + fn + 1e-9)),
               area_ratio=float(p.sum() / max(y.sum(), 1)))
    return out


def evaluate(fires, score_tile, max_pairs=None):
    """score_tile(f, dist_km) -> (h, w) scores for one tile. Returns pooled scores and targets."""
    S, Y = [], []
    n = 0
    for fire in fires:
        for k, gap in fire.pairs:
            reg = eval_region(fire, k)
            if reg is None:
                continue
            for (r0, r1, c0, c1) in tiles(reg, fire.H, fire.W):
                e0, e1 = max(r0 - TILE_MARGIN, 0), min(r1 + TILE_MARGIN, fire.H)
                e2, e3 = max(c0 - TILE_MARGIN, 0), min(c1 + TILE_MARGIN, fire.W)
                f, y, m, dist = fire.features(k, gap, (e0, e1, e2, e3))
                s = score_tile(f, dist)
                inner = (slice(r0 - e0, r1 - e0), slice(c0 - e2, c1 - e2))
                mi = m[inner]
                S.append(s[inner][mi])
                Y.append(y[inner][mi])
            n += 1
            if max_pairs and n >= max_pairs:
                break
    return np.concatenate(S), np.concatenate(Y)


def load(out, which):
    split = json.load(open(out / "splits.json"))
    return [Fire(out, f) for f, s in sorted(split.items()) if s in which]


def save_run(out, name, rec):
    (out / "runs").mkdir(exist_ok=True)
    json.dump(rec, open(out / "runs" / f"{name}.json", "w"), indent=1)


def show(name, m):
    print(f"[{name}] TEST AP {m['ap']:.3f}  F1 {m['f1']:.3f}  IoU {m['iou']:.3f}  P {m['precision']:.3f}  "
          f"R {m['recall']:.3f}  area ratio {m['area_ratio']:.2f}  (positives {m['pos_rate']:.2%})", flush=True)


def baselines(out, only=None):
    va, te = load(out, ["val"]), load(out, ["test"])
    if only != "logistic":
        baseline_distance(out, va, te)
    if only != "distance":
        baseline_logistic(out, va, te)


def baseline_distance(out, va, te):
    # 1) distance growth: the closer to the current fire, the more likely to burn next
    fn = lambda f, dist: -dist  # noqa: E731
    mv = metrics(*evaluate(va, fn))
    mt = metrics(*evaluate(te, fn), mv["thr"])
    save_run(out, "base_distance", {"name": "base_distance", "val": mv, "test": mt})
    show("distance growth", mt)


def baseline_logistic(out, va, te):
    # 2) logistic regression on every feature, pixel by pixel (no spatial context)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(SEED)
    X, Y = [], []
    for fire in load(out, ["train"]):
        for k, gap in fire.pairs:
            reg = eval_region(fire, k)
            if reg is None:
                continue
            for win in tiles(reg, fire.H, fire.W):
                f, y, m, _ = fire.features(k, gap, win)
                xs = np.concatenate([f[g] for g in GROUPS])[:, m].T
                yy = y[m]
                pos, neg = np.where(yy)[0], np.where(~yy)[0]
                pick = np.concatenate([rng.permutation(pos)[:40], rng.permutation(neg)[:120]])
                X.append(xs[pick].astype(np.float32))
                Y.append(yy[pick])
    X, Y = np.concatenate(X), np.concatenate(Y)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=3000, C=0.3).fit(sc.transform(X), Y)

    def fn2(f, dist):
        xs = np.concatenate([f[g] for g in GROUPS])
        C, h, w = xs.shape
        return lr.decision_function(sc.transform(xs.reshape(C, -1).T)).reshape(h, w)
    mv = metrics(*evaluate(va, fn2))
    mt = metrics(*evaluate(te, fn2), mv["thr"])
    save_run(out, "base_logistic", {"name": "base_logistic", "val": mv, "test": mt})
    show("logistic regression", mt)


def train(out, name, drop, epochs, width, lr, eval_every):
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    tr, va, te = load(out, ["train"]), load(out, ["val"]), load(out, ["test"])
    ga = [g for g in BRANCH_A if g not in drop]
    gb = [g for g in BRANCH_B if g not in drop]
    n_a, n_b = sum(GROUP_SIZES[g] for g in ga), sum(GROUP_SIZES[g] for g in gb)
    model = make_model(n_a, n_b, width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    items = [(fi, k, gap) for fi, fire in enumerate(tr) for k, gap in fire.pairs]
    print(f"[{name}] A={ga} ({n_a} ch)  B={gb} ({n_b} ch)  {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params  "
          f"{len(items)} train pairs, {sum(len(f.pairs) for f in va)} val, {sum(len(f.pairs) for f in te)} test", flush=True)
    pos_w = torch.tensor(8.0)

    def to_tensors(f):
        xa = torch.from_numpy(np.concatenate([f[g] for g in ga]))
        xb = torch.from_numpy(np.concatenate([f[g] for g in gb])) if gb else None
        return xa, xb

    def make_batch(idx):
        A, Bs, Ys, Ms = [], [], [], []
        for i in idx:
            fi, k, gap = items[i]
            fire = tr[fi]
            cur = fire.current(k)
            rows, cols = np.where(cur)
            j = np.random.randint(len(rows))
            r = int(np.clip(rows[j] + np.random.randint(-48, 49) - CROP // 2, 0, max(fire.H - CROP, 0)))
            c = int(np.clip(cols[j] + np.random.randint(-48, 49) - CROP // 2, 0, max(fire.W - CROP, 0)))
            f, y, m, _ = fire.features(k, gap, (r, min(r + CROP, fire.H), c, min(c + CROP, fire.W)))
            f = {g: pad_to(v) for g, v in f.items()}
            y, m = pad_to(y[None])[0], pad_to(m[None])[0]
            if np.random.rand() < 0.5:
                f, y, m = flip_inplace(f, True, False), y[:, ::-1].copy(), m[:, ::-1].copy()
            if np.random.rand() < 0.5:
                f, y, m = flip_inplace(f, False, True), y[::-1].copy(), m[::-1].copy()
            xa, xb = to_tensors(f)
            A.append(xa)
            Bs.append(xb)
            Ys.append(torch.from_numpy(y.astype(np.float32)))
            Ms.append(torch.from_numpy(m.astype(np.float32)))
        return torch.stack(A), (torch.stack(Bs) if gb else None), torch.stack(Ys), torch.stack(Ms)

    def loss_fn(logit, y, m):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w, reduction="none")
        bce = (bce * m).sum() / m.sum().clamp(min=1)
        p = torch.sigmoid(logit) * m
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
        return bce + dice

    def score_tile(f, dist):
        h, w = f["fire"].shape[1:]
        xa, xb = to_tensors({g: pad8(v) for g, v in f.items()})
        with torch.no_grad():
            s = model(xa[None], xb[None] if xb is not None else None)[0].numpy()
        return s[:h, :w]

    bs = 8
    steps = max(1, len(items) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.15)
    best, hist, start = (-1, None, None), [], 0
    ck = out / "runs" / f"{name}.ckpt"
    (out / "runs").mkdir(exist_ok=True)
    if ck.exists():
        c = torch.load(ck, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        best, hist, start = c["best"], c["hist"], c["epoch"]
        np.random.set_state(c["np_rng"]); torch.set_rng_state(c["torch_rng"])
        print(f"  resumed from epoch {start}", flush=True)
    for ep in range(start, epochs):
        model.train()
        order = np.random.permutation(len(items))
        tl, t1 = 0.0, time.time()
        for s in range(steps):
            xa, xb, y, m = make_batch(order[s * bs:(s + 1) * bs])
            loss = loss_fn(model(xa, xb), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
        rec = {"epoch": ep + 1, "loss": tl / steps, "train_s": time.time() - t1}
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            model.eval()
            mv = metrics(*evaluate(va, score_tile))
            rec.update(val_ap=mv["ap"], val_f1=mv["f1"])
            if mv["ap"] > best[0]:
                best = (mv["ap"], {k: v.clone() for k, v in model.state_dict().items()}, mv)
        hist.append(rec)
        print(f"  epoch {ep + 1}/{epochs} loss {rec['loss']:.3f}" +
              (f"  val AP {rec['val_ap']:.3f} F1 {rec['val_f1']:.3f}" if "val_ap" in rec else "") +
              f"  ({time.time() - t1:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best,
                    "hist": hist, "epoch": ep + 1, "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state()}, ck)
    model.load_state_dict(best[1])
    model.eval()
    mv = best[2]
    mt = metrics(*evaluate(te, score_tile), mv["thr"])
    torch.save(model.state_dict(), out / "runs" / f"{name}.pt") if (out / "runs").exists() else None
    save_run(out, name, {"name": name, "drop": drop, "width": width, "epochs": epochs, "val": mv, "test": mt,
                         "history": hist, "groups_a": ga, "groups_b": gb})
    (out / "runs").mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "runs" / f"{name}.pt")
    show(name, mt)


def report(out):
    rows = []
    for p in sorted((out / "runs").glob("*.json")):
        j = json.load(open(p))
        t, v = j["test"], j["val"]
        rows.append({"run": j["name"], "test_AP": t["ap"], "test_F1": t["f1"], "test_IoU": t["iou"],
                     "precision": t["precision"], "recall": t["recall"], "area_ratio": t["area_ratio"],
                     "val_AP": v["ap"], "pos_rate": t["pos_rate"]})
    df = pd.DataFrame(rows).sort_values("test_AP", ascending=False)
    print(df.round(3).to_string(index=False))
    df.to_csv(out / "runs" / "summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "base", "train", "report"])
    ap.add_argument("--out", default="nifc_out")
    ap.add_argument("--name", default="full")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--only", choices=["distance", "logistic"])
    a = ap.parse_args()
    out = Path(a.out)
    {"prep": lambda: prep(out), "base": lambda: baselines(out, a.only), "report": lambda: report(out),
     "train": lambda: train(out, a.name, a.drop, a.epochs, a.width, a.lr, a.eval_every)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
