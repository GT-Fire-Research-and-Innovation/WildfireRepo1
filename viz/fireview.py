"""
Look at one fire of the dataset the way the model sees it, to check that everything lines up.

  import fireview as V
  d = V.fetch("Dragon Bravo")                 # downloads that fire from Hugging Face (or pass a fire id)
  V.progression(d)                            # perimeters over time, gaps, label sources
  V.layers(d)                                 # every input layer with the final perimeter on top
  V.pairs(d)                                  # every training pair: old burn, new burn, excluded cells
  V.pair(d, 10)                               # one pair exactly as the training code builds it,
                                              #   with the physics-only forecast vs what happened
  V.crops(d)                                  # random training crops as the data loader feeds them
  V.prediction(d, 10, "hybrid_v1.pt", "hybrid", pca="tessera_pca.npz")   # a trained model's forecast

Works in Colab (the notebook clones the repo for the loader and the physics) or on ICE with a
local dataset folder: V.open_local("~/scratch/wildfire/my/fires/F6bd2bd1f45").
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HF_REPO = "Javyn123/wildfire-spread-2014-2025"
FUEL_NAMES = ["non-burnable", "grass", "grass-shrub", "shrub", "timber-understory", "timber litter", "slash-blowdown"]
FUEL_COLORS = ["#d9d9d9", "#f2e394", "#c9d17a", "#b0844e", "#5d8a3a", "#2f5d2a", "#7a4a7a"]


# ------------------------------------------------------------------------------- data access
def catalog(local_dir="hf_data"):
    """fires.csv merged with the QC rating (downloaded once)."""
    from huggingface_hub import hf_hub_download
    f = pd.read_csv(hf_hub_download(HF_REPO, "catalog/fires.csv", repo_type="dataset", local_dir=local_dir))
    q = pd.read_csv(hf_hub_download(HF_REPO, "catalog/qc_fires.csv", repo_type="dataset", local_dir=local_dir))
    return f.merge(q[["fire_id", "rating", "flags"]], on="fire_id", how="left")


def fetch(name_or_id, year=None, local_dir="hf_data"):
    """Download one fire folder from Hugging Face; returns its local path."""
    from huggingface_hub import snapshot_download
    fid = name_or_id
    if not (str(name_or_id).startswith("F") and len(str(name_or_id)) == 11):
        c = catalog(local_dir)
        m = c[c["name"].str.lower() == str(name_or_id).lower()]
        if year:
            m = m[m.year == year]
        if m.empty:
            raise SystemExit(f"no fire named {name_or_id!r}; try V.catalog() to browse")
        m = m.sort_values("perimeters", ascending=False)
        fid = m.iloc[0]["fire_id"]
        print(f"{m.iloc[0]['name']} {m.iloc[0]['year']}: {fid}, {m.iloc[0]['perimeters']} perimeters, "
              f"{m.iloc[0]['final_acres']:,.0f} acres, QC {m.iloc[0]['rating']}")
    snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[f"fires/{fid}/*"], local_dir=local_dir)
    return Path(local_dir) / "fires" / fid


def open_local(path):
    return Path(path).expanduser()


def _z(d, name):
    p = Path(d) / f"{name}.npz"
    return dict(np.load(p, allow_pickle=False)) if p.exists() else None


def _grid(d):
    return json.loads((Path(d) / "grid.json").read_text())


def _extent(g):
    dx, _, x0, _, dy, y0 = g["transform"]
    return [x0, x0 + dx * g["width"], y0 + dy * g["height"], y0]


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _outline(ax, mask, ext, color="k", lw=0.8):
    if mask.any():
        H, W = mask.shape
        xs = np.linspace(ext[0], ext[1], W)
        ys = np.linspace(ext[3], ext[2], H)
        ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=color, linewidths=lw)


def _clean(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _hillshade(elev, az=315, alt=45):
    gy, gx = np.gradient(elev.astype(float))
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy) / 30)
    aspect = np.arctan2(-gx, gy)
    a, z = np.radians(az), np.radians(alt)
    return np.clip(np.sin(z) * np.sin(slope) + np.cos(z) * np.cos(slope) * np.cos(a - aspect), 0, 1)


# ------------------------------------------------------------------------------- figures
def progression(d):
    """Perimeters over time (colour = which perimeter first contained the cell), the area curve,
    and the gaps between perimeters (the model's time step)."""
    plt = _plt()
    g, lab = _grid(d), _z(d, "labels")
    arr, times = lab["arrival"], pd.to_datetime(lab["times"], utc=True)
    ext = _extent(g)
    n = len(times)
    km2 = (g["transform"][0] * 111.32 * math.cos(math.radians(ext[2])) * abs(g["transform"][4]) * 111.32)
    area = [((arr >= 0) & (arr <= i)).sum() * km2 for i in range(n)]
    gaps = np.diff(times.view("int64")) / 3.6e12
    fig = plt.figure(figsize=(14, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    cm = plt.get_cmap("plasma", n)
    im = ax.imshow(np.ma.masked_less(arr, 0), cmap=cm, vmin=-0.5, vmax=n - 0.5, extent=ext, interpolation="nearest")
    if lab["other"].any():
        ax.imshow(np.ma.masked_where(~lab["other"], lab["other"]), cmap="Greys", alpha=0.5, extent=ext)
    _clean(ax, f"{g['name']} {g['year']}: {n} perimeters (grey = other fires)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    ticks = np.linspace(0, n - 1, min(n, 6)).round().astype(int)
    cb.set_ticks(ticks)
    cb.set_ticklabels([times[t].strftime("%b %d") for t in ticks])
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(times, area, "o-", ms=3)
    ax.set_ylabel("burned area (km²)")
    ax.set_title("Growth between perimeters", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax = fig.add_subplot(1, 3, 3)
    ax.bar(range(len(gaps)), gaps, color=["#c44" if (x < 3 or x > 168) else "#48a" for x in gaps])
    ax.axhline(24, color="k", lw=0.6, ls=":")
    ax.set_xlabel("pair")
    ax.set_ylabel("gap (hours)")
    ax.set_title("Time between perimeters (red = outside 3 h-7 d, not used)", fontsize=9)
    fig.tight_layout()
    src = pd.Series(lab["sources"]).value_counts().to_dict()
    meth = pd.Series(lab["map_methods"]).value_counts().head(3).to_dict()
    print(f"sources {src}; mapping methods {meth}; median gap {np.median(gaps):.1f} h")
    return fig


def layers(d, pair=None):
    """Every input layer on its own panel, the final perimeter (or perimeter `pair`) drawn on top:
    if a layer is shifted or empty it shows here."""
    plt = _plt()
    g, lab = _grid(d), _z(d, "labels")
    ext = _extent(g)
    arr = lab["arrival"]
    i = len(lab["times"]) - 1 if pair is None else pair
    fin = (arr >= 0) & (arr <= i)
    fig, axs = plt.subplots(3, 4, figsize=(16, 11.5))
    axs = axs.ravel()
    k = 0

    def panel(img, title, **kw):
        nonlocal k
        ax = axs[k]
        k += 1
        if img is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes, color="#a00")
        else:
            ax.imshow(img, extent=ext, interpolation="nearest", **kw)
        _outline(ax, fin, ext, "red" if kw.get("cmap") in ("gray", "Greys") else "k")
        _clean(ax, title)
        return ax

    t = _z(d, "terrain")
    panel(_hillshade(t["elev"]), f"terrain (hillshade, {t['elev'].min():.0f}-{t['elev'].max():.0f} m)", cmap="gray")
    lf = _z(d, "landfire")
    if lf is not None and not bool(lf.get("missing", True)):
        import data as D
        fam = np.full(lf["fbfm40"].shape, 0)
        for j, (_, a, b) in enumerate(D.FUEL_FAMILIES):
            fam[(lf["fbfm40"] >= a) & (lf["fbfm40"] <= b)] = j
        from matplotlib.colors import ListedColormap
        panel(fam, f"fuel family ({lf['version']})", cmap=ListedColormap(FUEL_COLORS), vmin=-0.5, vmax=6.5)
        panel(np.clip(lf["cc"], 0, 100), "canopy cover (%)", cmap="Greens", vmin=0, vmax=100)
    else:
        panel(None, "fuel family")
        panel(None, "canopy cover")
    tz = _z(d, "tessera")
    if tz is not None and not bool(tz.get("missing", True)):
        e = tz["embedding"].astype(np.float32)
        cov = tz["cover"]
        X = e[:, cov].T
        Xm = X.mean(0)
        _, _, vt = np.linalg.svd(X[:: max(1, len(X) // 20000)] - Xm, full_matrices=False)
        rgb = np.tensordot(vt[:3], e - Xm[:, None, None], axes=1).transpose(1, 2, 0)
        lo, hi = np.percentile(rgb[cov], 2, axis=0), np.percentile(rgb[cov], 98, axis=0)
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
        rgb[~cov] = 1
        panel(rgb, f"Tessera {int(tz['year'])} (first 3 components as colour)")
    else:
        panel(None, "Tessera")
    hz = _z(d, "human")
    if hz is not None:
        names = [str(x) for x in hz["channels"]]
        L = hz["layers"].astype(np.float32)
        get = lambda n: L[names.index(n)] if n in names else 0  # noqa: E731
        img = np.ones(L.shape[1:] + (3,))
        for n, col in (("developed", (0.85, 0.75, 0.75)), ("water", (0.6, 0.75, 0.95)), ("river", (0.3, 0.5, 0.9)),
                       ("track_trail", (0.6, 0.6, 0.6)), ("road_minor", (0.35, 0.35, 0.35)), ("road_major", (0.9, 0.4, 0.1)),
                       ("power_line", (0.6, 0.2, 0.8)), ("building", (0.1, 0.1, 0.1))):
            m = np.asarray(get(n)) > 0
            if np.ndim(m):
                img[m] = col
        panel(img, f"OpenStreetMap {str(hz.get('snapshot', ''))} (roads orange/grey, water blue)")
    else:
        panel(None, "OpenStreetMap")
    hi_ = _z(d, "history")
    if hi_ is not None:
        ys = hi_["years_since"].astype(float)
        panel(np.ma.masked_where(ys >= 255, ys), "years since last fire (blank = none)", cmap="viridis_r", vmin=0, vmax=30)
    else:
        panel(None, "fire history")
    v = _z(d, "viirs")
    if v is not None:
        cnt = (v["af_day"].astype(int) + v["af_night"]).sum(0)
        panel(np.ma.masked_equal(cnt, 0), f"VIIRS detections, all {len(v['dates'])} days", cmap="hot_r")
    else:
        panel(None, "VIIRS")
    gz = _z(d, "goes")
    ax = panel(np.zeros((g["height"], g["width"])), "GOES fire pixels (parallax-corrected)", cmap="Greys", vmin=0, vmax=1)
    if gz is not None and len(gz["t"]):
        ax.scatter(gz["lon"], gz["lat"], s=3, c=pd.to_datetime(gz["t"], utc=True).view("int64"), cmap="plasma")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    elif gz is None:
        ax.text(0.5, 0.5, "missing (GOES starts mid-2017)", ha="center", va="center", transform=ax.transAxes, color="#a00")
    sz = _z(d, "suppression")
    if sz is not None and bool(sz["has_spatial"]):
        n_ch = int(sz["n_channels"])
        lay = np.unpackbits(sz["layers"], axis=1)[:, :n_ch].max(0)
        names = [str(x) for x in sz["channels"]]
        img = np.ones(lay.shape[1:] + (3,))
        for n, col in (("dozer_line", (0.8, 0.2, 0.1)), ("hand_line", (0.9, 0.6, 0.1)), ("other_line", (0.5, 0.5, 0.5)),
                       ("burnout", (0.2, 0.2, 0.2)), ("retardant", (0.9, 0.1, 0.5)), ("uncontrolled_edge", (0.2, 0.4, 0.9))):
            if n in names:
                img[lay[names.index(n)] > 0] = col
        panel(img, "suppression lines, all days (dozer red, hand orange, retardant pink)")
    elif sz is not None and bool(sz["has_ics"]):
        panel(None, "suppression: daily ICS-209 numbers only (no line maps before 2018)")
    else:
        panel(None, "suppression")
    rz = _z(d, "hrrr")
    if rz is not None:
        F = rz["fields"].astype(np.float32)
        spd = np.nanmean(np.hypot(F[:, 1], F[:, 2]).reshape(F.shape[0], -1)[:, rz["idx"]], 0)
        ax = panel(spd, f"HRRR mean wind speed (m/s), {len(rz['times'])} hours", cmap="Blues")
        u = np.nanmean(F[:, 1].reshape(F.shape[0], -1)[:, rz["idx"]], 0)
        vv = np.nanmean(F[:, 2].reshape(F.shape[0], -1)[:, rz["idx"]], 0)
        s = max(1, min(u.shape) // 12)
        H, W = u.shape
        xs, ys = np.meshgrid(np.linspace(ext[0], ext[1], W)[::s], np.linspace(ext[3], ext[2], H)[::s])
        ax.quiver(xs, ys, u[::s, ::s], vv[::s, ::s], color="k", scale=60, width=0.004)
    else:
        panel(None, "HRRR (starts Aug 2014)")
    wz = _z(d, "weather")
    ax = axs[k]
    k += 1
    if wz is not None:
        names = [str(x) for x in wz["gm_vars"]]
        dates = pd.to_datetime(wz["dates"])
        gm = wz["gm"]
        ax.plot(dates, np.nanmean(gm[:, names.index("erc")], axis=(1, 2)), label="ERC")
        ax2 = ax.twinx()
        ax2.plot(dates, np.nanmean(gm[:, names.index("vpd")], axis=(1, 2)), color="#c63", label="VPD")
        t = pd.to_datetime(lab["times"], utc=True).tz_convert(None)
        ax.axvspan(t.min(), t.max(), color="#fdd", zorder=-1)
        ax.set_title("GRIDMET ERC (blue) and VPD (orange); red band = fire", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    for a in axs[k:]:
        a.axis("off")
    fig.tight_layout()
    return fig


def pairs(d, cols=8):
    """Every perimeter pair as one small panel: grey = burned before, red = new burn (the target),
    yellow = new burn excluded from scoring (far from the fire and no satellite fire nearby)."""
    plt = _plt()
    import data as D
    f = D.Fire(d)
    n = len(f.pairs)
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.2 * rows))
    axs = np.atleast_1d(axs).ravel()
    rr, cc = np.nonzero(f.arrival >= 0)
    r0, r1 = max(rr.min() - 10, 0), min(rr.max() + 10, f.H)
    c0, c1 = max(cc.min() - 10, 0), min(cc.max() + 10, f.W)
    for ax, (i, gap) in zip(axs, f.pairs):
        _, _, y, m, b0 = f.sample(i, gap, (r0, r1, c0, c1))
        img = np.ones(b0.shape + (3,))
        img[b0] = (0.7, 0.7, 0.7)
        img[y & m] = (0.85, 0.1, 0.1)
        img[y & ~m & ~b0] = (0.95, 0.8, 0.1)
        ax.imshow(img, interpolation="nearest")
        km2 = (y & m).sum() * f.cell_km ** 2
        _clean(ax, f"#{i} +{gap:.0f} h, {km2:.1f} km²")
    for ax in axs[n:]:
        ax.axis("off")
    fig.suptitle(f"{f.g['name']} {f.year}: {n} training pairs (grey before, red new, yellow excluded)", fontsize=11)
    fig.tight_layout()
    return fig


def _eval_window(f, i):
    import train_hybrid as TH
    TH.set_grid(f.cell_km)
    import evaluate as E
    return E.window(f, i)


def _physics_forecast(f, i, gap, win):
    """Rothermel + solver with no learning, the same as the physics_v1 run."""
    import torch
    import train_hybrid as TH
    import data as D
    feats, phys, y, m, b0 = f.sample(i, gap, win)
    model = TH.build_model("physics", sum(D.GROUPS.values()), 8, "cpu", dx=f.res_m)
    P = torch.from_numpy(TH.pad8(np.stack([phys["r_head"], phys["ecc"], phys["head_x"], phys["head_y"]])))[None].float()
    bb = torch.from_numpy(TH.pad8(b0[None]))[None, 0]
    x = torch.zeros(1, 1, *P.shape[-2:])
    with torch.no_grad():
        lg, _ = TH.forward(model, "physics", x, P, bb, torch.tensor([gap / 24]))
    return feats, phys, y, m, b0, lg[0, :y.shape[0], :y.shape[1]].numpy()


def _matched(score, y, m):
    """Flag exactly as many cells as actually burned (fair picture for any model's scale)."""
    k = int((y & m).sum())
    pred = np.zeros_like(m)
    if k:
        idx = np.argsort(-np.where(m, score, -np.inf), axis=None)[:k]
        pred.flat[idx] = True
    return pred


def _tpfp(ax, b0, pred, y, m, title):
    img = np.ones(b0.shape + (3,))
    img[b0] = (0.75, 0.75, 0.75)
    img[pred & y & m] = (0.1, 0.6, 0.1)        # hit
    img[pred & ~y & m] = (0.95, 0.6, 0.1)      # false alarm
    img[~pred & y & m] = (0.8, 0.1, 0.1)       # missed
    img[~m & ~b0] = img[~m & ~b0] * 0.92
    ax.imshow(img, interpolation="nearest")
    tp, fp, fn = (pred & y & m).sum(), (pred & ~y & m).sum(), (~pred & y & m).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    _clean(ax, f"{title}\nF1 {f1:.2f} (green hit, orange false alarm, red missed)")


def pair(d, i=None):
    """One pair exactly as the training and evaluation code build it."""
    plt = _plt()
    import data as D
    from scipy.ndimage import distance_transform_edt
    f = D.Fire(d)
    if i is None:
        i = max(f.pairs, key=lambda p: f.sample(p[0], p[1], (0, f.H, 0, f.W))[2].sum())[0]
    gap = dict(f.pairs)[i]
    win = _eval_window(f, i)
    feats, phys, y, m, b0, lg = _physics_forecast(f, i, gap, win)
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.6))
    axs = axs.ravel()
    fi = feats["fire"]
    img = np.ones(b0.shape + (3,))
    img[fi[2] > 0] = (0.55, 0.55, 0.55)
    img[(fi[1] > 0) & (fi[2] == 0)] = (0.7, 0.7, 0.7)
    img[(fi[0] > 0) & (fi[1] == 0)] = (0.85, 0.85, 0.85)
    axs[0].imshow(img, interpolation="nearest")
    _clean(axs[0], f"fire at perimeter #{i} (darker = older)\n{f.times[i]:%Y-%m-%d %H:%M} UTC, next in {gap:.0f} h")
    vi = feats["viirs"]
    axs[1].imshow(np.ma.masked_equal(vi[0] + vi[1], 0), cmap="hot_r", interpolation="nearest")
    _outline(axs[1], b0, [0, b0.shape[1], b0.shape[0], 0])
    axs[1].set_xlim(0, b0.shape[1]); axs[1].set_ylim(b0.shape[0], 0)
    _clean(axs[1], "VIIRS detections, 24 h before (input)")
    hr = feats["hrrr"]
    if hr[6].max() > 0:
        spd = hr[2] * 8
        axs[2].imshow(spd, cmap="Blues", interpolation="nearest")
        s = max(1, min(spd.shape) // 14)
        H, W = spd.shape
        X, Y = np.meshgrid(np.arange(W)[::s], np.arange(H)[::s])
        axs[2].quiver(X, Y, hr[0][::s, ::s], hr[1][::s, ::s], scale=12, width=0.004)
        _outline(axs[2], b0, [0, W, H, 0])
        axs[2].set_xlim(0, W); axs[2].set_ylim(H, 0)
        _clean(axs[2], f"HRRR wind over the gap (mean {spd.mean():.1f} m/s, arrows = toward)")
    else:
        axs[2].text(0.5, 0.5, "no HRRR for this pair", ha="center", transform=axs[2].transAxes)
        _clean(axs[2], "HRRR wind")
    R = phys["r_head"]
    im = axs[3].imshow(np.log10(np.clip(R, 1e-2, None)), cmap="inferno", vmin=-1, vmax=2, interpolation="nearest")
    s = max(1, min(R.shape) // 14)
    H, W = R.shape
    X, Y = np.meshgrid(np.arange(W)[::s], np.arange(H)[::s])
    axs[3].quiver(X, Y, phys["head_x"][::s, ::s], phys["head_y"][::s, ::s], color="c", scale=25, width=0.004)
    _outline(axs[3], b0, [0, W, H, 0], "w")
    axs[3].set_xlim(0, W); axs[3].set_ylim(H, 0)
    fig.colorbar(im, ax=axs[3], fraction=0.046, label="log10 m/min")
    _clean(axs[3], f"Rothermel head-fire spread rate (median {np.median(R[R > 0]):.1f} m/min)\narrows = head direction")
    img = np.ones(b0.shape + (3,))
    img[b0] = (0.75, 0.75, 0.75)
    img[y & m] = (0.85, 0.1, 0.1)
    img[y & ~m & ~b0] = (0.95, 0.8, 0.1)
    img[~m & ~b0 & ~y] = (0.93, 0.93, 0.97)
    axs[4].imshow(img, interpolation="nearest")
    _clean(axs[4], f"target: new burn by #{i + 1} (red, {(y & m).sum() * f.cell_km ** 2:.1f} km²)\n"
                   f"yellow/blue-grey = not scored")
    p = 1 / (1 + np.exp(-np.clip(lg, -30, 30)))
    im = axs[5].imshow(np.where(b0, np.nan, p), cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
    _outline(axs[5], y | b0, [0, W, H, 0], "c")
    axs[5].set_xlim(0, W); axs[5].set_ylim(H, 0)
    fig.colorbar(im, ax=axs[5], fraction=0.046)
    _clean(axs[5], "physics-only burn probability\n(cyan = actual next perimeter)")
    _tpfp(axs[6], b0, _matched(lg, y, m), y, m, "physics-only, same area as burned")
    dist = distance_transform_edt(~b0)
    _tpfp(axs[7], b0, _matched(-dist, y, m), y, m, "baseline: grow evenly outward")
    fig.tight_layout()
    print(f"pair #{i}: gap {gap:.1f} h, scored cells {m.sum():,}, new burned {(y & m).sum():,}, "
          f"excluded new burn {(y & ~m & ~b0).sum():,}")
    return fig


def crops(d, n=6, seed=0):
    """Random training crops as the data loader feeds them (after flips and hidden groups):
    input fire, target and the loss mask (crop edges are never scored)."""
    plt = _plt()
    import data as D
    import train_hybrid as TH
    f = D.Fire(d)
    TH.set_grid(f.cell_km)
    ds = TH.Crops([f], TH.groups_used(set()), n, seed=seed)
    fig, axs = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    for k in range(n):
        x, P, y, m, b0, gap = ds[k]
        img = np.ones(b0.shape + (3,))
        img[b0.numpy() > 0] = (0.7, 0.7, 0.7)
        img[(y.numpy() > 0) & (m.numpy() > 0)] = (0.85, 0.1, 0.1)
        axs[0, k].imshow(img, interpolation="nearest")
        _clean(axs[0, k], f"gap {float(gap) * 24:.0f} h")
        axs[1, k].imshow(m.numpy(), cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
        _clean(axs[1, k], "loss mask (white = scored)")
    fig.suptitle(f"Training crops ({TH.CROP} cells = {TH.CROP * f.cell_km:.0f} km): grey old burn, red target", fontsize=11)
    fig.tight_layout()
    return fig


def prediction(d, i, weights, mode="hybrid", pca=None, drop=()):
    """A trained model's forecast for pair i, raw and with per-fire calibration on pair i-1.
    weights: runs/<name>.pt (or .ckpt) copied from ICE; pca: runs/tessera_pca.npz from the same folder."""
    plt = _plt()
    import shutil
    import tempfile
    import data as D
    import evaluate as E
    import train_hybrid as TH
    pca_arr = None
    if pca:
        z = np.load(pca)
        pca_arr = (z["comp"], z["mean"])
    f = D.Fire(d, pca_arr)
    TH.set_grid(f.cell_km)
    tmp = Path(tempfile.mkdtemp())
    w = Path(weights)
    shutil.copy(w, tmp / f"m{w.suffix}")
    spec = f"m:{mode}" + (":" + "+".join(drop) if drop else "")
    mb = E.Member(tmp, spec, None, TH.groups_used(set()), "cpu", f.res_m, 32)
    gap = dict(f.pairs)[i]
    win = E.window(f, i)
    feats, phys, y, m, b0 = f.sample(i, gap, win)
    st = mb.state(feats, phys, b0, gap)
    raw = mb.logits(st).numpy()
    panels = [("raw", raw)]
    if (i - 1) in dict(f.pairs):
        g0 = dict(f.pairs)[i - 1]
        f0 = f.sample(i - 1, g0, E.window(f, i - 1))
        a, s = mb.fit([(mb.state(f0[0], f0[1], f0[4], g0), f0[2], f0[3])])
        panels.append((f"calibrated on pair {i - 1} (a={a:+.2f})", mb.logits(st, a, s).numpy()))
    fig, axs = plt.subplots(2, len(panels), figsize=(5.2 * len(panels), 9.4))
    axs = np.array(axs).reshape(2, -1)
    H, W = b0.shape
    for k, (name, lg) in enumerate(panels):
        p = 1 / (1 + np.exp(-np.clip(lg, -30, 30)))
        im = axs[0, k].imshow(np.where(b0, np.nan, p), cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
        _outline(axs[0, k], y | b0, [0, W, H, 0], "c")
        axs[0, k].set_xlim(0, W); axs[0, k].set_ylim(H, 0)
        fig.colorbar(im, ax=axs[0, k], fraction=0.046)
        _clean(axs[0, k], f"{mode} {name}: burn probability")
        _tpfp(axs[1, k], b0, _matched(lg, y, m), y, m, f"{mode} {name}, same area as burned")
    fig.tight_layout()
    return fig


def crownspot(d, i=None):
    """Physics v2 fields for one pair: crown fire probability, how much crowning speeds spread,
    fireline intensity, and where embers land (cyan = what actually burned next)."""
    plt = _plt()
    import data as D
    D.PHYSICS_V2 = True
    f = D.Fire(d)
    if i is None:
        i = max(f.pairs, key=lambda p: f.sample(p[0], p[1], (0, f.H, 0, f.W))[2].sum())[0]
    gap = dict(f.pairs)[i]
    feats, phys, y, m, b0 = f.sample(i, gap, _eval_window(f, i))
    cs = feats["crownspot"]
    inten = np.expm1(cs[3] * 3) * 1000
    boost = np.log10(np.clip(phys["r_crown"], 1e-2, None)) - np.log10(np.clip(phys["r_head"], 1e-2, None))
    E = phys["ember"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
    panels = [(cs[0], "crown fire probability", dict(cmap="YlOrRd", vmin=0, vmax=1)),
              (boost, "crowning speed-up (log10 crown-adjusted / surface)", dict(cmap="magma", vmin=0, vmax=1.5)),
              (np.log10(inten + 1), "fireline intensity (log10 kW/m)", dict(cmap="inferno", vmin=1, vmax=4.5)),
              (np.where(b0, np.nan, np.log10(E + 1e-4)), "ember landing (log), cyan = burned next", dict(cmap="viridis", vmin=-4, vmax=0))]
    for ax, (img, title, kw) in zip(axs, panels):
        im = ax.imshow(img, interpolation="nearest", **kw)
        _outline(ax, b0, [0, b0.shape[1], b0.shape[0], 0], "w")
        ax.set_xlim(0, b0.shape[1]); ax.set_ylim(b0.shape[0], 0)
        fig.colorbar(im, ax=ax, fraction=0.046)
        _clean(ax, title)
    _outline(axs[3], y & m, [0, b0.shape[1], b0.shape[0], 0], "c")
    fig.tight_layout()
    D.PHYSICS_V2 = False
    new = y & m
    print(f"pair #{i}: ember density on cells that burned next is "
          f"{E[new].mean() / max(E[m & ~y].mean(), 1e-9):.1f}x that on cells that did not")
    return fig
