"""
One-page picture of every layer for a built fire, to check the multi-year dataset by eye.
  python quicklook_multiyear.py --out data/my --fires F123 F456        -> OUT/quicklook/<fire>.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def show(ax, img, title, cmap="viridis", **kw):
    ax.imshow(img, cmap=cmap, interpolation="nearest", **kw)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def missing(ax, title):
    ax.text(0.5, 0.5, "not available\n(year / source)", ha="center", va="center", fontsize=8, color="#888")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def load(d, name):
    p = d / f"{name}.npz"
    return dict(np.load(p, allow_pickle=False)) if p.exists() else None


def quicklook(d, dst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = json.loads((d / "grid.json").read_text())
    lab = load(d, "labels")
    fig, axes = plt.subplots(3, 5, figsize=(17, 10.5))
    ax = iter(axes.ravel())
    arr = lab["arrival"].astype(float)
    arr[arr < 0] = np.nan
    a = next(ax)
    show(a, arr, f"perimeters: arrival index (0..{len(lab['times']) - 1})", "plasma")
    a.contour(lab["other"], levels=[0.5], colors="cyan", linewidths=0.6)

    ter = load(d, "terrain")
    show(next(ax), ter["elev"], "elevation (m)", "terrain")
    show(next(ax), ter["tpi_2km"].astype(float), "terrain position 2 km (ridge +)", "RdBu_r", vmin=-150, vmax=150)

    lf = load(d, "landfire")
    if lf is not None and not bool(lf["missing"]):
        fam = np.digitize(lf["fbfm40"], [91, 100, 120, 140, 160, 180, 200, 210])
        show(next(ax), fam, f"LANDFIRE fuel family ({lf['version']}): NB/GR/GS/SH/TU/TL/SB", "tab10", vmin=0, vmax=9)
        show(next(ax), lf["cc"], "canopy cover (%)", "Greens", vmin=0, vmax=100)
    else:
        missing(next(ax), "LANDFIRE fuels")
        missing(next(ax), "canopy cover")

    tz = load(d, "tessera")
    a = next(ax)
    if tz is not None and not bool(tz["missing"]):
        e = tz["embedding"].astype(np.float32)
        X = e.reshape(e.shape[0], -1).T
        Xc = X - X.mean(0)
        _, _, vt = np.linalg.svd(Xc[np.random.default_rng(0).choice(len(Xc), min(5000, len(Xc)), replace=False)], full_matrices=False)
        rgb = (Xc @ vt[:3].T).reshape(e.shape[1], e.shape[2], 3)
        lo, hi = np.percentile(rgb, 2, (0, 1)), np.percentile(rgb, 98, (0, 1))
        show(a, np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1), f"Tessera {int(tz['year'])} (first 3 PCs)")
    else:
        missing(a, "Tessera")

    hi_ = load(d, "history")
    ys = hi_["years_since"].astype(float)
    ys[ys == 255] = np.nan
    show(next(ax), ys, f"years since last fire ({int(hi_['n_polygons'])} past perimeters)", "magma_r", vmin=0, vmax=40)

    v = load(d, "viirs")
    show(next(ax), np.log1p(v["af_day"].sum(0) + v["af_night"].sum(0)), f"VIIRS detections, all days ({int(v['n_points'])} pts)", "hot")

    gz = load(d, "goes")
    a = next(ax)
    if gz is not None and len(gz["t"]):
        a.imshow(np.isnan(arr), cmap="Greys", alpha=0.15)
        dx, _, x0, _, ey, y0 = g["transform"]
        c = (gz["lon"] - x0) / dx
        r = (gz["lat"] - y0) / ey
        t = (pd.to_datetime(gz["t"]) - pd.to_datetime(gz["t"]).min()).total_seconds() / 86400
        a.scatter(c, r, c=t, s=2, cmap="viridis")
        a.set_xlim(0, g["width"])
        a.set_ylim(g["height"], 0)
        a.set_title(f"GOES fire pixels ({len(gz['t'])}, colour = day, sats {sorted(set(gz['sat']))})", fontsize=8)
        a.set_xticks([])
        a.set_yticks([])
    else:
        missing(a, "GOES fire pixels")

    rz = load(d, "hrrr")
    a = next(ax)
    if rz is not None:
        F = rz["fields"].astype(np.float32)
        spd = np.hypot(F[:, 1], F[:, 2])
        a.plot(pd.to_datetime(rz["times"]), np.nanmean(spd, (1, 2)), lw=0.8, label="mean wind")
        a.plot(pd.to_datetime(rz["times"]), np.nanmax(F[:, 0], (1, 2)), lw=0.8, label="max gust")
        for t in pd.to_datetime(lab["times"]):
            a.axvline(t, color="#ccc", lw=0.4)
        a.legend(fontsize=6)
        a.set_title("HRRR every 3 h (m/s); grey = perimeter times", fontsize=8)
        a.tick_params(labelsize=6)
    else:
        missing(a, "HRRR")

    wz = load(d, "weather")
    a = next(ax)
    names = [str(x) for x in wz["gm_vars"]]
    gm = wz["gm"]
    dates = pd.to_datetime(wz["dates"])
    a.plot(dates, np.nanmean(gm[:, names.index("fm100")], (1, 2)), label="100-h fuel moisture %")
    a.plot(dates, np.nanmean(gm[:, names.index("erc")], (1, 2)) / 5, label="ERC / 5")
    a.legend(fontsize=6)
    a.tick_params(labelsize=6)
    a.set_title("GRIDMET (14 days before to end)", fontsize=8)

    sz = load(d, "suppression")
    a = next(ax)
    if sz is not None and bool(sz["has_spatial"]):
        n = int(sz["n_channels"])
        lay = np.unpackbits(sz["layers"], axis=1)[:, :n]
        show(a, lay[-1].max(0), f"suppression lines/drops, last day ({', '.join(str(c) for c in sz['channels'][:3])}...)", "Reds")
    else:
        ics = "with ICS-209 crews/aircraft" if sz is not None and bool(sz["has_ics"]) else "no ICS-209 match"
        missing(a, f"suppression lines ({ics})")

    hz = load(d, "human")
    a = next(ax)
    if hz is not None:
        L = hz["layers"]
        ch = [str(c) for c in hz["channels"]]
        img = np.zeros(L.shape[1:] + (3,))
        img[..., 0] = L[[ch.index(c) for c in ch if c.startswith("road") or c == "track_trail"]].max(0)
        img[..., 1] = L[ch.index("building")] if "building" in ch else 0
        if "water" in ch:
            img[..., 2] = np.maximum(L[ch.index("water")], L[ch.index("river")])
        show(a, img, "OSM: roads (red), buildings (green), water (blue)")
    else:
        missing(a, "OSM human features")

    for a in ax:
        a.axis("off")
    t0, t1 = pd.to_datetime(lab["times"][0]), pd.to_datetime(lab["times"][-1])
    fig.suptitle(f"{g['name']} ({g['year']}) {g['fire_id']}  {g['height']}x{g['width']} cells at 150 m  "
                 f"{len(lab['times'])} perimeters {t0:%Y-%m-%d} to {t1:%Y-%m-%d}", fontsize=11)
    fig.tight_layout()
    fig.savefig(dst, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fires", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "quicklook").mkdir(exist_ok=True)
    dirs = [out / "fires" / f for f in a.fires] if a.fires else sorted((out / "fires").glob("*"))
    for d in dirs:
        if (d / "labels.npz").exists():
            quicklook(d, out / "quicklook" / f"{d.name}.png")
            print("wrote", out / "quicklook" / f"{d.name}.png")


if __name__ == "__main__":
    main()
