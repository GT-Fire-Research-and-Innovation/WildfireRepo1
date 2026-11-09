"""
Quality checks for every fire in the catalog, so bad fires can be reviewed and dropped.

Level 1, shape (catalog only, runs anywhere in minutes):
  placeholder shapes   circles (circularity), boxes (rectangle fill with few vertices), evenly spaced
                       vertices (buffer-made shapes)
  sequence             overlap with the previous perimeter, area growth, centroid jumps,
                       identical repeats (stale copies with new dates)
Level 2, satellite evidence (needs the built dataset, run on ICE after build_multiyear.py):
  share of each day's new burn with a VIIRS or GOES fire detection within 1 km, and share of
  detections near the fire that fall outside the perimeter

  python qc_fires.py --catalog data/catalog                       -> catalog/qc_perimeters.csv, qc_fires.csv
  python qc_fires.py --catalog data/catalog --built data/my       -> adds satellite agreement

Placeholder perimeters and exact repeats get drop=True (build_multiyear.py leaves them out of the labels).
Fire rating, from the kept perimeters: bad (< 3 usable or > 30 % jumps/shifts; not built), review
(any jump, shift or x30 growth), good.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# thresholds (natural fire perimeters are ragged: circularity typically < 0.4)
CIRCLE = 0.85            # 4*pi*A/P^2 above this = circle-like
BOX_FILL = 0.97          # area / minimum rotated rectangle above this, with <= BOX_VERTS vertices = box
BOX_VERTS = 8
EVEN_EDGES_CV = 0.08     # coefficient of variation of edge lengths below this (>= 16 vertices) = generated shape
SIMPLE_VERTS = 6         # this few vertices on a >= 500 acre perimeter = hand-drawn placeholder
OVERLAP_MIN = 0.6        # share of the previous perimeter still inside the new one
JUMP_KM = 5.0            # centroid moves farther than this AND more than 1 fire radius
GROWTH_MAX = 30.0        # area x30 between consecutive perimeters (only flagged above 1,000 acres)


def shape_metrics(geom):
    """Metrics in EPSG:5070 metres for one (multi)polygon."""
    import shapely
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    big = max(polys, key=lambda p: p.area)
    area, perim = geom.area, geom.length
    ring = np.asarray(big.exterior.coords)
    edges = np.hypot(*np.diff(ring, axis=0).T)
    n = len(ring) - 1
    mrr = shapely.minimum_rotated_rectangle(big).area
    return dict(
        acres=area / 4046.86,
        vertices=n,
        parts=len(polys),
        circularity=4 * math.pi * big.area / max(big.length, 1e-9) ** 2,
        rect_fill=big.area / max(mrr, 1e-9),
        hull_fill=big.area / max(big.convex_hull.area, 1e-9),
        edge_cv=float(edges.std() / max(edges.mean(), 1e-9)) if n >= 3 else 0.0,
        perim_m=perim,
    )


def shape_flags(m):
    f = []
    if m["circularity"] > CIRCLE:
        f.append("circle")
    if m["rect_fill"] > BOX_FILL and m["vertices"] <= BOX_VERTS:
        f.append("box")
    if m["vertices"] >= 16 and m["edge_cv"] < EVEN_EDGES_CV and m["hull_fill"] > 0.97:
        f.append("generated")
    if m["vertices"] <= SIMPLE_VERTS and m["acres"] >= 500 and "box" not in f:
        f.append("too_simple")
    return f


PLACEHOLDER = ("circle", "box", "generated", "too_simple")


def level1(catalog):
    """Per perimeter: shape flags; then sequence checks against the previous KEPT perimeter
    (placeholders and exact repeats are dropped, so they never make a real perimeter look wrong)."""
    import geopandas as gpd
    per = gpd.read_parquet(Path(catalog) / "perimeters.parquet").to_crs(5070)
    per = per.sort_values(["fire_id", "perim_time"]).reset_index(drop=True)
    rows = []
    for fid, grp in per.groupby("fire_id", sort=False):
        prev = None
        for k, (idx, r) in enumerate(grp.iterrows()):
            g = r.geometry
            m = shape_metrics(g)
            flags = shape_flags(m)
            m.update(fire_id=fid, k=k, perim_time=r.perim_time, map_method=r.map_method, source=r.source)
            placeholder = any(f in PLACEHOLDER for f in flags)
            repeat = False
            if prev is not None and not placeholder:
                inter = prev.intersection(g).area if prev.intersects(g) else 0.0
                m["overlap_prev"] = inter / max(prev.area, 1e-9)
                m["growth"] = g.area / max(prev.area, 1e-9)
                m["jump_km"] = prev.centroid.distance(g.centroid) / 1000
                radius_km = math.sqrt(max(prev.area, 1.0) / math.pi) / 1000
                repeat = bool(abs(g.area - prev.area) < 1e-3 * max(prev.area, 1) and m["overlap_prev"] > 0.999)
                if m["overlap_prev"] < OVERLAP_MIN:
                    flags.append("shifted")
                if m["jump_km"] > JUMP_KM and m["jump_km"] > radius_km:
                    flags.append("jump")
                if m["growth"] > GROWTH_MAX and m["acres"] > 1000:
                    flags.append("big_growth")
            if repeat:
                flags.append("repeat")
            m["flags"] = ",".join(flags)
            m["drop"] = placeholder or repeat
            rows.append(m)
            if not m["drop"]:
                prev = g
    return pd.DataFrame(rows)


def rate(pq):
    out = []
    for fid, g in pq.groupby("fire_id"):
        n = len(g)
        fl = g["flags"].fillna("")
        placeholder = int(fl.str.contains("circle|box|generated|too_simple").sum())
        repeats = int(fl.str.contains("repeat").sum())
        kept = g[~g["drop"]]
        kfl = kept["flags"].fillna("")
        seq = int(kfl.str.contains("shifted|jump").sum())
        growth = int(kfl.str.contains("big_growth").sum())
        usable = len(kept)
        if usable < 3 or seq / max(usable, 1) > 0.3:
            rating = "bad"
        elif seq or growth:
            rating = "review"
        else:
            rating = "good"
        out.append(dict(fire_id=fid, perimeters=n, placeholder=placeholder, repeats=repeats, usable=usable,
                        sequence_flags=seq, big_growth=growth, rating=rating,
                        flags=";".join(f"{k}:{v}" for k, v in fl.str.split(",").explode().value_counts().items() if k)))
    return pd.DataFrame(out)


def level2(built, fires):
    """Satellite agreement from the built dataset (labels, viirs, goes)."""
    from scipy.ndimage import binary_dilation
    rows = []
    for fid in fires:
        d = Path(built) / "fires" / fid
        if not (d / "labels.npz").exists() or not (d / "viirs.npz").exists():
            continue
        lab = np.load(d / "labels.npz")
        v = np.load(d / "viirs.npz")
        arr = lab["arrival"]
        det = (v["af_day"].sum(0) + v["af_night"].sum(0)) > 0
        if (d / "goes.npz").exists():
            gz = np.load(d / "goes.npz")
            if len(gz["t"]):
                import json
                gg = json.loads((d / "grid.json").read_text())
                dx, _, x0, _, ey, y0 = gg["transform"]
                c = np.floor((gz["lon"] - x0) / dx).astype(int)
                r = np.floor((gz["lat"] - y0) / ey).astype(int)
                ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
                gd = np.zeros_like(det)
                gd[r[ok], c[ok]] = True
                det |= binary_dilation(gd, iterations=6)
        near = binary_dilation(det, iterations=7)                 # ~1 km
        burned = arr >= 1                                         # burned after the first perimeter
        new_with_fire = float(near[burned].mean()) if burned.any() else np.nan
        fire_zone = binary_dilation(arr >= 0, iterations=20)       # 3 km around the final perimeter
        outside = float((det & fire_zone & ~binary_dilation(arr >= 0, iterations=3)).sum() / max((det & fire_zone).sum(), 1))
        rows.append(dict(fire_id=fid, new_burn_with_satellite_fire=new_with_fire, detections_outside_perimeter=outside))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog")
    ap.add_argument("--built", help="built dataset folder (adds satellite agreement)")
    a = ap.parse_args()
    cat = Path(a.catalog)
    pq = level1(cat)
    pq.to_csv(cat / "qc_perimeters.csv", index=False)
    fires = pd.read_csv(cat / "fires.csv")
    qf = fires.merge(rate(pq), on=["fire_id"], suffixes=("", "_qc"))
    if a.built:
        l2 = level2(a.built, qf["fire_id"])
        qf = qf.merge(l2, on="fire_id", how="left")
        weak = qf["new_burn_with_satellite_fire"] < 0.5
        qf.loc[weak & (qf["rating"] == "good"), "rating"] = "review"
        qf.loc[qf["new_burn_with_satellite_fire"] < 0.2, "rating"] = "bad"
    qf.to_csv(cat / "qc_fires.csv", index=False)
    print(qf["rating"].value_counts().to_string())
    print("\nperimeter flags:", pq["flags"].str.split(",").explode().value_counts().drop("", errors="ignore").to_dict())
    print("perimeters dropped (placeholder or exact repeat):", int(pq["drop"].sum()), "of", len(pq))
    print("wrote", cat / "qc_fires.csv", cat / "qc_perimeters.csv")


if __name__ == "__main__":
    main()
