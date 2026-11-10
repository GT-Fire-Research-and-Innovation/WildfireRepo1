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

Placeholder shapes, stale repeats and jumping / shifted perimeters get drop=True (build_multiyear.py leaves
them out of the labels). A fire is kept when at least 2 good perimeters remain, else rated bad and not built;
after the build, a fire whose new burn had satellite fire nearby less than 20 % of the time is also bad.
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
    """Per perimeter, three passes:
    1. shape: placeholder circles / boxes / triangles / generated shapes are dropped;
    2. forward: exact repeats of the previous kept perimeter (stale copies) are dropped;
    3. anchored on the fire's largest real perimeter (its true footprint): earlier perimeters, walking
       backwards, must sit inside the next kept one (>= OVERLAP_MIN) and not jump (centroid > JUMP_KM and
       farther than that fire's radius); later re-maps must cover the same ground as the anchor.
    Big growth is kept (real blow-up days) but flagged."""
    import geopandas as gpd
    per = gpd.read_parquet(Path(catalog) / "perimeters.parquet").to_crs(5070)
    per = per.sort_values(["fire_id", "perim_time"]).reset_index(drop=True)
    rows = []
    for fid, grp in per.groupby("fire_id", sort=False):
        geoms = list(grp.geometry)
        recs = []
        for k, (idx, r) in enumerate(grp.iterrows()):
            m = shape_metrics(r.geometry)
            m.update(fire_id=fid, k=k, perim_time=r.perim_time, map_method=r.map_method, source=r.source)
            m["_flags"] = shape_flags(m)
            m["drop"] = any(f in PLACEHOLDER for f in m["_flags"])
            recs.append(m)
        prev = None                                   # pass 2: stale repeats
        for m, g in zip(recs, geoms):
            if m["drop"]:
                continue
            if prev is not None and abs(g.area - prev.area) < 1e-3 * max(prev.area, 1) and \
                    prev.intersects(g) and prev.intersection(g).area / max(prev.area, 1e-9) > 0.999:
                m["_flags"].append("repeat")
                m["drop"] = True
                continue
            prev = g
        # pass 3: consistency with the fire's footprint, anchored on its LARGEST real perimeter (a final
        # perimeter can be a small partial re-map; the largest one is the fire's true extent)
        live = [i for i, m in enumerate(recs) if not m["drop"]]
        if live:
            anchor = max(live, key=lambda i: geoms[i].area)
            A = geoms[anchor]
            nxt = A                                   # before the anchor: walk backwards, each inside the next kept
            for i in reversed([i for i in live if i < anchor]):
                g, m = geoms[i], recs[i]
                inside = g.intersection(nxt).area / max(g.area, 1e-9) if g.intersects(nxt) else 0.0
                jump = g.centroid.distance(nxt.centroid) / 1000
                radius = math.sqrt(max(nxt.area, 1.0) / math.pi) / 1000
                m["inside_next"], m["jump_km"] = round(inside, 3), round(jump, 2)
                if inside < OVERLAP_MIN:
                    # not inside the later footprint: misplaced / shifted (a jump on top of that = "jump")
                    m["_flags"].append("jump" if (jump > JUMP_KM and jump > radius) else "shifted")
                    m["drop"] = True
                else:
                    if jump > JUMP_KM and jump > radius:
                        m["_flags"].append("lopsided_growth")   # inside the later fire: real one-sided run, kept
                    nxt = g
            for i in [i for i in live if i > anchor]:  # after the anchor: re-maps must cover the same ground
                g, m = geoms[i], recs[i]
                inter = g.intersection(A).area if g.intersects(A) else 0.0
                m["inside_next"] = round(inter / max(g.area, 1e-9), 3)
                if inter / max(g.area, 1e-9) < OVERLAP_MIN or inter / max(A.area, 1e-9) < OVERLAP_MIN:
                    m["_flags"].append("shifted")
                    m["drop"] = True
        prev = None                                   # report growth / overlap vs the previous kept one
        for m, g in zip(recs, geoms):
            if m["drop"]:
                continue
            if prev is not None:
                m["overlap_prev"] = prev.intersection(g).area / max(prev.area, 1e-9) if prev.intersects(g) else 0.0
                m["growth"] = g.area / max(prev.area, 1e-9)
                if m["growth"] > GROWTH_MAX and m["acres"] > 1000:
                    m["_flags"].append("big_growth")
            prev = g
        for m in recs:
            m["flags"] = ",".join(m.pop("_flags"))
            rows.append(m)
    return pd.DataFrame(rows)


MIN_KEEP = 2            # a fire with at least 2 good perimeters (one learnable pair) is kept


def rate(pq):
    out = []
    for fid, g in pq.groupby("fire_id"):
        fl = g["flags"].fillna("")
        usable = int((~g["drop"]).sum())
        out.append(dict(fire_id=fid, perimeters=len(g), usable=usable,
                        placeholder=int(fl.str.contains("circle|box|generated|too_simple").sum()),
                        repeats=int(fl.str.contains("repeat").sum()),
                        jumps_shifts=int(fl.str.contains("jump|shifted").sum()),
                        big_growth=int(fl.str.contains("big_growth").sum()),
                        rating="keep" if usable >= MIN_KEEP else "bad",
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
        qf.loc[qf["new_burn_with_satellite_fire"] < 0.2, "rating"] = "bad"   # grew on paper, no satellite fire
    qf.to_csv(cat / "qc_fires.csv", index=False)
    print(qf["rating"].value_counts().to_string())
    print("\nperimeter flags:", pq["flags"].str.split(",").explode().value_counts().drop("", errors="ignore").to_dict())
    print("perimeters dropped (placeholder or exact repeat):", int(pq["drop"].sum()), "of", len(pq))
    print("wrote", cat / "qc_fires.csv", cat / "qc_perimeters.csv")


if __name__ == "__main__":
    main()
