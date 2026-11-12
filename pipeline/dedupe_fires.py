"""
Find fires that are in the catalog twice: the same fire under two incident records, or a member fire
that is also inside its complex's perimeters. Both copies describe the same burn, so training on both
counts it twice and testing on both scores it twice.

Rules (same year only; reburns in later years are real and kept):
  duplicate  final perimeters overlap with IoU > 0.5                      -> keep the one with more perimeters
  nested     > 80% of the smaller fire lies inside the larger one AND the two were mapped over
             overlapping dates (at least half of the smaller fire's span)   -> keep the larger
A fire that burned first and was absorbed by a complex later (no date overlap) is kept: its early
perimeters are not in the complex's record.

  python pipeline/dedupe_fires.py --catalog ~/scratch/wildfire/catalog
Writes <catalog>/duplicates.csv (fire_id = the copy to leave out, kept = its twin). train_hybrid.py and
evaluate.py leave these fires out.
"""
import argparse
from pathlib import Path

import pandas as pd


def find(catalog):
    import geopandas as gpd
    cat = Path(catalog)
    p = gpd.read_parquet(cat / "perimeters.parquet")
    if (cat / "qc_fires.csv").exists():
        q = pd.read_csv(cat / "qc_fires.csv")
        p = p[~p.fire_id.isin(set(q.loc[q["rating"] == "bad", "fire_id"]))]
    span = p.groupby("fire_id").agg(t0=("perim_time", "min"), t1=("perim_time", "max"), n=("perim_time", "size"))
    fin = p.sort_values("acres_calc").groupby("fire_id").tail(1).reset_index(drop=True)
    fin["year"] = fin.perim_time.dt.year
    fin = fin.to_crs(5070)
    fin["geometry"] = fin.geometry.buffer(0)
    geoms, ids, years, names = fin.geometry.values, fin.fire_id.values, fin.year.values, fin.incident_name.values
    idx = fin.sindex
    rows, dropped = [], set()
    cand = []
    for i, ga in enumerate(geoms):
        for j in idx.query(ga, predicate="intersects"):
            if j <= i or years[i] != years[j]:
                continue
            inter = ga.intersection(geoms[j]).area
            if inter > 0:
                cand.append((inter / ga.union(geoms[j]).area, i, j, inter))
    for iou, i, j, inter in sorted(cand, reverse=True):               # strongest overlaps first
        a, b = ids[i], ids[j]
        if a in dropped or b in dropped:
            continue
        small, big = (i, j) if geoms[i].area <= geoms[j].area else (j, i)
        frac = inter / geoms[small].area
        if iou > 0.5:
            keep, drop = (a, b) if (span.n[a], geoms[i].area) >= (span.n[b], geoms[j].area) else (b, a)
            why = "duplicate"
        elif frac > 0.8:
            s, g = span.loc[ids[small]], span.loc[ids[big]]
            both = (min(s.t1, g.t1) - max(s.t0, g.t0)).total_seconds()
            if both < 0.5 * max((s.t1 - s.t0).total_seconds(), 1.0):
                continue                                              # burned first, absorbed later: keep both
            keep, drop, why = ids[big], ids[small], "nested"
        else:
            continue
        dropped.add(drop)
        k, d = list(ids).index(keep), list(ids).index(drop)
        rows.append(dict(fire_id=drop, name=names[d], year=int(years[d]), kept=keep, kept_name=names[k], why=why,
                         iou=round(iou, 3), inside=round(frac, 3), perimeters=int(span.n[drop]), kept_perimeters=int(span.n[keep])))
    return pd.DataFrame(rows, columns=["fire_id", "name", "year", "kept", "kept_name", "why", "iou", "inside", "perimeters",
                                       "kept_perimeters"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    a = ap.parse_args()
    df = find(a.catalog)
    out = Path(a.catalog) / "duplicates.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} fires to leave out ({(df.why == 'duplicate').sum()} duplicates, {(df.why == 'nested').sum()} nested) -> {out}")
    if len(df):
        split = df.year.map(lambda y: "test" if y >= 2024 else ("val" if y == 2023 else "train"))
        print("by split:", split.value_counts().to_dict())
        print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
