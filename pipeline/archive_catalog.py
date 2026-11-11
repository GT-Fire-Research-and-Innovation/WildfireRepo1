"""
Turn the live perimeter archive (perimeter-archive branch, filled every 2 h by live/wfigs_poller.py)
into a training catalog in the same format as pipeline/catalog.py, so new fires can be built with
build_multiyear.py and added to training (the continuous-learning loop, scripts/ice_continual.sh).

Cleaning matches the training catalog's rules where the archive has the information:
  - wildfires only (no prescribed burns), no hand sketches / modeled perimeters
  - one perimeter per local burn day (the last one that day)
  - shrinking re-maps dropped (a perimeter smaller than 90% of the largest so far)
  - kept only when the fire has >= --min-days burn days and >= --min-acres
  - fires still growing in the last --settle-days are left for a later run (their labels would change)

  python pipeline/archive_catalog.py --archive ~/scratch/wildfire/archive --out ~/scratch/wildfire/live_catalog
Writes OUT/fires.csv and OUT/perimeters.parquet (append-safe: existing fire ids are replaced).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DROP_METHODS = ("hand sketch", "modeled", "estimated")


def load_archive(arch):
    import geopandas as gpd
    from shapely.geometry import shape
    idx = pd.read_csv(Path(arch) / "perimeters" / "index.csv")
    rows = []
    for r in idx.itertuples():
        try:
            fc = json.loads((Path(arch) / "perimeters" / r.file).read_text())
            f = fc["features"][0]
            g = shape(f["geometry"]) if f.get("geometry") else None
        except (OSError, ValueError, KeyError, IndexError):
            continue
        if g is None or g.is_empty:
            continue
        p = f.get("properties", {})
        rows.append(dict(incident=str(r.fire_id), incident_name=r.name, category=str(r.category or ""),
                         map_method=str(r.map_method or ""), perim_time=pd.Timestamp(r.polygon_time),
                         type=p.get("attr_IncidentTypeCategory") or ("RX" if "prescribed" in str(r.category).lower() else "WF"),
                         irwin_id=str(r.irwin_id or "").strip("{}"), source="wfigs_archive", geometry=g.buffer(0)))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326) if rows else gpd.GeoDataFrame(columns=["geometry"], crs=4326)


def clean(gdf, min_days, min_acres, settle_days, now):
    out_fires, out_per, why = [], [], {"prescribed": 0, "method": 0, "short": 0, "small": 0, "still_growing": 0}
    if gdf.empty:
        return out_fires, out_per, why
    gdf = gdf[gdf["type"] != "RX"]
    keep = ~gdf["map_method"].str.lower().str.contains("|".join(DROP_METHODS), na=False)
    why["method"] = int((~keep).sum())
    gdf = gdf[keep].copy()
    gdf["acres_calc"] = gdf.to_crs(5070).area / 4046.856
    b = gdf.geometry.bounds
    lon = (b.minx + b.maxx) / 2
    gdf["burn_day"] = (gdf["perim_time"].dt.tz_convert(None) + pd.to_timedelta(lon / 15, unit="h")).dt.date.astype(str)
    for inc, grp in gdf.groupby("incident"):
        grp = grp.sort_values("perim_time").groupby("burn_day", as_index=False).tail(1)
        peak, keep = 0.0, []
        for a in grp["acres_calc"]:
            keep.append(a >= 0.9 * peak)
            peak = max(peak, a)
        grp = grp[np.array(keep)]
        if grp["burn_day"].nunique() < min_days:
            why["short"] += 1
            continue
        if grp["acres_calc"].max() < min_acres:
            why["small"] += 1
            continue
        last = grp["perim_time"].iloc[-1]
        if (now - last).days < settle_days and grp["acres_calc"].iloc[-1] > 1.02 * grp["acres_calc"].iloc[-2]:
            why["still_growing"] += 1
            continue
        fid = "L" + hashlib.sha1(inc.encode()).hexdigest()[:10]
        bb = grp.total_bounds
        gaps = grp["perim_time"].diff().dt.total_seconds().dropna() / 3600
        out_fires.append(dict(fire_id=fid, incident=inc, name=grp["incident_name"].iloc[-1], year=last.year,
                              first=grp["perim_time"].iloc[0], last=last, days=grp["burn_day"].nunique(), perimeters=len(grp),
                              final_acres=float(grp["acres_calc"].max()), first_acres=float(grp["acres_calc"].iloc[0]),
                              median_gap_h=float(gaps.median()) if len(gaps) else np.nan,
                              west=bb[0], south=bb[1], east=bb[2], north=bb[3], sources="wfigs_archive",
                              methods=",".join(sorted(grp["map_method"].str.lower().unique()))))
        out_per.append(grp[["incident", "incident_name", "source", "category", "map_method", "perim_time", "burn_day",
                            "acres_calc", "irwin_id", "geometry"]].assign(fire_id=fid))
    return out_fires, out_per, why


def main():
    import geopandas as gpd
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="checkout of the perimeter-archive branch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-days", type=int, default=3)
    ap.add_argument("--min-acres", type=float, default=300)
    ap.add_argument("--settle-days", type=int, default=3, help="skip fires that grew in the last N days")
    a = ap.parse_args()
    now = pd.Timestamp.now(tz="UTC")
    gdf = load_archive(a.archive)
    print(f"archive: {len(gdf)} perimeter versions, {gdf['incident'].nunique() if len(gdf) else 0} fires")
    fires, per, why = clean(gdf, a.min_days, a.min_acres, a.settle_days, now)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f_new = pd.DataFrame(fires)
    if (out / "fires.csv").exists() and len(f_new):
        old = pd.read_csv(out / "fires.csv")
        f_new = pd.concat([old[~old.fire_id.isin(f_new.fire_id)], f_new], ignore_index=True)
    if per:
        p_new = gpd.GeoDataFrame(pd.concat(per, ignore_index=True), geometry="geometry", crs=4326)
        if (out / "perimeters.parquet").exists():
            old = gpd.read_parquet(out / "perimeters.parquet")
            p_new = pd.concat([old[~old.fire_id.isin(p_new.fire_id)], p_new], ignore_index=True)
        p_new.to_parquet(out / "perimeters.parquet", index=False)
    if len(f_new):
        f_new.to_csv(out / "fires.csv", index=False)
    print(f"kept {len(fires)} fires this run; left out: {why}; catalog now {len(f_new)} fires")


if __name__ == "__main__":
    main()
