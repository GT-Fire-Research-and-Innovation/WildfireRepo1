"""
Daily suppression layers for every fire, on the same 75 m Tessera grid as the
NIFC perimeter labels. Run nifc_perimeters.py first: this script reuses its
output folder (fires.json, labels/, perimeters/, cache/).

Two kinds of suppression data:

A. Spatial, from the NIFC Operational Data Archive (2018 onward only).
   EventLine and EventPolygon layers record fire lines, burnouts and drops.
   The archive keeps every version of every feature with the dates it existed
   (GDB_FROM_DATE / GDB_TO_DATE), so for each day we rebuild the map exactly
   as it stood at the end of that burn day (06:00 local the next morning).
   Channels (CHANNELS below):
     hand_line, dozer_line, other_line, burnout, retardant, water_foam,
     uncontrolled_edge

B. Daily resources, from ICS-209-PLUS (St. Denis et al., 1999-2020).
   Per-day personnel, aircraft, % contained, etc. for the matched incidents,
   using the latest report whose period ended on or before that day.

Outputs in OUT_DIR/suppression/<fire_id>.npz:
  dates        (T,)            calendar days, 'YYYY-MM-DD'
  layers       (T, C, H, W)    uint8 0/1 rasters, C = len(channels)
  channels     (C,)            channel names
  ics_values   (T, F) float32  ICS-209 values per day (NaN = no report yet)
  ics_fields   (F,)            their names
  ics_incidents                ICS-209 incident ids used
  has_spatial                  False when the archive had no lines (e.g. 2017)
Plus OUT_DIR/suppression_summary.csv.

To avoid leaking the answer, use day t's suppression to predict growth from
day t to t+1, never day t+1's.

Run in Colab (after nifc_perimeters.py, with both files uploaded):
  !python suppression_layers.py --out /content/drive/MyDrive/nifc_perimeters
"""

import argparse
import json
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import nifc_perimeters as NP

warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Error parsing datetimes")  # 9999-12-31 = current version

# The archive's category names changed in 2021 (new NWCG schema). Both spellings are
# listed so 2018-2020 and 2021+ fires fill the same channels. Counts checked against the
# 2018, 2019, 2021 and 2022 archives.
CHANNELS = {
    "hand_line": {"Completed Hand Line", "Hand Line"},
    "dozer_line": {"Completed Dozer Line", "Completed Plow Line"},
    "other_line": {"Completed Line", "Line Break Completed", "Explosive Line", "Road as Completed Line",
                   # 2021+ names
                   "Completed Road as Line", "Completed Mixed Construction Line", "Completed Fuel Break",
                   "Contained Line"},
    "burnout": {"Completed Burnout", "Active Burnout"},
    "retardant": {"Aerial Retardant Drop", "Air Tanker Retardant", "Retardant Drop"},
    "water_foam": {"Aerial Water Drop", "Helitanker Water", "Aerial Foam Drop", "Air Tanker Foam", "Helitanker Foam",
                   "Water Drop", "Foam Drop"},
    # 'Fire Edge' replaced 'Uncontrolled Fire Edge' in 2021; 'Fire Edge (Field Collection)' is the 2022+ form.
    "uncontrolled_edge": {"Uncontrolled Fire Edge", "Fire Edge", "Fire Edge (Field Collection)"},
}
CHANNEL_NAMES = list(CHANNELS)
CAT_TO_CH = {c: i for i, ch in enumerate(CHANNEL_NAMES) for c in CHANNELS[ch]}
FAR_FUTURE = pd.Timestamp("2262-01-01", tz="UTC")

ICS_FIGSHARE_ARTICLE = 19858927
# Candidate column names (first match wins); the script prints what it picked.
ICS_SITREP_COLS = {
    "incident_id": ["INCIDENT_ID"],
    "report_to": ["REPORT_TO_DATE", "REPORT_DATE", "REPORT_TO"],
    "report_from": ["REPORT_FROM_DATE", "REPORT_FROM"],
}
ICS_VALUE_COLS = {
    "total_personnel": ["TOTAL_PERSONNEL", "TOTAL_PERSONNEL_SUM"],
    "total_aerial": ["TOTAL_AERIAL", "TOTAL_AERIAL_SUM"],
    "pct_contained": ["PCT_CONTAINED_COMPLETED", "PERCENT_CONTAINED", "PCT_CONTAINED"],
    "acres": ["ACRES", "CURR_INCIDENT_AREA", "INC_AREA"],
    "est_cost_to_date": ["EST_IM_COST_TO_DATE", "EST_COST_TO_DATE", "COST_TO_DATE"],
    "fatalities": ["FATALITIES"],
    "structures_destroyed": ["STR_DESTROYED", "STR_DESTROYED_TOTAL"],
}
ICS_INCIDENT_COLS = {
    "incident_id": ["INCIDENT_ID"],
    "name": ["INCIDENT_NAME", "INCIDENT_NAME_CORRECTED"],
    "lat": ["POO_LATITUDE", "LATITUDE", "POO_LAT"],
    "lon": ["POO_LONGITUDE", "LONGITUDE", "POO_LON", "POO_LONG"],
    "start": ["DISCOVERY_DATE", "START_DATE", "DISC_DATE"],
    "year": ["START_YEAR", "FIRE_YEAR", "YEAR"],
    "irwin": ["IRWIN_IDENTIFIER", "IRWIN_ID", "IRWINID", "IRWIN_IDS"],
}


# ---------------------------------------------------------------------------
# A. Spatial layers from the archive
# ---------------------------------------------------------------------------

def discover_archive_layers(sess, years):
    """{year: [(layer_url, 'line'|'polygon')]} for Operational Data Archive services."""
    js = NP.get_json(sess, NP.NIFC_ROOT)
    names = {s["name"] for s in js.get("services", []) if s.get("type") == "FeatureServer"}
    out = {}
    for y in sorted(years):
        found = []
        for n in sorted(n for n in names if str(y) in n and re.search(
                r"(National_Incident_Feature_Service|NIFS_OpenData|NIFS_Archive|Operational_Data_Archive)", n, re.I)):
            fs = NP.get_json(sess, f"{NP.NIFC_ROOT}/{n}/FeatureServer")
            for lyr in fs.get("layers", []):
                if NP.layer_is(lyr["name"], "eventline"):
                    found.append((f"{NP.NIFC_ROOT}/{n}/FeatureServer/{lyr['id']}", "line"))
                elif NP.layer_is(lyr["name"], "eventpolygon"):
                    found.append((f"{NP.NIFC_ROOT}/{n}/FeatureServer/{lyr['id']}", "polygon"))
        if not found and y in NP.ARCHIVE_FGDB_ITEMS:
            for u in NP.archive_fgdb_layers(y, ("eventline", "eventpolygon")):
                found.append((u, "line" if "line" in NP.layer_label(u).lower() else "polygon"))
        out[y] = found
        print(f"  {y}: " + (", ".join(NP.layer_label(u) for u, _ in found) or "no archive (no spatial suppression)"))
    return out


def archive_features(feats):
    """GeoJSON archive features -> GeoDataFrame of suppression features with validity times."""
    import geopandas as gpd
    from shapely.geometry import shape
    rows = []
    for f in feats:
        if not f.get("geometry"):
            continue
        p = {k.lower(): v for k, v in (f.get("properties") or {}).items()}
        cat = p.get("featurecategory")
        if cat not in CAT_TO_CH:
            continue
        if str(p.get("deletethis") or "").strip().lower() in {"yes", "y", "true", "1"}:
            continue
        t_from = NP._ms_to_dt(p.get("gdb_from_date"))
        t_to = NP._ms_to_dt(p.get("gdb_to_date"))
        if pd.isna(t_from):
            t_from = NP._ms_to_dt(p.get("createdate"))
        if pd.isna(t_from):
            continue
        if re.search(NP.TEST_NAME_RE, p.get("incidentname") or "", re.I):
            continue  # placeholder / practice incident
        if pd.isna(t_to) or t_to.year >= 9000:
            t_to = FAR_FUTURE
        rows.append({
            "category": cat, "channel": CAT_TO_CH[cat],
            "incident_name": (p.get("incidentname") or "").strip(),
            "irwin_id": NP._norm_id(p.get("irwinid")),
            "t_from": t_from, "t_to": t_to,
            "geometry": shape(f["geometry"]),
        })
    if not rows:
        return gpd.GeoDataFrame(columns=["category", "channel", "t_from", "t_to", "geometry"],
                                geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def snapshot_time(day, lon):
    """UTC instant at the end of burn day `day`: 06:00 local solar time the next morning."""
    local = pd.Timestamp(day) + pd.Timedelta(days=1, hours=NP.BURN_DAY_SHIFT_H)
    return (local - pd.Timedelta(hours=lon / 15.0)).tz_localize("UTC")


def rasterize_layers(feats, fire, dates, incidents=None):
    from affine import Affine
    from rasterio.features import rasterize
    H, W = fire["height"], fire["width"]
    tr = Affine(*fire["transform"])
    lon = float(np.mean(fire["bounds_lonlat"][0::2]))
    layers = np.zeros((len(dates), len(CHANNEL_NAMES), H, W), dtype=np.uint8)
    if feats.empty:
        return layers
    if incidents:
        ok = feats["irwin_id"].isin(incidents["irwin"]) | feats["incident_name"].str.lower().isin(incidents["names"])
        feats = feats[ok]
    for t, d in enumerate(dates):
        snap = snapshot_time(d, lon)
        live = feats[(feats["t_from"] <= snap) & (feats["t_to"] > snap)]
        for ch, grp in live.groupby("channel"):
            layers[t, ch] = rasterize([(g, 1) for g in grp.geometry], out_shape=(H, W), transform=tr,
                                      fill=0, all_touched=True, dtype="uint8")
    return layers


# ---------------------------------------------------------------------------
# B. ICS-209-PLUS daily resources
# ---------------------------------------------------------------------------

def _pick(cols, cands, required=True, table=""):
    up = {c.upper(): c for c in cols}
    for c in cands:
        if c.upper() in up:
            return up[c.upper()]
    if required:
        raise KeyError(f"None of {cands} in {table} columns: {sorted(cols)}")
    return None


def load_ics209(out_dir):
    """Download (once) and load the ICS-209-PLUS wildfire sitrep + incident tables."""
    d = out_dir / "ics209"
    d.mkdir(exist_ok=True)
    zpath = d / "ics209-plus-wildfire.zip"
    if not zpath.exists():
        sess = NP._session()
        art = NP.get_json(sess, f"https://api.figshare.com/v2/articles/{ICS_FIGSHARE_ARTICLE}", {"f": None})
        files = {f["name"]: f["download_url"] for f in art["files"]}
        print("  ICS-209-PLUS files:", list(files))
        name = next(n for n in files if "wildfire" in n.lower() and n.endswith(".zip"))
        print(f"  downloading {name} ...")
        with sess.get(files[name], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zpath, "wb") as fh:
                for chunk in r.iter_content(2 ** 20):
                    fh.write(chunk)
    z = zipfile.ZipFile(zpath)
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    print("  CSVs in zip:", csvs)
    usable = [n for n in csvs if not re.search(r"(delet|complex|tract|county|cbg|block|link)", n.lower())]
    sit = next(n for n in usable if "sitrep" in n.lower())
    # The incident summary table is the one with INCIDENT_ID plus an incident name column.
    name_cols = {c.upper() for c in ICS_INCIDENT_COLS["name"]}
    inc = None
    for n in sorted(usable, key=lambda n: ("incident" not in n.lower(), n)):
        if n == sit:
            continue
        cols = {c.upper() for c in pd.read_csv(z.open(n), nrows=0).columns}
        if "INCIDENT_ID" in cols and cols & name_cols:
            inc = n
            break
    if inc is None:
        raise RuntimeError(f"No incident summary table with INCIDENT_ID and a name column among {csvs}")
    print(f"  ICS-209 tables: {sit} | {inc}")
    sit_df = pd.read_csv(z.open(sit), low_memory=False)
    inc_df = pd.read_csv(z.open(inc), low_memory=False)

    sc = {k: _pick(sit_df.columns, v, k != "report_from", "sitrep") for k, v in ICS_SITREP_COLS.items()}
    vc = {k: _pick(sit_df.columns, v, False) for k, v in ICS_VALUE_COLS.items()}
    vc = {k: v for k, v in vc.items() if v}
    ic = {k: _pick(inc_df.columns, v, k in {"incident_id", "name"}, "incident") for k, v in ICS_INCIDENT_COLS.items()}
    print("  sitrep columns used:", {**sc, **vc})
    print("  incident columns used:", ic)

    sit_out = pd.DataFrame({"incident_id": sit_df[sc["incident_id"]].astype(str),
                            "report_to": pd.to_datetime(sit_df[sc["report_to"]], errors="coerce", format="mixed").dt.tz_localize(None)})
    for k, c in vc.items():
        sit_out[k] = pd.to_numeric(sit_df[c], errors="coerce")
    sit_out = sit_out.dropna(subset=["report_to"])

    inc_out = pd.DataFrame({"incident_id": inc_df[ic["incident_id"]].astype(str),
                            "name": inc_df[ic["name"]].astype(str)})
    for k in ["lat", "lon"]:
        inc_out[k] = pd.to_numeric(inc_df[ic[k]], errors="coerce") if ic[k] else np.nan
    inc_out["start"] = pd.to_datetime(inc_df[ic["start"]], errors="coerce", format="mixed") if ic["start"] else pd.NaT
    inc_out["year"] = pd.to_numeric(inc_df[ic["year"]], errors="coerce") if ic["year"] else inc_out["start"].dt.year
    inc_out["irwin"] = inc_df[ic["irwin"]].astype(str).str.upper().str.strip("{} ") if ic["irwin"] else ""
    return sit_out, inc_out, list(vc)


def _norm_name(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    s = re.sub(r"\b(fire|complex|wildfire|incident)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match_ics(inc_df, incidents, fire, start, end):
    """ICS-209 incident ids for this fire: by IRWIN id, else by name + year + location."""
    l, b, r, t = fire["bounds_lonlat"]
    pad = 0.25
    year_ok = inc_df["year"].between(start.year - 1, end.year)
    in_box = inc_df["lat"].between(b - pad, t + pad) & inc_df["lon"].between(l - pad, r + pad)
    ids = set()
    if incidents["irwin"]:
        ids |= set(inc_df.loc[inc_df["irwin"].apply(
            lambda v: any(i in v for i in incidents["irwin"]) if v else False), "incident_id"])
    if not ids:
        names = {_norm_name(n) for n in incidents["names"]} - {""}
        cand = inc_df[year_ok & inc_df["name"].map(_norm_name).isin(names)]
        if inc_df["lat"].notna().any():
            cand = cand[in_box[cand.index] | cand["lat"].isna()]
        ids |= set(cand["incident_id"])
    return sorted(ids)


def daily_ics(sit_df, ids, dates, fields, primary_ids):
    """(T, F) values on each day from the latest report ending on or before that day."""
    vals = np.full((len(dates), len(fields) + 1), np.nan, dtype=np.float32)
    if not ids:
        return vals
    s = sit_df[sit_df["incident_id"].isin(ids)].sort_values("report_to")
    days = pd.to_datetime(dates)
    per_inc = []
    for iid, grp in s.groupby("incident_id"):
        g = grp.groupby(grp["report_to"].dt.normalize()).last()
        g = g.reindex(g.index.union(days)).sort_index().ffill().reindex(days)
        per_inc.append((iid, g))
    if not per_inc:
        print(f"    note: ICS-209 incidents {ids} matched but have no sitreps")
        return vals
    for j, f in enumerate(fields):
        if f == "pct_contained":
            src = [g for iid, g in per_inc if iid in primary_ids] or [g for _, g in per_inc]
            vals[:, j] = np.nanmax(np.stack([g[f].to_numpy(float) for g in src]), axis=0)
        else:
            stack = np.stack([g[f].to_numpy(float) for _, g in per_inc])
            vals[:, j] = np.where(np.isnan(stack).all(0), np.nan, np.nansum(stack, axis=0))
    # days since the most recent report (any matched incident)
    rep_days = s["report_to"].dt.normalize().drop_duplicates().sort_values().to_numpy()
    for t, d in enumerate(days):
        prior = rep_days[rep_days <= np.datetime64(d)]
        vals[t, -1] = (np.datetime64(d) - prior[-1]).astype("timedelta64[D]").astype(int) if len(prior) else np.nan
    return vals


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def fire_dates(fire, label_dates):
    start, _ = NP.fire_window(fire)
    start = start + pd.Timedelta(days=NP.PAD_BEFORE_DAYS)  # back to the first TS-SatFire day
    end = pd.Timestamp(fire["last_day"]) if fire["last_day"] else start
    if label_dates:
        # Never reach outside the fire's search window, even if a label file has stray dates.
        w_start, w_end = NP.fire_window(fire)
        start = max(min(start, pd.Timestamp(label_dates[0])), w_start)
        end = min(max(end, pd.Timestamp(label_dates[-1])), w_end)
    return start, end, [str(d.date()) for d in pd.date_range(start, end)]


def fire_incidents(out_dir, fid):
    p = out_dir / "perimeters" / f"{fid}.gpkg"
    if not p.exists():
        return {"irwin": set(), "names": set()}, set()
    import geopandas as gpd
    g = gpd.read_file(p)
    if "main" in g:  # only the labelled fire, not separate fires sharing the box
        g = g[g["main"].astype(int) == 1]
    lab = np.load(out_dir / "labels" / f"{fid}.npz")
    prim = str(lab["primary_irwin_id"])
    return ({"irwin": {i for i in g["irwin_id"].dropna() if i},
             "names": {n.lower() for n in g["incident_name"].dropna() if n}},
            {prim} if prim else set())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nifc_out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fires", nargs="*")
    ap.add_argument("--skip-ics", action="store_true", help="spatial layers only")
    ap.add_argument("--years", nargs="*", type=int, help="only fires from these years")
    ap.add_argument("--redo", action="store_true", help="rebuild fires that are already done")
    args = ap.parse_args()
    out = Path(args.out)
    NP.FGDB_DIR = out / "fgdb"
    (out / "suppression").mkdir(parents=True, exist_ok=True)
    fires = json.loads((out / "fires.json").read_text())
    ids = sorted(fires)
    if args.fires:
        ids = [i for i in ids if i in set(args.fires)]
    if args.years:
        ids = [i for i in ids if fires[i]["fire_year"] in set(args.years)]
    if args.limit:
        ids = ids[: args.limit]
    print(f"{len(ids)} fires")

    import geopandas as gpd
    years = set()
    for fid in ids:
        s, e = NP.fire_window(fires[fid])
        years |= set(range(s.year, e.year + 1))
    sess = NP._session()
    print("Archive layers:")
    layers_by_year = discover_archive_layers(sess, years)

    if args.skip_ics:
        sit_df = inc_df = None
        fields = []
    else:
        print("Loading ICS-209-PLUS:")
        sit_df, inc_df, fields = load_ics209(out)
    ics_fields = fields + ["days_since_report"]

    rows = []
    for k, fid in enumerate(ids):
        fire = fires[fid]
        target = out / "suppression" / f"{fid}.npz"
        if target.exists() and not args.redo:
            print(f"[{k + 1}/{len(ids)}] {fid}: already done")
            continue
        lab_p = out / "labels" / f"{fid}.npz"
        label_dates = [str(d) for d in np.load(lab_p)["dates"]] if lab_p.exists() else []
        start, end, dates = fire_dates(fire, label_dates)
        incidents, primary_irwin = fire_incidents(out, fid)
        print(f"[{k + 1}/{len(ids)}] {fid}: {dates[0]} to {dates[-1]}, incidents {sorted(incidents['names'])}")

        parts = []
        for y in range(start.year, end.year + 1):
            for url, _ in layers_by_year.get(y, []):
                parts.append(archive_features(NP.query_layer(sess, url, fire["bounds_lonlat"], out / "cache")))
        parts = [p for p in parts if not p.empty]
        feats = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326) if parts \
            else archive_features([])
        has_spatial = bool(layers_by_year.get(start.year))
        layers = rasterize_layers(feats, fire, dates, incidents if incidents["names"] else None)

        ics_ids, prim_ids = [], set()
        vals = np.full((len(dates), len(ics_fields)), np.nan, dtype=np.float32)
        if inc_df is not None and incidents["names"]:
            ics_ids = match_ics(inc_df, incidents, fire, start, end)
            prim_ids = set(match_ics(inc_df, {"irwin": primary_irwin, "names": set()}, fire, start, end)) \
                if primary_irwin else set()
            vals = daily_ics(sit_df, ics_ids, dates, fields, prim_ids or set(ics_ids))

        np.savez_compressed(
            target, dates=np.array(dates), layers=layers, channels=np.array(CHANNEL_NAMES),
            ics_values=vals, ics_fields=np.array(ics_fields), ics_incidents=np.array(ics_ids),
            has_spatial=np.bool_(has_spatial), transform=np.array(fire["transform"]), fire_id=np.array(fid))
        px = layers.reshape(len(dates), len(CHANNEL_NAMES), -1).sum(-1)
        row = {"fire_id": fid, "fire_year": fire["fire_year"], "days": len(dates), "has_spatial": has_spatial,
               "archive_features": len(feats), "ics_incidents": ";".join(ics_ids),
               "ics_days_with_report": int(np.isfinite(vals[:, 0]).sum()) if fields else 0}
        for c, name in enumerate(CHANNEL_NAMES):
            row[f"{name}_days"] = int((px[:, c] > 0).sum())
        if "total_personnel" in fields:
            row["max_personnel"] = float(np.nanmax(vals[:, fields.index("total_personnel")])) \
                if np.isfinite(vals[:, fields.index("total_personnel")]).any() else None
        print("    " + ", ".join(f"{n}:{row[n + '_days']}d" for n in CHANNEL_NAMES)
              + f" | ICS-209 incidents {ics_ids or 'none'}")
        rows.append(row)
        NP.write_summary(out / "suppression_summary.csv", rows)
    print("Done:", out / "suppression_summary.csv")


if __name__ == "__main__":
    sys.exit(main())
