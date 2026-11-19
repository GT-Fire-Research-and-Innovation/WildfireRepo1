"""
NIFC suppression archive layers and ICS-209-PLUS incident reports for the raw grab (raw_channels.py:
suppression, ics209).

  discover_archive_layers  {year: [(layer_url, 'line' | 'polygon')]}: each year's Operational Data Archive
                           EventLine / EventPolygon services (2018+)
  load_ics209              download (once) and read the ICS-209-PLUS wildfire tables (St. Denis et al.)
  match_ics                ICS-209 incident ids for a fire: by IRWIN id, else by name, year and location

Each archived suppression version keeps GDB_FROM_DATE / GDB_TO_DATE, CreateDate and DateCurrent. A model may
use a version only from the burn day after the latest of the three (docs/RAW_DATA_CARD.md).
"""
import re
import warnings
import zipfile

import numpy as np
import pandas as pd

import nifc_perimeters as NP

warnings.filterwarnings("ignore", message="Error parsing datetimes")  # ICS-209 dates come in mixed formats

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
# Suppression archive layers
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


# ---------------------------------------------------------------------------
# ICS-209-PLUS incident reports
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
