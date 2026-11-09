"""
NIFC fire perimeters for every fire in the ts-tesserafire-5 dataset.

For each Tessera fire file (tessera/p<fire_id>.npz) this script:
  1. reads the fire's grid (bounds, transform, size) and dates from the .npz
     without downloading the 365 MB embedding,
  2. queries the NIFC perimeter archives for that box:
       - Historic GeoMAC Perimeters (one service per year, 2000-2019)
       - Operational Data Archive / NIFS EventPolygon layers (2018 onward)
  3. cleans the perimeters (drops hand sketches, modeled, test edits,
     deleted features, shrinking re-maps, duplicate archive versions),
  4. keeps one perimeter per incident per burn day,
  5. keeps the main fire (the incident covering most of the box plus incidents
     touching it); other fires in the box go to other_fire_mask,
  6. rasterizes cumulative burned area onto the exact Tessera 75 m grid.

Outputs in OUT_DIR:
  perimeters/<fire_id>.gpkg   cleaned perimeter polygons for that fire
  labels/<fire_id>.npz        arrival_day (H, W) int16, dates, per-day stats,
                              transform/crs identical to the Tessera file,
                              other_fire_mask (H, W) bool: separate fires in the
                              box (mask these out of the loss)
  quicklook/<fire_id>.png     arrival-day map for eyeballing
  summary.csv                 one row per fire with match and QA stats
  cache/                      raw NIFC query results (lets reruns resume)

Burned mask for day t:  (arrival_day >= 0) & (arrival_day <= t)
At 375 m (VIIRS grid):  burned_fraction_375(arrival_day, t, cells_per_pixel)

Run in Colab:
  !pip -q install zipfile-zstd geopandas rasterio huggingface_hub
  !python nifc_perimeters.py --out /content/drive/MyDrive/nifc_perimeters
"""

import argparse
import io
import json
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import zipfile_zstd  # noqa: F401  (the Tessera .npz files use Zstandard)
except ImportError:
    pass

HF_REPO = "SamuelWu318/ts-tesserafire-5"
HF_TESSERA_DIR = f"datasets/{HF_REPO}/tessera"
NIFC_ROOT = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
# Years whose Operational Data Archive is published only as a File Geodatabase
# download (no feature service): ArcGIS Online item ids, owner NIFC_Authoritative.
ARCHIVE_FGDB_ITEMS = {2020: "ea843f7f091f4c7f9743798b64c864be", 2021: "af727c41d79643b091cee372233110d4"}
ARCGIS_ITEM_DATA = "https://www.arcgis.com/sharing/rest/content/items/{}/data"
FGDB_DIR = Path("nifc_out") / "fgdb"  # set from --out in main()

# Perimeter categories kept from the Operational Data Archive EventPolygon layers.
NIFS_CATEGORIES = {"Wildfire Daily Fire Perimeter", "Wildfire Final Fire Perimeter"}
# Map methods dropped by default (low positional accuracy).
DROP_MAP_METHODS = {"hand sketch", "modeled"}
# Ranking used when an incident has several perimeters on the same burn day
# at the same time: lower is better.
MAP_METHOD_RANK = {
    "infrared image": 0, "gps-flight": 1, "gps-walked": 1, "gps-driven": 1,
    "gps-walked/driven": 1, "image interpretation": 2, "mixed methods": 3,
    "digitized-topo": 4, "digitized-other": 4, "other": 5, "unknown": 6,
}
SHRINK_TOL = 0.8        # drop a perimeter smaller than 80% of the incident's prior max
MIN_ACRES = 1.0         # drop slivers / test edits
PAD_BEFORE_DAYS = 10    # search window around the fire's TS-SatFire dates
PAD_AFTER_DAYS = 30
BURN_DAY_SHIFT_H = 6    # perimeters drawn before 06:00 local count for the previous day
MAIN_TOUCH_M = 500      # an incident joins the main fire if its footprint comes within this of it
# Placeholder / practice incidents seen in the archives (e.g. an incident named "test").
TEST_NAME_RE = r"^\s*(?:test|testing|exercise|training|practice|delete|dummy)\b"
ALL_INCIDENTS = False   # --all-incidents: label every incident in the box (old behaviour)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "tesserafire-nifc-perimeters/1.0"
    return s


def get_json(sess, url, params=None, tries=5):
    params = dict(params or {})
    params.setdefault("f", "json")
    for i in range(tries):
        try:
            r = sess.get(url, params=params, timeout=120)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict) and "error" in js:
                raise RuntimeError(js["error"])
            return js
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            wait = 2 ** (i + 1)
            print(f"    retry {i + 1} after error: {e} (waiting {wait}s)")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# 1. Fire list from the Tessera files
# ---------------------------------------------------------------------------

def _read_member(z, name):
    with z.open(name) as m:
        return np.lib.format.read_array(io.BytesIO(m.read()), allow_pickle=False)


def _member_shape(z, name):
    with z.open(name) as m:
        version = np.lib.format.read_magic(m)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(m)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(m)
    return shape


def _date_from_path(p):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(p))
    return m.group(1) if m else None


def read_fire_meta_from_zip(f, src):
    z = zipfile.ZipFile(f)
    names = {n[:-4] for n in z.namelist() if n.endswith(".npy")}
    meta = {k: _read_member(z, f"{k}.npy") for k in
            ["fire_id", "fire_year", "transform", "bounds_lonlat", "crs",
             "day_files", "cells_per_pixel"] if k in names}
    shape = _member_shape(z, "coverage.npy")
    days = sorted(d for d in (_date_from_path(p) for p in np.atleast_1d(meta.get("day_files", []))) if d)
    return {
        "fire_id": str(meta["fire_id"]),
        "fire_year": int(meta["fire_year"]),
        "transform": [float(x) for x in meta["transform"]],
        "bounds_lonlat": [float(x) for x in meta["bounds_lonlat"]],
        "crs": str(meta.get("crs", "EPSG:4326")),
        "height": int(shape[-2]),
        "width": int(shape[-1]),
        "cells_per_pixel": int(meta.get("cells_per_pixel", 5)),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "source": src,
    }


def list_fires(out_dir, limit=None, only=None):
    """Read metadata for every Tessera file on Hugging Face (cached to fires.json)."""
    cache = out_dir / "fires.json"
    fires = json.loads(cache.read_text()) if cache.exists() else {}
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    paths = sorted(p for p in fs.ls(HF_TESSERA_DIR, detail=False) if p.endswith(".npz"))
    print(f"Found {len(paths)} Tessera fire files on Hugging Face")
    if only:
        paths = [p for p in paths if Path(p).stem.lstrip("p") in only]
    if limit:
        paths = paths[:limit]
    for i, p in enumerate(paths):
        fid = Path(p).stem.lstrip("p")
        if fid in fires:
            continue
        print(f"  [{i + 1}/{len(paths)}] reading metadata for {fid}")
        with fs.open(p, "rb", block_size=2 ** 20) as f:
            fires[fid] = read_fire_meta_from_zip(f, p)
        cache.write_text(json.dumps(fires, indent=1))
    return [fires[Path(p).stem.lstrip("p")] for p in paths]


# ---------------------------------------------------------------------------
# 2. NIFC sources and queries
# ---------------------------------------------------------------------------

def layer_is(name, *kinds):
    """True when an archive layer name is one of `kinds`, however it is spelled.

    The archive names the same layer differently by year: EventPolygon (2020 gdb),
    EventPolygon2018 / EventLine2019 (services) and Event_Polygon_2021 (2021 gdb).
    Comparing letters only makes all of these match 'eventpolygon' / 'eventline'.
    """
    letters = re.sub(r"[^a-z]", "", name.lower())
    return any(letters.startswith(k) for k in kinds) and "attach" not in letters


def discover_sources(sess, years):
    """Return {year: [(kind, layer_url), ...]} for GeoMAC and NIFS archive layers."""
    js = get_json(sess, NIFC_ROOT)
    names = {s["name"] for s in js.get("services", []) if s.get("type") == "FeatureServer"}
    sources = {}
    for y in sorted(years):
        found = []
        # the 2019 service is spelled "GeoMAC", earlier years "Geomac"
        g = next((n for n in names if n.lower() == f"historic_geomac_perimeters_{y}"), None)
        if g:
            found.append(("geomac", f"{NIFC_ROOT}/{g}/FeatureServer/0"))
        nifs_names = [n for n in names if str(y) in n and re.search(
            r"(National_Incident_Feature_Service|NIFS_OpenData|NIFS_Archive|Operational_Data_Archive)", n, re.I)]
        for n in sorted(nifs_names):
            try:
                fs = get_json(sess, f"{NIFC_ROOT}/{n}/FeatureServer")
            except Exception as e:  # noqa: BLE001
                print(f"  could not open {n}: {e}")
                continue
            for lyr in fs.get("layers", []):
                if layer_is(lyr["name"], "eventpolygon"):
                    found.append(("nifs", f"{NIFC_ROOT}/{n}/FeatureServer/{lyr['id']}"))
        if not any(k == "nifs" for k, _ in found) and y in ARCHIVE_FGDB_ITEMS:
            found += [("nifs", u) for u in archive_fgdb_layers(y, ("eventpolygon",))]
        if not any(k == "nifs" for k, _ in found) and y >= 2018:
            hints = sorted(n for n in names if str(y) in n and re.search(r"(nifs|incident|archive|event)", n, re.I))
            print(f"  WARNING: no Operational Data Archive layer found for {y}. Candidates: {hints}")
        sources[y] = found
        print(f"  {y}: " + (", ".join(layer_label(u) for _, u in found) or "nothing found"))
    return sources


def layer_label(url):
    if url.startswith("fgdb://"):
        gdb, layer = url[len("fgdb://"):].split("|")
        return f"{Path(gdb).name}/{layer}"
    return url.split("/services/")[1]


def archive_fgdb_layers(year, prefixes):
    """Download (once) the year's archive File Geodatabase; return fgdb:// urls of matching layers."""
    import pyogrio
    d = FGDB_DIR / str(year)
    d.mkdir(parents=True, exist_ok=True)
    gdbs = sorted(d.rglob("*.gdb"))
    if not gdbs:
        zpath = d / "archive.zip"
        if not zpath.exists():
            url = ARCGIS_ITEM_DATA.format(ARCHIVE_FGDB_ITEMS[year])
            print(f"  downloading {year} Operational Data Archive geodatabase (one time, can be large) ...")
            sess = _session()
            with sess.get(url, stream=True, timeout=1800) as r:
                r.raise_for_status()
                with open(zpath.with_suffix(".part"), "wb") as fh:
                    for chunk in r.iter_content(2 ** 22):
                        fh.write(chunk)
            zpath.with_suffix(".part").rename(zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(d)
        gdbs = sorted(d.rglob("*.gdb"))
    if not gdbs:
        print(f"  WARNING: no .gdb found in the {year} archive download")
        return []
    out = []
    for gdb in gdbs:
        layers = [row[0] for row in pyogrio.list_layers(gdb)]
        print(f"  {year} geodatabase {gdb.name} layers: {layers}")
        for name in layers:
            if layer_is(name, *prefixes):
                out.append(f"fgdb://{gdb}|{name}")
    return out


FGDB_CHUNK = 20000  # features read per pass when caching a geodatabase layer


def _fgdb_cache(gdb, layer):
    """Read a geodatabase layer once, in chunks, into lon/lat Parquet parts with a box per feature.

    Needed because bounding-box reads do not work on the 2021 archive geodatabase (its spatial
    index returns nothing, even for a box drawn around a known fire), and reading the whole
    layer at once needs 4+ GB of memory. The cache sits next to the .gdb on Drive, so this
    runs once per layer.
    """
    import warnings
    import pyogrio
    import shapely
    d = Path(gdb).parent / f"{Path(gdb).stem}__{layer}.parquet.d"
    done = d / "_COMPLETE"
    if done.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    n = pyogrio.read_info(gdb, layer=layer)["features"]
    print(f"  caching {Path(gdb).name}/{layer}: {n} features (one time) ...")
    for k, skip in enumerate(range(0, n, FGDB_CHUNK)):
        part = d / f"part-{k:04d}.parquet"
        if part.exists():
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="organizePolygons")
            g = pyogrio.read_dataframe(gdb, layer=layer, skip_features=skip, max_features=FGDB_CHUNK,
                                       datetime_as_string=True)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        if g.crs is not None and g.crs.to_epsg() != 4326:
            g = g.to_crs(4326)
        b = g.geometry.bounds
        df = pd.DataFrame(g.drop(columns="geometry"))
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("string")
        df["xmin"], df["ymin"], df["xmax"], df["ymax"] = b["minx"].values, b["miny"].values, b["maxx"].values, b["maxy"].values
        df["wkb"] = shapely.to_wkb(g.geometry.values)
        df.to_parquet(part, index=False)
    done.write_text("ok")
    return d


def _read_fgdb(layer_url, bbox):
    import geopandas as gpd
    import shapely
    from shapely.geometry import box
    gdb, layer = layer_url[len("fgdb://"):].split("|")
    d = _fgdb_cache(gdb, layer)
    l, b, r, t = bbox
    flt = [("xmax", ">=", l), ("xmin", "<=", r), ("ymax", ">=", b), ("ymin", "<=", t)]
    parts = [pd.read_parquet(p, filters=flt) for p in sorted(d.glob("part-*.parquet"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return []
    df = pd.concat(parts, ignore_index=True)
    geom = shapely.from_wkb(df.pop("wkb").values)
    g = gpd.GeoDataFrame(df.drop(columns=["xmin", "ymin", "xmax", "ymax"]), geometry=geom, crs=4326)
    g = g[g.intersects(box(*bbox))]
    if g.empty:
        return []
    feats = []
    for props, geom in zip(g.drop(columns="geometry").to_dict("records"), g.geometry):
        if geom is None:
            continue
        for k, v in props.items():
            if v is pd.NA or v is pd.NaT:
                props[k] = None
            elif isinstance(v, pd.Timestamp):
                props[k] = None if pd.isna(v) else int((v.tz_localize("UTC") if v.tzinfo is None else v).timestamp() * 1000)
            elif isinstance(v, float) and np.isnan(v):
                props[k] = None
        feats.append({"type": "Feature", "geometry": geom.__geo_interface__, "properties": props})
    return feats


def query_layer(sess, layer_url, bbox, cache_dir):
    """All features intersecting bbox (lon/lat), as a list of GeoJSON features. Cached."""
    if layer_url.startswith("fgdb://"):
        return _read_fgdb(layer_url, bbox)
    key = re.sub(r"[^A-Za-z0-9]+", "_", layer_url.split("/services/")[1])
    cache = cache_dir / f"{key}__{'_'.join(f'{b:.4f}' for b in bbox)}.geojson"
    if cache.exists():
        return json.loads(cache.read_text())["features"]
    feats, offset, page = [], 0, 1000
    while True:
        js = get_json(sess, f"{layer_url}/query", {
            "where": "1=1",
            "geometry": ",".join(str(b) for b in bbox),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": page,
            "f": "geojson",
        })
        batch = js.get("features", [])
        feats.extend(batch)
        more = js.get("exceededTransferLimit") or js.get("properties", {}).get("exceededTransferLimit")
        if not batch or (len(batch) < page and not more):
            break
        offset += len(batch)
    cache.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return feats


# ---------------------------------------------------------------------------
# 3. Normalize + clean
# ---------------------------------------------------------------------------

def _ms_to_dt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    if isinstance(v, (int, float, np.integer, np.floating)):
        return pd.Timestamp(int(v), unit="ms", tz="UTC")
    return pd.to_datetime(v, utc=True, errors="coerce")


def _norm_id(v):
    if not v or not isinstance(v, str):
        return None
    v = v.strip().strip("{}").upper()
    return v or None


def name_problem(name):
    """Why an incident name looks like an editor's note rather than a fire name, or ''.

    Seen in the 2019 archive: 'Kendra Fallon @ 8/6/2019' and blank names.
    """
    n = (name or "").strip()
    if not n:
        return "blank name"
    if "@" in n:
        return "editor note in name"
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", n):
        return "date in name"
    return ""


def normalize(feats, kind, layer_url):
    """GeoJSON features from either archive -> one GeoDataFrame with common columns."""
    import geopandas as gpd
    from shapely.geometry import shape
    rows = []
    for f in feats:
        if not f.get("geometry"):
            continue
        p = {k.lower(): v for k, v in (f.get("properties") or {}).items()}
        if kind == "geomac":
            category = "Wildfire Daily Fire Perimeter"
            t = _ms_to_dt(p.get("perimeterdatetime"))
            if pd.isna(t):
                t = _ms_to_dt(p.get("datecurrent"))
            deleted = False
            version_time = _ms_to_dt(p.get("datecurrent"))
            geometry_id = f"geomac_{p.get('objectid')}"
        else:
            category = p.get("featurecategory")
            t = _ms_to_dt(p.get("polygondatetime"))
            if pd.isna(t):
                t = _ms_to_dt(p.get("datecurrent"))
            if pd.isna(t):
                t = _ms_to_dt(p.get("gdb_from_date"))
            deleted = str(p.get("deletethis") or "").strip().lower() in {"yes", "y", "true", "1"}
            version_time = _ms_to_dt(p.get("gdb_from_date"))
            geometry_id = p.get("geometryid") or p.get("globalid")
        rows.append({
            "source": kind,
            "layer": layer_label(layer_url),
            "incident_name": (p.get("incidentname") or "").strip(),
            "name_problem": name_problem(p.get("incidentname")),
            "irwin_id": _norm_id(p.get("irwinid")),
            "unique_fire_id": p.get("uniquefireidentifier") or p.get("localincidentid") or p.get("localincidentidentifier"),
            "complex_name": p.get("complexname"),
            "category": category,
            "map_method": (p.get("mapmethod") or "Unknown").strip(),
            "comments": p.get("comments") or "",
            "acres": p.get("gisacres"),
            "perim_time": t,
            "version_time": version_time,
            "geometry_id": str(geometry_id),
            "deleted": deleted,
            "geometry": shape(f["geometry"]),
        })
    if not rows:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def incident_key(row):
    if row["irwin_id"]:
        return row["irwin_id"]
    name = re.sub(r"\s+", " ", (row["incident_name"] or "").lower()).strip()
    return f"name:{name}|{row['unique_fire_id'] or ''}"


def burn_day(t, lon):
    """Local solar date, shifted so night IR flights count toward the day that just burned."""
    local = t + pd.Timedelta(hours=lon / 15.0) - pd.Timedelta(hours=BURN_DAY_SHIFT_H)
    return local.tz_localize(None).normalize()


def clean(gdf, keep_methods=None):
    """Returns (clean_gdf, counts_of_what_was_dropped)."""
    import shapely
    drops = {}

    def drop(mask, why):
        n = int(mask.sum())
        if n:
            drops[why] = drops.get(why, 0) + n
        return gdf[~mask]

    if gdf.empty:
        return gdf, drops
    gdf = gdf.copy()
    gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    gdf = drop(gdf.geometry.is_empty | gdf.geometry.isna(), "empty geometry")
    gdf = drop(~gdf["category"].isin(NIFS_CATEGORIES), "not a wildfire perimeter")
    gdf = drop(gdf["deleted"], "marked DeleteThis")
    gdf = drop(gdf["perim_time"].isna(), "no date")
    # 1899-12-29 / 1970-01-01 are 'empty date' placeholders in some archive records.
    gdf = drop(gdf["perim_time"] < pd.Timestamp("2000-01-01", tz="UTC"), "placeholder date (before 2000)")
    gdf = drop(gdf["comments"].str.contains(r"\btest\b", case=False, na=False), "test edit")
    gdf = drop(gdf["incident_name"].str.contains(TEST_NAME_RE, case=False, na=False, regex=True),
               "test / placeholder incident")
    methods = gdf["map_method"].str.lower()
    if keep_methods is None:
        gdf = drop(methods.isin(DROP_MAP_METHODS), "hand sketch / modeled")
    else:
        gdf = drop(~methods.isin({m.lower() for m in keep_methods}), "map method not kept")
    area_acres = gdf.to_crs(5070).area / 4046.8564224
    gdf["acres_calc"] = area_acres
    gdf = drop(gdf["acres_calc"] < MIN_ACRES, f"under {MIN_ACRES} acre")
    if gdf.empty:
        return gdf, drops

    gdf["incident"] = gdf.apply(incident_key, axis=1)
    lon = gdf.geometry.representative_point().x
    gdf["burn_day"] = [burn_day(t, x) for t, x in zip(gdf["perim_time"], lon)]
    gdf["method_rank"] = gdf["map_method"].str.lower().map(MAP_METHOD_RANK).fillna(5)

    # Archive edits: the same feature at the same perimeter time saved several times.
    gdf = gdf.sort_values(["version_time"], na_position="first")
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["geometry_id", "perim_time"], keep="last")
    if before - len(gdf):
        drops["older archive version"] = before - len(gdf)

    # One perimeter per incident per burn day: the latest, then best map method, then largest.
    gdf = gdf.sort_values(["incident", "burn_day", "perim_time", "method_rank", "acres_calc"],
                          ascending=[True, True, False, True, False])
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["incident", "burn_day"], keep="first")
    if before - len(gdf):
        drops["extra perimeter same day"] = before - len(gdf)

    # Shrinking re-maps: an incident's perimeter that is much smaller than one it already had.
    keep = []
    for _, grp in gdf.sort_values("burn_day").groupby("incident", sort=False):
        best = 0.0
        for idx, a in zip(grp.index, grp["acres_calc"]):
            if best and a < SHRINK_TOL * best:
                continue
            best = max(best, a)
            keep.append(idx)
    n_shrink = len(gdf) - len(keep)
    if n_shrink:
        drops["shrinking re-map"] = n_shrink
    gdf = gdf.loc[keep].sort_values(["incident", "burn_day"])
    return gdf, drops


# ---------------------------------------------------------------------------
# 4. Match to the fire + rasterize
# ---------------------------------------------------------------------------

def fire_window(fire):
    if fire["first_day"]:
        start = pd.Timestamp(fire["first_day"]) - pd.Timedelta(days=PAD_BEFORE_DAYS)
        end = pd.Timestamp(fire["last_day"]) + pd.Timedelta(days=PAD_AFTER_DAYS)
    else:
        start = pd.Timestamp(f"{fire['fire_year']}-01-01")
        end = pd.Timestamp(f"{fire['fire_year']}-12-31")
    return start, end


def match_incidents(gdf, fire):
    """Keep incidents with a perimeter inside the fire's box and date window."""
    from shapely.geometry import box
    if gdf.empty:
        return gdf, None
    start, end = fire_window(fire)
    bb = box(*fire["bounds_lonlat"])
    in_win = gdf[(gdf["burn_day"] >= start) & (gdf["burn_day"] <= end) & gdf.intersects(bb)]
    if in_win.empty:
        return in_win, None
    keep_inc = set(in_win["incident"])
    # Only perimeters inside the fire's window. Without the lower bound, a matched incident
    # brought in every older perimeter it ever had (one 2020 fire reached back to 1899),
    # which stretched the label dates and the suppression stacks by months or decades.
    # Area burned before the window is still captured: perimeters are cumulative outlines,
    # so the first one in the window already contains it (it becomes the day-0 burned state).
    out = gdf[gdf["incident"].isin(keep_inc) & (gdf["burn_day"] >= start) & (gdf["burn_day"] <= end)].copy()
    clipped = in_win.to_crs(5070).intersection(bb_5070(bb)).area
    primary = clipped.groupby(in_win["incident"]).max().idxmax()
    return out, primary


def main_incidents(gdf, primary):
    """The incidents that make up the fire TS-SatFire tracked: the primary incident plus any
    incident whose footprint comes within MAIN_TOUCH_M of it (fires that merged into it, or a
    spot fire / split-off with its own name), grown until nothing else joins.

    Separate fires that only share the bounding box are left out of the labels; they are
    returned in other_fire_mask so they can be ignored in the loss instead of counted as
    unburned.
    """
    if ALL_INCIDENTS or primary is None:
        return set(gdf["incident"])
    g = gdf.to_crs(5070)
    foot = g.geometry.buffer(0).groupby(g["incident"]).apply(lambda s: s.union_all() if hasattr(s, "union_all") else s.unary_union)
    group = {primary}
    grown = foot[primary].buffer(MAIN_TOUCH_M)
    changed = True
    while changed:
        changed = False
        for inc, f in foot.items():
            if inc not in group and f.intersects(grown):
                group.add(inc)
                grown = grown.union(f.buffer(MAIN_TOUCH_M))
                changed = True
    return group


def bb_5070(bb):
    import geopandas as gpd
    return gpd.GeoSeries([bb], crs=4326).to_crs(5070).iloc[0]


def rasterize_arrival(gdf, fire):
    """arrival_day[y, x] = index into `dates` of the first day the pixel is inside a perimeter."""
    from affine import Affine
    from rasterio.features import rasterize
    from shapely.ops import unary_union
    H, W = fire["height"], fire["width"]
    tr = Affine(*fire["transform"])
    dates = sorted(gdf["burn_day"].unique())
    arrival = np.full((H, W), -1, dtype=np.int16)
    cum = {}  # incident -> cumulative geometry
    stats = []
    for t, d in enumerate(dates):
        today = gdf[gdf["burn_day"] == d]
        for inc, g in zip(today["incident"], today.geometry):
            cum[inc] = g if inc not in cum else unary_union([cum[inc], g])
        geom = unary_union(list(cum.values()))
        m = rasterize([(geom, 1)], out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
        new = m & (arrival < 0)
        arrival[new] = t
        stats.append({"date": str(pd.Timestamp(d).date()), "n_new_perimeters": len(today),
                      "incidents_updated": ";".join(sorted(set(today["incident_name"]))),
                      "burned_px": int(m.sum()), "new_px": int(new.sum())})
    return arrival, [str(pd.Timestamp(d).date()) for d in dates], stats


def burned_mask(arrival_day, t):
    return (arrival_day >= 0) & (arrival_day <= t)


def burned_fraction_375(arrival_day, t, cells_per_pixel=5):
    m = burned_mask(arrival_day, t).astype(np.float32)
    H, W = m.shape
    c = cells_per_pixel
    return m[: H // c * c, : W // c * c].reshape(H // c, c, W // c, c).mean(axis=(1, 3))


def quicklook(arrival, dates, fire, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    a = np.ma.masked_less(arrival, 0)
    l, b, r, t = fire["bounds_lonlat"]
    im = ax.imshow(a, extent=[l, r, b, t], cmap="inferno_r", interpolation="nearest")
    if dates:
        cb = fig.colorbar(im, ax=ax, shrink=0.7)
        ticks = np.linspace(0, len(dates) - 1, min(len(dates), 6)).astype(int)
        cb.set_ticks(ticks)
        cb.set_ticklabels([dates[i] for i in ticks])
    ax.set_title(f"Fire {fire['fire_id']} ({fire['fire_year']}): NIFC arrival day")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_fire(fire, raw, out_dir):
    """raw: GeoDataFrame from normalize() for all sources covering this fire."""
    gdf, drops = clean(raw)
    gdf, primary = match_incidents(gdf, fire)
    row = {k: fire[k] for k in ["fire_id", "fire_year", "first_day", "last_day"]}
    row.update({"raw_features": len(raw), "dropped": json.dumps(drops)})
    if gdf is None or gdf.empty:
        row.update({"status": "no NIFC perimeters matched", "n_incidents": 0, "n_perimeter_days": 0})
        return row
    main_set = main_incidents(gdf, primary)
    gdf["main"] = gdf["incident"].isin(main_set)
    others = gdf[~gdf["main"]]
    all_rows = gdf
    gdf = gdf[gdf["main"]].copy()
    arrival, dates, stats = rasterize_arrival(gdf, fire)
    other_mask = np.zeros_like(arrival, dtype=bool)
    if len(others):
        from affine import Affine
        from rasterio.features import rasterize
        other_mask = rasterize([(g, 1) for g in others.geometry], out_shape=arrival.shape,
                               transform=Affine(*fire["transform"]), fill=0, dtype="uint8").astype(bool)
        other_mask &= arrival < 0  # pixels the main fire reached stay labelled
    prim_rows = gdf[gdf["incident"] == primary]
    prim = prim_rows.iloc[0].copy()
    good = prim_rows.loc[prim_rows["name_problem"] == "", "incident_name"]
    # Primary incident keyed by an editor's note (e.g. 'Kendra Fallon @ 8/13/2019'): take the
    # name of the largest well-named incident in the main fire instead.
    if not len(good):
        named = gdf[gdf["name_problem"] == ""]
        if len(named):
            good = named.sort_values("acres_calc", ascending=False)["incident_name"].iloc[:1]
    if len(good):
        prim["incident_name"] = good.value_counts().idxmax()
        primary_name_flag = ""
    else:
        primary_name_flag = prim_rows["name_problem"].iloc[0]
        prim["incident_name"] = f"(unnamed, IRWIN {prim['irwin_id']})" if prim["irwin_id"] else "(unnamed)"
    odd = sorted({f"{n or '(blank)'}: {w}" for n, w in zip(gdf["incident_name"], gdf["name_problem"]) if w})
    start, end = pd.Timestamp(fire["first_day"] or dates[0]), pd.Timestamp(fire["last_day"] or dates[-1])
    days_in_ts = pd.date_range(start, end)
    covered = sum(pd.Timestamp(d) in set(pd.to_datetime(dates)) for d in days_in_ts)
    row.update({
        "status": "ok",
        "n_incidents": gdf["incident"].nunique(),
        "incidents": "; ".join(sorted(set(gdf["incident_name"]))),
        "n_other_incidents": others["incident"].nunique(),
        "other_incidents": "; ".join(sorted(set(others["incident_name"]))),
        "other_fire_km2": round(float(other_mask.sum()) * fire_pixel_km2(fire), 1),
        "primary_incident": prim["incident_name"],
        "primary_irwin_id": prim["irwin_id"],
        "primary_name_flag": primary_name_flag,
        "suspect_names": "; ".join(odd),
        "n_perimeter_days": len(dates),
        "first_perimeter_day": dates[0],
        "last_perimeter_day": dates[-1],
        "ts_days_with_new_perimeter": f"{covered}/{len(days_in_ts)}",
        "final_burned_km2": round(float((arrival >= 0).sum()) * fire_pixel_km2(fire), 1),
        "map_methods": "; ".join(f"{k}:{v}" for k, v in gdf["map_method"].value_counts().items()),
        "sources": "; ".join(sorted(set(gdf["source"]))),
    })
    fid = fire["fire_id"]
    cols = ["source", "layer", "incident", "incident_name", "name_problem", "irwin_id", "main", "category",
            "map_method", "perim_time", "burn_day", "acres_calc", "geometry"]
    g = all_rows[cols].copy()
    g["main"] = g["main"].astype(int)  # 1 = part of the labelled fire, 0 = other fire in the box
    g["perim_time"] = g["perim_time"].astype(str)
    g["burn_day"] = g["burn_day"].astype(str)
    g.to_file(out_dir / "perimeters" / f"{fid}.gpkg", driver="GPKG")
    np.savez_compressed(
        out_dir / "labels" / f"{fid}.npz",
        arrival_day=arrival, dates=np.array(dates),
        n_new_perimeters=np.array([s["n_new_perimeters"] for s in stats], dtype=np.int16),
        burned_px=np.array([s["burned_px"] for s in stats], dtype=np.int64),
        transform=np.array(fire["transform"]), crs=np.array(fire["crs"]),
        bounds_lonlat=np.array(fire["bounds_lonlat"]), cells_per_pixel=np.int16(fire["cells_per_pixel"]),
        other_fire_mask=other_mask,
        fire_id=np.array(fid), primary_incident=np.array(prim["incident_name"]),
        primary_irwin_id=np.array(prim["irwin_id"] or ""),
    )
    pd.DataFrame(stats).to_csv(out_dir / "labels" / f"{fid}_days.csv", index=False)
    quicklook(arrival, dates, fire, out_dir / "quicklook" / f"{fid}.png")
    return row


def fire_pixel_km2(fire):
    a, _, _, _, e, _ = fire["transform"][:6]
    lat = np.deg2rad(np.mean(fire["bounds_lonlat"][1::2]))
    return abs(a) * 111.32 * np.cos(lat) * abs(e) * 110.57


def write_summary(path, rows):
    """Write this run's rows, keeping rows for fires from earlier runs that were not redone."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, dtype={"fire_id": str})
        old = old[~old["fire_id"].astype(str).isin(new["fire_id"].astype(str))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


SHARED_OVERLAP = 0.3  # share of the smaller fire's burned area that two fires must share


def find_shared_fires(out_dir):
    """Fires whose NIFC perimeters are the same real fire (e.g. one incident split into two
    TS-SatFire ids). They must go in the same train/validation/test split.

    Writes shared_fires.csv (one row per overlapping pair) and split_groups.csv
    (fire_id -> split_group); adds split_group to summary.csv.
    """
    import geopandas as gpd
    from shapely.ops import unary_union
    summ = pd.read_csv(out_dir / "summary.csv", dtype={"fire_id": str})
    ok = summ[summ["status"] == "ok"]
    foot = {}
    for fid in ok["fire_id"]:
        p = out_dir / "perimeters" / f"{fid}.gpkg"
        if p.exists():
            g = gpd.read_file(p).to_crs(5070)
            if "main" in g:
                g = g[g["main"].astype(int) == 1]
            foot[fid] = unary_union(list(g.geometry.buffer(0)))
    irwin = dict(zip(ok["fire_id"], ok.get("primary_irwin_id", pd.Series(dtype=str)).fillna("")))
    ids = sorted(foot)
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            same_irwin = bool(irwin.get(a)) and irwin.get(a) == irwin.get(b)
            if not same_irwin and not foot[a].intersects(foot[b]):
                continue
            inter = foot[a].intersection(foot[b]).area
            small = min(foot[a].area, foot[b].area) or 1.0
            share = inter / small
            if same_irwin or share >= SHARED_OVERLAP:
                pairs.append({"fire_a": a, "fire_b": b, "same_primary_irwin": same_irwin,
                              "overlap_of_smaller": round(share, 3),
                              "incident_a": ok.set_index("fire_id").at[a, "primary_incident"],
                              "incident_b": ok.set_index("fire_id").at[b, "primary_incident"]})
    parent = {f: f for f in summ["fire_id"]}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in pairs:
        parent[root(p["fire_a"])] = root(p["fire_b"])
    groups = {f: root(f) for f in summ["fire_id"]}
    pd.DataFrame(pairs, columns=["fire_a", "fire_b", "same_primary_irwin", "overlap_of_smaller",
                                 "incident_a", "incident_b"]).to_csv(out_dir / "shared_fires.csv", index=False)
    pd.DataFrame({"fire_id": list(groups), "split_group": list(groups.values())}).to_csv(
        out_dir / "split_groups.csv", index=False)
    summ["split_group"] = summ["fire_id"].map(groups)
    summ.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nShared-fire check: {len(pairs)} overlapping pair(s); split by split_group, not fire_id.")
    for p in pairs:
        print(f"  {p['fire_a']} ({p['incident_a']}) and {p['fire_b']} ({p['incident_b']}): "
              f"{p['overlap_of_smaller']:.0%} overlap" + (", same IRWIN id" if p["same_primary_irwin"] else ""))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nifc_out")
    ap.add_argument("--qa-only", action="store_true",
                    help="skip downloading; only run the shared-fire check on existing outputs")
    ap.add_argument("--limit", type=int, help="only the first N fires (for a test run)")
    ap.add_argument("--fires", nargs="*", help="only these fire ids")
    ap.add_argument("--years", nargs="*", type=int, help="only fires from these years")
    ap.add_argument("--all-incidents", action="store_true",
                    help="label every incident in the box, not just the main fire")
    args = ap.parse_args()
    out = Path(args.out)
    global FGDB_DIR, ALL_INCIDENTS
    FGDB_DIR = out / "fgdb"
    ALL_INCIDENTS = args.all_incidents
    for sub in ["perimeters", "labels", "quicklook", "cache"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    if args.qa_only:
        find_shared_fires(out)
        return

    import geopandas as gpd
    fires = list_fires(out, args.limit, set(args.fires) if args.fires else None)
    if args.years:
        fires = [f for f in fires if f["fire_year"] in set(args.years)]
        print(f"{len(fires)} fires in {args.years}")
    years = set()
    for f in fires:
        s, e = fire_window(f)
        years |= set(range(s.year, e.year + 1))
    print(f"\nFire years needed: {sorted(years)}")
    sess = _session()
    sources = discover_sources(sess, years)

    rows = []
    for i, fire in enumerate(fires):
        fid = fire["fire_id"]
        print(f"\n[{i + 1}/{len(fires)}] fire {fid} ({fire['fire_year']}, {fire['first_day']} to {fire['last_day']})")
        s, e = fire_window(fire)
        parts = []
        for y in range(s.year, e.year + 1):
            for kind, url in sources.get(y, []):
                feats = query_layer(sess, url, fire["bounds_lonlat"], out / "cache")
                print(f"    {layer_label(url)}: {len(feats)} features")
                parts.append(normalize(feats, kind, url))
        parts = [p for p in parts if not p.empty]
        raw = pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame(geometry=[], crs=4326)
        raw = gpd.GeoDataFrame(raw, geometry="geometry", crs=4326)
        try:
            row = process_fire(fire, raw, out)
        except Exception as ex:  # noqa: BLE001
            row = {"fire_id": fid, "fire_year": fire["fire_year"], "status": f"error: {ex}"}
        print(f"    -> {row.get('status')}: {row.get('primary_incident', '')} "
              f"{row.get('n_perimeter_days', 0)} perimeter days, coverage {row.get('ts_days_with_new_perimeter', '-')}")
        rows.append(row)
        write_summary(out / "summary.csv", rows)

    df = pd.read_csv(out / "summary.csv")
    print("\nDone.", (df["status"] == "ok").sum(), "of", len(df), "fires matched. Summary:", out / "summary.csv")
    flag_col = df["primary_name_flag"] if "primary_name_flag" in df else pd.Series("", index=df.index)
    flagged = df[flag_col.fillna("").astype(str) != ""]
    if len(flagged):
        print(f"{len(flagged)} fire(s) whose primary incident has no usable name (check them):",
              ", ".join(flagged["fire_id"].astype(str)))
    find_shared_fires(out)


if __name__ == "__main__":
    sys.exit(main())
