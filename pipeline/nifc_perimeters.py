"""
NIFC perimeter and event-layer access for the raw grab (raw_channels.py: perimeters, suppression).

  discover_sources       {year: [(kind, layer_url)]}: each year's Historic GeoMAC perimeter service and the
                         NIFS / Operational Data Archive layers published as feature services
  archive_fgdb_layers    years the archive publishes only as a File Geodatabase download (ARCHIVE_FGDB_ITEMS):
                         download it once, return fgdb:// urls of its matching layers
  query_layer            every feature intersecting a lon/lat box, from a feature service (paged, cached) or
                         from a downloaded geodatabase, as GeoJSON features
  layer_is, layer_label  a layer's kind however the year spells it, and its short name for manifests
"""
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import zipfile_zstd  # noqa: F401  (lets zipfile read Zstandard members, when installed)
except ImportError:
    pass

NIFC_ROOT = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
# Years whose Operational Data Archive is published only as a File Geodatabase
# download (no feature service): ArcGIS Online item ids, owner NIFC_Authoritative.
ARCHIVE_FGDB_ITEMS = {2020: "ea843f7f091f4c7f9743798b64c864be", 2021: "af727c41d79643b091cee372233110d4"}
ARCGIS_ITEM_DATA = "https://www.arcgis.com/sharing/rest/content/items/{}/data"
FGDB_DIR = Path("nifc_out") / "fgdb"  # raw_channels.Ctx points this at its cache


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
# NIFC sources and queries
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
