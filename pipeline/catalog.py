"""
Multi-year catalog of fires with daily perimeters (2014-2025), the training-set expansion.

Sources (NIFC ArcGIS Online, owner NIFC_Authoritative):
  GeoMAC historic perimeters            2014-2019   Historic_Geomac_Perimeters_<year>
  Operational Data Archive (NIFS)       2018-2025   services, or File Geodatabase for 2020 and 2021

Steps (each resumable; outputs in OUT/catalog):
  scan     attributes only (no geometry) of every wildfire perimeter record -> attrs_<year>.parquet
           (FGDB years are read directly, geometry included -> geoms_<year>.parquet)
  select   candidate incidents: final size >= --min-acres and >= --min-days distinct burn days
  geoms    fetch (lightly simplified, 20 m) geometry for the candidates' records only
  clean    same cleaning as the TS-SatFire labels (nifc_perimeters.clean): drop hand sketch / modeled,
           archive edit duplicates, one perimeter per burn day, shrinking re-maps -> perimeters.parquet
           + fires.csv (one row per fire: id, name, year, first/last perimeter time, days, acres, box,
           median hours between perimeters)

  python catalog.py all --out data --years 2014-2025
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import nifc_perimeters as N

CATS = ("Wildfire Daily Fire Perimeter", "Wildfire Final Fire Perimeter")
CONUS = (-125.0, 24.0, -66.0, 50.0)
SIMPLIFY_DEG = 0.0002            # ~20 m; labels are rasterised at 150 m


def years_arg(s):
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",")]


def paged(sess, url, params, page=500):
    """All records matching params: list every OBJECTID first, then fetch them in POSTed chunks.
    (Offset paging is unreliable on some archive services: the 2018 NIFS layer repeats records.)"""
    import requests
    ids = N.get_json(sess, f"{url}/query", {"where": params.get("where", "1=1"), "returnIdsOnly": "true", "f": "json"})
    ids = sorted(set(ids.get("objectIds") or []))
    out = []
    for i in range(0, len(ids), page):
        body = dict(params, objectIds=",".join(map(str, ids[i:i + page])))
        body.pop("where", None)
        for t in range(5):
            try:
                r = sess.post(f"{url}/query", data=body, timeout=300)
                r.raise_for_status()
                js = r.json()
                if "error" in js:
                    raise RuntimeError(js["error"])
                break
            except (requests.RequestException, RuntimeError, ValueError) as ex:
                if t == 4:
                    raise
                print(f"    retry {t + 1}: {str(ex)[:120]}", flush=True)
                time.sleep(2 ** (t + 1))
        out += js.get("features", [])
    return out


def layer_fields(sess, url):
    return {f["name"].lower(): f["name"] for f in N.get_json(sess, url).get("fields", [])}


def scan(out, years):
    d = out / "catalog"
    d.mkdir(parents=True, exist_ok=True)
    sess = N._session()
    N.FGDB_DIR = out / "nifc_fgdb"
    sources = N.discover_sources(sess, years)
    for y in years:
        dst = d / f"attrs_{y}.parquet"
        if dst.exists():
            continue
        frames = []
        for kind, url in sources.get(y, []):
            t0 = time.time()
            if url.startswith("fgdb://"):
                frames.append(scan_fgdb(out, y, url))
                continue
            f = layer_fields(sess, url)
            want = [f[k] for k in ("objectid", "incidentname", "irwinid", "uniquefireidentifier", "localincidentidentifier",
                                   "featurecategory", "mapmethod", "gisacres", "perimeterdatetime", "polygondatetime",
                                   "datecurrent", "deletethis", "gdb_from_date", "comments", "complexname", "cpxname",
                                   "state", "globalid", "sourceglobalid") if k in f]
            where = "1=1"
            if "featurecategory" in f:
                where = "FeatureCategory IN ('" + "','".join(CATS) + "')"
            feats = paged(sess, url, {"where": where, "outFields": ",".join(want), "returnGeometry": "false",
                                      "returnCentroid": "true", "outSR": 4326, "f": "json"})
            rows = []
            for ft in feats:
                a = {k.lower(): v for k, v in ft["attributes"].items()}
                c = ft.get("centroid") or {}
                a["cx"], a["cy"] = c.get("x"), c.get("y")
                rows.append(a)
            df = pd.DataFrame(rows)
            df["source"], df["layer_url"] = kind, url
            frames.append(df)
            print(f"  {y} {N.layer_label(url)}: {len(df)} records ({time.time() - t0:.0f}s)", flush=True)
        if frames:
            df = pd.concat(frames, ignore_index=True)
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].astype("string")
            df.to_parquet(dst, index=False)


def scan_fgdb(out, y, url):
    """Read the year's archive geodatabase in chunks: perimeter features only, simplified, with geometry."""
    import warnings

    import pyogrio
    import shapely
    gdb, layer = url[len("fgdb://"):].split("|")
    d = out / "catalog"
    gpath = d / f"geoms_{y}.parquet"
    info = pyogrio.read_info(gdb, layer=layer)
    n = info["features"]
    parts = []
    for skip in range(0, n, 5000):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            g = pyogrio.read_dataframe(gdb, layer=layer, skip_features=skip, max_features=5000, datetime_as_string=True)
        g.columns = [c if c == "geometry" else c.lower() for c in g.columns]
        g = g[g["featurecategory"].isin(CATS) & g.geometry.notna() & ~g.geometry.is_empty]
        if g.empty:
            continue
        if g.crs is not None and g.crs.to_epsg() != 4326:
            g = g.to_crs(4326)
        geom = shapely.simplify(shapely.make_valid(g.geometry.values), SIMPLIFY_DEG, preserve_topology=True)
        ok = ~shapely.is_empty(geom)
        g, geom = g[ok], geom[ok]
        if g.empty:
            continue
        c = shapely.centroid(geom)
        df = pd.DataFrame(g.drop(columns="geometry"))
        df["cx"], df["cy"] = shapely.get_x(c), shapely.get_y(c)
        df["wkb"] = shapely.to_wkb(geom)
        parts.append(df)
        print(f"  {y} fgdb {skip + 5000}/{n}: kept {sum(len(p) for p in parts)}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["source"], df["layer_url"] = "nifs", url
    for col in df.columns:
        if df[col].dtype == object and col != "wkb":
            df[col] = df[col].astype("string")
    df.to_parquet(gpath, index=False)
    return df.drop(columns="wkb")


def geomac_zone(lon, lat, year):
    """Clock a GeoMAC perimeter time was written in (same rule as model/data.py geomac_zone). GeoMAC kept
    local clock times with no zone and the archive serves them as UTC: 2019 records are Denver time
    for every fire (250 perimeters also in NIFS differ by exactly 6 h); earlier years are the fire's
    own local time (NIROPS comments like '2338hrs' equal the stored time)."""
    if year >= 2019:
        return "America/Denver"
    if lon < -130:
        return "America/Anchorage"
    if lon > -85.5:
        return "America/New_York"
    if lon > -101.5:
        return "America/Chicago"
    if lat < 37.0 and -114.8 < lon < -109.05:
        return "America/Phoenix"
    if lon > -114.05 or (lon > -117.03 and 42.0 <= lat <= 45.6):
        return "America/Denver"
    return "America/Los_Angeles"


def _time(df, with_field=False):
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    field = pd.Series(None, index=df.index, dtype="object")
    for c in ("perimeterdatetime", "polygondatetime", "datecurrent", "gdb_from_date"):
        if c in df:
            v = df[c]
            num = pd.to_numeric(v, errors="coerce")
            # milliseconds since 1970 between 2000 and 2030; anything else is an archive placeholder
            ok = num.between(946684800000, 1893456000000)
            t = pd.to_datetime(num.where(ok), unit="ms", utc=True, errors="coerce")
            s = v.where(num.isna()).astype("string")
            yr = pd.to_numeric(s.str.extract(r"^(\d{4})", expand=False), errors="coerce")
            s = s.where(yr.between(2000, 2030))
            t2 = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
            t = t.fillna(t2.astype("datetime64[ns, UTC]"))
            field = field.where(out.notna() | t.isna(), c)
            out = out.fillna(t.astype("datetime64[ns, UTC]"))
    return (out, field) if with_field else out


def incident_keys(df):
    irwin = df.get("irwinid", pd.Series(None, index=df.index)).astype("string").str.strip("{} ").str.upper()
    name = df["incidentname"].astype("string").fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    uid = df.get("uniquefireidentifier", pd.Series("", index=df.index)).astype("string").fillna("")
    return irwin.where(irwin.notna() & (irwin != ""), "name:" + name + "|" + uid)


def select(out, years, min_acres, min_days):
    d = out / "catalog"
    rows = []
    for y in years:
        p = d / f"attrs_{y}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["year"] = y
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["t"] = _time(df)
    df = df[df["t"].notna() & (df["t"] >= pd.Timestamp("2000-01-01", tz="UTC"))]
    cat = df.get("featurecategory")
    if cat is not None:
        df = df[cat.isna() | cat.isin(CATS)]
    ok_method = ~df["mapmethod"].astype("string").str.lower().isin(N.DROP_MAP_METHODS)
    df["key"] = incident_keys(df)
    lon = pd.to_numeric(df["cx"], errors="coerce")
    lat = pd.to_numeric(df["cy"], errors="coerce")
    df = df[lon.between(CONUS[0], CONUS[2]) & lat.between(CONUS[1], CONUS[3])]
    df["day"] = (df["t"] + pd.to_timedelta(lon.reindex(df.index) / 15.0, unit="h") - pd.Timedelta(hours=6)).dt.date
    acres = pd.to_numeric(df["gisacres"], errors="coerce")
    g = df[ok_method.reindex(df.index)].assign(acres=acres).groupby("key")
    summ = g.agg(year=("year", "max"), name=("incidentname", "first"), days=("day", "nunique"),
                 max_acres=("acres", "max"), n=("t", "size"))
    cand = summ[(summ["max_acres"] >= min_acres) & (summ["days"] >= min_days)]
    keep = df[df["key"].isin(cand.index)].copy()
    for c in keep.columns:                    # archives mix ms numbers and ISO strings: store as text
        if keep[c].dtype == object or c in ("perimeterdatetime", "polygondatetime", "datecurrent", "gdb_from_date", "day"):
            keep[c] = keep[c].astype("string")
    keep.to_parquet(d / "candidate_records.parquet", index=False)
    cand.to_csv(d / "candidates.csv")
    print(f"candidate incidents: {len(cand)} (from {summ.shape[0]}); per year:")
    print(cand.groupby("year").size().to_string())
    return cand


def geoms(out, years):
    d = out / "catalog"
    rec = pd.read_parquet(d / "candidate_records.parquet")
    sess = N._session()
    for y in years:
        dst = d / f"geoms_{y}.parquet"
        r = rec[(rec["year"] == y) & ~rec["layer_url"].str.startswith("fgdb://")]
        if dst.exists() or r.empty:
            continue
        parts = []
        for url, grp in r.groupby("layer_url"):
            ids = sorted(set(pd.to_numeric(grp["objectid"]).astype(int).tolist()))
            t0 = time.time()
            for i in range(0, len(ids), 200):
                for t in range(5):
                    try:
                        r = sess.post(f"{url}/query", data={"objectIds": ",".join(map(str, ids[i:i + 200])), "outFields": "OBJECTID",
                                                            "returnGeometry": "true", "outSR": 4326, "f": "geojson",
                                                            "maxAllowableOffset": SIMPLIFY_DEG, "geometryPrecision": 6}, timeout=300)
                        r.raise_for_status()
                        js = r.json()
                        break
                    except Exception as ex:  # noqa: BLE001
                        if t == 4:
                            raise
                        print(f"    retry {t + 1}: {str(ex)[:120]}", flush=True)
                        time.sleep(2 ** (t + 1))
                for ft in js.get("features", []):
                    if ft.get("geometry"):
                        oid = ft["properties"].get("OBJECTID", ft.get("id"))
                        parts.append((url, int(oid), json.dumps(ft["geometry"])))
            print(f"  {y} {N.layer_label(url)}: {len(ids)} geometries ({time.time() - t0:.0f}s)", flush=True)
        import shapely
        from shapely.geometry import shape
        g = pd.DataFrame(parts, columns=["layer_url", "objectid", "gj"])
        g["wkb"] = shapely.to_wkb([shape(json.loads(s)) for s in g.pop("gj")])
        g.to_parquet(dst, index=False)


def norm_frame(g, geom, year):
    """Vectorised equivalent of nifc_perimeters.normalize (same columns), without GeoJSON round trips."""
    import geopandas as gpd
    g = g.reset_index(drop=True)
    col = lambda c: g[c] if c in g else pd.Series(None, index=g.index, dtype="object")  # noqa: E731
    src = col("source").astype("string").fillna("nifs")
    is_g = src == "geomac"
    t, field = _time(g, with_field=True)
    # where the time came from: the mapping time, a database edit time (hours after the flight), or a bare date
    date_only = (t.dt.hour == 0) & (t.dt.minute == 0) & (t.dt.second == 0)
    tsrc = np.where(field.isin(["perimeterdatetime", "polygondatetime"]), np.where(date_only, "date_only", "mapped"), "edit")
    if is_g.any():                           # GeoMAC: local clock time stored as UTC -> true UTC
        import shapely
        pt = shapely.point_on_surface(np.asarray(list(geom), dtype=object))
        gx, gy = shapely.get_x(pt), shapely.get_y(pt)
        fixed = [ti.tz_localize(None).tz_localize(geomac_zone(x, y, year), ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
                 if (gm and pd.notna(ti)) else ti for ti, gm, x, y in zip(t, is_g, gx, gy)]
        t = pd.Series(pd.DatetimeIndex(fixed), index=g.index).astype("datetime64[ns, UTC]")
    vt = _time(pd.DataFrame({"gdb_from_date": col("gdb_from_date")}))
    vt = vt.where(~is_g, _time(pd.DataFrame({"datecurrent": col("datecurrent")})))
    cat = col("featurecategory").astype("string").where(~is_g, "Wildfire Daily Fire Perimeter")
    deleted = col("deletethis").astype("string").str.strip().str.lower().isin(["yes", "y", "true", "1"])
    oid = col("objectid").astype("string")
    gid = col("sourceglobalid").astype("string").fillna(col("globalid").astype("string")).fillna(col("layer_url").astype("string") + "#" + oid)
    gid = gid.where(~is_g, "geomac_" + oid)
    name = col("incidentname").astype("string").fillna("").str.strip()
    irwin = col("irwinid").astype("string").str.strip("{} ").str.upper()
    irwin = irwin.where(irwin.notna() & (irwin != ""), None)
    uid = col("uniquefireidentifier").astype("string").fillna(col("localincidentidentifier").astype("string"))
    df = pd.DataFrame({"source": src, "layer": f"archive_{year}", "incident_name": name.astype(object),
                       "name_problem": [N.name_problem(n) for n in name], "irwin_id": irwin.astype(object),
                       "unique_fire_id": uid.astype(object), "complex_name": None, "category": cat.astype(object),
                       "map_method": col("mapmethod").astype("string").fillna("Unknown").str.strip().astype(object),
                       "comments": col("comments").astype("string").fillna("").astype(object), "acres": pd.to_numeric(col("gisacres"), errors="coerce"),
                       "perim_time": t, "time_source": tsrc, "times_utc": True,
                       "version_time": vt, "geometry_id": gid.astype(object), "deleted": deleted.values})
    return gpd.GeoDataFrame(df, geometry=list(geom), crs=4326)


def polygonal(g):
    """Polygon parts only (None if there are none)."""
    import shapely
    from shapely.geometry import MultiPolygon
    if g is None or g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [p for p in shapely.get_parts(g) if p.geom_type in ("Polygon", "MultiPolygon")]
    polys = [q for p in parts for q in (p.geoms if p.geom_type == "MultiPolygon" else [p])]
    return MultiPolygon(polys) if polys else None


N_BUCKETS = 8
MAX_GAP_DAYS = 10        # perimeters further apart than this start a new episode


def main_episode(grp):
    """Split an incident at gaps > MAX_GAP_DAYS and keep the episode with the most perimeter days
    (ties: the most growth). Removes misdated stray perimeters, e.g. a 'March' record on an August fire."""
    gap = grp["perim_time"].diff().dt.total_seconds().fillna(0) / 86400
    seg = (gap > MAX_GAP_DAYS).cumsum()
    if seg.max() == 0:
        return grp
    best = max(seg.unique(), key=lambda k: (grp.loc[seg == k, "burn_day"].nunique(),
                                             grp.loc[seg == k, "acres_calc"].max() - grp.loc[seg == k, "acres_calc"].min()))
    return grp[seg == best]



def _bucket(df):
    """Same incident key as nifc_perimeters.incident_key, hashed into N_BUCKETS (cleaning is per incident)."""
    import zlib
    name = df["incident_name"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    key = df["irwin_id"].where(df["irwin_id"].notna(), "name:" + name + "|" + df["unique_fire_id"].fillna("").astype(str))
    return key.map(lambda k: zlib.crc32(str(k).encode()) % N_BUCKETS).astype("int8")


def clean_all(out, years, min_acres, min_days):
    """Normalise each year to disk, then clean in incident buckets (keeps memory under ~2 GB)."""
    import geopandas as gpd
    import shapely
    d = out / "catalog"
    nd = d / "norm"
    nd.mkdir(exist_ok=True)
    rec = pd.read_parquet(d / "candidate_records.parquet")
    keys = set(rec["key"])
    for y in years:
        p, q = d / f"geoms_{y}.parquet", nd / f"norm_{y}.parquet"
        if not p.exists() or q.exists():
            continue
        g = pd.read_parquet(p)
        if "featurecategory" in g:     # FGDB year: attributes and geometry together
            a = g.drop(columns="wkb")
            a["key"] = incident_keys(a)
            g = g.loc[a.index[a["key"].isin(keys)]]
        else:
            r = rec[rec["year"] == y].copy()
            r["objectid"] = pd.to_numeric(r["objectid"], errors="coerce").astype("Int64")
            r = r.drop_duplicates(["layer_url", "objectid"])
            g["objectid"] = pd.to_numeric(g["objectid"], errors="coerce").astype("Int64")
            g = g.merge(r, on=["layer_url", "objectid"])
        if g.empty:
            continue
        geom = shapely.from_wkb(g.pop("wkb").values)
        nf = norm_frame(g, geom, y)
        nf["bucket"] = _bucket(nf)
        nf.to_parquet(q, index=False)
        print(f"  {y}: {len(nf)} records", flush=True)
        del g, geom, nf
    rows, parts, drops_all = [], [], {}
    for b in range(N_BUCKETS):
        frames = [gpd.read_parquet(q, filters=[("bucket", "==", b)]) for q in sorted(nd.glob("norm_*.parquet"))]
        gdf = gpd.GeoDataFrame(pd.concat([f for f in frames if len(f)], ignore_index=True), geometry="geometry", crs=4326)
        del frames
        gdf["geometry"] = [polygonal(x) for x in gdf.geometry]
        gdf = gdf[gdf.geometry.notna()]
        gdf, drops = N.clean(gdf)
        for k, v in drops.items():
            drops_all[k] = drops_all.get(k, 0) + v
        for inc, grp in gdf.groupby("incident"):
            grp = main_episode(grp.sort_values("perim_time"))
            if grp["acres_calc"].max() < min_acres or grp["burn_day"].nunique() < min_days:
                continue
            bb = grp.total_bounds
            gaps = grp["perim_time"].diff().dt.total_seconds().dropna() / 3600
            fid = "F" + hashlib.sha1(inc.encode()).hexdigest()[:10]
            rows.append(dict(fire_id=fid, incident=inc, name=grp["incident_name"].iloc[-1], year=grp["perim_time"].iloc[-1].year,
                             first=grp["perim_time"].iloc[0], last=grp["perim_time"].iloc[-1], days=grp["burn_day"].nunique(),
                             perimeters=len(grp), final_acres=float(grp["acres_calc"].max()),
                             first_acres=float(grp["acres_calc"].iloc[0]),
                             median_gap_h=float(gaps.median()) if len(gaps) else np.nan,
                             west=bb[0], south=bb[1], east=bb[2], north=bb[3], sources=",".join(sorted(grp["source"].unique())),
                             methods=",".join(sorted(grp["map_method"].str.lower().unique()))))
            keep = ["incident", "incident_name", "source", "category", "map_method", "perim_time", "burn_day",
                    "acres_calc", "irwin_id", "geometry"] + [c for c in ("time_source", "times_utc") if c in grp]
            parts.append(grp[keep].assign(fire_id=fid))
        print(f"  bucket {b + 1}/{N_BUCKETS}: {len(rows)} fires so far", flush=True)
        del gdf
    print("dropped:", drops_all)
    fires = pd.DataFrame(rows)
    per = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326)
    per.to_parquet(d / "perimeters.parquet", index=False)
    fires.to_csv(d / "fires.csv", index=False)
    print(f"\nfires: {len(fires)}  perimeters: {len(per)}")
    print(fires.groupby("year").agg(fires=("fire_id", "size"), median_days=("days", "median"),
                                    median_acres=("final_acres", "median"), median_gap_h=("median_gap_h", "median")).round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["scan", "select", "geoms", "clean", "all"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--years", default="2014-2025")
    ap.add_argument("--min-acres", type=float, default=1000)
    ap.add_argument("--min-days", type=int, default=3)
    a = ap.parse_args()
    out, years = Path(a.out), years_arg(a.years)
    if a.step in ("scan", "all"):
        scan(out, years)
    if a.step in ("select", "all"):
        select(out, years, a.min_acres, a.min_days)
    if a.step in ("geoms", "all"):
        geoms(out, years)
    if a.step in ("clean", "all"):
        clean_all(out, years, a.min_acres, a.min_days)


if __name__ == "__main__":
    sys.exit(main())
