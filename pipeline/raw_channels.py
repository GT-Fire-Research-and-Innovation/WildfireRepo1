"""
Upstream raw data for one fire, one channel at a time (driven by raw_grab.py).

"Raw" means the source's own records, values and grid, cut to the fire: the same box (perimeter union
+ 9.6 km) and time windows the built dataset uses (build_multiyear.make_grid / days_of / _fire_span), so a
raw file and the built layer of the same fire line up. Whole-CONUS sources (HRRR, GOES, DEM, OSM) are
cropped on their native grid with untouched values; every file's source URL is logged in the manifest.

Channels (cheap -> heavy) and what each writes into fires/<fire_id>/<channel>/:
  history      Interagency Fire Perimeter History, every prior fire in the box   history.geojson
  perimeters   NIFC GeoMAC / NIFS / Operational Data Archive records of the fire  records.geojson
  firms        NASA FIRMS VIIRS active fire (SP, + NRT fill 2024+)               <SOURCE>.csv.gz
  gridmet      GRIDMET daily, 9 variables (OPeNDAP subset)                       gridmet.nc
  ics209       ICS-209-PLUS incident + sitrep rows (fires <= 2020)              incidents.csv, sitreps.csv
  suppression  NIFS / ODA EventLine + EventPolygon, every archived version (2018+) event_lines.geojson, event_polygons.geojson
  tessera      GeoTessera v1.1 reference only (no embedding data)                tessera_ref.json
  terrain      Copernicus GLO-30 DEM, native 1 arc-second window                 dem_glo30.tif
  landfire     LANDFIRE (LFPS) zip as delivered, native 30 m (fires 2017+)       landfire_<version>.zip
  goes         GOES-16/17/18/19 ABI L2 FDCC, hourly scan, fixed-grid crop        goes_<sat>.nc
  hrrr         HRRR gust/u10/v10/rh2, 3-hourly analyses + 12z+9h forecasts       hrrr_ana.nc, hrrr_fc.nc
  osm          OpenStreetMap 1 January snapshot of the fire's year, full extract osm_<YYYY>-01-01.osm.pbf

Per-fire channels: grab_<channel>(fire, out_dir, ctx) -> (status, sources), status "ok" or "empty".
Group channels (goes, hrrr, landfire, osm) share downloads across fires: grab_<channel>_many(fires, ctx)
yields (fire, status, sources, error). Secrets (FIRMS map key, LANDFIRE email) never reach the manifest.
"""
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

CHANNELS = ["history", "perimeters", "firms", "gridmet", "ics209", "suppression", "tessera", "terrain",
            "landfire", "goes", "hrrr", "osm"]
GROUP = {"goes", "hrrr", "landfire", "osm"}
GOES_VARS = ("Mask", "Power", "Area", "Temp", "DQF")
HRRR_START = "2014-08-01"            # first HRRR files on AWS used by the build
GOES_START = "2017-06-01"
TESSERA_FIRST_YEAR = 2017
UA = "wildfire-raw-grab/1.0"


# ------------------------------------------------------------------------------------- pure helpers

def norm_irwin(s):
    """IRWIN id without braces or whitespace, upper case. The catalog hashed 14 keys with trailing junk
    ('}\\r\\n'); matching on this form finds their records anyway."""
    return re.sub(r"[{}\s]", "", str(s or "")).upper()


def fire_key(fire):
    """(irwin, None) for IRWIN-keyed incidents, (None, (name, uid)) for 'name:<name>|<uid>' keys."""
    inc = str(fire["incident"])
    if inc.startswith("name:"):
        name, _, uid = inc[5:].partition("|")
        return None, (name.strip(), uid.strip())
    return norm_irwin(inc), None


def _ts(v):
    import pandas as pd
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def span(fire):
    """build_multiyear._fire_span: one day before the first perimeter to the last (hour ceiling), UTC."""
    import pandas as pd
    return _ts(fire["first"]).floor("D") - pd.Timedelta(days=1), _ts(fire["last"]).ceil("h")


def days(fire, before=14, after=2):
    """build_multiyear.days_of: calendar days from first - 14 to last + 2."""
    import pandas as pd
    d0 = _ts(fire["first"]).tz_convert(None).normalize() - pd.Timedelta(days=before)
    d1 = _ts(fire["last"]).tz_convert(None).normalize() + pd.Timedelta(days=after)
    return pd.date_range(d0, d1)


def landfire_version(year):
    """Same rule as build_multiyear.landfire_version: newest LANDFIRE that predates the fire."""
    if year <= 2016:
        return None
    if year <= 2022:
        return "LF2016"
    return f"LF{min(year - 1, 2025)}"


def applies(channel, fire):
    """None if the channel has data for this fire, else the reason (recorded as status n/a)."""
    import pandas as pd
    y = int(fire["year"])
    t0, t1 = span(fire)
    if channel == "ics209" and _ts(fire["first"]).year > 2020:
        return "ICS-209-PLUS covers 1999-2020"
    if channel == "suppression" and t1.year < 2018:
        return "no spatial suppression archive before 2018"
    if channel == "landfire" and landfire_version(y) is None:
        return "no LANDFIRE version predates fires before 2017"
    if channel == "tessera" and y - 1 < TESSERA_FIRST_YEAR:
        return "GeoTessera v1.1 starts in 2017"
    if channel == "goes" and t1 < pd.Timestamp(GOES_START, tz="UTC"):
        return "GOES-16 fire product starts 2017-06-01"
    if channel == "hrrr" and t1 < pd.Timestamp(HRRR_START, tz="UTC"):
        return "HRRR archive starts 2014-08"
    if channel == "osm":
        import human_features as HF
        try:
            HF.region_file({"bounds_lonlat": fire["bounds"]})
        except RuntimeError as ex:
            return str(ex)
    return None


def firms_calls(fire):
    """(source, first day, n days) for the standard-product requests: 5-day calls over days()."""
    ds = days(fire)
    srcs = ["VIIRS_SNPP_SP"] + (["VIIRS_NOAA20_SP"] if int(fire["year"]) >= 2018 else [])
    out = []
    for src in srcs:
        i = 0
        while i < len(ds):
            k = min(5, len(ds) - i)
            out.append((src, ds[i].date().isoformat(), k))
            i += k
    return out


def hrrr_times(fire, step_h=3):
    """Analyses every step_h hours across span(), and the 12z run's 9 h forecast for each day."""
    import pandas as pd
    t0, t1 = span(fire)
    ana = list(pd.date_range(t0, t1, freq=f"{step_h}h"))
    fc = [d + pd.Timedelta(hours=12) for d in pd.date_range(t0.floor("D"), t1.floor("D"))]
    return ana, fc


GOES_BUCKETS = [("G16", "noaa-goes16", "2017-06-01", "2025-04-06"), ("G19", "noaa-goes19", "2025-04-07", "2030-01-01"),
                ("G17", "noaa-goes17", "2018-11-15", "2023-01-09"), ("G18", "noaa-goes18", "2023-01-10", "2030-01-01")]


def goes_hours(fire):
    """[(hour, [(sat, bucket), ...])] hourly across span(), with the satellites operating then."""
    import pandas as pd
    t0, t1 = span(fire)
    out = []
    for t in pd.date_range(t0, t1, freq="60min"):
        sats = [(s, b) for s, b, a, z in GOES_BUCKETS if pd.Timestamp(a, tz="UTC") <= t <= pd.Timestamp(z, tz="UTC")]
        if sats:
            out.append((t, sats))
    return out


def redact(url):
    """Source URL safe to publish: no FIRMS map key, no LANDFIRE email."""
    url = re.sub(r"(/api/area/csv/)[^/]+/", r"\1{MAP_KEY}/", url)
    return re.sub(r"([?&])Email=[^&]*&?", r"\1", url).rstrip("&?")


# ------------------------------------------------------------------------------------- context

class Ctx:
    """Per-run state: work dirs, HTTP session, memoised shared resources, deadline."""

    def __init__(self, channel, work, deadline=None, reserve_s=900, threads=16):
        import requests
        import nifc_perimeters as NP
        self.channel, self.work = channel, Path(work)
        self.cache = self.work / "cache" / channel
        self.stage_root = self.work / "stage"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.deadline, self.reserve_s, self.threads = deadline, reserve_s, threads
        self.sess = requests.Session()
        self.sess.headers["User-Agent"] = UA
        NP.FGDB_DIR = self.cache / "fgdb"          # 2020/2021 archive geodatabases land here, deleted at the end
        self._memo = {}
        self._last_call = 0.0

    def stage(self, fid):
        p = self.stage_root / fid / self.channel
        p.mkdir(parents=True, exist_ok=True)
        return p

    def drop_stage(self, fid):
        shutil.rmtree(self.stage_root / fid, ignore_errors=True)

    def expired(self):
        return self.deadline is not None and time.time() > self.deadline - self.reserve_s

    def memo(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def throttle(self, min_gap_s):
        dt = min_gap_s - (time.time() - self._last_call)
        if dt > 0:
            time.sleep(dt)
        self._last_call = time.time()

    def tmp(self):
        p = self.cache / "q"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def cleanup(self):
        shutil.rmtree(self.cache, ignore_errors=True)
        shutil.rmtree(self.stage_root, ignore_errors=True)


def requirements(channel):
    """Raise SystemExit when the channel cannot run here (missing key or tool)."""
    if channel == "firms":
        from live import firms
        firms.map_key()
    if channel == "landfire":
        import landfire_features as L
        L.email()
    if channel == "osm" and not shutil.which("osmium"):
        raise SystemExit("osm needs osmium-tool (apt-get install -y osmium-tool, or conda install osmium-tool)")
    if channel == "hrrr":
        import pygrib  # noqa: F401


def shared_files(channel, ctx):
    """[(local file, repo path)] uploaded once per repo with the first batch."""
    if channel == "ics209":
        _ics_tables(ctx)
        return [(ctx.cache / "ics209" / "ics209-plus-wildfire.zip", "shared/ics209/ics209plus-wildfire.zip")]
    return []


# ------------------------------------------------------------------------------------- helpers

def _write_geojson(path, feats):
    with open(path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, default=str)


def _layer_source(url, bbox):
    import nifc_perimeters as NP
    if url.startswith("fgdb://"):
        gdb, layer = url[len("fgdb://"):].split("|")
        year = next((y for y in NP.ARCHIVE_FGDB_ITEMS if f"/{y}/" in gdb.replace(os.sep, "/")), None)
        item = NP.ARCGIS_ITEM_DATA.format(NP.ARCHIVE_FGDB_ITEMS[year]) if year else gdb
        return f"{item}#{layer}"
    return f"{url}/query?geometry={','.join(f'{b:.5f}' for b in bbox)}&geometryType=esriGeometryEnvelope&inSR=4326&outFields=*"


def _years(fire):
    t0, t1 = span(fire)
    return list(range(_ts(fire["first"]).year, t1.year + 1))


def _arcgis_paged(sess, url, params, page=1000):
    feats, off = [], 0
    while True:
        for attempt in range(6):
            try:
                r = sess.get(f"{url}/query", params=dict(params, resultOffset=off, resultRecordCount=page), timeout=180)
                js = r.json()
                if "error" in js:
                    raise RuntimeError(js["error"])
                break
            except (ValueError, RuntimeError, OSError) as ex:
                if attempt == 5:
                    raise
                print(f"    retry {attempt + 1}: {str(ex)[:120]}", flush=True)
                time.sleep(5 * 2 ** attempt)
        batch = js.get("features", [])
        feats += batch
        more = js.get("exceededTransferLimit") or js.get("properties", {}).get("exceededTransferLimit")
        if not batch or (len(batch) < page and not more):
            return feats
        off += len(batch)


# ------------------------------------------------------------------------------------- per-fire channels

def grab_history(fire, out, ctx):
    import build_multiyear as BM
    w, s, e, n = fire["bounds"]
    params = {"where": f"FIRE_YEAR_INT < {int(fire['year'])}", "geometry": f"{w},{s},{e},{n}",
              "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
              "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "returnGeometry": "true", "f": "geojson"}
    feats = _arcgis_paged(ctx.sess, BM.HIST_URL, params)
    _write_geojson(out / "history.geojson", feats)
    src = f"{BM.HIST_URL}/query?where=FIRE_YEAR_INT<{int(fire['year'])}&geometry={w:.5f},{s:.5f},{e:.5f},{n:.5f}&outFields=*"
    return ("ok" if feats else "empty"), [src]


def _perimeter_layers(ctx, year):
    import nifc_perimeters as NP
    return ctx.memo(("perim_layers", year), lambda: NP.discover_sources(ctx.sess, [year]).get(year, []))


def grab_perimeters(fire, out, ctx):
    """Every raw record of the fire's incident in each archive layer of its years (all attributes, full
    geometry), before any of the catalog's cleaning. _source_layer / _source_kind say where each came from."""
    import nifc_perimeters as NP
    irwin, name_uid = fire_key(fire)
    bbox = tuple(fire["bounds"])
    keep, sources = [], []
    for y in _years(fire):
        for kind, url in _perimeter_layers(ctx, y):
            feats = NP.query_layer(ctx.sess, url, bbox, ctx.tmp())
            sources.append(_layer_source(url, bbox))
            for f in feats:
                p = {k.lower(): v for k, v in (f.get("properties") or {}).items()}
                if irwin:
                    hit = norm_irwin(p.get("irwinid")) == irwin
                else:
                    nm = re.sub(r"\s+", " ", str(p.get("incidentname") or "").lower()).strip()
                    uid = str(p.get("uniquefireidentifier") or p.get("localincidentidentifier") or "")
                    hit = (nm, uid) == name_uid
                if hit:
                    f = dict(f)
                    f["properties"] = dict(f.get("properties") or {}, _source_layer=NP.layer_label(url), _source_kind=kind)
                    keep.append(f)
    shutil.rmtree(ctx.tmp(), ignore_errors=True)
    _write_geojson(out / "records.geojson", keep)
    return ("ok" if keep else "empty"), sources


def grab_firms(fire, out, ctx):
    """FIRMS area API, 5-day calls; NRT fills the days the standard product has not reached (2024+)."""
    import pandas as pd
    from live import firms
    frames, sources = {}, []

    def call(src, day, k):
        ctx.throttle(0.15)                              # 5000 transactions / 10 min per key
        df = firms.get(src, fire["bounds"], k, day)
        bbox = ",".join(f"{v:.4f}" for v in fire["bounds"])
        sources.append(redact(firms.API.format(key="KEY", src=src, bbox=bbox, days=k) + f"/{day}"))
        if len(df):
            frames.setdefault(src, []).append(df)

    for src, day, k in firms_calls(fire):
        call(src, day, k)
    ds = days(fire)
    if int(fire["year"]) >= 2024:
        sp = [pd.Timestamp(d["acq_date"].max()) for fs in frames.values() for d in fs]
        start = (max(sp) + pd.Timedelta(days=1)) if sp else ds[0]
        while start <= ds[-1]:
            k = min(5, (ds[-1] - start).days + 1)
            for src in firms.NRT:
                call(src, start.date().isoformat(), k)
            start += pd.Timedelta(days=k)
    n = 0
    for src, fs in frames.items():
        df = pd.concat(fs, ignore_index=True)
        n += len(df)
        with gzip.open(out / f"{src}.csv.gz", "wt") as fh:
            df.to_csv(fh, index=False)
    return ("ok" if n else "empty"), sources


def grab_gridmet(fire, out, ctx):
    import xarray as xr
    import weather_features as WF
    w, s, e, n = fire["bounds"]
    ds = days(fire)
    pad = WF.PAD_DEG
    parts, sources = [], []
    for short, var in WF.GM_VARS.items():
        url = WF.GRIDMET.format(v=short)
        src = ctx.memo(("gm", short), lambda u=url: xr.open_dataset(u, decode_times=True))
        sub = src[[var]].sel(lat=slice(n + pad, s - pad), lon=slice(w - pad, e + pad), day=slice(ds[0], ds[-1]))
        for t in range(5):
            try:
                parts.append(sub.load())
                break
            except Exception as ex:  # noqa: BLE001
                if t == 4:
                    raise
                print(f"    gridmet {short} retry ({str(ex)[:100]})", flush=True)
                time.sleep(5 * (t + 1))
        sources.append(f"{url}?{var}[day={ds[0].date()}..{ds[-1].date()}][lat={s - pad:.4f}..{n + pad:.4f}][lon={w - pad:.4f}..{e + pad:.4f}]")
    m = xr.merge(parts, compat="override", combine_attrs="drop_conflicts")
    m.attrs.update(source="GRIDMET (University of Idaho), THREDDS OPeNDAP aggregation", fire_id=fire["fire_id"])
    m.to_netcdf(out / "gridmet.nc", encoding={v: {"zlib": True, "complevel": 4} for v in m.data_vars})
    return "ok", sources


def _ics_tables(ctx):
    """(sitrep DataFrame, incident DataFrame, processed incident table for matching), loaded once."""
    def load():
        import zipfile
        import pandas as pd
        import suppression_layers as SL
        d = ctx.cache
        _, inc_proc, _ = SL.load_ics209(d)          # downloads ics209/ics209-plus-wildfire.zip once
        z = zipfile.ZipFile(d / "ics209" / "ics209-plus-wildfire.zip")
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        usable = [n for n in csvs if not re.search(r"(delet|complex|tract|county|cbg|block|link)", n.lower())]
        sit = next(n for n in usable if "sitrep" in n.lower())
        names = {c.upper() for c in SL.ICS_INCIDENT_COLS["name"]}
        inc = next(n for n in sorted(usable, key=lambda n: ("incident" not in n.lower(), n)) if n != sit and
                   "INCIDENT_ID" in {c.upper() for c in pd.read_csv(z.open(n), nrows=0).columns} and
                   names & {c.upper() for c in pd.read_csv(z.open(n), nrows=0).columns})
        return (pd.read_csv(z.open(sit), dtype=str, keep_default_na=False),
                pd.read_csv(z.open(inc), dtype=str, keep_default_na=False), inc_proc, (sit, inc))
    return ctx.memo("ics", load)


def grab_ics209(fire, out, ctx):
    """Original ICS-209-PLUS rows (every column, as text) for the incidents matched to this fire."""
    import suppression_layers as SL
    sit, inc, inc_proc, (sit_name, inc_name) = _ics_tables(ctx)
    irwin, _ = fire_key(fire)
    incidents = {"irwin": {irwin} if irwin else set(), "names": {str(fire["name"]).lower()}}
    ids = SL.match_ics(inc_proc, incidents, {"bounds_lonlat": fire["bounds"]},
                       _ts(fire["first"]).tz_convert(None), _ts(fire["last"]).tz_convert(None))
    ids = {str(i) for i in ids}
    inc[inc["INCIDENT_ID"].astype(str).isin(ids)].to_csv(out / "incidents.csv", index=False)
    sit[sit["INCIDENT_ID"].astype(str).isin(ids)].to_csv(out / "sitreps.csv", index=False)
    src = [f"https://ndownloader.figshare.com/files/38766504#{inc_name}", f"https://ndownloader.figshare.com/files/38766504#{sit_name}"]
    return ("ok" if ids else "empty"), src


def grab_suppression(fire, out, ctx):
    """Every archived version of every fire line / burnout / drop in the box (all attributes, incl.
    GDB_FROM_DATE / GDB_TO_DATE), for each archive year the fire burned in."""
    import nifc_perimeters as NP
    import suppression_layers as SL
    bbox = tuple(fire["bounds"])
    lines, polys, sources = [], [], []
    for y in [y for y in _years(fire) if y >= 2018]:
        layers = ctx.memo(("supp", y), lambda y=y: SL.discover_archive_layers(ctx.sess, [y]).get(y, []))
        for url, kind in layers:
            feats = NP.query_layer(ctx.sess, url, bbox, ctx.tmp())
            sources.append(_layer_source(url, bbox))
            for f in feats:
                f = dict(f)
                f["properties"] = dict(f.get("properties") or {}, _source_layer=NP.layer_label(url))
                (lines if kind == "line" else polys).append(f)
    shutil.rmtree(ctx.tmp(), ignore_errors=True)
    _write_geojson(out / "event_lines.geojson", lines)
    _write_geojson(out / "event_polygons.geojson", polys)
    return ("ok" if lines or polys else "empty"), sources


def grab_tessera(fire, out, ctx):
    ref = {"fire_id": fire["fire_id"], "fire_year": int(fire["year"]), "embedding_year": int(fire["year"]) - 1,
           "bbox_lonlat": list(fire["bounds"]), "dataset": "GeoTessera v1.1 (128-d annual embeddings, 10 m)",
           "access": "from geotessera.store import GeoTesseraZarr; GeoTesseraZarr().iter_region(tuple(bbox_lonlat), embedding_year)",
           "links": ["https://github.com/ucam-eo/geotessera", "https://data.source.coop/tessera/tessera",
                     "https://registry.opendata.aws/tessera/"],
           "note": "References only: the 10 m embeddings for one fire box are gigabytes. The built dataset's tessera.npz "
                   "holds the 150 m block means."}
    (out / "tessera_ref.json").write_text(json.dumps(ref, indent=1))
    return "ok", ref["links"][:1]


def grab_terrain(fire, out, ctx):
    """Copernicus GLO-30 at its native 1 arc-second grid, cut to the box from the 1-degree COG tiles."""
    import rasterio
    from rasterio.merge import merge
    import build_dataset as B
    w, s, e, n = fire["bounds"]
    srcs, sources = [], []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        for lat in range(math.floor(s), math.floor(n) + 1):
            for lon in range(math.floor(w), math.floor(e) + 1):
                url = B.DEM_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="W" if lon < 0 else "E", lon=abs(lon))
                try:
                    srcs.append(rasterio.open("/vsicurl/" + url))
                    sources.append(url)
                except rasterio.errors.RasterioIOError:
                    pass                                    # ocean: no tile
        if not srcs:
            return "empty", []
        arr, tr = merge(srcs, bounds=(w, s, e, n))
        prof = srcs[0].profile.copy()
        for d in srcs:
            d.close()
    prof.update(driver="GTiff", height=arr.shape[1], width=arr.shape[2], transform=tr, count=arr.shape[0],
                compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    with rasterio.open(out / "dem_glo30.tif", "w", **prof) as dst:
        dst.write(arr)
        dst.update_tags(source="Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus 2014-2018, ESA licence",
                        tiles=";".join(sources))
    return "ok", sources


PER_FIRE = {"history": grab_history, "perimeters": grab_perimeters, "firms": grab_firms, "gridmet": grab_gridmet,
            "ics209": grab_ics209, "suppression": grab_suppression, "tessera": grab_tessera, "terrain": grab_terrain}


# ------------------------------------------------------------------------------------- group channels

class NcSeries:
    """A (time, y, x) netCDF written one time step at a time, so long fires never sit in memory.
    Values are written exactly as given (auto mask/scale off): packed GOES integers stay packed, with
    their scale_factor / add_offset / _FillValue attributes copied from the source."""

    def __init__(self, path, lat, lon, attrs, static=None):
        import netCDF4
        import numpy as np
        self.ds = netCDF4.Dataset(path, "w")
        self.ds.setncatts({k: v for k, v in attrs.items() if v is not None})
        h, w = lat.shape
        self.ds.createDimension("time", None)
        self.ds.createDimension("y", h)
        self.ds.createDimension("x", w)
        for nm, a in (("lat", lat), ("lon", lon)):
            v = self.ds.createVariable(nm, "f4", ("y", "x"), zlib=True)
            v[:] = np.asarray(a, np.float32)
        for nm, (dims, a, at) in (static or {}).items():
            v = self.ds.createVariable(nm, np.asarray(a).dtype if dims else "i4", dims)
            if dims:
                v[:] = a
            v.setncatts(at)
        t = self.ds.createVariable("time", "i8", ("time",))
        t.units, t.long_name = "nanoseconds since 1970-01-01 00:00:00 UTC", "scan start / valid time"
        self.ds.createVariable("source", str, ("time",))
        self.n = 0

    def append(self, t_ns, source, fields):
        """fields: {name: (2-D array, dtype, attrs)}"""
        for nm, (arr, dtype, attrs) in fields.items():
            if nm not in self.ds.variables:
                fill = attrs.get("_FillValue")
                v = self.ds.createVariable(nm, dtype, ("time", "y", "x"), zlib=True, complevel=4,
                                           fill_value=fill if fill is not None else None)
                v.set_auto_maskandscale(False)
                v.setncatts({k: a for k, a in attrs.items() if k != "_FillValue"})
            self.ds.variables[nm][self.n] = arr
        self.ds.variables["time"][self.n] = t_ns
        self.ds.variables["source"][self.n] = source
        self.n += 1

    def close(self):
        self.ds.close()


def _chunks(fires, size):
    fires = sorted(fires, key=lambda f: span(f)[0])
    for i in range(0, len(fires), size):
        yield fires[i:i + size]


def _window(lat, lon, bounds, pad=0.0):
    import numpy as np
    w, s, e, n = bounds
    with np.errstate(invalid="ignore"):
        m = (lat >= s - pad) & (lat <= n + pad) & (lon >= w - pad) & (lon <= e + pad)
    r, c = np.where(m)
    if not len(r):
        return None
    return r.min(), r.max() + 1, c.min(), c.max() + 1


def _windowed_map(ex, fn, keys, ahead):
    """Like ex.map, but never more than `ahead` results waiting (bounds memory when one fetch is slow)."""
    from collections import deque
    q, it = deque(), iter(keys)
    for k in it:
        q.append(ex.submit(fn, k))
        if len(q) >= ahead:
            break
    while q:
        yield q.popleft().result()
        for k in it:
            q.append(ex.submit(fn, k))
            break


def grab_goes_many(fires, ctx, chunk=25):
    """Per fire and satellite: the scan nearest each hour, cropped to the box on the ABI fixed grid, with
    the fire product's own variables (Mask, Power, Area, Temp, DQF) left packed as in the source file."""
    import netCDF4
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    import build_multiyear as BM

    def fetch(item):
        t, sat, bucket = item
        prefix = f"ABI-L2-FDCC/{t.year}/{t.dayofyear:03d}/{t.hour:02d}/"
        for attempt in range(3):
            try:
                r = ctx.sess.get(f"https://{bucket}.s3.amazonaws.com/", params={"list-type": 2, "prefix": prefix}, timeout=60)
                r.raise_for_status()
                keys = re.findall(r"<Key>([^<]+)</Key>", r.text)
                if not keys:
                    return item, None, None
                key = min(keys, key=lambda k: abs(int(k.split("_s")[1][9:11]) - t.minute))
                rr = ctx.sess.get(f"https://{bucket}.s3.amazonaws.com/{key}", timeout=120)
                rr.raise_for_status()
                return item, key, rr.content
            except Exception as ex:  # noqa: BLE001
                if attempt == 2:
                    print(f"    GOES {sat} {t}: {str(ex)[:120]}", flush=True)
                    return item, None, None
                time.sleep(3 * (attempt + 1))

    grids = {}
    for group in _chunks(fires, chunk):
        if ctx.expired():
            return
        need, left = {}, {}
        for f in group:
            hrs = goes_hours(f)
            left[f["fire_id"]] = sum(len(s) for _, s in hrs)
            for t, sats in hrs:
                for sat, bucket in sats:
                    need.setdefault((t, sat, bucket), []).append(f)
        series, scans = {}, {f["fire_id"]: {} for f in group}
        stopped = False
        with ThreadPoolExecutor(ctx.threads) as ex:
            for (t, sat, bucket), key, data in _windowed_map(ex, fetch, sorted(need), 4 * ctx.threads):
                if data is not None:
                    ds = netCDF4.Dataset("mem", memory=data)
                    try:
                        shape = ds.variables["Mask"].shape
                        if (sat, shape) not in grids:
                            grids[(sat, shape)] = BM._goes_latlon(ds)
                        lat, lon = grids[(sat, shape)]
                        raw = {}
                        for nm in GOES_VARS:
                            if nm in ds.variables:
                                v = ds.variables[nm]
                                v.set_auto_maskandscale(False)
                                raw[nm] = (v, v.dtype, {a: v.getncattr(a) for a in v.ncattrs()})
                        for f in need[(t, sat, bucket)]:
                            fid = f["fire_id"]
                            k = (fid, sat)
                            if k not in series:
                                win = _window(lat, lon, f["bounds"])
                                if win is None:
                                    series[k] = None
                                else:
                                    r0, r1, c0, c1 = win
                                    xv, yv = ds.variables["x"][c0:c1], ds.variables["y"][r0:r1]
                                    proj = ds.variables["goes_imager_projection"]
                                    static = {"x": (("x",), xv.astype("f8"), {"units": "rad", "long_name": "ABI fixed grid x"}),
                                              "y": (("y",), yv.astype("f8"), {"units": "rad", "long_name": "ABI fixed grid y"}),
                                              "goes_imager_projection": ((), 0, {a: proj.getncattr(a) for a in proj.ncattrs()})}
                                    path = ctx.stage(fid) / f"goes_{sat}.nc"
                                    s = NcSeries(path, lat[r0:r1, c0:c1], lon[r0:r1, c0:c1],
                                                 {"title": f"GOES ABI L2 FDCC ({sat}) cropped to {fid}", "fire_id": fid,
                                                  "source": f"s3://{bucket}/ABI-L2-FDCC/", "window_rows_cols": f"{r0}:{r1},{c0}:{c1}"},
                                                 static)
                                    series[k] = (s, win)
                            if series[k] is not None:
                                s, (r0, r1, c0, c1) = series[k]
                                t_scan = pd.to_datetime(key.split("_s")[1][:13], format="%Y%j%H%M%S", utc=True).value
                                s.append(t_scan, f"s3://{bucket}/{key}",
                                         {nm: (v[r0:r1, c0:c1], dt, at) for nm, (v, dt, at) in raw.items()})
                                scans[fid][bucket] = scans[fid].get(bucket, 0) + 1
                    finally:
                        ds.close()
                for f in need[(t, sat, bucket)]:
                    fid = f["fire_id"]
                    left[fid] -= 1
                    if left[fid] == 0:
                        for (ff, _), v in list(series.items()):
                            if ff == fid and v is not None:
                                v[0].close()
                        yield f, ("ok" if scans[fid] else "empty"), \
                            [f"s3://{b}/ABI-L2-FDCC/ ({n} scans)" for b, n in sorted(scans[fid].items())], None
                if ctx.expired():
                    stopped = True
                    break
        if stopped:
            for (fid, _), v in series.items():
                if v is not None and left.get(fid, 0) > 0:
                    v[0].close()
                    ctx.drop_stage(fid)
            return


def grab_hrrr_many(fires, ctx, chunk=25, step_h=3):
    """gust, u10, v10, rh2 decoded from each HRRR file's own GRIB2 messages (fetched by .idx byte range),
    cropped to the fire on HRRR's native 3 km Lambert grid: analyses (f00) every step_h hours and the 12z
    run's f09 forecast each day."""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    import weather_features as WF
    lat, lon = ctx.memo("hrrr_grid", lambda: WF.hrrr_latlon(ctx.sess))
    names = [n for n, _ in WF.HRRR_FIELDS]

    def fetch(key):
        t, f = key
        url = WF.HRRR.format(d=t.strftime("%Y%m%d"), h=t.hour, f=f)
        return key, url, WF.hrrr_fetch(ctx.sess, url)

    for group in _chunks(fires, chunk):
        if ctx.expired():
            return
        need, left, wins, series, got_n = {}, {}, {}, {}, {}
        for f in group:
            fid = f["fire_id"]
            ana, fc = hrrr_times(f, step_h)
            ana = [t for t in ana if t >= _ts(HRRR_START)]
            fc = [t for t in fc if t >= _ts(HRRR_START)]
            wins[fid] = WF.hrrr_window(lat, lon, f["bounds"])
            left[fid], got_n[fid] = len(ana) + len(fc), 0
            for t in ana:
                need.setdefault((t, 0), []).append((f, "ana"))
            for t in fc:
                need.setdefault((t, 9), []).append((f, "fc"))
        stopped = False
        with ThreadPoolExecutor(ctx.threads) as ex:
            for (t, fh), url, got in _windowed_map(ex, fetch, sorted(need), 2 * ctx.threads):
                for f, which in need[(t, fh)]:
                    fid = f["fire_id"]
                    win = wins[fid]
                    if got and win is not None:
                        r0, r1, c0, c1 = win
                        k = (fid, which)
                        if k not in series:
                            series[k] = NcSeries(ctx.stage(fid) / f"hrrr_{which}.nc", lat[r0:r1, c0:c1], lon[r0:r1, c0:c1],
                                                 {"title": f"HRRR {'analyses (f00)' if which == 'ana' else '12z run, 9 h forecast'} cropped to {fid}",
                                                  "fire_id": fid, "source": "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/",
                                                  "fields": "GUST surface; UGRD/VGRD 10 m; RH 2 m",
                                                  "window_rows_cols": f"{r0}:{r1},{c0}:{c1}"})
                        valid = t + pd.Timedelta(hours=fh)
                        series[k].append(valid.value, url, {n: (got[n][r0:r1, c0:c1], "f4", {}) for n in names if n in got})
                        got_n[fid] += 1
                    left[fid] -= 1
                    if left[fid] == 0:
                        for w in ("ana", "fc"):
                            if (fid, w) in series:
                                series.pop((fid, w)).close()
                        yield f, ("ok" if got_n[fid] else "empty"), [f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/ ({got_n[fid]} files)"], None
                if ctx.expired():
                    stopped = True
                    break
        if stopped:
            for (fid, _), s in series.items():
                s.close()
                ctx.drop_stage(fid)
            return


def grab_landfire_many(fires, ctx, parallel=3):
    """LFPS jobs, a few at a time: FBFM40, CC, CH, CBH, CBD of the newest LANDFIRE predating the fire, on
    LANDFIRE's own 30 m grid and projection, kept as the zip LFPS delivers."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    import landfire_features as L

    def job(f):
        v = landfire_version(int(f["year"]))
        layers = [f"{v}_{x}" for x in ("FBFM40", "CC", "CH", "CBH", "CBD")]
        L.fetch(f["bounds"], ctx.stage(f["fire_id"]) / f"landfire_{v}.zip", layers=layers, native=True)
        w, s, e, n = f["bounds"]
        return redact(f"{L.LFPS}/submit?Email=x&Layer_List={';'.join(layers)}&Area_of_Interest={w:.4f} {s:.4f} {e:.4f} {n:.4f}")

    it = iter(fires)
    with ThreadPoolExecutor(parallel) as ex:
        running = {}
        while True:
            while len(running) < parallel and not ctx.expired():
                f = next(it, None)
                if f is None:
                    break
                running[ex.submit(job, f)] = f
            if not running:
                return
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                f = running.pop(fut)
                try:
                    yield f, "ok", [fut.result()], None
                except Exception as ex_:  # noqa: BLE001
                    yield f, "error", [], str(ex_)[:300]


def grab_osm_many(fires, ctx):
    """Full (unfiltered) OpenStreetMap extract of each box from the Geofabrik snapshot of 1 January of the
    fire's year. Fires are grouped by (region, year) so each ~2-3 GB snapshot downloads once, then it is
    deleted before the next one."""
    import human_features as HF
    groups = {}
    for f in fires:
        groups.setdefault((HF.region_file({"bounds_lonlat": f["bounds"]}), int(f["year"])), []).append(f)
    for (region, year), group in sorted(groups.items()):
        if ctx.expired():
            return
        tag = f"{str(year)[2:]}0101"
        url = f"{HF.GEOFABRIK}/{region}-{tag}.osm.pbf"
        snap = ctx.cache / f"{region.replace('/', '_')}-{tag}.osm.pbf"
        try:
            if not snap.exists():
                print(f"  downloading {url}", flush=True)
                if shutil.which("aria2c"):
                    subprocess.run(["aria2c", "-q", "-c", "--file-allocation=none", "-x", "8", "-s", "8", "-k", "5M",
                                    "--max-tries=20", "--retry-wait=10", "-d", str(snap.parent), "-o", snap.name, url], check=True)
                else:
                    with ctx.sess.get(url, stream=True, timeout=1800) as r:
                        r.raise_for_status()
                        with open(snap, "wb") as fh:
                            for b in r.iter_content(2 ** 22):
                                fh.write(b)
        except Exception as ex:  # noqa: BLE001
            for f in group:
                yield f, "error", [], f"snapshot {url}: {str(ex)[:200]}"
            continue
        for f in group:
            if ctx.expired():
                break
            w, s, e, n = f["bounds"]
            dst = ctx.stage(f["fire_id"]) / f"osm_{year}-01-01.osm.pbf"
            r = subprocess.run(["osmium", "extract", "--overwrite", "-s", "smart", "-b", f"{w},{s},{e},{n}", str(snap), "-o", str(dst)],
                               capture_output=True, text=True)
            if r.returncode:
                yield f, "error", [], r.stderr[-300:]
            else:
                yield f, "ok", [url], None
        snap.unlink(missing_ok=True)


GROUP_FUNCS = {"goes": grab_goes_many, "hrrr": grab_hrrr_many, "landfire": grab_landfire_many, "osm": grab_osm_many}
