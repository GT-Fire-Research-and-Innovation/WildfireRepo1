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
  gridmet      GRIDMET daily, 9 variables (THREDDS NetCDF Subset Service)        gridmet_<var>.nc
  ics209       ICS-209-PLUS incident + sitrep rows (fires <= 2020)              incidents.csv, sitreps.csv
  suppression  NIFS / ODA EventLine + EventPolygon, every archived version (2018+) event_lines.geojson, event_polygons.geojson
  tessera      GeoTessera v1.1 reference only (no embedding data)                tessera_ref.json
  terrain      Copernicus GLO-30 DEM, native 1 arc-second window                 dem_glo30.tif
  landfire     LANDFIRE image services, native 30 m EPSG:5070 (fires 2017+)      landfire_<version>_<layer>.tif
  goes         GOES-16/17/18/19 ABI L2 FDCC, hourly scan, fixed-grid crop        goes_<sat>.nc
  hrrr         HRRR gust/u10/v10/rh2, 3-hourly analyses + 12z+9h forecasts       hrrr_ana.nc, hrrr_fc.nc
  osm          OpenStreetMap 1 January snapshot of the fire's year, full extract osm_<YYYY>-01-01.osm.pbf

Per-fire channels: grab_<channel>(fire, out_dir, ctx) -> (status, sources), status "ok" or "empty".
Group channels (goes, hrrr, osm) share downloads across fires: grab_<channel>_many(fires, ctx)
yields (fire, status, sources, error). Secrets (FIRMS map key, HF token) never reach the manifest.
"""
import gzip
import hashlib
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
GROUP = {"goes", "hrrr", "osm"}
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


SECRET_NAMES = {"FIRMS_MAP_KEY": "{MAP_KEY}", "LANDFIRE_EMAIL": "{EMAIL}", "HF_TOKEN": "{HF_TOKEN}"}


def _secret_values():
    """{value: placeholder} for every secret this process can see (environment, .env, ~/.config/wildfire/env)."""
    vals = {}
    for name, ph in SECRET_NAMES.items():
        v = os.environ.get(name)
        if not v:
            for p in (Path(".env"), Path.home() / ".config" / "wildfire" / "env"):
                try:
                    for line in p.read_text().splitlines():
                        if line.startswith(name + "="):
                            v = line.split("=", 1)[1].split("#")[0].strip()
                except OSError:
                    pass
        if v and len(v) >= 4:
            from urllib.parse import quote, quote_plus
            for form in (v, quote(v, safe=""), quote_plus(v)):     # as typed, and as it appears inside URLs
                vals[form] = ph
    return vals


def redact(text):
    """Text safe to publish (manifests, logs): no FIRMS map key, LANDFIRE email or HF token, whether they
    appear as URL parts (also inside exception messages) or literally."""
    text = str(text or "")
    for v, ph in _secret_values().items():
        text = text.replace(v, ph)
    text = re.sub(r"(/api/area/csv/)[^/\s]+/", r"\1{MAP_KEY}/", text)
    text = re.sub(r"([?&])Email=[^&\s]*&?", r"\1", text)
    return re.sub(r"[?&]+(?=\s|$)", "", text)


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
    return arcgis_query_url(url, bbox)


def arcgis_query_url(layer_url, bbox, where="1=1", fields="*"):
    """Clickable ArcGIS query returning the box's features as GeoJSON (what the channel downloaded)."""
    from urllib.parse import urlencode
    q = {"where": where, "geometry": ",".join(f"{b:.5f}" for b in bbox), "geometryType": "esriGeometryEnvelope",
         "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": fields, "outSR": 4326, "f": "geojson"}
    return f"{layer_url}/query?{urlencode(q)}"


def _years(fire):
    t0, t1 = span(fire)
    return list(range(_ts(fire["first"]).year, t1.year + 1))


def _arcgis_call(sess, layer_url, params, post=False, tries=6):
    """One ArcGIS query, retried; ArcGIS reports errors with HTTP 200 and an 'error' body, raised here."""
    for attempt in range(tries):
        try:
            r = (sess.post(f"{layer_url}/query", data=params, timeout=300) if post
                 else sess.get(f"{layer_url}/query", params=params, timeout=180))
            js = r.json()
            if "error" in js:
                raise RuntimeError(js["error"])
            return js
        except (ValueError, RuntimeError, OSError) as ex:
            if attempt == tries - 1:
                raise
            print(f"    retry {attempt + 1}: {str(ex)[:120]}", flush=True)
            time.sleep(5 * 2 ** attempt)


def arcgis_box_features(sess, layer_url, bbox, where="1=1", chunk=200):
    """Every feature of a layer intersecting bbox (lon/lat), as GeoJSON, all fields, full geometry.
    Pages by OBJECTID (list the ids, then fetch them in chunks), not by resultOffset: offset paging on the
    2018 NIFS archive repeats records and silently misses others (California, 2018 EventLine: 137,930 rows
    returned, 92,926 unique). Raises if any listed id does not come back."""
    q = {"where": where, "geometry": ",".join(f"{b}" for b in bbox), "geometryType": "esriGeometryEnvelope",
         "inSR": 4326, "spatialRel": "esriSpatialRelIntersects"}
    ids = sorted(set(_arcgis_call(sess, layer_url, dict(q, returnIdsOnly="true", f="json")).get("objectIds") or []))
    feats = []
    for i in range(0, len(ids), chunk):
        js = _arcgis_call(sess, layer_url, {"objectIds": ",".join(map(str, ids[i:i + chunk])), "outFields": "*",
                                            "returnGeometry": "true", "outSR": 4326, "f": "geojson"}, post=True)
        feats += js.get("features", [])
    if len(feats) != len(ids):
        raise RuntimeError(f"{layer_url}: listed {len(ids)} features, received {len(feats)}")
    return feats


def _box_query(ctx, url, bbox):
    """Archive layer features in the box: ArcGIS services by id, 2020/2021 geodatabases from the local copy."""
    import nifc_perimeters as NP
    if url.startswith("fgdb://"):
        return NP.query_layer(ctx.sess, url, bbox, ctx.tmp())
    return arcgis_box_features(ctx.sess, url, bbox)


# ------------------------------------------------------------------------------------- per-fire channels

def grab_history(fire, out, ctx):
    import build_multiyear as BM
    where = f"FIRE_YEAR_INT < {int(fire['year'])}"
    feats = arcgis_box_features(ctx.sess, BM.HIST_URL, fire["bounds"], where=where, chunk=100)
    _write_geojson(out / "history.geojson", feats)
    return ("ok" if feats else "empty"), [arcgis_query_url(BM.HIST_URL, fire["bounds"], where=where)]


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
            feats = _box_query(ctx, url, bbox)
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


def gridmet_url(short, var, bounds, d0, d1, pad):
    """GRIDMET subset from the THREDDS NetCDF Subset Service: plain HTTPS returning a netCDF file (no
    OPeNDAP client needed; the OPeNDAP route failed on Kaggle). Same box (+pad) and days as the build."""
    import weather_features as WF
    w, s, e, n = bounds
    return (WF.GRIDMET.format(v=short).replace("/dodsC/", "/ncss/grid/") +
            f"?var={var}&north={n + pad:.4f}&west={w - pad:.4f}&east={e + pad:.4f}&south={s - pad:.4f}"
            f"&horizStride=1&time_start={d0.date()}T00:00:00Z&time_end={d1.date()}T00:00:00Z&accept=netcdf")


def http_get(ctx, url, tries=5, timeout=300):
    """Body of a GET, retried on network errors and 429 / 5xx; other HTTP errors raise with the reply text."""
    import requests
    for t in range(tries):
        try:
            r = ctx.sess.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            err = f"HTTP {r.status_code}"
        except requests.RequestException as ex:
            err = f"{type(ex).__name__}: {str(ex)[:120]}"
        if t == tries - 1:
            raise RuntimeError(f"{err} after {tries} tries")
        time.sleep(5 * 2 ** t)


def grab_gridmet(fire, out, ctx):
    """One netCDF per variable, exactly as the server returns it: gridmet_<var>.nc."""
    import weather_features as WF
    ds = days(fire)
    sources = []
    for short, var in WF.GM_VARS.items():
        url = gridmet_url(short, var, fire["bounds"], ds[0], ds[-1], WF.PAD_DEG)
        data = http_get(ctx, url)
        if not (data[:3] == b"CDF" or data[:4] == b"\x89HDF"):
            raise RuntimeError(f"GRIDMET {short}: reply is not netCDF: {data[:120]!r}")
        (out / f"gridmet_{short}.nc").write_bytes(data)
        sources.append(url)
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
            feats = _box_query(ctx, url, bbox)
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
                code = ctx.sess.head(url, timeout=60).status_code
                if code in (403, 404):
                    continue                                 # ocean: the archive has no tile there
                if code != 200:
                    raise RuntimeError(f"DEM tile HEAD {code}: {url}")
                try:
                    srcs.append(rasterio.open("/vsicurl/" + url))    # COG: reads only the window it needs
                except rasterio.errors.RasterioIOError:
                    tile = ctx.cache / "dem" / url.rsplit("/", 1)[1]   # no remote reads here: fetch the whole tile
                    if not tile.exists():
                        tile.parent.mkdir(parents=True, exist_ok=True)
                        tile.write_bytes(http_get(ctx, url, timeout=600))
                    srcs.append(rasterio.open(tile))
                sources.append(url)
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


LANDFIRE_IMAGE = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_{v}/{v}_{layer}_CONUS/ImageServer"
LANDFIRE_LAYERS = ("FBFM40", "CC", "CH", "CBH", "CBD")


def native_window(bounds_ll, origin, res, crs=5070):
    """Box in the service's own projection, snapped outward to its pixel edges (origin = top-left corner of
    the service extent), so the export is the stored pixels with no resampling. -> (x0, y0, x1, y1, w, h)"""
    from pyproj import Transformer
    xs0, ys0, xs1, ys1 = Transformer.from_crs(4326, crs, always_xy=True).transform_bounds(*bounds_ll, densify_pts=21)
    X0, Y1 = origin
    x0 = X0 + math.floor((xs0 - X0) / res) * res
    x1 = X0 + math.ceil((xs1 - X0) / res) * res
    y1 = Y1 - math.floor((Y1 - ys1) / res) * res
    y0 = Y1 - math.ceil((Y1 - ys0) / res) * res
    return x0, y0, x1, y1, int(round((x1 - x0) / res)), int(round((y1 - y0) / res))


def landfire_export_url(v, layer, bounds_ll, info):
    from urllib.parse import urlencode
    ext, res = info["extent"], info["pixelSizeX"]
    x0, y0, x1, y1, w, h = native_window(bounds_ll, (ext["xmin"], ext["ymax"]), res, info["spatialReference"]["wkid"])
    q = {"bbox": f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}", "bboxSR": 5070, "imageSR": 5070, "size": f"{w},{h}",
         "format": "tiff", "compression": "LZW", "pixelType": info["pixelType"], "noData": info.get("noDataValue", -9999),
         "interpolation": "RSP_NearestNeighbor", "f": "image"}
    return f"{LANDFIRE_IMAGE.format(v=v, layer=layer)}/exportImage?{urlencode(q)}"


def grab_landfire(fire, out, ctx):
    """FBFM40, CC, CH, CBH, CBD of the newest LANDFIRE version before the fire, from LANDFIRE's own image
    services: native 30 m pixels in EPSG:5070, nearest neighbour on the stored grid (no resampling), LZW
    GeoTIFF exactly as the server returns it. No email or job queue (the LFPS route needed both)."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    v = landfire_version(int(fire["year"]))

    def one(layer):
        info = ctx.memo(("lf", v, layer), lambda: ctx.sess.get(LANDFIRE_IMAGE.format(v=v, layer=layer),
                                                                params={"f": "json"}, timeout=60).json())
        if "extent" not in info:
            raise RuntimeError(f"LANDFIRE {v}_{layer}: service info unavailable: {str(info)[:150]}")
        url = landfire_export_url(v, layer, fire["bounds"], info)
        data = http_get(ctx, url, timeout=600)
        if data[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00"):
            raise RuntimeError(f"LANDFIRE {v}_{layer}: reply is not a GeoTIFF: {data[:150]!r}")
        (out / f"landfire_{v}_{layer}.tif").write_bytes(data)
        return url

    try:
        with ThreadPoolExecutor(len(LANDFIRE_LAYERS)) as ex:
            return "ok", list(ex.map(one, LANDFIRE_LAYERS))
    except requests.RequestException as ex:
        raise RuntimeError(f"LANDFIRE {v}: {ex}") from ex


PER_FIRE = {"history": grab_history, "perimeters": grab_perimeters, "firms": grab_firms, "gridmet": grab_gridmet,
            "ics209": grab_ics209, "suppression": grab_suppression, "tessera": grab_tessera, "terrain": grab_terrain,
            "landfire": grab_landfire}


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


def _open_nc_bytes(netCDF4, data, ctx):
    """Open a downloaded netCDF from memory; if this netCDF4 build cannot (some preinstalled builds), from a
    temporary file instead. -> (dataset, temp path or None)"""
    try:
        return netCDF4.Dataset("mem", memory=data), None
    except (OSError, ValueError, RuntimeError):
        import tempfile
        fd, p = tempfile.mkstemp(suffix=".nc", dir=ctx.tmp())
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return netCDF4.Dataset(p), Path(p)


def _s3_get(ctx, bucket, key):
    """One object of a public S3 bucket, checked against its ETag (the MD5 of a single-part upload, which
    every GOES file is): bytes cut short or garbled on the way raise here, so the caller downloads again,
    instead of failing later inside the netCDF reader."""
    rr = ctx.sess.get(f"https://{bucket}.s3.amazonaws.com/{key}", timeout=120)
    rr.raise_for_status()
    data = rr.content
    etag = rr.headers.get("ETag", "").strip('"')
    if re.fullmatch(r"[0-9a-f]{32}", etag) and hashlib.md5(data).hexdigest() != etag:
        raise OSError(f"{key.rsplit('/', 1)[-1]}: {len(data)} bytes do not match the object's MD5 (ETag)")
    return data


def _read_goes_scan(netCDF4, data, ctx, grids, sat):
    """Everything grab_goes_many takes from one FDCC file, read in full and the file closed again:
    {"vars": {name: (array, dtype, attrs)} packed as in the file, "x", "y", "proj": the projection's
    attributes, "latlon": (lat, lon) of its grid}. A file with a damaged compressed chunk opens fine and fails
    only when that chunk is read ("RuntimeError: NetCDF: HDF error"); reading it all here makes such a file
    fail as a whole, before any fire's series gets part of the scan."""
    import build_multiyear as BM
    ds, tmp = _open_nc_bytes(netCDF4, data, ctx)
    try:
        shape = ds.variables["Mask"].shape
        if (sat, shape) not in grids:
            grids[(sat, shape)] = BM._goes_latlon(ds)
        proj = ds.variables["goes_imager_projection"]
        scan = {"vars": {}, "latlon": grids[(sat, shape)], "x": ds.variables["x"][:], "y": ds.variables["y"][:],
                "proj": {a: proj.getncattr(a) for a in proj.ncattrs()}}
        for nm in GOES_VARS:
            if nm in ds.variables:
                v = ds.variables[nm]
                v.set_auto_maskandscale(False)
                scan["vars"][nm] = (v[:], v.dtype, {a: v.getncattr(a) for a in v.ncattrs()})
        return scan
    finally:
        ds.close()
        if tmp:
            tmp.unlink(missing_ok=True)


def _goes_scan(netCDF4, ctx, grids, sat, bucket, keys, data, tries=3):
    """(scan, key) of the nearest readable scan of the hour. keys: the hour's scans, nearest first; data: the
    first one's bytes (None if its download failed). A file that does not read, i.e. one stored corrupt in
    the bucket (its bytes match the ETag yet a chunk does not decompress), is logged and replaced by the next
    nearest scan, up to `tries` files. (None, None) when none reads: the hour is left out."""
    for i, key in enumerate(keys[:tries]):
        try:
            if i:
                data = _s3_get(ctx, bucket, key)
            if data is None:
                break
            return _read_goes_scan(netCDF4, data, ctx, grids, sat), key
        except (OSError, RuntimeError, KeyError, ValueError, IndexError) as ex:
            then = "trying the next scan of the hour" if i + 1 < min(tries, len(keys)) else "hour left out"
            print(f"    GOES {sat} {key.rsplit('/', 1)[-1]}: unreadable ({type(ex).__name__}: {str(ex)[:100]}), {then}",
                  flush=True)
    return None, None


def grab_goes_many(fires, ctx, chunk=25):
    """Per fire and satellite: the scan nearest each hour, cropped to the box on the ABI fixed grid, with
    the fire product's own variables (Mask, Power, Area, Temp, DQF) left packed as in the source file.
    A scan that is corrupt in the bucket is replaced by the next nearest of the same hour (_goes_scan)."""
    import netCDF4
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    def fetch(item):
        """-> (item, the hour's keys nearest first, the nearest one's bytes); ([], None) when nothing came."""
        t, sat, bucket = item
        prefix = f"ABI-L2-FDCC/{t.year}/{t.dayofyear:03d}/{t.hour:02d}/"
        for attempt in range(3):
            try:
                r = ctx.sess.get(f"https://{bucket}.s3.amazonaws.com/", params={"list-type": 2, "prefix": prefix}, timeout=60)
                r.raise_for_status()
                keys = sorted(re.findall(r"<Key>([^<]+)</Key>", r.text),
                              key=lambda k: abs(int(k.split("_s")[1][9:11]) - t.minute))
                if not keys:
                    return item, [], None
                return item, keys, _s3_get(ctx, bucket, keys[0])
            except Exception as ex:  # noqa: BLE001
                if attempt == 2:
                    print(f"    GOES {sat} {t}: {str(ex)[:120]}", flush=True)
                    return item, [], None
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
            for (t, sat, bucket), keys, data in _windowed_map(ex, fetch, sorted(need), 4 * ctx.threads):
                scan, key = _goes_scan(netCDF4, ctx, grids, sat, bucket, keys, data)
                if scan is not None:
                    lat, lon = scan["latlon"]
                    for f in need[(t, sat, bucket)]:
                        fid = f["fire_id"]
                        k = (fid, sat)
                        if k not in series:
                            win = _window(lat, lon, f["bounds"])
                            if win is None:
                                series[k] = None
                            else:
                                r0, r1, c0, c1 = win
                                static = {"x": (("x",), scan["x"][c0:c1].astype("f8"), {"units": "rad", "long_name": "ABI fixed grid x"}),
                                          "y": (("y",), scan["y"][r0:r1].astype("f8"), {"units": "rad", "long_name": "ABI fixed grid y"}),
                                          "goes_imager_projection": ((), 0, scan["proj"])}
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
                                     {nm: (a[r0:r1, c0:c1], dt, at) for nm, (a, dt, at) in scan["vars"].items()})
                            day = f"ABI-L2-FDCC/{t.year}/{t.dayofyear:03d}/"
                            scans[fid][(bucket, day)] = scans[fid].get((bucket, day), 0) + 1
                for f in need[(t, sat, bucket)]:
                    fid = f["fire_id"]
                    left[fid] -= 1
                    if left[fid] == 0:
                        for (ff, _), v in list(series.items()):
                            if ff == fid and v is not None:
                                v[0].close()
                        # one listing link per satellite-day (the exact scan of each hour is in the file's `source`)
                        yield f, ("ok" if scans[fid] else "empty"), \
                            [f"https://{b}.s3.amazonaws.com/?list-type=2&prefix={d}" for b, d in sorted(scans[fid])], None
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
            left[fid], got_n[fid] = len(ana) + len(fc), []
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
                        got_n[fid].append(url)
                    left[fid] -= 1
                    if left[fid] == 0:
                        for w in ("ana", "fc"):
                            if (fid, w) in series:
                                series.pop((fid, w)).close()
                        yield f, ("ok" if got_n[fid] else "empty"), sorted(got_n[fid]), None
                if ctx.expired():
                    stopped = True
                    break
        if stopped:
            for (fid, _), s in series.items():
                s.close()
                ctx.drop_stage(fid)
            return


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


GROUP_FUNCS = {"goes": grab_goes_many, "hrrr": grab_hrrr_many, "osm": grab_osm_many}
