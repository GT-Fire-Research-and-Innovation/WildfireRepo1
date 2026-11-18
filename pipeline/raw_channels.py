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
  gridmet      GRIDMET daily, 13 variables incl. pr/tmmn/sph/pdsi (THREDDS NCSS) gridmet_<var>.nc
  ics209       ICS-209-PLUS incident + sitrep rows (fires <= 2020)              incidents.csv, sitreps.csv
  suppression  NIFS / ODA EventLine + EventPolygon, every archived version (2018+) event_lines.geojson, event_polygons.geojson
  tessera      GeoTessera v1.1 reference only (no embedding data)                tessera_ref.json
  terrain      Copernicus GLO-30 DEM, native 1 arc-second window                 dem_glo30.tif
  landfire     LANDFIRE image services, native 30 m EPSG:5070 (fires 2017+)      landfire_<version>_<layer>.tif
  lulc         Esri / Impact Observatory 10 m annual land cover (fires 2017+)    io_lulc_<year>_<tile>.tif
  landcover    MODIS MCD12Q1 annual land cover, sinusoidal 463 m (Earthdata)     mcd12q1_<year>.nc
  vegetation   VIIRS VNP13A1 16-day vegetation indices, sinusoidal 463 m (Earthdata) vnp13a1.nc
  goes         GOES-16/17/18/19 ABI L2 FDCC, hourly scan, fixed-grid crop        goes_<sat>.nc
  hrrr         HRRR gust/u10/v10/rh2, 3-hourly analyses + 12z+9h forecasts       hrrr_ana.nc, hrrr_fc.nc
  hrrr_next    HRRR 12z run every 3 h out to 36 h: + t2/q2/apcp                  hrrr_next.nc
  osm          OpenStreetMap 1 January snapshot of the fire's year, full extract osm_<YYYY>-01-01.osm.pbf
  viirs        VIIRS L1B swaths day + night: I1-I5, M11, geolocation (Earthdata) <SAT>_<date>T<HHMM>_<day|night>.nc

lulc, landcover, vegetation, hrrr_next and viirs bring in what TS-SatFire (Zhao et al. 2024) has and the other
channels lacked: its VIIRS_Day / VIIRS_Night bands, its FirePred bands (NDVI, EVI2, gridMET pr / tmmn / sph /
pdsi, MODIS LC_Type1, GFS-style next-day forecasts) and its ESRI_LULC map.

Per-fire channels: grab_<channel>(fire, out_dir, ctx) -> (status, sources), status "ok" or "empty".
Group channels (goes, hrrr, hrrr_next, osm, viirs) share downloads across fires: grab_<channel>_many(fires, ctx)
yields (fire, status, sources, error). Secrets (FIRMS map key, Earthdata token, HF token) never reach the manifest.
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
            "landfire", "lulc", "landcover", "vegetation", "goes", "hrrr", "hrrr_next", "osm", "viirs"]
GROUP = {"goes", "hrrr", "hrrr_next", "osm", "viirs"}
EARTHDATA_CHANNELS = {"landcover", "vegetation", "viirs"}        # need EARTHDATA_TOKEN
GOES_VARS = ("Mask", "Power", "Area", "Temp", "DQF")
HRRR_START = "2014-08-01"            # first HRRR files on AWS used by the build
GOES_START = "2017-06-01"
TESSERA_FIRST_YEAR = 2017
LULC_FIRST_YEAR, LULC_LAST_YEAR = 2017, 2023      # io-lulc-annual-v02 maps on Planetary Computer
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
    if channel in ("hrrr", "hrrr_next") and t1 < pd.Timestamp(HRRR_START, tz="UTC"):
        return "HRRR archive starts 2014-08"
    if channel == "lulc" and y < LULC_FIRST_YEAR:
        return "Esri 10 m land cover starts 2017"
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


HRRR_NEXT_LEADS = tuple(range(3, 37, 3))
# How far the 12z run reached: 15 h in HRRRv1, 18 h from HRRRv2 (2016-08-23), 36 h from HRRRv3 (2018-07-12).
# Leads past these are not requested (a missing file would only be skipped anyway).
HRRR_MAX_LEAD = (("2014-01-01", 15), ("2016-08-23", 18), ("2018-07-12", 36))


def hrrr_max_lead(run):
    import pandas as pd
    return [h for d, h in HRRR_MAX_LEAD if _ts(run) >= pd.Timestamp(d, tz="UTC")][-1]


def hrrr_next_times(fire):
    """[(12z run, lead hours)]: each day of span() the 12z run's forecasts every 3 h out to 36 h (the next
    day's afternoon), as far as that HRRR version ran."""
    _, fc = hrrr_times(fire)
    return [(run, h) for run in fc for h in HRRR_NEXT_LEADS if h <= hrrr_max_lead(run)]


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


SECRET_NAMES = {"FIRMS_MAP_KEY": "{MAP_KEY}", "LANDFIRE_EMAIL": "{EMAIL}", "HF_TOKEN": "{HF_TOKEN}",
                "EARTHDATA_TOKEN": "{EARTHDATA_TOKEN}"}


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
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1{EARTHDATA_TOKEN}", text)
    text = re.sub(r"(?<=[?&])(st|se|sp|sv|sr|skoid|sktid|skt|ske|sks|skv|sig|X-Amz-[A-Za-z]+)=[^&\s]*&?", "", text)  # signed URLs
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
    if channel in ("hrrr", "hrrr_next"):
        import pygrib  # noqa: F401
    if channel in EARTHDATA_CHANNELS:
        import earthdata as ED
        ED.token()                                   # SystemExit with instructions when missing
    if channel in ("vegetation", "viirs"):
        import h5py  # noqa: F401
    if channel == "landcover":
        try:
            import pyhdf.SD  # noqa: F401
        except ImportError:
            raise SystemExit("landcover reads MCD12Q1 HDF4 tiles: pip install pyhdf")


def expected_files(channel):
    """Files an 'ok' fire must list in its manifest row to count as done (None: any 'ok' row counts). When a
    channel grows new files (gridmet gained pr / tmmn / sph / pdsi), its next run fetches again just the
    fires that miss them."""
    if channel == "gridmet":
        return {f"gridmet_{v}.nc" for v in gridmet_vars()}
    return None


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


# Beyond the build's 9 (weather_features.GM_VARS): the 4 TS-SatFire FirePred takes from GRIDMET. PDSI is its
# own aggregation (gridMET drought), one value per 5-day period.
GRIDMET_EXTRA = {"pr": "precipitation_amount", "tmmn": "daily_minimum_temperature",
                 "sph": "daily_mean_specific_humidity", "pdsi": "daily_mean_palmer_drought_severity_index"}
GRIDMET_LEAD_DAYS = {"pdsi": 10}       # start earlier so the window holds the PDSI value in force before the fire


def gridmet_vars():
    import weather_features as WF
    return {**WF.GM_VARS, **GRIDMET_EXTRA}


def grab_gridmet(fire, out, ctx):
    """One netCDF per variable, exactly as the server returns it: gridmet_<var>.nc."""
    import pandas as pd
    import weather_features as WF
    ds = days(fire)
    sources = []
    for short, var in gridmet_vars().items():
        d0 = ds[0] - pd.Timedelta(days=GRIDMET_LEAD_DAYS.get(short, 0))
        url = gridmet_url(short, var, fire["bounds"], d0, ds[-1], WF.PAD_DEG)
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


# ------------------------------------------------------------------------------------- source-grid helpers

_SKIP_ATTRS = {"DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME", "_Netcdf4Dimid", "_Netcdf4Coordinates",
               "_nc3_strict", "_FillValue"}


def clean_attrs(attrs):
    """Source attributes as netCDF can store them: text decoded, HDF5 dimension bookkeeping dropped,
    one-element arrays as scalars, booleans as int8 (_FillValue is passed to createVariable instead)."""
    import numpy as np
    out = {}
    for k, v in dict(attrs).items():
        if k in _SKIP_ATTRS:
            continue
        if isinstance(v, bytes):
            v = v.decode(errors="replace")
        elif isinstance(v, (list, tuple)):
            v = np.asarray(v)
        if isinstance(v, np.ndarray):
            if v.dtype.kind in "SOU":
                v = "; ".join(x.decode(errors="replace") if isinstance(x, bytes) else str(x) for x in v.ravel())
            elif v.dtype.kind == "b":
                v = v.astype("i1")
            elif v.dtype.kind not in "iuf":
                continue
            if isinstance(v, np.ndarray) and v.size == 1:
                v = v.reshape(-1)[0]
        elif isinstance(v, np.generic) and v.dtype.kind == "b":
            v = np.int8(v)
        elif not isinstance(v, (str, int, float, np.generic)):
            continue
        out[k] = v
    return out


def fill_value(attrs, dtype):
    """The source's _FillValue in the variable's dtype, or None."""
    import numpy as np
    f = dict(attrs).get("_FillValue")
    if f is None or np.dtype(dtype).kind not in "iuf":
        return None
    return np.asarray(f).astype(dtype).reshape(-1)[0]


def _var_name(name):
    """A product's layer name as a netCDF variable name: '500 m 16 days NDVI' -> 'NDVI'."""
    name = re.sub(r"^\d+ m \d+ days ", "", str(name).split("/")[-1])
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


class SinuMosaic:
    """Layers of one global sinusoidal window (earthdata.sinu_window), filled tile by tile pixel for pixel
    (the tiles share one grid, so nothing is resampled); with nt, one slot per time step. Pixels no tile
    covers keep the layer's _FillValue (netCDF's default fill when the source has none)."""

    def __init__(self, win, nt=None):
        self.win, self.nt, self.layers = win, nt, {}

    def put(self, name, src, attrs, h, v, t=None, source_name=None):
        """src: a 2400 x 2400 dataset of tile (h, v) (h5py, pyhdf or numpy); only the window's part is read."""
        import netCDF4
        import numpy as np
        import earthdata as ED
        part = ED.tile_part(self.win, h, v)
        if part is None:
            return False
        (tr0, tr1, tc0, tc1), (or0, or1, oc0, oc1) = part
        block = np.asarray(src[tr0:tr1, tc0:tc1])
        if name not in self.layers:
            fill = fill_value(attrs, block.dtype)
            if fill is None:
                fill = np.asarray(netCDF4.default_fillvals[block.dtype.str[1:]]).astype(block.dtype)
            r0, r1, c0, c1 = self.win
            shape = (r1 - r0, c1 - c0) if self.nt is None else (self.nt, r1 - r0, c1 - c0)
            at = clean_attrs(attrs)
            if source_name and source_name != name:
                at["source_name"] = str(source_name)
            self.layers[name] = (np.full(shape, fill, block.dtype), fill, at)
        arr = self.layers[name][0]
        (arr if t is None else arr[t])[or0:or1, oc0:oc1] = block
        return True


def write_sinu_nc(path, win, layers, attrs, times_ns=None):
    """A netCDF of SinuMosaic layers on the sinusoidal grid: x / y pixel centres (m), a CF `crs` variable,
    values exactly as stored in the source with its attributes (scale_factor etc. are not applied)."""
    import netCDF4
    import earthdata as ED
    x, y = ED.sinu_coords(win)
    r0, r1, c0, c1 = win
    with netCDF4.Dataset(path, "w") as ds:
        ds.setncatts({**attrs, "sinusoidal_rows_cols": f"{r0}:{r1},{c0}:{c1}",
                      "grid": "MODIS/VIIRS sinusoidal land grid, 2400 x 2400 pixels per tile (pixel_size_m 463.3127)"})
        ds.createDimension("y", len(y))
        ds.createDimension("x", len(x))
        for nm, vals, ln in (("x", x, "projection x (pixel centre)"), ("y", y, "projection y (pixel centre)")):
            v = ds.createVariable(nm, "f8", (nm,))
            v[:] = vals
            v.setncatts({"units": "m", "long_name": ln, "standard_name": f"projection_{nm}_coordinate"})
        crs = ds.createVariable("crs", "i4")
        crs.setncatts({"grid_mapping_name": "sinusoidal", "longitude_of_central_meridian": 0.0, "false_easting": 0.0,
                       "false_northing": 0.0, "earth_radius": 6371007.181, "proj4": ED.SINU_PROJ})
        if times_ns is not None:
            ds.createDimension("time", len(times_ns))
            t = ds.createVariable("time", "i8", ("time",))
            t[:] = times_ns
            t.setncatts({"units": "nanoseconds since 1970-01-01 00:00:00 UTC", "long_name": "composite start"})
        for nm, (arr, fill, at) in layers.items():
            v = ds.createVariable(nm, arr.dtype, ("time", "y", "x") if arr.ndim == 3 else ("y", "x"),
                                  zlib=True, complevel=4, fill_value=fill)
            v.set_auto_maskandscale(False)
            v.setncatts({**at, "grid_mapping": "crs"})
            v[:] = arr


def cached_file(ctx, url, fetch, max_gb=3.0):
    """A whole remote file kept in ctx.cache/files for this run (shared by every fire that needs it); the
    least recently used files go when the folder passes max_gb."""
    d = ctx.cache / "files"
    p = d / Path(url.split("?")[0]).name
    if not p.exists():
        fetch(p)
        files = sorted((q for q in d.iterdir() if q.is_file()), key=lambda q: q.stat().st_atime)
        total = sum(q.stat().st_size for q in files)
        for q in files:
            if total <= max_gb * 1e9 or q == p:
                break
            total -= q.stat().st_size
            q.unlink(missing_ok=True)
    os.utime(p)
    return p


def pixel_window(transform, width, height, bounds):
    """(r0, r1, c0, c1) of a raster's pixels covering bounds (in the raster's CRS), snapped outward and
    clipped to the raster, or None."""
    inv = ~transform
    x0, y0, x1, y1 = bounds
    corners = [(x, y) for x in (x0, x1) for y in (y0, y1)]
    cs = [inv.a * x + inv.b * y + inv.c for x, y in corners]
    rs = [inv.d * x + inv.e * y + inv.f for x, y in corners]
    c0, c1 = max(math.floor(min(cs)), 0), min(math.ceil(max(cs)), width)
    r0, r1 = max(math.floor(min(rs)), 0), min(math.ceil(max(rs)), height)
    return (r0, r1, c0, c1) if r0 < r1 and c0 < c1 else None


# ------------------------------------------------------------------------------------- land cover, vegetation

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
LULC_COLLECTION = "io-lulc-annual-v02"


def lulc_year(year):
    """Map year for a fire: the year before it, within the maps that exist (2017 fires get the first map,
    2017, as TS-SatFire's do)."""
    return min(max(int(year) - 1, LULC_FIRST_YEAR), LULC_LAST_YEAR)


def _pc_token(ctx, collection):
    """Planetary Computer read token (anonymous, valid about an hour), renewed after 30 minutes."""
    key = ("pc_sas", collection)
    tok, t = ctx._memo.get(key, (None, 0.0))
    if tok is None or time.time() - t > 1800:
        r = ctx.sess.get(PC_SAS.format(collection=collection), timeout=60)
        r.raise_for_status()
        tok = r.json()["token"]
        ctx._memo[key] = (tok, time.time())
    return tok


def grab_lulc(fire, out, ctx):
    """Esri / Impact Observatory 10 m annual land use / land cover (io-lulc-annual-v02, the map behind
    TS-SatFire's ESRI_LULC), from Microsoft Planetary Computer: for each source tile (one UTM zone each) the
    box's pixels read from the cloud-optimized GeoTIFF on its own grid, no resampling. Class legend in the
    file's `classes` tag."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window
    yr = lulc_year(fire["year"])
    r = ctx.sess.post(f"{PC_STAC}/search", timeout=120,
                      json={"collections": [LULC_COLLECTION], "bbox": list(fire["bounds"]), "limit": 100,
                            "datetime": f"{yr}-01-01T00:00:00Z/{yr}-12-31T23:59:59Z"})
    r.raise_for_status()
    sources = []
    for it in sorted(r.json().get("features", []), key=lambda i: i["id"]):
        a = it["assets"]["data"]
        href = a["href"]
        legend = {str(x["values"][0]): x["summary"] for x in a.get("file:values", [])}
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="5"):
            with rasterio.open(f"/vsicurl/{href}?{_pc_token(ctx, LULC_COLLECTION)}") as src:
                win = pixel_window(src.transform, src.width, src.height,
                                   transform_bounds(4326, src.crs, *fire["bounds"], densify_pts=21))
                if win is None:
                    continue
                r0, r1, c0, c1 = win
                w = Window(c0, r0, c1 - c0, r1 - r0)
                arr = src.read(window=w)
                prof = {k: v for k, v in src.profile.items() if k not in ("blockxsize", "blockysize", "tiled", "interleave")}
                prof.update(driver="GTiff", width=c1 - c0, height=r1 - r0, transform=src.window_transform(w),
                            compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        tile = it["id"].rsplit("-", 1)[0]
        with rasterio.open(out / f"io_lulc_{yr}_{tile}.tif", "w", **prof) as dst:
            dst.write(arr)
            dst.update_tags(source=href, stac_item=it["id"], collection=LULC_COLLECTION, map_year=str(yr),
                            fire_year=str(fire["year"]), classes=json.dumps(legend), window_rows_cols=f"{r0}:{r1},{c0}:{c1}",
                            licence="CC BY 4.0: Impact Observatory, Microsoft and Esri")
        sources.append(href)
    return ("ok" if sources else "empty"), sources


def grab_landcover(fire, out, ctx):
    """MODIS MCD12Q1 v061 annual land cover for the year before the fire: every layer of the product
    (LC_Type1, TS-SatFire's land-cover band, LC_Type2-5, LC_Prop1-3 and their assessments, QC, LW) on the
    sinusoidal 463 m grid, the box cut from each tile and joined pixel for pixel: mcd12q1_<year>.nc. Tiles
    are LP DAAC HDF4 files (13 MB), downloaded whole once per run and shared by every fire on them."""
    import earthdata as ED
    from pyhdf.SD import SD, SDC
    yr = int(fire["year"]) - 1
    win = ED.sinu_window(fire["bounds"])
    grans = [g for g in ED.cmr_granules(ctx.sess, "MCD12Q1", f"{yr}-01-01", f"{yr}-12-31T23:59:59", fire["bounds"])
             if ED.granule_parts(g["url"])[1] == f"A{yr}001" and ED.tile_part(win, *ED.tile_hv(g["name"]))]
    tok = ED.token()
    m, sources = SinuMosaic(win), []
    for g in grans:
        h, v = ED.tile_hv(g["name"])
        path = cached_file(ctx, g["url"], lambda p, u=g["url"]: ED.download(ctx.sess, u, p, tok))
        hdf = SD(str(path), SDC.READ)
        try:
            for name in hdf.datasets():
                sds = hdf.select(name)
                try:
                    if list(sds.info()[2]) == [ED.SINU_N, ED.SINU_N]:
                        m.put(name, sds, sds.attributes(), h, v)
                finally:
                    sds.endaccess()
        finally:
            hdf.end()
        sources.append(g["url"])
    if not m.layers:
        return "empty", sources
    write_sinu_nc(out / f"mcd12q1_{yr}.nc", win, m.layers,
                  {"title": f"MODIS MCD12Q1 v061 land cover {yr} cropped to {fire['fire_id']}", "fire_id": fire["fire_id"],
                   "map_year": yr, "sources": " | ".join(sources)})
    return "ok", sources


VEG_PRODUCT = "VNP13A1"


def _h5_grids(f, n):
    """(path, dataset) of every n x n dataset in an HDF5 file."""
    import h5py
    found = []
    f.visititems(lambda p, o: found.append((p, o)) if isinstance(o, h5py.Dataset) and o.shape == (n, n) else None)
    return found


def grab_vegetation(fire, out, ctx):
    """VIIRS VNP13A1 v002 16-day vegetation indices on the sinusoidal 463 m grid: TS-SatFire's NDVI and EVI2
    bands and every other layer of the product (EVI, the reflectances, angles, composite day, VI quality,
    pixel reliability), for each composite (a new one starts every 8 days) overlapping the 16 days before
    span() through its end. Only the box's part of each tile is read (HTTP byte ranges): vnp13a1.nc,
    time = composite start."""
    import h5py
    import pandas as pd
    import earthdata as ED
    t0, t1 = span(fire)
    win = ED.sinu_window(fire["bounds"])
    grans = [g for g in ED.cmr_granules(ctx.sess, VEG_PRODUCT, t0 - pd.Timedelta(days=16), t1, fire["bounds"])
             if ED.tile_part(win, *ED.tile_hv(g["name"]))]
    dates = sorted({ED.granule_parts(g["url"])[1] for g in grans})
    if not dates:
        return "empty", []
    tok = ED.token()
    m, sources = SinuMosaic(win, nt=len(dates)), []
    for g in grans:
        k = dates.index(ED.granule_parts(g["url"])[1])
        h, v = ED.tile_hv(g["name"])
        with ED.open_remote(ctx.sess, g["url"], tok) as fo, h5py.File(fo, "r") as f:
            for path, ds in _h5_grids(f, ED.SINU_N):
                m.put(_var_name(path), ds, dict(ds.attrs), h, v, t=k, source_name=path.split("/")[-1])
        sources.append(g["url"])
    times = [ED.granule_day(next(g["url"] for g in grans if ED.granule_parts(g["url"])[1] == d)).value for d in dates]
    write_sinu_nc(out / "vnp13a1.nc", win, m.layers,
                  {"title": f"VIIRS {VEG_PRODUCT} v002 16-day vegetation indices cropped to {fire['fire_id']}",
                   "fire_id": fire["fire_id"], "sources": " | ".join(sources)}, times_ns=times)
    return "ok", sources


PER_FIRE = {"history": grab_history, "perimeters": grab_perimeters, "firms": grab_firms, "gridmet": grab_gridmet,
            "ics209": grab_ics209, "suppression": grab_suppression, "tessera": grab_tessera, "terrain": grab_terrain,
            "landfire": grab_landfire, "lulc": grab_lulc, "landcover": grab_landcover, "vegetation": grab_vegetation}


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

    def append(self, t_ns, source, fields, extra=None):
        """fields: {name: (2-D array, dtype, attrs)}; extra: {name: (scalar, dtype, attrs)}, one value per step."""
        for nm, (arr, dtype, attrs) in fields.items():
            if nm not in self.ds.variables:
                fill = attrs.get("_FillValue")
                v = self.ds.createVariable(nm, dtype, ("time", "y", "x"), zlib=True, complevel=4,
                                           fill_value=fill if fill is not None else None)
                v.set_auto_maskandscale(False)
                v.setncatts({k: a for k, a in attrs.items() if k != "_FillValue"})
            self.ds.variables[nm][self.n] = arr
        for nm, (val, dtype, attrs) in (extra or {}).items():
            if nm not in self.ds.variables:
                self.ds.createVariable(nm, dtype, ("time",)).setncatts(attrs)
            self.ds.variables[nm][self.n] = val
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
                    ds, tmp = _open_nc_bytes(netCDF4, data, ctx)
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
                                day = f"ABI-L2-FDCC/{t.year}/{t.dayofyear:03d}/"
                                scans[fid][(bucket, day)] = scans[fid].get((bucket, day), 0) + 1
                    finally:
                        ds.close()
                        if tmp:
                            tmp.unlink(missing_ok=True)
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


HRRR_NEXT_UNITS = {"gust": "m s-1", "u10": "m s-1", "v10": "m s-1", "rh2": "%", "t2": "K", "q2": "kg kg-1",
                   "apcp": "kg m-2 (accumulated from the run start to the valid time)"}


def hrrr_next_fields():
    """The hrrr channel's 4 fields plus TS-SatFire's forecast temperature, specific humidity and precipitation.
    APCP is the message accumulated from the run start ('0-9 hour acc', '0-1 day acc'), not the last hour's."""
    import weather_features as WF
    return WF.HRRR_FIELDS + [("t2", ":TMP:2 m above ground:"), ("q2", ":SPFH:2 m above ground:"),
                             ("apcp", ":APCP:surface:0-")]


def grab_hrrr_next_many(fires, ctx, chunk=25):
    """The 12z run's forecasts every 3 h out to 36 h for each day of span() (hrrr_next_times): gust, u10, v10,
    rh2, t2, q2 and apcp decoded from each file's own GRIB2 messages and cropped on HRRR's native 3 km grid,
    one series per fire: hrrr_next.nc with time = valid time, lead_h and run_time per step. This is the
    next-day forecast TS-SatFire takes from GFS, at HRRR's resolution."""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    import weather_features as WF
    lat, lon = ctx.memo("hrrr_grid", lambda: WF.hrrr_latlon(ctx.sess))
    fields = hrrr_next_fields()
    names = [n for n, _ in fields]

    def fetch(key):
        run, h = key
        url = WF.HRRR.format(d=run.strftime("%Y%m%d"), h=run.hour, f=h)
        return key, url, WF.hrrr_fetch(ctx.sess, url, fields)

    for group in _chunks(fires, chunk):
        if ctx.expired():
            return
        need, left, wins, series, got_n = {}, {}, {}, {}, {}
        for f in group:
            fid = f["fire_id"]
            keys = [(run, h) for run, h in hrrr_next_times(f) if run >= _ts(HRRR_START)]
            wins[fid] = WF.hrrr_window(lat, lon, f["bounds"])
            left[fid], got_n[fid] = len(keys), []
            if not keys:
                yield f, "empty", [], None
            for k in keys:
                need.setdefault(k, []).append(f)
        stopped = False
        with ThreadPoolExecutor(ctx.threads) as ex:
            for (run, h), url, got in _windowed_map(ex, fetch, sorted(need), 2 * ctx.threads):
                for f in need[(run, h)]:
                    fid = f["fire_id"]
                    win = wins[fid]
                    if got and win is not None:
                        r0, r1, c0, c1 = win
                        if fid not in series:
                            series[fid] = NcSeries(ctx.stage(fid) / "hrrr_next.nc", lat[r0:r1, c0:c1], lon[r0:r1, c0:c1],
                                                   {"title": f"HRRR 12z run, forecasts every 3 h out to 36 h, cropped to {fid}",
                                                    "fire_id": fid, "source": "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/",
                                                    "fields": "GUST surface; UGRD/VGRD 10 m; RH 2 m; TMP 2 m; SPFH 2 m; "
                                                              "APCP surface accumulated from the run start",
                                                    "window_rows_cols": f"{r0}:{r1},{c0}:{c1}"})
                        valid = run + pd.Timedelta(hours=h)
                        series[fid].append(
                            valid.value, url,
                            {n: (got[n][r0:r1, c0:c1], "f4", {"units": HRRR_NEXT_UNITS[n]}) for n in names if n in got},
                            extra={"lead_h": (h, "i2", {"long_name": "forecast lead", "units": "hours"}),
                                   "run_time": (run.value, "i8", {"long_name": "model run (initialisation) time",
                                                                  "units": "nanoseconds since 1970-01-01 00:00:00 UTC"})})
                        got_n[fid].append(url)
                    left[fid] -= 1
                    if left[fid] == 0:
                        if fid in series:
                            series.pop(fid).close()
                        yield f, ("ok" if got_n[fid] else "empty"), sorted(got_n[fid]), None
                if ctx.expired():
                    stopped = True
                    break
        if stopped:
            for fid, s in series.items():
                s.close()
                ctx.drop_stage(fid)
            return


# ------------------------------------------------------------------------------------- VIIRS swaths

# (name, LAADS product prefix, first day of data). Products <prefix>02IMG (I1-I5, 375 m), 03IMG (its
# geolocation), 02MOD (M-bands, 750 m), 03MOD (its geolocation); the newest collection version of each.
VIIRS_SATS = (("SNPP", "VNP", "2012-01-19"), ("NOAA20", "VJ1", "2018-01-05"), ("NOAA21", "VJ2", "2023-02-10"))
VIIRS_PRODUCTS = ("02IMG", "03IMG", "02MOD", "03MOD")
VIIRS_I_BANDS = ("I01", "I02", "I03", "I04", "I05")     # with their _quality_flags, _uncert_index, LUTs
VIIRS_M_BANDS = ("M11",)
VIIRS_GEO = ("latitude", "longitude", "sensor_zenith", "solar_zenith")
VIIRS_POOL = "process"        # h5py serialises reads within a process: granules go to worker processes
VIIRS_WORKERS = 8


def viirs_granules(sess, fire):
    """Swath granules over the box during span(): {(sat, 'A2018314', '2048'): {'sat', 'daynight', 'start',
    '02IMG': url, '03IMG': url, '02MOD': url, '03MOD': url}}, only those with I-bands and their geolocation."""
    import pandas as pd
    import earthdata as ED
    t0, t1 = span(fire)
    out = {}
    for sat, prefix, first in VIIRS_SATS:
        if t1 < pd.Timestamp(first, tz="UTC"):
            continue
        for prod in VIIRS_PRODUCTS:
            for g in ED.cmr_granules(sess, prefix + prod, t0, t1, fire["bounds"]):
                _, a, hhmm, _ = ED.granule_parts(g["url"])
                e = out.setdefault((sat, a, hhmm), {"sat": sat})
                e[prod] = g["url"]
                if prod == "02IMG":
                    e["daynight"], e["start"] = g["daynight"], g["start"]
    return {k: e for k, e in out.items() if "02IMG" in e and "03IMG" in e}


def _grow(win, k, shape):
    r0, r1, c0, c1 = win
    return max(r0 - k, 0), min(r1 + k, shape[0]), max(c0 - k, 0), min(c1 + k, shape[1])


def _h5_window(group, prefixes, shape, win, luts):
    """{name: (array, attrs)}: the window of every 2-D dataset of `shape` whose name starts with one of
    prefixes, and every 1-D numeric dataset with such a name (look-up tables) whole, read once per granule."""
    import h5py
    r0, r1, c0, c1 = win
    out = {}
    for name, ds in group.items():
        if not isinstance(ds, h5py.Dataset) or not name.startswith(prefixes) or ds.dtype.kind not in "iuf":
            continue
        if ds.shape == tuple(shape):
            out[name] = (ds[r0:r1, c0:c1], dict(ds.attrs))
        elif ds.ndim == 1:
            if name not in luts:
                luts[name] = (ds[:], dict(ds.attrs))
            out[name] = luts[name]
    return out


def _viirs_read(job):
    """One granule for every fire on it (runs in a worker). job = (key, entry, [(fire_id, bounds)]).
    -> (key, {fire_id: crop}, bytes downloaded, error or None). The fire windows come from the 750 m
    geolocation (a quarter of the 375 m one); I-band windows follow from them (earthdata.m_to_i)."""
    import h5py
    import requests
    import earthdata as ED
    key, e, fires = job
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    opened = []

    def open_h5(prod):
        fo = ED.open_remote(sess, e[prod], ED.token())
        opened.append(fo)
        return h5py.File(fo, "r")

    try:
        has_m = "02MOD" in e and "03MOD" in e
        with open_h5("03MOD" if has_m else "03IMG") as g:
            gl = g["geolocation_data"]
            lat, lon = gl["latitude"][:], gl["longitude"][:]
        wins = {fid: w for fid, b in fires if (w := ED.swath_window(lat, lon, b)) is not None}
        crops = {}
        if wins:
            with open_h5("02IMG") as fi, open_h5("03IMG") as gi:
                obs_i, geo_i = fi["observation_data"], gi["geolocation_data"]
                i_shape = geo_i["latitude"].shape
                glob = clean_attrs(fi.attrs)
                luts = {}
                for fid, w in wins.items():
                    m_win = _grow(w, 1, lat.shape) if has_m else None
                    i_win = ED.m_to_i(m_win, i_shape) if has_m else _grow(w, 2, i_shape)
                    crops[fid] = {"i_win": i_win, "m_win": m_win, "glob": glob,
                                  "i": {**_h5_window(geo_i, VIIRS_GEO, i_shape, i_win, {}),
                                        **_h5_window(obs_i, VIIRS_I_BANDS, i_shape, i_win, luts)}, "m": {}}
            if has_m:
                with open_h5("02MOD") as fm, open_h5("03MOD") as gm:
                    obs_m, geo_m = fm["observation_data"], gm["geolocation_data"]
                    luts = {}
                    for fid, c in crops.items():
                        geo = _h5_window(geo_m, VIIRS_GEO, lat.shape, c["m_win"], {})
                        c["m"] = {**{f"m_{k}": v for k, v in geo.items()},
                                  **_h5_window(obs_m, VIIRS_M_BANDS, lat.shape, c["m_win"], luts)}
        return key, crops, sum(getattr(f, "fetched", 0) for f in opened), None
    except Exception as ex:  # noqa: BLE001  (reported per granule; the pool must not stop)
        return key, {}, sum(getattr(f, "fetched", 0) for f in opened), redact(f"{e.get('02IMG')}: {type(ex).__name__}: {ex}")[:400]
    finally:
        for f in opened:
            f.close()


def viirs_file_name(key, e):
    import pandas as pd
    sat, a, hhmm = key
    day = (pd.Timestamp(f"{a[1:5]}-01-01") + pd.Timedelta(days=int(a[5:8]) - 1)).strftime("%Y-%m-%d")
    return f"{sat}_{day}T{hhmm}_{(e.get('daynight') or 'unknown').lower()}.nc"


def _viirs_write(path, fid, key, e, c):
    """One granule's crop: I grid (iy, ix) with I01-I05 (+ flags), latitude, longitude, zeniths; M grid
    (my, mx) with M11 (+ flags) and m_latitude etc.; look-up tables whole. Values as stored in the source."""
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        r0, r1, c0, c1 = c["i_win"]
        at = {"title": f"VIIRS {key[0]} L1B granule {key[1]}.{key[2]} cropped to {fid}", "fire_id": fid,
              "satellite": key[0], "day_night": e.get("daynight") or "", "granule_start": e.get("start") or "",
              "i_window_rows_cols": f"{r0}:{r1},{c0}:{c1}",
              "sources": " | ".join(e[p] for p in VIIRS_PRODUCTS if p in e)}
        if c.get("m_win"):
            mr0, mr1, mc0, mc1 = c["m_win"]
            at["m_window_rows_cols"] = f"{mr0}:{mr1},{mc0}:{mc1}"
        ds.setncatts({**{f"source_{k}": v for k, v in c["glob"].items()}, **at})
        for grid, dims in (("i", ("iy", "ix")), ("m", ("my", "mx"))):
            if not c.get(grid):
                continue
            made = False
            for name, (arr, attrs) in c[grid].items():
                if arr.ndim == 2 and not made:
                    ds.createDimension(dims[0], arr.shape[0])
                    ds.createDimension(dims[1], arr.shape[1])
                    made = True
                dd = dims if arr.ndim == 2 else (f"{name}_n",)
                if arr.ndim == 1:
                    ds.createDimension(dd[0], arr.shape[0])
                v = ds.createVariable(name, arr.dtype, dd, zlib=True, complevel=4, fill_value=fill_value(attrs, arr.dtype))
                v.set_auto_maskandscale(False)
                v.setncatts(clean_attrs(attrs))
                v[:] = arr


def grab_viirs_many(fires, ctx, chunk=25):
    """VIIRS L1B swaths (TS-SatFire's VIIRS_Day / VIIRS_Night source): every S-NPP, NOAA-20 and NOAA-21
    granule over the box during span(), day and night, cropped on the swath itself (no regridding): I1-I5
    at 375 m and M11 at 750 m as stored (scaled integers, quality flags, the brightness-temperature look-up
    tables of I4 / I5), with each pixel's latitude, longitude and sensor / solar zenith. Remote files are
    read by byte range (a fire's part of a 250 MB granule is a few MB); each granule is read once for every
    fire on it. One file per granule: <SAT>_<YYYY-MM-DD>T<HHMM>_<day|night>.nc. Granules that fail are
    named in the manifest note; a fire is an error only when all of them fail."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    workers = max(1, min(ctx.threads, VIIRS_WORKERS))
    if VIIRS_POOL == "process":          # spawn: forking a process that runs threads can deadlock
        def pool(n):
            return ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"))
    else:
        pool = ThreadPoolExecutor
    t_start = time.time()

    def look(f):
        try:
            return f, viirs_granules(ctx.sess, f), None
        except Exception as ex:  # noqa: BLE001
            return f, None, redact(f"CMR: {type(ex).__name__}: {ex}")

    for group in _chunks(fires, chunk):
        if ctx.expired():
            return
        need, left, written, srcs, failed = {}, {}, {}, {}, {}
        with ThreadPoolExecutor(8) as ex:
            for f, grans, err in ex.map(look, group):
                fid = f["fire_id"]
                if err or not grans:
                    yield f, ("error" if err else "empty"), [], err
                    continue
                left[fid], written[fid], srcs[fid], failed[fid] = len(grans), [], [], []
                for k, e in grans.items():
                    need.setdefault(k, (e, []))[1].append(f)
        jobs = [(k, e, [(f["fire_id"], f["bounds"]) for f in fs])
                for k, (e, fs) in sorted(need.items(), key=lambda kv: (str(kv[1][0].get("start")), kv[0]))]
        if jobs:
            print(f"  viirs: {len(jobs)} granules for {len(left)} fires", flush=True)
        stopped, got = False, 0
        with pool(workers) as ex:
            for key, crops, nbytes, err in _windowed_map(ex, _viirs_read, jobs, 2 * workers):
                got += nbytes
                e, fs = need[key]
                for fid, c in crops.items():
                    name = viirs_file_name(key, e)
                    _viirs_write(ctx.stage(fid) / name, fid, key, e, c)
                    written[fid].append(name)
                    srcs[fid].append(e["02IMG"])
                for f in fs:
                    fid = f["fire_id"]
                    if err:
                        failed[fid].append(err)
                    left[fid] -= 1
                    if left[fid] == 0:
                        note = (f"{len(failed[fid])} granule(s) failed: {failed[fid][0]}" if failed[fid] else None)
                        status = "ok" if written[fid] else ("error" if failed[fid] else "empty")
                        yield f, status, srcs[fid], note
                if ctx.expired():
                    stopped = True
                    break
        if jobs:
            print(f"  viirs: {got / 1e9:.2f} GB read in {(time.time() - t_start) / 60:.0f} min so far", flush=True)
        if stopped:
            for fid, n in left.items():
                if n > 0:
                    ctx.drop_stage(fid)
            return


GROUP_FUNCS = {"goes": grab_goes_many, "hrrr": grab_hrrr_many, "hrrr_next": grab_hrrr_next_many, "osm": grab_osm_many,
               "viirs": grab_viirs_many}
