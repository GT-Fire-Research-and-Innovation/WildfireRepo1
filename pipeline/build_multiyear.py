"""
Multi-year training set (2014-2025) built from the fire catalog (catalog.py), using only inputs that
exist for every year. Runs anywhere: a laptop, a 4090 desktop, RunPod, or PACE (see scripts/).

Per fire, in OUT/fires/<fire_id>/ (one file per stage, so every stage is resumable):
  grid.json      150 m lon/lat grid: transform, height, width, fire year, perimeter times
  labels.npz     arrival (H, W) int16: index of the first perimeter containing the cell (-1 = never)
                 times (P,) int64 ns UTC, burn_days (P,), map_methods (P,), other (H, W) bool
                 (other fires' perimeters that year: excluded from the loss)
  history.npz    years_since (H, W) uint8 (255 = no fire on record), n_fires (H, W) uint8
                 from the Interagency Fire Perimeter History, prior years only
  viirs.npz      FIRMS VIIRS 375 m active fire per local solar day: af_day, af_night (counts),
                 frp_max (MW), dates; points.csv.gz keeps every detection with its time
  weather.npz    GRIDMET daily (9 variables), 14 days before the first perimeter to 2 days after the last
  hrrr.npz       HRRR gust, u10, v10, rh2 every 3 h across the fire (+ the 12z+9h forecast each day)
  goes.npz       every GOES-East/West ABI fire pixel in the box, hourly (2017+), sparse with times
  tessera.npz    128-d Tessera embedding of the year before the fire at 150 m (fires 2018+)
  suppression.npz daily fire lines / burnouts / drops (2018+) and ICS-209 crews, aircraft, containment
                 (1999-2020), bit-packed
  terrain.npz    elevation (m), slope (deg), aspect (deg, downslope direction), terrain position
                 (0.5 km, 2 km) and roughness, from Copernicus GLO-30
  landfire.npz   FBFM40, CC, CH, CBH, CBD from the newest LANDFIRE version that predates the fire
                 (LF2016 for 2017-2022, LF{year-1} from 2023; none for 2014-2016 -> missing)
  human.npz      9 OSM channels (roads x3, rail, power lines, buildings, developed, water, rivers),
                 1 January snapshot of the fire's year
Every layer that can be missing for a fire says so (missing / has_* flags) so the model masks it.

  python build_multiyear.py --catalog data/catalog --out data_my --stages all --workers 6
  python build_multiyear.py ... --limit 3            # quick test
"""
import argparse
import gzip
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

RES_M = 150.0
PAD = 64                     # cells (9.6 km) around the union of all perimeters
MAX_CELLS = 1600             # per side (240 km); larger fires are skipped and logged
DAYS_BEFORE, DAYS_AFTER = 14, 2
STAGES = ["grid", "labels", "history", "viirs", "terrain", "weather", "landfire", "tessera", "suppression",
          "hrrr", "goes", "human"]
DATE_MAJOR = ["hrrr", "goes", "human"]
HIST_URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
            "InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0")


# ---------------------------------------------------------------------------- grid

def make_grid(row):
    lat0 = (row.south + row.north) / 2
    dy = RES_M / 110570.0
    dx = RES_M / (111320.0 * math.cos(math.radians(lat0)))
    W = int(math.ceil((row.east - row.west) / dx)) + 2 * PAD
    H = int(math.ceil((row.north - row.south) / dy)) + 2 * PAD
    x0, y0 = row.west - PAD * dx, row.north + PAD * dy
    return dict(fire_id=row.fire_id, name=row["name"], year=int(row.year), transform=[dx, 0.0, x0, 0.0, -dy, y0],
                height=H, width=W, bounds_lonlat=[x0, y0 - H * dy, x0 + W * dx, y0],
                first=str(row["first"]), last=str(row["last"]))


def centres(g):
    dx, _, x0, _, e, y0 = g["transform"]
    lon = x0 + (np.arange(g["width"]) + 0.5) * dx
    lat = y0 + (np.arange(g["height"]) + 0.5) * e
    return np.meshgrid(lon, lat)


def rasterize(geoms, g, all_touched=False):
    from affine import Affine
    from rasterio.features import rasterize as rz
    if not len(geoms):
        return np.zeros((g["height"], g["width"]), bool)
    return rz([(gm, 1) for gm in geoms], out_shape=(g["height"], g["width"]), transform=Affine(*g["transform"]),
              fill=0, dtype="uint8", all_touched=all_touched).astype(bool)


def days_of(g):
    d0 = pd.Timestamp(g["first"]).tz_convert(None).normalize() - pd.Timedelta(days=DAYS_BEFORE)
    d1 = pd.Timestamp(g["last"]).tz_convert(None).normalize() + pd.Timedelta(days=DAYS_AFTER)
    return pd.date_range(d0, d1)


# ---------------------------------------------------------------------------- stages

def st_labels(g, ctx):
    per = ctx["per"]
    p = per[per.fire_id == g["fire_id"]].sort_values("perim_time")
    arrival = np.full((g["height"], g["width"]), -1, np.int16)
    for i, gm in enumerate(p.geometry.values):
        m = rasterize([gm], g) & (arrival < 0)
        arrival[m] = i
    w, s, e, n = g["bounds_lonlat"]
    t0, t1 = pd.Timestamp(f"{g['year']}-01-01", tz="UTC"), pd.Timestamp(g["last"])
    cand = per.iloc[list(per.sindex.intersection((w, s, e, n)))]
    cand = cand[(cand.fire_id != g["fire_id"]) & (cand.perim_time >= t0) & (cand.perim_time <= t1 + pd.Timedelta(days=30))]
    other = rasterize(list(cand.geometry.values), g) & (arrival < 0)
    txt = lambda c: np.array([str(v) for v in p[c]], dtype="U")  # noqa: E731  (plain unicode: loads without pickle)
    return dict(arrival=arrival, other=other, times=p.perim_time.astype("int64").values,
                burn_days=txt("burn_day"), map_methods=txt("map_method"), sources=txt("source"))


def st_history(g, ctx):
    import requests
    from shapely.geometry import shape
    w, s, e, n = g["bounds_lonlat"]
    params = {"where": f"FIRE_YEAR_INT < {g['year']} AND FIRE_YEAR_INT >= {g['year'] - 40}",
              "geometry": f"{w},{s},{e},{n}", "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
              "spatialRel": "esriSpatialRelIntersects", "outFields": "FIRE_YEAR_INT", "returnGeometry": "true",
              "maxAllowableOffset": 0.0005, "f": "geojson", "resultRecordCount": 1000}
    feats, off = [], 0
    while True:
        js = requests.get(f"{HIST_URL}/query", params=dict(params, resultOffset=off), timeout=180).json()
        if "error" in js:
            raise RuntimeError(js["error"])
        batch = js.get("features", [])
        feats += batch
        if len(batch) < 1000:
            break
        off += len(batch)
    since = np.full((g["height"], g["width"]), 255, np.uint8)
    count = np.zeros((g["height"], g["width"]), np.uint8)
    for f in feats:
        y = f["properties"].get("FIRE_YEAR_INT")
        if not y or not f.get("geometry"):
            continue
        m = rasterize([shape(f["geometry"])], g)
        since[m] = np.minimum(since[m], g["year"] - int(y))
        count[m] = np.minimum(count[m].astype(int) + 1, 255)
    return dict(years_since=since, n_fires=count, n_polygons=len(feats))


def st_viirs(g, ctx):
    from live import firms
    w, s, e, n = g["bounds_lonlat"]
    days = days_of(g)
    d0, d1 = days[0] + pd.Timedelta(days=DAYS_BEFORE - 7), days[-1]
    srcs = ["VIIRS_SNPP_SP"] + (["VIIRS_NOAA20_SP"] if g["year"] >= 2018 else [])
    frames = []
    for src in srcs:
        d = d0
        while d <= d1:
            k = min(5, (d1 - d).days + 1)            # FIRMS archive requests: at most 5 days
            ctx["firms_limiter"]()
            df = firms.get(src, (w, s, e, n), k, d.date().isoformat())
            if len(df):
                frames.append(df)
            d += pd.Timedelta(days=k)
    if not frames and g["year"] >= 2024:      # standard product not processed yet -> near real time
        for src in firms.NRT:
            d = d0
            while d <= d1:
                k = min(5, (d1 - d).days + 1)            # FIRMS archive requests: at most 5 days
                ctx["firms_limiter"]()
                df = firms.get(src, (w, s, e, n), k, d.date().isoformat())
                if len(df):
                    frames.append(df)
                d += pd.Timedelta(days=k)
    pts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["latitude", "longitude", "frp", "acq_date", "acq_time", "daynight", "confidence", "source"])
    dates = pd.date_range(d0, d1)
    D, H, W = len(dates), g["height"], g["width"]
    af_day = np.zeros((D, H, W), np.uint8)
    af_night = np.zeros((D, H, W), np.uint8)
    frp = np.zeros((D, H, W), np.float16)
    if len(pts):
        t = pd.to_datetime(pts["acq_date"]) + pd.to_timedelta(pts["acq_time"].astype(int).astype(str).str.zfill(4).str[:2].astype(int), unit="h") \
            + pd.to_timedelta(pts["acq_time"].astype(int).astype(str).str.zfill(4).str[2:].astype(int), unit="m")
        local = t + pd.to_timedelta(pts["longitude"] / 15.0, unit="h")
        di = ((local.dt.normalize() - dates[0]).dt.days).values
        dx, _, x0, _, ey, y0 = g["transform"]
        c = np.floor((pts["longitude"].values - x0) / dx).astype(int)
        r = np.floor((pts["latitude"].values - y0) / ey).astype(int)
        night = (pts["daynight"].astype(str) == "N").values
        f = pd.to_numeric(pts["frp"], errors="coerce").fillna(0).values
        for dr in (-1, 0, 1):                 # 375 m footprint ~ 3 x 3 cells of 150 m
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                ok = (di >= 0) & (di < D) & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                np.add.at(af_night, (di[ok & night], rr[ok & night], cc[ok & night]), 1)
                np.add.at(af_day, (di[ok & ~night], rr[ok & ~night], cc[ok & ~night]), 1)
                np.maximum.at(frp, (di[ok], rr[ok], cc[ok]), f[ok].astype(np.float16))
    ctx["write_points"](pts)
    return dict(af_day=af_day, af_night=af_night, frp_max=frp, dates=np.array([str(d.date()) for d in dates]),
                n_points=len(pts))


def st_terrain(g, ctx):
    import build_dataset as B
    from scipy.ndimage import uniform_filter
    elev, slope, aspect = B.dem_region(g["transform"], (0, g["height"], 0, g["width"]))
    # terrain position: elevation minus the mean around it (ridge > 0, valley < 0), at 0.5 km and 2 km
    tpi_s = elev - uniform_filter(elev, 7, mode="nearest")
    tpi_l = elev - uniform_filter(elev, 27, mode="nearest")
    rough = np.sqrt(np.maximum(uniform_filter(elev ** 2, 5, mode="nearest") - uniform_filter(elev, 5, mode="nearest") ** 2, 0))
    return dict(elev=elev, slope=slope.astype(np.float16), aspect=aspect.astype(np.float16),
                tpi_500m=tpi_s.astype(np.float16), tpi_2km=tpi_l.astype(np.float16), rough=rough.astype(np.float16))


def st_weather(g, ctx):
    """GRIDMET daily, 14 days before the first perimeter to 2 days after the last (HRRR is its own
    date-major stage so each HRRR file is downloaded once for all fires)."""
    import weather_features as WF
    days = days_of(g)
    gm, glat, glon = WF.gridmet_fire(ctx["gridmet"](), {"bounds_lonlat": g["bounds_lonlat"]}, days)
    lon, lat = centres(g)
    gr = np.abs(glat[:, None] - lat[:, 0][None, :]).argmin(0)
    gc = np.abs(glon[:, None] - lon[0, :][None, :]).argmin(0)
    return dict(dates=np.array([str(d.date()) for d in days]), gm=gm.astype(np.float32),
                gm_vars=np.array(list(WF.GM_VARS)), gm_row=gr.astype(np.int16), gm_col=gc.astype(np.int16))


TESSERA_FIRST_YEAR = 2017        # first year in the GeoTessera v1.1 Zarr store


def st_tessera(g, ctx):
    """128-d Tessera embedding of the year BEFORE the fire (no burn scar), block-averaged from 10 m
    to 150 m in UTM, then resampled onto the fire grid."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject
    ty = g["year"] - 1
    if ty < TESSERA_FIRST_YEAR:
        return dict(missing=np.array(True), year=np.array(ty))
    gt = ctx["tessera"]()
    k = 15                                        # 15 x 10 m = 150 m blocks
    acc = cnt = None
    tr0 = crs = None
    for strip, tr, crs in gt.iter_region(tuple(g["bounds_lonlat"]), ty, strip_rows=10 * k):
        if tr0 is None:
            tr0 = tr
        r_off = int(round((tr.f - tr0.f) / tr0.e))   # rows below the first strip
        h, w, b = strip.shape
        H2, W2 = -(-h // k) * k, -(-w // k) * k
        pad = np.full((H2, W2, b), np.nan, np.float32)
        pad[:h, :w] = strip
        blk = pad.reshape(H2 // k, k, W2 // k, k, b)
        s = np.nansum(blk, axis=(1, 3))
        n = np.isfinite(blk[..., 0]).sum(axis=(1, 3))
        br = r_off // k
        if acc is None:
            acc = np.zeros((0, W2 // k, b), np.float32)
            cnt = np.zeros((0, W2 // k), np.int32)
        need = br + s.shape[0]
        if need > acc.shape[0]:
            acc = np.concatenate([acc, np.zeros((need - acc.shape[0],) + acc.shape[1:], np.float32)])
            cnt = np.concatenate([cnt, np.zeros((need - cnt.shape[0], cnt.shape[1]), np.int32)])
        acc[br:need] += s
        cnt[br:need] += n
    if acc is None:
        return dict(missing=np.array(True), year=np.array(ty))
    mean = np.where(cnt[..., None] > 0, acc / np.maximum(cnt[..., None], 1), np.nan).transpose(2, 0, 1)
    src_tr = Affine(tr0.a * k, 0, tr0.c, 0, tr0.e * k, tr0.f)
    dst = np.full((mean.shape[0], g["height"], g["width"]), np.nan, np.float32)
    reproject(mean, dst, src_transform=src_tr, src_crs=crs, dst_transform=Affine(*g["transform"]), dst_crs="EPSG:4326",
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    cover = np.isfinite(dst[0])
    return dict(embedding=np.nan_to_num(dst).astype(np.float16), cover=cover, year=np.array(ty), missing=np.array(False))


def st_suppression(g, ctx):
    """Daily suppression: fire lines, burnouts and drops from the NIFC Operational Data Archive (2018+),
    and crews / aircraft / containment from ICS-209-PLUS (1999-2020). Same rules as suppression_layers.py:
    the map for day d is the archive as it stood at 06:00 local on d+1; use day k's layers for k -> k+1."""
    import geopandas as gpd
    import nifc_perimeters as NP
    import suppression_layers as SL
    per = ctx["per"]
    p = per[per.fire_id == g["fire_id"]]
    start, end = pd.Timestamp(p.burn_day.min()), pd.Timestamp(p.burn_day.max())
    dates = [str(d.date()) for d in pd.date_range(start, end)]
    incidents = {"irwin": {i for i in p.irwin_id.dropna() if i}, "names": {n.lower() for n in p.incident_name.dropna() if n}}
    layers = np.zeros((len(dates), len(SL.CHANNEL_NAMES), g["height"], g["width"]), np.uint8)
    has_spatial = False
    if g["year"] >= 2018:
        parts = []
        for y in range(start.year, end.year + 1):
            for url, _ in ctx["supp_layers"](y):
                parts.append(SL.archive_features(NP.query_layer(ctx["sess"], url, g["bounds_lonlat"], ctx["nifc_cache"])))
        parts = [x for x in parts if not x.empty]
        feats = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326) if parts \
            else SL.archive_features([])
        layers = SL.rasterize_layers(feats, g, dates, incidents if incidents["names"] else None)
        has_spatial = True
    ics = ctx["ics"]()
    fields = ics[2] if ics else []
    vals = np.full((len(dates), len(fields) + 1), np.nan, np.float32)
    ids = []
    if ics and g["year"] <= 2020 and incidents["names"]:
        sit_df, inc_df, fields = ics
        ids = SL.match_ics(inc_df, incidents, g, start, end)
        vals = SL.daily_ics(sit_df, ids, dates, fields, set(ids))
    return dict(dates=np.array(dates), layers=np.packbits(layers, axis=1), n_channels=np.array(len(SL.CHANNEL_NAMES)),
                channels=np.array(SL.CHANNEL_NAMES), ics_values=vals, ics_fields=np.array(list(fields) + ["days_since_report"]),
                ics_incidents=np.array(ids), has_spatial=np.array(has_spatial), has_ics=np.array(bool(ids)))


def landfire_version(year):
    if year <= 2016:
        return None
    if year <= 2022:
        return "LF2016"
    return f"LF{min(year - 1, 2025)}"


def st_landfire(g, ctx):
    import rasterio
    import landfire_features as L
    v = landfire_version(g["year"])
    if v is None:
        return dict(missing=np.array(True), version=np.array("none"))
    tif = ctx["dir"] / "landfire_raw.tif"
    L.fetch(g["bounds_lonlat"], tif, layers=[f"{v}_{x}" for x in ("FBFM40", "CC", "CH", "CBH", "CBD")])
    lon, lat = centres(g)
    with rasterio.open(tif) as d:
        a = d.read()
        tr = d.transform
        col = np.clip(np.floor((lon - tr.c) / tr.a).astype(int), 0, d.width - 1)
        row = np.clip(np.floor((lat - tr.f) / tr.e).astype(int), 0, d.height - 1)
    s = a[:, row, col]
    tif.unlink()
    return dict(fbfm40=s[0].astype(np.int16), cc=s[1].astype(np.int16), ch=s[2].astype(np.int16),
                cbh=s[3].astype(np.int16), cbd=s[4].astype(np.int16), version=np.array(v), missing=np.array(False))


FUNCS = {"labels": st_labels, "history": st_history, "viirs": st_viirs, "terrain": st_terrain,
         "weather": st_weather, "landfire": st_landfire, "tessera": st_tessera, "suppression": st_suppression}


# ---------------------------------------------------------------------------- shared resources (per process)

_CTX = {}


def _ctx(out, catalog):
    if _CTX:
        return _CTX
    import geopandas as gpd
    import requests
    import weather_features as WF
    per = gpd.read_parquet(Path(catalog) / "perimeters.parquet")
    _CTX["per"] = per
    sess = requests.Session()
    cache = Path(out) / "cache" / "hrrr"
    cache.mkdir(parents=True, exist_ok=True)
    state = {}

    def gridmet():
        if "gm" not in state:
            import xarray as xr
            state["gm"] = {k: xr.open_dataset(WF.GRIDMET.format(v=k), decode_times=True) for k in WF.GM_VARS}
        return state["gm"]

    def hrrr_grid():
        p = cache / "latlon.npz"
        if not p.exists():
            lat, lon = WF.hrrr_latlon(sess)
            np.savez(p, lat=lat.astype(np.float32), lon=lon.astype(np.float32))
        z = np.load(p)
        return z["lat"], z["lon"]

    def hrrr(url):
        """Fields of one HRRR file, cached as float16 per date-hour (shared by every fire)."""
        key = cache / (url.split("hrrr.")[1].replace("/", "_").replace(".grib2", "") + ".npz")
        if key.exists():
            z = np.load(key)
            return {k: z[k].astype(np.float32) for k in z.files}
        got = WF.hrrr_fetch(sess, url)
        if got:
            tmp = key.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, **{k: v.astype(np.float16) for k, v in got.items()})
            tmp.rename(key)
        return got

    last = [0.0]

    def firms_limiter():
        # 5000 transactions / 10 min per key, shared by all workers: stay under ~6/s overall
        dt = 0.15 * int(os.environ.get("BUILD_WORKERS", "4")) - (time.time() - last[0])
        if dt > 0:
            time.sleep(dt)
        last[0] = time.time()

    def tessera():
        if "tess" not in state:
            from geotessera.store import GeoTesseraZarr
            state["tess"] = GeoTesseraZarr()
        return state["tess"]

    def supp_layers(year):
        if "supp" not in state:
            state["supp"] = {}
        if year not in state["supp"]:
            import nifc_perimeters as NP
            import suppression_layers as SL
            NP.FGDB_DIR = Path(out) / "nifc_fgdb"
            state["supp"].update(SL.discover_archive_layers(sess, [year]))
        return state["supp"].get(year, [])

    def ics():
        if "ics" not in state:
            import suppression_layers as SL
            try:
                state["ics"] = SL.load_ics209(Path(out))
            except Exception as ex:  # noqa: BLE001
                print("ICS-209-PLUS unavailable:", ex, flush=True)
                state["ics"] = None
        return state["ics"]

    nifc_cache = Path(out) / "cache" / "nifc"
    nifc_cache.mkdir(parents=True, exist_ok=True)
    _CTX.update(gridmet=gridmet, hrrr_grid=hrrr_grid, hrrr=hrrr, firms_limiter=firms_limiter, tessera=tessera,
                supp_layers=supp_layers, ics=ics, sess=sess, nifc_cache=nifc_cache)
    return _CTX


def build_one(out, catalog, g, stages):
    out = Path(out)
    d = out / "fires" / g["fire_id"]
    d.mkdir(parents=True, exist_ok=True)
    ctx = dict(_ctx(out, catalog))
    ctx["dir"] = d
    ctx["write_points"] = lambda pts: pts.to_csv(gzip.open(d / "points.csv.gz", "wt"), index=False)
    done = []
    for st in stages:
        dst = d / f"{st}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        try:
            res = FUNCS[st](g, ctx)
        except Exception as ex:  # noqa: BLE001
            (d / f"{st}.error.txt").write_text(traceback.format_exc())
            done.append(f"{st}:ERROR({str(ex)[:80]})")
            continue
        tmp = d / f"{st}.tmp.npz"
        np.savez_compressed(tmp, **res)
        tmp.rename(dst)
        (d / f"{st}.error.txt").unlink(missing_ok=True)
        done.append(f"{st}:{time.time() - t0:.0f}s")
    return g["fire_id"], done


# ---------------------------------------------------------------------------- date-major stages
# HRRR and GOES files cover all of CONUS, and fire seasons overlap, so these stages loop over time:
# each file is downloaded once and cut for every fire that needs it.

HRRR_STEP_H = 3                  # analysis every 3 h from 1 day before the first perimeter to the last
GOES_STEP_MIN = 60               # one GOES scan per hour per satellite
GOES_FIRE_MASK = {10, 11, 12, 13, 14, 30, 31, 32, 33, 34}   # ABI fire categories, low-probability (15, 35) excluded


def _fire_span(g):
    t0 = pd.Timestamp(g["first"]).tz_convert("UTC").floor("D") - pd.Timedelta(days=1)
    t1 = pd.Timestamp(g["last"]).tz_convert("UTC").ceil("h")
    return t0, t1


def hrrr_stage(out, grids, threads=16):
    """Per fire hrrr.npz: times (T,) ns UTC, fields (T, 4, hh, wh) float16 [gust, u10, v10, rh2] on the
    HRRR 3 km window, valid hourly analyses every HRRR_STEP_H hours; plus the 12z+9h forecast for each
    day (fc_times, fc), and idx (H, W): flat index into the window for every 150 m cell."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from scipy.spatial import cKDTree
    import weather_features as WF
    out = Path(out)
    todo = [g for g in grids if not (out / "fires" / g["fire_id"] / "hrrr.npz").exists()
            and _fire_span(g)[1] >= pd.Timestamp("2014-08-01", tz="UTC")]
    if not todo:
        return
    sess = requests.Session()
    lat, lon = WF.hrrr_latlon(sess)
    need, per = {}, {}
    for g in todo:
        win = WF.hrrr_window(lat, lon, g["bounds_lonlat"])
        if win is None:
            continue
        t0, t1 = _fire_span(g)
        ana = list(pd.date_range(t0, t1, freq=f"{HRRR_STEP_H}h"))
        fc = [d + pd.Timedelta(hours=12) for d in pd.date_range(t0.floor("D"), t1.floor("D"))]
        r0, r1, c0, c1 = win
        per[g["fire_id"]] = dict(g=g, win=win, ana=ana, fc=fc,
                                 A=np.full((len(ana), 4, r1 - r0, c1 - c0), np.nan, np.float16),
                                 F=np.full((len(fc), 4, r1 - r0, c1 - c0), np.nan, np.float16))
        for i, t in enumerate(ana):
            need.setdefault((t, 0), []).append((g["fire_id"], "A", i))
        for i, t in enumerate(fc):
            need.setdefault((t, 9), []).append((g["fire_id"], "F", i))
    keys = sorted(need)
    print(f"HRRR: {len(per)} fires, {len(keys)} distinct files", flush=True)

    def fetch(key):
        t, f = key
        return key, WF.hrrr_fetch(sess, WF.HRRR.format(d=t.strftime("%Y%m%d"), h=t.hour, f=f))

    t_start = time.time()
    with ThreadPoolExecutor(threads) as ex:
        for k, (key, got) in enumerate(ex.map(fetch, keys)):
            if got:
                for fid, which, i in need[key]:
                    p = per[fid]
                    r0, r1, c0, c1 = p["win"]
                    for j, (name, _) in enumerate(WF.HRRR_FIELDS):
                        if name in got:
                            p[which][i, j] = got[name][r0:r1, c0:c1]
            for fid, _, _ in need[key]:          # write a fire as soon as its last file is in
                p = per[fid]
                p["left"] = p.get("left", len(p["ana"]) + len(p["fc"])) - 1
                if p["left"] == 0:
                    _save_hrrr(out, p, lat, lon, cKDTree)
            if k % 500 == 0:
                print(f"  HRRR {k}/{len(keys)} files ({(time.time() - t_start) / 60:.0f} min)", flush=True)


def _save_hrrr(out, p, lat, lon, cKDTree):
    g = p["g"]
    r0, r1, c0, c1 = p["win"]
    hlat, hlon = lat[r0:r1, c0:c1], lon[r0:r1, c0:c1]
    glon, glat = centres(g)
    cs = math.cos(math.radians(float(glat.mean())))
    _, idx = cKDTree(np.c_[hlat.ravel(), hlon.ravel() * cs]).query(np.c_[glat.ravel(), glon.ravel() * cs])
    np.savez_compressed(out / "fires" / g["fire_id"] / "hrrr.npz",
                        times=np.array([t.value for t in p["ana"]], np.int64), fields=p["A"],
                        fc_times=np.array([t.value + 9 * 3600 * 10 ** 9 for t in p["fc"]], np.int64), fc=p["F"],
                        vars=np.array([n for n, _ in WF_FIELDS()]), idx=idx.reshape(g["height"], g["width"]).astype(np.int32))


def WF_FIELDS():
    import weather_features as WF
    return WF.HRRR_FIELDS


GOES_BUCKETS = [  # (satellite, bucket, first day, last day)
    ("G16", "noaa-goes16", "2017-06-01", "2025-04-06"),
    ("G19", "noaa-goes19", "2025-04-07", "2030-01-01"),
    ("G17", "noaa-goes17", "2018-11-15", "2023-01-09"),
    ("G18", "noaa-goes18", "2023-01-10", "2030-01-01"),
]


def _goes_latlon(ds):
    """Latitude/longitude of the ABI fixed grid (GOES-R PUG formulas)."""
    p = ds.variables["goes_imager_projection"]
    H = p.perspective_point_height + p.semi_major_axis
    req, rpol = p.semi_major_axis, p.semi_minor_axis
    lon0 = math.radians(p.longitude_of_projection_origin)
    x = ds.variables["x"][:].astype(np.float64)
    y = ds.variables["y"][:].astype(np.float64)
    X, Y = np.meshgrid(x, y)
    a = np.sin(X) ** 2 + np.cos(X) ** 2 * (np.cos(Y) ** 2 + (req / rpol) ** 2 * np.sin(Y) ** 2)
    b = -2 * H * np.cos(X) * np.cos(Y)
    c = H ** 2 - req ** 2
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        sx = rs * np.cos(X) * np.cos(Y)
        sy = -rs * np.sin(X)
        sz = rs * np.cos(X) * np.sin(Y)
        lat = np.degrees(np.arctan((req / rpol) ** 2 * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
        lon = np.degrees(lon0 - np.arctan(sy / (H - sx)))
    return lat.astype(np.float32), lon.astype(np.float32)


def _goes_one(sess, bucket, sat, prefix, t, found, grids_ll, h5lock, netCDF4):
    """Fire pixels from the scan nearest to time t of one satellite (appended to found)."""
    import re
    r = sess.get(f"https://{bucket}.s3.amazonaws.com/", params={"list-type": 2, "prefix": prefix}, timeout=60)
    r.raise_for_status()
    keys = re.findall(r"<Key>([^<]+)</Key>", r.text)
    if not keys:
        return
    key = min(keys, key=lambda k: abs(int(k.split("_s")[1][9:11]) - t.minute))
    rr = sess.get(f"https://{bucket}.s3.amazonaws.com/{key}", timeout=120)
    rr.raise_for_status()
    nc = rr.content
    with h5lock:
        ds = netCDF4.Dataset("x", memory=nc)
        try:
            mask = np.asarray(np.ma.filled(ds.variables["Mask"][:], 0))
            powr = np.asarray(np.ma.filled(ds.variables["Power"][:], 0)) if "Power" in ds.variables \
                else np.zeros(mask.shape, np.float32)
            fire = np.isin(mask, list(GOES_FIRE_MASK))
            if fire.any() and (sat, mask.shape) not in grids_ll:
                grids_ll[(sat, mask.shape)] = _goes_latlon(ds)
        finally:
            ds.close()
    if fire.any():
        la, lo = grids_ll[(sat, mask.shape)]
        ts = pd.to_datetime(key.split("_s")[1][:13], format="%Y%j%H%M%S", utc=True)
        found.append((sat, ts, la[fire], lo[fire], mask[fire], powr[fire]))


def goes_stage(out, grids, threads=16):
    """Per fire goes.npz: every GOES ABI fire pixel (CONUS product, hourly scans from both the East and
    West satellites) inside the fire's box: t (ns UTC), lat, lon, mask category, power (MW), sat.
    Sparse, so the loader can make 'hours since GOES saw fire here' at any perimeter time."""
    import netCDF4
    import requests
    from concurrent.futures import ThreadPoolExecutor
    out = Path(out)
    todo = [g for g in grids if not (out / "fires" / g["fire_id"] / "goes.npz").exists()
            and _fire_span(g)[1] >= pd.Timestamp("2017-06-01", tz="UTC")]
    if not todo:
        return
    hours = {}
    for g in todo:
        t0, t1 = _fire_span(g)
        for t in pd.date_range(t0, t1, freq=f"{GOES_STEP_MIN}min"):
            hours.setdefault(t, []).append(g["fire_id"])
    boxes = {g["fire_id"]: g["bounds_lonlat"] for g in todo}
    left = {g["fire_id"]: 0 for g in todo}
    for fids in hours.values():
        for f in fids:
            left[f] += 1
    rows = {f: [] for f in boxes}
    sess = requests.Session()
    grids_ll = {}
    import threading
    h5lock = threading.Lock()              # netCDF4/HDF5 is not thread-safe: download in parallel, parse one at a time
    print(f"GOES: {len(todo)} fires, {len(hours)} distinct hours", flush=True)

    def one(t):
        found = []
        for sat, bucket, a, b in GOES_BUCKETS:
            if not (pd.Timestamp(a, tz="UTC") <= t <= pd.Timestamp(b, tz="UTC")):
                continue
            prefix = f"ABI-L2-FDCC/{t.year}/{t.dayofyear:03d}/{t.hour:02d}/"
            for attempt in range(3):          # transient network errors: retry this satellite-hour
                try:
                    found_before = len(found)
                    _goes_one(sess, bucket, sat, prefix, t, found, grids_ll, h5lock, netCDF4)
                    break
                except Exception as ex:  # noqa: BLE001
                    if attempt == 2:
                        print(f"  GOES {sat} {t}: {str(ex)[:150]}", flush=True)
                    del found[found_before:]
                    time.sleep(3 * (attempt + 1))
        return t, found


    t_start = time.time()
    with ThreadPoolExecutor(threads) as ex:
        for k, (t, found) in enumerate(ex.map(one, sorted(hours))):
            for fid in hours[t]:
                w, s, e, n = boxes[fid]
                for sat, ts, la, lo, m, pw in found:
                    ok = (la >= s) & (la <= n) & (lo >= w) & (lo <= e)
                    if ok.any():
                        rows[fid].append((sat, ts.value, la[ok], lo[ok], m[ok], pw[ok]))
                left[fid] -= 1
                if left[fid] == 0:
                    _save_goes(out, fid, rows.pop(fid))
            if k % 500 == 0:
                print(f"  GOES {k}/{len(hours)} hours ({(time.time() - t_start) / 60:.0f} min)", flush=True)


def _save_goes(out, fid, rows):
    if rows:
        n = [len(r[2]) for r in rows]
        arr = dict(t=np.repeat([r[1] for r in rows], n).astype(np.int64), lat=np.concatenate([r[2] for r in rows]),
                   lon=np.concatenate([r[3] for r in rows]), mask=np.concatenate([r[4] for r in rows]).astype(np.int16),
                   power=np.concatenate([r[5] for r in rows]).astype(np.float32),
                   sat=np.repeat([r[0] for r in rows], n))
    else:
        arr = dict(t=np.zeros(0, np.int64), lat=np.zeros(0, np.float32), lon=np.zeros(0, np.float32),
                   mask=np.zeros(0, np.int16), power=np.zeros(0, np.float32), sat=np.zeros(0, "<U3"))
    np.savez_compressed(out / "fires" / fid / "goes.npz", **arr)


def human_stage(out, grids):
    """OSM layers, grouped by Geofabrik region file and year so each ~2 GB snapshot downloads once."""
    import human_features as HF
    work = Path(out) / "osm_work"
    work.mkdir(parents=True, exist_ok=True)
    todo = [g for g in grids if not (Path(out) / "fires" / g["fire_id"] / "human.npz").exists()]
    for g in todo:
        g["fire_year"] = g["year"]
        try:
            g["_region"] = HF.region_file(g)
        except RuntimeError:
            g["_region"] = None
    todo = [g for g in todo if g["_region"]]
    todo.sort(key=lambda g: (g["_region"], g["year"]))
    current = None
    for k, g in enumerate(todo):
        grp = (g["_region"], g["year"])
        if current and grp != current:
            for f in work.glob(f"{current[0].replace('/', '_')}-{str(current[1])[2:]}0101*"):
                f.unlink()
        current = grp
        t0 = time.time()
        try:
            filt = HF.prepare_region(grp[0], g["year"], work)
            layers, counts = HF.rasterize_fire(HF.fire_features(filt, g, work), g)
            np.savez_compressed(Path(out) / "fires" / g["fire_id"] / "human.npz", layers=layers,
                                channels=np.array(HF.CHANNEL_NAMES), snapshot=np.array(f"{g['year']}-01-01"))
            print(f"[human {k + 1}/{len(todo)}] {g['fire_id']} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[human {k + 1}/{len(todo)}] {g['fire_id']}: ERROR {ex}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog")
    ap.add_argument("--out", default="data_my")
    ap.add_argument("--stages", default="all", help=f"comma list from {STAGES} or 'all'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fires", nargs="*")
    ap.add_argument("--years")
    a = ap.parse_args()
    os.environ["BUILD_WORKERS"] = str(a.workers)
    out = Path(a.out)
    (out / "fires").mkdir(parents=True, exist_ok=True)
    stages = STAGES[1:] if a.stages == "all" else a.stages.split(",")
    fires = pd.read_csv(Path(a.catalog) / "fires.csv")
    if a.years:
        y0, y1 = map(int, a.years.split("-"))
        fires = fires[fires.year.between(y0, y1)]
    if a.fires:
        fires = fires[fires.fire_id.isin(a.fires)]
    fires = fires.sort_values("final_acres", ascending=False)
    if a.limit:
        fires = fires.head(a.limit)
    grids, skipped = [], []
    for _, row in fires.iterrows():
        gp = out / "fires" / row.fire_id / "grid.json"
        if gp.exists():
            grids.append(json.loads(gp.read_text()))
            continue
        g = make_grid(row)
        if max(g["height"], g["width"]) > MAX_CELLS:
            skipped.append((row.fire_id, row["name"], g["height"], g["width"]))
            continue
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(json.dumps(g))
        grids.append(g)
    print(f"{len(grids)} fires to build, {len(skipped)} skipped as larger than {MAX_CELLS} cells a side", flush=True)
    pd.DataFrame(skipped, columns=["fire_id", "name", "height", "width"]).to_csv(out / "skipped_large.csv", index=False)
    pstages = [s for s in stages if s not in DATE_MAJOR]
    if pstages:
        t0 = time.time()
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(build_one, str(out), a.catalog, g, pstages) for g in grids]
            for k, f in enumerate(as_completed(futs)):
                fid, done = f.result()
                print(f"[{k + 1}/{len(grids)}] {fid}: {' '.join(done) or 'already done'} "
                      f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    if "hrrr" in stages:
        hrrr_stage(out, grids)
    if "goes" in stages:
        goes_stage(out, grids)
    if "human" in stages:
        human_stage(out, grids)
    status = []
    for g in grids:
        d = out / "fires" / g["fire_id"]
        status.append({"fire_id": g["fire_id"], "year": g["year"], "H": g["height"], "W": g["width"],
                       **{s: (d / f"{s}.npz").exists() for s in stages}})
    pd.DataFrame(status).to_csv(out / "build_status.csv", index=False)
    print("Done:", out / "build_status.csv")


if __name__ == "__main__":
    sys.exit(main())
