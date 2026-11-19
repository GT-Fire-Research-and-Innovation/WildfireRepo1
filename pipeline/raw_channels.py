"""
Upstream raw data for one fire, one channel at a time (driven by raw_grab.py).

"Raw" means the source's own records, values and grid, cut to the fire's region: the box around all its
perimeters pushed out 32 km on every side (area), so any 64 km tile centred anywhere in the fire, on its
ignition point or on its front on any day, can be cut from these files without fetching again. Over a time
window that also holds the days the fire was active before its first perimeter and after its last (FIRMS, see
span / fire_extent), plus a day on each side. Whole-CONUS sources (HRRR, GFS, GOES, DEM, OSM) are cropped on
their native grid with untouched values; every file's source URL is logged in the manifest.

Channels (cheap -> heavy) and what each writes into fires/<fire_id>/<channel>/:
  history      Interagency Fire Perimeter History, every prior fire in the box   history.geojson.gz
  perimeters   NIFC GeoMAC / NIFS / Operational Data Archive records of the fire  records.geojson.gz
  firms        NASA FIRMS VIIRS active fire (SP, + NRT fill 2024+)               <SOURCE>.csv.gz
  gridmet      GRIDMET daily, 13 variables incl. pr/tmmn/sph/pdsi (THREDDS NCSS) gridmet_<var>.nc
  ics209       ICS-209-PLUS incident + sitrep rows (fires <= 2020)              incidents.csv, sitreps.csv
  suppression  NIFS / ODA EventLine + EventPolygon, every archived version (2018+) event_lines.geojson.gz, event_polygons.geojson.gz
  tessera      GeoTessera v1.1 reference only (no embedding data)                tessera_ref.json
  terrain      Copernicus GLO-30 DEM, native 1 arc-second window                 dem_glo30.tif
  landfire     LANDFIRE image services, native 30 m EPSG:5070 (fires 2017+)      landfire_<version>_<layer>.tif
  lulc         Esri / Impact Observatory 10 m annual land cover (fires 2017+)    io_lulc_<year>_<tile>.tif
  landcover    MODIS MCD12Q1 annual land cover, sinusoidal 463 m (Earthdata)     mcd12q1_<year>.nc
  vegetation   VIIRS VNP13A1 16-day vegetation indices, sinusoidal 463 m (Earthdata) vnp13a1.nc
  goes         GOES-16/17/18/19 ABI L2 FDCC, hourly scan, fixed-grid crop        goes_<sat>.nc
  hrrr         HRRR gust/u10/v10/rh2, 3-hourly analyses + 12z+9h forecasts       hrrr_ana.nc, hrrr_fc.nc
  hrrr_next    HRRR 12z run every 3 h out to 36 h: + t2/q2/apcp                  hrrr_next.nc
  gfs          NOAA GFS 0.25 deg 12z run every 3 h out to 10 days (Earth Engine)   gfs.nc
  osm          OpenStreetMap 1 January snapshot of the fire's year, full extract osm_<YYYY>-01-01.osm.pbf
  viirs        VIIRS L1B swaths day + night: I1-I5, M11, geolocation (Earthdata) viirs.nc (a group per granule)

GeoJSON is gzipped with a fixed header (same features, same bytes): json.load(gzip.open(p)) or
geopandas.read_file('/vsigzip/' + p).

lulc, landcover, vegetation, hrrr_next, gfs and viirs bring in what TS-SatFire (Zhao et al. 2024) has and the
other channels lacked: its VIIRS_Day / VIIRS_Night bands, its FirePred bands (NDVI, EVI2, gridMET pr / tmmn /
sph / pdsi, MODIS LC_Type1, GFS forecasts: TS-SatFire's own source in gfs, at 3 km and hourly in hrrr_next) and
its ESRI_LULC map.

Per-fire channels: grab_<channel>(fire, out_dir, ctx) -> (status, sources), status "ok" or "empty".
Group channels (goes, hrrr, hrrr_next, gfs, osm, viirs) share downloads across fires: grab_<channel>_many(fires, ctx)
yields (fire, status, sources, error). Secrets (FIRMS map key, Earthdata token, Earth Engine key, HF token) never
reach the manifest.
"""
import contextlib
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
            "landfire", "lulc", "landcover", "vegetation", "goes", "hrrr", "hrrr_next", "gfs", "osm", "viirs"]
GROUP = {"goes", "hrrr", "hrrr_next", "gfs", "osm", "viirs"}
EARTHDATA_CHANNELS = {"landcover", "vegetation", "viirs"}        # need EARTHDATA_TOKEN
# Channels whose files depend on the fire's time window. A manifest row records the window it was fetched for
# (window_label); when the window changes (catalog/fire_extents.csv), done_ids sends the fire back to the queue.
WINDOW_CHANNELS = {"firms", "gridmet", "goes", "hrrr", "hrrr_next", "gfs", "vegetation", "viirs"}
# Channels cut to the fire's region (area). perimeters keeps the build's box (it keeps only the fire's own
# records, which all lie in it) and ics209 matches incidents by the build's box (a wider one would match
# neighbours); the manifest records the region, so a region change sends the fire back to the queue.
REGION_CHANNELS = {"history", "firms", "gridmet", "suppression", "tessera", "terrain", "landfire", "lulc", "landcover",
                   "vegetation", "goes", "hrrr", "hrrr_next", "gfs", "osm", "viirs"}
GOES_VARS = ("Mask", "Power", "Area", "Temp", "DQF")
HRRR_START = "2014-08-01"            # first HRRR files on AWS used by the build
GFS_START = "2015-07-01"             # first NOAA/GFS0P25 image in Earth Engine
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


# ------------------------------------------------------------------------------------- time window
# A fire often burns before its first mapped perimeter, and on after its last. The raw window therefore runs
# from the earlier of the first perimeter day and the first FIRMS-active day that leads into it, to the later
# of the last perimeter day and the last active day that follows it, plus one day on each side. FIRMS itself
# is fetched over a fixed span around the perimeter days (firms_days), so the window never depends on itself.

REGION_BUFFER_KM = 32.0              # half a 64 km tile: the region reaches this far past the perimeters' box
FIRMS_BEFORE, FIRMS_AFTER = 14, 14   # FIRMS days fetched around the perimeter days
ACTIVE_MIN_DETECTIONS = 2            # an active day: at least this many detections that are not low confidence
ACTIVE_MAX_GAP_DAYS = 2              # activity joins the fire across at most this many quiet days (clouds, gaps)
ACTIVE_PAD_DEG = 0.01                # around the perimeters' own box (~1 km: a 375 m pixel centre can sit outside)
WINDOW_PAD_DAYS = 1                  # added on both sides
EXTENTS_FILE = "catalog/fire_extents.csv"
IGNITION_OVERPASS_MIN = 15           # detections this close to the earliest one belong to the same overpass


def buffer_box(bounds, km):
    """bounds (lon/lat) pushed out km on every side; longitudes are widened for the latitude farthest from the
    equator, so the margin is at least km everywhere in the box."""
    w, s, e, n = bounds
    dlat = km / 110.574
    far = min(max(abs(s - dlat), abs(n + dlat)), 89.0)
    dlon = km / (111.320 * math.cos(math.radians(far)))
    return [w - dlon, s - dlat, e + dlon, n + dlat]


def area(fire):
    """The region every spatial channel is cut to (lon/lat box): the perimeters' own box, grown by the
    detection margin (ACTIVE_PAD_DEG, so an ignition point found from FIRMS lies inside it) and then by
    REGION_BUFFER_KM on every side, and never smaller than the build's box. A tile of 2 x REGION_BUFFER_KM
    centred anywhere in the fire lies inside it. Depends on the perimeters only, so it is known before any
    channel runs."""
    g = fire["bounds"]
    base = fire.get("perim_bounds") or g
    p = ACTIVE_PAD_DEG
    w, s, e, n = buffer_box([base[0] - p, base[1] - p, base[2] + p, base[3] + p], REGION_BUFFER_KM)
    return [min(w, g[0]), min(s, g[1]), max(e, g[2]), max(n, g[3])]


def region_label(channel, fire):
    """The region a channel's files cover, as the manifest records it ('' for channels on the build's box)."""
    return ",".join(f"{v:.4f}" for v in area(fire)) if channel in REGION_CHANNELS else ""


def perimeter_days(fire):
    """(first, last) perimeter day as naive UTC midnights."""
    return _ts(fire["first"]).tz_convert(None).normalize(), _ts(fire["last"]).tz_convert(None).normalize()


def span(fire):
    """The fire's time window, UTC: 00:00 of its first day to 23:00 of its last. The days come from
    fire["window"] (catalog/fire_extents.csv, already padded); without one, the perimeter days alone, one
    day added on each side."""
    import pandas as pd
    w = fire.get("window")
    if w:
        d0, d1 = pd.Timestamp(w[0]), pd.Timestamp(w[1])
    else:
        d0, d1 = perimeter_days(fire)
        d0, d1 = d0 - pd.Timedelta(days=WINDOW_PAD_DAYS), d1 + pd.Timedelta(days=WINDOW_PAD_DAYS)
    return d0.tz_localize("UTC"), d1.tz_localize("UTC") + pd.Timedelta(hours=23)


def days(fire, before=14, after=2):
    """Calendar days of the daily channels (gridmet): `before` days before the first perimeter day to `after`
    days after the last (14 / 2, as the build's days_of), moved out as far as span() is."""
    import pandas as pd
    t0, t1 = span(fire)
    d0 = t0.tz_convert(None).normalize() - pd.Timedelta(days=before - WINDOW_PAD_DAYS)
    d1 = t1.tz_convert(None).normalize() + pd.Timedelta(days=after - WINDOW_PAD_DAYS)
    return pd.date_range(d0, d1)


def firms_days(fire):
    """Days the firms channel fetches: FIRMS_BEFORE before the first perimeter day to FIRMS_AFTER after the
    last. Fixed by the perimeters, since the window is computed from what FIRMS finds here."""
    import pandas as pd
    d0, d1 = perimeter_days(fire)
    return pd.date_range(d0 - pd.Timedelta(days=FIRMS_BEFORE), d1 + pd.Timedelta(days=FIRMS_AFTER))


def active_days(df, bounds, pad=ACTIVE_PAD_DEG, min_n=ACTIVE_MIN_DETECTIONS):
    """Sorted UTC days (naive midnights) with at least min_n FIRMS detections inside bounds (+ pad) that are
    neither low confidence nor, where the source says so, a static land source or offshore (type 2 / 3)."""
    import pandas as pd
    if df is None or not len(df):
        return []
    m = _fire_detections(df, bounds, pad)
    count = pd.to_datetime(df.loc[m, "acq_date"]).dt.normalize().value_counts()
    return sorted(count[count >= min_n].index)


def _fire_detections(df, bounds, pad=ACTIVE_PAD_DEG):
    """Mask of detections that count for the fire: inside bounds (+ pad), not low confidence, not a static
    land source or offshore (type 2 / 3) where the source has a type column."""
    import pandas as pd
    w, s, e, n = bounds
    m = df["latitude"].between(s - pad, n + pad) & df["longitude"].between(w - pad, e + pad)
    if "confidence" in df:
        m &= ~df["confidence"].astype(str).str.lower().str.startswith("l")
    if "type" in df:
        m &= ~pd.to_numeric(df["type"], errors="coerce").isin([2, 3])
    return m


def first_perimeter_point(arrival, transform, times):
    """(time, lat, lon) of the first mapped perimeter: the centre of the cells the build's labels give the
    earliest arrival (arrival = index into times, -1 unburned), or None."""
    import numpy as np
    import pandas as pd
    a = np.asarray(arrival)
    if not (a >= 0).any():
        return None
    k = int(a[a >= 0].min())
    r, c = np.nonzero(a == k)
    dx, _, x0, _, dy, y0 = transform
    return pd.Timestamp(int(times[k]), tz="UTC"), float(y0 + (r.mean() + 0.5) * dy), float(x0 + (c.mean() + 0.5) * dx)


def ignition_point(fire, firms_df=None, active=(), first_perimeter=None):
    """(lat, lon, time UTC, method): the fire's earliest observation, known at its start, as the centre a
    tile can be cut around without seeing the future. The earlier of: the first VIIRS overpass that saw it
    (detections of the fire within IGNITION_OVERPASS_MIN minutes of the first one, on the first active day of
    its chained window: 'firms'; on any day of the perimeter span when no day is active: 'firms_single'), and
    its first mapped perimeter ('first_perimeter', from the build's labels). Without either, the centre of the
    perimeters' box ('perimeter_box_centre': uses the final extent, so not leak-free)."""
    import pandas as pd
    cands = []
    bounds = fire.get("perim_bounds") or fire["bounds"]
    if firms_df is not None and len(firms_df):
        d = firms_df[_fire_detections(firms_df, bounds)]
        day = pd.to_datetime(d["acq_date"]).dt.normalize()
        hhmm = pd.to_numeric(d["acq_time"], errors="coerce").fillna(0).astype(int) if "acq_time" in d \
            else pd.Series(0, index=d.index)
        t = (day + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="min")).dt.tz_localize("UTC")
        p0, p1 = perimeter_days(fire)
        a0, a1 = activity_window(p0, p1, list(active))
        sel, method = day.isin(set(active)) & day.between(a0, a1), "firms"
        if not sel.any():
            sel, method = day.between(p0 - pd.Timedelta(days=1), p1), "firms_single"
        if sel.any():
            t0 = t[sel].min()
            g = sel & (t <= t0 + pd.Timedelta(minutes=IGNITION_OVERPASS_MIN))
            cands.append((t0, float(d.loc[g, "latitude"].mean()), float(d.loc[g, "longitude"].mean()), method))
    if first_perimeter is not None:
        cands.append((*first_perimeter, "first_perimeter"))
    if cands:
        t0, lat, lon, method = min(cands, key=lambda c: c[0])
        return lat, lon, t0, method
    w, s, e, n = bounds
    return (s + n) / 2, (w + e) / 2, _ts(fire["first"]), "perimeter_box_centre"


def activity_window(first_day, last_day, active, max_gap=ACTIVE_MAX_GAP_DAYS):
    """(start, end): the perimeter days, stretched over active days that chain onto them with at most
    max_gap quiet days between links (activity weeks away belongs to another fire or a false alarm)."""
    import pandas as pd
    step = pd.Timedelta(days=max_gap + 1)
    s, e = first_day, last_day
    for d in sorted((d for d in active if d < first_day), reverse=True):
        if s - d > step:
            break
        s = d
    for d in sorted(d for d in active if d > last_day):
        if d - e > step:
            break
        e = d
    return s, e


def fire_extent(fire, firms_df=None, first_perimeter=None):
    """One row of catalog/fire_extents.csv: the fire's time window, ignition point and region. firms_df: the
    fire's FIRMS detections (None when it has none); first_perimeter: (time, lat, lon) of its first mapped
    perimeter (first_perimeter_point) or None. The activity test uses the perimeters' own box (perim_bounds),
    not the padded grid box, so a neighbour burning in the margin does not stretch this fire's window."""
    import pandas as pd
    p0, p1 = perimeter_days(fire)
    act = active_days(firms_df, fire.get("perim_bounds") or fire["bounds"])
    a0, a1 = activity_window(p0, p1, act)
    pad = pd.Timedelta(days=WINDOW_PAD_DAYS)
    lat, lon, t0, how = ignition_point(fire, firms_df, act, first_perimeter)
    w, s, e, n = area(fire)
    return {"fire_id": fire["fire_id"], "start_day": (a0 - pad).date().isoformat(), "end_day": (a1 + pad).date().isoformat(),
            "first_perimeter_day": p0.date().isoformat(), "last_perimeter_day": p1.date().isoformat(),
            "active_days_before": (p0 - a0).days, "active_days_after": (a1 - p1).days, "active_days": len(act),
            "window_rule": "perimeters+firms" if firms_df is not None else "perimeters",
            "ignition_lat": round(lat, 5), "ignition_lon": round(lon, 5),
            "ignition_time_utc": t0.strftime("%Y-%m-%dT%H:%MZ"), "ignition_method": how,
            "region_west": round(w, 5), "region_south": round(s, 5), "region_east": round(e, 5), "region_north": round(n, 5)}


def attach_extents(fires, table):
    """Put each fire's (start_day, end_day) from a fire_extents table ({fire_id: row}) on fire["window"] and
    its ignition point on fire["ignition"] (lat, lon, time, method)."""
    for f in fires:
        r = table.get(f["fire_id"]) if table else None
        if r is not None:
            f["window"] = (str(r["start_day"]), str(r["end_day"]))
            if str(r.get("ignition_lat", "")) not in ("", "nan"):
                f["ignition"] = (float(r["ignition_lat"]), float(r["ignition_lon"]), str(r["ignition_time_utc"]),
                                 str(r["ignition_method"]))
    return fires


def window_label(channel, fire):
    """The days a channel's files cover for this fire, as the manifest records them ('' for channels whose
    files do not depend on time)."""
    if channel == "firms":
        d = firms_days(fire)
    elif channel == "gridmet":
        d = days(fire)
    elif channel in WINDOW_CHANNELS:
        t0, t1 = span(fire)
        return f"{t0.date().isoformat()}/{t1.date().isoformat()}"
    else:
        return ""
    return f"{d[0].date().isoformat()}/{d[-1].date().isoformat()}"


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
    if channel == "gfs" and t1 < pd.Timestamp(GFS_START, tz="UTC"):
        return "GFS 0.25 deg in Earth Engine starts 2015-07-01"
    if channel == "lulc" and y < LULC_FIRST_YEAR:
        return "Esri 10 m land cover starts 2017"
    if channel == "osm":
        import human_features as HF
        try:
            HF.region_file({"bounds_lonlat": area(fire)})
        except RuntimeError as ex:
            return str(ex)
    return None


def firms_calls(fire):
    """(source, first day, n days) for the standard-product requests: 5-day calls over firms_days()."""
    ds = firms_days(fire)
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
                "EARTHDATA_TOKEN": "{EARTHDATA_TOKEN}", "GEE_SERVICE_ACCOUNT_KEY": "{GEE_KEY}"}


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
    text = re.sub(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", "{GEE_KEY}", text, flags=re.S)
    text = re.sub(r'("private_key(?:_id)?"\s*:\s*")[^"]*"', r'\1{GEE_KEY}"', text)
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
    if channel == "gfs":
        try:
            import ee  # noqa: F401
        except ImportError:
            raise SystemExit("gfs reads NOAA/GFS0P25 from Google Earth Engine: pip install earthengine-api")
        if not (os.environ.get("GEE_SERVICE_ACCOUNT_KEY") or (Path.home() / ".config" / "earthengine" / "credentials").exists()):
            raise SystemExit(GEE_HELP)
    if channel == "landcover":
        try:
            import pyhdf.SD  # noqa: F401
        except ImportError:
            raise SystemExit("landcover reads MCD12Q1 HDF4 tiles: pip install pyhdf")


# Files that replaced an older layout: rows that list the old names (plain .geojson, one file per VIIRS granule)
# are fetched again, or converted in place by raw_grab.py --repack.
EXPECTED_FILES = {"history": {"history.geojson.gz"}, "perimeters": {"records.geojson.gz"},
                  "suppression": {"event_lines.geojson.gz", "event_polygons.geojson.gz"}, "viirs": {"viirs.nc"}}


def expected_files(channel):
    """Files an 'ok' fire must list in its manifest row to count as done (None: any 'ok' row counts). When a
    channel grows new files (gridmet gained pr / tmmn / sph / pdsi), its next run fetches again just the
    fires that miss them."""
    if channel == "gridmet":
        return {f"gridmet_{v}.nc" for v in gridmet_vars()}
    return EXPECTED_FILES.get(channel)


def shared_files(channel, ctx):
    """[(local file, repo path)] uploaded once per repo with the first batch."""
    if channel == "ics209":
        _ics_tables(ctx)
        return [(ctx.cache / "ics209" / "ics209-plus-wildfire.zip", "shared/ics209/ics209plus-wildfire.zip")]
    return []


# ------------------------------------------------------------------------------------- helpers

GZIP_LEVEL = 6


@contextlib.contextmanager
def gzip_writer(path):
    """A gzip stream with no file name or time in its header, so the same content always gives the same bytes
    (a rerun that finds nothing new uploads nothing new)."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=GZIP_LEVEL,
                                                mtime=0) as gz:
        yield gz


def gzip_copy(src, dst):
    """dst = gzip of src, streamed, with gzip_writer's fixed header."""
    with open(src, "rb") as fi, gzip_writer(dst) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)


def _write_geojson(path, feats):
    """A FeatureCollection as <name>.geojson.gz (gzip_writer)."""
    import io
    with gzip_writer(path) as gz, io.TextIOWrapper(gz, encoding="utf-8") as fh:
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
    feats = arcgis_box_features(ctx.sess, BM.HIST_URL, area(fire), where=where, chunk=100)
    _write_geojson(out / "history.geojson.gz", feats)
    return ("ok" if feats else "empty"), [arcgis_query_url(BM.HIST_URL, area(fire), where=where)]


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
    _write_geojson(out / "records.geojson.gz", keep)
    return ("ok" if keep else "empty"), sources


def grab_firms(fire, out, ctx):
    """FIRMS area API, 5-day calls; NRT fills the days the standard product has not reached (2024+)."""
    import pandas as pd
    from live import firms
    frames, sources = {}, []

    def call(src, day, k):
        ctx.throttle(0.15)                              # 5000 transactions / 10 min per key
        df = firms.get(src, area(fire), k, day)
        bbox = ",".join(f"{v:.4f}" for v in area(fire))
        sources.append(redact(firms.API.format(key="KEY", src=src, bbox=bbox, days=k) + f"/{day}"))
        if len(df):
            frames.setdefault(src, []).append(df)

    for src, day, k in firms_calls(fire):
        call(src, day, k)
    ds = firms_days(fire)
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
        url = gridmet_url(short, var, area(fire), d0, ds[-1], WF.PAD_DEG)
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
    bbox = tuple(area(fire))
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
    _write_geojson(out / "event_lines.geojson.gz", lines)
    _write_geojson(out / "event_polygons.geojson.gz", polys)
    return ("ok" if lines or polys else "empty"), sources


def grab_tessera(fire, out, ctx):
    ref = {"fire_id": fire["fire_id"], "fire_year": int(fire["year"]), "embedding_year": int(fire["year"]) - 1,
           "bbox_lonlat": list(area(fire)), "dataset": "GeoTessera v1.1 (128-d annual embeddings, 10 m)",
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
    w, s, e, n = area(fire)
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
        url = landfire_export_url(v, layer, area(fire), info)
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
                      json={"collections": [LULC_COLLECTION], "bbox": list(area(fire)), "limit": 100,
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
                                   transform_bounds(4326, src.crs, *area(fire), densify_pts=21))
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
    win = ED.sinu_window(area(fire))
    grans = [g for g in ED.cmr_granules(ctx.sess, "MCD12Q1", f"{yr}-01-01", f"{yr}-12-31T23:59:59", area(fire))
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
    win = ED.sinu_window(area(fire))
    grans = [g for g in ED.cmr_granules(ctx.sess, VEG_PRODUCT, t0 - pd.Timedelta(days=16), t1, area(fire))
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
                            win = _window(lat, lon, area(f))
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
            wins[fid] = WF.hrrr_window(lat, lon, area(f))
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
        groups.setdefault((HF.region_file({"bounds_lonlat": area(f)}), int(f["year"])), []).append(f)
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
            w, s, e, n = area(f)
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
            wins[fid] = WF.hrrr_window(lat, lon, area(f))
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

# ------------------------------------------------------------------------------------- GFS (Earth Engine)

GFS_COLLECTION = "NOAA/GFS0P25"
GFS_LEADS = tuple(range(3, 241, 3))          # every 3 h out to 10 days
# Leads the collection holds (catalog page): 3-hourly to 36 h until 2017-07-09; from then hourly to 120 h,
# 3-hourly to 240 h (and on to 384 h, 12-hourly until 2021-06-11, 3-hourly after).
GFS_MAX_LEAD = (("2015-07-01", 36), ("2017-07-09", 240))
GFS_MAX_BANDS = 990                          # bands per Earth Engine request (22 per image since 2025-01-15)
GFS_MAX_CELLS = 2500                         # 0.25 deg cells per request: fires of one run share a request
GFS_UNITS = {
    "temperature_2m_above_ground": "degC", "dew_point_temperature_2m_above_ground": "degC",
    "maximum_temperature_2m_above_ground": "degC", "minimum_temperature_2m_above_ground": "degC",
    "specific_humidity_2m_above_ground": "kg kg-1", "relative_humidity_2m_above_ground": "%",
    "u_component_of_wind_10m_above_ground": "m s-1", "v_component_of_wind_10m_above_ground": "m s-1",
    "u_component_of_wind_planetary_boundary_layer": "m s-1", "v_component_of_wind_planetary_boundary_layer": "m s-1",
    "gust": "m s-1", "total_precipitation_surface": "kg m-2", "precipitation_rate": "kg m-2 s-1",
    "precipitable_water_entire_atmosphere": "kg m-2", "total_cloud_cover_entire_atmosphere": "%",
    "downward_shortwave_radiation_flux": "W m-2", "downward_longwave_radiation_flux": "W m-2",
    "upward_shortwave_radiation_flux": "W m-2", "upward_longwave_radiation_flux": "W m-2",
    "planetary_boundary_layer_height": "m", "haines_index": "1", "ventilation_rate": "m2 s-1"}
GEE_HELP = """gfs reads NOAA/GFS0P25 from Google Earth Engine (free for research and non-commercial use):
  1. Register a Google Cloud project for Earth Engine: https://code.earthengine.google.com/register
  2. Either create a service account in that project, give it the 'Earth Engine Resource Viewer' role and put
     its JSON key in GEE_SERVICE_ACCOUNT_KEY (a Kaggle secret of that name), or run `earthengine authenticate`
     on this machine.
  3. Set GEE_PROJECT to the project id."""


def gfs_max_lead(run):
    import pandas as pd
    return [h for d, h in GFS_MAX_LEAD if _ts(run) >= pd.Timestamp(d, tz="UTC")][-1]


def gfs_times(fire):
    """[(12z run, lead hours)]: each day of span() the 12z run's forecasts every 3 h, out to 10 days where
    the collection has them (36 h before 2017-07-09)."""
    _, fc = hrrr_times(fire)
    return [(run, h) for run in fc if run >= _ts(GFS_START) for h in GFS_LEADS if h <= gfs_max_lead(run)]


def gee_init():
    """Earth Engine on its high-volume endpoint: a service account's JSON key (GEE_SERVICE_ACCOUNT_KEY) or this
    machine's `earthengine authenticate` credentials, in GEE_PROJECT (default: the key's own project)."""
    import ee
    key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    project = os.environ.get("GEE_PROJECT") or None
    url = "https://earthengine-highvolume.googleapis.com"
    if key:
        info = json.loads(key)
        cred, project = ee.ServiceAccountCredentials(info["client_email"], key_data=key), project or info.get("project_id")
    else:
        cred = "persistent"
    try:
        ee.Initialize(cred, project=project, url=url)             # earthengine-api 1.x
    except TypeError:
        ee.Initialize(cred, project=project, opt_url=url)         # 0.1.x
    return ee


def gfs_grid(ee):
    """(crs, crs_transform (a, b, c, d, e, f), width, height) of the collection's native 0.25 deg grid."""
    band = ee.ImageCollection(GFS_COLLECTION).first().getInfo()["bands"][0]
    w, h = band["dimensions"]
    return band["crs"], tuple(float(v) for v in band["crs_transform"]), int(w), int(h)


def _east_lon(transform, width):
    """True when the grid's longitudes run 0..360 (then fire longitudes are shifted to match)."""
    a, _, c, *_ = transform
    return c > -90 and c + a * width > 270


def grid_window(transform, width, height, bounds):
    """(r0, r1, c0, c1) of the cells covering bounds (lon/lat) on a regular lon/lat grid, snapped outward."""
    from affine import Affine
    w, s, e, n = bounds
    if _east_lon(transform, width):
        w, e = w % 360, e % 360
    return pixel_window(Affine(*transform), width, height, (w, s, e, n))


def grid_latlon(transform, width, win):
    """Cell-centre lat / lon (2-D, longitudes in -180..180) of a window."""
    import numpy as np
    a, _, c, _, e, f = transform
    r0, r1, c0, c1 = win
    lon = c + a * (np.arange(c0, c1) + 0.5)
    lon = np.where(lon > 180, lon - 360, lon)
    lat = f + e * (np.arange(r0, r1) + 0.5)
    return np.meshgrid(lat, lon, indexing="ij")


def _union(w1, w2):
    return min(w1[0], w2[0]), max(w1[1], w2[1]), min(w1[2], w2[2]), max(w1[3], w2[3])


def gfs_clusters(wins, max_cells=GFS_MAX_CELLS):
    """[(union window, [fire_id])]: fires (fire_id -> window) grouped greedily, west to east, so one request
    covers each group and no group's window exceeds max_cells."""
    groups = []
    for fid, w in sorted(wins.items(), key=lambda kv: (kv[1][2], kv[1][0], kv[0])):
        for g in groups:
            u = _union(g[0], w)
            if (u[1] - u[0]) * (u[3] - u[2]) <= max_cells:
                g[0] = u
                g[1].append(fid)
                break
        else:
            groups.append([w, [fid]])
    return [(tuple(g[0]), g[1]) for g in groups]


def gfs_fetch(ee, run, leads, win, grid):
    """{lead: {band: 2-D array}} of one run over win (rows / cols of the native grid), one computePixels
    request: every band each image has, values as Earth Engine stores them. Leads the collection lacks are
    absent; a run it lacks gives {}."""
    crs, (a, b, c, d, e, f), _, _ = grid
    r0, r1, c0, c1 = win
    coll = (ee.ImageCollection(GFS_COLLECTION)
            .filter(ee.Filter.eq("creation_time", int(run.value // 10**6)))
            .filter(ee.Filter.inList("forecast_hours", list(leads))))

    def tag(im):        # band -> f<lead>__band, so toBands' own prefix (the image id) does not matter
        pre = ee.String("f").cat(ee.Number(im.get("forecast_hours")).int().format("%03d")).cat("__")
        return im.rename(im.bandNames().map(lambda bn: pre.cat(ee.String(bn))))

    req = {"expression": coll.map(tag).toBands(), "fileFormat": "NUMPY_NDARRAY",
           "grid": {"dimensions": {"width": c1 - c0, "height": r1 - r0}, "crsCode": crs,
                    "affineTransform": {"scaleX": a, "shearX": b, "translateX": c + c0 * a,
                                        "shearY": d, "scaleY": e, "translateY": f + r0 * e}}}
    for attempt in range(6):
        try:
            arr = ee.data.computePixels(req)
            break
        except Exception as ex:  # noqa: BLE001
            msg = str(ex).lower()
            if "no bands" in msg or "zero bands" in msg:
                return {}
            if attempt == 5 or not any(k in msg for k in ("429", "too many", "quota", "503", "500", "timed out", "deadline")):
                raise
            time.sleep(5 * 2 ** attempt)
    out = {}
    for name in arr.dtype.names or ():
        m = re.search(r"f(\d{3})__(\w+)$", name)
        if m:
            out.setdefault(int(m.group(1)), {}).setdefault(m.group(2), arr[name])
    return out


def _lead_chunks(leads, max_bands=GFS_MAX_BANDS, per_image=len(GFS_UNITS)):
    k = max(1, max_bands // per_image)
    return [tuple(leads[i:i + k]) for i in range(0, len(leads), k)]


def grab_gfs_many(fires, ctx, chunk=200):
    """NOAA GFS 0.25 deg (TS-SatFire's forecast source) from Earth Engine's NOAA/GFS0P25: for each day of
    span() the 12z run's forecasts every 3 h out to 10 days (gfs_times), every band each image has, on the
    native grid (no resampling; values as Earth Engine serves them, temperatures in degC), one series per fire:
    gfs.nc with time = valid time, lead_h and run_time per step. Fires of the same run share requests."""
    import numpy as np
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    ee = ctx.memo("gee", gee_init)
    grid = ctx.memo("gfs_grid", lambda: gfs_grid(ee))
    crs, tr, width, height = grid

    for group in _chunks(fires, chunk):
        if ctx.expired():
            return
        need, left, wins, series, got_n, by_id, errs = {}, {}, {}, {}, {}, {}, {}
        for f in group:
            fid = f["fire_id"]
            runs = sorted({r for r, _ in gfs_times(f)})
            wins[fid] = grid_window(tr, width, height, area(f))
            if not runs or wins[fid] is None:
                yield f, "empty", [], None
                continue
            by_id[fid], left[fid], got_n[fid], errs[fid] = f, len(runs), [], []
            for r in runs:
                need.setdefault(r, []).append(fid)

        def fetch(run):
            leads = [h for h in GFS_LEADS if h <= gfs_max_lead(run)]
            out = []
            for cwin, ids in gfs_clusters({fid: wins[fid] for fid in need[run]}):
                data, err = {}, None
                try:
                    for lc in _lead_chunks(leads):
                        for h, bands in gfs_fetch(ee, run, lc, cwin, grid).items():
                            data.setdefault(h, {}).update(bands)
                except Exception as ex:  # noqa: BLE001  (reported for the fires of this run; the others go on)
                    data, err = {}, redact(f"{run:%Y-%m-%dT%HZ}: {type(ex).__name__}: {ex}")[:300]
                out.append((cwin, ids, data, err))
            return run, out

        stopped = False
        with ThreadPoolExecutor(ctx.threads) as ex:
            for run, out in _windowed_map(ex, fetch, sorted(need), 2 * ctx.threads):
                src = f"ee:{GFS_COLLECTION} creation_time={run.strftime('%Y-%m-%dT%H:%MZ')}"
                for cwin, ids, data, err in out:
                    for fid in ids:
                        if err:
                            errs[fid].append(err)
                        r0, r1, c0, c1 = wins[fid]
                        sub = (slice(r0 - cwin[0], r1 - cwin[0]), slice(c0 - cwin[2], c1 - cwin[2]))
                        if data and fid not in series:
                            lat, lon = grid_latlon(tr, width, wins[fid])
                            series[fid] = NcSeries(
                                ctx.stage(fid) / "gfs.nc", lat, lon,
                                {"title": f"NOAA GFS 0.25 deg, 12z run, forecasts every 3 h out to 10 days, cropped to {fid}",
                                 "fire_id": fid, "source": f"Google Earth Engine {GFS_COLLECTION} (NOAA NCEP GFS)",
                                 "crs": crs, "crs_transform": " ".join(f"{v:g}" for v in tr),
                                 "window_rows_cols": f"{r0}:{r1},{c0}:{c1}",
                                 "total_precipitation_surface": "accumulated over the last ((lead_h - 1) % 6) + 1 hours "
                                                                "(Earth Engine catalog); sum the 6-hourly leads for totals",
                                 "note": "values as Earth Engine serves them (temperatures in degC); bands added to the "
                                         "collection on 2025-01-15 appear from then on"})
                        for h in sorted(data):
                            valid = run + pd.Timedelta(hours=h)
                            series[fid].append(
                                valid.value, f"{src} forecast_hours={h}",
                                {b: (np.asarray(v)[sub], np.asarray(v).dtype, {"units": GFS_UNITS.get(b, "")})
                                 for b, v in sorted(data[h].items())},
                                extra={"lead_h": (h, "i2", {"long_name": "forecast lead", "units": "hours"}),
                                       "run_time": (run.value, "i8", {"long_name": "model run (initialisation) time",
                                                                      "units": "nanoseconds since 1970-01-01 00:00:00 UTC"})})
                        if data:
                            got_n[fid].append(src)
                        left[fid] -= 1
                        if left[fid] == 0:
                            if fid in series:
                                series.pop(fid).close()
                            e = errs[fid]
                            note = f"{len(e)} run(s) failed: {e[0]}" if e else None
                            yield by_id[fid], ("ok" if got_n[fid] else "error" if e else "empty"), sorted(got_n[fid]), note
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
VIIRS_FILE = "viirs.nc"          # one per fire: a group per granule, look-up tables once in /luts
VIIRS_MASK_PAD_KM = 2.0          # pixels up to this far outside the region are kept (edge resampling); beyond: fill
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
            for g in ED.cmr_granules(sess, prefix + prod, t0, t1, area(fire)):
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


def viirs_group_name(key, e):
    """A granule's group in viirs.nc: <SAT>_<YYYY-MM-DD>T<HHMM>_<day|night>."""
    import pandas as pd
    sat, a, hhmm = key
    day = (pd.Timestamp(f"{a[1:5]}-01-01") + pd.Timedelta(days=int(a[5:8]) - 1)).strftime("%Y-%m-%d")
    return f"{sat}_{day}T{hhmm}_{(e.get('daynight') or 'unknown').lower()}"


def _viirs_mask(c, bounds):
    """Blank every pixel of a crop whose latitude / longitude lies outside bounds, to the source's own
    _FillValue, in each 2-D variable that has one. The swath window is a rectangle in scan coordinates, so it
    reaches past the region at the swath edges and on tilted passes; blanked pixels compress to almost nothing.
    Variables without a fill value (quality flags, uncertainty indices) are kept whole. -> {grid: pixels blanked}."""
    import numpy as np
    w, s, e, n = bounds
    out = {}
    for grid, la, lo in (("i", "latitude", "longitude"), ("m", "m_latitude", "m_longitude")):
        vs = c.get(grid) or {}
        if la not in vs or lo not in vs:
            continue
        lat, lon = (np.asarray(vs[k][0], dtype="f8") for k in (la, lo))
        outside = ~((lat >= s) & (lat <= n) & (lon >= w) & (lon <= e))
        out[grid] = int(outside.sum())
        if not out[grid]:
            continue
        for name, (arr, attrs) in list(vs.items()):
            fill = fill_value(attrs, arr.dtype)
            if arr.ndim == 2 and arr.shape == outside.shape and fill is not None:
                vs[name] = (np.where(outside, fill, arr).astype(arr.dtype), attrs)
    return out


def _viirs_lut(luts, name, arr, attrs):
    """A look-up table stored once per file in /luts, named by its content: -> its path. The tables only change
    with the calibration, so a fire's hundreds of granules share two or three of them."""
    import hashlib
    import numpy as np
    key = f"{name}_{hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]}"
    if key not in luts.variables:
        luts.createDimension(f"{key}_n", arr.shape[0])
        v = luts.createVariable(key, arr.dtype, (f"{key}_n",), zlib=True, complevel=4,
                                fill_value=fill_value(attrs, arr.dtype))
        v.set_auto_maskandscale(False)
        v.setncatts({**clean_attrs(attrs), "source_name": name})
        v[:] = arr
    return f"/luts/{key}"


def _viirs_write(path, fid, key, e, c, bounds=None):
    """Add one granule's crop to the fire's packed file as its own group (viirs_group_name): I grid (iy, ix) with
    I01-I05 (+ flags), latitude, longitude, zeniths; M grid (my, mx) with M11 (+ flags) and m_latitude etc.
    Values as stored in the source; with bounds, pixels outside them are blanked (_viirs_mask). Look-up tables
    go to /luts once (_viirs_lut): the group's attribute named after the table gives its path. -> group name."""
    import netCDF4
    name = viirs_group_name(key, e)
    blanked = _viirs_mask(c, bounds) if bounds else {}
    with netCDF4.Dataset(path, "a" if Path(path).exists() else "w") as ds:
        if name in ds.groups:
            return name
        luts = ds.groups.get("luts") or ds.createGroup("luts")
        g = ds.createGroup(name)
        r0, r1, c0, c1 = c["i_win"]
        at = {"title": f"VIIRS {key[0]} L1B granule {key[1]}.{key[2]} cropped to {fid}", "fire_id": fid,
              "satellite": key[0], "day_night": e.get("daynight") or "", "granule_start": e.get("start") or "",
              "i_window_rows_cols": f"{r0}:{r1},{c0}:{c1}",
              "sources": " | ".join(e[p] for p in VIIRS_PRODUCTS if p in e)}
        if c.get("m_win"):
            mr0, mr1, mc0, mc1 = c["m_win"]
            at["m_window_rows_cols"] = f"{mr0}:{mr1},{mc0}:{mc1}"
        if bounds:
            at["blanked_outside_lonlat"] = ",".join(f"{b:.5f}" for b in bounds)
            at.update({f"blanked_pixels_{k}": v for k, v in blanked.items()})
        g.setncatts({**{f"source_{k}": v for k, v in c["glob"].items()}, **at})
        for grid, dims in (("i", ("iy", "ix")), ("m", ("my", "mx"))):
            if not c.get(grid):
                continue
            made = False
            for vname, (arr, attrs) in c[grid].items():
                if arr.ndim == 1:
                    g.setncattr(vname, _viirs_lut(luts, vname, arr, attrs))
                    continue
                if not made:
                    g.createDimension(dims[0], arr.shape[0])
                    g.createDimension(dims[1], arr.shape[1])
                    made = True
                v = g.createVariable(vname, arr.dtype, dims, zlib=True, complevel=4, fill_value=fill_value(attrs, arr.dtype))
                v.set_auto_maskandscale(False)
                v.setncatts(clean_attrs(attrs))
                v[:] = arr
    return name


def _viirs_finish(part, fid, region):
    """Index the packed file (one row per granule group, in time order) and give it its final name, so a crash
    never leaves a half-written viirs.nc behind."""
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(part, "a") as ds:
        gs = ds.groups
        names = sorted((k for k in gs if k != "luts"), key=lambda k: (str(getattr(gs[k], "granule_start", "")), k))
        ds.createDimension("granule", len(names))
        for var, attr in (("granule", None), ("granule_start", "granule_start"), ("satellite", "satellite"),
                          ("day_night", "day_night")):
            v = ds.createVariable(var, str, ("granule",))
            v[:] = np.array([k if attr is None else str(getattr(gs[k], attr, "")) for k in names], dtype=object)
        ds.setncatts({"title": f"VIIRS L1B swath granules cropped to {fid}", "fire_id": fid, "n_granules": len(names),
                      "region_lonlat": ",".join(f"{b:.5f}" for b in region), "blanked_pad_km": VIIRS_MASK_PAD_KM,
                      "layout": "one group per granule, named as in the granule variable (I grid iy/ix, M grid my/mx); "
                                "/luts holds each look-up table once, and a group's attribute named after a table "
                                "(e.g. I04_brightness_temperature_lut) gives that table's path",
                      "blanking": "pixels farther than blanked_pad_km outside region_lonlat hold the source's _FillValue "
                                  "in every variable that has one"})
    final = Path(part).with_name(VIIRS_FILE)
    Path(part).rename(final)
    return final


def grab_viirs_many(fires, ctx, chunk=25):
    """VIIRS L1B swaths (TS-SatFire's VIIRS_Day / VIIRS_Night source): every S-NPP, NOAA-20 and NOAA-21
    granule over the box during span(), day and night, cropped on the swath itself (no regridding): I1-I5
    at 375 m and M11 at 750 m as stored (scaled integers, quality flags, the brightness-temperature look-up
    tables of I4 / I5), with each pixel's latitude, longitude and sensor / solar zenith. Remote files are
    read by byte range (a fire's part of a 250 MB granule is a few MB); each granule is read once for every
    fire on it. One file per fire, viirs.nc: a group per granule (<SAT>_<YYYY-MM-DD>T<HHMM>_<day|night>), the
    look-up tables once, pixels beyond the region + VIIRS_MASK_PAD_KM blanked. Granules that fail are named in
    the manifest note; a fire is an error only when all of them fail."""
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
        need, left, written, srcs, failed, region = {}, {}, {}, {}, {}, {}
        with ThreadPoolExecutor(8) as ex:
            for f, grans, err in ex.map(look, group):
                fid = f["fire_id"]
                if err or not grans:
                    yield f, ("error" if err else "empty"), [], err
                    continue
                left[fid], written[fid], srcs[fid], failed[fid] = len(grans), [], [], []
                region[fid] = area(f)
                for k, e in grans.items():
                    need.setdefault(k, (e, []))[1].append(f)
        jobs = [(k, e, [(f["fire_id"], area(f)) for f in fs])
                for k, (e, fs) in sorted(need.items(), key=lambda kv: (str(kv[1][0].get("start")), kv[0]))]
        if jobs:
            print(f"  viirs: {len(jobs)} granules for {len(left)} fires", flush=True)
        stopped, got = False, 0
        with pool(workers) as ex:
            for key, crops, nbytes, err in _windowed_map(ex, _viirs_read, jobs, 2 * workers):
                got += nbytes
                e, fs = need[key]
                for fid, c in crops.items():
                    part = ctx.stage(fid) / (VIIRS_FILE + ".part")
                    written[fid].append(_viirs_write(part, fid, key, e, c, buffer_box(region[fid], VIIRS_MASK_PAD_KM)))
                    srcs[fid].append(e["02IMG"])
                for f in fs:
                    fid = f["fire_id"]
                    if err:
                        failed[fid].append(err)
                    left[fid] -= 1
                    if left[fid] == 0:
                        note = (f"{len(failed[fid])} granule(s) failed: {failed[fid][0]}" if failed[fid] else None)
                        status = "ok" if written[fid] else ("error" if failed[fid] else "empty")
                        if written[fid]:
                            _viirs_finish(ctx.stage(fid) / (VIIRS_FILE + ".part"), fid, region[fid])
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


GROUP_FUNCS = {"goes": grab_goes_many, "hrrr": grab_hrrr_many, "hrrr_next": grab_hrrr_next_many, "gfs": grab_gfs_many,
               "osm": grab_osm_many, "viirs": grab_viirs_many}
