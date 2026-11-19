"""
GRIDMET and HRRR access for the raw grab (raw_channels.py: gridmet, hrrr, hrrr_next).

  GRIDMET, GM_VARS   THREDDS aggregation URL and the names of 9 variables inside it (raw_channels adds the rest)
  hrrr_fetch         chosen GRIB2 messages of one HRRR file, fetched by the byte ranges in its .idx
  hrrr_latlon        latitude / longitude of HRRR's 3 km Lambert grid
  hrrr_window        rows / cols of that grid covering a box (plus PAD_DEG)
"""
import time
import warnings

import numpy as np

GRIDMET = "https://thredds.northwestknowledge.net/thredds/dodsC/agg_met_{v}_1979_CurrentYear_CONUS.nc"
GM_VARS = {  # short name -> variable inside the aggregation
    "vpd": "daily_mean_vapor_pressure_deficit",
    "fm100": "dead_fuel_moisture_100hr",
    "fm1000": "dead_fuel_moisture_1000hr",
    "bi": "daily_mean_burning_index_g",
    "erc": "daily_mean_energy_release_component-g",
    "rmin": "daily_minimum_relative_humidity",
    "vs": "daily_mean_wind_speed",
    "th": "daily_mean_wind_direction",
    "tmmx": "daily_maximum_temperature",
}
HRRR = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d}/conus/hrrr.t{h:02d}z.wrfsfcf{f:02d}.grib2"
HRRR_FIELDS = [("gust", ":GUST:surface:"), ("u10", ":UGRD:10 m above ground:"),
               ("v10", ":VGRD:10 m above ground:"), ("rh2", ":RH:2 m above ground:")]
warnings.filterwarnings("ignore")
PAD_DEG = 0.05

# ---------------------------------------------------------------------------- HRRR

def _ranges(sess, url, fields=None):
    """Byte ranges of the wanted fields in a HRRR file, from its .idx (first message matching each key)."""
    r = sess.get(url + ".idx", timeout=60)
    if r.status_code != 200:
        return None
    lines = r.text.strip().splitlines()
    starts = [int(l.split(":")[1]) for l in lines]
    rng = {}
    for i, l in enumerate(lines):
        for name, key in fields or HRRR_FIELDS:
            if key in l and name not in rng:
                end = starts[i + 1] - 1 if i + 1 < len(lines) else None
                rng[name] = (starts[i], end)
    return rng


def hrrr_fetch(sess, url, fields=None):
    """{field: 2-D array} for one HRRR file (CONUS grid), or None if unavailable. fields: [(name, .idx key)],
    default HRRR_FIELDS."""
    import pygrib
    for tries in range(4):
        try:
            rng = _ranges(sess, url, fields)
            if not rng:
                return None
            out = {}
            for name, (a, b) in rng.items():
                hdr = {"Range": f"bytes={a}-{b if b is not None else ''}"}
                r = sess.get(url, headers=hdr, timeout=120)
                r.raise_for_status()
                g = pygrib.fromstring(r.content)
                out[name] = g.values.astype(np.float32)
            return out
        except Exception as ex:  # noqa: BLE001
            if tries == 3:
                print(f"    HRRR failed {url}: {ex}")
                return None
            time.sleep(5 * (tries + 1))


def hrrr_latlon(sess):
    import pygrib
    url = HRRR.format(d="20200820", h=12, f=0)
    a, b = _ranges(sess, url)["u10"]
    r = sess.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
    lat, lon = pygrib.fromstring(r.content).latlons()
    return lat, lon


def hrrr_window(lat, lon, bounds):
    w, s, e, n = bounds
    m = (lat >= s - PAD_DEG) & (lat <= n + PAD_DEG) & (lon >= w - PAD_DEG) & (lon <= e + PAD_DEG)
    rows, cols = np.where(m)
    if not len(rows):
        return None
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
