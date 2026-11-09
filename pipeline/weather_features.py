"""
Daily weather for every fire, beyond what TS-SatFire carries.

For each fire (from OUT_DIR/fires.json, written by nifc_perimeters.py) and each day from
6 days before its first TS-SatFire day to 2 days after its last:

  GRIDMET (daily, ~4 km; via the THREDDS OPeNDAP aggregations, only the fire's box is read)
    vpd      mean vapor pressure deficit (kPa)
    fm100    100-hour dead fuel moisture (%)
    fm1000   1000-hour dead fuel moisture (%)
    bi       burning index
    erc      energy release component
    rmin     minimum relative humidity (%)
    vs       mean wind speed at 10 m (m/s)
    th       wind direction (degrees, from)
    tmmx     maximum temperature (K)

  HRRR (hourly model, 3 km; only 4 fields are fetched per file using the .idx byte ranges)
    ana  = 21 UTC analysis of that day (early afternoon in the West): what the weather WAS
    fc   = 12 UTC run, 9-hour forecast, valid 21 UTC that day: what was FORECAST that morning
    fields: gust (m/s), u10, v10 (m/s), rh2 (%)

  To predict growth from day t to t+1 without peeking, use GRIDMET/HRRR-ana up to day t and
  HRRR-fc for day t+1 (issued 12 UTC on t+1, i.e. ~5-6 am in the West, around when the
  day-t perimeter is mapped).

Output: OUT_DIR/weather/<fire_id>.npz
  dates (D,), gm (D, 9, hg, wg) float32, gm_vars, gm_lat (hg,), gm_lon (wg,)
  hrrr_ana (D, 4, hh, wh), hrrr_fc (D, 4, hh, wh), hrrr_vars, hrrr_lat (hh, wh), hrrr_lon (hh, wh)
  NaN where data were missing.
"""
import argparse
import warnings
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

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
DAYS_BEFORE, DAYS_AFTER = 6, 2


def _session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "tesserafire-weather/1.0"
    return s


def fire_days(fire):
    d0 = pd.Timestamp(fire["first_day"] or f"{fire['fire_year']}-07-01") - pd.Timedelta(days=DAYS_BEFORE)
    d1 = pd.Timestamp(fire["last_day"] or fire["first_day"] or f"{fire['fire_year']}-07-01") + pd.Timedelta(days=DAYS_AFTER)
    return pd.date_range(d0, d1)


# ---------------------------------------------------------------------------- GRIDMET

def gridmet_fire(datasets, fire, days):
    w, s, e, n = fire["bounds_lonlat"]
    out, lat, lon = [], None, None
    for short, ds in datasets.items():
        var = GM_VARS[short]
        sub = ds[var].sel(lat=slice(n + PAD_DEG, s - PAD_DEG), lon=slice(w - PAD_DEG, e + PAD_DEG),
                          day=slice(days[0], days[-1]))
        for tries in range(5):
            try:
                arr = sub.load()
                break
            except Exception as ex:  # noqa: BLE001
                if tries == 4:
                    raise
                print(f"    gridmet {short} retry ({ex})")
                time.sleep(5 * (tries + 1))
        arr = arr.reindex(day=days)  # NaN for missing days
        out.append(arr.values.astype(np.float32))
        lat, lon = arr["lat"].values, arr["lon"].values
    return np.stack(out, axis=1), lat, lon  # (D, V, hg, wg)


# ---------------------------------------------------------------------------- HRRR

def _ranges(sess, url):
    """Byte ranges of the wanted fields in a HRRR file, from its .idx."""
    r = sess.get(url + ".idx", timeout=60)
    if r.status_code != 200:
        return None
    lines = r.text.strip().splitlines()
    starts = [int(l.split(":")[1]) for l in lines]
    rng = {}
    for i, l in enumerate(lines):
        for name, key in HRRR_FIELDS:
            if key in l and name not in rng:
                end = starts[i + 1] - 1 if i + 1 < len(lines) else None
                rng[name] = (starts[i], end)
    return rng


def hrrr_fetch(sess, url):
    """{field: 2-D array} for one HRRR file (CONUS grid), or None if unavailable."""
    import pygrib
    for tries in range(4):
        try:
            rng = _ranges(sess, url)
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


# ---------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nifc_out")
    ap.add_argument("--fires", nargs="*")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "weather").mkdir(parents=True, exist_ok=True)
    fires = json.loads((out / "fires.json").read_text())
    ids = sorted(fires) if not args.fires else [f for f in sorted(fires) if f in set(args.fires)]
    todo = [f for f in ids if args.redo or not (out / "weather" / f"{f}.npz").exists()]
    print(f"{len(todo)} of {len(ids)} fires to do")
    if not todo:
        return

    import xarray as xr
    gm = {k: xr.open_dataset(GRIDMET.format(v=k), engine="netcdf4") for k in GM_VARS}
    sess = _session()
    hlat, hlon = hrrr_latlon(sess)

    # HRRR: fetch each needed date once (analysis + morning forecast), crop for every fire.
    need = {}
    for fid in todo:
        for d in fire_days(fires[fid]):
            need.setdefault(d.strftime("%Y%m%d"), []).append(fid)
    wins = {fid: hrrr_window(hlat, hlon, fires[fid]["bounds_lonlat"]) for fid in todo}
    crops = {fid: {} for fid in todo}  # fid -> {(date, kind): (4, hh, wh)}

    def job(dk):
        d, kind = dk
        url = HRRR.format(d=d, h=21, f=0) if kind == "ana" else HRRR.format(d=d, h=12, f=9)
        return dk, hrrr_fetch(_session(), url)

    jobs = [(d, k) for d in sorted(need) for k in ("ana", "fc")]
    print(f"HRRR: {len(need)} dates, {len(jobs)} files")
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, ((d, kind), fields) in enumerate(ex.map(job, jobs)):
            if fields is not None:
                for fid in need[d]:
                    w = wins[fid]
                    if w is None:
                        continue
                    r0, r1, c0, c1 = w
                    crops[fid][(d, kind)] = np.stack([fields[n][r0:r1, c0:c1] for n, _ in HRRR_FIELDS])
            if (i + 1) % 100 == 0:
                print(f"  HRRR {i + 1}/{len(jobs)} files ({time.time() - t0:.0f}s)")

    for k, fid in enumerate(todo):
        fire = fires[fid]
        days = fire_days(fire)
        t1 = time.time()
        try:
            g, glat, glon = gridmet_fire(gm, fire, days)
        except Exception as ex:  # noqa: BLE001
            print(f"[{k + 1}/{len(todo)}] {fid}: GRIDMET ERROR {ex}")
            continue
        w = wins[fid]
        if w is None:
            ana = fc = np.zeros((len(days), 4, 0, 0), np.float32)
            hl = hn = np.zeros((0, 0), np.float32)
        else:
            r0, r1, c0, c1 = w
            shape = (4, r1 - r0, c1 - c0)
            blank = np.full(shape, np.nan, np.float32)
            ana = np.stack([crops[fid].get((d.strftime("%Y%m%d"), "ana"), blank) for d in days])
            fc = np.stack([crops[fid].get((d.strftime("%Y%m%d"), "fc"), blank) for d in days])
            hl, hn = hlat[r0:r1, c0:c1].astype(np.float32), hlon[r0:r1, c0:c1].astype(np.float32)
        np.savez_compressed(out / "weather" / f"{fid}.npz",
                            dates=np.array([str(d.date()) for d in days]), gm=g, gm_vars=np.array(list(GM_VARS)),
                            gm_lat=glat, gm_lon=glon, hrrr_ana=ana, hrrr_fc=fc,
                            hrrr_vars=np.array([n for n, _ in HRRR_FIELDS]), hrrr_lat=hl, hrrr_lon=hn)
        miss = float(np.isnan(ana).mean()) if ana.size else 1.0
        print(f"[{k + 1}/{len(todo)}] {fid}: {len(days)} days, gridmet {g.shape[2:]}, hrrr {ana.shape[2:]}, "
              f"hrrr missing {miss:.0%} ({time.time() - t1:.0f}s)")
    print("Done:", out / "weather")


if __name__ == "__main__":
    sys.exit(main())
