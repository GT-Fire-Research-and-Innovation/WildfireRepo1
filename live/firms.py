"""
VIIRS active-fire detections from NASA FIRMS (S-NPP, NOAA-20, NOAA-21; 375 m).

The MAP_KEY is read from the FIRMS_MAP_KEY environment variable or from a .env file
(never committed). Limit: 5000 transactions per 10 minutes.

  python live/firms.py latest --bbox -125 24 -66 50 --days 1        # near-real-time, last N days (1-10)
  python live/firms.py history --bbox -121.2 39.8 -120.3 40.7 --start 2019-09-01 --end 2019-09-20
  python live/firms.py fires --live data/live --days 2                # detections near every archived WFIGS fire

Output CSVs (one row per detection) go to OUT/firms/. Columns as FIRMS returns them
(latitude, longitude, bright_ti4, bright_ti5, frp, acq_date, acq_time, confidence, daynight, ...)
plus 'source'.
"""
import argparse
import io
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/{days}"
NRT = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
SP = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"]          # standard (archived) product for past years


def map_key():
    key = os.environ.get("FIRMS_MAP_KEY")
    for p in (Path(".env"), Path.home() / ".config" / "wildfire" / "env"):
        if not key and p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("FIRMS_MAP_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("Set FIRMS_MAP_KEY (environment or .env).")
    return key


def get(src, bbox, days, day=None, key=None):
    url = API.format(key=key or map_key(), src=src, bbox=",".join(f"{v:.4f}" for v in bbox), days=days)
    if day:
        url += f"/{day}"
    for t in range(4):
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and not r.text.startswith("Invalid"):
            df = pd.read_csv(io.StringIO(r.text)) if r.text.strip() else pd.DataFrame()
            if len(df):
                df["source"] = src
            return df
        time.sleep(10 * (t + 1))
    raise RuntimeError(f"FIRMS {src}: HTTP {r.status_code} {r.text[:200]}")


def latest(bbox, days, key=None):
    return pd.concat([get(s, bbox, days, key=key) for s in NRT], ignore_index=True)


def history(bbox, start, end, key=None):
    """Archived detections between two dates (inclusive), in 5-day requests (the API maximum with a date)."""
    out, d = [], date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    while d <= end_d:
        n = min(5, (end_d - d).days + 1)
        for s in SP:
            out.append(get(s, bbox, n, d.isoformat(), key=key))
        d += timedelta(days=n)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def near_live_fires(live, days, pad=0.1, key=None):
    """Detections within each archived WFIGS fire's bounding box (+pad degrees)."""
    import geopandas as gpd
    idx = pd.read_csv(Path(live) / "perimeters" / "index.csv")
    last = idx.sort_values("polygon_time").groupby("fire_id").tail(1)
    frames = []
    for _, r in last.iterrows():
        g = gpd.read_file(Path(live) / "perimeters" / r["file"])
        w, s, e, n = g.total_bounds
        df = latest((w - pad, s - pad, e + pad, n + pad), days, key=key)
        if len(df):
            df["fire_id"] = r["fire_id"]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["latest", "history", "fires"])
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), default=[-125, 24, -66, 50])
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--live", default="data/live")
    ap.add_argument("--out", default="data/live")
    a = ap.parse_args()
    if a.cmd == "latest":
        df = latest(a.bbox, a.days)
    elif a.cmd == "history":
        df = history(a.bbox, a.start, a.end)
    else:
        df = near_live_fires(a.live, a.days)
    od = Path(a.out) / "firms"
    od.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    fn = od / f"{a.cmd}_{stamp}.csv"
    df.to_csv(fn, index=False)
    print(f"{len(df)} detections -> {fn}")


if __name__ == "__main__":
    sys.exit(main())


SATELLITES = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT")


def recent(now, bbox=(-125, 24, -66, 50), days=2):
    """VIIRS detections from every satellite that has data (NOAA-20, NOAA-21, S-NPP) for the last `days`
    days: more passes means small fires are more likely to be caught. A satellite whose feed fails or is
    empty is skipped."""
    import pandas as pd
    parts = []
    for src in SATELLITES:
        for k in range(days):
            try:
                d = get(src, bbox, 1, (now - pd.Timedelta(days=k)).date().isoformat())
            except Exception as ex:                     # one feed down must not stop the others
                print(f"  FIRMS {src}: {type(ex).__name__}", flush=True)
                continue
            if len(d):
                parts.append(d.assign(source=src))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["latitude", "longitude"])
