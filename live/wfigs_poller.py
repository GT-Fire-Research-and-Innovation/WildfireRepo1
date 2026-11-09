"""
Polls the WFIGS Current Interagency Fire Perimeters service and archives every new polygon
version, so we build our own perimeter time series from day one (the live service keeps only
the latest polygon per fire).

  python live/wfigs_poller.py --out data/live            # poll every 15 minutes, forever
  python live/wfigs_poller.py --out data/live --once     # one poll (e.g. from cron)

Archive layout (OUT/perimeters):
  <UniqueFireIdentifier>/<PolygonDateTime UTC>_<hash8>.geojson   one file per new version
  index.csv   one row per archived version (fire id, IRWIN id, name, times, acres, map method,
              containment, file, first_seen)
A version is "new" when its geometry or polygon timestamp differs from every version already
archived for that fire.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
       "WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query")
KEEP = ["attr_UniqueFireIdentifier", "attr_IrwinID", "attr_IncidentName", "poly_PolygonDateTime",
        "poly_DateCurrent", "poly_GISAcres", "poly_MapMethod", "poly_FeatureCategory",
        "attr_PercentContained", "attr_FireDiscoveryDateTime", "attr_IncidentSize"]


def ms_to_iso(ms):
    return None if ms is None else datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch():
    feats, offset = [], 0
    while True:
        r = requests.get(URL, params={"where": "1=1", "outFields": ",".join(KEEP), "outSR": 4326,
                                      "f": "geojson", "resultOffset": offset, "resultRecordCount": 1000},
                         timeout=120)
        r.raise_for_status()
        page = r.json().get("features", [])
        feats += page
        if len(page) < 1000:
            return feats
        offset += len(page)


def poll(out):
    arch = out / "perimeters"
    arch.mkdir(parents=True, exist_ok=True)
    idx_path = arch / "index.csv"
    idx = pd.read_csv(idx_path) if idx_path.exists() else pd.DataFrame(columns=["fire_id", "hash"])
    seen = set(zip(idx["fire_id"].astype(str), idx["hash"].astype(str)))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, n_new = [], 0
    for f in fetch():
        p = f["properties"]
        fid = p.get("attr_UniqueFireIdentifier") or (p.get("attr_IrwinID") or "unknown").strip("{}")
        geom = json.dumps(f.get("geometry"), sort_keys=True)
        h = hashlib.sha1((geom + str(p.get("poly_PolygonDateTime"))).encode()).hexdigest()[:8]
        if (fid, h) in seen:
            continue
        t = ms_to_iso(p.get("poly_PolygonDateTime")) or now
        d = arch / fid
        d.mkdir(exist_ok=True)
        fn = d / f"{t.replace(':', '')}_{h}.geojson"
        fn.write_text(json.dumps({"type": "FeatureCollection", "features": [f]}))
        rows.append({"fire_id": fid, "irwin_id": p.get("attr_IrwinID"), "name": p.get("attr_IncidentName"),
                     "polygon_time": t, "date_current": ms_to_iso(p.get("poly_DateCurrent")),
                     "discovered": ms_to_iso(p.get("attr_FireDiscoveryDateTime")), "acres": p.get("poly_GISAcres"),
                     "map_method": p.get("poly_MapMethod"), "category": p.get("poly_FeatureCategory"),
                     "pct_contained": p.get("attr_PercentContained"), "file": str(fn.relative_to(arch)),
                     "hash": h, "first_seen": now})
        n_new += 1
    if rows:
        idx = pd.concat([idx, pd.DataFrame(rows)], ignore_index=True)
        idx.to_csv(idx_path, index=False)
    return n_new, idx["fire_id"].nunique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/live")
    ap.add_argument("--interval", type=int, default=900, help="seconds between polls")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    while True:
        t = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        try:
            n_new, n_fires = poll(out)
            print(f"{t}: {n_new} new perimeter versions; {n_fires} fires archived", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"{t}: poll failed: {ex}", flush=True)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
