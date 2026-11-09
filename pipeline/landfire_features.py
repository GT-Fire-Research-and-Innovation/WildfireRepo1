"""
LANDFIRE fuels and canopy for every fire box, from the LANDFIRE Product Service (LFPS).

Uses the LF 2016 Remap layers, which describe conditions before all 2018-2021 fires:
  FBFM40  Scott & Burgan 40 fire behavior fuel model (categorical)
  CC      canopy cover (%)
  CH      canopy height (m x 10)
  CBH     canopy base height (m x 10)
  CBD     canopy bulk density (kg/m3 x 100)
Output: OUT_DIR/landfire/<fire_id>.tif (5 bands, EPSG:4326, 90 m), one per fire.
"""
import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

LFPS = "https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/GPServer/LandfireProductService"
LAYERS = ["LF2016_FBFM40", "LF2016_CC", "LF2016_CH", "LF2016_CBH", "LF2016_CBD"]


def fetch(bounds, dst, tries=3):
    w, s, e, n = bounds
    params = {"Layer_List": ";".join(LAYERS), "Area_of_Interest": f"{w} {s} {e} {n}",
              "Output_Projection": "4326", "Resample_Resolution": "90", "f": "json"}
    for t in range(tries):
        try:
            job = requests.get(f"{LFPS}/submitJob", params=params, timeout=120).json()
            jid = job["jobId"]
            for _ in range(240):
                time.sleep(5)
                st = requests.get(f"{LFPS}/jobs/{jid}", params={"f": "json"}, timeout=60).json()
                if st["jobStatus"] in ("esriJobSucceeded", "esriJobFailed", "esriJobCancelled"):
                    break
            if st["jobStatus"] != "esriJobSucceeded":
                raise RuntimeError(f"job {jid}: {st['jobStatus']} {st.get('messages', [])[-2:]}")
            res = requests.get(f"{LFPS}/jobs/{jid}/results/Output_File", params={"f": "json"}, timeout=60).json()
            url = res["value"]["url"]
            z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=600).content))
            tif = [n for n in z.namelist() if n.lower().endswith(".tif")][0]
            dst.write_bytes(z.read(tif))
            return
        except Exception as ex:  # noqa: BLE001
            print(f"    retry {t + 1}: {ex}")
            time.sleep(20)
    raise RuntimeError("LFPS failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--fires", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "landfire").mkdir(exist_ok=True)
    fires = json.loads((out / "fires.json").read_text())
    ids = [p.stem for p in sorted((out / "train").glob("*.npz"))]
    if a.fires:
        ids = [i for i in ids if i in set(a.fires)]
    for k, fid in enumerate(ids):
        dst = out / "landfire" / f"{fid}.tif"
        if dst.exists():
            continue
        t0 = time.time()
        try:
            fetch(fires[fid]["bounds_lonlat"], dst)
            print(f"[{k + 1}/{len(ids)}] {fid}: {dst.stat().st_size / 1e6:.1f} MB ({time.time() - t0:.0f}s)", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[{k + 1}/{len(ids)}] {fid}: ERROR {ex}", flush=True)
    print("Done")


if __name__ == "__main__":
    sys.exit(main())
