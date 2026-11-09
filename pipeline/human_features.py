"""
Roads and other human features for every fire, on the same 75 m Tessera grid as the
NIFC perimeter labels and suppression layers. Run nifc_perimeters.py first: this script
reads the fire list and grids from its output folder (fires.json).

Source: OpenStreetMap as it was on 1 January of the fire's year (before the fire), from
Geofabrik's yearly snapshots (download.geofabrik.de, e.g. us-west-200101.osm.pbf), cut to
each fire's box with osmium. Using the pre-fire map avoids leaking post-fire changes
(buildings that burned and were removed, new fire roads). OSM data (c) OpenStreetMap
contributors, ODbL.

Outputs in OUT_DIR/human/<fire_id>.npz:
  layers     (C, H, W) uint8    0/1 rasters, C = len(channels)
  channels   (C,)               channel names (CHANNELS below)
  snapshot   ()                 OSM snapshot date used, 'YYYY-01-01'
  source     ()                 Geofabrik file the features came from
  transform, fire_id            grid georeferencing / id, as in the label files
Plus OUT_DIR/human_summary.csv (features and cells per channel per fire) and
OUT_DIR/human_quicklook/<fire_id>.png.

The layers are static: one map per fire, used for every day of that fire.
Distance-to-road / distance-to-building are cheap to compute in the loader with
scipy.ndimage.distance_transform_edt(1 - layer, sampling=(dy_m, dx_m)), so they are not stored.

Run (after nifc_perimeters.py; needs osmium-tool and aria2):
  apt-get install osmium-tool aria2
  python human_features.py --out data
Large downloads (~2 GB per region-year) go to --work and are deleted after use.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GEOFABRIK = "https://download.geofabrik.de/north-america"

# channel -> (OSM key, set of values or None for any value)
CHANNELS = {
    "road_major": ("highway", {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                               "secondary", "secondary_link"}),
    "road_minor": ("highway", {"tertiary", "tertiary_link", "unclassified", "residential", "living_street",
                               "service"}),
    "track_trail": ("highway", {"track", "path", "bridleway"}),  # dirt roads and trails; common fire access
    "railway": ("railway", {"rail", "light_rail", "narrow_gauge"}),
    "power_line": ("power", {"line", "minor_line"}),  # cleared corridors; also an ignition source
    "building": ("building", None),
    "developed": ("landuse", {"residential", "commercial", "industrial", "retail"}),
}
CHANNEL_NAMES = list(CHANNELS)
LINE_CHANNELS = ["road_major", "road_minor", "track_trail", "railway", "power_line"]


# Fires outside the us-west region file: (state file, lon_min, lat_min, lon_max, lat_max).
EAST_STATES = [("us/kansas", -102.06, 36.99, -94.58, 40.01), ("us/oklahoma", -103.0, 33.6, -94.4, 37.0),
               ("us/texas", -106.65, 25.8, -93.5, 36.5), ("us/florida", -87.7, 24.4, -79.9, 31.0),
               ("us/nebraska", -104.06, 40.0, -95.3, 43.0), ("us/missouri", -95.8, 35.99, -89.1, 40.6)]
TAG_FILTER = ["w/highway=" + ",".join(sorted(CHANNELS["road_major"][1] | CHANNELS["road_minor"][1]
                                             | CHANNELS["track_trail"][1])),
              "w/railway=" + ",".join(sorted(CHANNELS["railway"][1])),
              "w/power=" + ",".join(sorted(CHANNELS["power_line"][1])),
              "w/building",
              "w/landuse=" + ",".join(sorted(CHANNELS["developed"][1]))]
RASTER_BATCH = 50000  # geometries per rasterize call (bounds memory over dense cities)


def region_file(fire):
    """Geofabrik file (without date) covering the fire's box centre."""
    w, s, e, n = fire["bounds_lonlat"]
    lon, lat = (w + e) / 2, (s + n) / 2
    west_tx = lat < 32.0 and lon > -106.65  # west Texas is in us-south, not us-west
    if lon < -102.05 and not west_tx:
        return "us-west"
    for name, x0, y0, x1, y1 in EAST_STATES:
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            return name
    raise RuntimeError(f"no Geofabrik region for fire at {lon:.2f},{lat:.2f}")


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[:80]}... failed: {r.stderr[-400:]}")
    return r.stdout


def prepare_region(region, year, work):
    """Download the 1 Jan snapshot and keep only the tags we use. Returns the filtered file."""
    tag = f"{str(year)[2:]}0101"
    stem = f"{region.replace('/', '_')}-{tag}"
    filt = work / f"{stem}.filtered.osm.pbf"
    if filt.exists():
        return filt
    raw = work / f"{stem}.osm.pbf"
    url = f"{GEOFABRIK}/{region}-{tag}.osm.pbf"
    t0 = time.time()
    print(f"  downloading {url}", flush=True)
    # Geofabrik throttles single connections for old snapshots; aria2c with 8 connections is
    # ~3x faster. -c continues a partial file from an interrupted run.
    run(f"aria2c -q -c --file-allocation=none -x 8 -s 8 -k 5M --max-tries=20 --retry-wait=10 "
        f"-d {raw.parent} -o {raw.name} {url}")
    print(f"  filtering ({raw.stat().st_size / 1e9:.1f} GB, {time.time() - t0:.0f}s download)", flush=True)
    run(f"osmium tags-filter --overwrite {raw} {' '.join(TAG_FILTER)} -o {filt}")
    raw.unlink()
    print(f"  ready: {filt.name} ({filt.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s total)", flush=True)
    return filt


def fire_features(filt, fire, work):
    """Yields GeoJSON-like features inside the fire's box, streamed from the filtered region file
    (a box over a city can hold over a million buildings, so nothing is kept in memory)."""
    w, s, e, n = fire["bounds_lonlat"]
    cut = work / f"cut_{fire['fire_id']}.osm.pbf"
    seq = work / f"cut_{fire['fire_id']}.geojsonseq"
    run(f"osmium extract --overwrite -s complete_ways -b {w},{s},{e},{n} {filt} -o {cut}")
    run(f"osmium export --overwrite {cut} -f geojsonseq --geometry-types=linestring,polygon -o {seq}")
    cut.unlink()
    try:
        with open(seq) as fh:
            for line in fh:
                line = line.strip().lstrip("\x1e")
                if line:
                    yield json.loads(line)
    finally:
        seq.unlink(missing_ok=True)


def classify(props):
    """Channel names a feature belongs to (usually one)."""
    out = []
    for c in CHANNEL_NAMES:
        k, vals = CHANNELS[c]
        v = props.get(k)
        if v is not None and (vals is None or v in vals):
            out.append(c)
    return out


def rasterize_fire(feats, fire):
    from affine import Affine
    from rasterio.features import rasterize
    from shapely.geometry import shape
    H, W = fire["height"], fire["width"]
    tr = Affine(*fire["transform"])
    layers = np.zeros((len(CHANNEL_NAMES), H, W), dtype=np.uint8)
    counts = {c: 0 for c in CHANNEL_NAMES}
    pending = {c: [] for c in CHANNEL_NAMES}
    ix = {c: i for i, c in enumerate(CHANNEL_NAMES)}

    def flush(c):
        if pending[c]:
            # lines: every touched cell; areas: cells whose centre is inside, plus touched for
            # buildings (a house is smaller than a 75 m cell and would otherwise vanish)
            touched = c in LINE_CHANNELS or c == "building"
            burn = rasterize([(g, 1) for g in pending[c]], out_shape=(H, W), transform=tr, fill=0,
                             all_touched=touched, dtype="uint8")
            np.maximum(layers[ix[c]], burn, out=layers[ix[c]])
            pending[c] = []

    for f in feats:
        g = f.get("geometry")
        if not g:
            continue
        chans = classify(f.get("properties") or {})
        if not chans:
            continue
        try:
            geom = shape(g)
        except Exception:  # noqa: BLE001
            continue
        if geom.is_empty:
            continue
        for c in chans:
            pending[c].append(geom)
            counts[c] += 1
            if len(pending[c]) >= RASTER_BATCH:
                flush(c)
    for c in CHANNEL_NAMES:
        flush(c)
    return layers, counts


def quicklook(layers, fire, path, date):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = np.ones(layers.shape[1:] + (3,), dtype=np.float32)
    ix = {c: i for i, c in enumerate(CHANNEL_NAMES)}
    paint = [("developed", (0.93, 0.85, 0.75)), ("track_trail", (0.65, 0.5, 0.3)), ("road_minor", (0.5, 0.5, 0.5)),
             ("power_line", (0.2, 0.6, 0.9)), ("railway", (0.4, 0.2, 0.6)), ("road_major", (0.85, 0.1, 0.1)),
             ("building", (0.0, 0.0, 0.0))]
    for c, col in paint:
        rgb[layers[ix[c]] > 0] = col
    fig, ax = plt.subplots(figsize=(6, 6))
    l, b, r, t = fire["bounds_lonlat"]
    ax.imshow(rgb, extent=[l, r, b, t], interpolation="nearest")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col, label=c) for c, col in paint[::-1]], loc="lower left", fontsize=7)
    ax.set_title(f"Fire {fire['fire_id']}: OSM human features (map of {date})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--fires", nargs="*")
    ap.add_argument("--redo", action="store_true", help="rebuild fires that are already done")
    ap.add_argument("--work", default="osm_work", help="local scratch folder for downloads")
    args = ap.parse_args()
    out = Path(args.out)
    fires = json.loads((out / "fires.json").read_text())
    ids = sorted(fires)
    if args.fires:
        ids = [i for i in ids if i in set(args.fires)]
    for sub in ["human", "human_quicklook"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    # Process fires grouped by region file and year so each ~2 GB file is downloaded once.
    ids.sort(key=lambda i: (region_file(fires[i]), fires[i]["fire_year"], i))
    print(f"{len(ids)} fires; OSM snapshot = 1 January of each fire's year (Geofabrik)")
    current = None
    for k, fid in enumerate(ids):
        fire = fires[fid]
        group = (region_file(fire), fire["fire_year"])
        if current and group != current:  # done with the previous region-year: free the disk
            for f in work.glob(f"{current[0].replace('/', '_')}-{str(current[1])[2:]}0101*"):
                f.unlink()
        current = group
        target = out / "human" / f"{fid}.npz"
        if target.exists() and not args.redo:
            print(f"[{k + 1}/{len(ids)}] {fid}: already done")
            continue
        date = f"{fire['fire_year']}-01-01"
        t0 = time.time()
        try:
            filt = prepare_region(group[0], fire["fire_year"], work)
            feats = fire_features(filt, fire, work)
            source = filt.name.replace(".filtered.osm.pbf", ".osm.pbf")
            layers, counts = rasterize_fire(feats, fire)
        except Exception as ex:  # noqa: BLE001
            print(f"[{k + 1}/{len(ids)}] {fid}: ERROR {ex}")
            rows.append({"fire_id": fid, "status": f"error: {ex}"})
            continue
        np.savez_compressed(target, layers=layers, channels=np.array(CHANNEL_NAMES), snapshot=np.array(date),
                            source=np.array(source), transform=np.array(fire["transform"]), fire_id=np.array(fid))
        quicklook(layers, fire, out / "human_quicklook" / f"{fid}.png", date)
        cells = layers.reshape(len(CHANNEL_NAMES), -1).sum(1)
        row = {"fire_id": fid, "fire_year": fire["fire_year"], "snapshot": date, "source": source, "status": "ok"}
        for i, c in enumerate(CHANNEL_NAMES):
            row[f"{c}_features"] = counts[c]
            row[f"{c}_cells"] = int(cells[i])
        rows.append(row)
        print(f"[{k + 1}/{len(ids)}] {fid} ({source}, {time.time() - t0:.0f}s): "
              + ", ".join(f"{c}:{counts[c]}" for c in CHANNEL_NAMES), flush=True)
        new = pd.DataFrame(rows)
        sp = out / "human_summary.csv"
        if sp.exists():
            old = pd.read_csv(sp, dtype={"fire_id": str})
            new = pd.concat([old[~old["fire_id"].isin(new["fire_id"])], new], ignore_index=True)
        new.to_csv(sp, index=False)
    print("Done:", out / "human_summary.csv")


if __name__ == "__main__":
    sys.exit(main())
