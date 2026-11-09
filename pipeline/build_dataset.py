"""
Builds one compact training file per fire, everything on a common 150 m grid (the Tessera
75 m grid averaged 2x2), cropped to the area around the fire.

Inputs it reads (run the other scripts first, same --out folder):
  fires.json, summary.csv, split_groups.csv, labels/<fid>.npz      (nifc_perimeters.py)
  suppression/<fid>.npz                                             (suppression_layers.py)
  weather/<fid>.npz                                                 (weather_features.py)
  human/<fid>.npz  (optional)                                       (human_features.py)
And downloads per fire:
  Tessera embedding (Hugging Face SamuelWu318/ts-tesserafire-5)  -> 128 channels, averaged to 150 m
  TS-SatFire VIIRS day/night + FirePred (Hugging Face SamuelWu318/ts-satfire)
  Copernicus GLO-30 DEM (AWS open data)                            -> elevation, slope, aspect

Output OUT_DIR/train/<fid>.npz (see save() for keys). Tessera is kept at 128 channels here;
train.py projects it with a PCA fitted on training fires only.
"""
import argparse
import io
import json
import math
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    import zipfile_zstd  # noqa: F401
except ImportError:
    pass

POOL = 2            # 75 m -> 150 m
PAD = 64            # cells of 150 m (9.6 km) kept around the fire
TESSERA = "datasets/SamuelWu318/ts-tesserafire-5/tessera/p{fid}.npz"
TSSAT = "datasets/SamuelWu318/ts-satfire/ts-satfire/{fid}"
DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
# IGBP (MODIS LC_Type1) -> fuel groups
LC_GROUPS = {"forest": [1, 2, 3, 4, 5], "shrub": [6, 7], "savanna": [8, 9], "grass": [10],
             "crop": [12, 14], "urban": [13], "nonfuel": [11, 15, 16, 17]}


def pool_mean(a, k):
    *lead, H, W = a.shape
    return a[..., : H // k * k, : W // k * k].reshape(*lead, H // k, k, W // k, k).mean(axis=(-3, -1))


def pool_max(a, k):
    *lead, H, W = a.shape
    return a[..., : H // k * k, : W // k * k].reshape(*lead, H // k, k, W // k, k).max(axis=(-3, -1))


def pool_arrival(arr, k):
    H, W = arr.shape
    a = arr[: H // k * k, : W // k * k].reshape(H // k, k, W // k, k).astype(np.int32)
    a = np.where(a < 0, 10 ** 6, a).min(axis=(1, 3))
    return np.where(a >= 10 ** 6, -1, a).astype(np.int16)


def region_of(arrival150, other150):
    m = (arrival150 >= 0) | other150
    if not m.any():
        return None
    rows, cols = np.where(m)
    H, W = arrival150.shape
    r0, r1 = max(rows.min() - PAD, 0), min(rows.max() + PAD + 1, H)
    c0, c1 = max(cols.min() - PAD, 0), min(cols.max() + PAD + 1, W)
    return r0, r1, c0, c1


def grid_centres(transform, r0, r1, c0, c1):
    a, _, x0, _, e, y0 = transform[:6]
    cols = np.arange(c0, c1) + 0.5
    rows = np.arange(r0, r1) + 0.5
    lon = x0 + cols * a
    lat = y0 + rows * e
    return np.meshgrid(lon, lat)  # (h, w) each


def regular_index(lon, lat, x0, y0, dx, dy, nx, ny):
    """Nearest-cell row/col maps into a regular north-up grid (dy > 0 measured downward)."""
    col = np.clip(np.floor((lon - x0) / dx).astype(np.int32), 0, nx - 1)
    row = np.clip(np.floor((y0 - lat) / dy).astype(np.int32), 0, ny - 1)
    return row.astype(np.int16), col.astype(np.int16)


# ---------------------------------------------------------------------------- sources

def tessera_region(fs, fid, reg):
    """(128, h, w) float16 Tessera averaged to 150 m over the region, plus coverage."""
    with fs.open(TESSERA.format(fid=fid), "rb", block_size=2 ** 22) as f:
        z = zipfile.ZipFile(f)
        with z.open("embedding.npy") as m:
            emb = np.lib.format.read_array(io.BytesIO(m.read()))
        with z.open("coverage.npy") as m:
            cov = np.lib.format.read_array(io.BytesIO(m.read()))
    r0, r1, c0, c1 = reg
    emb = emb[:, r0 * POOL: r1 * POOL, c0 * POOL: c1 * POOL].astype(np.float32)
    emb = pool_mean(emb, POOL).astype(np.float16)
    cov = pool_mean(cov[r0 * POOL: r1 * POOL, c0 * POOL: c1 * POOL].astype(np.float32), POOL)
    return emb, cov.astype(np.float16)


def dem_region(transform150, reg):
    """Elevation (m), slope (deg), aspect (deg) on the region grid from Copernicus GLO-30."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject
    r0, r1, c0, c1 = reg
    h, w = r1 - r0, c1 - c0
    a, _, x0, _, e, y0 = transform150[:6]
    dst_tr = Affine(a, 0, x0 + c0 * a, 0, e, y0 + r0 * e)
    west, north = x0 + c0 * a, y0 + r0 * e
    east, south = west + w * a, north + h * e
    dst = np.full((h, w), np.nan, np.float32)
    for lat in range(math.floor(south), math.floor(north) + 1):
        for lon in range(math.floor(west), math.floor(east) + 1):
            url = DEM_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="W" if lon < 0 else "E", lon=abs(lon))
            try:
                with rasterio.open("/vsicurl/" + url) as src:
                    tmp = np.full((h, w), np.nan, np.float32)
                    reproject(rasterio.band(src, 1), tmp, dst_transform=dst_tr, dst_crs="EPSG:4326",
                              resampling=Resampling.average, dst_nodata=np.nan)
                    dst = np.where(np.isnan(dst), tmp, dst)
            except Exception as ex:  # noqa: BLE001 (ocean / missing tile)
                if "not recognized" not in str(ex) and "No such file" not in str(ex) and "HTTP" not in str(ex):
                    print(f"    DEM tile {lat},{lon}: {ex}")
    elev = np.nan_to_num(dst, nan=float(np.nanmean(dst)) if np.isfinite(dst).any() else 0.0)
    lat_c = (north + south) / 2
    dx = abs(a) * 111320 * math.cos(math.radians(lat_c))
    dy = abs(e) * 110570
    gy, gx = np.gradient(elev, dy, dx)          # gy: change per metre going south (row down)
    dzdx, dzdn = gx, -gy                        # east and north components of the gradient
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
    # aspect = direction the slope FACES (downhill), degrees clockwise from north
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdn)) + 360) % 360
    return elev.astype(np.float32), slope.astype(np.float32), aspect.astype(np.float32)


def tssat_files(fs, fid, sub):
    try:
        L = fs.ls(f"{TSSAT.format(fid=fid)}/{sub}", detail=False)
    except FileNotFoundError:
        return {}
    out = {}
    for p in L:
        name = p.split("/")[-1]
        out[name[:10]] = p
    return out


def read_tif(fs, path):
    import rasterio
    with fs.open(path, "rb") as f:
        b = f.read()
    with rasterio.MemoryFile(b) as mf, mf.open() as ds:
        return ds.read().astype(np.float32), ds.transform, list(ds.descriptions)


def viirs_and_static(fs, fid, lon, lat, dates):
    """VIIRS per label date (af, I4/I5 day, M11, I4/I5 night) and static NDVI/EVI2/land cover,
    all sampled onto the region grid (nearest 375 m pixel)."""
    day = tssat_files(fs, fid, "VIIRS_Day")
    night = tssat_files(fs, fid, "VIIRS_Night")
    pred = tssat_files(fs, fid, "FirePred")
    h, w = lon.shape
    viirs = np.zeros((len(dates), 6, h, w), np.float16)   # af, i4d, i5d, m11, i4n, i5n
    have = np.zeros(len(dates), bool)
    idx = None

    def index_for(tr, shape):
        return regular_index(lon, lat, tr.c, tr.f, tr.a, -tr.e, shape[1], shape[0])

    from concurrent.futures import ThreadPoolExecutor
    want = [day[d] for d in dates if d in day] + [night[d] for d in dates if d in night]
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(want, ex.map(lambda p: read_tif(fs, p), want)))
    for k, d in enumerate(dates):
        if d in day:
            a, tr, desc = got[day[d]]
            rr, cc = index_for(tr, a.shape[1:])
            g = a[:, rr, cc]
            di = {n: i for i, n in enumerate(desc)}
            af = g[di.get("af", 6)]
            viirs[k, 0] = np.isfinite(af) & (af > 0)
            for j, n in [(1, "b4"), (2, "b5"), (3, "m11")]:
                v = g[di[n]] if n in di else np.nan
                viirs[k, j] = np.nan_to_num(v, nan=0.0)
            have[k] = True
        if d in night:
            a, tr, _ = got[night[d]]
            rr, cc = index_for(tr, a.shape[1:])
            g = a[:, rr, cc]
            viirs[k, 4] = np.nan_to_num(g[0], nan=0.0)
            viirs[k, 5] = np.nan_to_num(g[1], nan=0.0)
    static = np.zeros((2 + len(LC_GROUPS), h, w), np.float16)
    if pred:
        first = sorted(pred)[0]
        a, tr, desc = read_tif(fs, pred[first])
        rr, cc = index_for(tr, a.shape[1:])
        di = {n: i for i, n in enumerate(desc)}
        static[0] = np.nan_to_num(a[di.get("NDVI_last", 0)][rr, cc] / 10000.0)
        static[1] = np.nan_to_num(a[di.get("EVI2_last", 1)][rr, cc] / 10000.0)
        lc = a[di.get("LC_Type1", 13)][rr, cc]
        for j, (g, vals) in enumerate(LC_GROUPS.items()):
            static[2 + j] = np.isin(np.round(lc), vals)
    return viirs, have, static


def weather_index(wz, lon, lat):
    """Maps from region cells to GRIDMET and HRRR cells."""
    from scipy.spatial import cKDTree
    glat, glon = wz["gm_lat"], wz["gm_lon"]
    dy = abs(glat[1] - glat[0]) if len(glat) > 1 else 1 / 24
    dx = abs(glon[1] - glon[0]) if len(glon) > 1 else 1 / 24
    g_row, g_col = regular_index(lon, lat, glon[0] - dx / 2, glat[0] + dy / 2, dx, dy, len(glon), len(glat))
    hl, hn = wz["hrrr_lat"], wz["hrrr_lon"]
    if hl.size:
        tree = cKDTree(np.c_[hl.ravel(), hn.ravel()])
        _, ii = tree.query(np.c_[lat.ravel(), lon.ravel()])
        h_idx = ii.reshape(lon.shape).astype(np.int32)
    else:
        h_idx = np.zeros(lon.shape, np.int32)
    return g_row, g_col, h_idx


# ---------------------------------------------------------------------------- driver

def build_fire(fs, out, fire, fid):
    lab = np.load(out / "labels" / f"{fid}.npz")
    arr150 = pool_arrival(lab["arrival_day"], POOL)
    oth150 = pool_max(lab["other_fire_mask"].astype(np.uint8), POOL).astype(bool)
    reg = region_of(arr150, oth150)
    if reg is None:
        return None
    r0, r1, c0, c1 = reg
    tr = list(fire["transform"])
    tr150 = [tr[0] * POOL, tr[1], tr[2], tr[3], tr[4] * POOL, tr[5]]
    lon, lat = grid_centres(tr150, r0, r1, c0, c1)
    dates = [str(d) for d in lab["dates"]]

    t = time.time()
    emb, cov = tessera_region(fs, fid, reg)
    t_tess = time.time() - t
    elev, slope, aspect = dem_region(tr150, reg)
    viirs, viirs_have, static = viirs_and_static(fs, fid, lon, lat, dates)

    rec = dict(fire_id=np.array(fid), dates=np.array(dates), region=np.array(reg), transform150=np.array(tr150),
               arrival=arr150[r0:r1, c0:c1], other=oth150[r0:r1, c0:c1], tessera=emb, tessera_cov=cov,
               elev=elev.astype(np.float16), slope=slope.astype(np.float16), aspect=aspect.astype(np.float16),
               static=static, static_names=np.array(["ndvi", "evi2"] + [f"lc_{k}" for k in LC_GROUPS]),
               viirs=viirs, viirs_have=viirs_have, viirs_names=np.array(["af", "i4d", "i5d", "m11", "i4n", "i5n"]))

    sp = out / "suppression" / f"{fid}.npz"
    if sp.exists():
        s = np.load(sp)
        layers = s["layers"][:, :, r0 * POOL: r1 * POOL, c0 * POOL: c1 * POOL]
        rec.update(sup_dates=s["dates"], sup=pool_max(layers, POOL).astype(np.uint8), sup_channels=s["channels"],
                   ics=s["ics_values"].astype(np.float32), ics_fields=s["ics_fields"])
    wp = out / "weather" / f"{fid}.npz"
    if wp.exists():
        wz = np.load(wp)
        g_row, g_col, h_idx = weather_index(wz, lon, lat)
        hh, hw = wz["hrrr_ana"].shape[2:] if wz["hrrr_ana"].ndim == 4 else (0, 0)
        rec.update(w_dates=wz["dates"], gm=wz["gm"], gm_vars=wz["gm_vars"], gm_row=g_row, gm_col=g_col,
                   hrrr_ana=wz["hrrr_ana"].reshape(len(wz["dates"]), 4, -1) if hh else wz["hrrr_ana"],
                   hrrr_fc=wz["hrrr_fc"].reshape(len(wz["dates"]), 4, -1) if hh else wz["hrrr_fc"],
                   hrrr_vars=wz["hrrr_vars"], hrrr_idx=h_idx)
    hp = out / "human" / f"{fid}.npz"
    if hp.exists():
        hz = np.load(hp)
        rec.update(human=pool_max(hz["layers"][:, r0 * POOL: r1 * POOL, c0 * POOL: c1 * POOL], POOL).astype(np.uint8),
                   human_channels=hz["channels"])
    np.savez_compressed(out / "train" / f"{fid}.npz", **rec)
    return dict(fid=fid, h=r1 - r0, w=c1 - c0, days=len(dates), tessera_s=round(t_tess),
                viirs_days=int(viirs_have.sum()), sup="sup" in rec, weather="gm" in rec, human="human" in rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nifc_out")
    ap.add_argument("--fires", nargs="*")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th fire starting at i")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    fires = json.loads((out / "fires.json").read_text())
    summ = pd.read_csv(out / "summary.csv", dtype={"fire_id": str})
    ok = summ.loc[summ["status"] == "ok", "fire_id"].tolist()
    if args.fires:
        ok = [f for f in ok if f in set(args.fires)]
    i, n = map(int, args.shard.split("/"))
    ok = ok[i::n]
    rows = []
    for k, fid in enumerate(ok):
        if (out / "train" / f"{fid}.npz").exists() and not args.redo:
            continue
        t0 = time.time()
        try:
            r = build_fire(fs, out, fires[fid], fid)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[{k + 1}/{len(ok)}] {fid}: ERROR {ex}")
            continue
        print(f"[{k + 1}/{len(ok)}] {fid}: {r} ({time.time() - t0:.0f}s)", flush=True)
        if r:
            rows.append(r)
    print("Done:", out / "train")


if __name__ == "__main__":
    sys.exit(main())
