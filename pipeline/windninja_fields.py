"""
Terrain-adjusted wind: how each fire's terrain reshapes the wind, from WindNinja (USFS Missoula Fire Lab).

HRRR gives one wind per 3 km square, about 400 of our cells, so every cell in a canyon, on a ridge and in
the lee of it gets the same wind. WindNinja's mass-conserving solver takes a broad wind plus the terrain and
returns the wind at every cell: faster over ridges and through gaps, slower in sheltered ground, turned
along drainages.

That solver is linear in the input wind (measured on the Feather River canyon: doubling the speed, or adding
two directions, reproduces a direct run to 0.1-0.3 % of the input speed). So two runs per fire describe every
wind: one from the west, one from the south. For a broad wind (u0, v0) the wind at a cell is
    u = u0 * resp[0, 0] + v0 * resp[1, 0]        v = u0 * resp[0, 1] + v0 * resp[1, 1]
A third run from the north-west checks the fire's own response against that rule (check_p95, check_max:
the error as a fraction of the input speed).

  python pipeline/windninja_fields.py --data ~/scratch/wildfire/my --wn ~/scratch/envs/windninja/bin/WindNinja_cli \
         --years 2024,2025 --workers 6
Writes fires/<fire_id>/terrain_wind.npz: resp (2, 2, H, W) float16 on the fire grid, valid, check_*, settings.
Install WindNinja once:  conda create -y -p ~/scratch/envs/windninja -c conda-forge windninja

What this is not: no daily upslope / downslope winds (they need the hour, sun and temperature per run), no
eddies behind ridges (the momentum solver does those, at 50-100x the cost), and the terrain is our 150 m
average, so features narrower than a few cells are already smoothed. Cells within about 3 km of the grid
edge feel the solver's boundary; the fire grids carry a 9.6 km margin for that reason.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SPEED = 10.0            # m/s given to WindNinja (the response is per unit of input speed)
HEIGHT = 10.0           # m above the vegetation, in and out: HRRR's 10 m wind
VEG = "grass"           # one roughness everywhere: this file isolates the terrain's effect
DIRS = (270.0, 180.0)   # winds FROM the west (blowing east: u = +1) and FROM the south (blowing north: v = +1)
CHECK_DIR = 315.0
MAX_CELLS = 400_000     # WindNinja needs about 11 GB per million cells: larger grids are solved on a coarser mesh
VERSION = 1


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def fill_nearest(a):
    """NaN cells take the value of the nearest valid cell (WindNinja needs a full elevation grid)."""
    from scipy.ndimage import distance_transform_edt
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    i = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(i)]


def to_utm(elev, g):
    """The fire's lon/lat elevation grid resampled to square cells in the local UTM zone (WindNinja needs
    a projected grid). Returns (dem, affine transform, crs)."""
    from affine import Affine
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    dx, _, x0, _, dy, y0 = g["transform"]
    H, W = elev.shape
    bounds = (x0, y0 + dy * H, x0 + dx * W, y0)
    crs = f"EPSG:{utm_epsg((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)}"
    res = float(g.get("res_m", 150.0))
    tr, w, h = calculate_default_transform("EPSG:4326", crs, W, H, *bounds, resolution=res)
    dem = np.full((h, w), np.nan, np.float32)
    reproject(fill_nearest(elev.astype(np.float32)), dem, src_transform=Affine(*g["transform"]), src_crs="EPSG:4326",
              dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return fill_nearest(dem), tr, crs


def wn_env(wn):
    """WindNinja run straight from its conda folder (not activated) needs to be told where its data is."""
    p = Path(wn).resolve().parent.parent
    e = dict(os.environ, PATH=f"{p / 'bin'}:{os.environ.get('PATH', '')}")
    for k, d in (("GDAL_DATA", "share/gdal"), ("PROJ_DATA", "share/proj"), ("PROJ_LIB", "share/proj"),
                 ("WINDNINJA_DATA", "share/windninja")):
        if (p / d).exists():
            e[k] = str(p / d)
    for k in ("LD_LIBRARY_PATH", "PYTHONPATH", "GDAL_DRIVER_PATH"):      # the caller's libraries are not WindNinja's
        e.pop(k, None)
    return e


def run_wn(wn, dem_tif, direction, work, res, threads=1):
    """One WindNinja run. Returns (2, h, w): u (east) and v (north) at 10 m, per unit of input speed, and the
    transform of that grid (WindNinja writes its output at the mesh resolution)."""
    import rasterio
    out = Path(work) / f"d{int(direction)}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(wn), "--num_threads", str(threads), "--elevation_file", str(dem_tif),
           "--initialization_method", "domainAverageInitialization",
           "--input_speed", str(SPEED), "--input_speed_units", "mps", "--output_speed_units", "mps",
           "--input_direction", str(direction),
           "--input_wind_height", str(HEIGHT), "--units_input_wind_height", "m",
           "--output_wind_height", str(HEIGHT), "--units_output_wind_height", "m",
           "--vegetation", VEG, "--mesh_resolution", str(res), "--units_mesh_resolution", "m",
           "--write_ascii_output", "true", "--ascii_out_uv", "true", "--output_path", str(out)]
    r = subprocess.run(cmd, env=wn_env(wn), capture_output=True, text=True)
    files = {k: sorted(out.glob(f"*_{k}.asc")) for k in ("u", "v")}
    if r.returncode != 0 or not files["u"] or not files["v"]:
        raise RuntimeError(f"WindNinja failed (exit {r.returncode}): {(r.stdout + r.stderr)[-400:]}")
    uv = []
    for k in ("u", "v"):
        with rasterio.open(files[k][0]) as s:
            a = s.read(1).astype(np.float32)
            a[a == s.nodata] = np.nan
            uv.append(a)
            tr = s.transform
    return np.stack(uv) / SPEED, tr


def unit(direction):
    """(u, v) of a unit wind FROM `direction` degrees."""
    d = np.radians(direction)
    return np.array([-np.sin(d), -np.cos(d)])


def response(elev, g, wn, work, threads=1, check=True):
    """The fire's wind response on its own grid, and the linearity check."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject
    dem, tr, crs = to_utm(elev, g)
    res = float(g.get("res_m", 150.0))
    mesh = res * max(1, int(np.ceil(np.sqrt(dem.size / MAX_CELLS))))
    tif = Path(work) / "dem.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1, dtype="float32",
                       crs=crs, transform=tr) as d:
        d.write(dem, 1)
    runs = [run_wn(wn, tif, a, work, mesh, threads) for a in DIRS]
    F, tr = np.stack([r[0] for r in runs]), runs[0][1]                          # (2 inputs, 2 components, h, w)
    chk = {}
    if check:
        a, b = unit(CHECK_DIR)                                                   # CHECK = a * west run + b * south run
        err = np.hypot(*(a * F[0] + b * F[1] - run_wn(wn, tif, CHECK_DIR, work, mesh, threads)[0]))
        chk = dict(check_p95=np.float32(np.nanpercentile(err, 95)), check_max=np.float32(np.nanmax(err)))
    H, W = elev.shape
    resp = np.full((2, 2, H, W), np.nan, np.float32)
    reproject(F.reshape(4, *F.shape[2:]), resp.reshape(4, H, W), src_transform=tr, src_crs=crs,
              dst_transform=Affine(*g["transform"]), dst_crs="EPSG:4326", resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    valid = np.isfinite(resp).all((0, 1))
    flat = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)[:, :, None, None]     # no terrain effect
    resp = np.where(valid, resp, flat)
    return dict(resp=resp.astype(np.float16), valid=valid, mesh_m=np.array(mesh), **chk)


def one(job):
    d, wn, threads, check, force = job
    d = Path(d)
    dst = d / "terrain_wind.npz"
    if dst.exists() and not force:
        return d.name, "exists", 0.0
    t0 = time.time()
    try:
        g = json.loads((d / "grid.json").read_text())
        elev = np.load(d / "terrain.npz")["elev"]
        work = tempfile.mkdtemp(prefix=f"wn_{d.name}_", dir=os.environ.get("TMPDIR"))
        try:
            out = response(elev, g, wn, work, threads, check)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        tmp = d / "terrain_wind.tmp.npz"
        np.savez_compressed(tmp, version=np.array(VERSION), speed=np.array(SPEED), height=np.array(HEIGHT),
                            vegetation=np.array(VEG), dirs=np.array(DIRS), **out)
        tmp.replace(dst)
        return d.name, f"ok p95 {float(out.get('check_p95', np.nan)):.4f} max {float(out.get('check_max', np.nan)):.3f}", time.time() - t0
    except Exception as ex:  # noqa: BLE001
        return d.name, f"ERROR {type(ex).__name__}: {str(ex)[:300]}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder (holds fires/<id>/grid.json and terrain.npz)")
    ap.add_argument("--wn", required=True, help="path to WindNinja_cli")
    ap.add_argument("--years", help="only fires of these years, e.g. 2024,2025")
    ap.add_argument("--fires", help="only these fire ids (comma separated, or a CSV with a fire_id column)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1, help="threads per WindNinja run")
    ap.add_argument("--no-check", action="store_true", help="skip the third run that checks linearity")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not Path(a.wn).exists():
        raise SystemExit(f"{a.wn} not found: conda create -y -p ~/scratch/envs/windninja -c conda-forge windninja")
    root = Path(a.data).expanduser() / "fires"
    dirs = sorted(p for p in root.iterdir() if (p / "grid.json").exists() and (p / "terrain.npz").exists())
    if a.years:
        ys = {int(y) for y in a.years.split(",")}
        dirs = [p for p in dirs if json.loads((p / "grid.json").read_text()).get("year") in ys]
    if a.fires:
        if Path(a.fires).exists():
            import pandas as pd
            keep = set(pd.read_csv(a.fires)["fire_id"].dropna().astype(str))
        else:
            keep = set(a.fires.split(","))
        dirs = [p for p in dirs if p.name in keep]
    print(f"{len(dirs)} fires, {a.workers} at a time", flush=True)
    jobs = [(str(p), a.wn, a.threads, not a.no_check, a.force) for p in dirs]
    n = dict(ok=0, exists=0, error=0)
    p95 = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (fid, msg, sec) in enumerate(ex.map(one, jobs), 1):
            n["error" if msg.startswith("ERROR") else msg.split()[0]] += 1
            if msg.startswith("ok p95"):
                p95.append(float(msg.split()[2]))
            if not msg.startswith("exists"):
                print(f"[{k}/{len(jobs)}] {fid}: {msg} ({sec:.0f} s)", flush=True)
    if p95 and np.isfinite(p95).any():
        print(f"linearity check (error as a share of the input speed, 95th percentile per fire): "
              f"median {np.nanmedian(p95):.4f}, worst {np.nanmax(p95):.4f}")
    print(f"done in {(time.time() - t0) / 60:.1f} min: {n['ok']} built, {n['exists']} already there, {n['error']} errors")


if __name__ == "__main__":
    main()
