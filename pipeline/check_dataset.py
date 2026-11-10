"""
Automated health check of the built dataset: is every fire processed correctly?

For each fire it checks, and summarises over all fires:
  labels      perimeter times increase, arrival indices valid, usable pairs exist, few excluded cells
  alignment   VIIRS / GOES detections fall on or near the burned area (a shifted layer fails here)
  coverage    HRRR hours cover each pair's time window, GRIDMET covers the fire dates,
              suppression and LANDFIRE follow the expected availability by year
  values      no NaN / inf, plausible terrain, fuels burnable inside the fire, Tessera cover
  loader      model/data.py builds every sampled pair: finite inputs, channel ranges, mask sizes,
              Rothermel spread rates in a plausible range

  python check_dataset.py --data ~/scratch/wildfire/my            # all fires
  python check_dataset.py --data ... --fires F6bd2bd1f45 F1c92720e56 --pairs 5
Writes <data>/check_dataset.csv (one row per fire) and prints the summary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "model"))

# expected availability (see build_multiyear.py)
GOES_FROM = pd.Timestamp("2017-06-01", tz="UTC")
HRRR_FROM = pd.Timestamp("2014-08-01", tz="UTC")
SUPPRESSION_MAPS_FROM = 2018
LANDFIRE_FROM = 2017


def _z(d, name):
    p = d / f"{name}.npz"
    return dict(np.load(p, allow_pickle=False)) if p.exists() else None


def near(mask, km, cell_km):
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=max(1, int(round(km / cell_km))))


def check_fire(d, n_pairs):
    import data as D
    g = json.loads((d / "grid.json").read_text())
    r = {"fire_id": g["fire_id"], "name": g["name"], "year": g["year"]}
    ck = float(g.get("res_m", 150)) / 1000
    lab = _z(d, "labels")
    arr, times = lab["arrival"], pd.to_datetime(lab["times"], utc=True)
    n = len(times)
    r["perimeters"] = n
    r["times_increasing"] = bool((np.diff(times.view("int64")) > 0).all())
    r["arrival_valid"] = bool(arr.min() >= -1 and arr.max() <= n - 1)
    burned = arr >= 0
    r["burned_km2"] = float(burned.sum() * ck * ck)
    t0, t1 = times.min(), times.max()

    # ---- alignment
    v = _z(d, "viirs")
    if v is not None:
        days = pd.to_datetime(v["dates"]).tz_localize("UTC")
        sel = (days >= t0.normalize() - pd.Timedelta(days=1)) & (days <= t1.normalize() + pd.Timedelta(days=1))
        det = ((v["af_day"][sel].astype(int) + v["af_night"][sel]) > 0).any(0) if sel.any() else np.zeros_like(burned)
        r["viirs_px"] = int(det.sum())
        fires_here = burned | lab["other"]                       # neighbouring fires in the window count too
        r["viirs_on_fire"] = float((det & near(fires_here, 1.0, ck)).sum() / max(det.sum(), 1)) if det.any() else np.nan
        new = arr >= 1                                            # growth after the first perimeter (VIIRS can see it)
        r["new_burn_km2"] = float(new.sum() * ck * ck)
        r["new_burn_with_viirs"] = float((new & near(det, 1.0, ck)).sum() / max(new.sum(), 1)) if new.any() else np.nan
    gz = _z(d, "goes")
    r["goes_expected"] = bool(t1 >= GOES_FROM)
    r["goes_present"] = gz is not None and len(gz["t"]) > 0
    r["goes_file"] = gz is not None
    if r["goes_present"]:
        dx, _, x0, _, ey, y0 = g["transform"]
        cc = np.floor((gz["lon"] - x0) / dx).astype(int)
        rr = np.floor((gz["lat"] - y0) / ey).astype(int)
        ins = (rr >= 0) & (rr < g["height"]) & (cc >= 0) & (cc < g["width"])
        nb = near(burned | lab["other"], 3.0, ck)
        r["goes_points"] = int(ins.sum())
        r["goes_on_fire"] = float(nb[rr[ins], cc[ins]].mean()) if ins.sum() >= 20 else np.nan   # a few strays mean nothing

    # ---- coverage
    hz = _z(d, "hrrr")
    r["hrrr_expected"] = bool(t0 >= HRRR_FROM)
    r["hrrr_present"] = hz is not None
    wz = _z(d, "weather")
    r["gridmet_covers"] = bool(wz is not None and pd.to_datetime(wz["dates"]).min() <= t0.tz_convert(None).normalize()
                               and pd.to_datetime(wz["dates"]).max() >= t1.tz_convert(None).normalize() - pd.Timedelta(days=1))
    r["gridmet_nan"] = float(np.isnan(wz["gm"]).mean()) if wz is not None else np.nan
    lf = _z(d, "landfire")
    r["landfire"] = "missing" if lf is None or bool(lf.get("missing", True)) else str(lf["version"])
    r["landfire_ok"] = (r["landfire"] == "missing") == (g["year"] < LANDFIRE_FROM)
    if lf is not None and not bool(lf.get("missing", True)):
        fb = lf["fbfm40"]
        r["burnable_in_fire"] = float(((fb >= 101) & (fb <= 204))[burned].mean()) if burned.any() else np.nan
    sz = _z(d, "suppression")
    r["supp_spatial"] = bool(sz is not None and bool(sz["has_spatial"]))
    r["supp_ics"] = bool(sz is not None and bool(sz["has_ics"]))
    r["supp_ok"] = r["supp_spatial"] or r["supp_ics"] or g["year"] >= 2021 or g["year"] < SUPPRESSION_MAPS_FROM
    hu = _z(d, "human")
    r["human_present"] = hu is not None
    tz = _z(d, "tessera")
    if tz is not None:
        r["tessera"] = "missing" if bool(tz.get("missing", True)) else f"{int(tz['year'])}"
        r["tessera_cover"] = float(tz["cover"].mean()) if not bool(tz.get("missing", True)) else np.nan
        r["tessera_finite"] = bool(np.isfinite(tz["embedding"]).all()) if not bool(tz.get("missing", True)) else True

    # ---- values
    t = _z(d, "terrain")
    r["elev_min"], r["elev_max"] = float(np.nanmin(t["elev"])), float(np.nanmax(t["elev"]))
    r["terrain_nan"] = float(np.mean([np.isnan(t[k].astype(np.float32)).mean() for k in ("elev", "slope", "aspect")]))

    # ---- loader: the exact samples training builds
    f = D.Fire(d)
    r["pairs"] = len(f.pairs)
    if f.pairs:
        idx = np.unique(np.linspace(0, len(f.pairs) - 1, min(n_pairs, len(f.pairs))).round().astype(int))
        bad_chan, excl, scored, rr_med, rr_zero_in_new, hrrr_cov, newb = [], [], [], [], [], [], []
        for k in idx:
            i, gap = f.pairs[k]
            feats, phys, y, m, b0 = f.sample(i, gap, (0, f.H, 0, f.W))
            for gname, a in feats.items():
                if not np.isfinite(a).all() or np.abs(a).max() > 50:
                    bad_chan.append(f"{gname}:{np.abs(a).max():.0f}")
            new = y & ~b0 & ~f.other
            newb.append(int(new.sum()))
            excl.append(float((new & ~m).sum() / max(new.sum(), 1)))
            scored.append(int(m.sum()))
            R = phys["r_head"]
            rr_med.append(float(np.median(R[new])) if new.any() else np.nan)
            rr_zero_in_new.append(float((R[new] < 0.05).mean()) if new.any() else np.nan)
            hrrr_cov.append(float(feats["hrrr"][6].max()))
        r["bad_channels"] = ";".join(sorted(set(bad_chan)))
        r["excluded_new_frac"] = float(np.mean(excl))
        r["pairs_without_growth"] = float(np.mean(np.array(newb) == 0))
        r["ros_median_new_m_min"] = float(np.nanmedian(rr_med)) if np.isfinite(rr_med).any() else np.nan
        r["ros_near_zero_in_new"] = float(np.nanmean(rr_zero_in_new)) if np.isfinite(rr_zero_in_new).any() else np.nan
        r["hrrr_pair_coverage"] = float(np.mean(hrrr_cov))
    return r


PROBLEMS = [  # (column, test returning True when the fire FAILS, description)
    ("times_increasing", lambda s: s == False, "perimeter times not increasing"),  # noqa: E712
    ("arrival_valid", lambda s: s == False, "invalid arrival index"),  # noqa: E712
    ("pairs", lambda s: s < 1, "no usable pairs"),
    ("viirs_on_fire", lambda s: s < 0.6, "under 60% of VIIRS detections within 1 km of a burn (misaligned?)"),
    ("new_burn_with_viirs", lambda s: s < 0.3, "under 30% of the growth has VIIRS within 1 km (re-mapping, not fire?)"),
    ("goes_on_fire", lambda s: s < 0.5, "under 50% of GOES pixels within 3 km of a burn (20+ pixels)"),
    ("gridmet_covers", lambda s: s == False, "GRIDMET does not cover the fire dates"),  # noqa: E712
    ("gridmet_nan", lambda s: s > 0.2, "GRIDMET over 20% NaN"),
    ("landfire_ok", lambda s: s == False, "LANDFIRE availability unexpected for the year"),  # noqa: E712
    ("burnable_in_fire", lambda s: s < 0.5, "under half the burned cells have a burnable fuel model"),
    ("terrain_nan", lambda s: s > 0.01, "terrain has NaN"),
    ("bad_channels", lambda s: s.fillna("").str.len() > 0, "loader produced non-finite or out-of-range inputs"),
    ("excluded_new_frac", lambda s: s > 0.3, "over 30% of new burn excluded from scoring"),
    ("ros_near_zero_in_new", lambda s: s > 0.5, "Rothermel ~0 on over half the cells that burned"),
    ("tessera_finite", lambda s: s == False, "Tessera has non-finite values"),  # noqa: E712
]


def summary(df):
    lines = [f"{len(df)} fires, {df['year'].min()}-{df['year'].max()}, {int(df['pairs'].sum())} pairs"]
    for col, test, desc in PROBLEMS:
        if col not in df:
            continue
        s = df[col]
        fail = test(s) & s.notna() if s.dtype != object else test(s)
        lines.append(f"  {'OK  ' if fail.sum() == 0 else 'WARN'} {desc}: {int(fail.sum())} fires"
                     + (f"  e.g. {', '.join(df.loc[fail, 'name'].astype(str).head(4))}" if fail.sum() else ""))
    ex = df[df.goes_expected & ~df.goes_file]
    lines.append(f"  {'OK  ' if ex.empty else 'WARN'} GOES expected (after mid-2017) but no file: {len(ex)} fires")
    ex = df[df.goes_expected & df.goes_file & ~df.goes_present]
    lines.append(f"  info GOES file but no detections (usually small fires): {len(ex)} fires, "
                 f"median {ex.burned_km2.median() if len(ex) else float('nan'):.0f} km²")
    ex = df[df.hrrr_expected & ~df.hrrr_present]
    lines.append(f"  {'OK  ' if ex.empty else 'WARN'} HRRR expected but missing: {len(ex)} fires")
    lines.append("  medians: " + ", ".join(f"{c} {df[c].median():.2f}" for c in
                 ("viirs_on_fire", "new_burn_with_viirs", "goes_on_fire", "burnable_in_fire", "excluded_new_frac",
                  "ros_median_new_m_min", "hrrr_pair_coverage") if c in df))
    lines.append("  by year (fires / with GOES / HRRR / suppression maps / ICS-209 / LANDFIRE):")
    for y, gdf in df.groupby("year"):
        lines.append(f"    {y}: {len(gdf):3d} / {gdf.goes_present.sum():3d} / {gdf.hrrr_present.sum():3d} / "
                     f"{gdf.supp_spatial.sum():3d} / {gdf.supp_ics.sum():3d} / {(gdf.landfire != 'missing').sum():3d}"
                     f"  ({', '.join(sorted(set(gdf.landfire)))})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fires", nargs="*")
    ap.add_argument("--pairs", type=int, default=6, help="pairs sampled per fire for the loader check")
    a = ap.parse_args()
    root = Path(a.data).expanduser()
    dirs = sorted(p.parent for p in (root / "fires").glob("*/labels.npz"))
    if a.fires:
        dirs = [d for d in dirs if d.name in set(a.fires)]
    rows = []
    for k, d in enumerate(dirs):
        try:
            rows.append(check_fire(d, a.pairs))
        except Exception as ex:  # noqa: BLE001
            rows.append({"fire_id": d.name, "error": repr(ex)})
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(dirs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(root / "check_dataset.csv", index=False)
    if "error" in df and df["error"].notna().any():
        print(f"ERRORS in {df['error'].notna().sum()} fires, e.g. {df.loc[df['error'].notna(), 'error'].iloc[0]}")
        df = df[df["error"].isna()]
    print(summary(df))
    print(f"per-fire table: {root / 'check_dataset.csv'}")


if __name__ == "__main__":
    main()
