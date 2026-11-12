"""
Live (and replay) fire-spread forecasts with the project's own data pipeline and models.

  live     today's active fires from WFIGS; for one fire, build its layers from the live sources
           (Copernicus DEM, LANDFIRE, GRIDMET, FIRMS VIIRS NRT, the latest HRRR forecast run) with the
           same stage code as the training dataset, then forecast the next N hours from now. The
           current fire = latest perimeter + VIIRS detections of the last 48 h.
  replay   the same forecast "as of" a past perimeter of a dataset fire, with the real next perimeter
           to compare against.

  python live/forecast_now.py list                                  # active fires
  python live/forecast_now.py live --fire "FOOT CREEK" --out data/live
  python live/forecast_now.py replay --fire-dir data/fires/Ff1ba680f61 --pair 22 --out data/live
Model: --weights runs/<name>.pt --mode physics2|hybrid2|unet2|... (default physics2 with its 5 learned
settings; any trained model from runs/ works the same way). Weather is always forecast-only.
Outputs (OUT/forecasts/<id>/): forecast.npz and meta.json, read by live/render_page.py.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "pipeline", ROOT / "model"):
    sys.path.insert(0, str(p))

WFIGS = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
         "WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query")
LIVE_STAGES = ["labels", "viirs", "terrain", "weather", "landfire", "history"]
NEAR_KM = 2.0          # satellite detections within this distance of the fire join its starting shape


# ------------------------------------------------------------------------------ active fires
def active_fires(min_acres=100, max_age_days=60):
    """Current WFIGS perimeters with their VIIRS detections of the last 48 h."""
    import requests
    from shapely.geometry import shape
    from live import firms
    from live import wfigs_poller as WP
    r = {"features": [f for f in WP.fetch(days=max_age_days) if (f["properties"].get("poly_GISAcres") or 0) >= min_acres]}
    now = pd.Timestamp.now(tz="UTC")
    det = firms.recent(now)
    rows = []
    for f in r.get("features", []):
        p = f["properties"]
        if not f.get("geometry") or not p.get("poly_PolygonDateTime"):
            continue
        t = pd.to_datetime(p["poly_PolygonDateTime"], unit="ms", utc=True)
        if (now - t).days > max_age_days:
            continue
        geom = shape(f["geometry"])
        w, s, e, n = geom.bounds
        pad = 0.05
        near = det[(det.longitude.between(w - pad, e + pad)) & (det.latitude.between(s - pad, n + pad))] if len(det) else det
        rows.append(dict(id=p["attr_UniqueFireIdentifier"], name=p["attr_IncidentName"], state=p.get("attr_POOState"),
                         acres=round(p["poly_GISAcres"] or 0), contained=p.get("attr_PercentContained"),
                         updated=t.isoformat(), lon=round(geom.centroid.x, 4), lat=round(geom.centroid.y, 4),
                         detections_48h=int(len(near)), feature=f,
                         discovered=pd.to_datetime(p["attr_FireDiscoveryDateTime"], unit="ms", utc=True).isoformat() if p.get("attr_FireDiscoveryDateTime") else None))
    rows.sort(key=lambda r: r["updated"], reverse=True)
    rows.sort(key=lambda r: -r["detections_48h"])            # most active first, then most recently mapped
    return rows


# ------------------------------------------------------------------------------ live build
def _hrrr_live(d, g, now, hours):
    """hrrr.npz for a live fire: analyses every 3 h over the last 36 h, and the latest HRRR run
    issued at least 90 min ago at +3 h steps up to `hours`, with fc_issued (the run time)."""
    import requests
    from scipy.spatial import cKDTree
    import build_multiyear as BM
    import weather_features as WF
    sess = requests.Session()
    lat, lon = WF.hrrr_latlon(sess)
    win = WF.hrrr_window(lat, lon, g["bounds_lonlat"])
    r0, r1, c0, c1 = win
    cut = lambda got: np.stack([got.get(n, np.full(lat.shape, np.nan))[r0:r1, c0:c1] for n, _ in WF.HRRR_FIELDS])  # noqa: E731
    ana_t, ana = [], []
    for t in pd.date_range((now - pd.Timedelta(hours=36)).floor("3h"), now.floor("h"), freq="3h"):
        got = WF.hrrr_fetch(sess, WF.HRRR.format(d=t.strftime("%Y%m%d"), h=t.hour, f=0))
        if got:
            ana_t.append(t); ana.append(cut(got))
    run = (now - pd.Timedelta(minutes=90)).floor("h")
    fc_t, fc, issued = [], [], None
    for back in range(0, 7):                              # newest run that has the forecast hours we need
        init = run - pd.Timedelta(hours=back)
        maxf = 48 if init.hour % 6 == 0 else 18
        steps = [f for f in range(3, min(hours, maxf) + 1, 3)]
        got = [WF.hrrr_fetch(sess, WF.HRRR.format(d=init.strftime("%Y%m%d"), h=init.hour, f=f)) for f in steps[:1]]
        if got and got[0]:
            issued = init
            for f in steps:
                gg = got[0] if f == steps[0] else WF.hrrr_fetch(sess, WF.HRRR.format(d=init.strftime("%Y%m%d"), h=init.hour, f=f))
                if gg:
                    fc_t.append(init + pd.Timedelta(hours=f)); fc.append(cut(gg))
            break
    hlat, hlon = lat[r0:r1, c0:c1], lon[r0:r1, c0:c1]
    glon, glat = BM.centres(g)
    cs = math.cos(math.radians(float(glat.mean())))
    _, idx = cKDTree(np.c_[hlat.ravel(), hlon.ravel() * cs]).query(np.c_[glat.ravel(), glon.ravel() * cs])
    shape4 = (0, 4, r1 - r0, c1 - c0)
    # The models were trained on the weather DURING the forecast window (HRRR analyses). Live, the HRRR
    # forecast for the window stands in for it: forecast steps go into `fields` after the analyses, so the
    # loader (WEATHER_MODE "observed") reads the window's weather from the forecast, not from yesterday.
    all_t, all_f = ana_t + fc_t, ana + fc
    np.savez_compressed(d / "hrrr.npz", times=np.array([t.value for t in all_t], np.int64),
                        fields=np.array(all_f, np.float16).reshape(-1, *shape4[1:]) if all_f else np.zeros(shape4, np.float16),
                        fc_times=np.array([t.value for t in fc_t], np.int64),
                        fc=np.array(fc, np.float16).reshape(-1, *shape4[1:]) if fc else np.zeros(shape4, np.float16),
                        fc_issued=np.array([issued.value] * len(fc_t), np.int64) if fc else np.zeros(0, np.int64),
                        vars=np.array([n for n, _ in WF.HRRR_FIELDS]), idx=idx.reshape(g["height"], g["width"]).astype(np.int32))
    return dict(analyses=len(ana_t), forecast_steps=len(fc_t), forecast_run=issued.isoformat() if issued else None)


def _points(d, g, now, hours_back=48):
    """VIIRS detections (NOAA-20, NOAA-21, S-NPP near real time) in the fire's grid over the last `hours_back`
    hours, with exact pass times. Falls back to the build's points.csv.gz if the live feed fails."""
    from live import firms
    pts = None
    try:
        pts = firms.recent(now, tuple(g["bounds_lonlat"]), days=int(np.ceil(hours_back / 24)) + 1)
    except Exception as ex:
        print(f"  FIRMS recent failed ({type(ex).__name__}); using the build's points", flush=True)
    if (pts is None or not len(pts)) and (Path(d) / "points.csv.gz").exists():
        pts = pd.read_csv(Path(d) / "points.csv.gz")
    if pts is None or not len(pts) or "acq_time" not in pts:
        return None
    hhmm = pts["acq_time"].astype(int).astype(str).str.zfill(4)
    pts = pts.assign(t=(pd.to_datetime(pts["acq_date"]) + pd.to_timedelta(hhmm.str[:2].astype(int), unit="h")
                        + pd.to_timedelta(hhmm.str[2:].astype(int), unit="m")).dt.tz_localize("UTC"))
    pts = pts.drop_duplicates(["latitude", "longitude", "t"])
    return pts[(pts["t"] <= now) & (pts["t"] >= now - pd.Timedelta(hours=hours_back))]


def start_fire(d, g, fire, lab, now, t_per):
    """The fire at forecast start, as accurately as the sources allow.
    With a perimeter: the perimeter, plus each satellite detection made AFTER it within NEAR_KM (the fire
    grew since it was mapped), one 150 m cell per detection at its exact location (not a 375 m pixel).
    Without one: a compact shape of the reported size, grown outward from the incident point and the exact
    detection points; if the detections cover more ground than the reported size (a stale report), they win.
    Returns (arrival raster, description, detection cells used)."""
    from scipy.ndimage import binary_dilation, distance_transform_edt
    H, W = g["height"], g["width"]
    dx, _, x0, _, ndy, y0 = g["transform"]
    cell_ac = g["res_m"] ** 2 / 4046.856
    arrival = lab["arrival"].copy()
    other = lab["other"].astype(bool)
    pts = _points(d, g, now)

    def cells(lon, lat):
        c = np.floor((np.asarray(lon) - x0) / dx).astype(int)
        r = np.floor((np.asarray(lat) - y0) / ndy).astype(int)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        return r[ok], c[ok]
    near_it = max(1, int(round(NEAR_KM * 1000 / g["res_m"])))
    if fire.get("source") == "incident":
        seed = np.zeros((H, W), bool)
        ri, ci = cells([fire["lon"]], [fire["lat"]])
        if not len(ri):
            ri, ci = np.array([H // 2]), np.array([W // 2])
        seed[ri, ci] = True
        reach = binary_dilation(seed, iterations=int(round(3000 / g["res_m"])))   # same 3 km search as the fire list
        nd = 0
        if pts is not None and len(pts):
            rr, cc = cells(pts["longitude"], pts["latitude"])
            keep = reach[rr, cc]
            seed[rr[keep], cc[keep]] = True
            nd = int(keep.sum())
        target = max(int(round((fire.get("acres") or 0) / cell_ac)), int(seed.sum()), 1)
        dist = distance_transform_edt(~seed)
        dist[other] = np.inf
        order = np.argsort(dist, axis=None, kind="stable")[:target]
        start = np.zeros(H * W, bool)
        start[order] = True
        arrival = np.where(start.reshape(H, W), 0, -1).astype(arrival.dtype)
        how = (f"reported size ({fire.get('acres') or 0:g} ac) around the incident and {nd} detections"
               + (" (detections cover more than the report)" if seed.sum() > round((fire.get("acres") or 0) / cell_ac) else ""))
        return arrival, how, nd
    core = arrival >= 0
    near = binary_dilation(core, iterations=near_it) if core.any() else np.ones((H, W), bool)
    added = 0
    if pts is not None and len(pts):
        newer = pts[pts["t"] > t_per]
        rr, cc = cells(newer["longitude"], newer["latitude"])
        ok = near[rr, cc] & ~other[rr, cc] & (arrival[rr, cc] < 0)
        arrival[rr[ok], cc[ok]] = 0
        added = int(ok.sum())
    return arrival, f"perimeter + {added} newer detections", added


def build_live(fire, out, hours=24, now=None):
    """Layers for one active fire (a row from active_fires) with the dataset's own stage code."""
    import geopandas as gpd
    from shapely.geometry import shape
    import build_multiyear as BM
    now = (now or pd.Timestamp.now(tz="UTC")).floor("min")
    geom = shape(fire["feature"]["geometry"])
    fid = "LIVE_" + fire["id"].replace("-", "")
    out = Path(out)
    cat = out / "catalog_live" / fid
    cat.mkdir(parents=True, exist_ok=True)
    t_per = pd.Timestamp(fire["updated"])
    per = gpd.GeoDataFrame(dict(fire_id=[fid], perim_time=[t_per], burn_day=[str(t_per.date())],
                                map_method=[fire["feature"]["properties"].get("poly_MapMethod") or ""], source=["wfigs"]),
                           geometry=[geom], crs="EPSG:4326")
    per.to_parquet(cat / "perimeters.parquet")
    w, s, e, n = geom.bounds
    row = pd.Series(dict(fire_id=fid, name=fire["name"], year=int(now.year), west=w, south=s, east=e, north=n,
                         first=str(t_per), last=str(now)))
    BM._CTX.clear()
    g = BM.make_grid(row)
    d = out / "fires" / fid
    d.mkdir(parents=True, exist_ok=True)
    (d / "grid.json").write_text(json.dumps(g))
    t0 = time.time()
    _, done = BM.build_one(out, cat, g, LIVE_STAGES)
    for attempt in (1, 2):                                # live sources drop connections now and then: retry failed stages
        failed = [s.split(":")[0] for s in done if "ERROR" in s]
        if not failed:
            break
        print(f"  retrying {failed} (attempt {attempt + 1})", flush=True)
        time.sleep(10 * attempt)
        _, again = BM.build_one(out, cat, g, failed)
        done = [s for s in done if s.split(":")[0] not in failed] + again
    print(f"  stages: {done} ({time.time() - t0:.0f}s)", flush=True)
    if not (d / "landfire.npz").exists():                  # newest LANDFIRE version may not exist yet: one older
        g2 = dict(g, year=g["year"] - 1)
        BM.build_one(out, cat, g2, ["landfire"])
    info = _hrrr_live(d, g, now, hours)
    print(f"  HRRR: {info}", flush=True)
    # the fire at forecast start (see start_fire): perimeter + newer satellite detections, or for a fire with no
    # perimeter, a compact shape of its reported size around the incident and its detections
    lab = dict(np.load(d / "labels.npz"))
    arrival, how, det_cells = start_fire(d, g, fire, lab, now, t_per)
    print(f"  start fire: {int((arrival >= 0).sum() * (g['res_m'] ** 2) / 4046.856)} ac ({how})", flush=True)
    lab.update(arrival=arrival, times=np.array([now.value, (now + pd.Timedelta(hours=hours)).value], np.int64),
               burn_days=np.array([str(now.date())] * 2), map_methods=np.array([how] * 2),
               sources=np.array(["live"] * 2))
    np.savez_compressed(d / "labels.npz", **lab)
    meta = dict(fire_id=fid, name=fire["name"], state=fire.get("state"), acres=fire.get("acres"), contained=fire.get("contained"),
                perimeter_time=str(t_per), origin=str(now), hours=hours, hrrr=info, viirs_cells_48h=det_cells, start_method=how,
                mode="live")
    (d / "live_meta.json").write_text(json.dumps(meta, indent=1))
    return d


# ------------------------------------------------------------------------------ forecast
def forecast(d, weights=None, mode="physics2", pair=0, hours=None, out=None):
    """Forecast for pair `pair` of fire folder d (live folders have one pair: now -> now + hours)."""
    import torch
    import data as D
    import evaluate as E
    import train_hybrid as TH
    D.WEATHER_MODE = "observed"      # live hrrr.npz carries the forecast as the window's weather (see _hrrr_live)
    D.PHYSICS_V2 = True
    pca = None
    f = D.Fire(d, pca)
    TH.set_grid(f.cell_km)
    gap = float(hours) if hours else (f.times[pair + 1] - f.times[pair]).total_seconds() / 3600
    runs = Path(weights).parent if weights else Path(out or d) / "_none"
    spec = f"{Path(weights).stem}:{mode}" if weights else f"_default:{mode}"
    mb = E.Member(runs, spec, None, TH.groups_used(set()), "cpu", f.res_m, 32)
    win = (0, f.H, 0, f.W)
    feats, phys, y, m, b0 = f.sample(pair, gap, win)
    st = mb.state(feats, phys, b0, gap)
    p = torch.sigmoid(mb.logits(st)).numpy().astype(np.float32)
    p[b0] = np.nan
    res = dict(burning=b0, prob=p, crown=feats["crownspot"][0], ember=phys.get("ember", np.zeros_like(p)),
               rate=phys.get("r_crown", phys["r_head"]), elev=f.z("terrain")["elev"].astype(np.float32))
    if (f.arrival == pair + 1).any():                      # replay: what really burned next
        res["actual_next"] = (f.arrival == pair + 1) & ~f.other
    v = f.z("viirs")
    t_i = f.times[pair]
    dates = pd.to_datetime(v["dates"])
    recent = (dates >= (t_i.tz_convert(None) - pd.Timedelta(days=2)).normalize()) & (dates <= t_i.tz_convert(None).normalize())
    res["viirs_48h"] = ((v["af_day"][recent].astype(int) + v["af_night"][recent]) > 0).any(0) if recent.any() else np.zeros_like(b0)
    meta = dict(fire_id=f.fid, name=f.g["name"], year=f.year, origin=str(t_i), hours=gap, pair=pair,
                transform=f.g["transform"], height=f.H, width=f.W, res_m=f.res_m,
                model=f"{mode}" + (f" ({Path(weights).name})" if weights else " (default physics settings)"),
                weather="HRRR forecast for the window (live) or analyses (replay)")
    if (Path(d) / "live_meta.json").exists():
        meta.update(json.loads((Path(d) / "live_meta.json").read_text()))
    meta["mode"] = "live" if (Path(d) / "live_meta.json").exists() else "replay"
    if "actual_next" in res:
        a = res["actual_next"]
        pp = np.nan_to_num(p)
        k = int(a.sum())
        if k:
            top = np.argsort(-pp, axis=None)[:k]
            hit = a.ravel()[top].mean()
            meta["replay_score"] = dict(actual_new_km2=round(k * (f.res_m / 1000) ** 2, 2), hit_rate_same_area=round(float(hit), 3))
    o = Path(out or d) / "forecasts" / f"{f.fid}_{pair}"
    o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / "forecast.npz", **res)
    (o / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"  forecast written to {o}", flush=True)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "live", "replay"])
    ap.add_argument("--fire", help="live: incident name or WFIGS id (default: most satellite detections)")
    ap.add_argument("--fire-dir", help="replay: a dataset fire folder")
    ap.add_argument("--pair", type=int, default=0, help="replay: forecast from this perimeter")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--weights", help="runs/<name>.pt of a trained model (default: physics2 default settings)")
    ap.add_argument("--mode", default="physics2")
    ap.add_argument("--out", default="data/live")
    a = ap.parse_args()
    if a.cmd == "list":
        for r in active_fires():
            print(f"{r['name'][:28]:28s} {r['state'] or '':6s} {r['acres']:>8,} ac  {r['contained'] or 0:>4}%  "
                  f"updated {r['updated'][:16]}  VIIRS 48 h: {r['detections_48h']}")
        return
    if a.cmd == "live":
        fires = active_fires()
        if a.fire:
            fires = [r for r in fires if a.fire.lower() in (r["name"].lower(), r["id"].lower())]
        if not fires:
            raise SystemExit("no matching active fire")
        fire = sorted(fires, key=lambda r: (-r["detections_48h"], -r["acres"]))[0]
        print(f"live forecast for {fire['name']} ({fire['acres']:,} acres, {fire['detections_48h']} detections in 48 h)")
        d = build_live(fire, a.out, a.hours)
        forecast(d, a.weights, a.mode, 0, a.hours, a.out)
    else:
        forecast(a.fire_dir, a.weights, a.mode, a.pair, None, a.out)


if __name__ == "__main__":
    main()
