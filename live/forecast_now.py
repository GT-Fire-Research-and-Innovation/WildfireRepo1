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


# ------------------------------------------------------------------------------ active fires
def active_fires(min_acres=100, max_age_days=60):
    """Current WFIGS perimeters with their VIIRS detections of the last 48 h."""
    import requests
    from shapely.geometry import shape
    from live import firms
    r = requests.get(WFIGS, params=dict(
        where=f"poly_GISAcres>={min_acres}", returnGeometry="true", outSR=4326, f="geojson", resultRecordCount=2000,
        outFields="attr_UniqueFireIdentifier,attr_IncidentName,poly_PolygonDateTime,poly_GISAcres,"
                  "attr_PercentContained,attr_POOState,poly_MapMethod"), timeout=120).json()
    now = pd.Timestamp.now(tz="UTC")
    det = pd.concat([firms.get("VIIRS_NOAA20_NRT", (-125, 24, -66, 50), 1, (now - pd.Timedelta(days=k)).date().isoformat())
                     for k in (1, 0)], ignore_index=True)
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
                         detections_48h=int(len(near)), feature=f))
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
    np.savez_compressed(d / "hrrr.npz", times=np.array([t.value for t in ana_t], np.int64),
                        fields=np.array(ana, np.float16).reshape(-1, *shape4[1:]) if ana else np.zeros(shape4, np.float16),
                        fc_times=np.array([t.value for t in fc_t], np.int64),
                        fc=np.array(fc, np.float16).reshape(-1, *shape4[1:]) if fc else np.zeros(shape4, np.float16),
                        fc_issued=np.array([issued.value] * len(fc_t), np.int64) if fc else np.zeros(0, np.int64),
                        vars=np.array([n for n, _ in WF.HRRR_FIELDS]), idx=idx.reshape(g["height"], g["width"]).astype(np.int32))
    return dict(analyses=len(ana_t), forecast_steps=len(fc_t), forecast_run=issued.isoformat() if issued else None)


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
    print(f"  stages: {done} ({time.time() - t0:.0f}s)", flush=True)
    if not (d / "landfire.npz").exists():                  # newest LANDFIRE version may not exist yet: one older
        g2 = dict(g, year=g["year"] - 1)
        BM.build_one(out, cat, g2, ["landfire"])
    info = _hrrr_live(d, g, now, hours)
    print(f"  HRRR: {info}", flush=True)
    # the current fire = latest perimeter + VIIRS detections of the last 48 h (dilated to the 375 m footprint)
    lab = dict(np.load(d / "labels.npz"))
    v = np.load(d / "viirs.npz")
    dates = pd.to_datetime(v["dates"])
    recent = dates >= (now.tz_convert(None) - pd.Timedelta(days=2)).normalize()
    det = ((v["af_day"][recent].astype(int) + v["af_night"][recent]) > 0).any(0) if recent.any() else np.zeros(lab["arrival"].shape, bool)
    arrival = lab["arrival"].copy()
    arrival[det & (arrival < 0) & ~lab["other"]] = 0
    lab.update(arrival=arrival, times=np.array([now.value, (now + pd.Timedelta(hours=hours)).value], np.int64),
               burn_days=np.array([str(now.date())] * 2), map_methods=np.array(["perimeter + VIIRS 48 h"] * 2),
               sources=np.array(["live"] * 2))
    np.savez_compressed(d / "labels.npz", **lab)
    meta = dict(fire_id=fid, name=fire["name"], state=fire.get("state"), acres=fire.get("acres"), contained=fire.get("contained"),
                perimeter_time=str(t_per), origin=str(now), hours=hours, hrrr=info, viirs_cells_48h=int(det.sum()),
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
    D.WEATHER_MODE = "forecast"
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
                weather="forecast-only (HRRR analyses up to the origin + forecasts issued before it)")
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
