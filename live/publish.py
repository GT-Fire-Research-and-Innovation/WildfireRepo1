"""
Publish live spread forecasts for the map app (site/). Run on a schedule (GitHub Actions).

  python live/publish.py --site site --work data/live_run --max-fires 12 --hours 24

1. Active fires from the NIFC WFIGS current perimeters (100+ acres, mapped in the last 60 days).
2. The most active ones (most VIIRS detections in 48 h) get a forecast: the fire's layers are
   built from live sources (DEM, LANDFIRE, GRIDMET, FIRMS VIIRS, latest HRRR forecast) with the
   training pipeline's own code, then the model forecasts the next --hours.
3. Model: the router in live/models/router.json (calibrated U-Net + hybrids, weighted by gap and
   wind, as fitted by model/combine.py) when its weights are there; otherwise the physics-only
   model with its default settings, so the app always has a forecast.
4. Writes site/data/index.json (run summary + one entry per fire) and
   site/data/forecast/<fire>.json: the forecast on the model's own 150 m cells (chance of burning per
   cell, the fire at forecast start, crown fire and ember levels), one byte per cell.
"""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "pipeline", ROOT / "model", ROOT / "live"):
    sys.path.insert(0, str(p))

BANDS = [0.1, 0.3, 0.5, 0.7]                 # forecast bands (probability of burning within the horizon)
ACRE_M2 = 4046.856
MPS_MPH = 2.23694
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


# ------------------------------------------------------------------------------ model
class Model:
    """The router (or a fallback) applied to one fire's inputs."""

    def __init__(self, models_dir):
        import evaluate as E
        import train_hybrid as TH
        import data as D
        D.WEATHER_MODE = "forecast"
        D.PHYSICS_V2 = True
        self.E, self.TH, self.D = E, TH, D
        md = Path(models_dir)
        cfg = md / "router.json"
        self.router = json.loads(cfg.read_text()) if cfg.exists() else None
        if self.router and not all((md / m["file"]).exists() for m in self.router["members"]):
            print("  !! router.json found but some weights are missing: using the physics-only model", flush=True)
            self.router = None
        if self.router:
            try:
                self.members = [E.Member(md, f"{m['name']}:{m['mode']}", None, TH.groups_used(set()), "cpu", 150.0, 32)
                                for m in self.router["members"]]
            except Exception as ex:                # a bad or mismatched weights file must not stop the forecasts
                print(f"  !! could not load the router models ({type(ex).__name__}: {ex}); using the physics-only model", flush=True)
                self.router = None
                D.TEMPORAL = False
        if self.router:
            names = ", ".join(m["name"] for m in self.router["members"])
            self.label = dict(kind="router", name=f"Router of {len(self.router['members'])} models ({names})",
                              note="Calibrated U-Nets (with 6-day fire history) and physics-guided hybrids, weighted by forecast length and wind.",
                              test_ap=(self.router.get("test_scores") or {}).get("router"))
        else:
            self.members = [E.Member(md / "_none", "_default:physics2", None, TH.groups_used(set()), "cpu", 150.0, 32)]
            self.label = dict(kind="physics", name="Physics only (Rothermel + crown fire + embers)",
                              note="Fallback until trained weights are published", test_ap=0.093)

    def run(self, d, hours):
        import torch
        E, TH, D = self.E, self.TH, self.D
        f = D.Fire(d, None)
        TH.set_grid(f.cell_km)
        feats, phys, y, m, b0 = f.sample(0, float(hours), (0, f.H, 0, f.W))
        logits = []
        for mb in self.members:
            st = mb.state(feats, phys, b0, float(hours))
            logits.append(mb.logits(st).detach().cpu().numpy().astype(np.float64))
        if self.router:
            r = self.router
            hr = feats["hrrr"]
            spd = hr[2] * 8 if hr[6].max() > 0 else feats["weather"][6] * 2 + 4
            wind = np.digitize(spd, E.WIND_EDGES).ravel()
            gap = int(np.digitize(hours, E.GAP_EDGES))
            Z = np.stack([np.clip(r["temps"][mm["eval"]][0] * lg.ravel() + r["temps"][mm["eval"]][1], -12, 12)
                          for mm, lg in zip(r["members"], logits)], 1)
            coef, nm = np.array(r["coef"]), Z.shape[1]
            s = Z @ coef[:nm] + r["intercept"]
            for j in range(nm):
                base = nm + 8 * j
                s += Z[:, j] * coef[base + gap] + Z[:, j] * coef[base + 4 + wind]
            logit = s.reshape(b0.shape)
        else:
            logit = logits[0]
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        p[b0] = 0.0
        return f, feats, phys, b0, p


# ------------------------------------------------------------------------------ vector layers
def _despeckle(mask, min_cells):
    """Drop blobs and fill holes smaller than min_cells (keeps the map readable and the files small)."""
    if min_cells <= 1:
        return mask
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = np.isin(lab, 1 + np.nonzero(sizes >= min_cells)[0])
    lab, n = ndi.label(~mask)
    if n:
        sizes = ndi.sum(~mask, lab, range(1, n + 1))
        mask = mask | np.isin(lab, 1 + np.nonzero(sizes < min_cells)[0])
    return mask


def polygons(mask, g, tol_cells=0.6, min_cells=1):
    """Boolean grid -> one (multi)polygon in lon/lat, simplified to about tol_cells cells."""
    from affine import Affine
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    if not mask.any():
        return None
    mask = _despeckle(mask, min_cells)
    if not mask.any():
        return None
    geoms = [shape(s) for s, v in shapes(mask.astype(np.uint8), mask=mask, transform=Affine(*g["transform"])) if v == 1]
    u = unary_union(geoms).simplify(abs(g["transform"][0]) * tol_cells, preserve_topology=True)
    return _round(mapping(u)) if not u.is_empty else None


def _round(geo, nd=5):
    def r(c):
        return [round(c[0], nd), round(c[1], nd)] if isinstance(c[0], (int, float)) else [r(x) for x in c]
    return dict(type=geo["type"], coordinates=r(list(geo["coordinates"])) if geo["type"] != "Point" else geo["coordinates"])


def vector_layers(g, b0, p, crown, ember, actual=None):
    """Map layers for one fire as GeoJSON features (lon/lat). Crown and ember zones are drawn
    coarser (they cover the whole grid and only need to show where, not the exact cell)."""
    out = []

    def add(mask, tol=0.6, min_cells=3, **props):
        geom = polygons(mask, g, tol, min_cells)
        if geom:
            out.append(dict(type="Feature", geometry=geom, properties=props))
    add(b0, 0.6, 1, layer="start", label="Fire at forecast start (perimeter + satellite detections, 48 h)")
    for lo in BANDS:
        add((p >= lo) & ~b0, layer="forecast", min_prob=lo, label=f"{int(lo * 100)}%+ chance of burning")
    crown = np.nan_to_num(crown)
    add(crown >= 0.5, 2.0, 25, layer="crown", level=2, label="Crown fire likely if reached")
    add((crown >= 0.2) & (crown < 0.5), 2.0, 25, layer="crown", level=1, label="Crown fire possible if reached")
    e = np.nan_to_num(ember)
    if e.max() > 0:
        rel = e / e.max()
        for lv, (lo, name) in enumerate([(1e-3, "Low"), (1e-2, "Moderate"), (1e-1, "High")], 1):
            add((rel >= lo) & ~b0, 1.5, 9, layer="ember", level=lv, label=f"{name} ember landing (relative)")
    if actual is not None:
        add(actual, layer="actual", label="What actually burned by the next perimeter")
    return out


def cell_grid(g, b0, p, crown, ember, actual=None, res_m=150.0, pad=12):
    """The forecast as the model's own grid cells, for the map: one byte per cell, base64.
    prob = chance of burning in percent (0-100), start = burning/burned at forecast start (0/1),
    crown = crown-fire probability in percent, ember = ember landing level 0-3, actual = replay only.
    Cropped to the fire and every cell with at least a 1% chance (plus a margin)."""
    import base64
    p = np.nan_to_num(p)
    keep = b0 | (p >= 0.01) | (actual if actual is not None else False)
    if not keep.any():
        keep = np.ones_like(b0)
    rr, cc = np.nonzero(keep)
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, b0.shape[0])
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, b0.shape[1])
    sl = (slice(r0, r1), slice(c0, c1))
    dx, _, x0, _, ndy, y0 = g["transform"]
    enc = lambda a: base64.b64encode(np.ascontiguousarray(a[sl], dtype=np.uint8).tobytes()).decode()  # noqa: E731
    pct = np.clip(np.round(p * 100), 0, 100)
    pct[(p >= 0.005) & (pct == 0)] = 1
    pct[b0] = 0
    e = np.nan_to_num(ember)
    lev = np.zeros(e.shape, np.uint8)
    if e.max() > 0:
        rel = e / e.max()
        lev = (rel >= 1e-3).astype(np.uint8) + (rel >= 1e-2) + (rel >= 1e-1)
        lev[b0] = 0
    out = dict(grid=dict(west=x0 + c0 * dx, north=y0 + r0 * ndy, dx=dx, dy=-ndy, width=int(c1 - c0), height=int(r1 - r0),
                         cell_m=res_m, order="rows from the north-west corner, one byte per cell, base64"),
               prob=enc(pct), start=enc(b0), crown=enc(np.clip(np.round(np.nan_to_num(crown) * 100), 0, 100)), ember=enc(lev))
    if actual is not None:
        out["actual"] = enc(actual)
    return out


def layers(f, feats, phys, b0, p):
    return vector_layers(f.g, b0, p, feats["crownspot"][0], phys.get("ember", np.zeros_like(p)))


def summary(f, feats, b0, p):
    cell_ac = (f.res_m ** 2) / ACRE_M2
    out = dict(expected_new_acres=int(round(float(p[~b0].sum()) * cell_ac)),
               likely_acres=int(round(float(((p >= 0.5) & ~b0).sum()) * cell_ac)),
               possible_acres=int(round(float(((p >= 0.1) & ~b0).sum()) * cell_ac)),
               start_acres=int(round(float(b0.sum()) * cell_ac)))
    hr = feats["hrrr"]
    if hr[6].max() > 0:
        near = b0 if b0.any() else np.ones_like(b0)
        from scipy.ndimage import binary_dilation
        near = binary_dilation(near, iterations=20)
        u, v = float(hr[0][near].mean() * 8), float(hr[1][near].mean() * 8)
        frm = (math.degrees(math.atan2(-u, -v)) + 360) % 360
        out["wind"] = dict(mean_mph=round(float(hr[2][near].mean() * 8 * MPS_MPH)),
                           peak_gust_mph=round(float(hr[4][near].max() * 20 * MPS_MPH)),
                           from_deg=round(frm), from_compass=COMPASS[int((frm + 11.25) // 22.5) % 16],
                           min_rh_pct=round(float(hr[5][near].min() * 100)))
    return out


# ------------------------------------------------------------------------------ which fires
INCIDENTS = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
             "WFIGS_Incident_Locations_Current/FeatureServer/0/query")


def all_fires(FN, now):
    """Every current wildfire worth considering: WFIGS perimeters of any size, plus WFIGS incidents
    that have no perimeter yet (most new fires, and most fires in the East). A fire without a
    perimeter gets its starting shape from its VIIRS detections of the last 48 h (375 m pixels)
    near the incident point, plus a circle of its reported size."""
    import requests
    from shapely import affinity
    from shapely.geometry import Point, box, mapping
    from shapely.ops import unary_union
    from live import firms
    fires = FN.active_fires(min_acres=0)
    for r in fires:
        r["source"] = "perimeter"
    have = {r["id"] for r in fires}
    det = pd.concat([firms.get("VIIRS_NOAA20_NRT", (-125, 24, -66, 50), 1, (now - pd.Timedelta(days=k)).date().isoformat())
                     for k in (1, 0)], ignore_index=True)
    since = (now - pd.Timedelta(days=21)).strftime("%Y-%m-%d %H:%M:%S")
    js = requests.get(INCIDENTS, params=dict(
        where=f"IncidentTypeCategory='WF' AND (PercentContained IS NULL OR PercentContained<100) AND ModifiedOnDateTime_dt >= TIMESTAMP '{since}'",
        outFields="IncidentName,IncidentSize,PercentContained,POOState,ModifiedOnDateTime_dt,UniqueFireIdentifier",
        outSR=4326, f="json", resultRecordCount=2000), timeout=120).json()
    added = 0
    for f in js.get("features", []):
        a, g = f["attributes"], f.get("geometry") or {}
        if "x" not in g or a["UniqueFireIdentifier"] in have:
            continue
        acres = float(a["IncidentSize"] or 0)
        rad_km = max(3.0, 1.5 * math.sqrt(acres * ACRE_M2 / 1e6 / math.pi) + 1.5)
        k = math.cos(math.radians(g["y"]))
        near = det[((det.latitude - g["y"]).abs() < rad_km / 111) & ((det.longitude - g["x"]).abs() < rad_km / (111 * k))] if len(det) else det
        if not len(near):
            continue
        hy, hx = 0.0017, 0.0017 / k                       # half a 375 m VIIRS pixel
        parts = [box(x - hx, y - hy, x + hx, y + hy) for x, y in zip(near.longitude, near.latitude)]
        r_deg = max(math.sqrt(acres * ACRE_M2 / math.pi), 100) / 111000
        parts.append(affinity.scale(Point(g["x"], g["y"]).buffer(r_deg, 16), xfact=1 / k, yfact=1))
        geom = unary_union(parts)
        t = pd.to_datetime(a["ModifiedOnDateTime_dt"], unit="ms", utc=True) if a["ModifiedOnDateTime_dt"] else now
        fires.append(dict(id=a["UniqueFireIdentifier"], name=a["IncidentName"], state=a["POOState"], acres=round(acres),
                          contained=a["PercentContained"], updated=min(t, now).isoformat(), lon=round(g["x"], 4), lat=round(g["y"], 4),
                          detections_48h=int(len(near)), source="incident",
                          feature=dict(type="Feature", geometry=mapping(geom),
                                       properties=dict(poly_MapMethod="VIIRS detections + incident location (no perimeter yet)"))))
        added += 1
    print(f"{len(fires) - added} perimeters + {added} incidents without a perimeter but with satellite detections", flush=True)
    return fires


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site")
    ap.add_argument("--work", default="data/live_run")
    ap.add_argument("--models", default=str(ROOT / "live" / "models"))
    ap.add_argument("--max-fires", type=int, default=12)
    ap.add_argument("--min-detections", type=int, default=3, help="VIIRS detections in 48 h to count as active")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--fire", help="forecast only this incident (name or WFIGS id)")
    a = ap.parse_args()
    import forecast_now as FN
    t0 = time.time()
    site = Path(a.site)
    (site / "data" / "forecast").mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    fires = all_fires(FN, now)
    pick = [r for r in fires if r["detections_48h"] >= a.min_detections]
    if a.fire:
        pick = [r for r in fires if a.fire.lower() in (r["name"].lower(), r["id"].lower())]
    pick = sorted(pick, key=lambda r: (-r["detections_48h"], -r["acres"]))[: a.max_fires]
    model = Model(a.models)
    print(f"model: {model.label['name']}; forecasting {len(pick)} fires: {', '.join(r['name'] for r in pick)}", flush=True)
    entries = []
    for r in fires:
        e = {k: r.get(k) for k in ("id", "name", "state", "acres", "contained", "updated", "lon", "lat", "detections_48h", "source")}
        e["forecast"] = None
        entries.append(e)
    by_id = {e["id"]: e for e in entries}
    for r in pick:
        e = by_id[r["id"]]
        t1 = time.time()
        print(f"\n== {r['name']} ({r['acres']:,} ac, {r['detections_48h']} detections in 48 h)", flush=True)
        try:
            d = FN.build_live(r, a.work, int(a.hours), now)
            f, feats, phys, b0, p = model.run(d, a.hours)
            cells = cell_grid(f.g, b0, p, feats["crownspot"][0], phys.get("ember", np.zeros_like(p)), res_m=f.res_m)
            name = r["id"].replace("/", "_") + ".json"
            (site / "data" / "forecast" / name).write_text(json.dumps(cells, separators=(",", ":")))
            meta = json.loads((Path(d) / "live_meta.json").read_text())
            e.update(forecast=f"data/forecast/{name}", stats=summary(f, feats, b0, p),
                     origin=meta["origin"], hrrr_run=(meta.get("hrrr") or {}).get("forecast_run"),
                     perimeter_time=meta["perimeter_time"], seconds=round(time.time() - t1))
            print(f"  done in {time.time() - t1:.0f}s: {e['stats']}", flush=True)
        except Exception as ex:                      # one bad fire never stops the run
            traceback.print_exc()
            e["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    index = dict(generated=now.isoformat(), horizon_hours=a.hours, model=model.label,
                 forecast_count=sum(1 for e in entries if e["forecast"]), fire_count=len(entries),
                 runtime_s=round(time.time() - t0), fires=entries)
    (site / "data" / "index.json").write_text(json.dumps(index, indent=1, default=str))
    print(f"\nwrote {site / 'data' / 'index.json'}: {index['forecast_count']} forecasts, {len(entries)} fires, "
          f"{index['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
