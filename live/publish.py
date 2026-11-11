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
   site/data/forecast/<fire>.geojson (forecast bands, crown fire, ember landing, starting fire),
   all in longitude/latitude.
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
            self.members = [E.Member(md, f"{m['name']}:{m['mode']}", None, TH.groups_used(set()), "cpu", 150.0, 32)
                            for m in self.router["members"]]
            names = ", ".join(m["name"] for m in self.router["members"])
            self.label = dict(kind="router", name=f"Router ({names})",
                              note="Calibrated U-Net + physics-guided hybrids, weighted by forecast length and wind",
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


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site")
    ap.add_argument("--work", default="data/live_run")
    ap.add_argument("--models", default=str(ROOT / "live" / "models"))
    ap.add_argument("--max-fires", type=int, default=12)
    ap.add_argument("--min-detections", type=int, default=5, help="VIIRS detections in 48 h to count as active")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--fire", help="forecast only this incident (name or WFIGS id)")
    a = ap.parse_args()
    import forecast_now as FN
    t0 = time.time()
    site = Path(a.site)
    (site / "data" / "forecast").mkdir(parents=True, exist_ok=True)
    fires = FN.active_fires()
    print(f"{len(fires)} active fires (100+ acres, mapped in the last 60 days)", flush=True)
    pick = [r for r in fires if r["detections_48h"] >= a.min_detections]
    if a.fire:
        pick = [r for r in fires if a.fire.lower() in (r["name"].lower(), r["id"].lower())]
    pick = sorted(pick, key=lambda r: (-r["detections_48h"], -r["acres"]))[: a.max_fires]
    model = Model(a.models)
    print(f"model: {model.label['name']}; forecasting {len(pick)} fires", flush=True)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    entries = []
    for r in fires:
        e = {k: r[k] for k in ("id", "name", "state", "acres", "contained", "updated", "lon", "lat", "detections_48h")}
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
            fc = dict(type="FeatureCollection", features=layers(f, feats, phys, b0, p))
            name = r["id"].replace("/", "_") + ".geojson"
            (site / "data" / "forecast" / name).write_text(json.dumps(fc, separators=(",", ":")))
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
