"""
Turn forecasts from live/forecast_now.py into one self-contained web page (map layers embedded).

  python live/render_page.py --forecasts OUT/forecasts/A OUT/forecasts/B --fires OUT/active_fires.json --page out.html
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

UP = 2          # image pixels per grid cell
TEMPLATE = Path(__file__).with_name("page_template.html")


def _png(rgba):
    im = Image.fromarray(rgba, "RGBA")
    im = im.resize((rgba.shape[1] * UP, rgba.shape[0] * UP), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jpg(rgb):
    im = Image.fromarray(rgb, "RGB").resize((rgb.shape[1] * UP, rgb.shape[0] * UP), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _ramp(v, stops):
    """v in [0,1] -> RGB from a list of (pos, (r,g,b))."""
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], float)
    out = np.stack([np.interp(v, pos, cols[:, k]) for k in range(3)], -1)
    return out.astype(np.uint8)


def _edge(mask):
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask, iterations=1, border_value=0)


def hillshade(elev):
    gy, gx = np.gradient(np.nan_to_num(elev.astype(float), nan=float(np.nanmean(elev))), 150)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy) * 1.6)
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    hs = np.clip(np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect), 0, 1)
    g = (110 + 130 * hs).astype(np.uint8)
    return np.stack([g, g, (g * 1.02).clip(0, 255).astype(np.uint8)], -1)


def layers(z, meta):
    H, W = z["burning"].shape
    out = {}
    out["terrain"] = _jpg(hillshade(z["elev"]))
    burn = z["burning"].astype(bool)
    a = np.zeros((H, W, 4), np.uint8)
    a[burn] = (60, 24, 18, 150)
    a[_edge(burn)] = (20, 8, 6, 255)
    out["burning"] = _png(a)
    p = np.nan_to_num(z["prob"].astype(float))
    a = np.zeros((H, W, 4), np.uint8)
    show = p >= 0.10
    a[..., :3] = _ramp(np.clip(p, 0, 1), [(0, (255, 236, 140)), (0.3, (253, 174, 60)), (0.6, (232, 89, 34)), (1, (150, 20, 30))])
    a[..., 3] = np.where(show, (70 + 170 * np.clip(p, 0, 1) ** 0.6), 0).astype(np.uint8)
    out["forecast"] = _png(a)
    c = np.nan_to_num(z["crown"].astype(float))
    a = np.zeros((H, W, 4), np.uint8)
    a[..., :3] = _ramp(np.clip(c, 0, 1), [(0, (200, 230, 160)), (1, (24, 110, 60))])
    a[..., 3] = np.where(c >= 0.1, 40 + 160 * c, 0).astype(np.uint8)
    out["crown"] = _png(a)
    e = np.nan_to_num(z["ember"].astype(float))
    le = np.clip((np.log10(e + 1e-6) + 4) / 4, 0, 1)
    a = np.zeros((H, W, 4), np.uint8)
    a[..., :3] = _ramp(le, [(0, (196, 181, 253)), (1, (88, 28, 135))])
    a[..., 3] = np.where(e > 1e-4, 50 + 170 * le, 0).astype(np.uint8)
    out["ember"] = _png(a)
    v = z["viirs_48h"].astype(bool) & ~burn
    a = np.zeros((H, W, 4), np.uint8)
    a[z["viirs_48h"].astype(bool)] = (214, 0, 120, 235)
    out["viirs"] = _png(a)
    if "actual_next" in z:
        a = np.zeros((H, W, 4), np.uint8)
        nxt = z["actual_next"].astype(bool) | burn
        a[_edge(nxt)] = (0, 200, 230, 255)
        out["actual"] = _png(a)
    km2 = (meta["res_m"] / 1000) ** 2
    stats = dict(expected_new_km2=round(float(p[~burn].sum() * km2), 1),
                 likely_km2=round(float((p[~burn] >= 0.5).sum() * km2), 1),
                 burning_km2=round(float(burn.sum() * km2), 1),
                 crown_km2=round(float((c >= 0.5).sum() * km2), 1),
                 viirs_cells=int(v.sum()), width_km=round(W * meta["res_m"] / 1000, 1),
                 height_km=round(H * meta["res_m"] / 1000, 1), px_w=W * UP, px_h=H * UP)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", nargs="+", required=True)
    ap.add_argument("--fires", help="active_fires.json from forecast_now.py")
    ap.add_argument("--page", required=True)
    a = ap.parse_args()
    items = []
    for d in a.forecasts:
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text())
        z = dict(np.load(d / "forecast.npz"))
        imgs, stats = layers(z, meta)
        items.append(dict(meta=meta, stats=stats, layers=imgs))
    fires = json.loads(Path(a.fires).read_text()) if a.fires else []
    for f in fires:
        f.pop("feature", None)
    data = json.dumps(dict(items=items, fires=fires))
    page = TEMPLATE.read_text().replace("/*__DATA__*/null", data)
    Path(a.page).write_text(page)
    print(f"wrote {a.page} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
