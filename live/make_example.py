"""
Turn a replay forecast (live/forecast_now.py replay) into the map app's example layer, shown when
no fire is actively burning: site/data/example/{index.json, example.geojson}, including what
really burned next so viewers can judge the forecast.

  python live/make_example.py --forecast data/live/forecasts/Ff1ba680f61_22 --site site
"""
import argparse
import json
from pathlib import Path

import numpy as np

import publish as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", required=True)
    ap.add_argument("--site", default="site")
    a = ap.parse_args()
    d = Path(a.forecast)
    z, meta = dict(np.load(d / "forecast.npz")), json.loads((d / "meta.json").read_text())
    g = dict(transform=meta["transform"], height=meta["height"], width=meta["width"])
    b0 = z["burning"].astype(bool)
    p = np.nan_to_num(z["prob"])
    actual = (z["actual_next"].astype(bool) | b0) if "actual_next" in z else None
    feats = P.vector_layers(g, b0, p, z["crown"], z["ember"], actual)
    cell_ac = meta["res_m"] ** 2 / P.ACRE_M2
    stats = dict(expected_new_acres=int(p[~b0].sum() * cell_ac), likely_acres=int(((p >= 0.5) & ~b0).sum() * cell_ac),
                 possible_acres=int(((p >= 0.1) & ~b0).sum() * cell_ac), start_acres=int(b0.sum() * cell_ac))
    if "actual_next" in z:
        stats["actual_new_acres"] = int(z["actual_next"].sum() * cell_ac)
    out = Path(a.site) / "data" / "example"
    out.mkdir(parents=True, exist_ok=True)
    (out / "example.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=feats), separators=(",", ":")))
    sc = meta.get("replay_score", {})
    note = (f"Replay of a past forecast, shown because no fire is actively burning. Forecast made {meta['origin'][:10]} "
            f"for the next {meta['hours']:.0f} h with {meta['model']}; the cyan line is what really burned"
            + (f" ({sc['hit_rate_same_area']:.0%} of it was in the forecast's top area)." if sc else "."))
    idx = dict(id="__example", name=f"{meta['name']} ({meta['year']})", state="", acres=stats["start_acres"], example=True,
               file="example.geojson", origin=meta["origin"], hours=round(meta["hours"]), stats=stats, note=note)
    (out / "index.json").write_text(json.dumps(idx, indent=1))
    print(f"wrote {out} ({len(feats)} features, {(out / 'example.geojson').stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
