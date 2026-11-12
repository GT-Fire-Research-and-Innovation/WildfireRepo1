"""What the fuel-moisture build stages actually wrote, for a few fires or all of them.
  python pipeline/check_fuel_inputs.py --data ~/scratch/wildfire/my [--fires ID ID ...]"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--fires", nargs="*")
a = ap.parse_args()
root = Path(a.data).expanduser() / "fires"
ids = a.fires or sorted(p.parent.name for p in root.glob("*/hrrr2.npz"))
print(f"{len(ids)} fires")
for fid in ids[:40]:
    d = root / fid
    year = json.loads((d / "grid.json").read_text())["year"]
    line = [f"{fid} {year}"]
    if (d / "hrrr2.npz").exists():
        z = np.load(d / "hrrr2.npz")
        n = len(z["times"])
        have = {k: float(np.isfinite(z[k].reshape(n, -1)).any(1).mean()) for k in ("t2", "dpt", "dswrf", "apcp3")}
        t2 = z["t2"].astype(np.float32)
        line.append(f"weather: {n} steps, share with data " + " ".join(f"{k} {v:.0%}" for k, v in have.items())
                    + f", temp {np.nanmin(t2):.0f}..{np.nanmax(t2):.0f} C, rain max {np.nanmax(z['apcp3'].astype(np.float32)):.1f} mm")
    else:
        line.append("weather: none")
    if (d / "green.npz").exists():
        z = np.load(d / "green.npz")
        if bool(z["missing"]):
            line.append("greenness: EMPTY")
        else:
            nd = z["ndvi"].astype(np.float32) * 1e-4
            line.append(f"greenness: {len(z['ends'])} composites, NDVI {np.nanmedian(nd[nd > -0.3]):.2f} median, "
                        f"{(z['ndvi'] > -3000).mean():.0%} of cells valid")
    else:
        err = d / "green.error.txt"
        line.append("greenness: " + ("ERROR " + err.read_text().strip().splitlines()[-1][:90] if err.exists() else "none"))
    old = str(np.load(d / "landfire.npz")["version"]) if (d / "landfire.npz").exists() else "none"
    if (d / "landfire2.npz").exists():
        z = np.load(d / "landfire2.npz")
        line.append(f"fuel map: {old} -> {z['version']}" + (" (EMPTY)" if bool(z["missing"]) else ""))
    else:
        err = d / "landfire2.error.txt"
        line.append(f"fuel map: {old} -> " + ("ERROR " + err.read_text().strip().splitlines()[-1][:90] if err.exists() else "none"))
    print(" | ".join(line))
