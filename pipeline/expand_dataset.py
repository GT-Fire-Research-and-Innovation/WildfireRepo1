"""
Start a second dataset folder for a rebuilt catalog without rebuilding what has not changed.

  python pipeline/expand_dataset.py --old ~/scratch/wildfire/my --new ~/scratch/wildfire/my2 \
         --catalog ~/scratch/wildfire/catalog2

A fire that is in both catalogs, whose new perimeters still fit inside the old grid's box and inside the dates
the old layers cover, gets its layer files hard-linked into the new folder (no extra disk, nothing downloaded again).
Only labels.npz is left out, because the perimeters changed (same-day perimeters, auto-generated circles
removed, pieces joined); the build then writes it from the new catalog. Fires that are new, or whose box or
dates changed, start empty and are built in full.

Also shared with the old folder: the download cache (HRRR and NIFC files are per date, not per fire) and
the land-cover PCA in runs/, so a model trained on one folder can be scored on the other with identical
inputs. catalog.txt in the new folder tells training and scoring which catalog's quality ratings and
duplicate list belong to it.

Nothing in the old folder is changed.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

SLACK_H = 12.0        # new perimeters may start / end this much outside the old span (the layers are padded by a day or more)
SKIP = {"labels.npz"}


def covers(g_old, row, pad):
    """Does the old grid still hold the fire with its full margin? True when the fire's new box lies inside the
    old one (the old grid minus its margin of `pad` cells), to within a cell."""
    dx, _, x0, _, dy, y0 = g_old["transform"]
    w, n = x0 + pad * dx, y0 + pad * dy
    e, s = x0 + (g_old["width"] - pad) * dx, y0 + (g_old["height"] - pad) * dy
    return row.west >= w - dx and row.east <= e + dx and row.north <= n - dy and row.south >= s + dy


def link_fire(old_dir, new_dir, row, pad):
    """Returns 'linked', 'new' or the reason the fire is rebuilt."""
    gp = old_dir / "grid.json"
    if not gp.exists():
        return "new"
    g_old = json.loads(gp.read_text())
    if not covers(g_old, row, pad):
        return "box grew"
    t0, t1 = pd.Timestamp(g_old["first"]), pd.Timestamp(g_old["last"])
    n0, n1 = pd.Timestamp(row["first"]), pd.Timestamp(row["last"])
    t0, t1, n0, n1 = (x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC") for x in (t0, t1, n0, n1))
    slack = pd.Timedelta(hours=SLACK_H)
    if n0 < t0 - slack or n1 > t1 + slack:
        return "dates changed"
    new_dir.mkdir(parents=True, exist_ok=True)
    if not (new_dir / "grid.json").exists():
        shutil.copy2(gp, new_dir / "grid.json")              # the old grid: its dates are the ones the layers cover
    for f in old_dir.iterdir():
        if f.name in SKIP or f.suffix not in (".npz", ".gz") or ".tmp" in f.name or f.name.startswith("labels"):
            continue
        dst = new_dir / f.name
        if not dst.exists():
            try:
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)
    return "linked"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="dataset folder already built")
    ap.add_argument("--new", required=True, help="dataset folder to start")
    ap.add_argument("--catalog", required=True, help="the rebuilt catalog (fires.csv)")
    a = ap.parse_args()
    old, new, cat = (Path(x).expanduser().resolve() for x in (a.old, a.new, a.catalog))
    if old == new:
        raise SystemExit("--old and --new are the same folder")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_multiyear as BM
    fires = pd.read_csv(cat / "fires.csv")
    (new / "fires").mkdir(parents=True, exist_ok=True)
    n = {}
    for _, row in fires.iterrows():
        r = link_fire(old / "fires" / row.fire_id, new / "fires" / row.fire_id, row, BM.PAD)
        n[r] = n.get(r, 0) + 1
    if (old / "cache").exists() and not (new / "cache").exists():
        os.symlink(old / "cache", new / "cache")
    (new / "runs").mkdir(exist_ok=True)
    pca = old / "runs" / "tessera_pca.npz"
    if pca.exists() and not (new / "runs" / "tessera_pca.npz").exists():
        shutil.copy2(pca, new / "runs" / "tessera_pca.npz")
    (new / "catalog.txt").write_text(str(cat) + "\n")
    gone = len(set(p.name for p in (old / "fires").iterdir()) - set(fires.fire_id))
    print(f"{len(fires)} fires in the new catalog: " + ", ".join(f"{v} {k}" for k, v in sorted(n.items())))
    print(f"{gone} fires of the old folder are not in the new catalog (left where they are)")
    print(f"next: build_multiyear.py --catalog {cat} --out {new} (only what is missing is built)")


if __name__ == "__main__":
    main()
