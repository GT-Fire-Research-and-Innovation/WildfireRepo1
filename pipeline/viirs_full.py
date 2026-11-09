"""All VIIRS bands per label date (the training set so far keeps only af, I4, M11, night I4).
Day file bands: b1-b5 = I1-I5, m11, af(+); night: b1, b2 (I4, I5 at night).
Output OUT_DIR/viirs_full/<fid>.npz: bands (D, 8, h, w) float16 on the region grid, names.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import build_dataset as B

NAMES = ["i1", "i2", "i3", "i4", "i5", "m11", "i4n", "i5n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "viirs_full").mkdir(exist_ok=True)
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    ids = [p.stem for p in sorted((out / "train").glob("*.npz"))]
    for n, fid in enumerate(ids):
        dst = out / "viirs_full" / f"{fid}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        z = np.load(out / "train" / f"{fid}.npz")
        dates = [str(d) for d in z["dates"]]
        r0, r1, c0, c1 = map(int, z["region"])
        lon, lat = B.grid_centres(list(z["transform150"]), r0, r1, c0, c1)
        day, night = B.tssat_files(fs, fid, "VIIRS_Day"), B.tssat_files(fs, fid, "VIIRS_Night")
        want = [day[d] for d in dates if d in day] + [night[d] for d in dates if d in night]
        with ThreadPoolExecutor(8) as ex:
            got = dict(zip(want, ex.map(lambda p: B.read_tif(fs, p), want)))
        out_arr = np.zeros((len(dates), 8, r1 - r0, c1 - c0), np.float16)
        for k, d in enumerate(dates):
            if d in day:
                arr, tr, desc = got[day[d]]
                rr, cc = B.regular_index(lon, lat, tr.c, tr.f, tr.a, -tr.e, arr.shape[2], arr.shape[1])
                di = {nm: i for i, nm in enumerate(desc)}
                for j, nm in enumerate(["b1", "b2", "b3", "b4", "b5", "m11"]):
                    if nm in di:
                        out_arr[k, j] = np.nan_to_num(arr[di[nm]][rr, cc])
            if d in night:
                arr, tr, _ = got[night[d]]
                rr, cc = B.regular_index(lon, lat, tr.c, tr.f, tr.a, -tr.e, arr.shape[2], arr.shape[1])
                out_arr[k, 6] = np.nan_to_num(arr[0][rr, cc])
                out_arr[k, 7] = np.nan_to_num(arr[1][rr, cc])
        np.savez_compressed(dst, bands=out_arr, names=np.array(NAMES), dates=np.array(dates))
        print(f"[{n + 1}/{len(ids)}] {fid}: {out_arr.shape} ({time.time() - t0:.0f}s)", flush=True)
    print("Done")


if __name__ == "__main__":
    sys.exit(main())
