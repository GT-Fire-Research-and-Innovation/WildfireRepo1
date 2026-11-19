"""
How long after the real mapping does a perimeter's edit-time stamp come? (sets the hazard model's edit slack)

A perimeter stamped with a database edit time was mapped at or before that stamp, by an unknown amount. For records
that carry BOTH a mapping time (perimeterdatetime / polygondatetime, not a bare date) and edit times, the delay from
mapping to the FIRST edit measures that amount. Later re-edits of old perimeters are excluded by taking the first
edit per perimeter (the same perimeter id over archive versions). The edit field is the one the catalog would have
used as the stamp: datecurrent if present, else gdb_from_date (catalog._time order); both are also reported.

Reads the attribute scans the catalog made (attrs_<year>.parquet in one or more catalog folders).
  python pipeline/measure_edit_delay.py ~/scratch/wildfire/catalog ~/scratch/wildfire/catalog2
Use about the 90th percentile as --edit-slack-h for train_hazard.py (err wide: too little slack teaches wrong timing).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def parse_time(v):
    """A catalog attribute column -> UTC times: epoch milliseconds (2000-2030) or date strings; anything else NaT."""
    v = pd.Series(v)
    num = pd.to_numeric(v, errors="coerce")
    t = pd.to_datetime(num.where(num.between(946684800000, 1893456000000)), unit="ms", utc=True, errors="coerce")
    s = v.where(num.isna()).astype("string")
    yr = pd.to_numeric(s.str.extract(r"^(\d{4})", expand=False), errors="coerce")
    t2 = pd.to_datetime(s.where(yr.between(2000, 2030)), utc=True, errors="coerce", format="mixed")
    return t.fillna(t2.astype("datetime64[ns, UTC]"))


def edit_delays(df):
    """One row per perimeter with a real mapping time: hours from mapping to its first edit (stamp field, and each
    field separately). df: raw attribute rows (several archive versions of a perimeter allowed)."""
    col = lambda c: df[c] if c in df else pd.Series(None, index=df.index, dtype="object")  # noqa: E731
    m = parse_time(col("perimeterdatetime")).fillna(parse_time(col("polygondatetime")))
    bare = (m.dt.hour == 0) & (m.dt.minute == 0) & (m.dt.second == 0)
    m = m.where(~bare)
    dc, gf = parse_time(col("datecurrent")), parse_time(col("gdb_from_date"))
    pid = col("sourceglobalid").astype("string").fillna(col("globalid").astype("string"))
    pid = pid.fillna(col("layer_url").astype("string").fillna("") + "#" + col("objectid").astype("string"))
    src = col("source").astype("string").fillna("unknown")
    year = m.dt.year
    d = pd.DataFrame({"id": pid, "source": src, "year": year, "mapped": m, "datecurrent": dc, "gdb_from_date": gf})
    d = d[d["mapped"].notna() & d["id"].notna()]
    g = d.groupby("id").agg(source=("source", "first"), year=("year", "first"), mapped=("mapped", "min"),
                            datecurrent=("datecurrent", "min"), gdb_from_date=("gdb_from_date", "min"))
    h = lambda a: (a - g["mapped"]).dt.total_seconds() / 3600  # noqa: E731
    g["delay_datecurrent_h"], g["delay_gdb_from_date_h"] = h(g["datecurrent"]), h(g["gdb_from_date"])
    g["delay_h"] = g["delay_datecurrent_h"].fillna(g["delay_gdb_from_date_h"])
    return g.reset_index()


def summary(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return "no records"
    q = np.quantile(x, QS)
    return (f"n {len(x)}  negative {np.mean(x < 0):.1%}  over 7 d {np.mean(x > 168):.1%}  "
            + "  ".join(f"p{int(p * 100)} {v:.1f} h" for p, v in zip(QS, q)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogs", nargs="+", help="catalog folders holding attrs_<year>.parquet")
    ap.add_argument("--out", help="CSV of per-perimeter delays")
    a = ap.parse_args()
    files = sorted({p.resolve() for c in a.catalogs for p in Path(c).expanduser().glob("attrs_*.parquet")})
    if not files:
        sys.exit(f"no attrs_*.parquet in {a.catalogs}")
    g = pd.concat([edit_delays(pd.read_parquet(f)) for f in files], ignore_index=True).drop_duplicates("id")
    print(f"{len(files)} attribute scans, {len(g)} perimeters with a real mapping time")
    print("first edit after mapping (the stamp field):  " + summary(g["delay_h"]))
    print("  datecurrent only:                          " + summary(g["delay_datecurrent_h"]))
    print("  gdb_from_date only:                        " + summary(g["delay_gdb_from_date_h"]))
    print("  (GeoMAC stores local clock times in both fields, so its delays are still valid)")
    for key in ("source", "year"):
        for v, sub in g.groupby(key):
            print(f"  {key} {v}: " + summary(sub["delay_h"]))
    ok = g["delay_h"][(g["delay_h"] >= 0) & (g["delay_h"] <= 168)]
    if len(ok):
        p90 = float(np.quantile(ok, 0.9))
        print(f"\nsuggested --edit-slack-h: {np.ceil(p90):.0f}  (90th percentile of delays in 0-7 days; "
              f"{np.mean(g['delay_h'] > 168):.1%} of first edits came over 7 days later and are left out of it)")
    if a.out:
        g.to_csv(a.out, index=False)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
