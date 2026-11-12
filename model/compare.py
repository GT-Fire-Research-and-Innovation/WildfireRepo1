"""
Side-by-side report of evaluate.py results (runs/eval_*.json).

  python compare.py --data ~/scratch/wildfire/my                      # every evaluation found
  python compare.py --data ... physics_eval unet_v1_eval hybrid_v1_before_eval

Prints the headline table (raw and calibrated per model, plus the baselines) and the breakdowns by
growth, fire size, fuel, wind and gap, and writes runs/compare.md with the same tables.
"""
import argparse
import json
from pathlib import Path

COLS = [("AP", "ap", 3), ("F1", "f1", 3), ("IoU", "iou", 3), ("P", "precision", 3), ("R", "recall", 3),
        ("area", "area_ratio", 2), ("R@area", "recall_matched_area", 3), ("F1 win", "f1_window", 3),
        ("edge km", "edge_km_mean", 2), ("Brier", "brier", 4), ("ECE", "ece", 3)]
BREAKDOWNS = [("by_growth", "growth in the pair"), ("by_fire_size", "final fire size"), ("by_fuel", "fuel family"),
              ("by_wind", "HRRR wind"), ("by_gap", "time gap")]


def fmt(v, d):
    return "-" if v is None or v != v else f"{v:.{d}f}"


def table(header, rows):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"  # noqa: E731
    return "\n".join([line(header), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("names", nargs="*", help="evaluation names (the --out of evaluate.py); default all")
    a = ap.parse_args()
    runs = Path(a.data).expanduser() / "runs"
    files = [runs / f"eval_{n}.json" for n in a.names] if a.names else sorted(runs.glob("eval_*.json"))
    evals = [(f.stem[5:], json.loads(f.read_text())) for f in files if f.exists()]
    if not evals:
        raise SystemExit(f"no evaluations in {runs}")
    out = []
    rows, base = [], None
    for name, e in evals:
        for v in ("raw", "calibrated"):
            if v in e:
                rows.append([f"{name} {v}"] + [fmt(e[v].get(k), d) for _, k, d in COLS])
        base = base or e
    for v, label in (("distance", "baseline: grow evenly outward"), ("no_growth", "baseline: no growth")):
        if v in base:
            rows.append([label] + [fmt(base[v].get(k), d) for _, k, d in COLS])
    n_pairs = len(base.get("pairs", []))
    versions = {name: e.get("data_version", 1) for name, e in evals}
    if len(set(versions.values())) > 1:
        out.append("WARNING: evaluations use different label versions (not the same cells): "
                   + ", ".join(f"{n} v{v}" for n, v in versions.items()) + "\n")
    leaky = [name for name, e in evals if e.get("input_version", 1) < 5]
    if leaky:
        out.append("WARNING: these evaluations used inputs from before the time-leak fixes (satellite passes or GOES checks "
                   "after the forecast start were visible); their scores are optimistic and not comparable: "
                   + ", ".join(leaky) + "\n")
    out.append(f"## Test years, {n_pairs} perimeter pairs (all models scored on the same cells)\n")
    out.append(table(["model"] + [c for c, _, _ in COLS], rows))
    for key, title in BREAKDOWNS:
        groups = []
        for _, e in evals:
            for b in e.get("calibrated", e.get("raw", {})).get(key, []):
                if b["group"] not in groups:
                    groups.append(b["group"])
        if not groups:
            continue
        rows = []
        for name, e in evals:
            for v in ("raw", "calibrated"):
                if v not in e:
                    continue
                by = {b["group"]: b for b in e[v].get(key, [])}
                rows.append([f"{name} {v}"] + [f"{fmt(by[g]['ap'], 2)} / {fmt(by[g]['f1'], 2)}" if g in by else "-" for g in groups])
        if "distance" in base:
            by = {b["group"]: b for b in base["distance"].get(key, [])}
            rows.append(["baseline: grow evenly"] + [f"{fmt(by[g]['ap'], 2)} / {fmt(by[g]['f1'], 2)}" if g in by else "-" for g in groups])
        out.append(f"\n## By {title} (AP / F1)\n")
        out.append(table(["model"] + groups, rows))
    for name, e in evals:
        c = e.get("calibration_params")
        if c:
            parts = [f"{k}: " + ", ".join(f"{kk} {vv:.2f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in q.items())
                     for k, q in c.items() if isinstance(q, dict)]
            out.append(f"\n{name}: calibrated {c['pairs_calibrated']}/{c['pairs_total']} pairs; " + "; ".join(parts))
    text = "\n".join(out)
    print(text)
    (runs / "compare.md").write_text(text + "\n")
    print(f"\nsaved {runs / 'compare.md'}")


if __name__ == "__main__":
    main()
