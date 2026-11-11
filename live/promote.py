"""
Promotion gate for the continuous-learning loop: should a fine-tuned model replace the live one?

Compares evaluation results (runs/eval_<name>.json from model/evaluate.py) of the candidate and the
incumbent on two sets:
  standard  the frozen 2024-2025 test fires: the candidate must not get worse (AP drop <= --tolerance)
  new       the newest fires from the live archive, held out of fine-tuning: the candidate must do
            at least as well (this is what learning from new fires should improve)
Prints a verdict and exits 0 to promote, 1 to keep the live model.

  python live/promote.py --runs RUNS --candidate ft_eval --incumbent live_eval \
                         --candidate-new ft_new_eval --incumbent-new live_new_eval
"""
import argparse
import json
import sys
from pathlib import Path


def ap_of(runs, name, variant="raw"):
    d = json.loads((Path(runs) / f"eval_{name}.json").read_text())
    return float(d[variant]["ap"]), d.get("test_fires") or d.get("n_fires")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--candidate-new")
    p.add_argument("--incumbent-new")
    p.add_argument("--tolerance", type=float, default=0.005)
    a = p.parse_args()
    c, _ = ap_of(a.runs, a.candidate)
    i, _ = ap_of(a.runs, a.incumbent)
    ok_std = c >= i - a.tolerance
    print(f"standard test AP: candidate {c:.3f} vs live {i:.3f} -> {'ok' if ok_std else 'WORSE'}")
    ok_new = True
    if a.candidate_new and a.incumbent_new and (Path(a.runs) / f"eval_{a.candidate_new}.json").exists():
        cn, _ = ap_of(a.runs, a.candidate_new)
        inn, _ = ap_of(a.runs, a.incumbent_new)
        ok_new = cn >= inn
        print(f"new fires AP:     candidate {cn:.3f} vs live {inn:.3f} -> {'better or equal' if ok_new else 'WORSE'}")
    else:
        print("new fires: none held out yet (gate uses the standard test only)")
    verdict = ok_std and ok_new
    print("PROMOTE" if verdict else "KEEP the live model")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
