"""
Package the models the live map uses (run on ICE after combine.py): copies each router member's
weights and writes router.json, so live/publish.py can apply the same router to live fires.

  python live/export_models.py --runs ~/scratch/wildfire/my/runs --combine combine --out live/models
Then add live/models/ to the repository (see live/README.md for the upload options).
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--combine", default="combine", help="runs/<name>.json written by combine.py")
    ap.add_argument("--out", default="live/models")
    a = ap.parse_args()
    runs, out = Path(a.runs).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((runs / f"{a.combine}.json").read_text())
    rm = res.get("router_model")
    if not rm:
        raise SystemExit(f"{a.combine}.json has no router_model: rerun combine.py with the current code")
    members = []
    for ev in rm["members"]:
        spec = json.loads((runs / f"eval_{ev}.json").read_text())["models"][0]
        name, _, mode = spec.partition(":")
        if not mode:
            mode = next(m for m in ("physics2", "hybrid2", "unet3", "unet2", "physics", "unet", "hybrid") if name.startswith(m))
        src = runs / f"{name}.pt"
        shutil.copy2(src, out / src.name)
        members.append(dict(eval=ev, name=name, mode=mode, file=src.name))
        print(f"  {ev}: {src.name} ({mode}, {src.stat().st_size / 1e6:.1f} MB)")
    router = dict(members=members, temps=rm["temps"], coef=rm["coef"], intercept=rm["intercept"],
                  threshold=rm.get("threshold"), features=rm["features"], source=f"{a.combine}.json",
                  test_scores={k: round(v["ap"], 3) for k, v in [("router", res["router"])]},
                  best_on_validation=res.get("best_on_validation"))
    (out / "router.json").write_text(json.dumps(router, indent=1))
    print(f"wrote {out / 'router.json'} ({len(members)} models)")


if __name__ == "__main__":
    main()
