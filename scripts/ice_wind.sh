#!/usr/bin/env bash
# Terrain-adjusted wind (WindNinja) on GT PACE-ICE: the first test, with no training.
#   bash scripts/ice_wind.sh            everything below, in order (the jobs wait for each other)
#   bash scripts/ice_wind.sh setup      install WindNinja once (login node: needs the internet, ~2 GB in scratch)
#   bash scripts/ice_wind.sh build      wind response of the 2023-2025 fires (validation + test), one CPU job
#   bash scripts/ice_wind.sh test       score with and without it (six short GPU jobs)
#   bash scripts/ice_wind.sh results    the comparison, once the jobs are done
#   YEARS=2014-2025 bash scripts/ice_wind.sh build     every fire: needed before training with --wind-terrain 1
#
# What the test can show. Physics-only models have no weights, so with and without is a clean comparison:
# physics_wn vs physics_base, physics2_wn vs physics2_base. hybrid_v3 and unet3_fm_v3 were TRAINED on the plain
# HRRR wind; scoring them on terrain wind is an indication only (a network sees inputs it never trained on).
# If physics improves, most of all in the high-wind bins, build every fire and train unet3_fm with --wind-terrain 1.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ENVS="$HOME/scratch/envs"
WN="$ENVS/windninja/bin/WindNinja_cli"
DATA="$HOME/scratch/wildfire/my"
BAD="--exclude=atl1-1-03-014-16-0"
YEARS="${YEARS:-2023,2024,2025}"
case "$YEARS" in *-*) YEARS=$(seq -s, "${YEARS%-*}" "${YEARS#*-}");; esac
step="${1:-all}"
build_job=""

if [ "$step" = setup ] || [ "$step" = all ]; then
  if [ ! -x "$WN" ]; then
    echo "installing WindNinja into $ENVS/windninja (conda-forge, a few minutes)"
    mkdir -p "$ENVS/mm" && curl -sSL https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64.tar.bz2 \
      | tar -xj -C "$ENVS/mm" bin/micromamba || { echo "!! could not download micromamba"; exit 1; }
    MAMBA_ROOT_PREFIX="$ENVS/mm" "$ENVS/mm/bin/micromamba" create -y -q -p "$ENVS/windninja" -c conda-forge windninja \
      || { echo "!! WindNinja install failed"; exit 1; }
  fi
  "$WN" --version || { echo "!! $WN does not run"; exit 1; }
fi

if [ "$step" = build ] || [ "$step" = all ]; then
  [ -x "$WN" ] || { echo "run: bash scripts/ice_wind.sh setup"; exit 1; }
  build_job=$(sbatch --parsable -p ice-cpu --qos=coe-ice -N 1 -c 8 --mem=64G -t 12:00:00 -o logs/wind-%j.out -J wf-wind \
      --wrap "module purge; export PATH=\$HOME/scratch/envs/wildfire/bin:\$PATH; echo \"code: \$(git log --oneline -1)\"; python pipeline/windninja_fields.py --data $DATA --wn $WN --years $YEARS --workers 8") || exit 1
  echo "wind response for $YEARS: job $build_job  (log: logs/wind-$build_job.out; resumable, built fires are skipped)"
fi

if [ "$step" = test ] || [ "$step" = all ]; then
  dep=${build_job:+--dependency=afterok:$build_job}
  ev() {   # ev <models> <out> <extra>
    local j; j=$(MODELS="$1" OUT="$2" EXTRA="$3" sbatch --parsable -J "wf-eval-$2" -t 4:00:00 $BAD $dep scripts/ice_eval.sbatch) && echo "  $2: $j"
  }
  echo "evaluations:"
  ev "physics_ref:physics"   physics_base        "--wind-terrain 0"
  ev "physics_ref:physics"   physics_wn          "--wind-terrain 1"
  ev "physics2_ref:physics2" physics2_base       "--wind-terrain 0"
  ev "physics2_ref:physics2" physics2_wn         "--wind-terrain 1"
  ev "hybrid_v3"             hybrid_v3_wn_eval   "--wind-terrain 1"
  ev "unet3_fm_v3"           unet3_fm_v3_wn_eval "--wind-terrain 1"
  echo "then: bash scripts/ice_wind.sh results"
fi

if [ "$step" = results ]; then
  export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
  echo "== wind build"; tail -2 "$(ls -t logs/wind-*.out 2>/dev/null | head -1)" 2>/dev/null
  echo "fires with a wind response: $(ls "$DATA"/fires/*/terrain_wind.npz 2>/dev/null | wc -l)"
  grep -h "terrain wind:" logs/eval-wf-eval-*_wn*.out 2>/dev/null | sort | uniq -c
  have=""
  for n in physics_base physics_wn physics2_base physics2_wn hybrid_v3_eval hybrid_v3_wn_eval unet3_fm_v3_eval unet3_fm_v3_wn_eval; do
    [ -f "$DATA/runs/eval_$n.json" ] && have="$have $n" || echo "(not finished: $n)"
  done
  [ -n "$have" ] && (cd model && python compare.py --data "$DATA" $have 2>&1 | awk '/^## /{p = ($0 ~ /Test years|HRRR wind|growth/)} p')
fi
