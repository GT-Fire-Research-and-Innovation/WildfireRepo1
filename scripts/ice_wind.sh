#!/usr/bin/env bash
# Terrain-adjusted wind (WindNinja) on GT PACE-ICE: the first test, with no training.
#   bash scripts/ice_wind.sh            everything below, in order (the jobs wait for each other)
#   bash scripts/ice_wind.sh setup      install WindNinja once (login node: needs the internet, ~2 GB in scratch)
#   bash scripts/ice_wind.sh build      wind response of the 2023-2025 fires (validation + test), one CPU job
#   bash scripts/ice_wind.sh test       score with and without it (six short GPU jobs)
#   bash scripts/ice_wind.sh results    the comparison, once the jobs are done
#   YEARS=2014-2025 bash scripts/ice_wind.sh build     every fire: needed before training with --wind-terrain 1
#   MESH=75 bash scripts/ice_wind.sh    the finer version: wind solved on a 75 m mesh from 30 m elevation, then
#                                       averaged into the 150 m cells (terrain_wind75.npz, results named ..._wn75).
#                                       About 5 times the computing of the default; both can be built and compared.
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
DATA="${WF_DATA:-$HOME/scratch/wildfire/my}"
BAD="--exclude=atl1-1-03-014-16-0"
YEARS="${YEARS:-2023,2024,2025}"
MESH="${MESH:-}"                              # empty: solve on the grid's own terrain
if [ -n "$MESH" ]; then WT=$MESH; TAG="wn$MESH"; BUILD="--mesh $MESH --max-cells 1200000 --workers 6 --no-check"; RES="-c 8 --mem=96G -t 16:00:00"
else WT=1; TAG="wn"; BUILD="--workers 8"; RES="-c 8 --mem=64G -t 12:00:00"; fi
case "$YEARS" in *-*) YEARS=$(seq -s, "${YEARS%-*}" "${YEARS#*-}");; esac
step="${1:-all}"
build_job=""
queued() { squeue -h -u "$USER" -o %j 2>/dev/null | grep -qx "$1"; }      # is a job with this name waiting or running?

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
  if queued "wf-$TAG" && [ "${FORCE:-0}" != 1 ]; then echo "!! a wf-$TAG build is already queued or running (two would overwrite each other's files): not submitted"; exit 1; fi
  build_job=$(sbatch --parsable -p ice-cpu --qos=coe-ice -N 1 $RES -o logs/wind-%j.out -J wf-$TAG \
      --wrap "module purge; export PATH=\$HOME/scratch/envs/wildfire/bin:\$PATH; echo \"code: \$(git log --oneline -1)\"; python pipeline/windninja_fields.py --data $DATA --wn $WN --years $YEARS $BUILD") || exit 1
  echo "wind response ($TAG) for $YEARS: job $build_job  (log: logs/wind-$build_job.out; resumable, built fires are skipped)"
fi

if [ "$step" = test ] || [ "$step" = all ]; then
  dep=${build_job:+--dependency=afterok:$build_job}
  ev() {   # ev <models> <out> <extra>
    local j; j=$(MODELS="$1" OUT="$2" EXTRA="$3" sbatch --parsable -J "wf-eval-$2" -t 4:00:00 $BAD $dep scripts/ice_eval.sbatch) && echo "  $2: $j"
  }
  echo "evaluations:"
  [ -f "$DATA/runs/eval_physics_base.json" ]  || dep="" ev "physics_ref:physics"   physics_base  "--wind-terrain 0"
  [ -f "$DATA/runs/eval_physics2_base.json" ] || dep="" ev "physics2_ref:physics2" physics2_base "--wind-terrain 0"
  ev "physics_ref:physics"   "physics_$TAG"          "--wind-terrain $WT"
  ev "physics2_ref:physics2" "physics2_$TAG"         "--wind-terrain $WT"
  ev "hybrid_v3"             "hybrid_v3_${TAG}_eval"   "--wind-terrain $WT"
  ev "unet3_fm_v3"           "unet3_fm_v3_${TAG}_eval" "--wind-terrain $WT"
  echo "then: bash scripts/ice_wind.sh results"
fi

if [ "$step" = results ]; then
  export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
  echo "== wind build"; tail -2 "$(ls -t logs/wind-*.out 2>/dev/null | head -1)" 2>/dev/null
  echo "fires with a wind response: grid terrain $(ls "$DATA"/fires/*/terrain_wind.npz 2>/dev/null | wc -l), finer $(ls "$DATA"/fires/*/terrain_wind[0-9]*.npz 2>/dev/null | wc -l)"
  grep -h "terrain wind:" logs/eval-wf-eval-*_wn*.out 2>/dev/null | sort | uniq -c
  have=""
  extra=$(cd "$DATA/runs" && ls eval_*_wn[0-9]*.json 2>/dev/null | sed 's/^eval_//; s/\.json$//')
  for n in physics_base physics_wn physics2_base physics2_wn hybrid_v3_eval hybrid_v3_wn_eval unet3_fm_v3_eval unet3_fm_v3_wn_eval $extra; do
    [ -f "$DATA/runs/eval_$n.json" ] && have="$have $n" || echo "(not finished: $n)"
  done
  [ -n "$have" ] && (cd model && python compare.py --data "$DATA" $have 2>&1 | awk '/^## /{p = ($0 ~ /Test years|HRRR wind|growth/)} p')
fi
