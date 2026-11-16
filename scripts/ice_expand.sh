#!/usr/bin/env bash
# The expanded fire set on GT PACE-ICE: smaller fires (100 acres and up, two real mappings 3 h apart), same-day
# perimeters, and no auto-generated circles (catalog rules v2, pipeline/catalog.py). It is built into a second
# pair of folders, catalog2 and my2, so the first dataset and every score made on it stay as they are.
#
#   bash scripts/ice_expand.sh catalog   1. new catalog, quality ratings, duplicates, and my2 started from my by links (one CPU job)
#   bash scripts/ice_expand.sh build     2. everything missing in my2: all layers of new fires, labels of every fire (CPU job, resumable)
#   bash scripts/ice_expand.sh extras    3. link again (picks up fuel shares / wind finished in my since), then fuel-moisture
#                                           weather, greenness, fuel shares and terrain wind for the rest
#   bash scripts/ice_expand.sh status    fires and layers in my2
#   then:  WF_DATA=$HOME/scratch/wildfire/my2 TAG=_x bash scripts/ice_all.sh train
#          (scored on my2's test fires AND on the original test fires, like for like with every earlier model)
# Run each step after the one before it has finished. Steps 2 and 3 can be submitted again after a timeout.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
W="$HOME/scratch/wildfire"
OLD="$W/my"; NEW="$W/my2"; CAT="$W/catalog2"
MIN_ACRES="${MIN_ACRES:-100}"
PY='module purge; export PATH=$HOME/scratch/envs/wildfire/bin:$PATH; set -a; [ -f $HOME/.config/wildfire/env ] && . $HOME/.config/wildfire/env; set +a; echo "code: $(git log --oneline -1)"; cd pipeline'
queued() { squeue -h -u "$USER" -o %j 2>/dev/null | grep -qx "$1"; }
submit() {   # submit <job name> <cpus> <mem> <hours> <commands>
  if queued "$1"; then echo "!! $1 is already queued or running: not submitted again" >&2; return 1; fi
  sbatch --parsable -p ice-cpu --qos=coe-ice -N 1 -c "$2" --mem="$3" -t "$4":00:00 -o "logs/expand-%j.out" -J "$1" --wrap "$PY; $5"
}
count() { ls "$NEW"/fires/*/"$1".npz 2>/dev/null | wc -l; }

case "${1:-status}" in
catalog)
  j=$(submit wf-catalog2 4 64G 8 "python catalog.py all --out $W --name catalog2 --from-catalog $W/catalog --rules v2 --min-acres $MIN_ACRES --years 2014-2025 \
      && python qc_fires.py --catalog $CAT && python dedupe_fires.py --catalog $CAT \
      && python expand_dataset.py --old $OLD --new $NEW --catalog $CAT") || exit 1
  echo "catalog2 + my2: job $j  (log: logs/expand-$j.out). When it has finished: bash scripts/ice_expand.sh build"
  ;;
build)
  [ -f "$CAT/fires.csv" ] && [ -f "$NEW/catalog.txt" ] || { echo "!! run 'bash scripts/ice_expand.sh catalog' first and wait for it"; exit 1; }
  j=$(submit wf-build2 16 96G 16 "python build_multiyear.py --catalog $CAT --out $NEW --years 2014-2025 \
      --stages labels,history,viirs,terrain,weather,landfire,tessera,suppression,hrrr,goes,human --workers 8 2>&1 | tee -a $W/build2.log; \
      python qc_fires.py --catalog $CAT --built $NEW 2>&1 | tail -8") || exit 1
  echo "build of what is missing in my2: job $j  (log: logs/expand-$j.out; 16 h limit, submit again to continue)"
  ;;
extras)
  [ -f "$NEW/catalog.txt" ] || { echo "!! run the catalog step first"; exit 1; }
  export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
  python pipeline/expand_dataset.py --old "$OLD" --new "$NEW" --catalog "$CAT" | head -2
  j=$(submit wf-extras2 16 96G 16 "python build_multiyear.py --catalog $CAT --out $NEW --years 2014-2025 --stages fuelwx,green,fuelfrac --workers 6 2>&1 | tee -a $W/build2.log") \
    && echo "fuel-moisture weather, greenness, fuel shares for my2: job $j  (log: logs/expand-$j.out; submit again to continue)"
  WF_DATA="$NEW" YEARS=2014-2025 MESH="${MESH:-}" bash scripts/ice_wind.sh build
  ;;
status)
  [ -d "$NEW/fires" ] || { echo "my2 does not exist yet: bash scripts/ice_expand.sh catalog"; exit 0; }
  n=$(ls -d "$NEW"/fires/*/ 2>/dev/null | wc -l)
  echo "catalog2 fires: $([ -f "$CAT/fires.csv" ] && tail -n +2 "$CAT/fires.csv" | wc -l)   folders in my2: $n"
  for s in labels terrain weather viirs landfire tessera suppression hrrr goes human hrrr2 green fuelfrac terrain_wind; do printf "  %-13s %s\n" "$s" "$(count $s)"; done
  ;;
*) echo "usage: bash scripts/ice_expand.sh catalog | build | extras | status"; exit 1 ;;
esac
