#!/usr/bin/env bash
# Train and score the fuel-moisture model: unet3_time with the hourly, slope-aware fuel moisture, satellite
# greenness and the fuel map of each fire's own year. Compare with unet3_time_v3 (same network without them).
# Run from the fuel-moisture checkout, after scripts/ice_fuelbuild.sbatch has finished:
#   bash scripts/ice_fm.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
RUNS="$HOME/scratch/wildfire/my/runs"
BAD="--exclude=atl1-1-03-014-16-0"
sub() { sbatch --parsable "$@"; }
prev=""; all=""
if [ ! -f "$RUNS/unet3_fm_v3.pt" ]; then
  for k in 1 2; do
    j=$(NAME=unet3_fm_v3 MODE=unet3 EXTRA="--width 64 --u3-time 1 --u3-attn 0 --fuel-moisture 1" \
        sub -J wf-unet3_fm_v3 -t 8:00:00 $BAD ${prev:+--dependency=afterany:$prev} scripts/ice_train.sbatch) || exit 1
    prev=$j; all="${all:+$all:}$j"
  done
  echo "unet3_fm_v3: queued ($all)"
fi
e=$(PULL=1 MODELS=unet3_fm_v3 OUT=unet3_fm_v3_eval sub -J wf-eval-unet3_fm_v3 -t 4:00:00 $BAD ${all:+--dependency=afterany:$all} scripts/ice_eval.sbatch)
echo "evaluation: $e  ->  $RUNS/eval_unet3_fm_v3_eval.json (compare with eval_unet3_time_v3_eval.json)"
