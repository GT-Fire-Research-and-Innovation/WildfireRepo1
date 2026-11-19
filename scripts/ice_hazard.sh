#!/usr/bin/env bash
# FireStep Stage 1 on GT PACE-ICE: UNet3 + per-bin hazard head trained on interval-censored perimeters
# (model/train_hazard.py), same inputs as unet3_fm_v3 (fuel moisture, 6-day history, no attention, no late start),
# on the original fires so it is scored on the same 295 test fires as the 0.261 baseline.
#   bash scripts/ice_hazard.sh smoke        ~1 h on a few fires: run this FIRST, paste the log tail
#   bash scripts/ice_hazard.sh train        seeds 0 1 2 (or: train 1 2): training + evaluate.py for each
#   bash scripts/ice_hazard.sh baseline     re-score unet3_fm_v3 (3 seeds) with the new code: adds calibration by gap
#   bash scripts/ice_hazard.sh results      the hazard seeds next to unet3_fm_v3 (AP, calibration by gap and horizon)
# Success (the plan): matches or beats 0.261 on every seed AND is better calibrated across gaps. Clearly worse: stop.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export WF_DATA="${WF_DATA:-$HOME/scratch/wildfire/my}"
RUNS="$WF_DATA/runs"
BAD="--exclude=atl1-1-03-014-16-0"
BASE="hazard_v1${TAG:-}"
X="--width 64 --u3-time 1 --u3-attn 0 --fuel-moisture 1"
step="${1:-}"; shift || true
queued() { squeue -h -u "$USER" -o %j 2>/dev/null | grep -qx "$1"; }

case "$step" in
smoke)
  N="${BASE}_smoke"
  if queued "wf-$N"; then echo "$N: already queued or running (not submitted again)"; exit 0; fi
  rm -f "$RUNS/$N.ckpt" "$RUNS/$N.pt"
  j=$(SCRIPT=train_hazard.py NAME=$N EXTRA="$X --max-fires 40 --epochs 2 --steps 20 --eval-every 1 --max-pairs 60" \
      sbatch --parsable -J "wf-$N" -t 1:30:00 $BAD scripts/ice_train.sbatch) || exit 1
  e=$(MODELS=$N OUT=${N}_eval EXTRA="--max-pairs 60" sbatch --parsable -J "wf-eval-$N" -t 1:00:00 $BAD --dependency=afterok:$j scripts/ice_eval.sbatch)
  echo "$N: train $j, evaluate.py $e   ->  tail -30 logs/train-wf-$N-$j.out logs/eval-wf-eval-$N-$e.out"
  ;;
train)
  for s in "${@:-0 1 2}"; do
    for s1 in $s; do
      N="$BASE$([ "$s1" = 0 ] || echo "s$s1")"
      if queued "wf-$N"; then echo "$N: already queued or running (not submitted again)"; continue; fi
      prev=""; all=""
      if [ ! -f "$RUNS/$N.pt" ]; then
        for k in 1 2; do              # two chained 8 h jobs; the second resumes from the checkpoint
          j=$(SCRIPT=train_hazard.py NAME=$N EXTRA="$X --seed $s1" sbatch --parsable -J "wf-$N" -t 8:00:00 $BAD \
              ${prev:+--dependency=afterany:$prev} scripts/ice_train.sbatch) || exit 1
          prev=$j; all="${all:+$all:}$j"
        done
        echo "$N: queued ($all)   [$X --seed $s1]"
      fi
      if queued "wf-eval-$N"; then echo "  evaluation of $N already queued"; continue; fi
      e=$(MODELS=$N OUT=${N}_eval sbatch --parsable -J "wf-eval-$N" -t 6:00:00 $BAD ${all:+--dependency=afterany:$all} scripts/ice_eval.sbatch)
      echo "  evaluate.py: ${N}_eval $e"
    done
  done
  ;;
baseline)
  for N in unet3_fm_v3 unet3_fm_v3s1 unet3_fm_v3s2; do
    if queued "wf-eval-${N}_cal"; then echo "${N}_cal: already queued"; continue; fi
    [ -f "$RUNS/$N.pt" ] || { echo "!! $RUNS/$N.pt missing: skipped"; continue; }
    e=$(MODELS=$N OUT=${N}_cal sbatch --parsable -J "wf-eval-${N}_cal" -t 4:00:00 $BAD scripts/ice_eval.sbatch)
    echo "${N}_cal: $e"
  done
  ;;
results)
  export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
  have=""
  for n in ${BASE}_eval ${BASE}s1_eval ${BASE}s2_eval unet3_fm_v3_cal unet3_fm_v3s1_cal unet3_fm_v3s2_cal \
           unet3_fm_v3_eval unet3_fm_v3s1_eval unet3_fm_v3s2_eval; do
    [ -f "$RUNS/eval_$n.json" ] && have="$have $n"
  done
  [ -n "$have" ] || { echo "no evaluations yet"; exit 1; }
  (cd model && python compare.py --data "$WF_DATA" $have 2>&1 | awk '/^## /{p = ($0 ~ /Test years|time gap|Calibration|HRRR wind|growth/)} p')
  echo "== calibration by horizon (every later perimeter within 72 h; from each training run's own test pass)"
  for N in $BASE ${BASE}s1 ${BASE}s2; do
    [ -f "$RUNS/$N.json" ] && python -c "
import json, sys; j = json.load(open('$RUNS/$N.json')); t = j['test']
print('$N', 'AP %.3f' % t['ap'], 'interval NLL %.4f' % t['interval_nll'], '| ' + '  '.join('%s ECE %.3f pred/obs %.3f/%.3f' % (r['group'], r['ece'], r['mean_pred'], r['observed']) for r in t['by_horizon']))"
  done
  ;;
*) echo "usage: bash scripts/ice_hazard.sh smoke | train [seeds] | baseline | results"; exit 1 ;;
esac
