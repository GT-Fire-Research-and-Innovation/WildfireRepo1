#!/usr/bin/env bash
# Train everything first, then one evaluation period for every model, then the comparison report.
#
#   training (cleaned data, data version 2):
#     hybrid_v1     hybrid, original physics          (restarted on the cleaned data)
#     hybrid2_v1    hybrid, crown fire + spotting
#     physics2_v1   physics only, crown fire + spotting (5 learned settings)
#     unet_h21      U-Net with 2021 held out, for the head-to-head with the group's TS-SatFire model
#     unet2_v1      U-Net that also sees the crown-fire and ember maps (physics as inputs, no solver)
#     unet_fc       U-Net given only the weather known at forecast time (how much observed weather flatters scores)
#   evaluation period (starts by itself when ALL training above has finished; pulls the latest code first):
#     physics_v1, unet_v1, hybrid_v1_before, hybrid_v1, hybrid2_v1, physics2_v1  on the 2024-2025 test years
#     unet_h21 on the 19 TS-SatFire test fires (2021)
#   report: runs/compare.md once every evaluation is done
#
#   bash scripts/ice_plan.sh          (safe to rerun: nothing is queued twice)
set -uo pipefail          # no -e: one slow Slurm reply must not stop the plan halfway
cd "$(dirname "$0")/.."
mkdir -p logs
RUNS="$HOME/scratch/wildfire/my/runs"
PY="$HOME/scratch/envs/wildfire/bin/python"
q() { local out; for k in 1 2 3; do out=$(squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null) && { [ -n "$out" ] && echo "$out" | sort -n; return 0; }; sleep 3; done; }
sub() { local j; for k in 1 2 3; do j=$(sbatch --parsable "$@") && [ -n "$j" ] && { echo "$j"; return 0; }; echo "  (sbatch failed, retrying)" >&2; sleep 5; done
        echo "  !! could not submit: sbatch $*" >&2; return 1; }

# 0. clear the old one-by-one evaluations; old-label results are kept as *_oldlabels.json
EV_OLD=$(squeue -h -u "$USER" -o "%i %j" | awk '$2 ~ /^wf-eval/ || $2 == "wf-compare" {print $1}' | tr '\n' ' ')
[ -n "$EV_OLD" ] && scancel $EV_OLD && echo "cancelled queued evaluations (they move to the evaluation period): $EV_OLD"
for f in "$RUNS"/eval_*.json; do
  [ -e "$f" ] || continue
  case "$f" in *_oldlabels.json) continue ;; esac
  grep -q '"data_version": 2' "$f" || { mv "$f" "${f%.json}_oldlabels.json"; echo "kept old-label result as $(basename "${f%.json}")_oldlabels.json"; }
done
# old seeds and any run of the old hybrid still queued
SEEDS=$(squeue -h -u "$USER" -n wf-hyb-s1,wf-hyb-s2 -o %i | tr '\n' ' ')
[ -n "$SEEDS" ] && scancel $SEEDS && echo "cancelled old-physics seeds: $SEEDS"
if [ -f "$RUNS/hybrid_v1.ckpt" ] && [ ! -f "$RUNS/hybrid_v1_before.ckpt" ]; then
  OLD=$(q wf-hybrid | tr '\n' ' '); [ -n "$OLD" ] && scancel $OLD && sleep 5
  cp "$RUNS/hybrid_v1.ckpt" "$RUNS/hybrid_v1_before.ckpt"; rm -f "$RUNS"/hybrid_v1.{ckpt,json,pt}
  echo "saved the old hybrid_v1 as hybrid_v1_before"
fi
# v2 runs finished with the pre-fix spotting head can't be evaluated: set them aside and retrain
for n in physics2_v1 hybrid2_v1; do
  if [ -f "$RUNS/$n.pt" ] && ! "$PY" -c "import torch,sys; sys.exit(0 if 'log_k' in torch.load('$RUNS/$n.pt',weights_only=False) else 1)" 2>/dev/null; then
    mkdir -p "$RUNS/old"; mv "$RUNS"/$n.* "$RUNS/old/"; echo "$n was trained with the old spotting head: moved to runs/old, retraining"
  fi
done

# 1. training
TRAIN=()
train() { local name="$1" mode="$2" jobname="$3" chain="$4" extra="${5:-}" t="${6:-16:00:00}"
  local ids; ids=$(q "$jobname" | tr '\n' ' ')
  if [ -n "${ids// /}" ]; then TRAIN+=($ids); echo "$name: already queued/running ($ids)"; return; fi
  [ -f "$RUNS/$name.pt" ] && { echo "$name: already trained"; return; }
  local prev="" j
  for k in $(seq 1 "$chain"); do
    j=$(NAME=$name MODE=$mode EXTRA="$extra" sub -J "$jobname" -t "$t" ${prev:+--dependency=afterany:$prev} scripts/ice_train.sbatch)
    TRAIN+=("$j"); prev=$j
  done
  echo "$name: queued ($(q "$jobname" | tr '\n' ' '))"; }
train hybrid_v1   hybrid   wf-hybrid   2
train hybrid2_v1  hybrid2  wf-hybrid2  2
train physics2_v1 physics2 wf-phys2    1 "--epochs 4 --steps 200 --lr 0.02 --eval-every 2" 4:00:00
train unet_h21    unet     wf-unet-h21 1 "--val-years 2020 --test-years 2021" 6:00:00
train unet2_v1    unet2    wf-unet2    1 "" 8:00:00
train unet_fc     unet     wf-unet-fc  1 "--weather forecast" 8:00:00
DEP=""; [ ${#TRAIN[@]} -gt 0 ] && DEP="--dependency=afterany:$(IFS=:; echo "${TRAIN[*]}")"

# 2. evaluation period
EV=()
ev() { local models="$1" out="$2" extra="${3:-}"
  [ -f "$RUNS/eval_$out.json" ] && { echo "eval_$out: already done"; return; }
  local j; j=$(PULL=1 MODELS="$models" OUT="$out" EXTRA="$extra" sub -J "wf-eval-$out" -t 4:00:00 $DEP scripts/ice_eval.sbatch)
  EV+=("$j"); echo "evaluation $out: $j"; }
ev "physics_v1:physics"      physics_eval
ev "unet_v1"                 unet_v1_eval
[ -f "$RUNS/hybrid_v1_before.ckpt" ] && ev "hybrid_v1_before:hybrid" hybrid_v1_before_eval
ev "hybrid_v1"               hybrid_v1_eval
ev "hybrid2_v1"              hybrid2_eval
ev "physics2_v1"             physics2_eval
ev "unet2_v1"                unet2_eval
ev "unet_fc"                 unet_fc_eval "--weather forecast"
ev "unet_h21"                unet_h21_vs_tssatfire "--val-years 2020 --test-years 2021 --fires ../benchmarks/tssatfire_2021_test_match.csv"

# 3. report
if [ ${#EV[@]} -gt 0 ]; then
  C=$(sub -p ice-cpu --qos=coe-ice -c 2 --mem=8G -t 0:30:00 -o logs/compare-%j.out -J wf-compare \
        --dependency=afterany:$(IFS=:; echo "${EV[*]}") \
        --wrap "export PATH=\$HOME/scratch/envs/wildfire/bin:\$PATH; cd model; python compare.py --data \$HOME/scratch/wildfire/my")
  echo "comparison report: $C  ->  $RUNS/compare.md"
fi
echo; echo "Training runs first; the evaluation period starts by itself when all of it is done. Progress: bash scripts/status.sh"
