#!/usr/bin/env bash
# Round 2: cleaned data (labels v2) + new physics, with detailed "before" evaluations.
#   1. snapshot hybrid_v1 as it is now (best weights so far) -> hybrid_v1_before, then restart hybrid_v1
#      from scratch on the cleaned data (so hybrid vs hybrid2 differ only in the physics)
#   2. make sure physics2_v1 and hybrid2_v1 (crown fire + spotting) are queued
#   3. evaluate the "before" models: physics_v1, unet_v1, hybrid_v1_before
#   4. evaluate hybrid_v1, hybrid2_v1, physics2_v1 automatically when their training ends
#   5. write runs/compare.md (all evaluations side by side) after each batch of evaluations
#   bash scripts/ice_round2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
RUNS="$HOME/scratch/wildfire/my/runs"
q() { squeue -h -u "$USER" -n "$1" -o %i | sort -n; }
sub() { sbatch --parsable "$@"; }

# 1. snapshot + restart hybrid_v1
if [ -f "$RUNS/hybrid_v1.ckpt" ] && [ ! -f "$RUNS/hybrid_v1_before.ckpt" ]; then
  OLD=$(q wf-hybrid | tr '\n' ' ')
  [ -n "$OLD" ] && scancel $OLD && echo "cancelled old hybrid_v1 jobs: $OLD" && sleep 5
  cp "$RUNS/hybrid_v1.ckpt" "$RUNS/hybrid_v1_before.ckpt"
  rm -f "$RUNS/hybrid_v1.ckpt" "$RUNS/hybrid_v1.json" "$RUNS/hybrid_v1.pt"
  echo "saved the current hybrid_v1 as hybrid_v1_before"
fi
if [ -z "$(q wf-hybrid)" ] && [ ! -f "$RUNS/hybrid_v1.pt" ]; then
  H1=$(NAME=hybrid_v1 MODE=hybrid sub -J wf-hybrid scripts/ice_train.sbatch)
  H2=$(NAME=hybrid_v1 MODE=hybrid sub -J wf-hybrid --dependency=afterany:$H1 scripts/ice_train.sbatch)
  echo "hybrid_v1 (cleaned data): $H1 $H2"
fi
# 2. new physics
if [ -z "$(q wf-phys2)" ] && [ ! -f "$RUNS/physics2_v1.pt" ]; then
  P=$(NAME=physics2_v1 MODE=physics2 EXTRA="--epochs 4 --steps 200 --lr 0.02 --eval-every 2" sub -J wf-phys2 -t 4:00:00 scripts/ice_train.sbatch)
  echo "physics2_v1: $P"
fi
if [ -z "$(q wf-hybrid2)" ] && [ ! -f "$RUNS/hybrid2_v1.pt" ]; then
  A=$(NAME=hybrid2_v1 MODE=hybrid2 sub -J wf-hybrid2 scripts/ice_train.sbatch)
  B=$(NAME=hybrid2_v1 MODE=hybrid2 sub -J wf-hybrid2 --dependency=afterany:$A scripts/ice_train.sbatch)
  echo "hybrid2_v1: $A $B"
fi
# 3. "before" evaluations (all scored on the cleaned labels, same cells as everything after)
EV=()
ev() { local models="$1" out="$2" dep="${3:-}"
  [ -f "$RUNS/eval_$out.json" ] && return
  [ -n "$(q wf-eval-$out)" ] && return                                  # already queued
  [ "$out" = physics2_eval ] && [ -n "$(q wf-eval-p2)" ] && return       # queued earlier by ice_v2.sh
  local j; j=$(MODELS="$models" OUT="$out" sub -J wf-eval-$out -t 4:00:00 ${dep:+--dependency=afterany:$dep} scripts/ice_eval.sbatch)
  EV+=("$j"); echo "evaluate $models -> eval_$out ($j${dep:+, after $dep})"; }
OLDPHYS=$(squeue -h -u "$USER" -n wf-eval -o %i | tr '\n' ' ')
[ -n "$OLDPHYS" ] && scancel $OLDPHYS && echo "replaced the earlier physics evaluation ($OLDPHYS)"
ev "physics_v1:physics" physics_eval
ev "unet_v1" unet_v1_eval
ev "hybrid_v1_before:hybrid" hybrid_v1_before_eval
# 4. "after" evaluations, when training ends
last() { q "$1" | tail -1; }
[ -n "$(last wf-phys2)" ] && ev "physics2_v1" physics2_eval "$(last wf-phys2)"
[ -n "$(last wf-hybrid)" ] && ev "hybrid_v1" hybrid_v1_eval "$(last wf-hybrid)"
[ -n "$(last wf-hybrid2)" ] && ev "hybrid2_v1" hybrid2_eval "$(last wf-hybrid2)"
# 5. comparison report after every evaluation finishes (cheap, CPU)
for j in "${EV[@]}"; do
  sbatch -p ice-cpu --qos=coe-ice -c 2 --mem=8G -t 0:20:00 -o logs/compare-%j.out --dependency=afterany:$j -J wf-compare \
    --wrap "export PATH=\$HOME/scratch/envs/wildfire/bin:\$PATH; cd model; python compare.py --data \$HOME/scratch/wildfire/my" >/dev/null
done
echo "done. 'bash scripts/status.sh' for progress; the comparison is in $RUNS/compare.md (also: tail -60 logs/compare-*.out)"
