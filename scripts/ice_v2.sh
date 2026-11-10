#!/usr/bin/env bash
# Physics v2 (crown fire + spotting) on ICE.
#   physics2_v1   physics only, learns 5 scalars (burning time, softness, crown weight, ember alpha/beta); ~1-2 h
#                 then evaluated (raw / calibrated / baselines / breakdowns) to compare with physics_v1
#   hybrid2_v1    the hybrid with crown fire and spotting (three chained 16 h jobs, like hybrid_v1)
#   bash scripts/ice_v2.sh            # both
#   bash scripts/ice_v2.sh physics    # only the physics-only check (cheap)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
P=$(NAME=physics2_v1 MODE=physics2 EXTRA="--epochs 4 --steps 200 --lr 0.02 --eval-every 2" \
      sbatch --parsable -J wf-phys2 -t 4:00:00 scripts/ice_train.sbatch)
E=$(MODELS="physics2_v1" OUT=physics2_eval sbatch --parsable -J wf-eval-p2 -t 4:00:00 --dependency=afterok:$P scripts/ice_eval.sbatch)
echo "physics2_v1: train $P, evaluate $E"
if [ "${1:-all}" != "physics" ]; then
  A=$(NAME=hybrid2_v1 MODE=hybrid2 sbatch --parsable -J wf-hybrid2 scripts/ice_train.sbatch)
  B=$(NAME=hybrid2_v1 MODE=hybrid2 sbatch --parsable -J wf-hybrid2 --dependency=afterany:$A scripts/ice_train.sbatch)
  C=$(NAME=hybrid2_v1 MODE=hybrid2 sbatch --parsable -J wf-hybrid2 --dependency=afterany:$B scripts/ice_train.sbatch)
  echo "hybrid2_v1: $A $B $C"
fi
