#!/usr/bin/env bash
# Submit the first round of training on ICE:
#   1. a short smoke test on the real data (one GPU, ~10-20 min)
#   2. if it passes: physics-only baseline, plain U-Net, hybrid model (one GPU each),
#      each U-Net / hybrid with two automatic follow-up jobs for the 16 h limit (they resume)
#   bash scripts/ice_train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
sub() { sbatch --parsable "$@"; }
S=$(NAME=smoke MODE=hybrid EXTRA="--max-fires 60 --epochs 1 --steps 20 --eval-every 1 --batch 4" sub -J wf-smoke -t 1:00:00 scripts/ice_train.sbatch)
P=$(NAME=physics_v1 MODE=physics sub -J wf-physics -t 4:00:00 --dependency=afterok:$S scripts/ice_train.sbatch)
U1=$(NAME=unet_v1 MODE=unet sub -J wf-unet --dependency=afterok:$S scripts/ice_train.sbatch)
U2=$(NAME=unet_v1 MODE=unet sub -J wf-unet --dependency=afterany:$U1 scripts/ice_train.sbatch)
U3=$(NAME=unet_v1 MODE=unet sub -J wf-unet --dependency=afterany:$U2 scripts/ice_train.sbatch)
H1=$(NAME=hybrid_v1 MODE=hybrid sub -J wf-hybrid --dependency=afterok:$S scripts/ice_train.sbatch)
H2=$(NAME=hybrid_v1 MODE=hybrid sub -J wf-hybrid --dependency=afterany:$H1 scripts/ice_train.sbatch)
H3=$(NAME=hybrid_v1 MODE=hybrid sub -J wf-hybrid --dependency=afterany:$H2 scripts/ice_train.sbatch)
echo "smoke $S | physics $P | unet $U1 $U2 $U3 | hybrid $H1 $H2 $H3"
echo "If the smoke test fails, the others are cancelled automatically by Slurm (DependencyNeverSatisfied):"
echo "  squeue -u \$USER ; scancel -u \$USER  (then send the smoke log)"
