#!/usr/bin/env bash
# Extra hybrid seeds for the ensemble (same settings as hybrid_v1, different random seed).
# Each seed gets the usual three chained 16 h jobs. Runs alongside the current jobs.
#   bash scripts/ice_seeds.sh 1 2        # hybrid_s1, hybrid_s2
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
for s in "${@:-1 2}"; do
  A=$(NAME=hybrid_s$s MODE=hybrid EXTRA="--seed $s" sbatch --parsable -J wf-hyb-s$s scripts/ice_train.sbatch)
  B=$(NAME=hybrid_s$s MODE=hybrid EXTRA="--seed $s" sbatch --parsable -J wf-hyb-s$s --dependency=afterany:$A scripts/ice_train.sbatch)
  C=$(NAME=hybrid_s$s MODE=hybrid EXTRA="--seed $s" sbatch --parsable -J wf-hyb-s$s --dependency=afterany:$B scripts/ice_train.sbatch)
  echo "hybrid_s$s: $A $B $C"
done
