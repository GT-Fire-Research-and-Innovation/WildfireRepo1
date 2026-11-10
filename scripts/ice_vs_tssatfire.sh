#!/usr/bin/env bash
# Head-to-head with the group's TS-SatFire SwinUNETR on its 2021 test fires (19 of the 24 are in our data).
# 1) U-Net trained with 2021 held out (validation 2020), ~2-3 h on one GPU
# 2) evaluate it on the 19 matched fires (TS-SatFire per-window F1/IoU and our pooled metrics)
#   bash scripts/ice_vs_tssatfire.sh            (add "hybrid" as an argument to also run the hybrid, ~8 h)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
for MODE in unet ${1:-}; do
  T=$(NAME=${MODE}_h21 MODE=$MODE EXTRA="--val-years 2020 --test-years 2021" sbatch --parsable -J wf-$MODE-h21 \
        -t $([ "$MODE" = unet ] && echo 4:00:00 || echo 12:00:00) scripts/ice_train.sbatch)
  E=$(MODELS="${MODE}_h21" OUT=${MODE}_h21_vs_tssatfire EXTRA="--val-years 2020 --test-years 2021 --fires ../benchmarks/tssatfire_2021_test_match.csv" \
        sbatch --parsable -J wf-eval-ts -t 4:00:00 --dependency=afterok:$T scripts/ice_eval.sbatch)
  echo "${MODE}_h21: train $T, evaluate $E"
done
