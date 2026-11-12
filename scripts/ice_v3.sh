#!/usr/bin/env bash
# v3: retrain the main models on leak-free inputs (INPUT_VERSION 5: no satellite pass, GOES check or
# suppression map from after the forecast start; perimeter times corrected), evaluate them, and refit the router with split-half validation. Every score before this round
# was optimistic (25% of windows saw a later pass), so these are the numbers to report.
#
#   bash scripts/ice_v3.sh          seed 0: unet2_v3, unet3_time_v3, unet3_full_v3, hybrid_v3, plus unet3_live_v3
#                                   (website inputs only), unet3_fc_v3 (forecast weather), unet3_ops_v3
#                                   (forecasts from an ageing perimeter) and unet_h21_v3
#   bash scripts/ice_v3.sh 1 2      extra seeds of the same models (…_v3s1, …_v3s2) for error bars
# Safe to rerun: trained models, finished evaluations and queued jobs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
RUNS="$HOME/scratch/wildfire/my/runs"
q() { local out; for k in 1 2 3; do out=$(squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null) && { [ -n "$out" ] && echo "$out" | sort -n; return 0; }; sleep 3; done; }
sub() { local j; for k in 1 2 3; do j=$(sbatch --parsable "$@") && [ -n "$j" ] && { echo "$j"; return 0; }; echo "  (sbatch failed, retrying)" >&2; sleep 5; done
        echo "  !! could not submit: sbatch $*" >&2; return 1; }
BAD="--exclude=atl1-1-03-014-16-0"        # node whose GPU failed on 2026-11-11

declare -A TJ
# train NAME MODE CHAIN TIME EXTRA: CHAIN jobs in a row (each resumes from the last checkpoint)
train() { local name="$1" mode="$2" chain="$3" t="$4" extra="$5"
  local ids; ids=$(q "wf-$name" | tr '\n' ':'); ids="${ids%:}"
  if [ -n "$ids" ]; then TJ[$name]="$ids"; echo "$name: already queued/running ($ids)"; return; fi
  [ -f "$RUNS/$name.pt" ] && { echo "$name: already trained"; return; }
  local prev="" j all=""
  for k in $(seq 1 "$chain"); do
    j=$(NAME=$name MODE=$mode EXTRA="$extra" sub -J "wf-$name" -t "$t" $BAD ${prev:+--dependency=afterany:$prev} scripts/ice_train.sbatch) || return
    prev=$j; all="${all:+$all:}$j"
  done
  TJ[$name]="$all"; echo "$name: queued ($all)"; }

EV=(); MEMBERS=()
ev() { local name="$1" out="$2" extra="${3:-}"
  [ -f "$RUNS/eval_$out.json" ] && [ -f "$RUNS/preds_$out.npz" ] && { echo "eval $out: already done"; return; }
  local dep=${TJ[$name]:-} j
  j=$(PULL=1 MODELS=$name OUT=$out EXTRA="$extra" sub -J "wf-eval-$out" -t 4:00:00 $BAD ${dep:+--dependency=afterany:$dep} scripts/ice_eval.sbatch) || return
  EV+=("$j"); echo "evaluation $out: $j"; }

SEEDS=("${@:-0}")
for s in "${SEEDS[@]}"; do
  sfx=$([ "$s" = 0 ] && echo "_v3" || echo "_v3s$s")
  seed="--seed $s"
  train unet2$sfx      unet2  1 8:00:00  "$seed"
  train unet3_time$sfx unet3  2 8:00:00  "--width 64 --u3-time 1 --u3-attn 0 $seed"
  train unet3_full$sfx unet3  2 8:00:00  "--width 64 --u3-time 1 --u3-attn 1 $seed"
  train hybrid$sfx     hybrid 2 16:00:00 "$seed"
  for m in unet2 unet3_time unet3_full hybrid; do ev "$m$sfx" "$m${sfx}_eval"; done
  if [ "$s" = 0 ]; then
    # like-for-like rerun of the head-to-head with Samuel's TS-SatFire fires (val 2020, test 2021)
    # the model the website can actually feed: only inputs that exist live (no GOES, Tessera, human layers,
    # suppression). Its score next to unet3_time_v3 shows what those inputs are worth.
    train unet3_live_v3 unet3 2 8:00:00 "--width 64 --u3-time 1 --u3-attn 0 --drop goes tessera human suppression"
    ev unet3_live_v3 unet3_live_v3_eval
    # same model given only forecast weather (what a forecaster has at the start), not the observed weather
    train unet3_fc_v3 unet3 2 8:00:00 "--width 64 --u3-time 1 --u3-attn 0 --weather forecast"
    ev unet3_fc_v3 unet3_fc_v3_eval "--weather forecast"
    # trained for the live situation: the forecast starts hours after the perimeter was mapped. Scored the
    # normal way and with every forecast starting 12 h late; unet3_time_v3 gets the same 12 h test, which
    # is how the website uses a normally trained model today.
    train unet3_ops_v3 unet3 2 8:00:00 "--width 64 --u3-time 1 --u3-attn 0 --origin-lag 1"
    ev unet3_ops_v3 unet3_ops_v3_eval
    ev unet3_ops_v3 unet3_ops_v3_lag12 "--lag-h 12"
    ev unet3_time_v3 unet3_time_v3_lag12 "--lag-h 12"
    train unet_h21_v3 unet 1 6:00:00 "--val-years 2020 --test-years 2021"
    ev unet_h21_v3 unet_h21_v3_vs_tssatfire "--val-years 2020 --test-years 2021 --fires ../benchmarks/tssatfire_2021_test_match.csv"
  fi
  MEMBERS=(unet2${sfx}_eval unet3_time${sfx}_eval unet3_full${sfx}_eval hybrid${sfx}_eval)
  DEPS=$(printf '%s\n' "${EV[@]}" | paste -sd:)
  H2H=$([ "$s" = 0 ] && echo "--h2h unet_h21_v3_vs_tssatfire" || echo "")
  C=$(sub -p ice-cpu --qos=coe-ice -c 4 --mem=64G -t 3:00:00 -o logs/combine-%j.out -J wf-combine$sfx ${DEPS:+--dependency=afterany:$DEPS} \
        --wrap "export PATH=\$HOME/scratch/envs/wildfire/bin:\$PATH; cd model; python combine.py --data \$HOME/scratch/wildfire/my --members ${MEMBERS[*]} $H2H --out combine$sfx")
  echo "router refit (seed $s): $C  ->  $RUNS/combine$sfx.md"
  EV=()
done
echo; echo "Progress: bash scripts/status.sh    Results: runs/combine_v3.md (seeds: combine_v3s1.md, combine_v3s2.md)"
