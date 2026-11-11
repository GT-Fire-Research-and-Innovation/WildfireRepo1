#!/usr/bin/env bash
# unet3 round: four runs that separate what each upgrade adds, then evaluation, then the router.
#   unet3_wide   width 64, no history, no attention       (bigger network alone)
#   unet3_time   width 64 + 6-day history with temporal attention + gap weather in 4 segments
#   unet3_attn   width 64 + transformer bottleneck
#   unet3_full   width 64 + both
# Each is compared with unet3_wide, so time and attention are measured on top of the wider network.
# Evaluations start by themselves after training; then combine.py refits the router with the new models.
#   bash scripts/ice_unet3.sh        (safe to rerun: nothing is queued twice)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
RUNS="$HOME/scratch/wildfire/my/runs"
q() { local out; for k in 1 2 3; do out=$(squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null) && { [ -n "$out" ] && echo "$out" | sort -n; return 0; }; sleep 3; done; }
sub() { local j; for k in 1 2 3; do j=$(sbatch --parsable "$@") && [ -n "$j" ] && { echo "$j"; return 0; }; echo "  (sbatch failed, retrying)" >&2; sleep 5; done
        echo "  !! could not submit: sbatch $*" >&2; return 1; }
BAD="--exclude=atl1-1-03-014-16-0"        # node whose GPU failed on 2026-11-11

declare -A TJ
train() { local name="$1" extra="$2"
  local ids; ids=$(q "wf-$name" | tr '\n' ' ')
  if [ -n "${ids// /}" ]; then TJ[$name]="${ids// /:}"; TJ[$name]="${TJ[$name]%:}"; echo "$name: already queued/running ($ids)"; return; fi
  [ -f "$RUNS/$name.pt" ] && { echo "$name: already trained"; return; }
  local a b
  a=$(NAME=$name MODE=unet3 EXTRA="--width 64 $extra" sub -J "wf-$name" -t 8:00:00 $BAD scripts/ice_train.sbatch)
  b=$(NAME=$name MODE=unet3 EXTRA="--width 64 $extra" sub -J "wf-$name" -t 8:00:00 $BAD --dependency=afterany:$a scripts/ice_train.sbatch)
  TJ[$name]="$a:$b"; echo "$name: queued ($a, then $b if it needs more time)"; }
train unet3_wide "--u3-time 0 --u3-attn 0"
train unet3_time "--u3-time 1 --u3-attn 0"
train unet3_attn "--u3-time 0 --u3-attn 1"
train unet3_full "--u3-time 1 --u3-attn 1"

EV=()
for name in unet3_wide unet3_time unet3_attn unet3_full; do
  [ -f "$RUNS/eval_${name}_eval.json" ] && [ -f "$RUNS/preds_${name}_eval.npz" ] && { echo "eval $name: already done"; EV+=("done"); continue; }
  dep=${TJ[$name]:-}
  j=$(MODELS=$name OUT=${name}_eval sub -J "wf-eval-$name" -t 3:00:00 $BAD ${dep:+--dependency=afterany:$dep} scripts/ice_eval.sbatch)
  EV+=("$j"); echo "evaluation ${name}_eval: $j"
done
DEPS=$(printf '%s\n' "${EV[@]}" | grep -v done | paste -sd:)
C=$(sub -p ice-cpu --qos=coe-ice -c 4 --mem=64G -t 2:00:00 -o logs/combine-%j.out -J wf-combine ${DEPS:+--dependency=afterany:$DEPS} \
      --wrap "export PATH=\$HOME/scratch/envs/wildfire/bin:\$PATH; cd model; python combine.py --data \$HOME/scratch/wildfire/my --members unet2_eval hybrid_v1_eval unet3_wide_eval unet3_time_eval unet3_attn_eval unet3_full_eval --out combine_unet3")
echo "router refit: $C  ->  $RUNS/combine_unet3.md"
echo; echo "Progress: bash scripts/status.sh   (unet3 epochs are slower than unet2: expect several hours per run)"
