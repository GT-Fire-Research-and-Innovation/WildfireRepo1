#!/usr/bin/env bash
# The model for the website: trained for what the site actually does.
#   - forecasts start hours after the perimeter was mapped (--origin-lag; v3: 0.190 vs 0.117 AP on
#     not-yet-detected cells when the perimeter is 12 h old)
#   - only the inputs the site has (no GOES, Tessera, roads/buildings, suppression)
#   - the attention network, the best single model in v3 (0.252 vs 0.229)
# Scored the normal way and with every forecast starting 12 h late (compare: unet3_ops_v3, unet3_live_v3).
#   bash scripts/ice_site.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
RUNS="$HOME/scratch/wildfire/my/runs"
BAD="--exclude=atl1-1-03-014-16-0"
N=unet3_site_v3
prev=""; all=""
if [ ! -f "$RUNS/$N.pt" ]; then
  for k in 1 2; do
    j=$(NAME=$N MODE=unet3 EXTRA="--width 64 --u3-time 1 --u3-attn 1 --origin-lag 1 --drop goes tessera human suppression" \
        sbatch --parsable -J wf-$N -t 8:00:00 $BAD ${prev:+--dependency=afterany:$prev} scripts/ice_train.sbatch) || exit 1
    prev=$j; all="${all:+$all:}$j"
  done
  echo "$N: queued ($all)"
fi
for v in "eval:" "lag12:--lag-h 12"; do
  out="${N}_${v%%:*}"; extra="${v#*:}"
  e=$(PULL=1 MODELS=$N OUT=$out EXTRA="$extra" sbatch --parsable -J wf-eval-$out -t 4:00:00 $BAD ${all:+--dependency=afterany:$all} scripts/ice_eval.sbatch)
  echo "evaluation $out: $e"
done
