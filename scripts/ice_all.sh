#!/usr/bin/env bash
# One model with every gain confirmed so far, on GT PACE-ICE:
#   fuel moisture + greenness (+0.035 AP over three seeds), training for late starts (the largest gain measured),
#   terrain-adjusted wind (WindNinja), and fuel shares per cell in place of one picked pixel.
#   bash scripts/ice_all.sh build      the two inputs the model needs for EVERY fire: terrain wind and fuel shares
#   bash scripts/ice_all.sh status     how many fires have each input
#   bash scripts/ice_all.sh train      train and score unet3_all_v4 (refuses while inputs are missing; FORCE=1 overrides)
#   bash scripts/ice_all.sh train 1 2  more seeds
#   bash scripts/ice_all.sh results    scores next to the models it should beat
# Settings:
#   WIND=1 (default) wind solved on the grid's terrain; WIND=75 the 75 m version; WIND=0 leaves terrain wind out
#   FRAC=1 (default) fuel shares; FRAC=0 the picked pixel
#   WF_DATA=<dataset folder>   default ~/scratch/wildfire/my; the model's name gets TAG (e.g. TAG=_small) so runs
#                              on another dataset do not overwrite these
# Compare with: unet3_fm_v3 (fuel moisture only, 0.261 mean of 3 seeds), unet3_ops_v3_lag12 (late start, 0.324).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export WF_DATA="${WF_DATA:-$HOME/scratch/wildfire/my}"
RUNS="$WF_DATA/runs"
BAD="--exclude=atl1-1-03-014-16-0"
WIND="${WIND:-1}"; FRAC="${FRAC:-1}"; TAG="${TAG:-}"
WFILE=$([ "$WIND" = 1 ] && echo terrain_wind || echo "terrain_wind$WIND")
step="${1:-status}"; shift || true

count() { ls "$WF_DATA"/fires/*/"$1".npz 2>/dev/null | wc -l; }
status() {
  local n; n=$(ls -d "$WF_DATA"/fires/*/ 2>/dev/null | wc -l)
  echo "fires: $n   fuels (landfire): $(count landfire)   fuel shares: $(count fuelfrac)   terrain wind ($WFILE): $(count "$WFILE")   fuel moisture weather: $(count hrrr2)   greenness: $(count green)"
}

case "$step" in
build)
  if [ "$WIND" != 0 ]; then YEARS=2014-2025 MESH=$([ "$WIND" = 1 ] || echo "$WIND") bash scripts/ice_wind.sh build; fi
  if [ "$FRAC" = 1 ]; then
    j=$(STAGES=fuelfrac sbatch --parsable -J wf-fuelfrac scripts/ice_fuelbuild.sbatch) && \
      echo "fuel shares (30 m LANDFIRE, one request per fire, about a day): job $j  (log: logs/fuelbuild-$j.out; resumable: submit again after a timeout)"
  fi
  ;;
status) status ;;
train)
  status
  if [ "${FORCE:-0}" != 1 ]; then
    need=$(count landfire)
    [ "$FRAC" = 1 ] && [ "$(count fuelfrac)" -lt $(( need * 97 / 100 )) ] && { echo "!! fuel shares are not built for every fire yet (bash scripts/ice_all.sh build; FORCE=1 to train anyway)"; exit 1; }
    [ "$WIND" != 0 ] && [ "$(count "$WFILE")" -lt $(( $(ls -d "$WF_DATA"/fires/*/ | wc -l) * 97 / 100 )) ] && { echo "!! terrain wind ($WFILE) is not built for every fire yet (FORCE=1 to train anyway)"; exit 1; }
  fi
  for s in "${@:-0}"; do
    N="unet3_all_v4${TAG}$([ "$s" = 0 ] || echo "s$s")"
    X="--width 64 --u3-time 1 --u3-attn 0 --fuel-moisture 1 --origin-lag 1 --wind-terrain $WIND --fuel-frac $FRAC --seed $s"
    prev=""; all=""
    if [ ! -f "$RUNS/$N.pt" ]; then
      for k in 1 2; do
        j=$(NAME=$N MODE=unet3 EXTRA="$X" sbatch --parsable -J "wf-$N" -t 8:00:00 $BAD ${prev:+--dependency=afterany:$prev} scripts/ice_train.sbatch) || exit 1
        prev=$j; all="${all:+$all:}$j"
      done
      echo "$N: queued ($all)   [$X]"
    fi
    e1=$(MODELS=$N OUT=${N}_eval sbatch --parsable -J "wf-eval-$N" -t 4:00:00 $BAD ${all:+--dependency=afterany:$all} scripts/ice_eval.sbatch)
    e2=$(MODELS=$N OUT=${N}_lag12 EXTRA="--lag-h 12" sbatch --parsable -J "wf-eval-${N}_lag12" -t 4:00:00 $BAD ${all:+--dependency=afterany:$all} scripts/ice_eval.sbatch)
    echo "  evaluations: ${N}_eval $e1, ${N}_lag12 $e2"
  done
  ;;
results)
  export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
  have=""
  for n in $(cd "$RUNS" && ls eval_unet3_all_v4*.json 2>/dev/null | sed 's/^eval_//; s/\.json$//') \
           unet3_fm_v3_eval unet3_fm_v3s1_eval unet3_fm_v3s2_eval unet3_time_v3_eval unet3_ops_v3_lag12 unet3_site_v3_lag12; do
    [ -f "$RUNS/eval_$n.json" ] && have="$have $n"
  done
  grep -h "fuel shares:\|terrain wind:" logs/eval-wf-eval-unet3_all_v4*.out 2>/dev/null | sort | uniq -c
  (cd model && python compare.py --data "$WF_DATA" $have 2>&1 | awk '/^## /{p = ($0 ~ /Test years|HRRR wind|growth|time gap/)} p')
  ;;
*) echo "usage: bash scripts/ice_all.sh build | status | train [seeds] | results"; exit 1 ;;
esac
