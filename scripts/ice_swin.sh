#!/usr/bin/env bash
# Swin transformer model (nets.SwinUNet): same inputs, fire history and training recipe as unet3_full_v3,
# with Swin window-attention stages at the two coarse levels in place of convolutions and the whole-map
# attention bottleneck. Compare eval_unet3_swin_v3_eval.json with unet3_full_v3 (AP 0.252, 0.226-0.279)
# and unet3_time_v3 (0.229). Then the router is refitted with it as a fifth member.
#   bash scripts/ice_swin.sh            seed 0
#   bash scripts/ice_swin.sh 1 2        more seeds (unet3_swin_v3s1, ...)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
RUNS="$HOME/scratch/wildfire/my/runs"
BAD="--exclude=atl1-1-03-014-16-0"
for s in "${@:-0}"; do
  sfx=$([ "$s" = 0 ] && echo "_v3" || echo "_v3s$s")
  N=unet3_swin$sfx
  prev=""; all=""
  if [ ! -f "$RUNS/$N.pt" ]; then
    for k in 1 2; do
      j=$(NAME=$N MODE=unet3 EXTRA="--width 64 --u3-time 1 --u3-swin 1 --seed $s" \
          sbatch --parsable -J wf-$N -t 8:00:00 $BAD ${prev:+--dependency=afterany:$prev} scripts/ice_train.sbatch) || exit 1
      prev=$j; all="${all:+$all:}$j"
    done
    echo "$N: queued ($all)"
  fi
  e=$(PULL=1 MODELS=$N OUT=${N}_eval sbatch --parsable -J wf-eval-$N -t 4:00:00 $BAD ${all:+--dependency=afterany:$all} scripts/ice_eval.sbatch)
  echo "evaluation ${N}_eval: $e"
  M="unet2${sfx}_eval unet3_time${sfx}_eval unet3_full${sfx}_eval hybrid${sfx}_eval ${N}_eval"
  c=$(sbatch --parsable -p ice-cpu --qos=coe-ice -c 4 --mem=64G -t 3:00:00 -o logs/combine-%j.out -J wf-combine-swin$sfx --dependency=afterany:$e \
        --wrap "export PATH=\$HOME/scratch/envs/wildfire/bin:\$PATH; cd model; python combine.py --data \$HOME/scratch/wildfire/my --members $M --out combine${sfx}_swin")
  echo "router with Swin as a fifth model: $c  ->  $RUNS/combine${sfx}_swin.md"
done
