#!/bin/bash
# GT PACE-ICE: the 75 m dataset (same fires, same stages, LANDFIRE at its native 30 m summarised per cell).
# CPU only, so it does not touch the GPU budget. Resumable: every stage skips what is built, so after a
# 16 h timeout just submit again (or queue a few in a row with --dependency=afterany).
#   sbatch scripts/ice_build75.sbatch
#   J=$(sbatch --parsable scripts/ice_build75.sbatch); for i in 1 2 3; do J=$(sbatch --parsable --dependency=afterany:$J scripts/ice_build75.sbatch); done
# Needs ~185 GB more scratch. HRRR files are shared with the 150 m build through its cache.
#SBATCH -J wf-build75
#SBATCH -p ice-cpu
#SBATCH --qos=coe-ice
#SBATCH -N 1 --cpus-per-task=16 --mem=128G
#SBATCH -t 16:00:00
#SBATCH -o logs/build75-%j.out
module purge
export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
set -a; [ -f "$HOME/.config/wildfire/env" ] && . "$HOME/.config/wildfire/env"; set +a
DATA="$HOME/scratch/wildfire"
mkdir -p "$DATA/my75"
[ -e "$DATA/my75/cache" ] || ln -s "$DATA/my/cache" "$DATA/my75/cache"      # reuse downloaded HRRR files
echo "scratch in use: $(du -sh "$DATA" 2>/dev/null | cut -f1)"
cd "$SLURM_SUBMIT_DIR/pipeline"
python build_multiyear.py --catalog "$DATA/catalog" --out "$DATA/my75" --res 75 --years 2014-2025 \
       --stages labels,history,viirs,terrain,weather,landfire,tessera,suppression,hrrr,goes,human \
       --workers 6 2>&1 | tee -a "$DATA/build75.log"
python check_dataset.py --data "$DATA/my75" --pairs 4 2>&1 | tail -30
