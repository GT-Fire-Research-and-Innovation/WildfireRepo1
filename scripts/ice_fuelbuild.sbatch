#!/bin/bash
# GT PACE-ICE: the fuel-moisture inputs for every fire already built (CPU only, new files only):
#   hrrr2.npz      air temperature, dew point, sunshine and rain every 3 h   (stage fuelwx)
#   green.npz      satellite greenness, a composite every 8 days             (stage green)
# (stage landfire2, the fuel map of the fire's own year, is NOT run: LANDFIRE's service only has the 2016,
#  2022, 2023, 2024 and 2025 fuel maps, checked 2026-11-12, so there is nothing older to fetch.
#  STAGES=fuelwx,green,landfire2 runs it anyway.)
# Nothing existing is rewritten, so models training on the current files are not affected.
# Resumable: built files are skipped, so after a timeout just submit again.
#   sbatch scripts/ice_fuelbuild.sbatch                 everything
#   FIRES="<fire_id> <fire_id>" sbatch scripts/ice_fuelbuild.sbatch     a few fires first (recommended once)
#SBATCH -J wf-fuelbuild
#SBATCH -p ice-cpu
#SBATCH --qos=coe-ice
#SBATCH -N 1 --cpus-per-task=16 --mem=96G
#SBATCH -t 16:00:00
#SBATCH -o logs/fuelbuild-%j.out
module purge
export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
set -a; [ -f "$HOME/.config/wildfire/env" ] && . "$HOME/.config/wildfire/env"; set +a
DATA="$HOME/scratch/wildfire"
cd "$SLURM_SUBMIT_DIR/pipeline"
echo "code: $(git -C "$SLURM_SUBMIT_DIR" log --oneline -1)"
python build_multiyear.py --catalog "$DATA/catalog" --out "$DATA/my" --years 2014-2025 \
       --stages "${STAGES:-fuelwx,green}" --workers 6 ${FIRES:+--fires $FIRES} 2>&1 | tee -a "$DATA/fuelbuild.log"
cd "$DATA/my/fires" && echo "built so far: hrrr2 $(ls */hrrr2.npz 2>/dev/null | wc -l), green $(ls */green.npz 2>/dev/null | wc -l), landfire2 $(ls */landfire2.npz 2>/dev/null | wc -l) of $(ls -d */ | wc -l) fires; errors: $(ls */green.error.txt */landfire2.error.txt 2>/dev/null | wc -l)"
