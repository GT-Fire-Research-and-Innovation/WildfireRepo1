#!/bin/bash
# GT PACE-ICE: dataset build on a CPU node (compute nodes have internet). 16 h job limit:
# resubmit after a timeout and every stage continues where it stopped.
#   sbatch scripts/ice_build.sbatch
#SBATCH -J wf-build
#SBATCH -p ice-cpu
#SBATCH --qos=coe-ice
#SBATCH -N 1 --cpus-per-task=16 --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/build-%j.out
module load anaconda3
conda activate "$HOME/scratch/envs/wildfire"
cd "$SLURM_SUBMIT_DIR"
bash scripts/build_all.sh "$HOME/scratch/wildfire" 12
