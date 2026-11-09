#!/bin/bash
# Georgia Tech PACE (Phoenix): dataset build on CPU nodes. Submit: sbatch scripts/pace_build.sbatch
# Set the account to your PI's charge account (pace-whoami / pace-quota show it).
#SBATCH -J wildfire-build
#SBATCH -A gts-CHANGE_ME
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=16 --mem=64G
#SBATCH -t 16:00:00
#SBATCH -q inferno
#SBATCH -o build-%j.out
module load anaconda3
conda activate wildfire
cd "$SLURM_SUBMIT_DIR"
# FIRMS_MAP_KEY and LANDFIRE_EMAIL come from ~/.config/wildfire/env (never committed)
bash scripts/build_all.sh "${DATA:-$HOME/scratch/wildfire}" 12
# Hit the wall-clock limit? Just resubmit: every stage skips what is already built.
