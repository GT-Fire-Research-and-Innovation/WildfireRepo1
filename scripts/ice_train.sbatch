#!/bin/bash
# GT PACE-ICE: one training run on one H100 or H200 (coe-gpu partition, coe-ice QOS). Checkpoints every
# epoch; resubmit to continue after the 16 h limit.
#   NAME=hybrid_v1 sbatch scripts/ice_train.sbatch
#SBATCH -J wf-train
#SBATCH -p coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:1                 # coe-gpu = H100 or H200, whichever frees up first
#SBATCH -N 1 --cpus-per-task=8 --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/train-%j.out
module purge
export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
cd "$SLURM_SUBMIT_DIR/model"
python train_hybrid.py --data "$HOME/scratch/wildfire/my" --name "${NAME:-hybrid}" --device cuda
