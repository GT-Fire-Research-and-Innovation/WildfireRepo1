#!/bin/bash
# Georgia Tech PACE / AI Makerspace: one training run on one GPU. Checkpoints every epoch, so a
# resubmit resumes. Submit: NAME=hybrid_v1 sbatch scripts/pace_train.sbatch
#SBATCH -J wildfire-train
#SBATCH -A gts-CHANGE_ME
#SBATCH -N 1 --gres=gpu:1 --cpus-per-task=8 --mem=64G
#SBATCH -t 12:00:00
#SBATCH -q inferno
#SBATCH -o train-%j.out
module load anaconda3
conda activate wildfire
cd "$SLURM_SUBMIT_DIR/model"
python train_hybrid.py --data "${DATA:-$HOME/scratch/wildfire}/my" --name "${NAME:-hybrid}" --device cuda
