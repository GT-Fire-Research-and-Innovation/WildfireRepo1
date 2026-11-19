#!/bin/bash
# GT PACE-ICE: one training run on one H100 or H200 (coe-gpu partition, coe-ice QOS).
# Checkpoints every epoch; resubmitting the same NAME continues it after the 16 h limit.
#   NAME=hybrid_v1 MODE=hybrid sbatch scripts/ice_train.sbatch
#   SCRIPT=train_hazard.py NAME=hazard_v1 EXTRA="--width 64" sbatch scripts/ice_train.sbatch
#   NAME=smoke MODE=hybrid EXTRA="--max-fires 40 --epochs 1 --steps 20" sbatch scripts/ice_train.sbatch
#SBATCH -J wf-train
#SBATCH -p coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:1                 # coe-gpu = H100 or H200, whichever frees up first
#SBATCH -N 1 --cpus-per-task=8 --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/train-%x-%j.out
module purge
export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
cd "$SLURM_SUBMIT_DIR/model"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SCRIPT="${SCRIPT:-train_hybrid.py}"                 # train_hazard.py for the hazard model (it has no --mode)
MODEARG=$([ "$SCRIPT" = train_hybrid.py ] && echo "--mode ${MODE:-hybrid}")
echo "code: $(git -C "$SLURM_SUBMIT_DIR" log --oneline -1)"
python "$SCRIPT" --data "${WF_DATA:-$HOME/scratch/wildfire/my}" --name "${NAME:-hybrid}" $MODEARG \
       --device cuda --workers 6 ${EXTRA:-}
