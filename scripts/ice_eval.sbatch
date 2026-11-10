#!/bin/bash
# GT PACE-ICE: score models on the test years with model/evaluate.py (raw + per-fire calibrated +
# baselines + breakdowns). Works on finished runs (.pt) and on runs still training (best .ckpt).
#   MODELS="physics_v1:physics" OUT=physics_eval sbatch scripts/ice_eval.sbatch
#   MODELS="hybrid_v1" OUT=hybrid_v1_eval sbatch scripts/ice_eval.sbatch
#   MODELS="hybrid_v1 hybrid_s1 hybrid_s2" OUT=hybrid_ens3 sbatch scripts/ice_eval.sbatch   # ensemble
#SBATCH -J wf-eval
#SBATCH -p coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:1
#SBATCH -N 1 --cpus-per-task=8 --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/eval-%x-%j.out
module purge
export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
cd "$SLURM_SUBMIT_DIR/model"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python evaluate.py --data "$HOME/scratch/wildfire/my" --model ${MODELS:?set MODELS} --out "${OUT:?set OUT}" \
       --device cuda ${EXTRA:-}
