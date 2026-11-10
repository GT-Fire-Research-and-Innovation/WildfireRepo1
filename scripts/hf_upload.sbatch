#!/bin/bash
# Upload the built dataset to a Hugging Face dataset repo (resumable: resubmit if it stops).
#   HF_REPO=yourname/wildfire-spread-2014-2025 sbatch scripts/hf_upload.sbatch
# Needs a one-time `hf auth login` on the login node (the token is stored in ~/.cache/huggingface).
#SBATCH -J wf-hf-upload
#SBATCH -p ice-cpu
#SBATCH --qos=coe-ice
#SBATCH -N 1 --cpus-per-task=8 --mem=16G
#SBATCH -t 16:00:00
#SBATCH -o logs/hf-upload-%j.out
set -euo pipefail
module purge
export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
: "${HF_REPO:?set HF_REPO=yourname/reponame}"
STAGE="$HOME/scratch/wildfire/hf_stage"
# stage = links to what gets published (no copies): catalog, fires, quicklook, dataset card
mkdir -p "$STAGE"
ln -sfn "$HOME/scratch/wildfire/catalog" "$STAGE/catalog"
ln -sfn "$HOME/scratch/wildfire/my/fires" "$STAGE/fires"
ln -sfn "$HOME/scratch/wildfire/my/quicklook" "$STAGE/quicklook"
cp "$SLURM_SUBMIT_DIR/docs/DATASET_CARD.md" "$STAGE/README.md"
hf repo create "$HF_REPO" --repo-type dataset --private 2>/dev/null || true
hf upload-large-folder "$HF_REPO" "$STAGE" --repo-type dataset --num-workers 8 \
   --exclude "catalog/norm/*" "catalog/attrs_*" "catalog/geoms_*" "catalog/candidate_records.parquet" "fires/*/*.tmp.npz" "fires/*/*.error.txt"
echo "Done: https://huggingface.co/datasets/$HF_REPO"
