#!/bin/bash
# Upload the built dataset to a private Hugging Face dataset repo. Resumable: if it stops, resubmit.
#   HF_REPO=yourname/wildfire-spread-2014-2025 sbatch scripts/hf_upload.sbatch
# One-time setup on the login node: `hf auth login` (paste your write token in the terminal, never in chat).
#SBATCH -J wf-hf-upload
#SBATCH -p ice-cpu
#SBATCH --qos=coe-ice
#SBATCH -N 1 --cpus-per-task=8 --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/hf-upload-%j.out
set -euo pipefail
module purge
export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
: "${HF_REPO:?set HF_REPO=yourname/reponame}"
W="$HOME/scratch/wildfire"
hf repos create "$HF_REPO" --repo-type dataset --private 2>/dev/null || true
hf upload "$HF_REPO" "$SLURM_SUBMIT_DIR/docs/DATASET_CARD.md" README.md --repo-type dataset
hf upload "$HF_REPO" "$W/catalog" catalog --repo-type dataset \
   --include "fires.csv" --include "perimeters.parquet" --include "qc_fires.csv" --include "qc_perimeters.csv" --include "candidates.csv"
hf upload "$HF_REPO" "$W/my/quicklook" quicklook --repo-type dataset
hf upload "$HF_REPO" "$W/my/fires" fires --repo-type dataset \
   --exclude "*/*.tmp.npz" --exclude "*/*.error.txt"
echo "Done: https://huggingface.co/datasets/$HF_REPO"
