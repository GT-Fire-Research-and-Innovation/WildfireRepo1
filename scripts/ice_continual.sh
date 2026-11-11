#!/usr/bin/env bash
# Continuous learning (run weekly on PACE-ICE):
#   new NIFC perimeters (archived every 2 h by the website's job) -> new training fires -> fine-tune
#   the live model on old + new fires -> score it against the live model -> promote only if it is
#   no worse on the frozen 2024-25 test fires AND better on the newest fires it never saw -> refit the
#   router -> export to live/models and push (the website picks it up on its next run).
#
#   bash scripts/ice_continual.sh            BASE=unet3_time MODE=unet3 EXTRA="--width 64 --u3-attn 0" (defaults)
# Every step is a Slurm job chained on the previous one; logs in logs/. Safe to rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
D="$HOME/scratch/wildfire/my"; RUNS="$D/runs"
ARCH="$HOME/scratch/wildfire/archive"; CAT="$HOME/scratch/wildfire/live_catalog"
BASE="${BASE:-unet3_time}"; MODE="${MODE:-unet3}"; EXTRA="${EXTRA:---width 64 --u3-attn 0}"
MEMBERS="${MEMBERS:-unet3_time_eval hybrid_v1_eval}"          # router members; BASE's eval is swapped for the candidate's
TAG=$(date +%Y%m%d); CAND="${BASE}_ft${TAG}"
PY="export PATH=\$HOME/scratch/envs/wildfire/bin:\$PATH;"
export PATH="$HOME/scratch/envs/wildfire/bin:$PATH"
git pull -q --ff-only || echo "(git pull failed: using the code on disk)"

echo "1) perimeter archive"
if [ -d "$ARCH/.git" ]; then git -C "$ARCH" pull -q --ff-only; else git clone -q --depth 1 --branch perimeter-archive https://github.com/Javyn05/WildfireRepo1.git "$ARCH"; fi

echo "2) archive -> training catalog"
python pipeline/archive_catalog.py --archive "$ARCH" --out "$CAT" || exit 1
[ -f "$CAT/fires.csv" ] || { echo "no new fires qualify yet (need 3+ daily perimeters, 300+ acres, finished growing); nothing to do"; exit 0; }
python - "$CAT" <<'PY'
import sys, pandas as pd
cat = sys.argv[1]
f = pd.read_csv(f"{cat}/fires.csv").sort_values("last")
k = max(1, len(f) // 5)                       # newest 20% are held out to judge the fine-tuned model
f.tail(k)[["fire_id", "name", "last"]].to_csv(f"{cat}/holdout.csv", index=False)
print(f"   {len(f)} archive fires; {k} newest held out for the promotion check")
PY

sub() { sbatch --parsable --exclude=atl1-1-03-014-16-0 "$@"; }
echo "3) build the new fires' inputs (CPU)"
B=$(sub -p ice-cpu --qos=coe-ice -c 8 --mem=48G -t 12:00:00 -J wf-cl-build -o logs/cl-build-%j.out \
     --wrap "$PY cd pipeline; python build_multiyear.py --catalog $CAT --out $D --workers 6")
echo "4) fine-tune $BASE -> $CAND (GPU)"
T=$(NAME=$CAND MODE=$MODE EXTRA="$EXTRA --init $RUNS/$BASE.pt --epochs 6 --lr 5e-5 --holdout-fires $CAT/holdout.csv" \
     sub -J "wf-$CAND" -t 6:00:00 --dependency=afterok:$B scripts/ice_train.sbatch)
echo "5) score candidate and live model: standard test + new fires"
E1=$(MODELS=$CAND OUT=${CAND}_eval sub -J wf-cl-eval -t 3:00:00 --dependency=afterok:$T scripts/ice_eval.sbatch)
E2=$(MODELS=$CAND OUT=${CAND}_new_eval EXTRA="--test-years 2026 --fires $CAT/holdout.csv" sub -J wf-cl-eval -t 2:00:00 --dependency=afterok:$T scripts/ice_eval.sbatch)
E3=$(MODELS=$BASE OUT=${BASE}_new_eval EXTRA="--test-years 2026 --fires $CAT/holdout.csv" sub -J wf-cl-eval -t 2:00:00 --dependency=afterok:$B scripts/ice_eval.sbatch)
NEWM=$(echo "$MEMBERS" | sed "s/${BASE}_eval/${CAND}_eval/")
echo "6) gate -> router refit -> export -> push (only if promoted)"
G=$(sub -p ice-cpu --qos=coe-ice -c 4 --mem=64G -t 2:00:00 -J wf-cl-promote -o logs/cl-promote-%j.out --dependency=afterok:$E1:$E2:$E3 \
     --wrap "$PY cd \$HOME/scratch/WildfireRepo1 && python live/promote.py --runs $RUNS --candidate ${CAND}_eval --incumbent ${BASE}_eval \
             --candidate-new ${CAND}_new_eval --incumbent-new ${BASE}_new_eval && \
             (cd model && python combine.py --data $D --members $NEWM --out combine_${CAND}) && \
             python live/export_models.py --runs $RUNS --combine combine_${CAND} --out live/models && \
             git add -f live/models && git commit -qm 'continual learning: promote $CAND' && \
             (git push -q || echo 'PUSH FAILED: run git push from the login node (needs your GitHub token)')")
echo "queued: build $B -> train $T -> evals $E1 $E2 $E3 -> gate $G   (watch: tail -f logs/cl-promote-$G.out)"
