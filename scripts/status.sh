#!/usr/bin/env bash
# One-look status of everything on ICE:   bash scripts/status.sh
cd "$(dirname "$0")/.."
echo "=== jobs"
squeue -u "$USER" -o "%.10i %.16j %.8T %.10M %.22R"
echo
echo "=== training (last epochs, best validation so far, test result if finished)"
for log in $(ls -t logs/train-*.out 2>/dev/null); do
  name=$(grep -o "^\[[^]]*\]" "$log" | head -1)
  [ -z "$name" ] && continue
  case " $seen " in *" $name "*) continue ;; esac; seen="$seen $name"
  last=$(grep -h "  epoch " "$log" | tail -1)
  best=$(grep -h "val AP" "$log" | sed 's/.*val AP \([0-9.]*\).*/\1/' | sort -n | tail -1)
  test=$(grep -h "TEST" "$log" | tail -1)
  err=$(grep -l "Traceback\|Error" "$log" >/dev/null && echo "  !! error in $log")
  jid=$(basename "$log" .out | grep -o "[0-9]*$")
  state=""
  if [ -z "$test" ] && ! squeue -h -j "$jid" >/dev/null 2>&1 | grep -q . && ! squeue -h -j "$jid" 2>/dev/null | grep -q .; then
    state="   (old log: that job ended without finishing, e.g. cancelled; a newer run may be waiting in the queue)"
  fi
  echo "$name ${last:-starting}   best val AP ${best:--}$state"
  [ -n "$test" ] && echo "    $test"
  [ -n "$err" ] && echo "$err"
done
echo
echo "=== evaluations (current labels only; earlier old-label runs are listed by name)"
RUNS="$HOME/scratch/wildfire/my/runs"
for log in $(ls -t logs/eval-*.out 2>/dev/null); do
  name=$(grep -o "^\[[^]]*\] TEST" "$log" | head -1 | sed 's/^\[\(.*\)\] TEST/\1/')
  if [ -z "$name" ]; then
    grep -q "validation pass\|test pass" "$log" && echo "$log: running ($(grep -c "pairs (" "$log") progress lines)"
    continue
  fi
  case " $shown " in *" $name "*) continue ;; esac; shown="$shown $name"
  if grep -q '"data_version": 2' "$RUNS/eval_$name.json" 2>/dev/null; then sed -n '/TEST (/,/saved/p' "$log"
  else echo "[$name] earlier run on the old labels (re-scored in the evaluation period)"; fi
done
[ -f "$RUNS/compare.md" ] && echo && echo "full comparison: cat $RUNS/compare.md"
