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
  echo "$name ${last:-starting}   best val AP ${best:--}"
  [ -n "$test" ] && echo "    $test"
  [ -n "$err" ] && echo "$err"
done
echo
echo "=== evaluations"
for log in $(ls -t logs/eval-*.out 2>/dev/null); do
  if grep -q "TEST (" "$log"; then sed -n '/TEST (/,/saved/p' "$log"; else echo "$log: running ($(grep -c pairs "$log") progress lines)"; fi
done
