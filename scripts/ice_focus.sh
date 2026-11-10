#!/usr/bin/env bash
# Give the two hybrids (hybrid_v1, hybrid2_v1) first claim on your GPU budget:
#   bash scripts/ice_focus.sh           hold every other pending GPU job (evaluations, physics2, head-to-head)
#   bash scripts/ice_focus.sh release   let them run again (do this once both hybrids show RUNNING)
set -uo pipefail
others=$(squeue -h -u "$USER" -t PENDING -o "%i %j" | awk '$2 !~ /^wf-hybrid2?$/ && $2 != "wf-compare" {print $1}' | tr '\n' ' ')
if [ "${1:-}" = release ]; then
  held=$(squeue -h -u "$USER" -t PENDING -o "%i %r" | awk '$2 ~ /JobHeldUser/ {print $1}' | tr '\n' ' ')
  [ -n "$held" ] && scontrol release $held && echo "released: $held" || echo "nothing held"
else
  [ -n "$others" ] && scontrol hold $others && echo "on hold until you release them: $others" || echo "nothing to hold"
fi
echo
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.24R"
