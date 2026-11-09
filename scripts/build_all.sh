#!/usr/bin/env bash
# Build the 2014-2025 multi-year dataset end to end. Resumable: rerun the same command after any stop.
#   bash scripts/build_all.sh /path/to/data 8        (data folder, parallel workers)
set -euo pipefail
DATA=${1:-data}; WORKERS=${2:-6}
cd "$(dirname "$0")/../pipeline"
python catalog.py all --out "$DATA" --years 2014-2025          2>&1 | tee -a "$DATA/catalog.log"
python build_multiyear.py --catalog "$DATA/catalog" --out "$DATA/my" \
       --stages labels,history,viirs,terrain,weather,landfire --workers "$WORKERS" 2>&1 | tee -a "$DATA/build.log"
python build_multiyear.py --catalog "$DATA/catalog" --out "$DATA/my" --stages human 2>&1 | tee -a "$DATA/build_human.log"
