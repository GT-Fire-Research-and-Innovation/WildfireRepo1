#!/usr/bin/env bash
# Build the 2014-2025 multi-year dataset end to end. Resumable: rerun the same command after any stop
# (every stage skips what is already built).
#   bash scripts/build_all.sh /path/to/data 12        (data folder, parallel workers)
set -euo pipefail
DATA=${1:-data}; WORKERS=${2:-6}
mkdir -p "$DATA"
cd "$(dirname "$0")/../pipeline"
if [ ! -f "$DATA/catalog/fires.csv" ]; then
  python catalog.py all --out "$DATA" --years 2014-2025                                    2>&1 | tee -a "$DATA/catalog.log"
fi
python build_multiyear.py --catalog "$DATA/catalog" --out "$DATA/my" --years 2014-2025 \
       --stages labels,history,viirs,terrain,weather,landfire,tessera,suppression,hrrr,goes,human \
       --workers "$WORKERS"                                                                 2>&1 | tee -a "$DATA/build.log"
python quicklook_multiyear.py --out "$DATA/my" --fires $(ls "$DATA/my/fires" | shuf -n 12 --random-source=<(yes)) \
                                                                                            2>&1 | tail -3
