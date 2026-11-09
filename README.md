# Physics-Informed Wildfire Spread Forecaster

Predicts where an active wildfire will have burned by the next day. A neural network corrects a
Rothermel spread-rate field; a differentiable level-set solver turns that field into an
arrival-time map. Trained on 123 western-US fires (2018–2021, TS-SatFire + NIFC perimeters),
then run live on current fires.

## Layout
| Folder | What it does |
|---|---|
| `live/` | Live ingestion: `wfigs_poller.py` archives every new WFIGS perimeter version; `firms.py` pulls VIIRS active-fire detections (near-real-time and archive) |
| `pipeline/` | Training-data builders: NIFC perimeter labels, suppression layers, OSM human features, GRIDMET + HRRR weather, VIIRS bands, LANDFIRE fuels, dataset assembly |
| `pipeline/train.py` | Two-branch gated U-Net (the plan's "plain U-Net" baseline), distance/logistic baselines, evaluation |
| `pipeline/analyze.py` | Error analysis by gap, distance from front, active fire, fire size, plus example maps |

Planned next: `physics/` (differentiable Rothermel + level-set solver), `model/` (CNN encoder +
temporal transformer), `app/` (web app).

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # add your FIRMS MAP_KEY
```

## Live data
```bash
python live/wfigs_poller.py --out data/live            # polls every 15 min; --once for cron
python live/firms.py latest --days 1                    # CONUS VIIRS, last 24 h (S-NPP, NOAA-20, NOAA-21)
python live/firms.py fires --live data/live --days 2    # detections around every archived fire
python live/firms.py history --bbox W S E N --start 2020-09-01 --end 2020-09-20
```
FIRMS allows 5000 transactions per 10 minutes per key.

## Training data and model
```bash
cd pipeline
python build_dataset.py --out ../data          # per-fire npz (needs labels, weather, human layers)
python train.py prep --out ../data              # splits, PCA, disk cache
python train.py base --out ../data              # baselines
python train.py train --out ../data --name full --epochs 40
python analyze.py --out ../data --name full
```

## Evaluation protocol
Scores only cells that newly burn between consecutive perimeters (already-burned cells are
excluded), masks unknown cells, and splits by fire (year split planned). Metrics: average
precision, F1 at the validation threshold, IoU, predicted/true area ratio.
