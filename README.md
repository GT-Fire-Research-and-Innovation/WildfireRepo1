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
| `pipeline/raw_grab.py`, `raw_channels.py`, `earthdata.py` | Upstream raw data of every fire (NIFC records, FIRMS, GOES, HRRR, GRIDMET, DEM, LANDFIRE, suppression, ICS-209, OSM, plus TS-SatFire's inputs: VIIRS L1B swaths, VNP13A1, MCD12Q1, Esri 10 m land cover), one channel at a time, uploaded to a Hugging Face repo; run on Kaggle with `notebooks/kaggle_data_grab.ipynb` |

| `physics/rothermel.py` | Differentiable Rothermel surface spread in PyTorch (Scott & Burgan 40 fuel models, dynamic curing, BehavePlus wind adjustment, wind + slope vectors, elliptical shape). Matches the pyretechnics reference to 0.013 % on 400 random cases |
| `physics/solver.py` | Differentiable arrival-time solver: 16-neighbour minimum travel time with an exact adjoint backward pass (memory does not grow with iterations) |
| `tests/test_physics.py` | Flat circle, wind ellipse (L/B), upslope speed-up, non-burnable barrier, finite-difference gradient checks |

Planned next: physics-only baseline on the 117 training fires, `model/` (CNN encoder + temporal
transformer predicting a spread-rate correction), `app/` (web app).

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # add your FIRMS MAP_KEY
```

## Live data
```bash
python live/wfigs_poller.py --out data/live            # saves every changed NIFC perimeter, polling every 5 min
python live/wfigs_poller.py --out data/live --report   # how complete the archive is (gaps between polls, versions)
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

## Raw source data
```bash
python pipeline/raw_grab.py --channel history --hf-repo you/wildfire-raw-2014-2025   # one channel, resumable
python pipeline/raw_grab.py --status --hf-repo you/wildfire-raw-2014-2025           # progress per channel
python pipeline/raw_grab.py --links --hf-repo you/wildfire-raw-2014-2025            # fires/<id>/LINKS.md + links.json
python pipeline/raw_grab.py --channel goes --ids F2003403265 --dry-run --work /tmp/raw  # one fire, no upload
```
Channels: history, perimeters, firms, gridmet, ics209, suppression, tessera (reference only), terrain, landfire,
lulc, landcover, vegetation, goes, hrrr, hrrr_next, osm, viirs. lulc, landcover, vegetation, hrrr_next, viirs and
gridmet's pr / tmmn / sph / pdsi carry every input TS-SatFire has (its VIIRS_Day / VIIRS_Night bands, FirePred
bands and ESRI_LULC map); viirs, vegetation and landcover need a free `EARTHDATA_TOKEN`
(https://urs.earthdata.nasa.gov, Generate Token). Each fire is cut to the built dataset's box and dates on the source's own grid. Batches are
uploaded and then deleted locally. Each fire gets one folder, `fires/<id>/`, with a subfolder per channel and a
`LINKS.md` that links every file and its upstream source. `notebooks/kaggle_data_grab.ipynb` runs all channels
on Kaggle. Repo layout and sources: `docs/RAW_DATA_CARD.md`.

## Evaluation protocol
Scores only cells that newly burn between consecutive perimeters (already-burned cells are
excluded), masks unknown cells, and splits by fire (year split planned). Metrics: average
precision, F1 at the validation threshold, IoU, predicted/true area ratio.
