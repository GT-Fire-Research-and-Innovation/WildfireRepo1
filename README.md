# Wildfire raw data grab

Downloads every upstream source behind each fire of the wildfire spread dataset
([Javyn123/wildfire-spread-2014-2025](https://huggingface.co/datasets/Javyn123/wildfire-spread-2014-2025)), one
channel at a time, into a Hugging Face dataset repo with one folder per fire. Nothing is regridded or cleaned:
each source keeps its own records, values and grid. This branch holds only the grab; the dataset build, the
model and the live forecast live on the other branches.

## Layout
| Path | What it does |
|---|---|
| `pipeline/raw_grab.py` | Command line: fire list, resumable runs per channel, batched uploads, manifests, status, extents, links, repack |
| `pipeline/raw_channels.py` | One function per channel: what to fetch for a fire, from where, and how it is stored |
| `pipeline/earthdata.py` | NASA Earthdata search, token and byte-range reads (viirs, vegetation, landcover) |
| `pipeline/nifc_perimeters.py` | NIFC perimeter and event-layer services and archive geodatabases (perimeters, suppression) |
| `pipeline/suppression_layers.py` | Suppression archive layers by year, ICS-209-PLUS tables and incident matching (suppression, ics209) |
| `pipeline/weather_features.py` | GRIDMET aggregation names, HRRR byte-range fetch and grid (gridmet, hrrr, hrrr_next) |
| `notebooks/kaggle_data_grab.ipynb` | Runs every channel on Kaggle, session after session |
| `docs/RAW_DATA_CARD.md` | The repo's data card: layout, every source, a guide for newcomers, TS-SatFire parity |
| `tests/` | Offline tests of the grab (`python -m pytest tests`) |

## Setup
```bash
pip install -r requirements.txt     # or: mamba env create -f environment.yml
cp .env.example .env                # FIRMS map key, Earthdata token
```
`osm` also needs `osmium-tool` (and uses `aria2c` when present). Uploads need `HF_TOKEN` with write access;
`gfs` needs Google Earth Engine access (`GEE_SERVICE_ACCOUNT_KEY` and `GEE_PROJECT`).

## Running
```bash
python pipeline/raw_grab.py --channel history --hf-repo you/wildfire-raw-2014-2025   # one channel, resumable
python pipeline/raw_grab.py --status --hf-repo you/wildfire-raw-2014-2025           # progress per channel
python pipeline/raw_grab.py --extents --hf-repo you/wildfire-raw-2014-2025          # after firms: catalog/fire_extents.csv
python pipeline/raw_grab.py --links --hf-repo you/wildfire-raw-2014-2025            # fires/<id>/LINKS.md + links.json
python pipeline/raw_grab.py --repack --channel perimeters --hf-repo you/wildfire-raw-2014-2025  # gzip old .geojson in place
python pipeline/raw_grab.py --channel goes --ids F2003403265 --dry-run --work /tmp/raw  # one fire, no upload
```
Channels: history, perimeters, firms, gridmet, ics209, suppression, tessera (reference only), terrain, landfire,
lulc, landcover, vegetation, goes, hrrr, hrrr_next, gfs, osm, viirs. lulc, landcover, vegetation, hrrr_next, gfs,
viirs and gridmet's pr / tmmn / sph / pdsi carry every input TS-SatFire has (its VIIRS_Day / VIIRS_Night bands,
FirePred bands and ESRI_LULC map).

Each fire is cut on the source's own grid to its region (the box around all its perimeters pushed out 32 km on
every side, so any 64 km tile centred in the fire fits), over a time window that adds the days FIRMS saw the
fire active before its first perimeter and after its last, plus a day on each side. Its ignition point is stored
for tiles that start from it (run firms, then `--extents`, before the other time-dependent channels).

GeoJSON is stored gzipped and viirs as one `viirs.nc` per fire (a group per satellite pass). Batches are uploaded
and then deleted locally; a fire fetched again replaces its older files. Each fire gets one folder,
`fires/<id>/`, with a subfolder per channel and a `LINKS.md` that links every file and its upstream source.
