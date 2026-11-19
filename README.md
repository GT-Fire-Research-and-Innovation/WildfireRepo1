# Wildfire raw data grab

Downloads every upstream source behind each fire of the wildfire spread dataset
([Javyn123/wildfire-spread-2014-2025](https://huggingface.co/datasets/Javyn123/wildfire-spread-2014-2025)), one
channel at a time, into a Hugging Face dataset repo with one folder per fire. Nothing is regridded or cleaned:
each source keeps its own records, values and grid. This branch holds only the grab; the dataset build, the
model and the live forecast live on the other branches.

## Layout
The code is one package, `fah/`, in layers. Each layer imports only from the layers below it, and every module
exposes classes (`tests/test_architecture.py` checks both).

| Layer | Path | What it holds |
|---|---|---|
| 5 | `fah/__main__.py` | The command line (`python -m fah`): flags, logging, exit codes |
| 4 | `fah/commands/` | One class per action: `ChannelRun` (fetch a channel), `StatusReport`, `LinksBuilder`, `ExtentsBuilder`, `Repacker`, `SetupCheck` |
| 3 | `fah/channels/` | One class per channel (`HistoryChannel` ... `ViirsChannel`): what it fetches, which fires it covers, the files it writes; `ChannelRegistry` lists them cheap to heavy |
| 3 | `fah/hub/` | The Hugging Face side: `HubRepo`, `ManifestBook` (what is done), `BatchUploader`, `FireCatalog` (the built dataset's fire list) |
| 2 | `fah/sources/` | One client per upstream service: NIFC, ICS-209-PLUS, FIRMS, GRIDMET, HRRR, GOES, GFS (Earth Engine), Earthdata, Planetary Computer, LANDFIRE, DEM, Geofabrik |
| 2 | `fah/storage/` | File formats: gzipped GeoJSON, netCDF time series, sinusoidal mosaics, the packed `viirs.nc` |
| 1 | `fah/core/` | `Fire` (region, window, days), boxes and pixel windows, fire extents from FIRMS, run context, results, secrets |

| Other | What it is |
|---|---|
| `notebooks/kaggle_data_grab.ipynb` | Runs every channel on Kaggle, session after session |
| `docs/RAW_DATA_CARD.md` | The repo's data card: layout, every source, a guide for newcomers, TS-SatFire parity |
| `tests/` | Offline tests, mirroring the layers (`python -m pytest`) |

A new channel is one class in `fah/channels/` (a `PerFireChannel` or `SharedDownloadChannel`) added to
`ChannelRegistry.ORDER`; a new upstream service is one client in `fah/sources/`.

## Setup
```bash
pip install -r requirements.txt     # or: mamba env create -f environment.yml, or pip install -e ".[channels,dev]"
cp .env.example .env                # FIRMS map key, Earthdata token
```
`osm` also needs `osmium-tool` (and uses `aria2c` when present). Uploads need `HF_TOKEN` with write access;
`gfs` needs Google Earth Engine access (`GEE_SERVICE_ACCOUNT_KEY` and `GEE_PROJECT`).

## Running
```bash
python -m fah --channel history --hf-repo you/wildfire-raw-2014-2025   # one channel, resumable
python -m fah --status --hf-repo you/wildfire-raw-2014-2025           # progress per channel
python -m fah --extents --hf-repo you/wildfire-raw-2014-2025          # after firms: catalog/fire_extents.csv
python -m fah --links --hf-repo you/wildfire-raw-2014-2025            # fires/<id>/LINKS.md + links.json
python -m fah --repack --channel perimeters --hf-repo you/wildfire-raw-2014-2025  # gzip old .geojson in place
python -m fah --channel goes --ids F2003403265 --dry-run --work /tmp/raw  # one fire, no upload
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
