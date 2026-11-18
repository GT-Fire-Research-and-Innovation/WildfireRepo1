---
license: other
pretty_name: Wildfire spread 2014-2025, upstream raw sources per fire
tags: [wildfire, remote-sensing, geospatial, fire-spread, raw-data]
---
# Wildfire spread 2014–2025: upstream raw sources per fire

This repo holds the upstream data behind the built dataset
[{SOURCE_REPO}](https://huggingface.co/datasets/{SOURCE_REPO}), fire by fire.
Each fire gets the original records, values and grids of every source, cut to its region (the box around all
its perimeters pushed out 32 km on every side) over its time window (below). Nothing is regridded or derived:
this is a raw layer to cut training tiles from, never to train on as it stands.
Made by `pipeline/raw_grab.py` on the `data_grab_satfire` branch of
https://github.com/GT-Fire-Research-and-Innovation/WildfireRepo1, one channel at a time
(`notebooks/kaggle_data_grab.ipynb`).

## Layout
```
catalog/fires.csv, qc_fires.csv      the fire list (fire_id, IRWIN incident, name, dates, box, QC rating)
catalog/raw_index.csv                one row per fire and channel: status, number of files, bytes
catalog/fire_extents.csv             each fire's time window, ignition point and region, and how each was found
fires/<fire_id>/
  LINKS.md, links.json               every channel of this fire: status, links to its files here, and the
                                     upstream URLs they came from (pending = channel not run yet)
  <channel>/...                      raw files, below
manifest/<channel>/<batch>.csv       one row per fire and channel: status (ok / empty / n/a / error),
                                     files, bytes, source URLs (separated by " | "), fetch time, and the
                                     days (window) and box (region) the channel covers
shared/ics209/                       source files used by many fires, stored once
```
To get one fire, download `fires/<fire_id>/`, for example with
`huggingface_hub.snapshot_download(repo, repo_type="dataset", allow_patterns=["fires/<fire_id>/*"])`.
FIRMS links contain `{MAP_KEY}` where your own free FIRMS key goes. NASA Earthdata links (viirs, vegetation,
landcover) need a free Earthdata Login account or token (https://urs.earthdata.nasa.gov).
`fire_id` = `"F"` followed by the first 10 hex characters of the SHA-1 of the incident key (its IRWIN ID).
It matches the built dataset.

## Region, time window, ignition point
**Region.** The box around all of a fire's perimeters, pushed out 32 km on every side (at least 32 km at every
latitude of the box), so any 64 km × 64 km tile centred anywhere in the fire fits inside: on its ignition point,
on its front on any day, or shifted at random. A fire too small to measure gets a 64 km square; larger fires get
64 km plus their own width. `perimeters` and `ics209` stay on the build's tighter box (perimeter union + 9.6 km):
they keep only the fire's own records, and a wider box would match neighbouring incidents. 27 fires burned wider
than 64 km; their region still holds all their data, but no single 64 km tile holds the whole fire.

**Time window.** A fire often burns before its first mapped perimeter and keeps burning after its last. Each fire's window runs
from the earlier of its first perimeter day and the first FIRMS-active day that leads into it, to the later of
its last perimeter day and the last active day that follows it, plus one day on each side (00:00 UTC of the first
day to 23:00 UTC of the last). An active day has at least 2 VIIRS detections inside the perimeters' own box
(+ about 1 km) that are not low confidence or a static land source; activity joins the fire across at most 2
quiet days, so a separate fire weeks earlier does not stretch the window. FIRMS itself covers a fixed 14 days
before the first perimeter to 14 days after the last.

**Ignition point.** The fire's earliest observation, which is known when it starts: the earlier of the first VIIRS
overpass that saw it (the centre of its detections within 15 minutes of the first one; `ignition_method` =
`firms`, or `firms_single` when no day had 2 detections) and its first mapped perimeter (`first_perimeter`, the
centre of the cells the built dataset's labels give the earliest arrival). Without either, the centre of the
perimeters' box (`perimeter_box_centre`, which uses the final extent: avoid it as a tile centre).

`catalog/fire_extents.csv` lists, per fire: `start_day` / `end_day`, the perimeter days, how many days activity
added on each side, `window_rule` (`perimeters+firms`, or `perimeters` without FIRMS data), `ignition_lat` /
`ignition_lon` / `ignition_time_utc` / `ignition_method`, and `region_west` / `south` / `east` / `north`.

The region and window are fixed after the fact (from the final perimeters and all the detections), so they
describe the whole fire. To train without seeing the future, cut your own inputs from these files using only what was
known at each step.

## Channels
| channel | source | fires | files |
|---|---|---|---|
| history | NIFC Interagency Fire Perimeter History (every earlier fire in the region) | all | `history.geojson` |
| perimeters | NIFC GeoMAC (2014–2019), NIFS (2018–2019), Operational Data Archive (2020–2025): every record of the fire's incident, before cleaning | all | `records.geojson` (`_source_layer` per record) |
| firms | NASA FIRMS VIIRS 375 m active fire: S-NPP and NOAA-20 standard product, plus near-real-time for 2024+, from 14 days before the first perimeter to 14 days after the last | all | `<SOURCE>.csv.gz` |
| gridmet | GRIDMET daily, 13 variables (vpd, fm100, fm1000, bi, erc, rmin, vs, th, tmmx, pr, tmmn, sph, and the 5-day drought index pdsi), from 13 days before the window (pdsi: 23) to 1 day after it, as the THREDDS subset service returns it | all | `gridmet_<var>.nc` (13 files) |
| ics209 | ICS-209-PLUS (St. Denis et al.): incident and situation-report rows | ≤ 2020 | `incidents.csv`, `sitreps.csv` |
| suppression | NIFS / ODA EventLine and EventPolygon: every archived version (GDB_FROM_DATE / GDB_TO_DATE) | 2018+ | `event_lines.geojson`, `event_polygons.geojson` |
| tessera | GeoTessera v1.1: reference only (year, region, how to fetch) | 2018+ | `tessera_ref.json` |
| terrain | Copernicus GLO-30 DEM, native 1 arc-second | all | `dem_glo30.tif` |
| landfire | LANDFIRE FBFM40, CC, CH, CBH, CBD from the newest version before the fire: the stored 30 m pixels in EPSG:5070 from LANDFIRE's image services (snapped to its grid, nearest neighbour), LZW GeoTIFF as the server returns it | 2017+ | `landfire_<version>_<layer>.tif` (5 files) |
| lulc | Esri / Impact Observatory 10 m annual land use / land cover (io-lulc-annual-v02, Microsoft Planetary Computer), map of the year before the fire (2017 for 2017 fires, 2023 at most): the box's pixels of each source tile on its own UTM grid; class legend in the `classes` tag | 2017+ | `io_lulc_<year>_<tile>.tif` |
| landcover | MODIS MCD12Q1 v061 annual land cover of the year before the fire, every layer (LC_Type1-5, LC_Prop1-3 and assessments, QC, LW), on the sinusoidal 463 m grid; tiles joined pixel for pixel | all | `mcd12q1_<year>.nc` |
| vegetation | VIIRS VNP13A1 v002 16-day vegetation indices, every layer (NDVI, EVI, EVI2, reflectances, angles, composite day, VI quality, pixel reliability), each composite overlapping 16 days before the window to its end, sinusoidal 463 m grid | all | `vnp13a1.nc` (`time` = composite start) |
| goes | GOES-16/17/18/19 ABI L2 fire product (FDCC): the scan nearest each hour of the window, cropped on the fixed grid; Mask/Power/Area/Temp/DQF stay packed as in the source | window ≥ 2017-06 | `goes_<sat>.nc` |
| hrrr | NOAA HRRR GUST, UGRD/VGRD 10 m, RH 2 m, decoded from the files' own GRIB2 messages and cropped on the native 3 km grid: analyses every 3 h and the 12z run's 9 h forecast each day of the window | window ≥ 2014-08 | `hrrr_ana.nc`, `hrrr_fc.nc` |
| hrrr_next | NOAA HRRR, the 12z run's forecasts every 3 h out to 36 h for each day (to 15 h before 2016-08-23, 18 h before 2018-07-12): GUST, UGRD/VGRD 10 m, RH 2 m, TMP 2 m, SPFH 2 m, APCP accumulated from the run start; `lead_h` and `run_time` per step | window ≥ 2014-08 | `hrrr_next.nc` |
| gfs | NOAA GFS 0.25° (TS-SatFire's forecast source) from Google Earth Engine `NOAA/GFS0P25`: the 12z run's forecasts every 3 h out to 10 days for each day of the window (to 36 h before 2017-07-09, as the collection holds them), every band of each image (2 m temperature in °C, specific / relative humidity, 10 m wind, precipitation accumulated over the last 1–6 h, precipitable water, cloud cover, radiation; more bands from 2025-01-15) on the native 0.25° grid; `lead_h` and `run_time` per step | window ≥ 2015-07 | `gfs.nc` |
| osm | OpenStreetMap, Geofabrik snapshot of 1 January of the fire's year: full extract of the region | all | `osm_<YYYY>-01-01.osm.pbf` |
| viirs | VIIRS L1B swaths of S-NPP (VNP02IMG / 03IMG / 02MOD / 03MOD), NOAA-20 (VJ1…) and NOAA-21 (VJ2…), every day and night granule over the region during the fire's window: I1–I5 (375 m) and M11 (750 m) as stored (scaled integers, quality flags, I4 / I5 brightness-temperature look-up tables) with each pixel's latitude, longitude and sensor / solar zenith, cut on the swath itself | all | `<SAT>_<YYYY-MM-DD>T<HHMM>_<day\|night>.nc` |

The netCDF time series hold `time` (ns UTC), `lat`/`lon` per cell, and `source` (the exact file each step came from).
The sinusoidal files (landcover, vegetation) hold `x`/`y` pixel centres in metres and a CF `crs` variable; values are
stored as in the source, so apply each variable's `scale_factor` / `add_offset` yourself.

### TS-SatFire inputs
Every input of TS-SatFire (Zhao, Gerard and Ban 2024) has a raw counterpart here:

| TS-SatFire | here |
|---|---|
| VIIRS_Day I1–I5, M11; VIIRS_Night I4, I5 | `viirs` (day and night granules, all three satellites) |
| active fire, burned area labels | `firms`, `perimeters`, `history` |
| NDVI, EVI2 | `vegetation` |
| gridMET precipitation, wind speed / direction, min / max temperature, ERC, specific humidity, PDSI | `gridmet` (`pr`, `vs`, `th`, `tmmn`, `tmmx`, `erc`, `sph`, `pdsi`) |
| slope, aspect, elevation (SRTM, 2000) | `terrain` (Copernicus GLO-30, 2011–2015, the newer 30 m model; slope and aspect follow from it) |
| land cover LC_Type1 (MODIS) | `landcover` |
| GFS forecast precipitation, wind, temperature, specific humidity | `gfs` (the same GFS 0.25° collection, 10 days ahead), and `hrrr_next` (HRRR 3 km, hourly-resolved, 36 h ahead) |
| ESRI_LULC | `lulc` |

## Sources and terms
This repo is a derivative of public data: NIFC / USFS / DOI perimeter and incident archives, NASA FIRMS, NASA LAADS
DAAC (VIIRS L1B) and LP DAAC (VNP13A1, MCD12Q1), NOAA GOES and HRRR (NOAA Open Data Dissemination), NOAA NCEP GFS
(public domain, via Google Earth Engine `NOAA/GFS0P25`), GRIDMET and
gridMET drought (University of Idaho), Esri / Impact Observatory 10 m land cover (CC BY 4.0, via Microsoft Planetary Computer), Copernicus DEM GLO-30
(© DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union
and ESA), LANDFIRE (USGS / USFS), ICS-209-PLUS (St. Denis et al., figshare 10.6084/m9.figshare.19858927), and
OpenStreetMap (© OpenStreetMap contributors, ODbL: the `osm` files are a derivative database under the ODbL).
Cite the original sources when you use these files.
