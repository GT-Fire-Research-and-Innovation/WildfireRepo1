---
license: other
pretty_name: Wildfire spread 2014-2025, upstream raw sources per fire
tags: [wildfire, remote-sensing, geospatial, fire-spread, raw-data]
---
# Wildfire spread 2014–2025: upstream raw sources per fire

This repo holds the upstream data behind the built dataset
[{SOURCE_REPO}](https://huggingface.co/datasets/{SOURCE_REPO}), fire by fire.
Each fire gets the original records, values and grids of every source, cut to the box and dates the build used
(the perimeter union plus 9.6 km). Nothing is regridded or derived.
Made by `pipeline/raw_grab.py` on the `data-grab` branch of
https://github.com/GT-Fire-Research-and-Innovation/WildfireRepo1, one channel at a time
(`notebooks/kaggle_data_grab.ipynb`).

## Layout
```
catalog/fires.csv, qc_fires.csv      the fire list (fire_id, IRWIN incident, name, dates, box, QC rating)
catalog/raw_index.csv                one row per fire and channel: status, number of files, bytes
fires/<fire_id>/
  LINKS.md, links.json               every channel of this fire: status, links to its files here, and the
                                     upstream URLs they came from (pending = channel not run yet)
  <channel>/...                      raw files, below
manifest/<channel>/<batch>.csv       one row per fire and channel: status (ok / empty / n/a / error),
                                     files, bytes, source URLs (separated by " | "), fetch time
shared/ics209/                       source files used by many fires, stored once
```
To get one fire, download `fires/<fire_id>/`, for example with
`huggingface_hub.snapshot_download(repo, repo_type="dataset", allow_patterns=["fires/<fire_id>/*"])`.
FIRMS links contain `{MAP_KEY}` where your own free FIRMS key goes. The LANDFIRE link is the job request;
add `&Email=<you>` to submit it.
`fire_id` = `"F"` followed by the first 10 hex characters of the SHA-1 of the incident key (its IRWIN ID).
It matches the built dataset.

## Channels
| channel | source | fires | files |
|---|---|---|---|
| history | NIFC Interagency Fire Perimeter History (every earlier fire in the box) | all | `history.geojson` |
| perimeters | NIFC GeoMAC (2014–2019), NIFS (2018–2019), Operational Data Archive (2020–2025): every record of the fire's incident, before cleaning | all | `records.geojson` (`_source_layer` per record) |
| firms | NASA FIRMS VIIRS 375 m active fire: S-NPP and NOAA-20 standard product, plus near-real-time for 2024+ | all | `<SOURCE>.csv.gz` |
| gridmet | GRIDMET daily, 9 variables (vpd, fm100, fm1000, bi, erc, rmin, vs, th, tmmx), from 14 days before the first perimeter to 2 days after the last, as the THREDDS subset service returns it | all | `gridmet_<var>.nc` (9 files) |
| ics209 | ICS-209-PLUS (St. Denis et al.): incident and situation-report rows | ≤ 2020 | `incidents.csv`, `sitreps.csv` |
| suppression | NIFS / ODA EventLine and EventPolygon: every archived version (GDB_FROM_DATE / GDB_TO_DATE) | 2018+ | `event_lines.geojson`, `event_polygons.geojson` |
| tessera | GeoTessera v1.1: reference only (year, box, how to fetch) | 2018+ | `tessera_ref.json` |
| terrain | Copernicus GLO-30 DEM, native 1 arc-second | all | `dem_glo30.tif` |
| landfire | LANDFIRE FBFM40, CC, CH, CBH, CBD from the newest version before the fire, native 30 m, as LFPS delivers it | 2017+ | `landfire_<version>.zip` |
| goes | GOES-16/17/18/19 ABI L2 fire product (FDCC): the scan nearest each hour, cropped on the fixed grid; Mask/Power/Area/Temp/DQF stay packed as in the source | span ≥ 2017-06 | `goes_<sat>.nc` |
| hrrr | NOAA HRRR GUST, UGRD/VGRD 10 m, RH 2 m, decoded from the files' own GRIB2 messages and cropped on the native 3 km grid: analyses every 3 h and the 12z run's 9 h forecast each day | span ≥ 2014-08 | `hrrr_ana.nc`, `hrrr_fc.nc` |
| osm | OpenStreetMap, Geofabrik snapshot of 1 January of the fire's year: full extract of the box | all | `osm_<YYYY>-01-01.osm.pbf` |

The netCDF time series hold `time` (ns UTC), `lat`/`lon` per cell, and `source` (the exact file each step came from).

## Sources and terms
This repo is a derivative of public data: NIFC / USFS / DOI perimeter and incident archives, NASA FIRMS, NOAA GOES
and HRRR (NOAA Open Data Dissemination), GRIDMET (University of Idaho), Copernicus DEM GLO-30
(© DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union
and ESA), LANDFIRE (USGS / USFS), ICS-209-PLUS (St. Denis et al., figshare 10.6084/m9.figshare.19858927), and
OpenStreetMap (© OpenStreetMap contributors, ODbL: the `osm` files are a derivative database under the ODbL).
Cite the original sources when you use these files.
