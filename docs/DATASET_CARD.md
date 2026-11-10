---
license: other
pretty_name: Wildfire spread 2014-2025 (perimeters + 11 input layers, 150 m)
tags: [wildfire, remote-sensing, geospatial, fire-spread]
---
# Wildfire spread dataset, 2014–2025 (150 m)

Daily fire perimeters for 1,841 U.S. wildfires (≥ 1,000 acres, ≥ 2 good perimeters after quality control)
with 11 co-registered input layers, built for next-perimeter fire spread prediction.
Code: https://github.com/rafiki230/WildfireRepo1 (`pipeline/catalog.py`, `pipeline/qc_fires.py`,
`pipeline/build_multiyear.py`).

## Layout
```
catalog/fires.csv            one row per fire: id, name, year, first/last perimeter, acres, box, sources
catalog/perimeters.parquet   cleaned daily perimeters (GeoParquet, EPSG:4326)
catalog/qc_fires.csv         per-fire quality rating (keep / bad) and flags
catalog/qc_perimeters.csv    per-perimeter checks; drop=True perimeters are left out of the labels
fires/<fire_id>/grid.json    150 m lon/lat grid (transform, size), name, year, perimeter time span
fires/<fire_id>/<layer>.npz  one NumPy archive per layer (below)
quicklook/<fire_id>.png      one-page picture of every layer for a sample of fires
```
Load a fire: `np.load("fires/F6ec7489178/labels.npz")` → `arrival` (index of the first perimeter containing
each cell, -1 = never), `times` (perimeter times, ns UTC), `other` (other fires' burns, exclude from losses).

## Layers
| file | source | years | contents |
|---|---|---|---|
| labels | NIFC GeoMAC + Operational Data Archive, cleaned | 2014–2025 | arrival index, perimeter times, other-fire mask |
| viirs | NASA FIRMS VIIRS 375 m (S-NPP, NOAA-20) | all | day / night detections and max FRP per local day; `points.csv.gz` raw detections |
| goes | NOAA GOES-16/17/18/19 ABI fire product | 2017+ | hourly fire pixels (sparse), terrain-parallax corrected |
| weather | GRIDMET | all | 9 daily variables, 14 days before the first perimeter to 2 days after the last |
| hrrr | NOAA HRRR | Aug 2014+ | gust, u10, v10, rh2 every 3 h across the fire + 12z+9h forecasts |
| terrain | Copernicus GLO-30 DEM | all | elevation, slope, aspect, terrain position (0.5 / 2 km), roughness |
| landfire | LANDFIRE (pre-fire version) | fires 2017+ | FBFM40 fuel model, canopy cover / height / base height / bulk density |
| tessera | GeoTessera v1.1 | fires 2018+ | 128-d embedding of the year before the fire, 150 m means |
| suppression | NIFC archive lines; ICS-209-PLUS | 2018+ / ≤ 2020 | daily fire lines, burnouts, drops (bit-packed); crews, aircraft, containment |
| human | OpenStreetMap, 1 Jan of the fire year | all | roads ×3, rail, power lines, buildings, developed land, water, rivers |
| history | NIFC Interagency Fire Perimeter History | all | years since last fire, number of past fires |
Layers that do not exist for a fire's year carry `missing=True` (or are absent).

## Quality control
Placeholder perimeters (triangles, circles, boxes, buffer-generated shapes), stale repeats, and perimeters
outside the fire's footprint are dropped; fires with fewer than 2 good perimeters are rated bad. After the
build, fires whose new burn had VIIRS/GOES fire within 1 km less than 20 % of the time are also rated bad.
Training code skips fires rated bad in `catalog/qc_fires.csv`.

## Sources and terms
Derived from public data: NIFC / USFS / DOI perimeter archives, NASA FIRMS, NOAA GOES and HRRR (NOAA Open Data
Dissemination), GRIDMET (University of Idaho), Copernicus DEM GLO-30 (© DLR / Airbus, ESA licence), LANDFIRE
(USGS / USFS), GeoTessera (University of Cambridge), ICS-209-PLUS (St. Denis et al.), and OpenStreetMap
(© OpenStreetMap contributors, ODbL: the `human` layer is a derivative database under the ODbL).
Cite the original sources when using these layers.
