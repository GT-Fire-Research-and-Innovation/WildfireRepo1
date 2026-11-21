# WildfireRepo1: data sources, fire naming, per-fire link chain

Repo: https://github.com/GT-Fire-Research-and-Innovation/WildfireRepo1 (default branch `phase1-live-ingest`,
commit `88247ef`, 2026-11-14). Public built dataset: https://huggingface.co/datasets/Javyn123/wildfire-spread-2014-2025
(checked 2026-11-15: 1,994 fires in `catalog/fires.csv`, 1,819 rated `keep`, 175 `bad`).

All links below were taken from the code; the ones marked ✔ were requested on 2026-11-15 and answered.

---

## 1. How fires are named

The repo has four fire-ID schemes, one per dataset.

| Dataset | ID format | Example | Made in |
|---|---|---|---|
| Multi-year training set (2014–2025) | `F` + first 10 hex chars of SHA-1(incident key) | `F2003403265` (Camp, 2018) | `pipeline/catalog.py:411` |
| Live-archive retraining catalog | `L` + first 10 hex of SHA-1(WFIGS UniqueFireIdentifier) | — | `pipeline/archive_catalog.py:82` |
| Live forecast (website) | `LIVE_` + UniqueFireIdentifier without dashes; JSON file = `<UniqueFireIdentifier>.json` | `LIVE_2025CASHU012345` | `live/forecast_now.py:212`, `live/publish.py:328` |
| Legacy 123-fire set (TS-SatFire based) | TS-SatFire folder name | `US_2021_ID4558511544420210705`, `24463187` | upstream TS-SatFire |

### Multi-year set (`F…` IDs): exact rule

1. **Incident key** (`catalog.incident_keys`, `nifc_perimeters.incident_key`):
   - IRWIN ID, with `{`, `}` and spaces stripped from the ends, upper-cased: `75E64DB8-9B75-4A68-BEDD-67CC62658E38`
   - if a record has no IRWIN ID: `name:<incident name, lower-case, single spaces>|<UniqueFireIdentifier>`
2. **fire_id** = `"F" + sha1(incident_key).hexdigest()[:10]`.
3. **name** = `IncidentName` of the fire's **last** kept perimeter, exactly as the archive spells it (so `CAMP`, `Caldor`,
   `Haypress (River Complex)`). It is a label only, not unique.

Checked against the public `fires.csv`: SHA-1 of the `incident` column reproduces all 1,994 `fire_id`s; every fire
in the current catalog is keyed by IRWIN ID (none by name).

Things to know:
- **Names are not unique.** Two 2021 fires are named "Dixie": `Fcff391c0fa` (Idaho, 43k acres) and `F1df87a4ce9`
  (California, 963k acres). The TS-SatFire benchmark maps `US_2021_ID4558511544420210705` to the Idaho one. Join on
  `fire_id` or IRWIN ID, never the name.
- **14 fires have dirty keys.** Their IRWIN field held trailing junk (`}\n`, `}\r\n`, `} \n`, `\n`) that
  `.str.strip("{} ")` does not remove, so the junk is hashed into the ID. One fire got **two IDs** because of this:
  Cronan (River Complex) 2021 is `F3e9841215e` (key `…74547C\r\n`) and `F926614fdfe` (key `…74547C}\r\n`), both rated
  `keep`. Others: Quartz Ridge, Big Knife, Hogpen, Haypress, Bear Creek, Dale, Trail Springs, Wolf, Stoner Mesa,
  Table Rock, Summer, Peak.
- **The same fire can have more than one IRWIN ID across archives.** For the Camp Fire, NIFS 2018 holds 51 "Camp"
  perimeters under `{36D31BE1-6FF5-434C-9874-18E5EABF070D}` (with no PolygonDateTime) plus one auto-generated record
  under the GeoMAC IRWIN ID. Only the GeoMAC IRWIN ID became a catalog fire.

### Legacy TS-SatFire IDs
- TS-SatFire's 2017–2020 lists (`roi/us_fire_20xx_out_new.csv` in zhaoyutim/TS-SatFire) use numeric event IDs
  (e.g. `24463187`).
- The 2021 list uses `US_<year>_<ST><lat×1000><lon×1000><YYYYMMDD>`.
  For example, `ID4558511544420210705` means Idaho, 45.585 N, 115.444 W, start 2021-07-05. This matches the MTBS
  Event_ID layout. That match comes from the format only; I did not confirm it with TS-SatFire.
- `nifc_perimeters.py` matched each TS-SatFire fire to NIFC perimeters by its box and dates, and
  `benchmarks/tssatfire_2021_test_match.csv` maps `ts_id` to the `F…` ID.

---

## 2. Every data source (master link list)

### 2a. Fire perimeters (labels): NIFC ArcGIS Online, owner NIFC_Authoritative
REST root ✔: https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services

| Years | Archive | Perimeter layer | Line layer (suppression) |
|---|---|---|---|
| 2014–2018 | GeoMAC historic | `Historic_Geomac_Perimeters_<year>/FeatureServer/0` ✔ | — |
| 2019 | GeoMAC historic | `Historic_GeoMAC_Perimeters_2019/FeatureServer/0` ✔ | — |
| 2018 | NIFS | `National_Incident_Feature_Service_2018/FeatureServer/2` (EventPolygon2018) ✔ | `/1` (EventLine2018) ✔ |
| 2019 | NIFS | `2019_NIFS_OpenData/FeatureServer/2` (EventPolygon2019) ✔ | `/0` (EventLine2019) ✔ |
| 2020 | Operational Data Archive, File Geodatabase only (0.47 GB) ✔ | https://www.arcgis.com/sharing/rest/content/items/ea843f7f091f4c7f9743798b64c864be/data | same gdb |
| 2021 | Operational Data Archive, File Geodatabase only (1.34 GB) ✔ | https://www.arcgis.com/sharing/rest/content/items/af727c41d79643b091cee372233110d4/data | same gdb |
| 2022 | Operational Data Archive | `Operational_Data_Archive_2022/FeatureServer/7` (EventPolygon2022) ✔ | `/5` ✔ |
| 2023, 2024 | Operational Data Archive | `Operational_Data_Archive_2023`, `_2024` (layer found at run time by name) ✔ | same |
| 2025 | Operational Data Archive | `Operational_Data_Archive_2025/FeatureServer/6` (Event_Polygon) ✔ | `/4` ✔ |

Records kept: `FeatureCategory` in {Wildfire Daily Fire Perimeter, Wildfire Final Fire Perimeter}; drop hand
sketch / modeled, deleted, test edits; one perimeter per burn day; shrinking re-maps dropped; then `qc_fires.py`.

### 2b. Input layers
| Layer | Source | Link / template |
|---|---|---|
| history | Interagency Fire Perimeter History | https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0 ✔ |
| viirs | NASA FIRMS area API (needs free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/) | `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SRC}/{w,s,e,n}/{days≤5}/{YYYY-MM-DD}`, SRC = `VIIRS_SNPP_SP`, `VIIRS_NOAA20_SP` (2018+); NRT fill 2024+: `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `VIIRS_NOAA21_NRT` |
| goes | NOAA GOES ABI L2 fire (FDCC, CONUS) on AWS | `https://noaa-goes16.s3.amazonaws.com/?list-type=2&prefix=ABI-L2-FDCC/{YYYY}/{DOY}/{HH}/` ✔; buckets `noaa-goes16` (2017-06-01→2025-04-06), `noaa-goes17` (2018-11-15→2023-01-09), `noaa-goes18` (2023-01-10→), `noaa-goes19` (2025-04-07→) |
| weather | GRIDMET (Univ. of Idaho) THREDDS OPeNDAP | `https://thredds.northwestknowledge.net/thredds/dodsC/agg_met_{v}_1979_CurrentYear_CONUS.nc` ✔, v = vpd, fm100, fm1000, bi, erc, rmin, vs, th, tmmx |
| hrrr | NOAA HRRR on AWS (byte ranges from `.idx`) | `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{YYYYMMDD}/conus/hrrr.t{HH}z.wrfsfcf{FF}.grib2` ✔ (+ `.idx`); fields GUST, UGRD/VGRD 10 m, RH 2 m |
| terrain | Copernicus GLO-30 DEM on AWS | `https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{N/S}{lat:02}_00_{E/W}{lon:03}_00_DEM/Copernicus_DSM_COG_10_{N/S}{lat:02}_00_{E/W}{lon:03}_00_DEM.tif` ✔ |
| landfire | LANDFIRE Product Service (needs contact email) | `https://lfps.usgs.gov/api/job/submit` → `/status?JobId=` → output zip; version LF2016 for 2017–2022 fires, LF{year−1} for 2023+, none ≤2016 |
| tessera | GeoTessera v1.1 Zarr via `geotessera.store.GeoTesseraZarr()` (library default store; the URL is not in this repo) | https://github.com/ucam-eo/geotessera, https://data.source.coop/tessera/tessera, https://registry.opendata.aws/tessera/ |
| suppression (spatial) | NIFS / Operational Data Archive EventLine + EventPolygon (2018+) | table 2a |
| suppression (ICS) | ICS-209-PLUS (St. Denis et al.), figshare 19858927, 1999–2020 | https://api.figshare.com/v2/articles/19858927 ✔ → `ics209plus-wildfire.zip` https://ndownloader.figshare.com/files/38766504 ✔ |
| human | OpenStreetMap, Geofabrik 1-Jan snapshots | `https://download.geofabrik.de/north-america/{region}-{YY}0101.osm.pbf` ✔; region `us-west`, else `us/kansas`, `us/oklahoma`, `us/texas`, `us/florida`, `us/nebraska`, `us/missouri` |

### 2c. Built outputs and files in the repo
- Built dataset: https://huggingface.co/datasets/Javyn123/wildfire-spread-2014-2025 (its card says the code is at
  `github.com/rafiki230/WildfireRepo1`; the repo copy says GT-Fire-Research-and-Innovation)
- Files in the repo: `pipeline/excluded_fires.csv` (dedupe drops), `pipeline/perimeter_time_source.csv.gz`
  (21,104 rows: per-perimeter time source, `mapped` / `edit` / `date_only`), `benchmarks/tssatfire_2021_test_match.csv`

### 2d. Legacy 123-fire set (README "2018–2021")
- TS-SatFire mirror: https://huggingface.co/datasets/SamuelWu318/ts-satfire (per fire: `VIIRS_Day`, `VIIRS_Night`,
  `FirePred`, `ESRI_LULC`)
- Tessera per fire: https://huggingface.co/datasets/SamuelWu318/ts-tesserafire-5 (`tessera/p<fire_id>.npz`)
- Upstream: https://github.com/zhaoyutim/TS-SatFire, paper https://arxiv.org/abs/2412.11555
- Labels: the same NIFC archives (GeoMAC 2000–2019 + NIFS 2018+), queried by each fire's box

### 2e. Live system
- WFIGS perimeters: https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0
  and `WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0`
- WFIGS incident points: `WFIGS_Incident_Locations_Current/FeatureServer/0`
- FIRMS NRT (above); Esri VIIRS hotspots (map only): https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services/Satellite_VIIRS_Thermal_Hotspots_and_Fire_Activity/FeatureServer/0
- Perimeter version archive (polled every 5 min): https://github.com/GT-Fire-Research-and-Innovation/WildfireRepo1/tree/perimeter-archive
  (layout `perimeters/<UniqueFireIdentifier>/<PolygonDateTime>_<hash8>.geojson`)
- Site: https://gt-fire-research-and-innovation.github.io/WildfireRepo1/
- Display only, not model inputs: NASA GIBS VIIRS true color, OpenFreeMap, AWS terrarium tiles, Esri and USGS
  basemaps, and Google searches for InciWeb and Watch Duty

---

## 3. One fire, end to end: Camp Fire 2018 (`F2003403265`)

Catalog row (HF `catalog/fires.csv`): incident `75E64DB8-9B75-4A68-BEDD-67CC62658E38`, name `CAMP`, 15 perimeters
over 15 burn days, 2018-11-09 00:00 → 2018-11-25 18:22 UTC, 153,299 acres, source `geomac`, QC `keep`.
Grid (`grid.json`): 150 m, 349 × 371 cells, box **W −121.88999, S 39.51118, E −121.23980, N 39.98464**
(the perimeter union plus 9.6 km).

| # | Layer | Exact request(s) |
|---|---|---|
| 1 | **Identity** | `"F" + sha1("75E64DB8-9B75-4A68-BEDD-67CC62658E38")[:10]` = `F2003403265` |
| 2 | **labels** (28 raw records ✔) | https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Historic_Geomac_Perimeters_2018/FeatureServer/0/query?where=irwinid%3D%2775e64db8-9b75-4a68-bedd-67cc62658e38%27&outFields=*&f=geojson |
| 2b | labels, also scanned (1 auto-generated record ✔, lost in cleaning) | https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/National_Incident_Feature_Service_2018/FeatureServer/2/query?where=IRWINID%3D%2775e64db8-9b75-4a68-bedd-67cc62658e38%27&outFields=*&f=geojson |
| 2c | `other` mask | other catalog fires in the box during 2018: https://huggingface.co/datasets/Javyn123/wildfire-spread-2014-2025/blob/main/catalog/perimeters.parquet |
| 3 | **history** (151 polygons ✔) | https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query?where=FIRE_YEAR_INT+%3C+2018+AND+FIRE_YEAR_INT+%3E%3D+1978&geometry=-121.88999,39.51118,-121.23980,39.98464&geometryType=esriGeometryEnvelope&inSR=4326&outSR=4326&spatialRel=esriSpatialRelIntersects&outFields=FIRE_YEAR_INT&returnGeometry=true&maxAllowableOffset=0.0005&f=geojson |
| 4 | **viirs** (12 calls) | `https://firms.modaps.eosdis.nasa.gov/api/area/csv/<MAP_KEY>/VIIRS_SNPP_SP/-121.8900,39.5112,-121.2398,39.9846/5/2018-11-02`; the same for start dates 11-07, 11-12, 11-17, 11-22 (5 days each) and `/1/2018-11-27`; then all 6 again with `VIIRS_NOAA20_SP` |
| 5 | **goes** (428 hours; G17 from 11-15) | https://noaa-goes16.s3.amazonaws.com/?list-type=2&prefix=ABI-L2-FDCC/2018/312/00/ ✔ … through `2018/329/19/`; https://noaa-goes17.s3.amazonaws.com/?list-type=2&prefix=ABI-L2-FDCC/2018/319/00/ ✔ … through `2018/329/19/`. Keeps the scan nearest to :00 each hour |
| 6 | **weather** (GRIDMET, 2018-10-26 → 2018-11-27) | `https://thredds.northwestknowledge.net/thredds/dodsC/agg_met_{vpd,fm100,fm1000,bi,erc,rmin,vs,th,tmmx}_1979_CurrentYear_CONUS.nc` (9 OPeNDAP reads of the box; add `.html` for the query form) |
| 7 | **hrrr** (143 analyses + 18 forecasts) | Analyses every 3 h, 2018-11-08 00z → 2018-11-25 18z: https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20181108/conus/hrrr.t00z.wrfsfcf00.grib2 (+ `.idx` ✔) … ; forecasts 12z+9h for each day 11-08 → 11-25: https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20181108/conus/hrrr.t12z.wrfsfcf09.grib2 … |
| 8 | **terrain** (1 tile ✔) | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N39_00_W122_00_DEM/Copernicus_DSM_COG_10_N39_00_W122_00_DEM.tif |
| 9 | **landfire** (LF2016, 90 m) | `https://lfps.usgs.gov/api/job/submit?Email=<LANDFIRE_EMAIL>&Layer_List=LF2016_FBFM40;LF2016_CC;LF2016_CH;LF2016_CBH;LF2016_CBD&Area_of_Interest=-121.8900%2039.5112%20-121.2398%2039.9846&Output_Projection=4326&Resample_Resolution=90`, then poll `/api/job/status?JobId=…` and download `outputFile` |
| 10 | **tessera** | the 2017 embedding (year before the fire) for the box, through `GeoTesseraZarr().iter_region(box, 2017)` (store: https://data.source.coop/tessera/tessera) |
| 11 | **suppression**, spatial (1,793 line features in the box ✔) | https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/National_Incident_Feature_Service_2018/FeatureServer/1/query?where=1%3D1&geometry=-121.88999,39.51118,-121.23980,39.98464&geometryType=esriGeometryEnvelope&inSR=4326&outSR=4326&spatialRel=esriSpatialRelIntersects&outFields=*&f=geojson (and `/2/query` with the same box for burnouts and drops; paged 1,000 at a time). Daily maps are rebuilt from GDB_FROM_DATE / GDB_TO_DATE |
| 12 | **suppression**, ICS-209 | https://ndownloader.figshare.com/files/38766504 (`ics209plus-wildfire.zip`), rows matched by IRWIN `75E64DB8…` (fallback: name + year + box ±0.25°) |
| 13 | **human** (OSM 2018-01-01 ✔) | https://download.geofabrik.de/north-america/us-west-180101.osm.pbf, cut to the box with osmium |
| = | **Result** | https://huggingface.co/datasets/Javyn123/wildfire-spread-2014-2025/tree/main/fires/F2003403265 (grid.json, labels, history, viirs + points.csv.gz, goes, weather, hrrr, terrain, landfire, tessera, suppression, human) |

### Recipe for any other fire
1. Look the fire up in `catalog/fires.csv`: `incident` (the IRWIN ID), `year`, `first`, `last`, box.
2. Read the grid box from `fires/<id>/grid.json`, or rebuild it: box ± 9.6 km at 150 m.
3. Fill the templates in section 2 with the box and dates:
   - GRIDMET: first − 14 days → last + 2 days
   - VIIRS: first − 7 days → last + 2 days, in 5-day calls
   - HRRR and GOES: first − 1 day → last
   - Perimeter archive: pick it by year (table 2a)
   - LANDFIRE: pick the version by year
   - Tessera: use year − 1
   - ICS-209: only for fires in 2020 or earlier
   - OSM: the region from the box centre, snapshot of 1 January of the fire's year
