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

## Guide for newcomers
**The task.** Next-day fire spread prediction asks one question: given everything known at the end of one day,
which land will burn the next day? A model needs five kinds of input to answer it:

- **Active fire:** where the fire is burning right now.
- **Burned area:** what has already burned. The next day's new burned area is also the answer a model is graded on.
- **Weather:** what pushes the fire: wind, heat, dry air, rain.
- **Land:** what there is to burn, and the shape of the ground it burns over.
- **Human:** what people did: control lines, roads, homes to protect.

Each folder in this repo supplies one or more of these. The sections below say what each source holds, how to tell
whether it is good enough, what one real fire looks like in it, and which inputs matter most.

### How to tell whether a data source is good
Ask six questions of any source. None needs a background in fire science.

| Question | Why it matters | Good sign | Warning sign |
|---|---|---|---|
| How big is one pixel? | A pixel is the smallest square a map can describe. Smaller pixels show more detail: a 30 m pixel can show a road, a 4 km pixel blends a whole hillside into one number. | 30 m or less for fuel and ground; 375 m or less for fire | Several kilometres for anything that changes across a hillside. Weather is the exception: air changes smoothly, so 3 km is fine. |
| How often is there a new value? | A fire can double in a few hours. | Every hour or every few hours for fire and weather | Once a year for something that changes daily |
| How many fires and years does it cover? | A source missing for half the fires forces you to drop those fires or guess. | Every fire, 2014–2025 | Only some years (the tables below say which) |
| Can it see through clouds, smoke and darkness? | Satellites that measure heat work at night; ones that measure visible light do not; thick cloud blocks both. | Heat bands, night passes, model data that covers every hour | Treating "no fire detected" as "no fire" when the sky was cloudy |
| Was it known before the moment you predict? | Using information from after that moment is cheating, called a **leak**: like grading a weather forecaster with tomorrow's newspaper. The model looks brilliant and then fails in real use. | Forecasts issued beforehand; maps made before the fire | The next day's observed weather; yearly maps made from imagery after the fire |
| Measured or modelled? | Measurements (satellites, mapping flights) are direct but have gaps. Models (HRRR, GFS, gridMET) fill every place and hour but can be wrong. | Knowing which one you are using | Treating model output as ground truth |

**Signs of trouble inside the files themselves**, all of which show up in the Camp Fire example below:
- Values that go backwards. A burned area cannot shrink, so a later map that is smaller is a mapping difference, not real.
- Counts that are not whole numbers, such as 1,645.3 people. They were estimated, not counted.
- Long runs of fill values, which mark "no data", not zero.
- Times recorded in local time but labelled as UTC, or the reverse. Check them against another source.

### Every source at a glance
"Snapshot" means one copy per fire, taken before it started. "Time series" means many copies, one per time step, across
the fire's window.

| Kind | Folder | What it tells you | Years here | Pixel size | Snapshot or time series |
|---|---|---|---|---|---|
| Active fire | `firms` | Points where a satellite saw fire burning, with how much heat each gave off (fire radiative power) | 2014–2025; a second satellite from 2018 | 375 m points | Time series: every satellite pass, about 2 a day per satellite (day and night) |
| Active fire | `goes` | Fire seen from a weather satellite parked over the Americas: coarse, but every hour | mid-2017–2025 | 2 km at best, about 3–4 km over the West | Time series: every hour |
| Active fire | `viirs` | The raw satellite images behind `firms`: five bands of reflected light and heat at 375 m, one more at 750 m, day and night | 2014–2025; second satellite from 2018, third from 2023 | 375 m (one band 750 m) | Time series: every pass, about 2–6 a day |
| Burned area | `perimeters` | The fire's outline each time an aircraft or crew mapped it | 2014–2025 | Exact outlines (no pixels) | Time series: each mapping, median 8 per fire, about a day apart |
| Weather | `hrrr` | Wind, gusts and humidity as they were, from NOAA's high-resolution weather model | mid-2014–2025 | 3 km | Time series: every 3 hours |
| Weather | `hrrr_next` | NOAA's forecast for the next 36 hours, made at noon UTC each day: wind, gusts, humidity, temperature, rain | mid-2014–2025; 36 h ahead from mid-2018 (15–18 h before) | 3 km | Time series: one forecast a day, a step every 3 hours |
| Weather | `gfs` | NOAA's global forecast out to 10 days, coarse, with extra fields for smoke plumes and the air's mixing height | mid-2015–2025 | about 28 km | Time series: one forecast a day, a step every 3 hours |
| Weather | `gridmet` | Daily weather and the fire-danger numbers crews use, including how dry dead wood and grass are | 2014–2025 (record from 1979) | about 4 km | Time series: daily |
| Land | `terrain` | Height of the ground; slope and the direction a slope faces follow from it | one map (measured 2011–2015) | about 30 m | Snapshot |
| Land | `landfire` | What fuel is on the ground (40 standard fuel types) and how the tree canopy is built | fires 2017–2025 | 30 m | Snapshot: one version per fire |
| Land | `vegetation` | How green the plants are, from satellite (NDVI, EVI2) | 2014–2025 | about 463 m (often called 500 m) | Time series: a new 16-day average every 8 days |
| Land | `landcover` | Broad land type: forest, shrubland, grassland, cropland, city | 2014–2025 | about 463 m | Snapshot: the year before the fire |
| Land | `lulc` | Detailed land type: trees, rangeland, crops, built-up, water | fires 2017–2025 (maps 2017–2023) | 10 m | Snapshot: the year before the fire (2017 fires: 2017) |
| Land | `history` | Outlines of earlier fires in the region | many decades back (Camp Fire: to 1900) | Exact outlines | Snapshot: every fire of earlier years |
| Land | `tessera` | A pointer to an AI summary of a year of satellite imagery (no data stored) | fires 2018–2025 | 10 m | Snapshot: the year before the fire |
| Human | `suppression` | Lines crews built or planned to stop the fire: bulldozer lines, hand lines, roads used as lines | 2018–2025 | Exact lines and shapes | Time series: every saved version, with the dates it was current |
| Human | `osm` | Roads, trails, buildings, power lines, rivers (OpenStreetMap) | 2014–2025 | Exact shapes | Snapshot: 1 January of the fire's year |
| Human | `ics209` | Incident reports: percent contained, number of crew, homes lost | 2014–2020 | A table, no map | Time series: about twice a day |

### One fire from start to finish: the Camp Fire
`fires/F2003403265/`. The Camp Fire started around 6:30 a.m. on 8 November 2018 in Butte County, California,
destroyed the town of Paradise, and burned about 153,300 acres (620 km²). In this repo:

- **Ignition point:** 39.770° N, 121.551° W, from the first satellite pass that saw it, at 11:50 a.m. local time,
  about five hours after it started. This is the earliest observation, not the true start.
- **Time window:** 7 to 26 November 2018, 20 days. The perimeters run from 8 to 25 November.
- **Region:** 103 km × 99 km around the fire.

**Day by day.** Every number below comes from this fire's files (`perimeters`, `firms`, `ics209`).

| Date (2018) | Infrared map (acres) | Other map (acres) | Day detections | Night detections | Fire power, day + night (MW) | Contained | Crew |
|---|---|---|---|---|---|---|---|
| Thu 8 Nov | 54,586 | 70,454 | 1,142 | 0 | 80,696 | 5% | 1,645 |
| Fri 9 Nov | 109,503 | 111,552 | 570 | 1,834 | 61,438 | 20% | 1,095 |
| Sat 10 Nov | 114,866 | – | 201 | 552 | 9,627 | 25% | 1,553 |
| Sun 11 Nov | – | 130,394 | 313 | 698 | 19,387 | 25% | 1,631 |
| Mon 12 Nov | 138,127 | 128,816 | 338 | 742 | 14,738 | 30% | 1,774 |
| Tue 13 Nov | 145,254 | 136,636 | 0 | 399 | 2,263 | 35% | 2,370 |
| Wed 14 Nov | 149,512 | 142,807 | 114 | 526 | 5,486 | 40% | 2,367 |
| Thu 15 Nov | 151,575 | – | 105 | 90 | 1,214 | 45% | 2,401 |
| Fri 16 Nov | 149,689 | 148,238 | 107 | 305 | 3,151 | 55% | 2,406 |
| Sat 17 Nov | 150,690 | 150,628 | 64 | 187 | 1,802 | 60% | 1,833 |
| Sun 18 Nov | 151,429 | 151,472 | 31 | 99 | 596 | 66% | 1,702 |
| Mon 19 Nov | 152,761 | 151,515 | 21 | 122 | 666 | 70% | 1,291 |
| Tue 20 Nov | 153,438 | 152,688 | 14 | 76 | 239 | 80% | 1,003 |
| Wed 21 Nov | – | 153,336 | 0 | 19 | 36 | 90% | 517 |
| Thu 22 Nov | – | – | 0 | 0 | 0 | 95% | 591 |
| Fri 23 Nov | – | – | 0 | 0 | 0 | 95% | 441 |
| Sat 24 Nov | – | – | 0 | 0 | 0 | 100% | 368 |
| Sun 25 Nov | – | 153,336 | 0 | 0 | 0 | 100% | 215 |

How to read it:
- **Map columns:** the last map of each day, split into maps from infrared flights (the most reliable kind) and maps by
  other methods.
- **Detections:** VIIRS 375 m pixels flagged as burning. Night passes happen around 1–3 a.m., so they count toward
  the next date; that is why 8 November has none.
- **Fire power:** the heat the satellites measured, in megawatts, summed over every detection that day.
- **Contained** and **Crew:** the last incident report of the day. Times are local (Pacific); perimeter times in the
  archives are sometimes entered in local time and stored as UTC, so trust the date more than the hour.

What this one fire teaches:
- **Most of the fire burned on day one.** The first map already showed 54,586 acres, about a third of the final
  area, and the day's fire power was the highest of the whole fire. The most important day is the one with the least
  data before it.
- **Maps disagree.** On 12 November the infrared map says 138,127 acres and a later map by other methods says 128,816.
  Burned land cannot un-burn: the second map is a different method, not the fire shrinking. Prefer infrared maps.
- **Fires burn at night.** On 9 November, night detections (1,834) outnumbered day detections (570) three to one.
- **No detection does not mean no fire.** On 13 November there were 0 day detections but 399 that night. Smoke or
  cloud likely hid it during the day; `firms` alone cannot say which, so a cloud mask is needed.
- **Estimated numbers look precise.** The incident reports give crew sizes like 1,645.3 people: ICS-209-PLUS fills in
  some values by estimation, so they are not exact counts.
- **The human record helps, and can leak.** Crews mapped 1,793 line records for this fire, as in the table below.
  Each record carries the time it was saved, and a model may see it only from the day after. The 41 records
  marked "Uncontrolled Fire Edge" show why: a day later they are just yesterday's edge, but on the day they were
  drawn they trace where the fire got to, which is the answer. From 19 Nov records were saved in batches of about
  400 every two days, so a line built days earlier counts only from the day after its batch was saved.

**What each folder holds for this fire**

| Folder | Snapshot or series | For the Camp Fire |
|---|---|---|
| `perimeters` | Series | 28 mapped outlines, 8–25 November: 12 from infrared flights, 16 by other methods |
| `firms` | Series | 8,669 detections inside the fire's box from 70 satellite passes; FIRMS fetched 26 October to 9 December |
| `goes` | Series | 480 hourly scans from GOES-16; 288 from GOES-17, whose data starts 15 November |
| `viirs` | Series | 130 satellite passes: 65 from S-NPP and 65 from NOAA-20 |
| `hrrr` | Series | 160 weather snapshots (every 3 hours) |
| `hrrr_next` | Series | 20 daily forecasts × 12 steps, out to 36 hours |
| `gfs` | Series | 20 daily forecasts × 80 steps, out to 10 days |
| `gridmet` | Series | 34 days, 25 October to 27 November, 13 variables |
| `vegetation` | Series | 7 greenness composites, starting 8 October to 25 November |
| `suppression` | Series | 1,793 line records: 933 completed bulldozer line, 282 proposed bulldozer line, 237 roads used as line, 170 completed hand line, 66 completed line, 64 planned secondary line, 41 uncontrolled fire edge |
| `ics209` | Series | 38 incident reports, 8–25 November |
| `landfire` | Snapshot | LANDFIRE 2016: fuel type and 4 canopy layers |
| `terrain` | Snapshot | One elevation map |
| `lulc` | Snapshot | The 2017 map |
| `landcover` | Snapshot | The 2017 map |
| `osm` | Snapshot | OpenStreetMap as of 1 January 2018 |
| `history` | Snapshot | 291 earlier fires in the region, 1900–2017; 75 since 2000 |
| `tessera` | Snapshot | A pointer to the 2017 embeddings |

### Ranked inputs, by kind
Each input below is something a model can be given, made from one or more folders. The rank says how much a
next-day spread model is likely to lose without it:

- **1, must have:** without it the model is guessing.
- **2, strong:** clearly improves predictions.
- **3, helpful:** improves them some of the time, or only for some fires.
- **4, optional:** worth testing, not worth depending on.
- **Skip:** another input says the same thing better, or the values are unreliable.
- **Never as input:** it contains the answer or comes from after the prediction moment. Fine for checking results.

**Active fire: where is it burning now?**

| Rank | Input | From | What it tells the model | Watch out for |
|---|---|---|---|---|
| 1 | Satellite fire detections, day and night | `firms` | Where the fire is burning, to 375 m, up to four times a day | Clouds and smoke hide fire; edge-of-view pixels are up to 4× larger |
| 2 | Clear / cloudy / not-seen mask | `viirs` (or NASA's fire-mask product, not fetched yet) | Whether "no detection" means no fire or could not see | Needs extra processing |
| 2 | Hourly fire detections and fire power | `goes` | When in the day the fire ran, and overnight activity | 2–4 km pixels; positions can be off by kilometres in mountains; only from mid-2017 |
| 2 | Overnight fire power | `firms`, `goes` | A fire still burning hard at night often makes a big run the next day | Night passes belong to the next date |
| 3 | How long each spot has been burning | `firms` | Separates the active edge from burned-out interior | Depends on cloud-free passes |
| 3 | Heat readings at the last pass (I4, I5) | `viirs` | Fire intensity before the satellite's own fire product | Raw data; most models do fine with `firms` instead |
| Skip | GOES estimated fire size and temperature | `goes` | Little: mostly missing values, and unstable where present | — |

**Burned area: what has burned, and the answer to predict**

| Rank | Input | From | What it tells the model | Watch out for |
|---|---|---|---|---|
| 1 | Burned area at the prediction moment | `perimeters` | The fire's current shape: the starting point of every prediction | Maps come about a day apart, at irregular times |
| 1 | New burned area the next day (the answer) | `perimeters` | What the model is graded on | Use infrared maps where possible; ignore apparent shrinking |
| 2 | Arrival-time map: when each spot burned | `perimeters` with `firms` / `goes` timing | Turns irregular maps into a clean day-by-day record | Has to be built; timing only as good as the detections |
| 2 | Detections added up since the start | `firms` | Burned area between maps, and where maps are missing | Overstates the area at swath edges |
| 3 | Burn-scar index from the satellite images | `viirs` | A daily burned-area estimate independent of the maps | Daytime and cloud-free passes only |
| 3 | How each map was made | `perimeters` | Lets the model trust infrared maps more | — |
| Never | The final perimeter | `perimeters` | It is the end of the story | Fine for analysis, never as input |

**Weather: what pushes the fire?**

| Rank | Input | From | What it tells the model | Watch out for |
|---|---|---|---|---|
| 1 | Next-day forecast: wind, gusts, humidity, temperature, rain | `hrrr_next` | The weather the fire will meet, as known the day before | Before mid-2018 the forecast stops short of the next afternoon (use `gfs`) |
| 2 | Today's wind, gusts and humidity | `hrrr` | The conditions that produced today's burning | — |
| 2 | Fire-danger numbers and dead-fuel dryness (ERC, BI, 100- and 1000-hour fuel moisture, vapor pressure deficit) | `gridmet` | How ready the fuel is to burn, built up over weeks | Daily values; use the prediction day's, never the next day's |
| 3 | Next-day forecast for early fires; plume and mixing-height fields | `gfs` | Fills about 750 fires before mid-2018; deep mixing goes with erratic growth | 28 km pixels: a few numbers per fire, not a map |
| 3 | Days since rain, 30-day rain, drought index (PDSI) | `gridmet` | Long-term dryness | Changes slowly |
| 3 | Critical fire-weather hours (low humidity with strong gusts) | `hrrr`, `hrrr_next` | How long conditions were extreme | Thresholds vary by region |
| 4 | Forecast steps beyond 2 days | `gfs` | Nothing for next-day prediction | Useful only for multi-day prediction |
| Never | The next day's observed weather | `hrrr`, `gridmet` | It is from the future | The most common leak in fire datasets |

**Land: what is there to burn, and how is the ground shaped?**

| Rank | Input | From | What it tells the model | Watch out for |
|---|---|---|---|---|
| 1 | Fuel type and its properties (how much fuel, how fine, how deep) | `landfire` | What burns and how fast | Missing for 2014–2016 fires; 2021–2022 fires get 2016 fuels, which miss recent burns |
| 1 | Slope, aspect, elevation | `terrain` | Fire races uphill and follows canyons | — |
| 2 | Physics-based spread estimate (Rothermel) | `landfire`, `terrain`, `hrrr`, `gridmet` | A textbook prediction of spread speed for the model to correct | Has to be computed |
| 2 | Tree canopy height, cover, base height, density | `landfire` | Whether fire can climb into the treetops | Same version issues as fuel type |
| 2 | Years since each spot last burned | `history` | Recently burned land has little left to burn | Fires earlier in the same year are missing |
| 3 | Plant greenness and moisture | `vegetation` | Green plants slow fire; cured grass speeds it | Use only averages that ended before the prediction day |
| 3 | Unburnable land: water, rock, farmland, city | `landfire`, `osm` | Natural barriers | — |
| 4 | Detailed 10 m land type | `lulc` | Small clearings and town edges | 2017 fires get a map from their own year |
| 4 | AI satellite summary | `tessera` | Learned description of the landscape | Only 2018+; not stored here |
| Skip | Broad 463 m land type | `landcover` | `landfire` and `lulc` say the same with far more detail | Kept only to match TS-SatFire |

**Human: what did people do?**

| Rank | Input | From | What it tells the model | Watch out for |
|---|---|---|---|---|
| 2 | Completed control lines at the prediction moment | `suppression` | Where crews have already stopped fuel from carrying fire | 2018+ only; show each version from the day after it was saved |
| 2 | Roads and trails | `osm` | Firebreaks, and where crews can reach | Rural maps can be incomplete |
| 3 | Planned control lines | `suppression` | Where crews intend to hold the fire | — |
| 3 | Buildings | `osm` | Homes at risk; crews concentrate there | — |
| 4 | Containment and crew size | `ics209` | How hard the fire was being fought | 2014–2020 only; one number per report, no map |
| 4 | Power lines | `osm` | Where fires start, not how they spread | — |
| 4 | Mapped fire edge from the day before | `suppression` | Where crews saw the edge yesterday | Repeats the previous perimeter |
| Never | Any line saved on or after the prediction day | `suppression` | Edge lines drawn that day trace where the fire went | Time each version by the latest of `GDB_FROM_DATE`, `CreateDate`, `DateCurrent` |

### Words used here
- **Active fire detection:** a satellite pixel flagged as burning at the moment of the pass.
- **Analysis vs forecast:** an analysis is a weather model's best estimate of conditions as they were; a forecast is
  its prediction, made in advance. Only forecasts are fair inputs for the day you are predicting.
- **Composite:** an image averaged over several days so clouds can be filled in.
- **ERC, BI:** energy release component and burning index, daily fire-danger numbers used by US fire agencies.
- **Fire radiative power (FRP):** heat given off by a burning pixel, in megawatts.
- **Fuel moisture (100-hour, 1000-hour):** how wet dead wood is; the hours describe how fast wood of that thickness
  dries out.
- **Fuel model:** one of 40 standard fuel types (grass, shrub, timber litter, …) that fire behaviour models use.
- **Infrared (IR) map:** a fire outline drawn from a heat camera flown over the fire, usually at night.
- **Leak:** information from after the prediction moment slipping into the inputs.
- **Perimeter:** the outline of the burned area at one moment.
- **Pixel size:** the side of the smallest square a gridded map describes.
- **Snapshot / time series:** one copy per fire, taken before it started / many copies over the fire's window.
- **Swath:** the strip of ground a polar-orbiting satellite sees on one pass, about 3,000 km wide for VIIRS.

## Channels
| channel | source | fires | files |
|---|---|---|---|
| history | NIFC Interagency Fire Perimeter History (every earlier fire in the region) | all | `history.geojson.gz` |
| perimeters | NIFC GeoMAC (2014–2019), NIFS (2018–2019), Operational Data Archive (2020–2025): every record of the fire's incident, before cleaning | all | `records.geojson.gz` (`_source_layer` per record) |
| firms | NASA FIRMS VIIRS 375 m active fire: S-NPP and NOAA-20 standard product, plus near-real-time for 2024+, from 14 days before the first perimeter to 14 days after the last | all | `<SOURCE>.csv.gz` |
| gridmet | GRIDMET daily, 13 variables (vpd, fm100, fm1000, bi, erc, rmin, vs, th, tmmx, pr, tmmn, sph, and the 5-day drought index pdsi), from 13 days before the window (pdsi: 23) to 1 day after it, as the THREDDS subset service returns it | all | `gridmet_<var>.nc` (13 files) |
| ics209 | ICS-209-PLUS (St. Denis et al.): incident and situation-report rows | ≤ 2020 | `incidents.csv`, `sitreps.csv` |
| suppression | NIFS / ODA EventLine and EventPolygon: every archived version (GDB_FROM_DATE / GDB_TO_DATE), with its CreateDate and DateCurrent; a model may use a version only from the burn day after the latest of the three | 2018+ | `event_lines.geojson.gz`, `event_polygons.geojson.gz` |
| tessera | GeoTessera v1.1: reference only (year, region, how to fetch) | 2018+ | `tessera_ref.json` |
| terrain | Copernicus GLO-30 DEM, native 1 arc-second | all | `dem_glo30.tif` |
| landfire | LANDFIRE FBFM40, CC, CH, CBH, CBD from the newest version before the fire: the stored 30 m pixels in EPSG:5070 from LANDFIRE's image services (snapped to its grid, nearest neighbour), LZW GeoTIFF as the server returns it | 2017+ | `landfire_<version>_<layer>.tif` (5 files) |
| lulc | Esri / Impact Observatory 10 m annual land use / land cover (io-lulc-annual-v02, Microsoft Planetary Computer), map of the year before the fire (2017 for 2017 fires, 2023 at most): the box's pixels of each source tile on its own UTM grid; class legend in the `classes` tag | 2017+ | `io_lulc_<year>_<tile>.tif` |
| landcover | MODIS MCD12Q1 v061 annual land cover of the year before the fire, every layer (LC_Type1-5, LC_Prop1-3 and assessments, QC, LW), on the sinusoidal 463 m grid; tiles joined pixel for pixel | all | `mcd12q1_<year>.nc` |
| vegetation | VIIRS VNP13A1 v002 16-day vegetation indices, every layer (NDVI, EVI, EVI2, reflectances, angles, composite day, VI quality, pixel reliability), each composite overlapping 16 days before the window to its end, sinusoidal 463 m grid | all | `vnp13a1.nc` (`time` = composite start) |
| goes | GOES-16/17/18/19 ABI L2 fire product (FDCC): the scan nearest each hour of the window, cropped on the fixed grid; Mask/Power/Area/Temp/DQF stay packed as in the source | window ≥ 2017-06 | `goes_<sat>.nc` |
| hrrr | NOAA HRRR GUST, UGRD/VGRD 10 m, RH 2 m, decoded from the files' own GRIB2 messages and cropped on the native 3 km grid: analyses every 3 h across the window (the 12z run's forecasts are in `hrrr_next`) | window ≥ 2014-08 | `hrrr_ana.nc` |
| hrrr_next | NOAA HRRR, the 12z run's forecasts every 3 h out to 36 h for each day (to 15 h before 2016-08-23, 18 h before 2018-07-12): GUST, UGRD/VGRD 10 m, RH 2 m, TMP 2 m, SPFH 2 m, APCP accumulated from the run start; `lead_h` and `run_time` per step | window ≥ 2014-08 | `hrrr_next.nc` |
| gfs | NOAA GFS 0.25° (TS-SatFire's forecast source) from Google Earth Engine `NOAA/GFS0P25`: the 12z run's forecasts every 3 h out to 10 days for each day of the window (to 36 h before 2017-07-09, as the collection holds them), every band of each image (2 m temperature in °C, specific / relative humidity, 10 m wind, precipitation accumulated over the last 1–6 h, precipitable water, cloud cover, radiation; more bands from 2025-01-15) on the native 0.25° grid; `lead_h` and `run_time` per step | window ≥ 2015-07 | `gfs.nc` |
| osm | OpenStreetMap, Geofabrik snapshot of 1 January of the fire's year: full extract of the region | all | `osm_<YYYY>-01-01.osm.pbf` |
| viirs | VIIRS L1B swaths of S-NPP (VNP02IMG / 03IMG / 02MOD / 03MOD), NOAA-20 (VJ1…) and NOAA-21 (VJ2…), every day and night granule over the region during the fire's window: I1–I5 (375 m) and M11 (750 m) as stored (scaled integers, quality flags, I4 / I5 brightness-temperature look-up tables) with each pixel's latitude, longitude and sensor / solar zenith, cut on the swath itself; pixels more than 2 km outside the region hold the source's fill value | all | `viirs.nc`: one group per granule (`<SAT>_<YYYY-MM-DD>T<HHMM>_<day\|night>`), the look-up tables once in `/luts` |

**Reading the files.** GeoJSON is gzipped: `json.load(gzip.open(path))`, or
`geopandas.read_file("/vsigzip/" + path)`. The gzip header has no name or time, so the same features always give the
same bytes. `viirs.nc` has one group per satellite pass: its root variables `granule`, `granule_start`, `satellite`
and `day_night` list them in time order, and each group holds that pass's I grid (`iy`, `ix`) and M grid (`my`, `mx`)
exactly as the source cut them. The I4 / I5 brightness-temperature tables are stored once in `/luts`; a group's
attribute `I04_brightness_temperature_lut` gives the path of its table (`ds[group.I04_brightness_temperature_lut]`
with netCDF4, or `xarray.open_dataset(path, group="luts")`).
```python
import netCDF4
ds = netCDF4.Dataset("fires/F2003403265/viirs/viirs.nc")
g = ds.groups[ds["granule"][0]]                     # first pass
v = g["I04"]; v.set_auto_maskandscale(False)        # the stored 16-bit codes
codes = v[:]
lut = ds[g.I04_brightness_temperature_lut]; lut.set_auto_maskandscale(False)
bt_i4 = lut[:][codes]                               # kelvin; mask codes == v._FillValue (no data / outside region)
```

The netCDF time series hold `time` (ns UTC), `lat`/`lon` per cell, and `source` (the exact file each step came from).
The sinusoidal files (landcover, vegetation) hold `x`/`y` pixel centres in metres and a CF `crs` variable; values are
stored as in the source, so apply each variable's `scale_factor` / `add_offset` yourself.

### TS-SatFire inputs
Every input of TS-SatFire (Zhao, Gerard and Ban 2024) has a raw counterpart here. TS-SatFire is built around VIIRS
images because two of its three tasks, finding active fire and mapping each day's burned area, ask a model to
recognise fire and burn scars in the images themselves; its next-day prediction task reuses the same images as input.

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
