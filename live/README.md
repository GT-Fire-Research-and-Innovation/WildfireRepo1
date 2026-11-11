# SpreadWatch: live wildfire map

A public map for crews and planners: live NIFC perimeters, incident locations and VIIRS satellite
hotspots, plus the project's 24-hour spread forecast for every actively burning fire.

- **Map app:** `site/` (static: Leaflet, no build step). The browser pulls perimeters, incidents and
  hotspots straight from the NIFC/Esri services every 5 minutes.
- **Forecasts:** `.github/workflows/live-forecast.yml` runs every 2 hours on GitHub Actions:
  `live/publish.py` lists active fires, builds each one's inputs from live sources (DEM, LANDFIRE,
  GRIDMET, FIRMS VIIRS, latest HRRR forecast) with the training pipeline's own code, runs the model,
  and writes `site/data/index.json` + `site/data/forecast/<fire>.json` (the forecast on the model's own 150 m cells). The workflow then deploys
  `site/` to GitHub Pages.
- **Retraining feed:** each run also archives every new NIFC perimeter version to the
  `perimeter-archive` branch (`live/wfigs_poller.py`), the raw material for adding new fires to the
  training set.
- **Off season:** when no fire is actively burning, the app shows a labelled replay example
  (`site/data/example/`, made with `live/make_example.py`).

## One-time setup (repository owner)

1. **Pages:** Settings → Pages → Build and deployment → Source: **GitHub Actions**.
2. **Secrets:** Settings → Secrets and variables → Actions → New repository secret:
   - `FIRMS_MAP_KEY`: your NASA FIRMS map key
   - `LANDFIRE_EMAIL`: the contact email the LANDFIRE service requires
3. **First run:** Actions → Live forecast → Run workflow. The site appears at
   `https://<owner>.github.io/<repo>/` (the run's deploy step prints the link).

Until trained weights are added, forecasts use the physics-only model (Rothermel + crown fire +
embers, test AP 0.093). Add the router for the real forecasts (test AP 0.336):

## Publishing the trained models (on PACE-ICE)

```bash
cd ~/scratch/WildfireRepo1 && git pull
python live/export_models.py --runs ~/scratch/wildfire/my/runs --combine combine_unet3 --out live/models
git add -f live/models && git commit -m "live models" && git push
```

`combine_unet3` is tonight's router refit; use `--combine combine` for the first router. `git push`
from ICE needs a GitHub personal access token (fine-grained, this repository, Contents: read and
write), typed when git asks for a password. The next scheduled run picks the models up.

## Run locally

```bash
python live/publish.py --site site --work /tmp/live --max-fires 3     # needs FIRMS_MAP_KEY, LANDFIRE_EMAIL
cd site && python -m http.server 8000                                  # open http://localhost:8000
```

Experimental research forecast, not an official product. The app says so on every view.

## Continuous learning

The model keeps learning from fires as they happen:

1. **Collect** (every 2 h, GitHub Actions): every new NIFC perimeter version is saved to the
   `perimeter-archive` branch. Perimeters come from NIFC's Current service *and* the Year-to-Date
   service, because Current drops active fires whose incident record went stale.
2. **Catalog** (weekly, ICE): `pipeline/archive_catalog.py` turns the archive into training fires with
   the same cleaning as the main catalog (wildfires only, no hand sketches, one perimeter per burn day,
   no shrinking re-maps, 3+ burn days, 300+ acres, finished growing).
3. **Build**: `pipeline/build_multiyear.py` builds each new fire's inputs (terrain, fuels, weather,
   satellite) into the training data.
4. **Fine-tune**: the live model continues training on old + new fires (`--init`, 6 epochs, low
   learning rate). The newest 20% of new fires are held out.
5. **Gate** (`live/promote.py`): promote only if the candidate is no worse on the frozen 2024-25 test
   fires (AP within 0.005) **and** at least as good on the held-out new fires.
6. **Ship**: router refit, `live/export_models.py`, push; the website uses it from its next run.

Run it with `bash scripts/ice_continual.sh` (weekly; every step is a chained Slurm job).
