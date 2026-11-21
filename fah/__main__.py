"""
Pull the upstream raw data of every fire, one channel at a time, into a Hugging Face dataset repo.
Built for Kaggle (notebooks/kaggle_data_grab.ipynb) but runs anywhere with internet access.

  python -m fah --catalog all --hf-repo you/wildfire-raw-2014-2025              # the fire list (first)
  python -m fah --channel firms --hf-repo you/wildfire-raw-2014-2025 --work /tmp/raw
  python -m fah --channel goes  --hf-repo you/wildfire-raw-2014-2025 --max-hours 11.5
  python -m fah --status --hf-repo you/wildfire-raw-2014-2025
  python -m fah --links  --hf-repo you/wildfire-raw-2014-2025          # per-fire links files
  python -m fah --extents --hf-repo you/wildfire-raw-2014-2025         # after firms: windows, ignition
  python -m fah --repack --channel perimeters --hf-repo you/wildfire-raw-2014-2025  # gzip old GeoJSON
  python -m fah --channel history --ids F2003403265 --dry-run --work /tmp/raw   # local test

How a run works
  1. The fire list is the repo's own catalog/fires.csv: --catalog scan crawls every record of the NIFC perimeter
     archives (GeoMAC 2014-2019, NIFS / Operational Data Archive 2018-2025) and the WFIGS incident of each, untouched,
     into catalog/raw/; --catalog build applies the gates (fah.core.incidents: 2014-2025, continental US, wildfire,
     >= 1,000 acres, >= 3 burn days). Each fire's region is the box of its perimeter records pushed out 32 km on
     every side (Fire.region), so any 64 km tile centred anywhere in the fire can be cut from the files. Its time
     window is the perimeter days stretched over the FIRMS activity that leads into and out of them, plus a day on
     each side: run firms first, then --extents, which writes catalog/fire_extents.csv
     (window, ignition point, region); the time-dependent channels (gridmet, goes, hrrr, hrrr_next, gfs,
     vegetation, viirs) read it.
  2. Resume: fires already in manifest/<channel>/*.csv of the target repo (ok / empty / n/a) are skipped;
     errors are retried. A row also has to match the fire's current window and region (columns window,
     region), so widening either sends the fire back to the queue.
  3. Each fire is fetched into a staging folder, then moved into the current batch. A batch is uploaded as one
     commit when it reaches --batch-gb or --batch-fires, when disk runs low, or before the deadline, together
     with its manifest file; the local copy is deleted right after. A failed upload keeps the batch on disk.
     A fire fetched again replaces its folder: the same commit deletes the channel's older files of that fire.
  4. Files: GeoJSON is gzipped with a fixed header (.geojson.gz); viirs is one viirs.nc per fire. Rows that
     list the older names (plain .geojson) count as not done; --repack gzips such files in place instead of
     fetching again.
  5. One-time shared downloads (archive geodatabases, ICS-209-PLUS, OSM snapshots) are deleted when the
     channel ends (OSM: after each region-year).
  6. Channels can run side by side, one process (or Kaggle copy) each, against the same repo: a channel writes
     only fires/<id>/<channel>/ and manifest/<channel>/, and every Hub write is retried when another commit or
     a rate limit gets in the way. Time-dependent channels refuse to start while some fire's window is not
     final (firms unfinished for it, or the catalog changed since --extents) unless --partial-windows.

Target repo layout
  README.md                          dataset card (docs/RAW_DATA_CARD.md)
  catalog/fires.csv                  --catalog build: the fires (gated incidents), one row each
  catalog/incidents.csv, gates.json  --catalog build: every incident with its gate values; the gate settings
  catalog/raw/perimeters, incidents  --catalog scan: every perimeter record and WFIGS incident, as served
  catalog/raw_index.csv              --links: one row per fire and channel (status, files, bytes)
  catalog/fire_extents.csv           --extents: each fire's time window, ignition point and region
  fires/<fire_id>/<channel>/...      raw files
  fires/<fire_id>/links.json, LINKS.md   --links: every channel of the fire, its files here and its upstream URLs
  manifest/<channel>/<batch>.csv     fire_id, status, files, bytes, sources, fetched_utc, error, window, region
  shared/ics209/...                  source files used by many fires, stored once

Needs HF_TOKEN (write access) unless --dry-run, FIRMS_MAP_KEY for firms, EARTHDATA_TOKEN (free Earthdata
Login token) for viirs, vegetation and landcover, and Google Earth Engine access for gfs (GEE_SERVICE_ACCOUNT_KEY,
the service account's JSON key, or `earthengine authenticate` on this machine; GEE_PROJECT, the Cloud project
registered for Earth Engine). No other keys.
"""

from __future__ import annotations

import argparse
import logging
import sys

from fah.channels.registry import ChannelRegistry
from fah.commands.catalog import CatalogBuilder
from fah.commands.check import SetupCheck
from fah.commands.extents import ExtentsBuilder
from fah.commands.links import LinksBuilder
from fah.commands.repack import Repacker
from fah.commands.run import ChannelRun
from fah.commands.settings import RunSettings
from fah.commands.status import StatusReport
from fah.core.errors import ChannelUnavailable, ExitCode, SetupError
from fah.hub.catalog import FireCatalog


def main(argv: list[str] | None = None) -> int:
    """The command line: parse, log to stdout (warnings and errors to stderr), run one command. -> exit code."""
    ap = argparse.ArgumentParser(
        prog="python -m fah", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--channel", choices=ChannelRegistry().names())
    ap.add_argument("--status", action="store_true", help="per-channel progress of --hf-repo")
    ap.add_argument(
        "--check", action="store_true", help="check HF_TOKEN can write to --hf-repo, and which keys are set"
    )
    ap.add_argument(
        "--links",
        action="store_true",
        help="write fires/<id>/links.json + LINKS.md (every channel's files and sources) from the manifests",
    )
    ap.add_argument("--links-chunk", type=int, default=500, help="fires per commit for --links")
    ap.add_argument(
        "--extents",
        action="store_true",
        help=f"write {FireCatalog.EXTENTS_FILE} (each fire's time window, ignition point, region) from the "
        "firms files; run after firms",
    )
    ap.add_argument(
        "--repack",
        action="store_true",
        help=f"gzip the plain .geojson files already in --hf-repo for --channel "
        f"({', '.join(Repacker.CHANNELS)}; default perimeters)",
    )
    ap.add_argument(
        "--extents-file",
        help=f"read the extents from this local file instead of {FireCatalog.EXTENTS_FILE} in --hf-repo",
    )
    ap.add_argument(
        "--partial-windows",
        action="store_true",
        help="time-dependent channels: run although firms is not finished for every fire (those are fetched again)",
    )
    ap.add_argument(
        "--local-manifests",
        help="--links / --extents: read manifests (and firms files) from local --dry-run batches under this folder",
    )
    ap.add_argument("--hf-repo", help="target dataset repo, e.g. you/wildfire-raw-2014-2025")
    ap.add_argument("--work", default="raw_work", help="local scratch (staging, batches, one-time downloads)")
    ap.add_argument(
        "--archive",
        choices=["geomac", "nifs", "both", "all"],
        default="all",
        help="only the catalog's fires of this perimeter archive",
    )
    ap.add_argument("--catalog-file", help="read the fire list from this local fires.csv instead of --hf-repo's")
    ap.add_argument(
        "--catalog",
        choices=["scan", "build", "all"],
        help="crawl the NIFC databases (scan) and / or write catalog/fires.csv from them (build)",
    )
    ap.add_argument("--years", default="2014-2025", help="--catalog: years to crawl and keep, e.g. 2014-2025")
    ap.add_argument("--min-acres", type=float, default=1000.0, help="--catalog gate: largest reported size")
    ap.add_argument("--min-days", type=int, default=3, help="--catalog gate: burn days with a perimeter")
    ap.add_argument("--ids", nargs="*", help="only these fire ids")
    ap.add_argument("--limit", type=int, help="at most N fires this run")
    ap.add_argument("--batch-gb", type=float, default=2.0)
    ap.add_argument("--batch-fires", type=int, default=100)
    ap.add_argument("--min-free-gb", type=float, default=5.0, help="upload early when free disk falls below this")
    ap.add_argument("--max-hours", type=float, help="stop cleanly before this many hours (Kaggle: 11.5)")
    ap.add_argument(
        "--threads",
        type=int,
        default=16,
        help="parallel downloads for goes / hrrr / hrrr_next / gfs; viirs uses up to 8 worker processes",
    )
    ap.add_argument("--private", action="store_true", help="create the target repo private")
    ap.add_argument("--redo", action="store_true", help="ignore manifests: fetch every fire again")
    ap.add_argument("--dry-run", action="store_true", help="no upload; batches are kept under WORK/batch")
    ap.add_argument("--keep", action="store_true", help="keep local batches after upload")
    ap.add_argument("--keep-cache", action="store_true", help="keep one-time downloads at the end")
    ap.add_argument(
        "--max-errors-in-row",
        type=int,
        default=10,
        help="stop the channel (exit 4) after this many fires fail in a row; 0 = never",
    )
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    out = logging.StreamHandler(sys.stdout)
    out.addFilter(lambda r: r.levelno < logging.WARNING)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[out, err], force=True)

    if a.status and not a.hf_repo:
        ap.error("--status needs --hf-repo")
    if a.links and not a.hf_repo and not a.dry_run:
        ap.error("--links needs --hf-repo (or --dry-run)")
    if a.extents and not a.hf_repo and not a.local_manifests:
        ap.error("--extents needs --hf-repo (or --local-manifests with --dry-run)")
    if a.repack and not a.hf_repo:
        ap.error("--repack needs --hf-repo")
    if a.catalog and not a.hf_repo and not a.dry_run:
        ap.error("--catalog needs --hf-repo (or --dry-run)")
    if not any((a.check, a.status, a.links, a.extents, a.repack, a.catalog)):
        if not a.channel:
            ap.error("--channel is required")
        if not a.hf_repo and not a.dry_run:
            ap.error("--hf-repo is required (or --dry-run)")

    settings = RunSettings.from_args(a)
    try:
        if a.check:
            return SetupCheck(settings).execute()
        if a.catalog:
            return CatalogBuilder(settings).execute()
        if a.status:
            return StatusReport(settings).execute()
        if a.links:
            return LinksBuilder(settings).execute()
        if a.extents:
            return ExtentsBuilder(settings).execute()
        if a.repack:
            return Repacker(settings).execute()
        return ChannelRun(settings).execute()
    except SetupError as ex:
        print(f"SETUP ERROR: {ex}", file=sys.stderr, flush=True)
        return ExitCode.SETUP
    except ChannelUnavailable as ex:
        print(ex, file=sys.stderr, flush=True)
        return ExitCode.FAILED


if __name__ == "__main__":
    sys.exit(main())
