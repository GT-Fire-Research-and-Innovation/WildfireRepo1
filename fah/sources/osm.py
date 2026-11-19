"""OpenStreetMap snapshots from Geofabrik (1 January of a year), cut to a box with osmium."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from fah.core.errors import ChannelUnavailable
from fah.core.geometry import BoundingBox
from fah.sources.http import HttpClient

log = logging.getLogger(__name__)


class Geofabrik:
    """Each snapshot (~2-3 GB) is downloaded once into cache_dir and deleted by the caller when done."""

    BASE = "https://download.geofabrik.de/north-america"
    # Fires outside the us-west region file: (state file, lon_min, lat_min, lon_max, lat_max).
    EAST_STATES = [
        ("us/kansas", -102.06, 36.99, -94.58, 40.01),
        ("us/oklahoma", -103.0, 33.6, -94.4, 37.0),
        ("us/texas", -106.65, 25.8, -93.5, 36.5),
        ("us/florida", -87.7, 24.4, -79.9, 31.0),
        ("us/nebraska", -104.06, 40.0, -95.3, 43.0),
        ("us/missouri", -95.8, 35.99, -89.1, 40.6),
    ]

    def __init__(self, http: HttpClient, cache_dir: Path) -> None:
        self.http, self.cache_dir = http, Path(cache_dir)

    @staticmethod
    def check_ready() -> None:
        if not shutil.which("osmium"):
            raise ChannelUnavailable(
                "osm needs osmium-tool (apt-get install -y osmium-tool, or conda install osmium-tool)"
            )

    @classmethod
    def region(cls, box: BoundingBox) -> str:
        """The Geofabrik file (without date) covering the box centre."""
        lat, lon = box.centre()
        west_tx = lat < 32.0 and lon > -106.65  # west Texas is in us-south, not us-west
        if lon < -102.05 and not west_tx:
            return "us-west"
        for name, x0, y0, x1, y1 in cls.EAST_STATES:
            if x0 <= lon <= x1 and y0 <= lat <= y1:
                return name
        raise RuntimeError(f"no Geofabrik region for fire at {lon:.2f},{lat:.2f}")

    @classmethod
    def snapshot_url(cls, region: str, year: int) -> str:
        return f"{cls.BASE}/{region}-{str(year)[2:]}0101.osm.pbf"

    def snapshot(self, region: str, year: int) -> Path:
        """The region's 1 January snapshot, downloaded when missing (aria2c when installed: faster, resumable)."""
        url = self.snapshot_url(region, year)
        path = self.cache_dir / f"{region.replace('/', '_')}-{str(year)[2:]}0101.osm.pbf"
        if not path.exists():
            log.info(f"  downloading {url}")
            if shutil.which("aria2c"):
                subprocess.run(
                    [
                        "aria2c",
                        "-q",
                        "-c",
                        "--file-allocation=none",
                        "-x",
                        "8",
                        "-s",
                        "8",
                        "-k",
                        "5M",
                        "--max-tries=20",
                        "--retry-wait=10",
                        "-d",
                        str(path.parent),
                        "-o",
                        path.name,
                        url,
                    ],
                    check=True,
                )
            else:
                self.http.download(url, path)
        return path

    @staticmethod
    def extract(snapshot: Path, box: BoundingBox, dst: Path) -> subprocess.CompletedProcess:
        """The full (unfiltered) extract of the box."""
        w, s, e, n = box
        return subprocess.run(
            [
                "osmium",
                "extract",
                "--overwrite",
                "-s",
                "smart",
                "-b",
                f"{w},{s},{e},{n}",
                str(snapshot),
                "-o",
                str(dst),
            ],
            capture_output=True,
            text=True,
        )
