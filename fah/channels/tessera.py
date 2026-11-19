"""tessera: a reference to the GeoTessera embeddings of the fire's region (no embedding data is stored)."""

from __future__ import annotations

import json
from pathlib import Path

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire


class TesseraChannel(PerFireChannel):
    """tessera_ref.json: the embedding year (the year before the fire), the region and how to fetch them."""

    name = "tessera"
    FIRST_YEAR = 2017
    LINKS = [
        "https://github.com/ucam-eo/geotessera",
        "https://data.source.coop/tessera/tessera",
        "https://registry.opendata.aws/tessera/",
    ]

    def skip_reason(self, fire: Fire) -> str | None:
        if int(fire.year) - 1 < self.FIRST_YEAR:
            return "GeoTessera v1.1 starts in 2017"
        return None

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        ref = {
            "fire_id": fire.fire_id,
            "fire_year": int(fire.year),
            "embedding_year": int(fire.year) - 1,
            "bbox_lonlat": list(fire.region()),
            "dataset": "GeoTessera v1.1 (128-d annual embeddings, 10 m)",
            "access": "from geotessera.store import GeoTesseraZarr; "
            "GeoTesseraZarr().iter_region(tuple(bbox_lonlat), embedding_year)",
            "links": self.LINKS,
            "note": "References only: the 10 m embeddings for one fire box are gigabytes. The built dataset's "
            "tessera.npz holds the 150 m block means.",
        }
        (out / "tessera_ref.json").write_text(json.dumps(ref, indent=1))
        return "ok", self.LINKS[:1]
