"""suppression: every archived version of every fire line, burnout and drop in the region (NIFS / ODA)."""

from __future__ import annotations

import shutil
from pathlib import Path

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.sources.nifc import NifcArchive
from fah.storage.geojson import GzipGeoJson


class SuppressionChannel(PerFireChannel):
    """event_lines.geojson.gz, event_polygons.geojson.gz: all attributes, incl. GDB_FROM_DATE / GDB_TO_DATE,
    CreateDate and DateCurrent, for each archive year the fire burned in. A model may use a version only from the
    burn day after the latest of those stamps (docs/RAW_DATA_CARD.md)."""

    name = "suppression"
    required_files = frozenset({"event_lines.geojson.gz", "event_polygons.geojson.gz"})
    FIRST_YEAR = 2018

    def skip_reason(self, fire: Fire) -> str | None:
        if fire.span().end.year < self.FIRST_YEAR:
            return "no spatial suppression archive before 2018"
        return None

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        nifc = NifcArchive.for_run(ctx)
        box = fire.region()
        lines, polys, sources = [], [], []
        for year in [y for y in fire.years() if y >= self.FIRST_YEAR]:
            for url, kind in ctx.memo(("supp", year), lambda y=year: nifc.event_layers(y)):
                feats = nifc.features(url, box)
                sources.append(nifc.source_link(url, box))
                for f in feats:
                    f = dict(f)
                    f["properties"] = dict(f.get("properties") or {}, _source_layer=nifc.layer_label(url))
                    (lines if kind == "line" else polys).append(f)
        shutil.rmtree(ctx.tmp(), ignore_errors=True)
        GzipGeoJson.write(out / "event_lines.geojson.gz", lines)
        GzipGeoJson.write(out / "event_polygons.geojson.gz", polys)
        return ("ok" if lines or polys else "empty"), sources
