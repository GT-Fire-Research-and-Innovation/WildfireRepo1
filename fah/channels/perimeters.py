"""perimeters: every raw record of the fire's incident in the NIFC perimeter archives, before any cleaning."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.sources.nifc import NifcArchive
from fah.storage.geojson import GzipGeoJson


class PerimetersChannel(PerFireChannel):
    """records.geojson.gz: the incident's records in each archive layer of its years (all attributes, full geometry);
    _source_layer / _source_kind say where each came from. Queried over the perimeters' box, which holds every record
    of the fire itself (a wider box would only add neighbours)."""

    name = "perimeters"
    region_cut = False
    required_files = frozenset({"records.geojson.gz"})

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        nifc = NifcArchive.for_run(ctx)
        irwin, name_uid = fire.incident_key()
        box = fire.box
        keep, sources = [], []
        for year in fire.years():
            for kind, url in ctx.memo(("perim_layers", year), lambda y=year: nifc.perimeter_layers(y)):
                feats = nifc.features(url, box)
                sources.append(nifc.source_link(url, box))
                for f in feats:
                    if self._is_this_fire(f.get("properties") or {}, irwin, name_uid):
                        f = dict(f)
                        f["properties"] = dict(
                            f.get("properties") or {}, _source_layer=nifc.layer_label(url), _source_kind=kind
                        )
                        keep.append(f)
        shutil.rmtree(ctx.tmp(), ignore_errors=True)
        GzipGeoJson.write(out / "records.geojson.gz", keep)
        return ("ok" if keep else "empty"), sources

    @staticmethod
    def _is_this_fire(properties: dict, irwin: str | None, name_uid: tuple[str, str] | None) -> bool:
        """By IRWIN id, or for name-keyed incidents by (normalised name, unique fire id)."""
        p = {k.lower(): v for k, v in properties.items()}
        if irwin:
            return Fire.normalise_irwin(p.get("irwinid")) == irwin
        name = re.sub(r"\s+", " ", str(p.get("incidentname") or "").lower()).strip()
        uid = str(p.get("uniquefireidentifier") or p.get("localincidentidentifier") or "")
        return (name, uid) == name_uid
