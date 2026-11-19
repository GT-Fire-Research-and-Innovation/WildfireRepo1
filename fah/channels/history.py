"""history: every earlier fire in the region, from the Interagency Fire Perimeter History."""

from __future__ import annotations

from pathlib import Path

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.sources.arcgis import ArcGisClient
from fah.storage.geojson import GzipGeoJson


class HistoryChannel(PerFireChannel):
    """history.geojson.gz: every perimeter of the years before the fire's in its region (all fields)."""

    name = "history"
    required_files = frozenset({"history.geojson.gz"})
    SERVICE = (
        "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
        "InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0"
    )

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        where = f"FIRE_YEAR_INT < {int(fire.year)}"
        region = fire.region()
        feats = ArcGisClient(ctx.session).box_features(self.SERVICE, region, where=where, chunk=100)
        GzipGeoJson.write(out / "history.geojson.gz", feats)
        return ("ok" if feats else "empty"), [ArcGisClient.query_url(self.SERVICE, region, where=where)]
