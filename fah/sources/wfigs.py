"""WFIGS Incident Locations: one record per incident (IRWIN), every field (discovery time, cause, type, final acres,
containment / control / out times, fuel model, complex, ...) and the incident's point location."""

from __future__ import annotations

import pandas as pd

from fah.sources.arcgis import ArcGisClient


class WfigsIncidents:
    """Incidents fetched by IRWIN id, all fields as served, plus `_lon`, `_lat` (the incident point)."""

    URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations/FeatureServer/0"
    CHUNK = 200  # ids per request (the query string stays well under URL limits when posted)

    def __init__(self, arcgis: ArcGisClient) -> None:
        self.arcgis = arcgis

    def by_irwin(self, irwin_ids: list[str]) -> pd.DataFrame:
        """The incidents with these IRWIN uuids (without braces; matched in the service's '{UUID}' form)."""
        rows = []
        ids = sorted({i.upper() for i in irwin_ids if i})
        for k in range(0, len(ids), self.CHUNK):
            where = "IrwinID IN (" + ",".join(f"'{{{i}}}'" for i in ids[k : k + self.CHUNK]) + ")"
            for f in self.arcgis.records(self.URL, where=where, geometry=True):
                a = dict(f.get("attributes") or {})
                g = f.get("geometry") or {}
                a.update(_lon=g.get("x"), _lat=g.get("y"))
                rows.append(a)
        return pd.DataFrame(rows)
