"""ArcGIS feature services: every feature of a layer in a box, and the clickable query that returns them."""

from __future__ import annotations

import logging
import time
from urllib.parse import urlencode

import requests

from fah.core.geometry import BoundingBox

log = logging.getLogger(__name__)


class ArcGisClient:
    """Queries on a shared session. ArcGIS reports errors with HTTP 200 and an 'error' body; they raise here."""

    def __init__(self, session: requests.Session) -> None:
        self.session = session

    @staticmethod
    def query_url(layer_url: str, box: BoundingBox, where: str = "1=1", fields: str = "*") -> str:
        """Clickable query returning the box's features as GeoJSON (what a channel downloaded)."""
        q = {
            "where": where,
            "geometry": ",".join(f"{b:.5f}" for b in box),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": fields,
            "outSR": 4326,
            "f": "geojson",
        }
        return f"{layer_url}/query?{urlencode(q)}"

    def get_json(self, url: str, params: dict | None = None, tries: int = 5) -> dict:
        """A service's JSON (f=json unless given), retried."""
        params = dict(params or {})
        params.setdefault("f", "json")
        for i in range(tries):
            try:
                r = self.session.get(url, params=params, timeout=120)
                r.raise_for_status()
                js = r.json()
                if isinstance(js, dict) and "error" in js:
                    raise RuntimeError(js["error"])
                return js
            except Exception as ex:  # noqa: BLE001  (retried, then raised)
                if i == tries - 1:
                    raise
                wait = 2 ** (i + 1)
                log.info(f"    retry {i + 1} after error: {ex} (waiting {wait}s)")
                time.sleep(wait)
        raise AssertionError("unreachable")

    def box_features(self, layer_url: str, box: BoundingBox, where: str = "1=1", chunk: int = 200) -> list[dict]:
        """Every feature of a layer intersecting the box, as GeoJSON, all fields, full geometry. Pages by OBJECTID
        (list the ids, then fetch them in chunks), not by resultOffset: offset paging on the 2018 NIFS archive
        repeats records and silently misses others (California, 2018 EventLine: 137,930 rows returned, 92,926
        unique). Raises if any listed id does not come back."""
        q = {
            "where": where,
            "geometry": ",".join(f"{b}" for b in box),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
        }
        ids = sorted(set(self._query(layer_url, dict(q, returnIdsOnly="true", f="json")).get("objectIds") or []))
        feats = []
        for i in range(0, len(ids), chunk):
            js = self._query(
                layer_url,
                {
                    "objectIds": ",".join(map(str, ids[i : i + chunk])),
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": 4326,
                    "f": "geojson",
                },
                post=True,
            )
            feats += js.get("features", [])
        if len(feats) != len(ids):
            raise RuntimeError(f"{layer_url}: listed {len(ids)} features, received {len(feats)}")
        return feats

    def _query(self, layer_url: str, params: dict, post: bool = False, tries: int = 6) -> dict:
        """One query, retried."""
        for attempt in range(tries):
            try:
                r = (
                    self.session.post(f"{layer_url}/query", data=params, timeout=300)
                    if post
                    else self.session.get(f"{layer_url}/query", params=params, timeout=180)
                )
                js = r.json()
                if "error" in js:
                    raise RuntimeError(js["error"])
                return js
            except (ValueError, RuntimeError, OSError) as ex:
                if attempt == tries - 1:
                    raise
                log.info(f"    retry {attempt + 1}: {str(ex)[:120]}")
                time.sleep(5 * 2**attempt)
        raise AssertionError("unreachable")
