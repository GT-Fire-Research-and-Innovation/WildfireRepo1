"""Microsoft Planetary Computer: STAC search and the anonymous read token of a collection."""

from __future__ import annotations

import time

import requests

from fah.core.geometry import BoundingBox


class PlanetaryComputer:
    """Items of a collection over a box and year, and the token that signs reads of their assets."""

    STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
    SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
    TOKEN_RENEW_S = 1800  # tokens last about an hour

    def __init__(self, session: requests.Session) -> None:
        self.session = session
        self._tokens: dict[str, tuple[str, float]] = {}

    def search(self, collection: str, box: BoundingBox, year: int) -> list[dict]:
        """The collection's items over the box during `year`, sorted by id."""
        r = self.session.post(
            f"{self.STAC}/search",
            timeout=120,
            json={
                "collections": [collection],
                "bbox": list(box),
                "limit": 100,
                "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
            },
        )
        r.raise_for_status()
        return sorted(r.json().get("features", []), key=lambda i: i["id"])

    def token(self, collection: str) -> str:
        """The read token, renewed after TOKEN_RENEW_S."""
        tok, t = self._tokens.get(collection, (None, 0.0))
        if tok is None or time.time() - t > self.TOKEN_RENEW_S:
            r = self.session.get(self.SAS.format(collection=collection), timeout=60)
            r.raise_for_status()
            tok = r.json()["token"]
            self._tokens[collection] = (tok, time.time())
        return tok
