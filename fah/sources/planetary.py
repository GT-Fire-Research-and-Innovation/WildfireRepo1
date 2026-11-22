"""Microsoft Planetary Computer: STAC search and the anonymous read token of a collection."""

from __future__ import annotations

import time
from datetime import datetime

import requests

from fah.core.geometry import BoundingBox


class PlanetaryComputer:
    """Items of a collection over a box and year, and the token that signs reads of their assets."""

    STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
    SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
    TOKEN_MARGIN_S = 300  # renew this long before the token's expiry, so no read outlives it
    TOKEN_LIFE_S = 900  # assumed lifetime when the reply gives no expiry (anonymous tokens last 15 min)

    def __init__(self, session: requests.Session) -> None:
        self.session = session
        self._tokens: dict[str, tuple[str, float]] = {}  # collection -> (token, expiry as time.time())

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
        """The read token, renewed TOKEN_MARGIN_S before the expiry the service states (msft:expiry). Anonymous
        tokens last 15 minutes; past that every read is refused (HTTP 403), which GDAL reports as a failed strip
        read partway through a file."""
        tok, expiry = self._tokens.get(collection, (None, 0.0))
        if tok is None or time.time() > expiry - self.TOKEN_MARGIN_S:
            r = self.session.get(self.SAS.format(collection=collection), timeout=60)
            r.raise_for_status()
            reply = r.json()
            tok = reply["token"]
            stated = reply.get("msft:expiry")
            if stated:
                expiry = datetime.fromisoformat(str(stated).replace("Z", "+00:00")).timestamp()
            else:
                expiry = time.time() + self.TOKEN_LIFE_S
            self._tokens[collection] = (tok, expiry)
        return tok
