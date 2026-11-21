"""The fire list: the repo's own catalog/fires.csv (written by `--catalog build`), and the extents table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fah.core.fire import Fire
from fah.core.geometry import BoundingBox
from fah.hub.repo import HubRepo

log = logging.getLogger(__name__)


class FireCatalog:
    """Fires of catalog/fires.csv, from the target repo or a local file. `archive` picks geomac, nifs, both or all."""

    FIRES_FILE = "catalog/fires.csv"
    EXTENTS_FILE = "catalog/fire_extents.csv"
    ARCHIVES = ("geomac", "nifs", "both")

    def __init__(self, hub: HubRepo, cache: Path, local_file: str | Path | None = None) -> None:
        self.hub, self.cache, self.local_file = hub, Path(cache), local_file

    def table(self) -> pd.DataFrame:
        path = self.local_file
        if path is None:
            if not self.hub.repo_id:
                raise FileNotFoundError(f"no fire list: give --hf-repo (its {self.FIRES_FILE}) or --catalog-file")
            path = self.hub.download(self.FIRES_FILE, self.cache)
        return pd.read_csv(path, dtype={"fire_id": str, "incident": str})

    def fires(self, archive: str = "all", ids: list[str] | None = None) -> list[Fire]:
        cat = self.table()
        if archive != "all":
            cat = cat[cat["archive"] == archive]
        if ids:
            cat = cat[cat["fire_id"].isin(set(ids))]
        boxed = cat[["west", "south", "east", "north"]].notna().all(axis=1)
        if (~boxed).any():
            log.warning(f"warning: {int((~boxed).sum())} fires have no box in {self.FIRES_FILE} and are left out")
        return [self.fire_from_row(row) for _, row in cat[boxed].iterrows()]

    @staticmethod
    def fire_from_row(row: pd.Series) -> Fire:
        def point(t: str, lat: str, lon: str) -> tuple | None:
            vals = (row.get(t), row.get(lat), row.get(lon))
            return vals if all(pd.notna(v) for v in vals) else None

        origin_lat, origin_lon = (
            ("initial_lat", "initial_lon") if pd.notna(row.get("initial_lat")) else ("incident_lat", "incident_lon")
        )
        return Fire(
            fire_id=str(row["fire_id"]),
            incident=str(row["incident"]),
            name=str(row["name"]),
            year=int(row["year"]),
            first=str(row["first"]),
            last=str(row["last"]),
            box=BoundingBox(*(float(row[k]) for k in ("west", "south", "east", "north"))),
            archive=str(row.get("archive", "")),
            first_record=point("first_record_time", "first_lat", "first_lon"),
            origin=point("discovery_utc", origin_lat, origin_lon),
        )

    @staticmethod
    def read_extents(path: Path | str) -> dict[str, dict]:
        """{fire_id: row} of a local fire_extents.csv."""
        t = pd.read_csv(path, dtype=str, keep_default_na=False)
        return {r["fire_id"]: r for r in t.to_dict("records")}

    @classmethod
    def load_extents(
        cls, hub: HubRepo | None, cache: Path | None = None, path: Path | str | None = None
    ) -> dict | None:
        """{fire_id: row} of catalog/fire_extents.csv from a local file or the target repo; None when absent."""
        if path is None:
            if hub is None or not hub.repo_id:
                return None
            try:
                path = hub.download(cls.EXTENTS_FILE, cache)
            except Exception as ex:  # noqa: BLE001  (not written yet)
                if (
                    "404" in str(ex)
                    or "EntryNotFound" in type(ex).__name__
                    or "RepositoryNotFound" in type(ex).__name__
                ):
                    return None
                raise
        return cls.read_extents(path)

    @staticmethod
    def with_extents(fires: list[Fire], table: dict | None) -> list[Fire]:
        """The fires with their extents rows attached (unchanged without a table)."""
        return [f.with_extent(table.get(f.fire_id)) if table else f for f in fires]
