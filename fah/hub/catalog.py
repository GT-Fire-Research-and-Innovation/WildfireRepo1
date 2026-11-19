"""The fire list of the built dataset (its catalog and grids), and the extents table this repo adds to it."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from fah.core.extent import FireExtentEstimator
from fah.core.fire import Fire
from fah.core.geometry import BoundingBox
from fah.hub.repo import HubRepo


class FireCatalog:
    """Fires of the built dataset (source repo): catalog/fires.csv joined with the QC ratings, each with the build's
    grid box. `which`: 'built' (fires with a built folder), 'keep' (QC rating keep) or 'all'."""

    CATALOG_FILES = ("catalog/fires.csv", "catalog/qc_fires.csv")
    EXTENTS_FILE = "catalog/fire_extents.csv"
    GRID_RES_M = 150.0  # the built dataset's cell size and margin: its grid box is each fire's build box
    GRID_PAD_KM = 9.6

    def __init__(self, source_repo: str, cache: Path) -> None:
        self.source = HubRepo.connect(source_repo)
        self.source_repo, self.cache = source_repo, Path(cache)

    @classmethod
    def build_grid(cls, row) -> dict:
        """The built dataset's grid for a catalog row: the perimeters' box plus GRID_PAD_KM on GRID_RES_M cells
        (north-up lon/lat). -> {"transform": [dx, 0, x0, 0, -dy, y0], "bounds_lonlat": [w, s, e, n]}"""
        pad = int(round(cls.GRID_PAD_KM * 1000 / cls.GRID_RES_M))
        lat0 = (row.south + row.north) / 2
        dy = cls.GRID_RES_M / 110570.0
        dx = cls.GRID_RES_M / (111320.0 * math.cos(math.radians(lat0)))
        w = int(math.ceil((row.east - row.west) / dx)) + 2 * pad
        h = int(math.ceil((row.north - row.south) / dy)) + 2 * pad
        x0, y0 = row.west - pad * dx, row.north + pad * dy
        return {"transform": [dx, 0.0, x0, 0.0, -dy, y0], "bounds_lonlat": [x0, y0 - h * dy, x0 + w * dx, y0]}

    def catalog_files(self) -> list[tuple[str, str]]:
        """[(repo path, local file)] of the catalog, downloaded once."""
        return [(f, self.source.download(f, self.cache)) for f in self.CATALOG_FILES]

    def fires(self, which: str = "built", ids: list[str] | None = None) -> list[Fire]:
        files = dict(self.catalog_files())
        cat = pd.read_csv(files["catalog/fires.csv"])
        qc = pd.read_csv(files["catalog/qc_fires.csv"], usecols=["fire_id", "rating"])
        cat = cat.merge(qc, on="fire_id", how="left")
        if which == "built":
            built = {p.split("/")[-1] for p in self.source.files("fires", recursive=False)}
            cat = cat[cat.fire_id.isin(built)]
        elif which == "keep":
            cat = cat[cat.rating == "keep"]
        if ids:
            cat = cat[cat.fire_id.isin(set(ids))]
        return [self.fire_from_row(row) for _, row in cat.iterrows()]

    @classmethod
    def fire_from_row(cls, row) -> Fire:
        g = cls.build_grid(row)
        sides = ("west", "south", "east", "north")
        own = (
            BoundingBox(*(float(row[k]) for k in sides)) if all(k in row and pd.notna(row[k]) for k in sides) else None
        )
        return Fire(
            fire_id=row.fire_id,
            incident=str(row.incident),
            name=str(row["name"]),
            year=int(row.year),
            first=str(row["first"]),
            last=str(row["last"]),
            build_box=BoundingBox(*g["bounds_lonlat"]),
            perimeter_box=own,
            grid_transform=tuple(g["transform"]),
            rating=str(row.rating),
        )

    def first_perimeters(self, fires: list[Fire]) -> dict[str, tuple]:
        """{fire_id: (time, lat, lon)} of each fire's first mapped perimeter, from the built dataset's labels.npz
        (arrival index per cell) on the fire's build grid. Fires without labels are left out."""
        self.source.snapshot([f"fires/{f.fire_id}/labels.npz" for f in fires], self.cache)
        out = {}
        for f in fires:
            p = self.cache / "fires" / f.fire_id / "labels.npz"
            if p.exists() and f.grid_transform:
                with np.load(p, allow_pickle=False) as z:
                    pt = FireExtentEstimator.first_perimeter_point(z["arrival"], f.grid_transform, z["times"])
                if pt is not None:
                    out[f.fire_id] = pt
        return out

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
