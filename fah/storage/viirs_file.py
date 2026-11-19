"""viirs.nc: one fire's VIIRS swath granules packed in one netCDF, a group per granule."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from fah.core.geometry import BoundingBox
from fah.storage.attrs import NetcdfAttrs


class ViirsFile:
    """Written as viirs.nc.part while granules arrive and renamed by finish(), so a crash never leaves a
    half-written viirs.nc. Each granule is a group (<SAT>_<YYYY-MM-DD>T<HHMM>_<day|night>) with the I grid
    (iy, ix) and M grid (my, mx) as cut from the swath; look-up tables are stored once in /luts and a group's
    attribute named after a table gives its path. A crop is {"i_win", "m_win", "glob", "i": {name: (array,
    attrs)}, "m": {...}} as the channel reads it."""

    NAME = "viirs.nc"

    def __init__(self, stage_dir: Path, fire_id: str) -> None:
        self.part = Path(stage_dir) / (self.NAME + ".part")
        self.fire_id = fire_id
        self.groups: list[str] = []

    @staticmethod
    def group_name(key: tuple[str, str, str], entry: Mapping) -> str:
        """<SAT>_<YYYY-MM-DD>T<HHMM>_<day|night> of a granule key (sat, 'A2018314', '2048')."""
        sat, a, hhmm = key
        day = (pd.Timestamp(f"{a[1:5]}-01-01") + pd.Timedelta(days=int(a[5:8]) - 1)).strftime("%Y-%m-%d")
        return f"{sat}_{day}T{hhmm}_{(entry.get('daynight') or 'unknown').lower()}"

    def add_granule(
        self,
        key: tuple[str, str, str],
        entry: Mapping,
        crop: dict,
        bounds: BoundingBox | None = None,
        sources: str = "",
    ) -> str:
        """Add one granule's crop as its own group. Values as stored in the source; with bounds, pixels outside
        them are blanked to the source's _FillValue. -> the group's name."""
        import netCDF4

        name = self.group_name(key, entry)
        blanked = self._blank_outside(crop, bounds) if bounds else {}
        with netCDF4.Dataset(self.part, "a" if self.part.exists() else "w") as ds:
            if name in ds.groups:
                return name
            luts = ds.groups.get("luts") or ds.createGroup("luts")
            g = ds.createGroup(name)
            at = {
                "title": f"VIIRS {key[0]} L1B granule {key[1]}.{key[2]} cropped to {self.fire_id}",
                "fire_id": self.fire_id,
                "satellite": key[0],
                "day_night": entry.get("daynight") or "",
                "granule_start": entry.get("start") or "",
                "i_window_rows_cols": crop["i_win"].label(),
                "sources": sources,
            }
            if crop.get("m_win"):
                at["m_window_rows_cols"] = crop["m_win"].label()
            if bounds:
                at["blanked_outside_lonlat"] = ",".join(f"{b:.5f}" for b in bounds)
                at.update({f"blanked_pixels_{k}": v for k, v in blanked.items()})
            g.setncatts({**{f"source_{k}": v for k, v in crop["glob"].items()}, **at})
            for grid, dims in (("i", ("iy", "ix")), ("m", ("my", "mx"))):
                if not crop.get(grid):
                    continue
                made = False
                for vname, (arr, attrs) in crop[grid].items():
                    if arr.ndim == 1:
                        g.setncattr(vname, self._lut(luts, vname, arr, attrs))
                        continue
                    if not made:
                        g.createDimension(dims[0], arr.shape[0])
                        g.createDimension(dims[1], arr.shape[1])
                        made = True
                    v = g.createVariable(
                        vname,
                        arr.dtype,
                        dims,
                        zlib=True,
                        complevel=4,
                        fill_value=NetcdfAttrs.fill_value(attrs, arr.dtype),
                    )
                    v.set_auto_maskandscale(False)
                    v.setncatts(NetcdfAttrs.clean(attrs))
                    v[:] = arr
        self.groups.append(name)
        return name

    def finish(self, region: BoundingBox, pad_km: float) -> Path:
        """Index the granule groups in time order (root variables granule, granule_start, satellite, day_night)
        and rename the file to viirs.nc."""
        import netCDF4

        with netCDF4.Dataset(self.part, "a") as ds:
            gs = ds.groups
            names = sorted((k for k in gs if k != "luts"), key=lambda k: (str(getattr(gs[k], "granule_start", "")), k))
            ds.createDimension("granule", len(names))
            for var, attr in (
                ("granule", None),
                ("granule_start", "granule_start"),
                ("satellite", "satellite"),
                ("day_night", "day_night"),
            ):
                v = ds.createVariable(var, str, ("granule",))
                v[:] = np.array([k if attr is None else str(getattr(gs[k], attr, "")) for k in names], dtype=object)
            ds.setncatts(
                {
                    "title": f"VIIRS L1B swath granules cropped to {self.fire_id}",
                    "fire_id": self.fire_id,
                    "n_granules": len(names),
                    "region_lonlat": ",".join(f"{b:.5f}" for b in region),
                    "blanked_pad_km": pad_km,
                    "layout": "one group per granule, named as in the granule variable (I grid iy/ix, M grid "
                    "my/mx); /luts holds each look-up table once, and a group's attribute named after a "
                    "table (e.g. I04_brightness_temperature_lut) gives that table's path",
                    "blanking": "pixels farther than blanked_pad_km outside region_lonlat hold the source's "
                    "_FillValue in every variable that has one",
                }
            )
        final = self.part.with_name(self.NAME)
        self.part.rename(final)
        return final

    @staticmethod
    def _blank_outside(crop: dict, bounds: BoundingBox) -> dict[str, int]:
        """Blank every pixel whose latitude / longitude lies outside bounds, in each 2-D variable with a fill value.
        The swath window is a rectangle in scan coordinates, so it reaches past the region at the swath edges and
        on tilted passes; blanked pixels compress to almost nothing. Variables without a fill value (quality flags,
        uncertainty indices) are kept whole. -> {grid: pixels blanked}."""
        out = {}
        for grid, la, lo in (("i", "latitude", "longitude"), ("m", "m_latitude", "m_longitude")):
            vs = crop.get(grid) or {}
            if la not in vs or lo not in vs:
                continue
            lat, lon = (np.asarray(vs[k][0], dtype="f8") for k in (la, lo))
            outside = ~bounds.contains(lat, lon)
            out[grid] = int(outside.sum())
            if not out[grid]:
                continue
            for name, (arr, attrs) in list(vs.items()):
                fill = NetcdfAttrs.fill_value(attrs, arr.dtype)
                if arr.ndim == 2 and arr.shape == outside.shape and fill is not None:
                    vs[name] = (np.where(outside, fill, arr).astype(arr.dtype), attrs)
        return out

    @staticmethod
    def _lut(luts, name: str, arr: np.ndarray, attrs: Mapping) -> str:
        """A look-up table stored once per file in /luts, named by its content: -> its path. The tables only change
        with the calibration, so a fire's hundreds of granules share two or three of them."""
        key = f"{name}_{hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]}"
        if key not in luts.variables:
            luts.createDimension(f"{key}_n", arr.shape[0])
            v = luts.createVariable(
                key,
                arr.dtype,
                (f"{key}_n",),
                zlib=True,
                complevel=4,
                fill_value=NetcdfAttrs.fill_value(attrs, arr.dtype),
            )
            v.set_auto_maskandscale(False)
            v.setncatts({**NetcdfAttrs.clean(attrs), "source_name": name})
            v[:] = arr
        return f"/luts/{key}"
