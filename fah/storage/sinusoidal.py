"""Layers of one window of the MODIS / VIIRS sinusoidal grid, filled tile by tile and written as one netCDF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from fah.core.geometry import PixelWindow, SinusoidalGrid
from fah.storage.attrs import NetcdfAttrs


class SinusoidalMosaic:
    """The tiles share one grid, so each is copied in pixel for pixel (nothing is resampled); with nt, one slot per
    time step. Pixels no tile covers keep the layer's _FillValue (netCDF's default fill when the source has none)."""

    def __init__(self, win: PixelWindow, nt: int | None = None) -> None:
        self.win, self.nt = win, nt
        self.layers: dict[str, tuple[np.ndarray, object, dict]] = {}

    def put(
        self, name: str, src, attrs: Mapping, h: int, v: int, t: int | None = None, source_name: str | None = None
    ) -> bool:
        """src: an N x N dataset of tile (h, v) (h5py, pyhdf or numpy); only the window's part is read."""
        import netCDF4

        part = SinusoidalGrid.tile_part(self.win, h, v)
        if part is None:
            return False
        (tr0, tr1, tc0, tc1), (or0, or1, oc0, oc1) = part
        block = np.asarray(src[tr0:tr1, tc0:tc1])
        if name not in self.layers:
            fill = NetcdfAttrs.fill_value(attrs, block.dtype)
            if fill is None:
                fill = np.asarray(netCDF4.default_fillvals[block.dtype.str[1:]]).astype(block.dtype)
            r0, r1, c0, c1 = self.win
            shape = (r1 - r0, c1 - c0) if self.nt is None else (self.nt, r1 - r0, c1 - c0)
            at = NetcdfAttrs.clean(attrs)
            if source_name and source_name != name:
                at["source_name"] = str(source_name)
            self.layers[name] = (np.full(shape, fill, block.dtype), fill, at)
        arr = self.layers[name][0]
        (arr if t is None else arr[t])[or0:or1, oc0:oc1] = block
        return True

    def write(self, path: Path, attrs: Mapping[str, object], times_ns: Sequence[int] | None = None) -> None:
        """The layers on the sinusoidal grid: x / y pixel centres (m), a CF `crs` variable, values exactly as
        stored in the source with its attributes (scale_factor etc. are not applied)."""
        import netCDF4

        x, y = SinusoidalGrid.coords(self.win)
        with netCDF4.Dataset(path, "w") as ds:
            ds.setncatts(
                {
                    **attrs,
                    "sinusoidal_rows_cols": self.win.label(),
                    "grid": "MODIS/VIIRS sinusoidal land grid, 2400 x 2400 pixels per tile (pixel_size_m 463.3127)",
                }
            )
            ds.createDimension("y", len(y))
            ds.createDimension("x", len(x))
            for nm, vals, ln in (("x", x, "projection x (pixel centre)"), ("y", y, "projection y (pixel centre)")):
                v = ds.createVariable(nm, "f8", (nm,))
                v[:] = vals
                v.setncatts({"units": "m", "long_name": ln, "standard_name": f"projection_{nm}_coordinate"})
            crs = ds.createVariable("crs", "i4")
            crs.setncatts(
                {
                    "grid_mapping_name": "sinusoidal",
                    "longitude_of_central_meridian": 0.0,
                    "false_easting": 0.0,
                    "false_northing": 0.0,
                    "earth_radius": 6371007.181,
                    "proj4": SinusoidalGrid.PROJ,
                }
            )
            if times_ns is not None:
                ds.createDimension("time", len(times_ns))
                t = ds.createVariable("time", "i8", ("time",))
                t[:] = times_ns
                t.setncatts({"units": "nanoseconds since 1970-01-01 00:00:00 UTC", "long_name": "composite start"})
            for nm, (arr, fill, at) in self.layers.items():
                v = ds.createVariable(
                    nm,
                    arr.dtype,
                    ("time", "y", "x") if arr.ndim == 3 else ("y", "x"),
                    zlib=True,
                    complevel=4,
                    fill_value=fill,
                )
                v.set_auto_maskandscale(False)
                v.setncatts({**at, "grid_mapping": "crs"})
                v[:] = arr
