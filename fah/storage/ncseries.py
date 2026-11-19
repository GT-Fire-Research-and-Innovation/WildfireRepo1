"""A (time, y, x) netCDF written one time step at a time, so a long fire never sits in memory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


class NcSeriesWriter:
    """Values are written exactly as given (auto mask / scale off): packed integers stay packed, with the
    scale_factor / add_offset / _FillValue attributes copied from the source. Each step also records `time`
    (ns, UTC) and `source` (the file it came from). Use as a context manager, or call close()."""

    def __init__(
        self,
        path: Path,
        lat: np.ndarray,
        lon: np.ndarray,
        attrs: Mapping[str, object],
        static: Mapping[str, tuple] | None = None,
    ) -> None:
        """static: {name: (dims, values, attrs)} written once (dims () for an attribute-only variable)."""
        import netCDF4

        self.ds = netCDF4.Dataset(path, "w")
        self.ds.setncatts({k: v for k, v in attrs.items() if v is not None})
        h, w = lat.shape
        self.ds.createDimension("time", None)
        self.ds.createDimension("y", h)
        self.ds.createDimension("x", w)
        for nm, a in (("lat", lat), ("lon", lon)):
            v = self.ds.createVariable(nm, "f4", ("y", "x"), zlib=True)
            v[:] = np.asarray(a, np.float32)
        for nm, (dims, a, at) in (static or {}).items():
            v = self.ds.createVariable(nm, np.asarray(a).dtype if dims else "i4", dims)
            if dims:
                v[:] = a
            v.setncatts(at)
        t = self.ds.createVariable("time", "i8", ("time",))
        t.units, t.long_name = "nanoseconds since 1970-01-01 00:00:00 UTC", "scan start / valid time"
        self.ds.createVariable("source", str, ("time",))
        self.n = 0

    def append(
        self, t_ns: int, source: str, fields: Mapping[str, tuple], extra: Mapping[str, tuple] | None = None
    ) -> None:
        """fields: {name: (2-D array, dtype, attrs)}; extra: {name: (scalar, dtype, attrs)}, one value per step."""
        for nm, (arr, dtype, attrs) in fields.items():
            if nm not in self.ds.variables:
                fill = attrs.get("_FillValue")
                v = self.ds.createVariable(
                    nm, dtype, ("time", "y", "x"), zlib=True, complevel=4, fill_value=fill if fill is not None else None
                )
                v.set_auto_maskandscale(False)
                v.setncatts({k: a for k, a in attrs.items() if k != "_FillValue"})
            self.ds.variables[nm][self.n] = arr
        for nm, (val, dtype, attrs) in (extra or {}).items():
            if nm not in self.ds.variables:
                self.ds.createVariable(nm, dtype, ("time",)).setncatts(attrs)
            self.ds.variables[nm][self.n] = val
        self.ds.variables["time"][self.n] = t_ns
        self.ds.variables["source"][self.n] = source
        self.n += 1

    def close(self) -> None:
        self.ds.close()

    def __enter__(self) -> NcSeriesWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
