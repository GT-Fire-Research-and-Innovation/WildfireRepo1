"""Source attributes and names as netCDF can store them."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np


class NetcdfAttrs:
    """Attributes copied from HDF4 / HDF5 / netCDF sources into the files written here."""

    SKIP = frozenset(
        {
            "DIMENSION_LIST",
            "REFERENCE_LIST",
            "CLASS",
            "NAME",
            "_Netcdf4Dimid",
            "_Netcdf4Coordinates",
            "_nc3_strict",
            "_FillValue",
        }
    )

    @classmethod
    def clean(cls, attrs: Mapping) -> dict:
        """Text decoded, HDF5 dimension bookkeeping dropped, one-element arrays as scalars, booleans as int8
        (_FillValue is passed to createVariable instead)."""
        out = {}
        for k, v in dict(attrs).items():
            if k in cls.SKIP:
                continue
            if isinstance(v, bytes):
                v = v.decode(errors="replace")
            elif isinstance(v, (list, tuple)):
                v = np.asarray(v)
            if isinstance(v, np.ndarray):
                if v.dtype.kind in "SOU":
                    v = "; ".join(x.decode(errors="replace") if isinstance(x, bytes) else str(x) for x in v.ravel())
                elif v.dtype.kind == "b":
                    v = v.astype("i1")
                elif v.dtype.kind not in "iuf":
                    continue
                if isinstance(v, np.ndarray) and v.size == 1:
                    v = v.reshape(-1)[0]
            elif isinstance(v, np.generic) and v.dtype.kind == "b":
                v = np.int8(v)
            elif not isinstance(v, (str, int, float, np.generic)):
                continue
            out[k] = v
        return out

    @staticmethod
    def fill_value(attrs: Mapping, dtype) -> object | None:
        """The source's _FillValue in the variable's dtype, or None."""
        f = dict(attrs).get("_FillValue")
        if f is None or np.dtype(dtype).kind not in "iuf":
            return None
        return np.asarray(f).astype(dtype).reshape(-1)[0]

    @staticmethod
    def variable_name(name: str) -> str:
        """A product's layer path as a netCDF variable name: '.../500 m 16 days NDVI' -> 'NDVI'."""
        name = re.sub(r"^\d+ m \d+ days ", "", str(name).split("/")[-1])
        return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
