"""Source attributes and names as netCDF can store them."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np


class NetcdfAttrs:
    """Attributes copied from HDF4 / HDF5 / netCDF sources into the files written here."""

    # HDF5 dimension-scale bookkeeping whose values are h5py object references, which cannot be pickled.
    REFERENCES = frozenset({"DIMENSION_LIST", "REFERENCE_LIST"})
    SKIP = REFERENCES | frozenset(
        {
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
        (_FillValue is passed to createVariable instead), names netCDF cannot hold made legal (attr_name)."""
        out = {}
        for k, v in dict(attrs).items():
            if k in cls.SKIP:
                continue
            k = cls.attr_name(k)
            if k in out:  # two source names that differ only in characters netCDF refuses: the first stays
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

    @classmethod
    def portable(cls, attrs: Mapping) -> dict:
        """The attributes as values a worker process can send back (pickle): everything but the h5py object
        references, which are not even read. Unlike clean(), _FillValue and the source names stay."""
        return {k: attrs[k] for k in attrs.keys() if k not in cls.REFERENCES}

    @staticmethod
    def attr_name(name: object) -> str:
        """A source attribute name as netCDF can store it, otherwise unchanged: '/' and control characters become
        '_', trailing spaces go, and a name that does not start with a letter, digit or '_' gets a leading '_'
        (MCD12Q1's 'Cropland/Natural Vegetation Mosaics' -> 'Cropland_Natural Vegetation Mosaics')."""
        s = re.sub(r"[/\x00-\x1f\x7f]", "_", str(name)).rstrip()
        if not s or (s[0].isascii() and not (s[0].isalnum() or s[0] == "_")):
            s = "_" + s
        return s

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
