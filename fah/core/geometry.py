"""Boxes and pixel windows: a lon/lat box, a window of rows and columns on a grid, and the MODIS / VIIRS
sinusoidal tile grid. Pure arithmetic; arrays are numpy."""

from __future__ import annotations

import math
import re
from typing import NamedTuple

import numpy as np


class BoundingBox(NamedTuple):
    """A lon/lat box (west, south, east, north), in degrees."""

    west: float
    south: float
    east: float
    north: float

    def buffered(self, km: float) -> BoundingBox:
        """The box pushed out km on every side; longitudes are widened for the latitude farthest from the
        equator, so the margin is at least km everywhere in the box."""
        dlat = km / 110.574
        far = min(max(abs(self.south - dlat), abs(self.north + dlat)), 89.0)
        dlon = km / (111.320 * math.cos(math.radians(far)))
        return BoundingBox(self.west - dlon, self.south - dlat, self.east + dlon, self.north + dlat)

    def padded(self, deg: float) -> BoundingBox:
        """The box pushed out deg degrees on every side."""
        return BoundingBox(self.west - deg, self.south - deg, self.east + deg, self.north + deg)

    def union(self, other: BoundingBox) -> BoundingBox:
        """The smallest box holding both."""
        return BoundingBox(
            min(self.west, other.west),
            min(self.south, other.south),
            max(self.east, other.east),
            max(self.north, other.north),
        )

    def centre(self) -> tuple[float, float]:
        """(lat, lon) of the box centre."""
        return (self.south + self.north) / 2, (self.west + self.east) / 2

    def label(self, digits: int = 4) -> str:
        """'-121.8900,39.5112,-121.2398,39.9846'."""
        return ",".join(f"{v:.{digits}f}" for v in self)

    def contains(self, lat: np.ndarray, lon: np.ndarray, pad: float = 0.0) -> np.ndarray:
        """Mask of the points inside the box (+ pad degrees); NaN coordinates are outside."""
        with np.errstate(invalid="ignore"):
            return (
                (lat >= self.south - pad)
                & (lat <= self.north + pad)
                & (lon >= self.west - pad)
                & (lon <= self.east + pad)
            )


class PixelWindow(NamedTuple):
    """Rows r0..r1 and columns c0..c1 (end-exclusive) of a grid."""

    r0: int
    r1: int
    c0: int
    c1: int

    @classmethod
    def of_points(cls, lat: np.ndarray, lon: np.ndarray, box: BoundingBox, pad: float = 0.0) -> PixelWindow | None:
        """The rows / columns of a 2-D grid whose point centres fall in the box (+ pad), or None."""
        m = box.contains(lat, lon, pad)
        rows = np.flatnonzero(m.any(axis=1))
        if not len(rows):
            return None
        cols = np.flatnonzero(m.any(axis=0))
        return cls(int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)

    @classmethod
    def covering(cls, transform, width: int, height: int, bounds) -> PixelWindow | None:
        """The pixels of a raster (affine transform, size) covering bounds in the raster's CRS, snapped outward
        and clipped to the raster, or None."""
        inv = ~transform
        x0, y0, x1, y1 = bounds
        corners = [(x, y) for x in (x0, x1) for y in (y0, y1)]
        cs = [inv.a * x + inv.b * y + inv.c for x, y in corners]
        rs = [inv.d * x + inv.e * y + inv.f for x, y in corners]
        c0, c1 = max(math.floor(min(cs)), 0), min(math.ceil(max(cs)), width)
        r0, r1 = max(math.floor(min(rs)), 0), min(math.ceil(max(rs)), height)
        return cls(r0, r1, c0, c1) if r0 < r1 and c0 < c1 else None

    def grown(self, k: int, shape: tuple[int, int]) -> PixelWindow:
        """k more pixels on every side, within a grid of `shape`."""
        return PixelWindow(
            max(self.r0 - k, 0), min(self.r1 + k, shape[0]), max(self.c0 - k, 0), min(self.c1 + k, shape[1])
        )

    def union(self, other: PixelWindow) -> PixelWindow:
        return PixelWindow(
            min(self.r0, other.r0), max(self.r1, other.r1), min(self.c0, other.c0), max(self.c1, other.c1)
        )

    def cells(self) -> int:
        return (self.r1 - self.r0) * (self.c1 - self.c0)

    def slices(self) -> tuple[slice, slice]:
        return slice(self.r0, self.r1), slice(self.c0, self.c1)

    def slices_within(self, outer: PixelWindow) -> tuple[slice, slice]:
        """This window's rows / columns inside an array cut to `outer`."""
        return (slice(self.r0 - outer.r0, self.r1 - outer.r0), slice(self.c0 - outer.c0, self.c1 - outer.c0))

    def label(self) -> str:
        """'r0:r1,c0:c1', as file attributes record a window."""
        return f"{self.r0}:{self.r1},{self.c0}:{self.c1}"

    def m_to_i(self, i_shape: tuple[int, int], margin: int = 2) -> PixelWindow:
        """This window on a VIIRS 750 m M-band swath as the window on the 375 m I-band swath of the same granule.
        VIIRS scans 16 M-band and 32 I-band detector rows at a time with twice the columns, so M pixel (r, c)
        covers I pixels (2r..2r+1, 2c..2c+1); `margin` I pixels are added on every side."""
        return PixelWindow(
            max(2 * self.r0 - margin, 0),
            min(2 * self.r1 + margin, i_shape[0]),
            max(2 * self.c0 - margin, 0),
            min(2 * self.c1 + margin, i_shape[1]),
        )


class SinusoidalGrid:
    """The MODIS / VIIRS land grid: 36 x 18 tiles of N x N pixels (~463 m) on a sphere of radius 6371007.181 m.
    Windows are global pixel rows / columns, so tiles join pixel for pixel without resampling."""

    PROJ = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    TILE_M = 1111950.5197665233
    X0, Y0 = -20015109.355798, 10007554.677899  # upper-left corner of tile h00v00
    N = 2400

    @classmethod
    def window(cls, box: BoundingBox) -> PixelWindow:
        """A lon/lat box as global pixel rows / columns, snapped outward."""
        from pyproj import Transformer

        x0, y0, x1, y1 = Transformer.from_crs(4326, cls.PROJ, always_xy=True).transform_bounds(*box, densify_pts=41)
        px = cls.TILE_M / cls.N
        return PixelWindow(
            math.floor((cls.Y0 - y1) / px),
            math.ceil((cls.Y0 - y0) / px),
            math.floor((x0 - cls.X0) / px),
            math.ceil((x1 - cls.X0) / px),
        )

    @staticmethod
    def tile_hv(name: str) -> tuple[int, int]:
        """(h, v) of a tile from a granule name such as VNP13A1.A2018305.h08v05.002.<prod>.h5."""
        m = re.search(r"\.h(\d\d)v(\d\d)\.", name)
        if not m:
            raise ValueError(f"no tile in {name}")
        return int(m.group(1)), int(m.group(2))

    @classmethod
    def tile_part(cls, win: PixelWindow, h: int, v: int) -> tuple[PixelWindow, PixelWindow] | None:
        """Overlap of a global window with tile (h, v): (tile rows / cols, output rows / cols), or None."""
        n = cls.N
        r0, r1, c0, c1 = win
        tr0, tr1, tc0, tc1 = max(r0, v * n), min(r1, (v + 1) * n), max(c0, h * n), min(c1, (h + 1) * n)
        if tr0 >= tr1 or tc0 >= tc1:
            return None
        return (
            PixelWindow(tr0 - v * n, tr1 - v * n, tc0 - h * n, tc1 - h * n),
            PixelWindow(tr0 - r0, tr1 - r0, tc0 - c0, tc1 - c0),
        )

    @classmethod
    def coords(cls, win: PixelWindow) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre x (columns) and y (rows) of a global window, in metres."""
        px = cls.TILE_M / cls.N
        return cls.X0 + (np.arange(win.c0, win.c1) + 0.5) * px, cls.Y0 - (np.arange(win.r0, win.r1) + 0.5) * px
