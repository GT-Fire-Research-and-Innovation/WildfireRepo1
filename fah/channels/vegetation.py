"""vegetation: VIIRS VNP13A1 v002 16-day vegetation indices on the sinusoidal 463 m grid."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fah.channels.base import PerFireChannel
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.core.geometry import SinusoidalGrid
from fah.sources.earthdata import EarthdataClient
from fah.storage.attrs import NetcdfAttrs
from fah.storage.sinusoidal import SinusoidalMosaic


class VegetationChannel(PerFireChannel):
    """vnp13a1.nc (time = composite start): TS-SatFire's NDVI and EVI2 bands and every other layer of the product
    (EVI, the reflectances, angles, composite day, VI quality, pixel reliability), for each composite (a new one
    starts every 8 days) overlapping the 16 days before the window through its end. Only the region's part of each
    tile is read (HTTP byte ranges)."""

    name = "vegetation"
    time_dependent = True
    PRODUCT = "VNP13A1"

    def check_ready(self) -> None:
        EarthdataClient.token()
        self.require_module("h5py", "vegetation reads HDF5 granules: pip install h5py")

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        import h5py

        earthdata = EarthdataClient(ctx.session)
        t0, t1 = fire.span()
        box = fire.region()
        win = SinusoidalGrid.window(box)
        grans = [
            g
            for g in earthdata.granules(self.PRODUCT, t0 - pd.Timedelta(days=16), t1, box)
            if SinusoidalGrid.tile_part(win, *SinusoidalGrid.tile_hv(g["name"]))
        ]
        dates = sorted({EarthdataClient.granule_parts(g["url"])[1] for g in grans})
        if not dates:
            return "empty", []
        token = EarthdataClient.token()
        mosaic, sources = SinusoidalMosaic(win, nt=len(dates)), []
        for g in grans:
            k = dates.index(EarthdataClient.granule_parts(g["url"])[1])
            h, v = SinusoidalGrid.tile_hv(g["name"])
            with earthdata.open_remote(g["url"], token) as fo, h5py.File(fo, "r") as f:
                for path, ds in self._grids(f, SinusoidalGrid.N):
                    mosaic.put(
                        NetcdfAttrs.variable_name(path), ds, dict(ds.attrs), h, v, t=k, source_name=path.split("/")[-1]
                    )
            sources.append(g["url"])
        times = [
            EarthdataClient.granule_day(
                next(g["url"] for g in grans if EarthdataClient.granule_parts(g["url"])[1] == d)
            ).value
            for d in dates
        ]
        mosaic.write(
            out / "vnp13a1.nc",
            {
                "title": f"VIIRS {self.PRODUCT} v002 16-day vegetation indices cropped to {fire.fire_id}",
                "fire_id": fire.fire_id,
                "sources": " | ".join(sources),
            },
            times_ns=times,
        )
        return "ok", sources

    @staticmethod
    def _grids(f, n: int) -> list[tuple[str, object]]:
        """(path, dataset) of every n x n dataset in an HDF5 file."""
        import h5py

        found = []
        f.visititems(lambda p, o: found.append((p, o)) if isinstance(o, h5py.Dataset) and o.shape == (n, n) else None)
        return found
