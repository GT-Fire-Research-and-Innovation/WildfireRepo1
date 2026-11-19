"""GRIDMET daily weather and fire danger (~4 km), from the THREDDS NetCDF Subset Service."""

from __future__ import annotations

import pandas as pd

from fah.core.geometry import BoundingBox
from fah.sources.http import HttpClient


class GridmetService:
    """Plain HTTPS subsets returning a netCDF file (no OPeNDAP client needed; the OPeNDAP route failed on Kaggle)."""

    AGGREGATION = "https://thredds.northwestknowledge.net/thredds/dodsC/agg_met_{v}_1979_CurrentYear_CONUS.nc"
    # short name -> variable inside the aggregation: the built dataset's 9, then the 4 TS-SatFire's FirePred takes
    # from GRIDMET. PDSI is its own aggregation (gridMET drought), one value per 5-day period.
    VARIABLES = {
        "vpd": "daily_mean_vapor_pressure_deficit",
        "fm100": "dead_fuel_moisture_100hr",
        "fm1000": "dead_fuel_moisture_1000hr",
        "bi": "daily_mean_burning_index_g",
        "erc": "daily_mean_energy_release_component-g",
        "rmin": "daily_minimum_relative_humidity",
        "vs": "daily_mean_wind_speed",
        "th": "daily_mean_wind_direction",
        "tmmx": "daily_maximum_temperature",
        "pr": "precipitation_amount",
        "tmmn": "daily_minimum_temperature",
        "sph": "daily_mean_specific_humidity",
        "pdsi": "daily_mean_palmer_drought_severity_index",
    }
    LEAD_DAYS = {"pdsi": 10}  # start earlier so the window holds the PDSI value in force before the fire
    PAD_DEG = 0.05

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    @classmethod
    def subset_url(
        cls, short: str, var: str, box: BoundingBox, d0: pd.Timestamp, d1: pd.Timestamp, pad: float | None = None
    ) -> str:
        """One variable over the box (+ pad) and days d0..d1."""
        pad = cls.PAD_DEG if pad is None else pad
        w, s, e, n = box
        return (
            cls.AGGREGATION.format(v=short).replace("/dodsC/", "/ncss/grid/")
            + f"?var={var}&north={n + pad:.4f}&west={w - pad:.4f}&east={e + pad:.4f}&south={s - pad:.4f}"
            f"&horizStride=1&time_start={d0.date()}T00:00:00Z&time_end={d1.date()}T00:00:00Z&accept=netcdf"
        )

    def subset(self, short: str, box: BoundingBox, d0: pd.Timestamp, d1: pd.Timestamp) -> tuple[str, bytes]:
        """(url, netCDF bytes) of one variable; the variable's lead days are added before d0."""
        d0 = d0 - pd.Timedelta(days=self.LEAD_DAYS.get(short, 0))
        url = self.subset_url(short, self.VARIABLES[short], box, d0, d1)
        data = self.http.get_bytes(url)
        if not (data[:3] == b"CDF" or data[:4] == b"\x89HDF"):
            raise RuntimeError(f"GRIDMET {short}: reply is not netCDF: {data[:120]!r}")
        return url, data
