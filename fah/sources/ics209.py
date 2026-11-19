"""ICS-209-PLUS (St. Denis et al.): the incident and situation-report tables, and matching incidents to a fire."""

from __future__ import annotations

import logging
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from fah.core.context import RunContext
from fah.core.geometry import BoundingBox
from fah.sources.http import HttpClient

log = logging.getLogger(__name__)


class Ics209Archive:
    """The ICS-209-PLUS wildfire zip from figshare, downloaded once into cache_dir/ics209 and read once."""

    ARTICLE = "https://api.figshare.com/v2/articles/19858927"
    DOWNLOAD = "https://ndownloader.figshare.com/files/38766504"
    # Candidate column names of the incident table (first match wins).
    INCIDENT_COLS = {
        "incident_id": ["INCIDENT_ID"],
        "name": ["INCIDENT_NAME", "INCIDENT_NAME_CORRECTED"],
        "lat": ["POO_LATITUDE", "LATITUDE", "POO_LAT"],
        "lon": ["POO_LONGITUDE", "LONGITUDE", "POO_LON", "POO_LONG"],
        "start": ["DISCOVERY_DATE", "START_DATE", "DISC_DATE"],
        "year": ["START_YEAR", "FIRE_YEAR", "YEAR"],
        "irwin": ["IRWIN_IDENTIFIER", "IRWIN_ID", "IRWINID", "IRWIN_IDS"],
    }
    NOT_TABLES = re.compile(r"(delet|complex|tract|county|cbg|block|link)")
    MATCH_PAD_DEG = 0.25

    def __init__(self, http: HttpClient, cache_dir: Path) -> None:
        self.http = http
        self.zip_path = Path(cache_dir) / "ics209" / "ics209-plus-wildfire.zip"
        self._tables: tuple | None = None

    @classmethod
    def for_run(cls, ctx: RunContext) -> Ics209Archive:
        """The run's archive: downloaded and read once, shared by every fire."""
        return ctx.memo("ics209", lambda: cls(HttpClient(ctx.session), ctx.cache))

    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, str]]:
        """(sitrep rows, incident rows) every column as text, the incident table for matching, and the two
        tables' names in the zip. Downloaded and read on first use."""
        if self._tables is None:
            self._download()
            z = zipfile.ZipFile(self.zip_path)
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            usable = [n for n in csvs if not self.NOT_TABLES.search(n.lower())]
            sit = next(n for n in usable if "sitrep" in n.lower())
            inc = self._incident_table(z, usable, sit, csvs)
            log.info(f"  ICS-209 tables: {sit} | {inc}")
            self._tables = (
                pd.read_csv(z.open(sit), dtype=str, keep_default_na=False),
                pd.read_csv(z.open(inc), dtype=str, keep_default_na=False),
                self._matching_table(pd.read_csv(z.open(inc), low_memory=False)),
                (sit, inc),
            )
        return self._tables

    def match(
        self,
        incidents: pd.DataFrame,
        irwins: set[str],
        names: set[str],
        box: BoundingBox,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[str]:
        """Incident ids of a fire: by IRWIN id, else by name + year + location."""
        pad = self.MATCH_PAD_DEG
        year_ok = incidents["year"].between(start.year - 1, end.year)
        in_box = incidents["lat"].between(box.south - pad, box.north + pad) & incidents["lon"].between(
            box.west - pad, box.east + pad
        )
        ids = set()
        if irwins:
            ids |= set(
                incidents.loc[
                    incidents["irwin"].apply(lambda v: any(i in v for i in irwins) if v else False), "incident_id"
                ]
            )
        if not ids:
            wanted = {self._norm_name(n) for n in names} - {""}
            cand = incidents[year_ok & incidents["name"].map(self._norm_name).isin(wanted)]
            if incidents["lat"].notna().any():
                cand = cand[in_box[cand.index] | cand["lat"].isna()]
            ids |= set(cand["incident_id"])
        return sorted(ids)

    def _download(self) -> None:
        if self.zip_path.exists():
            return
        self.zip_path.parent.mkdir(parents=True, exist_ok=True)
        art = self.http.get_json(self.ARTICLE)
        files = {f["name"]: f["download_url"] for f in art["files"]}
        log.info(f"  ICS-209-PLUS files: {list(files)}")
        name = next(n for n in files if "wildfire" in n.lower() and n.endswith(".zip"))
        log.info(f"  downloading {name} ...")
        self.http.download(files[name], self.zip_path, timeout=600, chunk=1 << 20)

    def _incident_table(self, z: zipfile.ZipFile, usable: list[str], sit: str, csvs: list[str]) -> str:
        """The incident summary: the table with INCIDENT_ID plus an incident name column."""
        name_cols = {c.upper() for c in self.INCIDENT_COLS["name"]}
        for n in sorted(usable, key=lambda n: ("incident" not in n.lower(), n)):
            if n == sit:
                continue
            cols = {c.upper() for c in pd.read_csv(z.open(n), nrows=0).columns}
            if "INCIDENT_ID" in cols and cols & name_cols:
                return n
        raise RuntimeError(f"No incident summary table with INCIDENT_ID and a name column among {csvs}")

    def _matching_table(self, inc: pd.DataFrame) -> pd.DataFrame:
        """incident_id, name, lat, lon, start, year, irwin: the columns matching uses, however the table names them."""
        ic = {k: self._pick(inc.columns, v, k in {"incident_id", "name"}) for k, v in self.INCIDENT_COLS.items()}
        out = pd.DataFrame({"incident_id": inc[ic["incident_id"]].astype(str), "name": inc[ic["name"]].astype(str)})
        for k in ("lat", "lon"):
            out[k] = pd.to_numeric(inc[ic[k]], errors="coerce") if ic[k] else np.nan
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Error parsing datetimes")  # dates come in mixed formats
            out["start"] = pd.to_datetime(inc[ic["start"]], errors="coerce", format="mixed") if ic["start"] else pd.NaT
        out["year"] = pd.to_numeric(inc[ic["year"]], errors="coerce") if ic["year"] else out["start"].dt.year
        out["irwin"] = inc[ic["irwin"]].astype(str).str.upper().str.strip("{} ") if ic["irwin"] else ""
        return out

    @staticmethod
    def _pick(cols, candidates: list[str], required: bool = True) -> str | None:
        up = {c.upper(): c for c in cols}
        for c in candidates:
            if c.upper() in up:
                return up[c.upper()]
        if required:
            raise KeyError(f"None of {candidates} in the incident table's columns: {sorted(cols)}")
        return None

    @staticmethod
    def _norm_name(s: object) -> str:
        s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
        s = re.sub(r"\b(fire|complex|wildfire|incident)\b", " ", s)
        return re.sub(r"\s+", " ", s).strip()
