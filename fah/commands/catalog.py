"""The fire catalog, crawled from the NIFC databases: raw tables first (scan), then the gated fire list (build)."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fah.commands.settings import RunSettings
from fah.core.context import RunContext
from fah.core.errors import ExitCode
from fah.core.incidents import ArchiveRules, CatalogGates, IncidentKey
from fah.hub.repo import HubRepo
from fah.sources.nifc import NifcArchive
from fah.sources.wfigs import WfigsIncidents

log = logging.getLogger(__name__)


class CatalogBuilder:
    """scan: every record of every perimeter layer of the years (all fields, as served) to
    catalog/raw/perimeters/<year>_<archive>_<layer>.parquet, and every WFIGS incident those records name to
    catalog/raw/incidents/wfigs_<year>.parquet. A table already in the repo is not fetched again.
    build: the gates (fah.core.incidents) over those tables -> catalog/fires.csv (the fires), catalog/incidents.csv
    (every incident, its gate values and why it is not a fire) and catalog/gates.json (settings and counts)."""

    RAW = "catalog/raw"
    READ_COLUMNS = {"incidentname", "irwinid", "uniquefireidentifier", "localincidentidentifier", "gisacres",
                    "mapmethod", "featurecategory", "deletethis", *ArchiveRules.TIME_FIELDS}  # fmt: skip
    # fires.csv name <- WFIGS field (all fields stay in the raw incident table)
    INCIDENT_COLUMNS = {
        "discovery_utc": "FireDiscoveryDateTime", "incident_type": "IncidentTypeCategory",
        "incident_kind": "IncidentTypeKind",
        "fire_cause": "FireCause", "fire_cause_general": "FireCauseGeneral", "fire_cause_specific": "FireCauseSpecific",
        "final_acres": "FinalAcres", "incident_size": "IncidentSize", "initial_lat": "InitialLatitude",
        "initial_lon": "InitialLongitude", "incident_lat": "_lat", "incident_lon": "_lon",
        "containment_utc": "ContainmentDateTime", "control_utc": "ControlDateTime", "out_utc": "FireOutDateTime",
        "predominant_fuel_group": "PredominantFuelGroup", "predominant_fuel_model": "PredominantFuelModel",
        "primary_fuel_model": "PrimaryFuelModel", "complexity_level": "IncidentComplexityLevel",
        "complex_name": "CpxName",
        "complex_id": "CpxID", "is_complex_child": "IsCpxChild", "poo_state": "POOState", "poo_county": "POOCounty",
        "gacc": "GACC", "unique_fire_id": "UniqueFireIdentifier", "local_incident_id": "LocalIncidentIdentifier",
    }  # fmt: skip
    TIME_COLUMNS = ("discovery_utc", "containment_utc", "control_utc", "out_utc")

    def __init__(self, settings: RunSettings) -> None:
        self.s = settings
        self.hub = HubRepo.connect(settings.hf_repo)
        self.root = Path(settings.work) / "catalog_repo"  # local copy laid out like the repo
        self.gates = CatalogGates(first_year=settings.first_year, last_year=settings.last_year,
                                  min_acres=settings.min_acres, min_days=settings.min_days)  # fmt: skip

    @property
    def uploads(self) -> bool:
        return bool(self.s.hf_repo) and not self.s.dry_run

    def execute(self) -> int:
        if self.uploads:
            self.hub.check_write()
            self.hub.ensure(self.s.private)
        if self.s.catalog in ("scan", "all"):
            self.scan()
        if self.s.catalog in ("build", "all"):
            self.build()
        return ExitCode.OK

    # ------------------------------------------------------------------------------------- scan

    def scan(self) -> None:
        ctx = RunContext("catalog", Path(self.s.work), threads=self.s.threads)
        nifc = NifcArchive.for_run(ctx)
        have = set(self.hub.files(self.RAW)) if self.uploads else set()
        years = range(self.s.first_year, self.s.last_year + 1)
        for year in years:
            for kind, url in nifc.perimeter_layers(year):
                slug = re.sub(r"[^A-Za-z0-9]+", "_", nifc.layer_label(url)).strip("_")
                path = f"{self.RAW}/perimeters/{year}_{kind}_{slug}.parquet"
                if path in have or (self.root / path).exists():
                    log.info(f"  {path}: already crawled")
                    continue
                df = nifc.layer_records(url).assign(_archive="geomac" if kind == "geomac" else "nifs", _year=year,
                                                    _layer=nifc.layer_label(url), _url=url)  # fmt: skip
                self._save(df, path, f"Catalog crawl: {year} {nifc.layer_label(url)} ({len(df)} records)")
        wfigs = WfigsIncidents(nifc.arcgis)
        for year in years:
            path = f"{self.RAW}/incidents/wfigs_{year}.parquet"
            if path in have or (self.root / path).exists():
                continue
            ids = {IncidentKey.irwin_uuid(k) for k in self._year_keys(year)} - {None}
            df = wfigs.by_irwin(sorted(ids))
            self._save(df, path, f"Catalog crawl: WFIGS incidents of {year} ({len(df)} of {len(ids)} IRWIN ids)")
        ctx.cleanup()

    def _year_keys(self, year: int) -> set[str]:
        tables = [self._table(p) for p in self._raw_paths("perimeters") if p.rsplit("/", 1)[-1].startswith(f"{year}_")]
        keys = set()
        for t in tables:
            low = t.rename(columns=str.lower)
            keys |= set(ArchiveRules().keys(low.reindex(columns=sorted(set(low.columns) | {"incidentname"}))))
        return keys

    def _save(self, df: pd.DataFrame, path: str, message: str) -> None:
        """A raw table as Parquet, values as served (mixed-type columns as text)."""
        dst = self.root / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("string")
        df.to_parquet(dst, index=False)
        log.info(f"  {path}: {len(df)} rows")
        if self.uploads:
            self.hub.upload_file(dst, path, message)

    # ------------------------------------------------------------------------------------- build

    def build(self) -> None:
        raw = self._evaluation_frame()
        incidents = self._incident_table()
        types = dict(zip(incidents["_uuid"], incidents.get("IncidentTypeCategory", pd.Series(dtype=str)), strict=True))
        records = self.gates.records(raw)
        report = self.gates.evaluate(records, types)
        fires = self._fires(report[report["reason"] == ""], records, incidents)
        out = self.root / "catalog"
        out.mkdir(parents=True, exist_ok=True)
        fires.to_csv(out / "fires.csv", index=False)
        report.drop(columns=["_run_index"]).to_csv(out / "incidents.csv", index=False)
        gates = {
            k: getattr(self.gates, k)
            for k in ("first_year", "last_year", "min_acres", "min_days", "max_gap_days", "max_link_km")
        }
        summary = {
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gates": gates | {"region": list(self.gates.region)},
            "incidents": len(report),
            "fires": len(fires),
            "fires_by_archive": fires["archive"].value_counts().to_dict(),
            "not_fires_by_reason": report.loc[report["reason"] != "", "reason"].value_counts().to_dict(),
        }
        (out / "gates.json").write_text(json.dumps(summary, indent=1, default=str))
        log.info(f"{len(fires)} fires from {len(report)} incidents; by archive {summary['fires_by_archive']}")
        for why, n in summary["not_fires_by_reason"].items():
            log.info(f"  not fires: {n:6d}  {why}")
        if self.uploads:
            stage = Path(self.s.work) / "catalog_upload"
            (stage / "catalog").mkdir(parents=True, exist_ok=True)
            for name in ("fires.csv", "incidents.csv", "gates.json"):
                (stage / "catalog" / name).write_bytes((out / name).read_bytes())
            self.hub.upload_folder(stage, f"Fire catalog: {len(fires)} fires from {len(report)} incidents")

    def _raw_paths(self, kind: str) -> list[str]:
        local = sorted(str(p.relative_to(self.root)) for p in (self.root / self.RAW / kind).glob("*.parquet"))
        remote = [p for p in self.hub.files(f"{self.RAW}/{kind}") if p.endswith(".parquet")] if self.uploads else []
        return sorted(set(local) | set(remote))

    def _table(self, path: str, columns: set[str] | None = None) -> pd.DataFrame:
        """A raw table (downloaded when not here yet), optionally only the columns named (any case)."""
        local = self.root / path
        if not local.exists():
            self.hub.download(path, self.root)
        if columns is None:
            return pd.read_parquet(local)
        import pyarrow.parquet as pq

        names = pq.read_schema(local).names
        return pd.read_parquet(local, columns=[n for n in names if n.lower() in columns or n.startswith("_")])

    def _evaluation_frame(self) -> pd.DataFrame:
        """Every perimeter record, field names in lower case, only the columns the gates read."""
        tables = [self._table(p, self.READ_COLUMNS).rename(columns=str.lower) for p in self._raw_paths("perimeters")]
        if not tables:
            raise RuntimeError(f"no raw perimeter tables under {self.RAW}: run --catalog scan first")
        return pd.concat(tables, ignore_index=True)

    def _incident_table(self) -> pd.DataFrame:
        tables = [self._table(p) for p in self._raw_paths("incidents")]
        inc = (
            pd.concat([t for t in tables if len(t)], ignore_index=True)
            if any(len(t) for t in tables)
            else pd.DataFrame()
        )
        if inc.empty:
            return pd.DataFrame(columns=["_uuid"])
        inc["_uuid"] = inc["IrwinID"].map(lambda v: IncidentKey.irwin_uuid(str(v)))
        return inc.drop_duplicates("_uuid", keep="last")

    def _fires(self, gated: pd.DataFrame, records: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
        """One row per fire: its main run of perimeters, exact box, and its WFIGS incident fields."""
        nifc = NifcArchive.for_run(RunContext("catalog", Path(self.s.work), threads=self.s.threads))
        wild = records[records["wildfire"] & records["t"].notna() & ~records["deleted"]]
        span = wild.groupby("key")["t"].agg(["min", "max"])
        inc = incidents.set_index("_uuid")

        def one(row: pd.Series) -> dict:
            run = records.loc[row["_run_index"]].sort_values("t")
            box = None
            for url, part in run.groupby("_url"):
                b = nifc.extent(url, part)
                box = b if box is None else (box.union(b) if b else box)
            first = run.iloc[0]
            out = {"fire_id": row["fire_id"], "incident": row["incident"],
                   "irwin_id": IncidentKey.irwin_uuid(row["incident"]),
                   "name": str(run["incidentname"].dropna().iloc[-1]) if run["incidentname"].notna().any() else "",
                   "archive": row["archive"], "year": int(row["year"]),
                   "first": run["t"].iloc[0].isoformat(), "last": run["t"].iloc[-1].isoformat(),
                   "first_record_utc": span.at[row["incident"], "min"].isoformat(),
                   "last_record_utc": span.at[row["incident"], "max"].isoformat(),
                   "days": int(row["days"]), "all_days": int(row["all_days"]), "records": len(run),
                   "all_records": int(row["all_records"]), "max_gis_acres": float(run["acres"].max()),
                   "map_methods": ",".join(sorted(run["mapmethod"].dropna().astype(str).str.lower().unique()))
                   if "mapmethod" in run else "",
                   "layers": ",".join(sorted(run["_layer"].unique())),
                   "west": box.west if box else None, "south": box.south if box else None,
                   "east": box.east if box else None, "north": box.north if box else None,
                   "first_lat": first["_cy"], "first_lon": first["_cx"],
                   "first_record_time": first["t"].isoformat()}  # fmt: skip
            r = inc.loc[out["irwin_id"]] if out["irwin_id"] in inc.index else None
            for col, field in self.INCIDENT_COLUMNS.items():
                v = r.get(field) if r is not None else None
                if col in self.TIME_COLUMNS and pd.notna(pd.to_numeric(v, errors="coerce")):
                    v = pd.Timestamp(int(float(v)), unit="ms", tz="UTC").isoformat()
                out[col] = v
            return out

        rows = []
        with ThreadPoolExecutor(self.s.threads) as ex:
            for k, out in enumerate(ex.map(one, (row for _, row in gated.iterrows())), 1):
                rows.append(out)
                if k % 250 == 0:
                    log.info(f"  boxes and incident fields: {k} of {len(gated)} fires")
        return pd.DataFrame(rows).sort_values("fire_id") if rows else pd.DataFrame(columns=["fire_id"])
