"""NIFC perimeter and event layers: Historic GeoMAC services, NIFS / Operational Data Archive services, and the
archive years published only as a File Geodatabase download."""

from __future__ import annotations

import logging
import re
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from fah.core.context import RunContext
from fah.core.geometry import BoundingBox
from fah.sources.arcgis import ArcGisClient
from fah.sources.http import HttpClient

log = logging.getLogger(__name__)


class NifcArchive:
    """Layer discovery and box queries. Geodatabase layers are addressed as 'fgdb://<path.gdb>|<layer>'; they are
    downloaded once into fgdb_dir and cached as Parquet parts next to the .gdb."""

    ROOT = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
    # Years whose Operational Data Archive exists only as a File Geodatabase download: ArcGIS Online item ids
    # (owner NIFC_Authoritative).
    FGDB_ITEMS = {2020: "ea843f7f091f4c7f9743798b64c864be", 2021: "af727c41d79643b091cee372233110d4"}
    ITEM_DATA = "https://www.arcgis.com/sharing/rest/content/items/{}/data"
    ARCHIVE_SERVICE = re.compile(
        r"(National_Incident_Feature_Service|NIFS_OpenData|NIFS_Archive|Operational_Data_Archive)", re.I
    )
    FGDB_CHUNK = 20000  # features read per pass when caching a geodatabase layer
    LISTING_TRIES = 4

    def __init__(self, arcgis: ArcGisClient, http: HttpClient, fgdb_dir: Path) -> None:
        self.arcgis, self.http, self.fgdb_dir = arcgis, http, Path(fgdb_dir)
        self._names: set[str] | None = None

    @classmethod
    def for_run(cls, ctx: RunContext) -> NifcArchive:
        """The run's archive: one per run, its geodatabase downloads in the channel cache (deleted at the end)."""
        return ctx.memo("nifc", lambda: cls(ArcGisClient(ctx.session), HttpClient(ctx.session), ctx.cache / "fgdb"))

    # ------------------------------------------------------------------------------------- names

    @staticmethod
    def layer_is(name: str, *kinds: str) -> bool:
        """True when an archive layer name is one of `kinds`, however the year spells it: EventPolygon (2020 gdb),
        EventPolygon2018 / EventLine2019 (services), Event_Polygon_2021 (2021 gdb) all match 'eventpolygon' /
        'eventline' once only letters are compared."""
        letters = re.sub(r"[^a-z]", "", name.lower())
        return any(letters.startswith(k) for k in kinds) and "attach" not in letters

    @staticmethod
    def layer_label(url: str) -> str:
        """A layer's short name for manifests: '<service>/FeatureServer/<n>' or '<gdb>/<layer>'."""
        if url.startswith("fgdb://"):
            gdb, layer = url[len("fgdb://") :].split("|")
            return f"{Path(gdb).name}/{layer}"
        return url.split("/services/")[1]

    def source_link(self, url: str, box: BoundingBox) -> str:
        """What a manifest cites for a layer queried over the box: the query itself, or the geodatabase download."""
        if url.startswith("fgdb://"):
            gdb, layer = url[len("fgdb://") :].split("|")
            year = next((y for y in self.FGDB_ITEMS if f"/{y}/" in gdb.replace("\\", "/")), None)
            item = self.ITEM_DATA.format(self.FGDB_ITEMS[year]) if year else gdb
            return f"{item}#{layer}"
        return self.arcgis.query_url(url, box)

    # ------------------------------------------------------------------------------------- discovery

    def service_names(self) -> set[str]:
        """Every feature service of the NIFC organisation, listed once per run. ArcGIS Online now and then answers
        this listing with no services at all; that is retried, then raised, so the fire is recorded as an error and
        fetched again on the next run instead of being recorded 'empty' for good."""
        if self._names is None:
            for attempt in range(self.LISTING_TRIES):
                js = self.arcgis.get_json(self.ROOT)
                names = {s["name"] for s in js.get("services", []) if s.get("type") == "FeatureServer"}
                if names:
                    self._names = names
                    break
                log.info(f"    NIFC service listing came back empty (try {attempt + 1}), asking again")
                time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"{self.ROOT}: the service listing came back empty {self.LISTING_TRIES} times")
        return self._names

    def _archive_services(self, year: int) -> tuple[list[str], set[str]]:
        """(the year's NIFS / Operational Data Archive services, every feature service name)."""
        names = self.service_names()
        return sorted(n for n in names if str(year) in n and self.ARCHIVE_SERVICE.search(n)), names

    def _service_layers(self, service: str) -> list[dict]:
        """A service's layers; an empty list is a failed answer (an archive service always has layers)."""
        layers = self.arcgis.get_json(f"{self.ROOT}/{service}/FeatureServer").get("layers", [])
        if not layers:
            raise RuntimeError(f"{service}: the service listed no layers")
        return layers

    def perimeter_layers(self, year: int) -> list[tuple[str, str]]:
        """[(kind, layer_url)] of the year's perimeter layers: 'geomac' (Historic GeoMAC) and 'nifs' (EventPolygon
        of the NIFS / Operational Data Archive services, or of the year's geodatabase)."""
        services, names = self._archive_services(year)
        found = []
        geomac = next((n for n in names if n.lower() == f"historic_geomac_perimeters_{year}"), None)  # 2019: "GeoMAC"
        if geomac:
            found.append(("geomac", f"{self.ROOT}/{geomac}/FeatureServer/0"))
        for n in services:
            try:
                layers = self._service_layers(n)
            except Exception as ex:  # noqa: BLE001  (one broken service must not hide the others)
                log.info(f"  could not open {n}: {ex}")
                continue
            for lyr in layers:
                if self.layer_is(lyr["name"], "eventpolygon"):
                    found.append(("nifs", f"{self.ROOT}/{n}/FeatureServer/{lyr['id']}"))
        if not any(k == "nifs" for k, _ in found) and year in self.FGDB_ITEMS:
            found += [("nifs", u) for u in self._fgdb_layers(year, ("eventpolygon",))]
        if not any(k == "nifs" for k, _ in found) and year >= 2018:
            hints = sorted(n for n in names if str(year) in n and re.search(r"(nifs|incident|archive|event)", n, re.I))
            log.info(f"  WARNING: no Operational Data Archive layer found for {year}. Candidates: {hints}")
        log.info(f"  {year}: " + (", ".join(self.layer_label(u) for _, u in found) or "nothing found"))
        return found

    def event_layers(self, year: int) -> list[tuple[str, str]]:
        """[(layer_url, 'line' | 'polygon')] of the year's suppression layers (EventLine / EventPolygon)."""
        services, _ = self._archive_services(year)
        found = []
        for n in services:
            for lyr in self._service_layers(n):
                if self.layer_is(lyr["name"], "eventline"):
                    found.append((f"{self.ROOT}/{n}/FeatureServer/{lyr['id']}", "line"))
                elif self.layer_is(lyr["name"], "eventpolygon"):
                    found.append((f"{self.ROOT}/{n}/FeatureServer/{lyr['id']}", "polygon"))
        if not found and year in self.FGDB_ITEMS:
            for u in self._fgdb_layers(year, ("eventline", "eventpolygon")):
                found.append((u, "line" if "line" in self.layer_label(u).lower() else "polygon"))
        log.info(
            f"  {year}: " + (", ".join(self.layer_label(u) for u, _ in found) or "no archive (no spatial suppression)")
        )
        return found

    # ------------------------------------------------------------------------------------- queries

    def features(self, url: str, box: BoundingBox) -> list[dict]:
        """Every feature of a layer in the box: services by id (ArcGisClient), geodatabases from the local cache."""
        if url.startswith("fgdb://"):
            return self._read_fgdb(url, box)
        return self.arcgis.box_features(url, box)

    def layer_records(self, url: str) -> pd.DataFrame:
        """Every record of a layer, all fields as served (no geometry), plus `_oid` (its object id), `_cx`, `_cy`
        (centroid, lon / lat) and, for geodatabase layers, `_xmin` .. `_ymax` (its box)."""
        if url.startswith("fgdb://"):
            return self._fgdb_records(url)
        oid_field = self.arcgis.get_json(url).get("objectIdField") or "OBJECTID"
        rows = []
        for f in self.arcgis.records(url, centroid=True):
            a = dict(f.get("attributes") or {})
            c = f.get("centroid") or {}
            a.update(_oid=a.get(oid_field), _cx=c.get("x"), _cy=c.get("y"))
            rows.append(a)
        return pd.DataFrame(rows)

    def extent(self, url: str, records: pd.DataFrame) -> BoundingBox | None:
        """The exact box of these records of one layer: computed by the service, or from the geodatabase boxes."""
        if url.startswith("fgdb://"):
            b = records[["_xmin", "_ymin", "_xmax", "_ymax"]].apply(pd.to_numeric, errors="coerce")
            if b.isna().all().any():
                return None
            return BoundingBox(b["_xmin"].min(), b["_ymin"].min(), b["_xmax"].max(), b["_ymax"].max())
        ids = sorted(int(i) for i in pd.to_numeric(records["_oid"], errors="coerce").dropna())
        return self.arcgis.extent(url, ids) if ids else None

    def _fgdb_records(self, url: str) -> pd.DataFrame:
        import shapely

        gdb, layer = url[len("fgdb://") :].split("|")
        parts = []
        for p in sorted(self._fgdb_cache(gdb, layer).glob("part-*.parquet")):
            df = pd.read_parquet(p)
            c = shapely.centroid(shapely.from_wkb(df.pop("wkb").values))
            df = df.rename(columns={k: f"_{k}" for k in ("xmin", "ymin", "xmax", "ymax")})
            df["_cx"], df["_cy"], df["_oid"] = shapely.get_x(c), shapely.get_y(c), None
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _fgdb_layers(self, year: int, prefixes: tuple[str, ...]) -> list[str]:
        """Download (once) the year's archive geodatabase; fgdb:// urls of its matching layers."""
        import pyogrio

        d = self.fgdb_dir / str(year)
        d.mkdir(parents=True, exist_ok=True)
        gdbs = sorted(d.rglob("*.gdb"))
        if not gdbs:
            zpath = d / "archive.zip"
            if not zpath.exists():
                log.info(f"  downloading {year} Operational Data Archive geodatabase (one time, can be large) ...")
                self.http.download(self.ITEM_DATA.format(self.FGDB_ITEMS[year]), zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(d)
            gdbs = sorted(d.rglob("*.gdb"))
        if not gdbs:
            log.info(f"  WARNING: no .gdb found in the {year} archive download")
            return []
        out = []
        for gdb in gdbs:
            layers = [row[0] for row in pyogrio.list_layers(gdb)]
            log.info(f"  {year} geodatabase {gdb.name} layers: {layers}")
            out += [f"fgdb://{gdb}|{name}" for name in layers if self.layer_is(name, *prefixes)]
        return out

    def _fgdb_cache(self, gdb: str, layer: str) -> Path:
        """A geodatabase layer read once, in chunks, into lon/lat Parquet parts with a box per feature. Box reads do
        not work on the 2021 archive geodatabase (its spatial index returns nothing), and reading the whole layer at
        once needs 4+ GB of memory."""
        import pyogrio
        import shapely

        d = Path(gdb).parent / f"{Path(gdb).stem}__{layer}.parquet.d"
        done = d / "_COMPLETE"
        if done.exists():
            return d
        d.mkdir(parents=True, exist_ok=True)
        n = pyogrio.read_info(gdb, layer=layer)["features"]
        log.info(f"  caching {Path(gdb).name}/{layer}: {n} features (one time) ...")
        for k, skip in enumerate(range(0, n, self.FGDB_CHUNK)):
            part = d / f"part-{k:04d}.parquet"
            if part.exists():
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="organizePolygons")
                g = pyogrio.read_dataframe(
                    gdb, layer=layer, skip_features=skip, max_features=self.FGDB_CHUNK, datetime_as_string=True
                )
            g = g[g.geometry.notna() & ~g.geometry.is_empty]
            if g.crs is not None and g.crs.to_epsg() != 4326:
                g = g.to_crs(4326)
            b = g.geometry.bounds
            df = pd.DataFrame(g.drop(columns="geometry"))
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].astype("string")
            df["xmin"], df["ymin"], df["xmax"], df["ymax"] = (
                b["minx"].values,
                b["miny"].values,
                b["maxx"].values,
                b["maxy"].values,
            )
            df["wkb"] = shapely.to_wkb(g.geometry.values)
            df.to_parquet(part, index=False)
        done.write_text("ok")
        return d

    def _read_fgdb(self, layer_url: str, box: BoundingBox) -> list[dict]:
        import geopandas as gpd
        import shapely
        from shapely.geometry import box as shapely_box

        gdb, layer = layer_url[len("fgdb://") :].split("|")
        d = self._fgdb_cache(gdb, layer)
        w, s, e, n = box
        flt = [("xmax", ">=", w), ("xmin", "<=", e), ("ymax", ">=", s), ("ymin", "<=", n)]
        parts = [p for p in (pd.read_parquet(p, filters=flt) for p in sorted(d.glob("part-*.parquet"))) if len(p)]
        if not parts:
            return []
        df = pd.concat(parts, ignore_index=True)
        geom = shapely.from_wkb(df.pop("wkb").values)
        g = gpd.GeoDataFrame(df.drop(columns=["xmin", "ymin", "xmax", "ymax"]), geometry=geom, crs=4326)
        g = g[g.intersects(shapely_box(*box))]
        if g.empty:
            return []
        feats = []
        for props, geom in zip(g.drop(columns="geometry").to_dict("records"), g.geometry, strict=True):
            if geom is None:
                continue
            for k, v in props.items():
                if v is pd.NA or v is pd.NaT:
                    props[k] = None
                elif isinstance(v, pd.Timestamp):
                    props[k] = (
                        None
                        if pd.isna(v)
                        else int((v.tz_localize("UTC") if v.tzinfo is None else v).timestamp() * 1000)
                    )
                elif isinstance(v, float) and np.isnan(v):
                    props[k] = None
            feats.append({"type": "Feature", "geometry": geom.__geo_interface__, "properties": props})
        return feats
