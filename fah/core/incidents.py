"""Which incidents of the NIFC perimeter archives are fires of this dataset: incident keys and fire ids, how each
archive's records are read, and the gates. Records are never changed; the interpreted times and days below decide
the gates and each fire's window only.

Two archives, read by their own rules:
  GeoMAC (2014-2019)  no category field; perimeterdatetime is a local clock time stored as if it were UTC (2019: Denver
                      time for every fire, earlier years the fire's own zone)
  NIFS / ODA (2018+)  FeatureCategory; PolygonDateTime in true UTC; every edit is a new version; DeleteThis flags
A fire's window and box come from its main run of perimeters: records chained in space (centroids within
max_link_km of one another) and in time (no gap over max_gap_days), so a record misdated by weeks, or one carrying
another fire's IRWIN id hundreds of km away, cannot stretch them; every record stays in the raw tables."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fah.core.geometry import BoundingBox


class IncidentKey:
    """The incident key and fire id of the earlier catalog, kept verbatim so existing fire folders keep their ids.
    An IRWIN id is stripped of braces and spaces only (a few keys keep trailing junk such as '}\\n'); without one, the
    key is 'name:<lower-case name>|<unique fire id>'."""

    @staticmethod
    def of(irwin: object, name: object, unique_id: object) -> str:
        irwin = None if pd.isna(irwin) else str(irwin).strip("{} ").upper()
        if irwin:
            return irwin
        uid = "" if pd.isna(unique_id) else str(unique_id)
        name = re.sub(r"\s+", " ", ("" if pd.isna(name) else str(name)).lower()).strip()
        return f"name:{name}|{uid}"

    @staticmethod
    def fire_id(key: str) -> str:
        return "F" + hashlib.sha1(key.encode()).hexdigest()[:10]

    @staticmethod
    def irwin_uuid(key: str) -> str | None:
        """The IRWIN uuid of a key (junk removed, for joining other tables), or None for a name key."""
        return None if key.startswith("name:") else re.sub(r"[^0-9A-F-]", "", key.upper()) or None


class ArchiveRules:
    """How one archive's records are read. `records` columns are the layer's own field names in lower case, plus
    `_cx`, `_cy` (record centroid, lon / lat)."""

    name = ""
    TIME_FIELDS = ("perimeterdatetime", "polygondatetime", "datecurrent", "gdb_from_date")
    WILDFIRE_CATEGORIES = frozenset({"Wildfire Daily Fire Perimeter", "Wildfire Final Fire Perimeter"})
    NOT_COUNTED_METHODS = frozenset({"hand sketch", "modeled"})  # not counted for size and days (still kept raw)
    BURN_DAY_SHIFT_H = 6  # perimeters drawn before 06:00 local count for the day before

    def record_time(self, records: pd.DataFrame) -> pd.Series:
        """Each record's time, UTC: the first of TIME_FIELDS that holds a date between 2000 and 2030."""
        out = pd.Series(pd.NaT, index=records.index, dtype="datetime64[ns, UTC]")
        for c in self.TIME_FIELDS:
            if c not in records:
                continue
            v = records[c]
            num = pd.to_numeric(v, errors="coerce")
            t = pd.to_datetime(
                num.where(num.between(946684800000, 1893456000000)), unit="ms", utc=True, errors="coerce"
            )
            s = v.where(num.isna()).astype("string")
            year = pd.to_numeric(s.str.extract(r"^(\d{4})", expand=False), errors="coerce")
            t2 = pd.to_datetime(s.where(year.between(2000, 2030)), utc=True, errors="coerce", format="mixed")
            out = out.fillna(t.fillna(t2.astype("datetime64[ns, UTC]")).astype("datetime64[ns, UTC]"))
        return out

    def is_wildfire(self, records: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=records.index)

    def deleted(self, records: pd.DataFrame) -> pd.Series:
        if "deletethis" not in records:
            return pd.Series(False, index=records.index)
        return (
            records["deletethis"].astype("string").str.strip().str.lower().isin(["yes", "y", "true", "1"]).fillna(False)
        )

    def counted(self, records: pd.DataFrame) -> pd.Series:
        """Records that count for the size and day gates (hand sketches and modeled shapes do not)."""
        method = records["mapmethod"] if "mapmethod" in records else pd.Series("", index=records.index)
        return ~method.astype("string").str.lower().isin(self.NOT_COUNTED_METHODS).fillna(False)

    def keys(self, records: pd.DataFrame) -> pd.Series:
        uid = records.get("uniquefireidentifier", pd.Series(None, index=records.index))
        uid = uid.where(uid.notna(), records.get("localincidentidentifier", pd.Series(None, index=records.index)))
        irwin = records.get("irwinid", pd.Series(None, index=records.index))
        return pd.Series([IncidentKey.of(i, n, u) for i, n, u in zip(irwin, records["incidentname"], uid, strict=True)],
                         index=records.index)  # fmt: skip

    @classmethod
    def burn_day(cls, t: pd.Series, lon: pd.Series) -> pd.Series:
        """Local solar date, shifted so night infrared flights count toward the day that just burned."""
        local = t + pd.to_timedelta(lon / 15.0, unit="h") - pd.Timedelta(hours=cls.BURN_DAY_SHIFT_H)
        return local.dt.tz_localize(None).dt.normalize()


class GeomacRules(ArchiveRules):
    name = "geomac"

    @staticmethod
    def zone(lon: float, lat: float, year: int) -> str:
        """The clock a GeoMAC perimeter time was written in: Denver time for every 2019 record, earlier years the
        fire's own local time (NIROPS comments like '2338hrs' equal the stored time)."""
        if year >= 2019:
            return "America/Denver"
        if lon < -130:
            return "America/Anchorage"
        if lon > -85.5:
            return "America/New_York"
        if lon > -101.5:
            return "America/Chicago"
        if lat < 37.0 and -114.8 < lon < -109.05:
            return "America/Phoenix"
        if lon > -114.05 or (lon > -117.03 and 42.0 <= lat <= 45.6):
            return "America/Denver"
        return "America/Los_Angeles"

    def record_time(self, records: pd.DataFrame) -> pd.Series:
        t = super().record_time(records)
        years = pd.to_numeric(records.get("_year"), errors="coerce").fillna(t.dt.year)
        out = [ti.tz_localize(None).tz_localize(self.zone(x, y, int(yr)), ambiguous=True, nonexistent="shift_forward")
               .tz_convert("UTC") if pd.notna(ti) and pd.notna(x) and pd.notna(y) else ti
               for ti, x, y, yr in zip(t, records["_cx"], records["_cy"], years, strict=True)]  # fmt: skip
        return pd.Series(pd.DatetimeIndex(out), index=records.index).astype("datetime64[ns, UTC]")


class NifsRules(ArchiveRules):
    name = "nifs"

    def is_wildfire(self, records: pd.DataFrame) -> pd.Series:
        if "featurecategory" not in records:
            return pd.Series(True, index=records.index)
        return records["featurecategory"].astype("string").isin(self.WILDFIRE_CATEGORIES).fillna(False)


@dataclass(frozen=True, slots=True)
class CatalogGates:
    """Which incidents are fires of the dataset. Defaults are those of the earlier catalog."""

    first_year: int = 2014
    last_year: int = 2025
    min_acres: float = 1000.0
    min_days: int = 3
    max_gap_days: int = 10  # perimeters further apart start a new run; the run with the most days is the fire's
    max_link_km: float = 25.0  # centroids not chained within this distance are separate groups (mis-tagged records)
    region: BoundingBox = BoundingBox(-125.0, 24.0, -66.0, 50.0)  # continental US, by record centroid
    RULES = {"geomac": GeomacRules(), "nifs": NifsRules()}

    def records(self, raw: pd.DataFrame) -> pd.DataFrame:
        """The raw records (one archive's tables concatenated, field names lower case, `_archive`, `_cx`, `_cy`,
        `_year` added) with the columns the gates read: key, t, day, wildfire, deleted, counted, acres."""
        parts = []
        for archive, rec in raw.groupby("_archive", sort=False):
            rules = self.RULES[archive]
            rec = rec.copy()
            rec["key"] = rules.keys(rec)
            rec["t"] = rules.record_time(rec)
            rec["wildfire"], rec["deleted"], rec["counted"] = (
                rules.is_wildfire(rec),
                rules.deleted(rec),
                rules.counted(rec),
            )
            rec["acres"] = pd.to_numeric(rec.get("gisacres"), errors="coerce")
            rec["day"] = ArchiveRules.burn_day(rec["t"], pd.to_numeric(rec["_cx"], errors="coerce"))
            parts.append(rec)
        return pd.concat(parts, ignore_index=True) if parts else raw.assign(key=[], t=[], day=[])

    def evaluate(self, records: pd.DataFrame, incident_types: dict[str, str] | None = None) -> pd.DataFrame:
        """One row per incident: archive, record counts, size and days, `reason` ('' when it is a fire). Fires also get
        their main run of perimeters: `days`, `year` and the record index labels of the run in `_run_index`. Whole-
        table gates run first; the main run is worked out only for incidents that pass them."""
        incident_types = incident_types or {}
        usable = records[records["t"].notna() & ~records["deleted"]]
        if usable.empty:
            return pd.DataFrame(columns=["incident", "fire_id", "archive", "reason"])
        cx, cy = pd.to_numeric(usable["_cx"], errors="coerce"), pd.to_numeric(usable["_cy"], errors="coerce")
        inside = cx.between(self.region.west, self.region.east) & cy.between(self.region.south, self.region.north)
        counted = usable[usable["wildfire"] & inside & usable["counted"]]
        g = usable.groupby("key")
        archive = g["_archive"].agg(lambda a: a.iloc[0] if a.nunique() == 1 else "both")
        out = pd.DataFrame({"all_records": g.size(), "wildfire_records": g["wildfire"].sum(), "archive": archive})
        c = counted.groupby("key")
        out["max_gis_acres"] = c["acres"].max()
        out["all_days"] = c["day"].nunique()
        out["year"] = c["t"].max().dt.year
        out["in_region"] = usable[usable["wildfire"] & inside].groupby("key").size().reindex(out.index).fillna(0) > 0
        out["type"] = [incident_types.get(IncidentKey.irwin_uuid(k) or "") for k in out.index]
        out["reason"] = ""
        rules = [(out["wildfire_records"] == 0, "not a wildfire perimeter"),
                 (out["type"] == "RX", "prescribed fire (WFIGS incident type)"),
                 (~out["in_region"], "outside the continental US"),
                 (out["all_days"].isna(), "only hand-sketched or modeled perimeters"),
                 (~out["year"].between(self.first_year, self.last_year), f"outside {self.first_year}-{self.last_year}"),
                 (~(out["max_gis_acres"] >= self.min_acres), f"under {self.min_acres:g} acres"),
                 (~(out["all_days"] >= self.min_days), f"under {self.min_days} burn days")]  # fmt: skip
        for fails, why in rules:
            out.loc[fails & (out["reason"] == ""), "reason"] = why
        out["days"], out["_run_index"] = out["all_days"], None
        by_key = counted.groupby("key").groups
        for key in out.index[out["reason"] == ""]:
            run = self._main_run(counted.loc[by_key[key]])
            out.at[key, "days"], out.at[key, "_run_index"] = run["day"].nunique(), list(run.index)
            out.at[key, "year"] = run["t"].max().year
            if not run["acres"].max() >= self.min_acres:
                out.at[key, "reason"] = f"under {self.min_acres:g} acres in its main run of perimeters"
            elif run["day"].nunique() < self.min_days:
                out.at[key, "reason"] = f"under {self.min_days} burn days in its main run of perimeters"
        out.index.name = "incident"
        out = out.reset_index()
        out.insert(1, "fire_id", out["incident"].map(IncidentKey.fire_id))
        return out

    def _main_run(self, rec: pd.DataFrame) -> pd.DataFrame:
        """The perimeters of the main spatial group, split where more than max_gap_days pass without one; the run
        with the most burn days (ties: the most growth)."""
        if rec.empty:
            return rec
        rec = self._main_group(rec).sort_values("t")
        gap = rec["t"].diff().dt.total_seconds().fillna(0) / 86400
        seg = (gap > self.max_gap_days).cumsum()
        if seg.max() == 0:
            return rec

        def score(k: int) -> tuple:
            part = rec[seg == k]
            return part["day"].nunique(), (part["acres"].max() - part["acres"].min())

        return rec[seg == max(seg.unique(), key=score)]

    def _main_group(self, rec: pd.DataFrame) -> pd.DataFrame:
        """Records whose centroids chain together within max_link_km (single linkage); the group with the most burn
        days (ties: the most acres). One far-off record, such as another fire's carrying this IRWIN id, is left out."""
        xy = rec[["_cx", "_cy"]].apply(pd.to_numeric, errors="coerce")
        points, which = np.unique(xy.round(3).to_numpy(), axis=0, return_inverse=True)
        which = which.reshape(-1)
        km = np.c_[points[:, 0] * 111.32 * np.cos(np.radians(np.nanmedian(points[:, 1]))), points[:, 1] * 110.57]
        group = np.full(len(points), -1)
        for seed in range(len(points)):
            if group[seed] >= 0:
                continue
            group[seed], frontier = seed, [seed]
            while frontier:
                near = np.zeros(len(points), bool)
                for i in frontier:
                    near |= np.hypot(*(km - km[i]).T) <= self.max_link_km
                frontier = np.flatnonzero(near & (group < 0)).tolist()
                group[frontier] = seed
        labels = pd.Series(group[which], index=rec.index)
        if labels.nunique() == 1:
            return rec

        def score(k: int) -> tuple:
            part = rec[labels == k]
            return part["day"].nunique(), part["acres"].max()

        return rec[labels == max(labels.unique(), key=score)]
