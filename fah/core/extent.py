"""A fire's extent from its FIRMS detections: the days it was active, the time window those days give, and its
ignition point. One row of catalog/fire_extents.csv per fire."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from fah.core.clock import Utc
from fah.core.fire import Fire, Ignition
from fah.core.geometry import BoundingBox


class FireExtentEstimator:
    """Time window and ignition point of a fire from its FIRMS detections and first mapped perimeter."""

    MIN_DETECTIONS = 2  # an active day: at least this many detections that are not low confidence
    MAX_GAP_DAYS = 2  # activity joins the fire across at most this many quiet days (clouds, gaps)
    IGNITION_OVERPASS_MIN = 15  # detections this close to the earliest one belong to the same overpass
    ORIGIN_MAX_LEAD_DAYS = 14  # a reported discovery earlier than this before the first perimeter is not trusted

    @staticmethod
    def fire_detections(df: pd.DataFrame, box: BoundingBox, pad: float = Fire.DETECTION_PAD_DEG) -> pd.Series:
        """Mask of the detections that count for the fire: inside the box (+ pad), not low confidence, and not a
        static land source or offshore (type 2 / 3) where the source has a type column."""
        m = df["latitude"].between(box.south - pad, box.north + pad) & df["longitude"].between(
            box.west - pad, box.east + pad
        )
        if "confidence" in df:
            m &= ~df["confidence"].astype(str).str.lower().str.startswith("l")
        if "type" in df:
            m &= ~pd.to_numeric(df["type"], errors="coerce").isin([2, 3])
        return m

    @classmethod
    def active_days(
        cls, df: pd.DataFrame | None, box: BoundingBox, pad: float = Fire.DETECTION_PAD_DEG, min_n: int | None = None
    ) -> list[pd.Timestamp]:
        """Sorted UTC days (naive midnights) with at least min_n detections that count for the fire."""
        if df is None or not len(df):
            return []
        m = cls.fire_detections(df, box, pad)
        count = pd.to_datetime(df.loc[m, "acq_date"]).dt.normalize().value_counts()
        return sorted(count[count >= (cls.MIN_DETECTIONS if min_n is None else min_n)].index)

    @classmethod
    def activity_window(
        cls, first_day: pd.Timestamp, last_day: pd.Timestamp, active: Sequence[pd.Timestamp], max_gap: int | None = None
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(start, end): the perimeter days, stretched over active days that chain onto them with at most max_gap
        quiet days between links (activity weeks away belongs to another fire or a false alarm)."""
        step = pd.Timedelta(days=(cls.MAX_GAP_DAYS if max_gap is None else max_gap) + 1)
        s, e = first_day, last_day
        for d in sorted((d for d in active if d < first_day), reverse=True):
            if s - d > step:
                break
            s = d
        for d in sorted(d for d in active if d > last_day):
            if d - e > step:
                break
            e = d
        return s, e

    @classmethod
    def ignition(
        cls,
        fire: Fire,
        firms_df: pd.DataFrame | None = None,
        active: Sequence[pd.Timestamp] = (),
        first_perimeter: tuple | None = None,
    ) -> Ignition:
        """The fire's earliest observation, known at its start. The earliest of: the first VIIRS overpass that saw
        it (detections within IGNITION_OVERPASS_MIN minutes of the first one, on the first active day of its
        chained window: 'firms'; on any day of the perimeter span when no day is active: 'firms_single'), its first
        perimeter record ('first_perimeter': the catalog's, unless one is given), and the incident's reported discovery
        time and location ('reported_origin', WFIGS; ignored when it falls outside ORIGIN_MAX_LEAD_DAYS before the first
        perimeter day to a day after the last). Without any, the centre of the perimeters' box ('perimeter_box_centre':
        uses the final extent, so not leak-free)."""
        cands = []
        box = fire.box
        if firms_df is not None and len(firms_df):
            d = firms_df[cls.fire_detections(firms_df, box)]
            day = pd.to_datetime(d["acq_date"]).dt.normalize()
            hhmm = (
                pd.to_numeric(d["acq_time"], errors="coerce").fillna(0).astype(int)
                if "acq_time" in d
                else pd.Series(0, index=d.index)
            )
            t = (day + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="min")).dt.tz_localize(
                "UTC"
            )
            p0, p1 = fire.perimeter_days()
            a0, a1 = cls.activity_window(p0, p1, list(active))
            sel, method = day.isin(set(active)) & day.between(a0, a1), "firms"
            if not sel.any():
                sel, method = day.between(p0 - pd.Timedelta(days=1), p1), "firms_single"
            if sel.any():
                t0 = t[sel].min()
                g = sel & (t <= t0 + pd.Timedelta(minutes=cls.IGNITION_OVERPASS_MIN))
                cands.append((t0, float(d.loc[g, "latitude"].mean()), float(d.loc[g, "longitude"].mean()), method))
        first_perimeter = first_perimeter if first_perimeter is not None else fire.first_record
        if first_perimeter is not None and all(pd.notna(v) for v in first_perimeter):
            t, lat, lon = first_perimeter
            cands.append((Utc.stamp(t), float(lat), float(lon), "first_perimeter"))
        if fire.origin is not None and all(pd.notna(v) for v in fire.origin):
            t, lat, lon = fire.origin
            t, (p0, p1) = Utc.stamp(t), fire.perimeter_days()
            if p0 - pd.Timedelta(days=cls.ORIGIN_MAX_LEAD_DAYS) <= t.tz_convert(None) <= p1 + pd.Timedelta(days=1):
                cands.append((t, float(lat), float(lon), "reported_origin"))
        if cands:
            t0, lat, lon, method = min(cands, key=lambda c: c[0])
            return Ignition(lat, lon, t0, method)
        lat, lon = box.centre()
        return Ignition(lat, lon, Utc.stamp(fire.first), "perimeter_box_centre")

    @staticmethod
    def is_final(fire: Fire, row: Mapping[str, object] | None) -> bool:
        """Whether a fire_extents.csv row is the fire's final window: its firms rows were current when the row was
        written (`firms_current`; rows from before that column count as current) and its perimeter days are still
        the catalog's. A window that is not final can still grow, and every time-dependent channel would fetch the
        fire again."""
        if row is None or str(row.get("firms_current", "")).strip().lower() in ("false", "0"):
            return False
        p0, p1 = fire.perimeter_days()
        days = (str(row.get("first_perimeter_day")), str(row.get("last_perimeter_day")))
        return days == (p0.date().isoformat(), p1.date().isoformat())

    @classmethod
    def row(cls, fire: Fire, firms_df: pd.DataFrame | None = None, first_perimeter: tuple | None = None) -> dict:
        """The fire's row of catalog/fire_extents.csv: time window, ignition point and region. The activity test
        uses the perimeters' box, not the region, so a neighbour burning in the margin does not stretch this fire's
        window."""
        p0, p1 = fire.perimeter_days()
        act = cls.active_days(firms_df, fire.box)
        a0, a1 = cls.activity_window(p0, p1, act)
        pad = pd.Timedelta(days=Fire.WINDOW_PAD_DAYS)
        lat, lon, t0, how = cls.ignition(fire, firms_df, act, first_perimeter)
        w, s, e, n = fire.region()
        return {
            "fire_id": fire.fire_id,
            "start_day": (a0 - pad).date().isoformat(),
            "end_day": (a1 + pad).date().isoformat(),
            "first_perimeter_day": p0.date().isoformat(),
            "last_perimeter_day": p1.date().isoformat(),
            "active_days_before": (p0 - a0).days,
            "active_days_after": (a1 - p1).days,
            "active_days": len(act),
            "window_rule": "perimeters+firms" if firms_df is not None else "perimeters",
            "ignition_lat": round(lat, 5),
            "ignition_lon": round(lon, 5),
            "ignition_time_utc": t0.strftime("%Y-%m-%dT%H:%MZ"),
            "ignition_method": how,
            "region_west": round(w, 5),
            "region_south": round(s, 5),
            "region_east": round(e, 5),
            "region_north": round(n, 5),
        }
