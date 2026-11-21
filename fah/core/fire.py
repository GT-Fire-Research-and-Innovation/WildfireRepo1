"""A fire of the catalog: its identity, boxes and time window, and the region and days every channel derives
from them.

A fire often burns before its first mapped perimeter and on after its last. Its time window is therefore the
perimeter days stretched over the FIRMS activity that chains onto them (catalog/fire_extents.csv, attached with
`with_extent`), plus one day on each side; without that table, the perimeter days plus one day on each side.
FIRMS itself is fetched over a fixed span around the perimeter days, so the window never depends on itself."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import pandas as pd

from fah.core.clock import TimeSpan, Utc
from fah.core.geometry import BoundingBox


class Ignition(NamedTuple):
    """The fire's earliest observation: where a tile can be centred without seeing the future."""

    lat: float
    lon: float
    time_utc: object  # a Timestamp when estimated, the table's text when read back
    method: str


class FireRef(NamedTuple):
    """What a manifest row records about a fire."""

    fire_id: str
    name: str
    year: object


@dataclass(frozen=True, slots=True)
class Fire:
    """One fire of the catalog (catalog/fires.csv). `box` is the exact extent of its main run of perimeter records;
    `first_record` is (time, lat, lon) of the earliest of them and `origin` (time, lat, lon) the incident's reported
    discovery time and location (WFIGS), when known."""

    REGION_BUFFER_KM: ClassVar[float] = 32.0  # half a 64 km tile: the region reaches this far past the perimeters
    DETECTION_PAD_DEG: ClassVar[float] = 0.01  # ~1 km around the perimeters' box: a 375 m pixel centre can sit outside
    WINDOW_PAD_DAYS: ClassVar[int] = 1  # added to both ends of the window
    FIRMS_BEFORE_DAYS: ClassVar[int] = 14  # FIRMS days fetched around the perimeter days
    FIRMS_AFTER_DAYS: ClassVar[int] = 14

    fire_id: str
    incident: str
    name: str
    year: int
    first: str
    last: str
    box: BoundingBox
    archive: str = ""
    first_record: tuple | None = None
    origin: tuple | None = None
    window_days: tuple[str, str] | None = None
    ignition: Ignition | None = None

    # ---------------------------------------------------------------------------------------- identity

    def ref(self) -> FireRef:
        return FireRef(self.fire_id, self.name, self.year)

    @staticmethod
    def normalise_irwin(value: object) -> str:
        """IRWIN id without braces or whitespace, upper case. The catalog hashed 14 keys with trailing junk
        ('}\\r\\n'); matching on this form finds their records anyway."""
        return re.sub(r"[{}\s]", "", str(value or "")).upper()

    def irwin(self) -> str:
        return self.normalise_irwin(self.incident)

    def incident_key(self) -> tuple[str | None, tuple[str, str] | None]:
        """(irwin, None) for IRWIN-keyed incidents, (None, (name, uid)) for 'name:<name>|<uid>' keys."""
        if self.incident.startswith("name:"):
            name, _, uid = self.incident[5:].partition("|")
            return None, (name.strip(), uid.strip())
        return self.irwin(), None

    def replace(self, **changes: object) -> Fire:
        return dataclasses.replace(self, **changes)

    def with_extent(self, row: Mapping[str, object] | None) -> Fire:
        """This fire with the window and ignition point of its catalog/fire_extents.csv row (unchanged without)."""
        if row is None:
            return self
        ignition = self.ignition
        if str(row.get("ignition_lat", "")) not in ("", "nan"):
            ignition = Ignition(
                float(row["ignition_lat"]),
                float(row["ignition_lon"]),
                str(row["ignition_time_utc"]),
                str(row["ignition_method"]),
            )
        return self.replace(window_days=(str(row["start_day"]), str(row["end_day"])), ignition=ignition)

    # ---------------------------------------------------------------------------------------- space

    def region(self) -> BoundingBox:
        """The box every spatial channel is cut to: the perimeters' box, grown by DETECTION_PAD_DEG (so an ignition
        point found from FIRMS lies inside it) and then by REGION_BUFFER_KM on every side. A tile of
        2 x REGION_BUFFER_KM centred anywhere in the fire lies inside it. Depends on the perimeters only, so it is known
        before any channel runs."""
        return self.box.padded(self.DETECTION_PAD_DEG).buffered(self.REGION_BUFFER_KM)

    # ---------------------------------------------------------------------------------------- time

    def perimeter_days(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(first, last) perimeter day as naive UTC midnights."""
        return Utc.stamp(self.first).tz_convert(None).normalize(), Utc.stamp(self.last).tz_convert(None).normalize()

    def span(self) -> TimeSpan:
        """The fire's time window, UTC: 00:00 of its first day to 23:00 of its last."""
        if self.window_days:
            d0, d1 = pd.Timestamp(self.window_days[0]), pd.Timestamp(self.window_days[1])
        else:
            d0, d1 = self.perimeter_days()
            pad = pd.Timedelta(days=self.WINDOW_PAD_DAYS)
            d0, d1 = d0 - pad, d1 + pad
        return TimeSpan(d0.tz_localize("UTC"), d1.tz_localize("UTC") + pd.Timedelta(hours=23))

    def days(self, before: int = 14, after: int = 2) -> pd.DatetimeIndex:
        """Calendar days of the daily channels: `before` days before the first perimeter day to `after` days after
        the last (14 / 2, as the built dataset's), moved out as far as the window is."""
        t0, t1 = self.span()
        d0 = t0.tz_convert(None).normalize() - pd.Timedelta(days=before - self.WINDOW_PAD_DAYS)
        d1 = t1.tz_convert(None).normalize() + pd.Timedelta(days=after - self.WINDOW_PAD_DAYS)
        return pd.date_range(d0, d1)

    def firms_days(self) -> pd.DatetimeIndex:
        """Days FIRMS is fetched for: fixed by the perimeters, since the window is computed from what FIRMS finds."""
        d0, d1 = self.perimeter_days()
        return pd.date_range(
            d0 - pd.Timedelta(days=self.FIRMS_BEFORE_DAYS), d1 + pd.Timedelta(days=self.FIRMS_AFTER_DAYS)
        )

    def years(self) -> list[int]:
        """Calendar years from the first perimeter to the end of the window."""
        return list(range(Utc.stamp(self.first).year, self.span().end.year + 1))
