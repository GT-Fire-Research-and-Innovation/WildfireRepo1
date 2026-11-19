"""UTC timestamps: every instant in the grab is a timezone-aware pandas Timestamp in UTC."""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd


class Utc:
    """Conversions to UTC pandas Timestamps."""

    @staticmethod
    def stamp(value: object) -> pd.Timestamp:
        """Anything pandas reads as a time, in UTC; a value without a time zone is taken as UTC."""
        t = pd.Timestamp(value)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    @staticmethod
    def iso_seconds(value: object) -> str:
        """'2018-11-08T00:00:00Z'."""
        return Utc.stamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


class TimeSpan(NamedTuple):
    """A closed time interval, UTC."""

    start: pd.Timestamp
    end: pd.Timestamp

    def label(self) -> str:
        """'2018-11-08/2018-11-26': the days the span covers, as a manifest records them."""
        return f"{self.start.date().isoformat()}/{self.end.date().isoformat()}"

    def hours(self, step_h: int) -> list[pd.Timestamp]:
        """Every step_h hours from the start to the end."""
        return list(pd.date_range(self.start, self.end, freq=f"{step_h}h"))

    def daily(self, hour: int) -> list[pd.Timestamp]:
        """That hour (UTC) of every day the span touches."""
        return [d + pd.Timedelta(hours=hour) for d in pd.date_range(self.start.floor("D"), self.end.floor("D"))]
