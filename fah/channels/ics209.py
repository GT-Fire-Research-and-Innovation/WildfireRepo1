"""ics209: the ICS-209-PLUS incident and situation-report rows of the incidents matched to the fire."""

from __future__ import annotations

from pathlib import Path

from fah.channels.base import PerFireChannel
from fah.core.clock import Utc
from fah.core.context import RunContext
from fah.core.fire import Fire
from fah.sources.ics209 import Ics209Archive


class Ics209Channel(PerFireChannel):
    """incidents.csv, sitreps.csv: the original rows, every column as text. Incidents are matched on the build box
    (a wider box would match neighbours). The source zip goes to shared/ once per repo."""

    name = "ics209"
    region_cut = False
    LAST_YEAR = 2020
    SHARED_PATH = "shared/ics209/ics209plus-wildfire.zip"

    def skip_reason(self, fire: Fire) -> str | None:
        if Utc.stamp(fire.first).year > self.LAST_YEAR:
            return "ICS-209-PLUS covers 1999-2020"
        return None

    def shared_files(self, ctx: RunContext) -> list[tuple[Path, str]]:
        archive = Ics209Archive.for_run(ctx)
        archive.tables()
        return [(archive.zip_path, self.SHARED_PATH)]

    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        archive = Ics209Archive.for_run(ctx)
        sit, inc, incidents, (sit_name, inc_name) = archive.tables()
        irwin, _ = fire.incident_key()
        ids = archive.match(
            incidents,
            {irwin} if irwin else set(),
            {str(fire.name).lower()},
            fire.build_box,
            Utc.stamp(fire.first).tz_convert(None),
            Utc.stamp(fire.last).tz_convert(None),
        )
        ids = {str(i) for i in ids}
        inc[inc["INCIDENT_ID"].astype(str).isin(ids)].to_csv(out / "incidents.csv", index=False)
        sit[sit["INCIDENT_ID"].astype(str).isin(ids)].to_csv(out / "sitreps.csv", index=False)
        return ("ok" if ids else "empty"), [
            f"{Ics209Archive.DOWNLOAD}#{inc_name}",
            f"{Ics209Archive.DOWNLOAD}#{sit_name}",
        ]
