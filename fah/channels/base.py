"""What every channel is: its facts (time-dependent, cut to the region, files it must write) and how it fetches.

Per-fire channels fetch each fire on its own (fetch_one). Shared-download channels read one upstream file for
many fires (a GOES scan, a HRRR file, an OSM snapshot) and so fetch a whole list of fires at once."""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import ClassVar

from fah.core.context import RunContext
from fah.core.errors import ChannelUnavailable
from fah.core.fire import Fire
from fah.core.result import FetchResult, Status

log = logging.getLogger(__name__)


class Channel(ABC):
    """A channel of the raw repo. A manifest row records the window and region a fire was fetched for
    (window_label / region_label); when either changes, the fire is fetched again."""

    name: ClassVar[str]
    time_dependent: ClassVar[bool] = False  # files depend on the fire's time window
    region_cut: ClassVar[bool] = True  # cut to Fire.region() (False: the perimeters' box, or not spatial)
    required_files: ClassVar[frozenset[str] | None] = None

    @property
    def expected_files(self) -> frozenset[str] | None:
        """Files an 'ok' row must list to count as done (None: any 'ok' row counts). A channel that gains files
        fetches again just the fires that miss them."""
        return self.required_files

    def skip_reason(self, fire: Fire) -> str | None:
        """None when the source covers this fire, else why not (recorded as status n/a)."""
        return None

    def check_ready(self) -> None:  # noqa: B027  (optional hook: most channels need nothing)
        """Raise ChannelUnavailable when a key, tool or package this channel needs is missing."""

    def shared_files(self, ctx: RunContext) -> list[tuple[Path, str]]:
        """[(local file, repo path)] uploaded once per repo, with the first batch."""
        return []

    def window_label(self, fire: Fire) -> str:
        """The days this channel's files cover for the fire ('' when they do not depend on time)."""
        return fire.span().label() if self.time_dependent else ""

    def region_label(self, fire: Fire) -> str:
        """The box this channel's files cover ('' for channels on the perimeters' box)."""
        return fire.region().label() if self.region_cut else ""

    def labels(self, fire: Fire) -> tuple[str, str]:
        return self.window_label(fire), self.region_label(fire)

    @abstractmethod
    def fetch(self, fires: list[Fire], ctx: RunContext) -> Iterator[FetchResult]:
        """One result per fire, each fire's files in ctx.stage_path(fire_id) when it is yielded."""

    @staticmethod
    def describe(ex: BaseException) -> str:
        """'Type: message' of an error, then of the errors it came from: rasterio's 'Read failed. See previous
        exception for details.' says nothing without GDAL's own error behind it."""
        parts: list[str] = []
        while ex is not None and len(parts) < 4:
            parts.append(f"{type(ex).__name__}: {ex}")
            ex = ex.__cause__ or (None if ex.__suppress_context__ else ex.__context__)
        return " <- ".join(parts)

    @staticmethod
    def require_module(module: str, message: str) -> None:
        """ChannelUnavailable(message) when `module` cannot be imported."""
        try:
            __import__(module)
        except ImportError as ex:
            raise ChannelUnavailable(message) from ex


class PerFireChannel(Channel):
    """Fetches the fires one by one; an exception is that fire's error, not the run's."""

    def fetch(self, fires: list[Fire], ctx: RunContext) -> Iterator[FetchResult]:
        for fire in fires:
            if ctx.expired():
                log.info("deadline reached: stopping; rerun to continue")
                return
            out = ctx.stage(fire.fire_id)
            try:
                status, sources = self.fetch_one(fire, out, ctx)
            except Exception as ex:  # noqa: BLE001  (recorded for this fire; the run goes on)
                yield FetchResult(fire, Status.ERROR, (), self.describe(ex), traceback.format_exc())
            else:
                yield FetchResult(fire, status, tuple(sources))

    @abstractmethod
    def fetch_one(self, fire: Fire, out: Path, ctx: RunContext) -> tuple[str, list[str]]:
        """Write the fire's files into `out`; -> (status 'ok' or 'empty', upstream URLs)."""


class SharedDownloadChannel(Channel):
    """Fetches groups of fires together, so each upstream file is read once for every fire on it."""

    @staticmethod
    def groups(fires: Iterable[Fire], size: int) -> Iterator[list[Fire]]:
        """The fires in time order, `size` at a time (fires close in time share upstream files)."""
        ordered = sorted(fires, key=lambda f: f.span().start)
        for i in range(0, len(ordered), size):
            yield ordered[i : i + size]
