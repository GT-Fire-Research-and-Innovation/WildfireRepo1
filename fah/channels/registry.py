"""Every channel, in the order a full run takes them (cheap to heavy)."""

from __future__ import annotations

from collections.abc import Iterator

from fah.channels.base import Channel
from fah.channels.firms import FirmsChannel
from fah.channels.gfs import GfsChannel
from fah.channels.goes import GoesChannel
from fah.channels.gridmet import GridmetChannel
from fah.channels.history import HistoryChannel
from fah.channels.hrrr import HrrrChannel
from fah.channels.hrrr_next import HrrrNextChannel
from fah.channels.ics209 import Ics209Channel
from fah.channels.landcover import LandcoverChannel
from fah.channels.landfire import LandfireChannel
from fah.channels.lulc import LulcChannel
from fah.channels.osm import OsmChannel
from fah.channels.perimeters import PerimetersChannel
from fah.channels.suppression import SuppressionChannel
from fah.channels.terrain import TerrainChannel
from fah.channels.tessera import TesseraChannel
from fah.channels.vegetation import VegetationChannel
from fah.channels.viirs import ViirsChannel


class ChannelRegistry:
    """Channels by name, in run order."""

    ORDER = (
        HistoryChannel,
        PerimetersChannel,
        FirmsChannel,
        GridmetChannel,
        Ics209Channel,
        SuppressionChannel,
        TesseraChannel,
        TerrainChannel,
        LandfireChannel,
        LulcChannel,
        LandcoverChannel,
        VegetationChannel,
        GoesChannel,
        HrrrChannel,
        HrrrNextChannel,
        GfsChannel,
        OsmChannel,
        ViirsChannel,
    )

    def __init__(self) -> None:
        self._channels = {cls.name: cls() for cls in self.ORDER}

    def names(self) -> list[str]:
        return list(self._channels)

    def get(self, name: str) -> Channel:
        return self._channels[name]

    def __iter__(self) -> Iterator[Channel]:
        return iter(self._channels.values())
