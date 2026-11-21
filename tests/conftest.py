"""Shared fixtures: the Camp Fire as catalog/fires.csv describes it."""

from __future__ import annotations

import pytest

from fah.channels.registry import ChannelRegistry
from fah.core.fire import Fire
from fah.core.geometry import BoundingBox


@pytest.fixture
def camp() -> Fire:
    """Camp Fire 2018: its catalog row (box = exact extent of its perimeter records)."""
    return Fire(
        fire_id="F2003403265",
        incident="75E64DB8-9B75-4A68-BEDD-67CC62658E38",
        name="CAMP",
        year=2018,
        first="2018-11-09 00:00:00.000+00:00",
        last="2018-11-25 18:22:00.000+00:00",
        box=BoundingBox(-121.777828, 39.598585, -121.352605, 39.897816),
        archive="both",
    )


@pytest.fixture
def camp_perimeters(camp: Fire) -> Fire:
    """The same fire (older tests asked for the perimeters' box by this name)."""
    return camp


@pytest.fixture
def channels() -> ChannelRegistry:
    return ChannelRegistry()
