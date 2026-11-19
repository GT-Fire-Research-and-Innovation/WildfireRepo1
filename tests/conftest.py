"""Shared fixtures: the Camp Fire as the catalog describes it, with and without its perimeters' own box."""

from __future__ import annotations

import pytest

from fah.channels.registry import ChannelRegistry
from fah.core.fire import Fire
from fah.core.geometry import BoundingBox


@pytest.fixture
def camp() -> Fire:
    """Camp Fire 2018 as the catalog and grid.json describe it (build box only)."""
    return Fire(
        fire_id="F2003403265",
        incident="75E64DB8-9B75-4A68-BEDD-67CC62658E38",
        name="CAMP",
        year=2018,
        first="2018-11-09 00:00:00.000+00:00",
        last="2018-11-25 18:22:00.000+00:00",
        build_box=BoundingBox(-121.88999101802999, 39.51118309776612, -121.23979602288742, 39.98463882716831),
        rating="keep",
    )


@pytest.fixture
def camp_perimeters(camp: Fire) -> Fire:
    """The Camp Fire with the box around all its perimeters (catalog west / south / east / north)."""
    return camp.replace(perimeter_box=BoundingBox(-121.777828, 39.598585, -121.352605, 39.897816))


@pytest.fixture
def channels() -> ChannelRegistry:
    return ChannelRegistry()
