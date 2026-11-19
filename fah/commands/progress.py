"""Which fires a channel has finished, by its rules (expected files, window, region)."""

from __future__ import annotations

import pandas as pd

from fah.channels.base import Channel
from fah.core.fire import Fire
from fah.hub.manifest import ManifestBook


class ChannelProgress:
    """Joins a channel's rules to the manifest: the hub knows rows, the channel knows labels and files."""

    @staticmethod
    def done_ids(
        manifest: pd.DataFrame | None, channel: Channel | None = None, fires: list[Fire] | None = None
    ) -> set[str]:
        """Fires finished in the channel. Given the fires, a row must also match each fire's current window (channels
        that depend on time) and region (channels cut to the region)."""
        expected = channel.expected_files if channel else None
        window = region = None
        if channel is not None and fires is not None:
            if channel.time_dependent:
                window = {f.fire_id: channel.window_label(f) for f in fires}
            if channel.region_cut:
                region = {f.fire_id: channel.region_label(f) for f in fires}
        return ManifestBook.done_ids(manifest, expected, window, region)
