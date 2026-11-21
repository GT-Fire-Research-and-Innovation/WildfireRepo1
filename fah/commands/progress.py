"""Which fires a channel has finished, by its rules (expected files, window, region)."""

from __future__ import annotations

import pandas as pd

from fah.channels.base import Channel
from fah.core.fire import Fire
from fah.core.result import Status
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

    @classmethod
    def gaps(cls, manifest: pd.DataFrame | None, channel: Channel, fires: list[Fire]) -> list[tuple[Fire, str, str]]:
        """The fires the channel missed, as (fire, status, note): first those not done (status of their latest row:
        'error'; 'outdated' when it was fetched for another window, region or file set; 'not fetched' without a
        row), then those done with nothing found ('empty'). Fires the source does not cover (n/a) are not listed."""
        done = cls.done_ids(manifest, channel, fires)
        last = ManifestBook.latest(manifest).set_index("fire_id") if manifest is not None and len(manifest) else None
        missing, empty = [], []
        for f in fires:
            if last is None or f.fire_id not in last.index:
                missing.append((f, "not fetched", ""))
                continue
            row = last.loc[f.fire_id]
            status, note = str(row["status"]), str(row.get("error", "") or "")
            if f.fire_id not in done:
                missing.append((f, "outdated" if status in Status.FINAL else status, note))
            elif status == Status.EMPTY:
                empty.append((f, status, note))
        return missing + empty
