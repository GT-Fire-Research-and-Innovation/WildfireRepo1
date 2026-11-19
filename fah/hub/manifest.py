"""manifest/<channel>/<batch>.csv: one row per fire and fetch, the record runs resume from."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from fah.core.result import Status
from fah.hub.repo import HubRepo


class ManifestBook:
    """Reading manifests and deciding which fires a channel has finished."""

    COLUMNS = [
        "fire_id",
        "name",
        "year",
        "channel",
        "status",
        "n_files",
        "bytes",
        "files",
        "sources",
        "fetched_utc",
        "error",
        "window",
        "region",
    ]  # window / region: the days and box the channel covered
    SEP = " | "  # between source URLs in a cell (URLs may hold ';' and ',')

    @classmethod
    def empty(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls.COLUMNS)

    @classmethod
    def read(cls, hub: HubRepo, channel: str | None = None) -> pd.DataFrame:
        """Every manifest row of the repo (one channel, or all)."""
        files = [p for p in hub.files(f"manifest/{channel}" if channel else "manifest") if p.endswith(".csv")]
        frames = [pd.read_csv(hub.download(f), dtype=str, keep_default_na=False) for f in files]
        return pd.concat(frames, ignore_index=True) if frames else cls.empty()

    @classmethod
    def read_local(cls, folder: Path, channel: str | None = None) -> pd.DataFrame:
        """Every manifest row of the --dry-run batches under folder (one channel, or all)."""
        files = sorted(Path(folder).glob(f"**/manifest/{channel or '*'}/*.csv"))
        frames = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]
        return pd.concat(frames, ignore_index=True) if frames else cls.empty()

    @staticmethod
    def latest(manifest: pd.DataFrame, by: tuple[str, ...] = ("fire_id",)) -> pd.DataFrame:
        """The newest row per fire (or per `by`)."""
        return manifest.sort_values("fetched_utc").groupby(list(by)).tail(1)

    @classmethod
    def done_ids(
        cls,
        manifest: pd.DataFrame | None,
        expected: frozenset[str] | None = None,
        window: Mapping[str, str] | None = None,
        region: Mapping[str, str] | None = None,
    ) -> set[str]:
        """Fires finished: the latest row has a final status (errors are retried). With `expected`, an 'ok' row must
        name all of those files. With `window` / `region` ({fire_id: label}), the row must have been fetched for the
        fire's current window / region: rows from before these rules, or for one since widened, are not done."""
        if manifest is None or not len(manifest):
            return set()
        last = cls.latest(manifest)
        done = last.status.isin(Status.FINAL)
        if expected:
            col = last["files"] if "files" in last else pd.Series("", index=last.index)
            listed = col.fillna("").astype(str).map(lambda s: set(s.split(";")))
            done &= (last.status != Status.OK) | listed.map(expected.issubset)
        for col, want in (("window", window), ("region", region)):
            if want is not None:
                have = last[col].fillna("").astype(str) if col in last else pd.Series("", index=last.index)
                done &= have == last["fire_id"].map(want)
        return set(last.loc[done, "fire_id"])
