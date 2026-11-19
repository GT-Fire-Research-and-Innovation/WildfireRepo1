"""Per-channel progress of the target repo, from its manifests."""

from __future__ import annotations

import logging

import pandas as pd

from fah.channels.registry import ChannelRegistry
from fah.commands.settings import RunSettings
from fah.core.errors import ExitCode
from fah.hub.manifest import ManifestBook
from fah.hub.repo import HubRepo

log = logging.getLogger(__name__)


class StatusReport:
    """Fires per channel and status (latest row per fire), and GB stored."""

    def __init__(self, settings: RunSettings) -> None:
        self.s = settings

    def execute(self) -> int:
        m = ManifestBook.read(HubRepo.connect(self.s.hf_repo))
        if not len(m):
            log.info("no manifests yet")
            return ExitCode.OK
        m["bytes"] = pd.to_numeric(m["bytes"], errors="coerce").fillna(0)
        last = ManifestBook.latest(m, by=("channel", "fire_id"))
        tab = last.pivot_table(index="channel", columns="status", values="fire_id", aggfunc="count", fill_value=0)
        tab["GB"] = last.groupby("channel")["bytes"].sum() / 1e9
        log.info(tab.reindex([c for c in ChannelRegistry().names() if c in tab.index]).round(2).to_string())
        return ExitCode.OK
