"""Check the token can write to the target repo, and which keys are set."""

from __future__ import annotations

import logging
import os

from fah.commands.settings import RunSettings
from fah.core.errors import ExitCode
from fah.hub.repo import HubRepo

log = logging.getLogger(__name__)


class SetupCheck:
    """Raises SetupError (exit 3) when the token cannot write; lists the channel keys (a missing one skips its
    channels in the notebook)."""

    KEYS = ("FIRMS_MAP_KEY", "EARTHDATA_TOKEN", "GEE_SERVICE_ACCOUNT_KEY", "GEE_PROJECT")

    def __init__(self, settings: RunSettings) -> None:
        self.s = settings

    def execute(self) -> int:
        who = HubRepo.connect(self.s.hf_repo).check_write()
        keys = {k: ("set" if os.environ.get(k) else "missing (channel skipped)") for k in self.KEYS}
        log.info(f"OK: HF_TOKEN can write as '{who}' to {self.s.hf_repo}; {keys}")
        return ExitCode.OK
