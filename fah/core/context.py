"""State of one channel run: work folders, the HTTP session, shared objects, the deadline."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import requests

T = TypeVar("T")


class RunContext:
    """Work folders of a run (`cache` for one-time downloads, `stage_root/<fire>/<channel>` for each fire's
    files), the shared HTTP session, objects shared by every fire (`memo`), and the deadline."""

    USER_AGENT = "wildfire-raw-grab/1.0"

    def __init__(
        self,
        channel: str,
        work: Path,
        deadline: float | None = None,
        reserve_s: float = 900,
        threads: int = 16,
        session: requests.Session | None = None,
    ) -> None:
        self.channel, self.work = channel, Path(work)
        self.cache = self.work / "cache" / channel
        self.stage_root = self.work / "stage"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.deadline, self.reserve_s, self.threads = deadline, reserve_s, threads
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = self.USER_AGENT
        self.session = session
        self._memo: dict[object, object] = {}
        self._last_call = 0.0

    def stage_path(self, fire_id: str) -> Path:
        """Where a fire's files for this channel are written (not created)."""
        return self.stage_root / fire_id / self.channel

    def stage(self, fire_id: str) -> Path:
        """stage_path, created."""
        p = self.stage_path(fire_id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def drop_stage(self, fire_id: str) -> None:
        shutil.rmtree(self.stage_root / fire_id, ignore_errors=True)

    def tmp(self) -> Path:
        """Scratch folder for query results, emptied by the channels that use it."""
        p = self.cache / "q"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def expired(self) -> bool:
        """True when the deadline (less the reserve kept for the last upload) has passed."""
        return self.deadline is not None and time.time() > self.deadline - self.reserve_s

    def memo(self, key: object, make: Callable[[], T]) -> T:
        """The object stored under key, made on first use (clients and tables shared by every fire)."""
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    def throttle(self, min_gap_s: float) -> None:
        """Wait so calls are at least min_gap_s apart."""
        dt = min_gap_s - (time.time() - self._last_call)
        if dt > 0:
            time.sleep(dt)
        self._last_call = time.time()

    def cleanup(self) -> None:
        shutil.rmtree(self.cache, ignore_errors=True)
        shutil.rmtree(self.stage_root, ignore_errors=True)
