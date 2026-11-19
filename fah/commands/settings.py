"""The options of one command line run."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class RunSettings:
    """Every command line option (defaults as the command line gives them)."""

    SOURCE_REPO: ClassVar[str] = "Javyn123/wildfire-spread-2014-2025"  # the built dataset

    channel: str | None = None
    hf_repo: str | None = None
    source_repo: str = SOURCE_REPO
    work: Path = Path("raw_work")
    fires: str = "built"
    ids: tuple[str, ...] | None = None
    limit: int | None = None
    batch_gb: float = 2.0
    batch_fires: int = 100
    min_free_gb: float = 5.0
    max_hours: float | None = None
    threads: int = 16
    private: bool = False
    redo: bool = False
    dry_run: bool = False
    keep: bool = False
    keep_cache: bool = False
    max_errors_in_row: int = 10
    verbose: bool = False
    links_chunk: int = 500
    extents_file: str | None = None
    local_manifests: str | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunSettings:
        values = {f.name: getattr(args, f.name) for f in dataclasses.fields(cls) if hasattr(args, f.name)}
        values["work"] = Path(values.get("work") or "raw_work")
        if values.get("ids") is not None:
            values["ids"] = tuple(values["ids"])
        return cls(**values)

    def replace(self, **changes: object) -> RunSettings:
        return dataclasses.replace(self, **changes)
