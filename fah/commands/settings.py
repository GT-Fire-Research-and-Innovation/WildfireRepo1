"""The options of one command line run."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunSettings:
    """Every command line option (defaults as the command line gives them)."""

    channel: str | None = None
    hf_repo: str | None = None
    work: Path = Path("raw_work")
    archive: str = "all"  # geomac | nifs | both | all: which fires of the catalog a run takes
    catalog_file: str | None = None  # a local fires.csv instead of the repo's catalog/fires.csv
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
    partial_windows: bool = False  # time-dependent channels: go ahead while some windows are not final
    local_manifests: str | None = None
    catalog: str | None = None  # scan | build | all
    first_year: int = 2014
    last_year: int = 2025
    min_acres: float = 1000.0
    min_days: int = 3

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunSettings:
        values = {f.name: getattr(args, f.name) for f in dataclasses.fields(cls) if hasattr(args, f.name)}
        values["work"] = Path(values.get("work") or "raw_work")
        if values.get("ids") is not None:
            values["ids"] = tuple(values["ids"])
        years = getattr(args, "years", None)
        if years:
            first, _, last = str(years).partition("-")
            values["first_year"], values["last_year"] = int(first), int(last or first)
        return cls(**values)

    def replace(self, **changes: object) -> RunSettings:
        return dataclasses.replace(self, **changes)
