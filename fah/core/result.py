"""What fetching one fire in one channel produced."""

from __future__ import annotations

from dataclasses import dataclass

from fah.core.fire import Fire


class Status:
    """Status of a fire in a channel, as manifests record it."""

    OK = "ok"  # files written
    EMPTY = "empty"  # the source has nothing for this fire
    NOT_APPLICABLE = "n/a"  # the source does not cover this fire (years, region)
    ERROR = "error"  # retried on the next run
    PENDING = "pending"  # not run yet (links only)
    FINAL = frozenset({OK, EMPTY, NOT_APPLICABLE})
    STORED = frozenset({OK, EMPTY})


@dataclass(frozen=True, slots=True)
class FetchResult:
    """One fire's outcome: its status, the upstream URLs its files came from, and an error or note (published,
    so it is redacted before it reaches a manifest). `trace` is the traceback of an error, for --verbose."""

    fire: Fire
    status: str
    sources: tuple[str, ...] = ()
    error: str = ""
    trace: str = ""
