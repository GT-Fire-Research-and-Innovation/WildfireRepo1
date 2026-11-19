"""HTTP building blocks shared by the clients: retried GETs, checked S3 objects, streamed downloads, a bounded
parallel map, and a size-capped cache of whole files."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor
from pathlib import Path
from typing import TypeVar

import requests

log = logging.getLogger(__name__)
K = TypeVar("K")
R = TypeVar("R")


class HttpClient:
    """GETs on a shared session, retried on network errors, 429 and 5xx."""

    def __init__(self, session: requests.Session) -> None:
        self.session = session

    def get_bytes(self, url: str, tries: int = 5, timeout: float = 300) -> bytes:
        """Body of a GET; other HTTP errors raise with the reply text."""
        for t in range(tries):
            try:
                r = self.session.get(url, timeout=timeout)
                if r.status_code == 200:
                    return r.content
                if r.status_code != 429 and r.status_code < 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                err = f"HTTP {r.status_code}"
            except requests.RequestException as ex:
                err = f"{type(ex).__name__}: {str(ex)[:120]}"
            if t == tries - 1:
                raise RuntimeError(f"{err} after {tries} tries")
            time.sleep(5 * 2**t)
        raise AssertionError("unreachable")

    def get_json(self, url: str, params: dict | None = None, tries: int = 5) -> object:
        """JSON of a GET, retried on any error."""
        for i in range(tries):
            try:
                r = self.session.get(url, params=params, timeout=120)
                r.raise_for_status()
                return r.json()
            except Exception as ex:  # noqa: BLE001  (retried, then raised)
                if i == tries - 1:
                    raise
                wait = 2 ** (i + 1)
                log.info(f"    retry {i + 1} after error: {ex} (waiting {wait}s)")
                time.sleep(wait)
        raise AssertionError("unreachable")

    def s3_object(self, bucket: str, key: str) -> bytes:
        """One object of a public S3 bucket, checked against its ETag (the MD5 of a single-part upload): bytes cut
        short or garbled on the way raise here, so the caller downloads again instead of failing later in a reader."""
        r = self.session.get(f"https://{bucket}.s3.amazonaws.com/{key}", timeout=120)
        r.raise_for_status()
        data = r.content
        etag = r.headers.get("ETag", "").strip('"')
        if re.fullmatch(r"[0-9a-f]{32}", etag) and hashlib.md5(data).hexdigest() != etag:
            raise OSError(f"{key.rsplit('/', 1)[-1]}: {len(data)} bytes do not match the object's MD5 (ETag)")
        return data

    def download(self, url: str, dst: Path, timeout: float = 1800, chunk: int = 1 << 22) -> Path:
        """Stream a whole file to dst (via dst.part, so a broken download never looks finished)."""
        part = dst.with_name(dst.name + ".part")
        with self.session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(part, "wb") as fh:
                for b in r.iter_content(chunk):
                    fh.write(b)
        part.replace(dst)
        return dst


class ParallelMap:
    """Executor maps that keep memory bounded."""

    @staticmethod
    def windowed(executor: Executor, fn: Callable[[K], R], keys: Iterable[K], ahead: int) -> Iterator[R]:
        """Like executor.map (results in order), but never more than `ahead` results waiting, so one slow fetch
        does not pile up the others in memory."""
        q, it = deque(), iter(keys)
        for k in it:
            q.append(executor.submit(fn, k))
            if len(q) >= ahead:
                break
        while q:
            yield q.popleft().result()
            for k in it:
                q.append(executor.submit(fn, k))
                break


class DownloadCache:
    """Whole remote files kept for this run and shared by every fire that needs them; the least recently used
    go when the folder passes max_gb."""

    def __init__(self, folder: Path, max_gb: float = 3.0) -> None:
        self.folder, self.max_gb = Path(folder), max_gb

    def get(self, url: str, fetch: Callable[[Path], object]) -> Path:
        """The local copy of url; fetch(path) writes it when missing."""
        p = self.folder / Path(url.split("?")[0]).name
        if not p.exists():
            fetch(p)
            files = sorted((q for q in self.folder.iterdir() if q.is_file()), key=lambda q: q.stat().st_atime)
            total = sum(q.stat().st_size for q in files)
            for q in files:
                if total <= self.max_gb * 1e9 or q == p:
                    break
                total -= q.stat().st_size
                q.unlink(missing_ok=True)
        os.utime(p)
        return p
