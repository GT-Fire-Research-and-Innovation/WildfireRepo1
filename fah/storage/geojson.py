"""GeoJSON as <name>.geojson.gz with a fixed gzip header: the same features always give the same bytes, so a
rerun that finds nothing new uploads nothing new. Read with json.load(gzip.open(p)) or
geopandas.read_file('/vsigzip/' + p)."""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path


class GzipGeoJson:
    """Deterministic gzip streams and FeatureCollections."""

    LEVEL = 6

    @classmethod
    @contextlib.contextmanager
    def writer(cls, path: Path) -> Iterator[gzip.GzipFile]:
        """A gzip stream with no file name or time in its header."""
        with (
            open(path, "wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=cls.LEVEL, mtime=0) as gz,
        ):
            yield gz

    @classmethod
    def write(cls, path: Path, features: Sequence[dict]) -> None:
        """A FeatureCollection of `features`."""
        with cls.writer(path) as gz, io.TextIOWrapper(gz, encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection", "features": list(features)}, fh, default=str)

    @classmethod
    def compress(cls, src: Path, dst: Path) -> None:
        """dst = gzip of src, streamed, with the fixed header."""
        with open(src, "rb") as fi, cls.writer(dst) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
