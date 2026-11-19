"""NASA Earthdata: granule search in the Common Metadata Repository, the Earthdata Login token, byte-range reads of
remote HDF5 / netCDF4 files and whole-file downloads.

Earthdata Cloud answers a file request with a redirect to a signed S3 URL (valid about an hour). The token goes
only to NASA hosts, never to the S3 URL, and never into a manifest (Secrets.redact). Free token:
https://urs.earthdata.nasa.gov > Generate Token (valid 60 days)."""

from __future__ import annotations

import io
import re
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from fah.core.clock import Utc
from fah.core.errors import ChannelUnavailable
from fah.core.geometry import BoundingBox
from fah.core.secrets import Secrets


class EarthdataClient:
    """CMR search (no login needed) and authenticated reads (LAADS DAAC, LP DAAC)."""

    CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
    TOKEN_ENV = "EARTHDATA_TOKEN"
    TOKEN_HELP = (
        "Set EARTHDATA_TOKEN: a free Earthdata Login token from https://urs.earthdata.nasa.gov "
        "(Generate Token). On Kaggle add it under Add-ons > Secrets."
    )
    NASA_HOSTS = (".earthdata.nasa.gov", ".earthdatacloud.nasa.gov", ".eosdis.nasa.gov")

    def __init__(self, session: requests.Session) -> None:
        self.session = session

    # ------------------------------------------------------------------------------------- login

    @classmethod
    def token(cls, required: bool = True) -> str | None:
        """The Earthdata Login bearer token from the environment, .env or ~/.config/wildfire/env."""
        v = Secrets.lookup(cls.TOKEN_ENV)
        if not v and required:
            raise ChannelUnavailable(cls.TOKEN_HELP)
        return v

    @classmethod
    def is_nasa(cls, url: str) -> bool:
        return urlparse(url).netloc.endswith(cls.NASA_HOSTS)

    # ------------------------------------------------------------------------------------- granule names

    @staticmethod
    def granule_parts(url: str) -> tuple[str, str, str, str]:
        """('VNP02IMG', 'A2018314', '2048', '002') from a granule file name; tiles give ('VNP13A1', 'A2018305',
        'h08v05', '002'). The first three identify the observation, the fourth is the collection version."""
        p = Path(urlparse(url).path).name.split(".")
        return p[0], p[1], p[2], p[3]

    @classmethod
    def granule_day(cls, url: str) -> pd.Timestamp:
        """The A<YYYY><DDD> date of a granule (UTC midnight)."""
        a = cls.granule_parts(url)[1]
        return pd.Timestamp(f"{a[1:5]}-01-01", tz="UTC") + pd.Timedelta(days=int(a[5:8]) - 1)

    @classmethod
    def newest_versions(cls, granules: list[dict]) -> list[dict]:
        """One granule per observation (product, date, time or tile): the newest collection version."""
        best: dict[tuple, dict] = {}
        for g in granules:
            p = cls.granule_parts(g["url"])
            if p[:3] not in best or cls._version_key(p[3]) > cls._version_key(cls.granule_parts(best[p[:3]]["url"])[3]):
                best[p[:3]] = g
        return sorted(best.values(), key=lambda g: (str(g["start"]), g["name"]))

    @staticmethod
    def _version_key(v: str) -> tuple:
        return tuple(int(x) if x.isdigit() else x for x in re.findall(r"\d+|[A-Za-z]+", v))

    # ------------------------------------------------------------------------------------- search

    def granules(
        self, short_name: str, t0: object, t1: object, box: BoundingBox, page_size: int = 500, tries: int = 5
    ) -> list[dict]:
        """Granules of a collection overlapping [t0, t1] and the box, newest collection version only:
        [{'url', 'name', 'start', 'end', 'daynight'}] in time order."""
        params = {
            "short_name": short_name,
            "temporal": f"{Utc.iso_seconds(t0)},{Utc.iso_seconds(t1)}",
            "page_size": page_size,
            "bounding_box": ",".join(f"{v:.5f}" for v in box),
        }
        out, after = [], None
        while True:
            hdr = {"CMR-Search-After": after} if after else {}
            for attempt in range(tries):
                try:
                    r = self.session.get(self.CMR, params=params, headers=hdr, timeout=120)
                    if r.status_code == 200:
                        break
                    err = f"HTTP {r.status_code}: {r.text[:200]}"
                except Exception as ex:  # noqa: BLE001  (network)
                    err = f"{type(ex).__name__}: {str(ex)[:150]}"
                if attempt == tries - 1:
                    raise RuntimeError(f"CMR {short_name}: {err}")
                time.sleep(3 * 2**attempt)
            items = r.json().get("items", [])
            for it in items:
                u = it["umm"]
                urls = [
                    x["URL"]
                    for x in u.get("RelatedUrls", [])
                    if x.get("Type") == "GET DATA" and x["URL"].startswith("https://")
                ]
                if not urls:
                    continue
                te = (u.get("TemporalExtent") or {}).get("RangeDateTime") or {}
                out.append(
                    {
                        "url": urls[0],
                        "name": Path(urlparse(urls[0]).path).name,
                        "start": te.get("BeginningDateTime"),
                        "end": te.get("EndingDateTime"),
                        "daynight": (u.get("DataGranule") or {}).get("DayNightFlag", ""),
                    }
                )
            after = r.headers.get("CMR-Search-After")
            if not after or len(items) < page_size:
                break
        return self.newest_versions(out)

    # ------------------------------------------------------------------------------------- reads

    def open_remote(self, url: str, token: str | None = None) -> RangeFile:
        """A file-like object h5py can read, fetched by byte range."""
        return RangeFile(self.session, url, token)

    def download(self, url: str, dst: Path, token: str | None = None, tries: int = 5, timeout: float = 600) -> Path:
        """A whole file (for formats h5py cannot read, such as HDF4). The token goes to NASA hosts only: requests
        drops the Authorization header when a redirect leaves the host."""
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        hdr = {"Authorization": f"Bearer {token}"} if token and self.is_nasa(url) else {}
        for attempt in range(tries):
            try:
                with self.session.get(url, headers=hdr, stream=True, timeout=timeout) as r:
                    if "urs.earthdata.nasa.gov" in urlparse(r.url).netloc:
                        raise PermissionError(f"Earthdata Login refused the request for {url}. {self.TOKEN_HELP}")
                    if r.status_code in (401, 403):
                        raise PermissionError(f"HTTP {r.status_code} for {url}. {self.TOKEN_HELP}")
                    r.raise_for_status()
                    tmp = dst.with_suffix(dst.suffix + ".part")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 22):
                            fh.write(chunk)
                    tmp.replace(dst)
                    return dst
            except PermissionError:
                raise
            except (requests.RequestException, OSError) as ex:
                if attempt == tries - 1:
                    raise RuntimeError(f"{url}: {type(ex).__name__}: {str(ex)[:150]}") from ex
                time.sleep(5 * 2**attempt)
        raise AssertionError("unreachable")


class RangeFile(io.RawIOBase):
    """A remote file read by HTTP byte ranges, for h5py.File(RangeFile(...), "r").

    The first request follows the redirects by hand: the token goes to NASA hosts only, and the final URL (a signed
    S3 URL on Earthdata Cloud) is kept for every later range. Reads go through an LRU cache of `block`-byte blocks;
    a read longer than two blocks is fetched in one request. A signed URL that has expired (403) is resolved again
    once. `fetched` counts the bytes downloaded."""

    def __init__(
        self,
        session: requests.Session,
        url: str,
        token: str | None = None,
        block: int = 4 << 20,
        cache_blocks: int = 48,
        timeout: float = 180,
        tries: int = 5,
    ) -> None:
        super().__init__()
        self.session, self.url, self.token = session, url, token
        self.block, self.cache_blocks, self.timeout, self.tries = block, cache_blocks, timeout, tries
        self.pos, self.fetched = 0, 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._resolve()

    def _resolve(self) -> None:
        url = self.url
        for _ in range(8):
            hdr = {"Range": "bytes=0-0"}
            if self.token and EarthdataClient.is_nasa(url):
                hdr["Authorization"] = f"Bearer {self.token}"
            r = self.session.get(url, headers=hdr, allow_redirects=False, stream=True, timeout=self.timeout)
            r.close()
            if r.status_code in (301, 302, 303, 307, 308):
                nxt = urljoin(url, r.headers["Location"])
                if "urs.earthdata.nasa.gov" in urlparse(nxt).netloc:
                    raise PermissionError(
                        f"Earthdata Login refused the request for {self.url}: "
                        f"EARTHDATA_TOKEN is missing, expired or not valid. {EarthdataClient.TOKEN_HELP}"
                    )
                url = nxt
                continue
            if r.status_code == 206:
                self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            elif r.status_code == 200:
                self.size = int(r.headers["Content-Length"])
            elif r.status_code in (401, 403):
                raise PermissionError(f"HTTP {r.status_code} for {self.url}: {EarthdataClient.TOKEN_HELP}")
            else:
                raise RuntimeError(f"HTTP {r.status_code} for {self.url}")
            self.final = url
            return
        raise RuntimeError(f"too many redirects for {self.url}")

    def _get(self, a: int, b: int) -> bytes:
        """Bytes a..b (inclusive) of the file."""
        err, resolved = "", False
        for attempt in range(self.tries):
            hdr = {"Range": f"bytes={a}-{b}"}
            if self.token and EarthdataClient.is_nasa(self.final):
                hdr["Authorization"] = f"Bearer {self.token}"
            try:
                r = self.session.get(self.final, headers=hdr, timeout=self.timeout)
                if r.status_code == 206 or (r.status_code == 200 and a == 0 and len(r.content) == b + 1):
                    self.fetched += len(r.content)
                    return r.content
                if r.status_code == 200:  # server ignored the range: cut it here
                    self.fetched += len(r.content)
                    return r.content[a : b + 1]
                if r.status_code == 403 and not resolved:  # signed URL expired
                    self._resolve()
                    resolved = True
                    continue
                if r.status_code != 429 and r.status_code < 500:
                    raise RuntimeError(f"HTTP {r.status_code} for {self.url} bytes {a}-{b}: {r.text[:150]}")
                err = f"HTTP {r.status_code}"
            except (OSError, ValueError) as ex:  # requests' errors are OSError / ValueError
                err = f"{type(ex).__name__}: {str(ex)[:120]}"
            if attempt == self.tries - 1:
                raise RuntimeError(f"{self.url} bytes {a}-{b}: {err} after {self.tries} tries")
            time.sleep(3 * 2**attempt)
        raise AssertionError("unreachable")

    def _block(self, k: int) -> bytes:
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        a = k * self.block
        data = self._get(a, min(a + self.block, self.size) - 1)
        self._cache[k] = data
        if len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return data

    # io interface used by h5py
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = (
            offset if whence == io.SEEK_SET else (self.pos + offset if whence == io.SEEK_CUR else self.size + offset)
        )
        return self.pos

    def readinto(self, b) -> int:
        out = memoryview(b).cast("B")
        n = min(len(out), self.size - self.pos)
        if n <= 0:
            return 0
        if n >= 2 * self.block:
            out[:n] = self._get(self.pos, self.pos + n - 1)
        else:
            done = 0
            while done < n:
                k, off = divmod(self.pos + done, self.block)
                blk = self._block(k)
                take = min(len(blk) - off, n - done)
                out[done : done + take] = blk[off : off + take]
                done += take
        self.pos += n
        return n
