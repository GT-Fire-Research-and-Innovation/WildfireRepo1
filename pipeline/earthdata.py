"""
NASA Earthdata access for the raw grab (raw_channels.py: viirs, vegetation, landcover).

  cmr_granules    granule search in NASA's Common Metadata Repository (no login needed)
  token           Earthdata Login (EDL) bearer token, needed to read LAADS DAAC and LP DAAC files
  RangeFile       read-only file over HTTP byte ranges, so h5py reads only the parts of a remote
                  HDF5 / netCDF4 file it needs (a fire's window of a 250 MB swath file is a few MB)
  sinu_*          the MODIS / VIIRS sinusoidal tile grid: a fire box as global pixel rows/cols, and
                  which part of each 2400 x 2400 tile falls inside it (tiles join pixel for pixel)
  swath_window    rows/cols of a VIIRS swath whose pixel centres fall in a fire box

Earthdata Cloud answers a file request with a redirect to a signed S3 URL (valid about an hour). The token
goes only to NASA hosts, never to the S3 URL, and never into a manifest (raw_channels.redact).
Free token: https://urs.earthdata.nasa.gov > Generate Token (valid 60 days).
"""
import io
import math
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urljoin, urlparse

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
TOKEN_ENV = "EARTHDATA_TOKEN"
TOKEN_HELP = ("Set EARTHDATA_TOKEN: a free Earthdata Login token from https://urs.earthdata.nasa.gov "
              "(Generate Token). On Kaggle add it under Add-ons > Secrets.")
NASA_HOSTS = (".earthdata.nasa.gov", ".earthdatacloud.nasa.gov", ".eosdis.nasa.gov")


# ------------------------------------------------------------------------------------- login

def token(required=True):
    """The EDL bearer token from the environment, .env or ~/.config/wildfire/env."""
    v = os.environ.get(TOKEN_ENV)
    for p in (Path(".env"), Path.home() / ".config" / "wildfire" / "env"):
        if not v and p.exists():
            for line in p.read_text().splitlines():
                if line.startswith(TOKEN_ENV + "="):
                    v = line.split("=", 1)[1].split("#")[0].strip()
    if not v and required:
        raise SystemExit(TOKEN_HELP)
    return v or None


def _nasa(url):
    return urlparse(url).netloc.endswith(NASA_HOSTS)


# ------------------------------------------------------------------------------------- CMR

def granule_parts(url):
    """('VNP02IMG', 'A2018314', '2048', '002') from a granule file name; tiles give ('VNP13A1', 'A2018305',
    'h08v05', '002'). The first three identify the observation, the fourth is the collection version."""
    p = Path(urlparse(url).path).name.split(".")
    return p[0], p[1], p[2], p[3]


def granule_day(url):
    """The A<YYYY><DDD> date of a granule as a pandas Timestamp (UTC midnight)."""
    import pandas as pd
    a = granule_parts(url)[1]
    return pd.Timestamp(f"{a[1:5]}-01-01", tz="UTC") + pd.Timedelta(days=int(a[5:8]) - 1)


def _version_key(v):
    return tuple(int(x) if x.isdigit() else x for x in re.findall(r"\d+|[A-Za-z]+", v))


def cmr_granules(sess, short_name, t0, t1, bbox, page_size=500, tries=5):
    """Granules of a collection that overlap [t0, t1] and the box (lon/lat), newest collection version only:
    [{'url', 'name', 'start', 'end', 'daynight'}] in time order. t0 / t1: anything pandas reads (UTC)."""
    import pandas as pd

    def iso(t):
        t = pd.Timestamp(t)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {"short_name": short_name, "temporal": f"{iso(t0)},{iso(t1)}", "page_size": page_size,
              "bounding_box": ",".join(f"{v:.5f}" for v in bbox)}
    out, after = [], None
    while True:
        hdr = {"CMR-Search-After": after} if after else {}
        for attempt in range(tries):
            try:
                r = sess.get(CMR, params=params, headers=hdr, timeout=120)
                if r.status_code == 200:
                    break
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as ex:  # noqa: BLE001  (network)
                err = f"{type(ex).__name__}: {str(ex)[:150]}"
            if attempt == tries - 1:
                raise RuntimeError(f"CMR {short_name}: {err}")
            time.sleep(3 * 2 ** attempt)
        items = r.json().get("items", [])
        for it in items:
            u = it["umm"]
            urls = [x["URL"] for x in u.get("RelatedUrls", [])
                    if x.get("Type") == "GET DATA" and x["URL"].startswith("https://")]
            if not urls:
                continue
            te = (u.get("TemporalExtent") or {}).get("RangeDateTime") or {}
            out.append({"url": urls[0], "name": Path(urlparse(urls[0]).path).name,
                        "start": te.get("BeginningDateTime"), "end": te.get("EndingDateTime"),
                        "daynight": (u.get("DataGranule") or {}).get("DayNightFlag", "")})
        after = r.headers.get("CMR-Search-After")
        if not after or len(items) < page_size:
            break
    return newest_versions(out)


def newest_versions(granules):
    """Keep one granule per observation (product, date, time or tile): the newest collection version."""
    best = {}
    for g in granules:
        p = granule_parts(g["url"])
        k = p[:3]
        if k not in best or _version_key(p[3]) > _version_key(granule_parts(best[k]["url"])[3]):
            best[k] = g
    return sorted(best.values(), key=lambda g: (str(g["start"]), g["name"]))


# ------------------------------------------------------------------------------------- byte ranges

class RangeFile(io.RawIOBase):
    """A remote file read by HTTP byte ranges, for h5py.File(RangeFile(...), "r").

    The first request follows the redirects by hand: the token goes to NASA hosts only, and the final URL
    (a signed S3 URL on Earthdata Cloud) is kept for every later range. Reads go through an LRU cache of
    `block`-byte blocks; a read longer than two blocks is fetched in one request. A signed URL that has
    expired (403) is resolved again once. `fetched` counts the bytes downloaded."""

    def __init__(self, sess, url, token=None, block=4 << 20, cache_blocks=48, timeout=180, tries=5):
        super().__init__()
        self.sess, self.url, self.token = sess, url, token
        self.block, self.cache_blocks, self.timeout, self.tries = block, cache_blocks, timeout, tries
        self.pos, self.fetched = 0, 0
        self._cache = OrderedDict()
        self._resolve()

    def _resolve(self):
        url = self.url
        for _ in range(8):
            hdr = {"Range": "bytes=0-0"}
            if self.token and _nasa(url):
                hdr["Authorization"] = f"Bearer {self.token}"
            r = self.sess.get(url, headers=hdr, allow_redirects=False, stream=True, timeout=self.timeout)
            r.close()
            if r.status_code in (301, 302, 303, 307, 308):
                nxt = urljoin(url, r.headers["Location"])
                if "urs.earthdata.nasa.gov" in urlparse(nxt).netloc:
                    raise PermissionError(f"Earthdata Login refused the request for {self.url}: "
                                          f"EARTHDATA_TOKEN is missing, expired or not valid. {TOKEN_HELP}")
                url = nxt
                continue
            if r.status_code == 206:
                self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            elif r.status_code == 200:
                self.size = int(r.headers["Content-Length"])
            elif r.status_code in (401, 403):
                raise PermissionError(f"HTTP {r.status_code} for {self.url}: {TOKEN_HELP}")
            else:
                raise RuntimeError(f"HTTP {r.status_code} for {self.url}")
            self.final = url
            return
        raise RuntimeError(f"too many redirects for {self.url}")

    def _get(self, a, b):
        """Bytes a..b (inclusive) of the file."""
        err, resolved = "", False
        for attempt in range(self.tries):
            hdr = {"Range": f"bytes={a}-{b}"}
            if self.token and _nasa(self.final):
                hdr["Authorization"] = f"Bearer {self.token}"
            try:
                r = self.sess.get(self.final, headers=hdr, timeout=self.timeout)
                if r.status_code == 206 or (r.status_code == 200 and a == 0 and len(r.content) == b + 1):
                    self.fetched += len(r.content)
                    return r.content
                if r.status_code == 200:                     # server ignored the range: cut it here
                    self.fetched += len(r.content)
                    return r.content[a:b + 1]
                if r.status_code == 403 and not resolved:    # signed URL expired
                    self._resolve()
                    resolved = True
                    continue
                if r.status_code != 429 and r.status_code < 500:
                    raise RuntimeError(f"HTTP {r.status_code} for {self.url} bytes {a}-{b}: {r.text[:150]}")
                err = f"HTTP {r.status_code}"
            except (OSError, ValueError) as ex:              # requests' errors are OSError / ValueError
                err = f"{type(ex).__name__}: {str(ex)[:120]}"
            if attempt == self.tries - 1:
                raise RuntimeError(f"{self.url} bytes {a}-{b}: {err} after {self.tries} tries")
            time.sleep(3 * 2 ** attempt)

    def _block(self, k):
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
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = offset if whence == io.SEEK_SET else (self.pos + offset if whence == io.SEEK_CUR else self.size + offset)
        return self.pos

    def readinto(self, b):
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
                out[done:done + take] = blk[off:off + take]
                done += take
        self.pos += n
        return n


def open_remote(sess, url, token=None):
    """A file-like object for h5py; tests replace this with a local opener."""
    return RangeFile(sess, url, token)


def download(sess, url, dst, token=None, tries=5, timeout=600):
    """Whole file to dst (for formats h5py cannot read, such as HDF4). The token goes to NASA hosts only:
    requests drops the Authorization header when a redirect leaves the host."""
    import requests
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    hdr = {"Authorization": f"Bearer {token}"} if token and _nasa(url) else {}
    for attempt in range(tries):
        try:
            with sess.get(url, headers=hdr, stream=True, timeout=timeout) as r:
                if "urs.earthdata.nasa.gov" in urlparse(r.url).netloc:
                    raise PermissionError(f"Earthdata Login refused the request for {url}. {TOKEN_HELP}")
                if r.status_code in (401, 403):
                    raise PermissionError(f"HTTP {r.status_code} for {url}. {TOKEN_HELP}")
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
            time.sleep(5 * 2 ** attempt)


# ------------------------------------------------------------------------------------- sinusoidal tiles

# MODIS / VIIRS land grid: 36 x 18 tiles of 2400 x 2400 pixels (~463 m) on a sphere of radius 6371007.181 m.
SINU_PROJ = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
SINU_TILE_M = 1111950.5197665233
SINU_X0, SINU_Y0 = -20015109.355798, 10007554.677899     # upper-left corner of tile h00v00
SINU_N = 2400


def sinu_window(bounds_ll, n=SINU_N):
    """A lon/lat box as global sinusoidal pixel rows/cols (r0, r1, c0, c1), end-exclusive, snapped outward."""
    from pyproj import Transformer
    x0, y0, x1, y1 = Transformer.from_crs(4326, SINU_PROJ, always_xy=True).transform_bounds(*bounds_ll, densify_pts=41)
    px = SINU_TILE_M / n
    return (math.floor((SINU_Y0 - y1) / px), math.ceil((SINU_Y0 - y0) / px),
            math.floor((x0 - SINU_X0) / px), math.ceil((x1 - SINU_X0) / px))


def tile_hv(name):
    """(h, v) of a tile from a granule name such as VNP13A1.A2018305.h08v05.002.<prod>.h5."""
    m = re.search(r"\.h(\d\d)v(\d\d)\.", name)
    if not m:
        raise ValueError(f"no tile in {name}")
    return int(m.group(1)), int(m.group(2))


def tile_part(win, h, v, n=SINU_N):
    """Overlap of a global window with tile (h, v): (tile rows/cols, output rows/cols) as two
    (r0, r1, c0, c1) tuples, or None."""
    r0, r1, c0, c1 = win
    tr0, tr1, tc0, tc1 = max(r0, v * n), min(r1, (v + 1) * n), max(c0, h * n), min(c1, (h + 1) * n)
    if tr0 >= tr1 or tc0 >= tc1:
        return None
    return (tr0 - v * n, tr1 - v * n, tc0 - h * n, tc1 - h * n), (tr0 - r0, tr1 - r0, tc0 - c0, tc1 - c0)


def sinu_coords(win, n=SINU_N):
    """Pixel-centre x (cols) and y (rows) in metres of a global window."""
    import numpy as np
    r0, r1, c0, c1 = win
    px = SINU_TILE_M / n
    return SINU_X0 + (np.arange(c0, c1) + 0.5) * px, SINU_Y0 - (np.arange(r0, r1) + 0.5) * px


# ------------------------------------------------------------------------------------- swaths

def swath_window(lat, lon, bounds, pad=0.0):
    """(r0, r1, c0, c1) of the swath pixels whose centres fall in the box (+pad degrees), or None."""
    import numpy as np
    w, s, e, n = bounds
    with np.errstate(invalid="ignore"):
        m = (lat >= s - pad) & (lat <= n + pad) & (lon >= w - pad) & (lon <= e + pad)
    rows = np.flatnonzero(m.any(axis=1))
    if not len(rows):
        return None
    cols = np.flatnonzero(m.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def m_to_i(win, i_shape, margin=2):
    """A window on the 750 m M-band swath as the window on the 375 m I-band swath of the same granule.
    VIIRS scans 16 M-band and 32 I-band detector rows at a time with twice the columns, so M pixel (r, c)
    covers I pixels (2r..2r+1, 2c..2c+1); `margin` I pixels are added on every side."""
    r0, r1, c0, c1 = win
    return (max(2 * r0 - margin, 0), min(2 * r1 + margin, i_shape[0]),
            max(2 * c0 - margin, 0), min(2 * c1 + margin, i_shape[1]))
