"""
Loader for the multi-year dataset (pipeline/build_multiyear.py output).

A sample is one pair of consecutive perimeters (i -> j) of one fire, at their real times (any gap
from 3 h to 4 days). Inputs describe the fire at perimeter i's time t_i and the weather over
[t_i, t_j]; the target is the cells newly burned by t_j.

Every input group that can be missing has a 0/1 mask channel, and during training whole groups are
randomly hidden (mask set to 0) so the model learns to work without them.
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MIN_GAP_H, MAX_GAP_H = 3.0, 168.0          # pairs up to 7 days apart (98 % of kept pairs); the gap is a model input
CLEAN_FAR_KM = 3.0                 # new burn farther than this with no satellite fire nearby -> unknown
DATA_VERSION = 2                   # 2: per-pair satellite-support check on labels + isolated GOES pixels removed
                                   # 1: original behaviour (kept for checkpoints trained before the change)
INPUT_VERSION = 5                  # 5: perimeter times corrected (GeoMAC local clock time -> UTC, about +6 h for 2014-2019);
                                   #    HRRR winds rotated to true north; hours with no HRRR file no longer count as calm, 0 % RH air;
                                   #    perimeters stamped with a database edit time: satellite inputs stop EDIT_LAG_H earlier
                                   # 4: suppression layers of the PREVIOUS burn day (the day's own layers are the map at 06:00
                                   #    the next morning, up to 24 h after the forecast start, incl. the new uncontrolled edge);
                                   #    fire size, distance to fire and relative elevation no longer depend on the crop window
                                   # 3: GOES input filter is causal (the isolated-pixel check looked up to 3 h ahead)
                                   # 2: satellite inputs exclude passes after the forecast start (time-leak fix, 2026-11-11)
                                   # 1: whole local day of the forecast start included (leaked up to ~22 h of the future)
SUPPORT_MIN_CELLS = 20             # v2: new-burn patches at least this big (0.45 km² at 150 m) ...
SUPPORT_KM, SUPPORT_FRAC = 1.0, 0.2    # ... with under 20 % of cells within 1 km of VIIRS/GOES fire -> unknown
GOES_ISOLATED_KM, GOES_ISOLATED_H = 4.0, 3.0   # v2: a GOES pixel with no other within 4 km and 3 h is dropped
EDIT_LAG_H = 8.0                   # an edit-time stamp is typically the 04-06 upload of a 21-23 flight
HERB, WOODY = 0.6, 0.9             # live fuel moisture defaults (fraction)
GM = ["vpd", "fm100", "fm1000", "bi", "erc", "rmin", "vs", "th", "tmmx"]
GM_SCALE = {"vpd": (1.5, 1.0), "fm100": (12, 5), "fm1000": (14, 5), "bi": (50, 25), "erc": (50, 25),
            "rmin": (25, 15), "vs": (4, 2), "th": (0, 1), "tmmx": (300, 8)}
FUEL_FAMILIES = [("NB", 91, 99), ("GR", 101, 109), ("GS", 121, 124), ("SH", 141, 149), ("TU", 161, 165),
                 ("TL", 181, 189), ("SB", 201, 204)]
OPTIONAL = ["goes", "fuels", "tessera", "human", "hrrr", "suppression", "history"]

GROUPS = {  # name -> number of channels (including its mask channel where optional)
    "fire": 11, "viirs": 3, "goes": 4, "terrain": 7, "fuels": 12, "tessera": 17, "human": 10, "history": 3,
    "weather": 14, "hrrr": 7, "suppression": 12, "physics": 4,
    "crownspot": 4,          # physics v2 only: crown probability, crown-adjusted spread, embers, intensity
    "origin": 1,             # stale-perimeter models only: age of the perimeter at the forecast start (h / 48)
    "fmoist": 15,            # fuel-moisture models only: hourly 1-h / 10-h dead fuel moisture, rain, sunshine (fuel_moisture.py)
    "green": 4,              # fuel-moisture models only: satellite greenness (NDVI, EVI, 32-day NDVI change, valid)
    "hrrr_seq": 16,          # unet3 (time) only: HRRR over the gap in 4 equal segments x (u, v, speed, RH)
    "seq": 24,               # unet3 (time) only: last 6 days x (burned, VIIRS detections, snapshot age, valid); kept LAST
}
V2_GROUPS = ["crownspot"]    # left out unless PHYSICS_V2 (keeps v1 models and checkpoints unchanged)
T3_GROUPS = ["hrrr_seq", "seq"]  # left out unless TEMPORAL (unet3 with its time dimension)
FM_GROUPS = ["fmoist", "green"]   # left out unless the model was trained with --fuel-moisture
FM_ON = False                # set when a loaded model uses the fuel-moisture groups (they cost time to compute)
FUEL_V2 = False              # --fuel-moisture: also read landfire2.npz (the right fuel map for the fire's year) when it exists
FM_PHYSICS = False           # --fm-physics: the spread equations use the hourly, slope-aware fine fuel moisture (driest in the window)
O_GROUPS = ["origin"]        # left out unless the model was trained with --origin-lag (forecasts from an ageing perimeter)
ORIGIN_MAX_H = 36.0          # training draws the perimeter's age from 0..this (and 0 half the time)
TEMPORAL = False             # set for unet3 runs that use the day-by-day history
SEQ_DAYS, SEQ_CH, SEQ_SEG = 6, 4, 4
PHYSICS_V2 = False           # set by the hybrid2 / physics2 modes: compute crown fire and spotting
HRRR_ROTATE = True           # rotate HRRR winds from the HRRR grid's axes to true north (up to 17 degrees in the West)
HRRR_CONE, HRRR_LON0 = 0.622515, -97.5     # HRRR Lambert conformal: sin(38.5 deg), central meridian
ROTH_WIND = "vector"         # wind speed given to Rothermel: "vector" (speed of the mean wind vector), "mean", "p90"
CANOPY_WAF = False           # shelter the midflame wind under forest canopy (LANDFIRE cover and height)
GM_DAY = "start"             # GRIDMET day: "start" (local day of t_i) or "window" (days whose afternoon is in the window)
WEATHER_MODE = "observed"    # "observed": HRRR analyses over the gap (oracle weather, best case)
                             # "forecast": only what a forecaster has at t_i: HRRR analyses of the last 24 h
                             #   plus the HRRR 12z+9h forecasts issued at least 2 h before t_i that are valid
                             #   during the gap; GRIDMET from the previous day unless t_i is after 18:00 local


from collections import OrderedDict

_TSRC = None


def time_sources():
    """{fire_id: {perimeter time (s, as stored in the catalog): 'mapped' | 'edit' | 'date_only'}} from
    pipeline/perimeter_time_source.csv.gz. 'mapped' = the time the perimeter was flown or mapped;
    'edit' = only a database edit time exists (NIFS, 28-59 % of perimeters depending on the year);
    'date_only' = a date with no time of day."""
    global _TSRC
    if _TSRC is None:
        _TSRC = {}
        p = Path(__file__).resolve().parent.parent / "pipeline" / "perimeter_time_source.csv.gz"
        if p.exists():
            t = pd.read_csv(p)
            for fid, g in t.groupby("fire_id"):
                _TSRC[fid] = dict(zip(g["t_s"].astype(np.int64), g["time_source"]))
    return _TSRC


def geomac_zone(lon, lat, year):
    """Time zone of the clock a GeoMAC perimeter time was written in. GeoMAC stored local clock times
    with no zone (the archive serves them as if UTC): 2019 records were converted to Denver time for
    every fire (250 perimeters also in NIFS differ by exactly 6 h); earlier years carry the fire's own
    local time (NIROPS comments such as '2338hrs' equal the stored time; infrared flights would
    otherwise cluster at 13-17 local instead of 20-24)."""
    if year >= 2019:
        return "America/Denver"
    if lon < -130:
        return "America/Anchorage"
    if lon > -85.5:
        return "America/New_York"
    if lon > -101.5:
        return "America/Chicago"
    if lat < 37.0 and -114.8 < lon < -109.05:
        return "America/Phoenix"
    if lon > -114.05 or (lon > -117.03 and 42.0 <= lat <= 45.6):
        return "America/Denver"
    return "America/Los_Angeles"


def fix_times(times_ns, sources, lon, lat, year):
    """Catalog times (ns, stored as UTC) -> true UTC: GeoMAC entries are local clock times."""
    t = pd.to_datetime(np.asarray(times_ns, np.int64), utc=True)
    if sources is None:
        return t
    zone = geomac_zone(lon, lat, year)
    out = []
    for ti, src in zip(t, sources):
        if str(src) == "geomac":
            ti = ti.tz_localize(None).tz_localize(zone, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
        out.append(ti)
    return pd.DatetimeIndex(out)

LAYER_CACHE = 24            # layers kept in memory per process (a fire has ~11)
_CACHE = OrderedDict()


class Fire:
    def __init__(self, d, pca=None):
        d = Path(d)
        self.d = d
        self.g = json.loads((d / "grid.json").read_text())
        self.fid = self.g["fire_id"]
        self.year = self.g["year"]
        self.H, self.W = self.g["height"], self.g["width"]
        self.res_m = float(self.g.get("res_m", 150.0))
        self.cell_km = self.res_m / 1000
        lab = np.load(d / "labels.npz")
        self.arrival, self.other = lab["arrival"], lab["other"]
        stored = np.asarray(lab["times"], np.int64)
        tr = self.g["transform"]
        lon_c, lat_c = tr[2] + tr[0] * self.W / 2, tr[5] + tr[4] * self.H / 2
        fixed = "times_utc" in lab.files                     # labels built after the catalog fix carry true UTC
        self.times = pd.to_datetime(stored, utc=True) if fixed or INPUT_VERSION < 5 or "sources" not in lab.files \
            else fix_times(stored, lab["sources"], lon_c, lat_c, self.year)
        if "time_sources" in lab.files:
            self.time_src = [str(v) for v in lab["time_sources"]]
        else:
            src = time_sources().get(self.fid, {})
            self.time_src = [src.get(int(t // 10 ** 9), "unknown") for t in stored]
        self.pca = pca
        self.pairs = []
        for i in range(len(self.times) - 1):
            gap = (self.times[i + 1] - self.times[i]).total_seconds() / 3600
            if MIN_GAP_H <= gap <= MAX_GAP_H and ((self.arrival >= 0) & (self.arrival <= i)).any():
                self.pairs.append((i, gap))

    def has(self, name):
        return (self.d / f"{name}.npz").exists()

    def z(self, name):
        """Layer arrays, from a small per-process cache shared by all fires (keeps RAM bounded:
        each DataLoader worker holds at most LAYER_CACHE layers at a time)."""
        key = (self.fid, name)
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
        if not self.has(name):
            return None
        arr = dict(np.load(self.d / f"{name}.npz", allow_pickle=False))
        _CACHE[key] = arr
        while len(_CACHE) > LAYER_CACHE:
            _CACHE.popitem(last=False)
        return arr

    def burned(self, i):
        return (self.arrival >= 0) & (self.arrival <= i)

    def front_cells(self, i):
        from scipy.ndimage import binary_dilation
        b = self.burned(i)
        return np.argwhere(binary_dilation(b, iterations=2) & ~b)

    # ------------------------------------------------------------------ features
    def sample(self, i, gap_h, win, hide=(), lag_h=0.0):
        """Inputs (dict of group -> (C, h, w) float32), physics fields, target, mask, burning.
        lag_h > 0: the forecast starts lag_h hours AFTER perimeter i was mapped, as it does on the live site:
        the fire shape is still perimeter i, but satellite and weather inputs run up to the later start,
        gap_h is the time left to perimeter i+1, and the target is everything that burns by then (what
        burned during the lag included: the model has to work that out from the satellite inputs)."""
        from scipy.ndimage import binary_dilation, distance_transform_edt
        r0, r1, c0, c1 = win
        sl = (slice(r0, r1), slice(c0, c1))
        h, w = r1 - r0, c1 - c0
        lag = pd.Timedelta(hours=float(lag_h))
        t_i, t_j = self.times[i] + lag, self.times[i + 1]    # t_i = the forecast start
        t_sat = self.sat_time(i) + lag                        # newest satellite observation the inputs may use
        f = {}
        B0 = self.burned(i)[sl]
        B1 = self.burned(i - 1)[sl] if i >= 1 else B0
        B2 = self.burned(i - 2)[sl] if i >= 2 else B1
        age1 = ((t_i - self.times[i - 1]).total_seconds() / 3600 if i >= 1 else 0.0) / 48
        age2 = ((t_i - self.times[i - 2]).total_seconds() / 3600 if i >= 2 else 0.0) / 48
        ck = self.cell_km
        dist = self.fire_distance(i)[sl]                      # from the whole fire, not just the part inside the window
        oth = self.other[sl]
        full = lambda v: np.full((h, w), v, np.float32)  # noqa: E731
        f["origin"] = full(min(float(lag_h), 96.0) / 48)[None]
        f["fire"] = np.stack([B0, B1, B2, full(age1), full(age2), full(gap_h / 48), np.clip(dist, 0, 15) / 15,
                              np.exp(-dist / 2), oth, full(math.log1p(self.burned(i).sum() * ck * ck) / 8),
                              full(math.sin(2 * math.pi * (t_i.hour + self.g["transform"][2] / 15) / 24))]).astype(np.float32)

        # VIIRS (FIRMS) over the 24 h before t_i (local solar days)
        v = self.z("viirs")
        af24 = np.zeros((h, w), np.float32)
        afn = np.zeros((h, w), np.float32)
        frp = np.zeros((h, w), np.float32)
        evidence = np.zeros((self.H, self.W), bool)
        if v is not None:
            days = pd.to_datetime(v["dates"])
            lon_c = self.g["transform"][2]
            local_i = (t_i.tz_convert(None) + pd.Timedelta(hours=lon_c / 15)).normalize()
            local_j = (t_j.tz_convert(None) + pd.Timedelta(hours=lon_c / 15)).normalize()
            for k, dday in enumerate(days):
                if local_i - pd.Timedelta(days=1) <= dday <= local_i:
                    use_d, use_n = self.passes_before(dday, t_sat)     # no satellite pass after the start (time leak fix)
                    if use_d:
                        af24 += v["af_day"][k][sl]
                    if use_n:
                        afn += v["af_night"][k][sl]
                    if use_d and use_n:                                # frp_max mixes both passes of the day
                        frp = np.maximum(frp, v["frp_max"][k][sl].astype(np.float32))
                if local_i - pd.Timedelta(days=1) <= dday <= local_j + pd.Timedelta(days=1):
                    evidence |= (v["af_day"][k] + v["af_night"][k]) > 0
        f["viirs"] = np.stack([np.minimum(af24, 4) / 4, np.minimum(afn, 4) / 4, np.log1p(frp) / 6]).astype(np.float32)
        if TEMPORAL:
            f["seq"] = self.history(i, sl, v, h, w, lag_h)

        # GOES: hours since a fire pixel was seen here (before t_i), detections and power in the last 6 h
        gz_raw = gz = self.z("goes")
        if gz is not None and DATA_VERSION >= 2 and len(gz["t"]):
            keep = self.goes_keep(gz)
            gz = {k: (v[keep] if getattr(v, "ndim", 0) == 1 and len(v) == len(keep) else v) for k, v in gz.items()}
        goes = np.zeros((4, h, w), np.float32)
        if gz is not None and "goes" not in hide and len(gz["t"]):
            if INPUT_VERSION >= 3 and DATA_VERSION >= 2:
                # input side: a pixel counts once a companion detection (4 km, 3 h) has been seen BY t_i;
                # the label-side filter also looks up to 3 h past each pixel, which would leak into inputs
                ok = self.goes_companion_t(gz_raw) <= pd.Timestamp(t_sat).value
                gz = {k: (v[ok] if getattr(v, "ndim", 0) == 1 and len(v) == len(ok) else v) for k, v in gz_raw.items()}
            t = pd.to_datetime(gz["t"], utc=True)
            before = (t <= t_sat) & (t >= t_i - pd.Timedelta(hours=72))
            if before.any():
                dx, _, x0, _, ey, y0 = self.g["transform"]
                cc = np.floor((gz["lon"][before] - x0) / dx).astype(int) - c0
                rr = np.floor((gz["lat"][before] - y0) / ey).astype(int) - r0
                hrs = (t_i - t[before]).total_seconds().values / 3600
                pw = gz["power"][before]
                since = np.full((h, w), 72.0, np.float32)
                cnt6 = np.zeros((h, w), np.float32)
                pw6 = np.zeros((h, w), np.float32)
                rad = max(1, int(round(1.05 / ck)))         # ~1 km: a GOES pixel is ~2-3 km
                for k in range(len(rr)):
                    a0, a1, b0, b1 = max(rr[k] - rad, 0), min(rr[k] + rad + 1, h), max(cc[k] - rad, 0), min(cc[k] + rad + 1, w)
                    if a0 >= a1 or b0 >= b1:
                        continue
                    since[a0:a1, b0:b1] = np.minimum(since[a0:a1, b0:b1], hrs[k])
                    if hrs[k] <= 6:
                        cnt6[a0:a1, b0:b1] += 1
                        pw6[a0:a1, b0:b1] = np.maximum(pw6[a0:a1, b0:b1], pw[k])
                goes = np.stack([1 - since / 72, np.minimum(cnt6, 6) / 6, np.log1p(pw6) / 7, full(1.0)])
            else:
                goes[3] = 1.0
        # GOES pixels during (t_i, t_j] count as label evidence too: v2 adds them even when the GOES
        # input is hidden, so labels never depend on which inputs were hidden (v1 kept the old coupling)
        if gz_raw is not None and DATA_VERSION >= 2 and len(gz_raw["t"]):      # labels: the full (two-sided) filter
            keep = self.goes_keep(gz_raw)
            gz = {k: (v[keep] if getattr(v, "ndim", 0) == 1 and len(v) == len(keep) else v) for k, v in gz_raw.items()}
        if gz is not None and len(gz["t"]) and ("goes" not in hide or DATA_VERSION >= 2):
            t = pd.to_datetime(gz["t"], utc=True)
            dur = (t > t_i - pd.Timedelta(hours=24)) & (t <= t_j + pd.Timedelta(hours=24))
            if dur.any():
                dx, _, x0, _, ey, y0 = self.g["transform"]
                cc = np.clip(np.floor((gz["lon"][dur] - x0) / dx).astype(int), 0, self.W - 1)
                rr = np.clip(np.floor((gz["lat"][dur] - y0) / ey).astype(int), 0, self.H - 1)
                gev = np.zeros((self.H, self.W), bool)
                gev[rr, cc] = True
                evidence |= binary_dilation(gev, iterations=max(1, int(round(1.5 / ck))))
        f["goes"] = goes

        ter = self.z("terrain")
        slope = ter["slope"][sl].astype(np.float32)
        asp = np.radians(ter["aspect"][sl].astype(np.float32))
        elev = ter["elev"][sl].astype(np.float32)
        f["terrain"] = np.stack([(elev - self.fire_elev(i, ter)) / 500,   # relative to the fire's own mean elevation
 slope / 45, np.sin(asp) * slope / 45, np.cos(asp) * slope / 45,
                                 ter["tpi_500m"][sl].astype(np.float32) / 100, ter["tpi_2km"][sl].astype(np.float32) / 300,
                                 ter["rough"][sl].astype(np.float32) / 50])

        lf = (self.z("landfire2") if FUEL_V2 and self.has("landfire2") else None) or self.z("landfire")
        fuels = np.zeros((12, h, w), np.float32)
        fb = None
        if lf is not None and not bool(lf.get("missing", True)) and "fuels" not in hide:
            fb = lf["fbfm40"][sl]
            for k, (_, a, b) in enumerate(FUEL_FAMILIES):
                fuels[k] = (fb >= a) & (fb <= b)
            fuels[7] = np.clip(lf["cc"][sl], 0, 100) / 100
            fuels[8] = np.clip(lf["ch"][sl], 0, 500) / 500
            fuels[9] = np.clip(lf["cbh"][sl], 0, 100) / 100
            fuels[10] = np.clip(lf["cbd"][sl], 0, 50) / 50
            fuels[11] = 1.0
        f["fuels"] = fuels

        tz = self.z("tessera")
        tess = np.zeros((17, h, w), np.float32)
        if tz is not None and not bool(tz.get("missing", True)) and "tessera" not in hide and self.pca is not None:
            e = tz["embedding"][:, r0:r1, c0:c1].astype(np.float32)
            comp, mean = self.pca
            tess[:16] = np.tensordot(comp, e - mean[:, None, None], axes=1) / 3.0
            tess[:16] *= tz["cover"][sl]
            tess[16] = tz["cover"][sl]
        f["tessera"] = tess

        hz = self.z("human")
        hum = np.zeros((10, h, w), np.float32)
        if hz is not None and "human" not in hide:
            L = hz["layers"][:, r0:r1, c0:c1].astype(np.float32)
            hum[:L.shape[0]] = L
            hum[9] = 1.0
        f["human"] = hum

        hi = self.z("history")
        hist = np.zeros((3, h, w), np.float32)
        if hi is not None and "history" not in hide:
            ys = hi["years_since"][sl].astype(np.float32)
            hist[0] = np.where(ys < 255, np.exp(-ys / 5), 0)
            hist[1] = np.minimum(hi["n_fires"][sl], 5) / 5
            hist[2] = 1.0
        f["history"] = hist

        # GRIDMET: day of t_i plus 7- and 14-day antecedent means
        wz = self.z("weather")
        weather = np.zeros((14, h, w), np.float32)
        gmv = {}
        if wz is not None:
            dates = pd.to_datetime(wz["dates"])
            local_i = (t_i.tz_convert(None) + pd.Timedelta(hours=self.g["transform"][2] / 15)).normalize()
            k = int(np.clip(np.searchsorted(dates, local_i), 0, len(dates) - 1))
            ks = [k]
            if GM_DAY == "window" and WEATHER_MODE != "forecast":
                off = pd.Timedelta(hours=self.g["transform"][2] / 15)
                li, lj = t_i.tz_convert(None) + off, t_j.tz_convert(None) + off
                dd = [d for d in pd.date_range(li.normalize(), lj.normalize()) if li <= d + pd.Timedelta(hours=15) <= lj]
                if not dd:
                    dd = [(li + (lj - li) / 2).normalize()]
                ks = sorted({int(np.clip(np.searchsorted(dates, d), 0, len(dates) - 1)) for d in dd})
                k = ks[-1]
            if WEATHER_MODE == "forecast":
                local_hour = (t_i.tz_convert(None) + pd.Timedelta(hours=self.g["transform"][2] / 15)).hour
                if local_hour < 18:                           # the day's weather is not observed yet
                    k = max(k - 1, 0)
            gr, gc = wz["gm_row"][r0:r1].astype(int), wz["gm_col"][c0:c1].astype(int)
            names = [str(x) for x in wz["gm_vars"]]
            gm = wz["gm"]
            for n in GM:
                if len(ks) > 1 and WEATHER_MODE != "forecast" and n != "th":      # a direction cannot be averaged in degrees
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        a = np.nanmean(gm[ks, names.index(n)], axis=0)[np.ix_(gr, gc)]
                else:
                    a = gm[k, names.index(n)][np.ix_(gr, gc)]
                gmv[n] = np.nan_to_num(a, nan=GM_SCALE[n][0])
            for c, n in enumerate(GM):
                mu, sd = GM_SCALE[n]
                weather[c] = np.sin(np.radians(gmv[n])) if n == "th" else (gmv[n] - mu) / sd
            for c, (n, days) in enumerate([("vpd", 7), ("rmin", 7), ("fm100", 7), ("erc", 7), ("fm1000", 14)]):
                with warnings.catch_warnings():          # all-NaN cells (gaps in GRIDMET) are filled below
                    warnings.simplefilter("ignore", RuntimeWarning)
                    a = np.nanmean(gm[max(0, k - days):k + 1, names.index(n)], axis=0)[np.ix_(gr, gc)]
                mu, sd = GM_SCALE[n]
                weather[9 + c] = (np.nan_to_num(a, nan=mu) - mu) / sd
        f["weather"] = weather

        # HRRR over [t_i, t_j]: mean u, v, mean and max speed, max gust, min RH
        rz = self.z("hrrr")
        hr = np.zeros((7, h, w), np.float32)
        u_mean = v_mean = None
        hseq = None
        self._spd_stats = None
        if rz is not None and "hrrr" not in hide:
            ts = pd.to_datetime(rz["times"], utc=True)
            if WEATHER_MODE == "forecast":
                sel = (ts >= t_i - pd.Timedelta(hours=24)) & (ts <= t_i)
            else:
                sel = (ts >= t_i - pd.Timedelta(hours=1)) & (ts <= t_j)
            if sel.sum() == 0:
                sel = np.abs((ts - t_i).total_seconds()) == np.abs((ts - t_i).total_seconds()).min()
                if WEATHER_MODE == "forecast":
                    sel &= (ts <= t_i)
            F = rz["fields"][sel].astype(np.float32)          # (T, 4, hh, wh): gust, u10, v10, rh2
            tF = ts[sel]
            if WEATHER_MODE == "forecast" and "fc" in rz and len(rz["fc_times"]):
                vt = pd.to_datetime(rz["fc_times"], utc=True)          # valid times (12z init + 9 h)
                issued = (pd.to_datetime(rz["fc_issued"], utc=True) if "fc_issued" in rz and len(rz["fc_issued"]) == len(vt)
                          else vt - pd.Timedelta(hours=9))     # dataset: 12z runs at +9 h; live: the real run time
                use = (vt > t_i) & (vt <= t_j) & (issued <= t_i - pd.Timedelta(hours=2))
                if use.any():
                    F = np.concatenate([F, rz["fc"][use].astype(np.float32)]) if len(F) else rz["fc"][use].astype(np.float32)
                    tF = tF.append(vt[use])
            idx = rz["idx"][sl]
            flat = F.reshape(F.shape[0], 4, -1)[:, :, idx]   # (T, 4, h, w)
            has_wind = np.isfinite(flat[:, 1]).any((1, 2)) & np.isfinite(flat[:, 2]).any((1, 2))
            if has_wind.any():
                # Hours whose file is missing are left out (they used to count as calm, 0 % humidity air).
                # HRRR had no 2 m humidity before Aug 2016 and has gaps later: missing humidity is filled
                # from the hours that have it, else from GRIDMET's daily minimum; never with zero.
                flat, tF = flat[has_wind], tF[has_wind]
                u, vv = np.nan_to_num(flat[:, 1]), np.nan_to_num(flat[:, 2])
                if HRRR_ROTATE:
                    # HRRR stores winds relative to its Lambert grid, not to true north: up to 17 degrees off in
                    # the West (checked against GRIDMET wind direction). Rotate to earth-relative east / north.
                    dxg, _, x0g = self.g["transform"][:3]
                    ang = math.radians(HRRR_CONE * (x0g + dxg * (c0 + c1) / 2 - HRRR_LON0))
                    u, vv = u * math.cos(ang) + vv * math.sin(ang), -u * math.sin(ang) + vv * math.cos(ang)
                spd = np.hypot(u, vv)
                gust = np.where(np.isfinite(flat[:, 0]), flat[:, 0], spd)
                rh = flat[:, 3]
                rh_ok = np.isfinite(rh)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rh_fill = np.nanmean(rh, 0) if rh_ok.any() else None
                if rh_fill is None or not np.isfinite(rh_fill).all():
                    alt = gmv["rmin"] if gmv else full(25.0)
                    rh_fill = alt if rh_fill is None else np.where(np.isfinite(rh_fill), rh_fill, alt)
                rh = np.where(rh_ok, rh, rh_fill[None])
                u_mean, v_mean = u.mean(0), vv.mean(0)
                hr = np.stack([u_mean / 8, v_mean / 8, spd.mean(0) / 8, spd.max(0) / 12, gust.max(0) / 20,
                               rh.min(0) / 100, full(1.0)])
                self._spd_stats = (spd.mean(0), np.quantile(spd, 0.9, axis=0) if len(spd) > 1 else spd[0])
                if TEMPORAL:
                    hseq = self.hrrr_segments(tF, u, vv, spd, rh, t_i, t_j, hr)
        f["hrrr"] = hr
        if TEMPORAL:
            f["hrrr_seq"] = hseq if hseq is not None else np.zeros((SEQ_SEG * 4, h, w), np.float32)

        # suppression on perimeter i's burn day
        sz = self.z("suppression")
        sup = np.zeros((12, h, w), np.float32)
        if sz is not None and "suppression" not in hide:
            dates = list(sz["dates"])
            # day d's layers = the map at 06:00 local on d+1, so the newest map that exists at t_i is the
            # PREVIOUS burn day's (as of 06:00 on the burn day containing t_i)
            local_i = str((t_i.tz_convert(None) + pd.Timedelta(hours=self.g["transform"][2] / 15 - 6 - 24)).date())
            if local_i in dates:
                k = dates.index(local_i)
                n_ch = int(sz["n_channels"])
                lay = np.unpackbits(sz["layers"][k], axis=0)[:n_ch]
                if bool(sz["has_spatial"]):
                    sup[:n_ch] = lay[:, r0:r1, c0:c1]
                    sup[10] = 1.0
                if bool(sz["has_ics"]):
                    fields = list(sz["ics_fields"])
                    vals = sz["ics_values"][k]
                    def gv(name):
                        return float(vals[fields.index(name)]) if name in fields and np.isfinite(vals[fields.index(name)]) else 0.0
                    sup[7] = math.log1p(gv("total_personnel")) / 8
                    sup[8] = math.log1p(gv("total_aerial")) / 4
                    sup[9] = gv("pct_contained") / 100
                    sup[11] = 1.0
        f["suppression"] = sup

        # Rothermel fields (physics backbone)
        self._canopy = (np.clip(lf["cc"][sl], 0, 100) / 100.0, np.clip(lf["ch"][sl], 0, 1000) / 10.0) if fb is not None else None
        if FM_ON or FM_PHYSICS:
            f["fmoist"], self._fm = self.fuel_moist(sl, t_i, t_j, ter, self._canopy[0] * 100 if fb is not None else 0.0, h, w)
            f["green"] = self.greenness(sl, t_sat, h, w)
        else:
            f["fmoist"], f["green"], self._fm = np.zeros((15, h, w), np.float32), np.zeros((4, h, w), np.float32), None
        phys = self.rothermel(fb, gmv, u_mean, v_mean, slope, ter["aspect"][sl].astype(np.float32), h, w)
        f["physics"] = np.stack([np.log1p(phys["r_head"]) / 3, phys["ecc"], phys["head_x"], phys["head_y"]]).astype(np.float32)
        if PHYSICS_V2:
            f["crownspot"] = self.crown_spot(phys, lf if fb is not None else None, sl, B0, hr, gmv, h, w)

        for k in f:                                   # last line of defence: no NaN / inf reaches the network
            f[k] = np.nan_to_num(f[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        phys = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in phys.items()}
        target = self.arrival[sl] == i + 1
        mask = ~B0 & ~oth
        far = target & (dist > CLEAN_FAR_KM) & ~evidence[sl]
        mask &= ~far
        if DATA_VERSION >= 2:
            mask &= ~self.unsupported(target & ~B0 & ~oth, evidence[sl])
        return f, phys, target, mask, B0

    def fuel_moist(self, sl, t_i, t_j, ter, canopy_cover, h, w):
        """The 'fmoist' input group (fuel_moisture.py): fine dead fuel moisture stepped through the weather
        from 72 h before the start to the end of the window, per cell. Needs hrrr2.npz (air temperature,
        dew point, sunshine, rain); zeros and no mask when it is missing. Forecast weather mode sees
        nothing after the start: the last 24 h of weather are repeated across the window, without rain."""
        import fuel_moisture as FM
        hz = self.z("hrrr2")
        zero = np.zeros((FM.N_CH, h, w), np.float32)
        if hz is None or not len(hz["times"]):
            return zero, None
        ts = pd.to_datetime(hz["times"], utc=True)
        idx = hz["idx"][sl]
        rz = self.z("hrrr")
        wind_at = {}
        if rz is not None:
            wind_at = {int(t): k for k, t in enumerate(rz["times"])}
            ridx = rz["idx"][sl]

        def fields(k):
            def load():
                t2 = hz["t2"][k].astype(np.float32).ravel()[idx]
                td = hz["dpt"][k].astype(np.float32).ravel()[idx] if "dpt" in hz else np.full((h, w), np.nan, np.float32)
                es = lambda x: 6.112 * np.exp(17.67 * x / (x + 243.5))  # noqa: E731
                rh = np.where(np.isfinite(td), 100 * es(td) / es(t2), np.nan)
                if not np.isfinite(rh).all() and rz is not None and int(hz["times"][k]) in wind_at:
                    rh2 = rz["fields"][wind_at[int(hz["times"][k])], 3].astype(np.float32).ravel()[ridx]
                    rh = np.where(np.isfinite(rh), rh, rh2)
                wind = 3.0
                if int(hz["times"][k]) in wind_at:
                    fz = rz["fields"][wind_at[int(hz["times"][k])]].astype(np.float32)
                    wind = np.nan_to_num(np.hypot(fz[1], fz[2]).ravel()[ridx], nan=3.0)
                return dict(temp_c=np.nan_to_num(t2, nan=20.0), rh=np.nan_to_num(rh, nan=30.0), wind_ms=wind,
                            rain_mm=hz["apcp3"][k].astype(np.float32).ravel()[idx] if "apcp3" in hz else 0.0,
                            ghi=hz["dswrf"][k].astype(np.float32).ravel()[idx] if "dswrf" in hz else None)
            return load

        ok = np.isfinite(hz["t2"].reshape(len(ts), -1)).any(1)
        lo = t_i - pd.Timedelta(hours=FM.SPINUP_H)
        if WEATHER_MODE == "forecast":
            ks = [k for k in range(len(ts)) if ok[k] and lo <= ts[k] <= t_i]
            steps = [dict(t=ts[k], load=fields(k)) for k in ks]
            day = [k for k in ks if ts[k] > t_i - pd.Timedelta(hours=24)]
            rep = 1
            while day and ts[day[0]] + pd.Timedelta(hours=24 * rep) <= t_j:
                for k in day:                               # yesterday's daily cycle again, dry
                    t = ts[k] + pd.Timedelta(hours=24 * rep)
                    if t <= t_j:
                        steps.append(dict(t=t, load=(lambda ld: (lambda: {**ld(), "rain_mm": 0.0}))(fields(k))))
                rep += 1
        else:
            steps = [dict(t=ts[k], load=fields(k)) for k in range(len(ts)) if ok[k] and lo <= ts[k] <= t_j]
        tr = self.g["transform"]
        lon_c, lat_c = tr[2] + tr[0] * (sl[1].start + w / 2), tr[5] + tr[4] * (sl[0].start + h / 2)
        elev = ter["elev"][sl].astype(np.float32)
        # the weather cell's own height: the mean elevation of the cells it covers (within this window)
        flat = idx.ravel()
        base = (np.bincount(flat, elev.ravel()) / np.maximum(np.bincount(flat), 1))[flat].reshape(h, w)
        out, fm = FM.integrate(steps, t_i, t_j, lat_c, lon_c, elev - base, ter["slope"][sl].astype(np.float32),
                               ter["aspect"][sl].astype(np.float32), canopy_cover)
        return (zero, None) if out is None else (out, fm)

    def greenness(self, sl, t_use, h, w):
        """The 'green' input group: the newest satellite greenness composite whose 16 days ended before
        t_use (NDVI, EVI), its NDVI change over the previous 32 days (curing), and a valid flag."""
        out = np.zeros((4, h, w), np.float32)
        gz = self.z("green")
        if gz is None or bool(gz.get("missing", False)):
            return out
        ends = pd.to_datetime(gz["ends"], utc=True) + pd.Timedelta(days=1)     # usable from the day after it ends
        past = np.flatnonzero(ends <= t_use)
        if not len(past):
            return out
        k = past[-1]
        nd = gz["ndvi"][k][sl].astype(np.float32) * 1e-4
        ok = gz["ndvi"][k][sl] > -3000
        out[0] = np.where(ok, nd, 0)
        out[1] = np.where(ok, gz["evi"][k][sl].astype(np.float32) * 1e-4, 0)
        older = np.flatnonzero(ends <= ends[k] - pd.Timedelta(days=32))
        if len(older):
            k0 = older[-1]
            both = ok & (gz["ndvi"][k0][sl] > -3000)
            out[2] = np.where(both, (nd - gz["ndvi"][k0][sl].astype(np.float32) * 1e-4) * 5, 0)
        out[3] = ok
        return out

    def sat_time(self, i):
        """Newest satellite observation the inputs of pair i may use. Normally the perimeter time. A perimeter
        stamped with a database edit time was really mapped hours earlier (typically flown 21-23 local,
        uploaded 04-06), so satellite passes just before the stamp show fire that the perimeter does not
        hold yet: growth the pair is asked to predict. Those inputs stop EDIT_LAG_H earlier."""
        t = self.times[i]
        if INPUT_VERSION >= 5 and getattr(self, "time_src", None) and self.time_src[i] == "edit":
            return t - pd.Timedelta(hours=EDIT_LAG_H)
        return t

    def fire_distance(self, i):
        """Distance (km) of every cell of the grid to the fire at perimeter i (30 when nothing has burned)."""
        if getattr(self, "_dist_i", None) != i:
            from scipy.ndimage import distance_transform_edt
            b = self.burned(i)
            self._dist = (distance_transform_edt(~b) * self.cell_km).astype(np.float32) if b.any() \
                else np.full((self.H, self.W), 30.0, np.float32)
            self._dist_i = i
        return self._dist

    def fire_elev(self, i, ter):
        """Mean elevation of the burned area at perimeter i (the whole grid's mean if nothing has burned)."""
        if getattr(self, "_elev_i", None) != i:
            b = self.burned(i)
            e = ter["elev"]
            self._elev = float(e[b].mean()) if b.any() else float(e.mean())
            self._elev_i = i
        return self._elev

    def passes_before(self, dday, t_i):
        """Which satellite passes of local solar day `dday` ended before t_i: (day pass, night pass).
        The daily VIIRS rasters hold a whole local day; a forecast starting at 02:00 must not see that day's
        13:30 pass (it falls inside the forecast window). Pass times come from the fire's detection points
        (points.csv.gz, the same local-day rule the rasters were built with). A pass with no detections has
        nothing to leak. Without the points file, the day of t_i is left out entirely (safe)."""
        ends = self.pass_ends()
        if ends is None:
            ok = dday.normalize() < (t_i.tz_convert(None) + pd.Timedelta(hours=self.g["transform"][2] / 15)).normalize()
            return ok, ok
        key = dday.strftime("%Y-%m-%d")
        tu = t_i.tz_convert(None)
        return (ends.get((key, "D"), tu) < tu or (key, "D") not in ends,
                ends.get((key, "N"), tu) < tu or (key, "N") not in ends)

    def pass_ends(self):
        if not hasattr(self, "_pass_ends"):
            self._pass_ends = None
            p = self.d / "points.csv.gz"
            if p.exists():
                try:
                    pts = pd.read_csv(p, usecols=["longitude", "acq_date", "acq_time", "daynight"])
                    hhmm = pts["acq_time"].astype(int).astype(str).str.zfill(4)
                    t = (pd.to_datetime(pts["acq_date"]) + pd.to_timedelta(hhmm.str[:2].astype(int), unit="h")
                         + pd.to_timedelta(hhmm.str[2:].astype(int), unit="m"))
                    local = (t + pd.to_timedelta(pts["longitude"] / 15.0, unit="h")).dt.strftime("%Y-%m-%d")
                    kind = np.where(pts["daynight"].astype(str) == "N", "N", "D")
                    self._pass_ends = {k: v for k, v in pd.Series(t.values, index=pd.MultiIndex.from_arrays([local, kind])).groupby(level=[0, 1]).max().items()}
                except (ValueError, KeyError, OSError):
                    self._pass_ends = None
        return self._pass_ends

    def history(self, i, sl, v, h, w, lag_h=0.0):
        """Day-by-day look back from t_i (unet3): for k = 0..5 days before t_i, the burned area at the
        latest observation not after t_i - k days, that local day's VIIRS detections (day + night),
        the snapshot's age and a valid flag. Uses nothing after t_i (the k = 0 VIIRS day is the same
        local day the 'viirs' group already uses). t_i is the forecast start (perimeter time + lag_h)."""
        lag = pd.Timedelta(hours=float(lag_h))
        t_i = self.times[i] + lag
        out = np.zeros((SEQ_DAYS, SEQ_CH, h, w), np.float32)
        days = pd.to_datetime(v["dates"]) if v is not None else None
        lon_c = self.g["transform"][2]
        local_i = (t_i.tz_convert(None) + pd.Timedelta(hours=lon_c / 15)).normalize()
        for k in range(SEQ_DAYS):
            cut = t_i - pd.Timedelta(days=k) + pd.Timedelta(minutes=1)
            js = np.nonzero(self.times[: i + 1] <= cut)[0]
            if len(js):
                j = int(js[-1])
                out[k, 0] = self.burned(j)[sl]
                out[k, 2] = (t_i - self.times[j]).total_seconds() / 3600 / (24 * SEQ_DAYS)
                out[k, 3] = 1.0
            if days is not None:
                hit = np.nonzero(days == local_i - pd.Timedelta(days=k))[0]
                if len(hit):
                    q = int(hit[0])
                    use_d, use_n = self.passes_before(days[q], self.sat_time(i) + lag)
                    af = (v["af_day"][q][sl] * use_d + v["af_night"][q][sl] * use_n).astype(np.float32)
                    out[k, 1] = np.minimum(np.log1p(af) / np.log(5), 1.0)
        return out.reshape(SEQ_DAYS * SEQ_CH, h, w)

    @staticmethod
    def hrrr_segments(tF, u, vv, spd, rh, t_i, t_j, hr):
        """HRRR over the gap in SEQ_SEG equal time segments (unet3): mean u, v, speed and RH per segment.
        In forecast mode only forecasts valid in the gap fall in it; an empty segment takes the gap mean."""
        tt = np.asarray(pd.DatetimeIndex(tF).asi8, dtype=np.float64)
        edges = np.linspace(t_i.value, t_j.value, SEQ_SEG + 1)
        segs = []
        for s in range(SEQ_SEG):
            sel = (tt >= edges[s]) & ((tt < edges[s + 1]) if s < SEQ_SEG - 1 else (tt <= edges[s + 1]))
            if sel.any():
                segs += [u[sel].mean(0) / 8, vv[sel].mean(0) / 8, spd[sel].mean(0) / 8, rh[sel].mean(0) / 100]
            else:
                segs += [hr[0], hr[1], hr[2], rh.mean(0) / 100]
        return np.stack(segs).astype(np.float32)

    def unsupported(self, new, evidence):
        """v2 label check: patches of new burn with almost no satellite fire nearby (re-mapping of
        area that burned earlier, or digitising errors). They become unknown, never 'did not burn'."""
        from scipy.ndimage import binary_dilation, label
        out = np.zeros_like(new)
        if not new.any():
            return out
        near = binary_dilation(evidence, iterations=max(1, int(round(SUPPORT_KM / self.cell_km))))
        lab, n = label(new, structure=np.ones((3, 3)))
        if n == 0:
            return out
        idx = np.arange(1, n + 1)
        from scipy.ndimage import sum as nsum
        size = nsum(new, lab, idx)
        sup = nsum(new & near, lab, idx) / np.maximum(size, 1)
        bad = idx[(size >= SUPPORT_MIN_CELLS) & (sup < SUPPORT_FRAC)]
        if len(bad):
            out = np.isin(lab, bad)
        return out

    def goes_keep(self, gz):
        """v2: drop GOES pixels with no other detection within 4 km and 3 h (single-frame false alarms)."""
        if getattr(self, "_goes_keep", None) is not None and len(self._goes_keep) == len(gz["t"]):
            return self._goes_keep
        from scipy.spatial import cKDTree
        lat0 = math.radians(float(np.mean(gz["lat"])))
        x = gz["lon"].astype(np.float64) * 111.32 * math.cos(lat0)
        y = gz["lat"].astype(np.float64) * 110.57
        ti = pd.to_datetime(gz["t"], utc=True)
        t = np.asarray((ti.as_unit("ns") if hasattr(ti, "as_unit") else ti).asi8, np.float64) / 3.6e12 * (GOES_ISOLATED_KM / GOES_ISOLATED_H)
        tree = cKDTree(np.c_[x, y, t])
        counts = np.array([len(c) for c in tree.query_ball_point(np.c_[x, y, t], r=GOES_ISOLATED_KM)])
        self._goes_keep = counts > 1
        return self._goes_keep

    def goes_companion_t(self, gz):
        """For each GOES pixel, the time (ns) at which it stops looking isolated: the later of its own time and
        the earliest OTHER detection within GOES_ISOLATED_KM / GOES_ISOLATED_H (int64 max if none). Used
        for inputs, so a pixel is only trusted once its companion has actually been observed."""
        if getattr(self, "_goes_comp", None) is not None and len(self._goes_comp) == len(gz["t"]):
            return self._goes_comp
        from scipy.spatial import cKDTree
        lat0 = math.radians(float(np.mean(gz["lat"])))
        x = gz["lon"].astype(np.float64) * 111.32 * math.cos(lat0)
        y = gz["lat"].astype(np.float64) * 110.57
        ti = pd.to_datetime(gz["t"], utc=True)
        tn = np.asarray((ti.as_unit("ns") if hasattr(ti, "as_unit") else ti).asi8, np.int64)   # ns whatever the stored unit
        t = tn / 3.6e12 * (GOES_ISOLATED_KM / GOES_ISOLATED_H)
        tree = cKDTree(np.c_[x, y, t])
        big = np.iinfo(np.int64).max
        comp = np.full(len(tn), big, np.int64)
        for k, nb in enumerate(tree.query_ball_point(np.c_[x, y, t], r=GOES_ISOLATED_KM)):
            others = [j for j in nb if j != k]
            if others:
                comp[k] = max(tn[k], tn[others].min())
        self._goes_comp = comp
        return comp

    def rothermel(self, fb, gmv, u, v, slope_deg, aspect_deg, h, w):
        import torch
        from physics import rothermel as R
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
        if fb is None or not gmv:
            return dict(r_head=np.ones((h, w), np.float32), ecc=np.zeros((h, w), np.float32),
                        head_x=np.zeros((h, w), np.float32), head_y=np.ones((h, w), np.float32),
                        i_r=np.zeros((h, w), np.float32), sigma=np.ones((h, w), np.float32),
                        effm=np.full((h, w), 8.0, np.float32), ws=np.zeros((h, w), np.float32),
                        burnable=np.ones((h, w), bool), sfc=np.zeros((h, w), np.float32))
        tf = (gmv["tmmx"] - 273.15) * 9 / 5 + 32
        rh = np.clip(gmv["rmin"], 1, 100)
        emc = np.where(rh < 10, 0.03229 + 0.281073 * rh - 0.000578 * rh * tf,
                       np.where(rh < 50, 2.22749 + 0.160107 * rh - 0.01478 * tf,
                                21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * tf - 0.483199 * rh))
        m1, m10 = 1.03 * emc, 1.28 * emc
        if FM_PHYSICS and getattr(self, "_fm", None) is not None:      # hourly, slope-aware moisture: driest in the window
            m1, m10 = np.clip(self._fm["m1_min"], 1.0, 40.0), np.clip(self._fm["m10_min"], 1.0, 40.0)
        if u is not None:
            ws, wd = np.hypot(u, v), (np.degrees(np.arctan2(-u, -v)) + 360) % 360
            if ROTH_WIND != "vector" and getattr(self, "_spd_stats", None) is not None:
                ws = self._spd_stats[0] if ROTH_WIND == "mean" else self._spd_stats[1]
        else:
            ws, wd = gmv["vs"], gmv["th"]
        idx = R.fuel_index(torch.from_numpy(fb.astype(np.int64)))
        canopy = {}
        if CANOPY_WAF and getattr(self, "_canopy", None) is not None:
            canopy = dict(canopy_cover=t(self._canopy[0]), canopy_height_m=t(self._canopy[1]))
        out = R.spread(idx, t(m1 / 100), t(m10 / 100),
                       t(gmv["fm100"] / 100), torch.full((h, w), HERB), torch.full((h, w), WOODY), t(ws), t(wd),
                       t(np.tan(np.radians(slope_deg))), t(aspect_deg), **canopy)
        res = {k: out[k].numpy().astype(np.float32) for k in ("r_head", "ecc", "head_x", "head_y", "i_r", "sigma")}
        res["burnable"] = out["burnable"].numpy()
        res["effm"] = (m1 * np.ones((h, w))).astype(np.float32)   # fine dead fuel moisture, %
        res["ws"] = np.asarray(ws, np.float32) * np.ones((h, w), np.float32)
        if PHYSICS_V2:
            from physics import crown_spot as CS
            load = R._params(idx, torch.float64)[3].numpy()
            res["sfc"] = CS.surface_fuel_kg_m2(load).astype(np.float32)
        return res

    def crown_spot(self, phys, lf, sl, B0, hr, gmv, h, w):
        """Physics v2 fields (see physics/crown_spot.py). Adds phys['r_crown'] (crown-adjusted head
        spread, m/min) and phys['ember'] (ember-landing density per day); returns the 4 input channels."""
        from physics import crown_spot as CS
        r = phys["r_head"]
        if lf is not None:
            cbh = np.clip(lf["cbh"][sl], 0, 1000) / 10.0
            cbd = np.clip(lf["cbd"][sl], 0, 100) / 100.0
            cc = np.clip(lf["cc"][sl], 0, 100).astype(np.float32)
            p, _, r_tot = CS.crown_fire(r, cbh, cbd, cc, phys["ws"] * 3.6, phys["effm"],
                                        phys.get("sfc", np.zeros_like(r)), phys["burnable"])
        else:
            p, r_tot = np.zeros_like(r), r
        inten = CS.fireline_intensity(phys["i_r"], phys["sigma"], r_tot)
        if hr[6].max() > 0:
            u, v = float(np.mean(hr[0]) * 8), float(np.mean(hr[1]) * 8)
        elif gmv:
            th, vs = float(np.mean(gmv["th"])), float(np.mean(gmv["vs"]))
            u, v = -vs * math.sin(math.radians(th)), -vs * math.cos(math.radians(th))
        else:
            u = v = 0.0
        ember = CS.ember_field(B0, inten, p, phys["ws"], u, v, self.res_m)
        phys["r_crown"] = np.nan_to_num(r_tot).astype(np.float32)
        phys["ember"] = np.nan_to_num(ember).astype(np.float32)
        return np.stack([p, np.log1p(r_tot) / 3, np.log1p(10 * ember) / 3,
                         np.log1p(inten / 1000.0) / 3]).astype(np.float32)


def stack(f, groups):
    return np.concatenate([f[g] for g in groups])


VAL_YEARS, TEST_YEARS = {2023}, {2024, 2025}


def year_split(year):
    """Default: train 2014-2022, validate 2023, test 2024-2025 (newest fires = the real test)."""
    if year in TEST_YEARS:
        return "test"
    if year in VAL_YEARS:
        return "val"
    return "train"
