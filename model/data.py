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
}
V2_GROUPS = ["crownspot"]    # left out unless PHYSICS_V2 (keeps v1 models and checkpoints unchanged)
PHYSICS_V2 = False           # set by the hybrid2 / physics2 modes: compute crown fire and spotting


from collections import OrderedDict

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
        self.times = pd.to_datetime(lab["times"], utc=True)
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
    def sample(self, i, gap_h, win, hide=()):
        """Inputs (dict of group -> (C, h, w) float32), physics fields, target, mask, burning."""
        from scipy.ndimage import binary_dilation, distance_transform_edt
        r0, r1, c0, c1 = win
        sl = (slice(r0, r1), slice(c0, c1))
        h, w = r1 - r0, c1 - c0
        t_i, t_j = self.times[i], self.times[i + 1]
        f = {}
        B0 = self.burned(i)[sl]
        B1 = self.burned(i - 1)[sl] if i >= 1 else B0
        B2 = self.burned(i - 2)[sl] if i >= 2 else B1
        age1 = ((t_i - self.times[i - 1]).total_seconds() / 3600 if i >= 1 else 0.0) / 48
        age2 = ((t_i - self.times[i - 2]).total_seconds() / 3600 if i >= 2 else 0.0) / 48
        ck = self.cell_km
        dist = distance_transform_edt(~B0) * ck if B0.any() else np.full((h, w), 30.0)
        oth = self.other[sl]
        full = lambda v: np.full((h, w), v, np.float32)  # noqa: E731
        f["fire"] = np.stack([B0, B1, B2, full(age1), full(age2), full(gap_h / 48), np.clip(dist, 0, 15) / 15,
                              np.exp(-dist / 2), oth, full(math.log1p(B0.sum() * ck * ck) / 8),
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
                    af24 += v["af_day"][k][sl]
                    afn += v["af_night"][k][sl]
                    frp = np.maximum(frp, v["frp_max"][k][sl].astype(np.float32))
                if local_i - pd.Timedelta(days=1) <= dday <= local_j + pd.Timedelta(days=1):
                    evidence |= (v["af_day"][k] + v["af_night"][k]) > 0
        f["viirs"] = np.stack([np.minimum(af24, 4) / 4, np.minimum(afn, 4) / 4, np.log1p(frp) / 6]).astype(np.float32)

        # GOES: hours since a fire pixel was seen here (before t_i), detections and power in the last 6 h
        gz = self.z("goes")
        goes = np.zeros((4, h, w), np.float32)
        if gz is not None and "goes" not in hide and len(gz["t"]):
            t = pd.to_datetime(gz["t"], utc=True)
            before = (t <= t_i) & (t >= t_i - pd.Timedelta(hours=72))
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
            # GOES pixels during (t_i, t_j] count as label evidence too
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
        f["terrain"] = np.stack([(elev - elev.mean()) / 500, slope / 45, np.sin(asp) * slope / 45, np.cos(asp) * slope / 45,
                                 ter["tpi_500m"][sl].astype(np.float32) / 100, ter["tpi_2km"][sl].astype(np.float32) / 300,
                                 ter["rough"][sl].astype(np.float32) / 50])

        lf = self.z("landfire")
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
            gr, gc = wz["gm_row"][r0:r1].astype(int), wz["gm_col"][c0:c1].astype(int)
            names = [str(x) for x in wz["gm_vars"]]
            gm = wz["gm"]
            for n in GM:
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
        if rz is not None and "hrrr" not in hide:
            ts = pd.to_datetime(rz["times"], utc=True)
            sel = (ts >= t_i - pd.Timedelta(hours=1)) & (ts <= t_j)
            if sel.sum() == 0:
                sel = np.abs((ts - t_i).total_seconds()) == np.abs((ts - t_i).total_seconds()).min()
            F = rz["fields"][sel].astype(np.float32)          # (T, 4, hh, wh): gust, u10, v10, rh2
            idx = rz["idx"][sl]
            flat = F.reshape(F.shape[0], 4, -1)[:, :, idx]   # (T, 4, h, w)
            if np.isfinite(flat).any():
                gust, u, vv, rh = (np.nan_to_num(flat[:, j]) for j in range(4))
                spd = np.hypot(u, vv)
                u_mean, v_mean = u.mean(0), vv.mean(0)
                hr = np.stack([u_mean / 8, v_mean / 8, spd.mean(0) / 8, spd.max(0) / 12, gust.max(0) / 20,
                               rh.min(0) / 100, full(1.0)])
        f["hrrr"] = hr

        # suppression on perimeter i's burn day
        sz = self.z("suppression")
        sup = np.zeros((12, h, w), np.float32)
        if sz is not None and "suppression" not in hide:
            dates = list(sz["dates"])
            local_i = str((t_i.tz_convert(None) + pd.Timedelta(hours=self.g["transform"][2] / 15 - 6)).date())
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
        return f, phys, target, mask, B0

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
        if u is not None:
            ws, wd = np.hypot(u, v), (np.degrees(np.arctan2(-u, -v)) + 360) % 360
        else:
            ws, wd = gmv["vs"], gmv["th"]
        idx = R.fuel_index(torch.from_numpy(fb.astype(np.int64)))
        out = R.spread(idx, t(1.03 * emc / 100), t(1.28 * emc / 100),
                       t(gmv["fm100"] / 100), torch.full((h, w), HERB), torch.full((h, w), WOODY), t(ws), t(wd),
                       t(np.tan(np.radians(slope_deg))), t(aspect_deg))
        res = {k: out[k].numpy().astype(np.float32) for k in ("r_head", "ecc", "head_x", "head_y", "i_r", "sigma")}
        res["burnable"] = out["burnable"].numpy()
        res["effm"] = (1.03 * emc).astype(np.float32)          # fine dead fuel moisture, %
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
