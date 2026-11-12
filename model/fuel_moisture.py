"""
Fine dead fuel moisture that follows the weather hour by hour, per 150 m cell.

The physics used to get ONE fine-fuel moisture per day, from the day's highest temperature and lowest
humidity on a 4 km grid: no overnight recovery, no rain, the same value on a north and a south slope.
This module steps through the weather between observations (every 3 h from the HRRR analyses, hourly
from a live forecast) and keeps, for every cell:

  1-h fuels  (grass, litter)  : follow the fuel-level humidity within about an hour
  10-h fuels (small branches) : lag about ten hours behind

Each step:
  1. air temperature and humidity of the weather cell are moved to the cell's own elevation
     (6.5 C per km; the amount of water vapour in the air is kept),
  2. sunshine reaching the fuel is worked out from the sun's position, the cell's slope and aspect,
     cloud (HRRR downward shortwave when stored, else clear sky) and canopy shade,
  3. sun-warmed fuel is hotter and effectively drier than the air around it (the BEHAVE fuel-level
     correction of Rothermel et al. 1986, after Byram and Jemison 1943),
  4. the moisture the fuel is heading for (equilibrium moisture content, the NFDRS equations of
     Simard 1968) is computed from that fuel-level temperature and humidity,
  5. each fuel class moves toward it at its own speed; rain wets them.

The rain response and the canopy-shade factor are simple approximations (stated where they are set),
not fitted values.
"""
import math

import numpy as np

TAU_1H, TAU_10H = 1.0, 10.0        # response times (h)
RAIN_MIN_MM = 0.25                 # rain below this in a step is ignored
SAT_1H, SAT_10H = 30.0, 35.0       # moisture (%) fine fuels reach when rained on
RAIN_E_FOLD_MM = 6.0               # 10-h fuels: this much rain moves them 63 % of the way to saturation
DIRECT_SHARE = 0.8                 # share of sunshine that is direct beam (the rest is not changed by slope)
CANOPY_SHADE = 0.8                 # a closed canopy blocks this share of the sunshine reaching surface fuels
LAPSE_C_PER_M = 0.0065
W_TO_CAL = 1.0 / 697.8             # W/m2 -> cal/cm2/min
MAX_HEATING_F = 30.0               # cap on fuel-above-air temperature (deg F)
SPINUP_H = 72.0                    # history used to bring the 10-h fuels to the right state
N_CH = 15


def emc(temp_f, rh):
    """Equilibrium moisture content (%), NFDRS (Simard 1968). temp in deg F, rh in %."""
    rh = np.clip(rh, 1.0, 100.0)
    return np.where(rh < 10, 0.03229 + 0.281073 * rh - 0.000578 * rh * temp_f,
                    np.where(rh < 50, 2.22749 + 0.160107 * rh - 0.014784 * temp_f,
                             21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * temp_f - 0.483199 * rh))


def sun(lat_deg, lon_deg, t):
    """(cos zenith, azimuth in degrees clockwise from north) of the sun at UTC time t (pandas Timestamp)."""
    doy = t.dayofyear
    hours = t.hour + t.minute / 60.0
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365.0)
    ha = math.radians(15.0 * (hours + lon_deg / 15.0 - 12.0))
    lat = math.radians(lat_deg)
    cz = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat))) + 180.0
    return cz, az % 360.0


def sunshine_on_slope(ghi, cos_zen, sun_az, slope_deg, aspect_deg, canopy_cover):
    """Sunshine (W/m2) reaching surface fuel on a slope, from the sunshine on flat open ground (ghi)."""
    if cos_zen <= 0.02:
        return np.zeros(np.broadcast(slope_deg, aspect_deg).shape, np.float32)
    b, zen = np.radians(slope_deg), math.acos(min(cos_zen, 1.0))
    cos_inc = np.cos(b) * cos_zen + np.sin(b) * math.sin(zen) * np.cos(np.radians(sun_az - aspect_deg))
    beam = np.clip(cos_inc, 0, None) / max(cos_zen, 0.1)
    on_slope = ghi * (DIRECT_SHARE * np.clip(beam, 0, 4.0) + (1 - DIRECT_SHARE))
    return (on_slope * (1 - CANOPY_SHADE * np.clip(canopy_cover, 0, 100) / 100.0)).astype(np.float32)


def to_elevation(temp_c, rh, dz_m):
    """Air temperature and humidity moved dz_m metres up (+) or down (-), keeping the water vapour."""
    es = lambda t: 6.112 * np.exp(17.67 * t / (t + 243.5))  # noqa: E731
    t2 = temp_c - LAPSE_C_PER_M * dz_m
    return t2, np.clip(rh * es(temp_c) / es(t2), 1.0, 100.0)


def fuel_level(temp_c, rh, sunshine, wind_ms):
    """Temperature (deg F) and humidity (%) at the fuel: sun-warmed fuel is hotter and drier than the air.
    Tf - Ta = I / (0.015 U + 0.026), Hf = Ha exp(-0.033 (Tf - Ta)); I in cal/cm2/min, U in mi/h at fuel level."""
    ta = temp_c * 9 / 5 + 32
    u = np.clip(wind_ms, 0, None) * 0.4 * 2.237            # 10 m wind -> fuel level, mi/h
    heat = np.clip(sunshine * W_TO_CAL / (0.015 * u + 0.026), 0, MAX_HEATING_F)
    return ta + heat, np.clip(rh * np.exp(-0.033 * heat), 1.0, 100.0)


def integrate(steps, t_i, t_j, lat, lon, elev_dz, slope_deg, aspect_deg, canopy_cover):
    """steps: time-ordered list of dicts {t, temp_c, rh, wind_ms, rain_mm, ghi}, each field a scalar or an
    array broadcastable to the cell grid; ghi may be None / NaN (clear sky is used). Steps before t_i are
    spin-up, steps in (t_i, t_j] are the forecast window. Returns (N_CH, h, w) float32 inputs and a dict
    of the moistures the physics can use, or None when there is no step at or before t_i."""
    shape = np.broadcast(elev_dz, slope_deg, aspect_deg, canopy_cover).shape
    if not steps or steps[0]["t"] > t_i:
        return None, None
    m1 = m10 = None
    last_t = None
    rain_before = np.zeros(shape, np.float32)
    rain_win = np.zeros(shape, np.float32)
    since_rain = np.full(shape, SPINUP_H, np.float32)
    at_start = None
    seg_sum, seg_n = np.zeros((4,) + shape, np.float32), np.zeros(4, np.int32)
    w_min1 = w_max1 = w_sum1 = w_min10 = sun_sum = None
    n_win = 0
    gap_h = max((t_j - t_i).total_seconds() / 3600, 1e-6)
    for s in steps:
        t = s["t"]
        if t > t_j:
            break
        if "load" in s:                                   # fields are read only when the step is reached
            s = {**s["load"](), "t": t}
        cz, az = sun(lat, lon, t)
        ghi = s.get("ghi")
        clear = 1000.0 * max(cz, 0.0) ** 1.15
        ghi = np.where(np.isfinite(ghi), ghi, clear) if ghi is not None else clear
        shine = sunshine_on_slope(ghi, cz, az, slope_deg, aspect_deg, canopy_cover)
        ta, rh = to_elevation(np.asarray(s["temp_c"], np.float32), np.asarray(s["rh"], np.float32), elev_dz)
        tf, hf = fuel_level(ta, rh, shine, s["wind_ms"])
        e = emc(tf, hf).astype(np.float32)
        rain = np.nan_to_num(np.asarray(s.get("rain_mm", 0.0), np.float32)) * np.ones(shape, np.float32)
        if m1 is None:
            m1, m10 = e.copy() * np.ones(shape, np.float32), 1.25 * e * np.ones(shape, np.float32)
        else:
            dt = max((t - last_t).total_seconds() / 3600, 0.0)
            m1 = e + (m1 - e) * math.exp(-dt / TAU_1H)
            m10 = e + (m10 - e) * math.exp(-dt / TAU_10H)
            since_rain = np.minimum(since_rain + dt, SPINUP_H)
        wet = rain >= RAIN_MIN_MM
        if wet.any():
            m1 = np.where(wet, np.maximum(m1, SAT_1H), m1)
            m10 = np.where(wet, m10 + (SAT_10H - m10).clip(0) * (1 - np.exp(-rain / RAIN_E_FOLD_MM)), m10)
            since_rain = np.where(rain >= 1.0, 0.0, since_rain)
        last_t = t
        if t <= t_i:
            rain_before += rain
            at_start = (m1.copy(), m10.copy(), since_rain.copy())
        else:
            rain_win += rain
            if n_win == 0:
                w_min1, w_max1, w_sum1, w_min10, sun_sum = m1.copy(), m1.copy(), m1.copy(), m10.copy(), shine.astype(np.float32).copy()
            else:
                w_min1, w_max1 = np.minimum(w_min1, m1), np.maximum(w_max1, m1)
                w_sum1 += m1
                w_min10 = np.minimum(w_min10, m10)
                sun_sum += shine
            n_win += 1
            k = min(int((t - t_i).total_seconds() / 3600 / gap_h * 4), 3)
            seg_sum[k] += m1
            seg_n[k] += 1
    if at_start is None:
        return None, None
    s1, s10, dry_h = at_start
    if n_win == 0:                                       # no weather step inside the window: hold the start state
        w_min1 = w_max1 = w_mean1 = s1
        w_min10, sun_mean = s10, np.zeros(shape, np.float32)
    else:
        w_mean1, sun_mean = w_sum1 / n_win, sun_sum / n_win
    segs = [seg_sum[k] / seg_n[k] if seg_n[k] else w_mean1 for k in range(4)]
    out = np.stack([s1 / 30, w_min1 / 30, w_mean1 / 30, w_max1 / 30, s10 / 30, w_min10 / 30,
                    segs[0] / 30, segs[1] / 30, segs[2] / 30, segs[3] / 30,
                    np.log1p(rain_before) / 3, np.log1p(rain_win) / 3, dry_h / SPINUP_H, sun_mean / 1000,
                    np.ones(shape, np.float32)]).astype(np.float32)
    return out, dict(m1_start=s1, m1_min=w_min1, m1_mean=w_mean1, m10_min=w_min10)
