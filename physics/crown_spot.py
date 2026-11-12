"""
Crown fire and spotting on top of Rothermel surface spread (NumPy, computed with the inputs).

Crown fire (published equations, as implemented in the firebehavioR package, cfis()):
  initiation probability   Cruz, Alexander & Wakimoto (2004), logistic in 10-m open wind (km/h),
                           fuel stratum gap ~ canopy base height (m), fine fuel moisture (%) and
                           surface fuel consumption class
  active crown spread      Cruz, Alexander & Wakimoto (2005):  CROS_A = 11.02 U10^0.90 CBD^0.19 e^(-0.17 EFFM)
  passive crown spread     CROS_P = CROS_A exp(-CROS_A CBD / 3);  active when CAC = CROS_A CBD / 3 >= 1
  The expected spread rate blends surface and crown spread by the crown probability.

Fireline intensity (Byram): I = H_A R with H_A = I_R * tau_r, tau_r = 384 / sigma minutes (Anderson 1969).

Spotting: embers launched from the burning front land downwind. Landing distance follows a
lognormal whose median grows like sqrt(intensity) x wind speed (the empirical form used by ELMFIRE:
mean distance = A I^0.5 U^1.0, bounded to 10-2000 m). Published coefficients vary widely between
fuels, so A is a starting value here and the hybrid model learns how strongly embers ignite.
Sources: torching (crown probability) throws many embers, intense surface fire a few.
The result is an ember-landing density map (relative units, per day).
"""
import math

import numpy as np

SPOT_A = 25.0                 # m per (MW/m)^0.5 per (m/s): ~500 m median at 4 MW/m and 10 m/s
SPOT_SIGMA_LN = 0.6           # lognormal spread of landing distance
SPOT_MIN, SPOT_MAX = 10.0, 2000.0
SPOT_CLASSES = (150.0, 450.0, 1300.0)   # median distances the sources are binned into (m)
CROSS_FRAC = 0.25             # crosswind spread = 25 % of the downwind distance
SURFACE_EMBERS = 0.1          # ember yield of an intense surface fire relative to a torching canopy
FMC = 100.0                   # foliar moisture content (%)


def fireline_intensity(i_r_kw_m2, sigma_per_ft, r_m_min):
    """Byram fireline intensity in kW/m."""
    tau_s = 384.0 / np.maximum(sigma_per_ft, 1.0) * 60.0
    return i_r_kw_m2 * tau_s * r_m_min / 60.0


def surface_fuel_kg_m2(load_lb_ft2):
    """Fuel likely consumed by a surface fire: fine dead + half the 100-h + live, kg/m2."""
    l1, l10, l100, lh, lw = load_lb_ft2
    return (l1 + l10 + 0.5 * l100 + lh + lw) * 4.8824


def crown_fire(r_surface, cbh_m, cbd, cc_pct, u10_kmh, effm_pct, sfc_kg_m2, burnable):
    """Returns crown probability, crown spread rate (m/min) and the blended expected spread rate."""
    u = np.clip(u10_kmh, 0.0, 80.0)
    effm = np.clip(effm_pct, 2.0, 20.0)
    sfc_term = np.where(sfc_kg_m2 < 1, -4.613, np.where(sfc_kg_m2 > 2, 0.0, -1.856))
    g = 4.236 + 0.357 * u - 0.71 * np.clip(cbh_m, 0, 12) - 0.331 * effm + sfc_term
    has_canopy = (cc_pct >= 10) & (cbd > 0.01) & burnable
    p = np.where(has_canopy, 1.0 / (1.0 + np.exp(-g)), 0.0)
    cbd_c = np.clip(cbd, 0.01, 1.0)
    cros_a = 11.02 * np.maximum(u, 0.1) ** 0.90 * cbd_c ** 0.19 * np.exp(-0.17 * effm)
    cac = cros_a * cbd_c / 3.0
    cros = np.where(cac >= 1, cros_a, cros_a * np.exp(-cac))
    cros = np.where(has_canopy, cros, 0.0)
    r_total = (1 - p) * r_surface + p * np.maximum(r_surface, cros)
    return p.astype(np.float32), cros.astype(np.float32), r_total.astype(np.float32)


def _kernel(median_m, u, v, cell_m):
    """Downwind landing kernel (sums to 1) for one median distance; wind (u east, v north)."""
    spd = math.hypot(u, v)
    reach = median_m * math.exp(2.5 * SPOT_SIGMA_LN)
    n = int(min(max(3, math.ceil(reach / cell_m)), 400))
    ys, xs = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
    east, north = xs * cell_m, -ys * cell_m          # rows run south
    if spd < 0.5:                                     # calm: embers fall all around, short range
        r = np.hypot(east, north)
        k = np.exp(-0.5 * (np.log(np.maximum(r, 1.0) / (0.3 * median_m)) / SPOT_SIGMA_LN) ** 2) / np.maximum(r, 1.0) ** 2
    else:
        ux, uy = u / spd, v / spd
        along = east * ux + north * uy
        cross = -east * uy + north * ux
        a = np.maximum(along, 1.0)
        k = np.exp(-0.5 * (np.log(a / median_m) / SPOT_SIGMA_LN) ** 2) / a
        k *= np.exp(-0.5 * (cross / (CROSS_FRAC * a)) ** 2) / (CROSS_FRAC * a)
        k[along <= 0] = 0.0
    k[n, n] = 0.0
    s = k.sum()
    return (k / s).astype(np.float32) if s > 0 else k.astype(np.float32)


def _fftconv(img, ker):
    from scipy.signal import fftconvolve
    return fftconvolve(img, ker, mode="same")


def ember_field(burning, intensity_kw_m, p_crown, wind_ms, u_mean, v_mean, cell_m):
    """Ember-landing density (relative units per day) from the burning front of `burning`.
    wind_ms: per-cell speed; u_mean, v_mean: window-mean wind (direction of the plume)."""
    from scipy.ndimage import binary_erosion
    out = np.zeros(burning.shape, np.float32)
    if not burning.any():
        return out
    front = burning & ~binary_erosion(burning, iterations=1, border_value=1)   # the window's edge is not a fire front
    strength = (p_crown + SURFACE_EMBERS * (1 - np.exp(-intensity_kw_m / 4000.0))) * front
    if strength.max() <= 0:
        return out
    dist = np.clip(SPOT_A * np.sqrt(np.maximum(intensity_kw_m, 0) / 1000.0) * np.maximum(wind_ms, 0.5), SPOT_MIN, SPOT_MAX)
    # each source's embers go to the two nearest distance classes (weights in log distance)
    logc = np.log(np.array(SPOT_CLASSES))
    ld = np.clip(np.log(dist), logc[0], logc[-1])
    for j, med in enumerate(SPOT_CLASSES):
        if j == 0:
            w = np.clip((logc[1] - ld) / (logc[1] - logc[0]), 0, 1)
        elif j == len(SPOT_CLASSES) - 1:
            w = np.clip((ld - logc[-2]) / (logc[-1] - logc[-2]), 0, 1)
        else:
            w = np.where(ld <= logc[j], np.clip((ld - logc[j - 1]) / (logc[j] - logc[j - 1]), 0, 1),
                         np.clip((logc[j + 1] - ld) / (logc[j + 1] - logc[j]), 0, 1))
        src = strength * w
        if src.max() <= 0:
            continue
        out += np.maximum(_fftconv(src, _kernel(med, u_mean, v_mean, cell_m)), 0).astype(np.float32)
    out[burning] = 0.0
    return out
