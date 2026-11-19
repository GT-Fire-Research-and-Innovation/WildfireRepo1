"""
Stage 0 of the hazard model: perimeters as interval-censored burn times.

A perimeter series only says that a cell burned somewhere between the last perimeter that did not cover it and the
first one that did. For a forecast that starts at perimeter i, every cell gets, in hours from the start:

  burned within the horizon:   (t_lo, t_hi]   t_lo = earliest possible time of the perimeter before first cover,
                                              t_hi = latest possible time of the first covering perimeter
  not burned by L:             right-censored at t_c = earliest time of L, the last perimeter inside the horizon
                               (it had not ignited by t_c), returned as t_lo = t_c, t_hi = inf
  burned at the start, or "other" (another fire), or no information: valid = False

Every later perimeter inside the horizon supervises the sample, not only the next one. A cell whose interval runs
past the horizon (the model's hazard stops there) contributes "had not ignited by t_lo" (= lo[L]), so windows longer than the
horizon are kept, not dropped; a sample whose next perimeter is already past the horizon carries no information. Perimeter times that are uncertain (database edit times, bare dates) widen the
bounds outward by TIME_SLACK, so a label never claims more precision than the stamp has.
"""
import numpy as np

HORIZON_H = 72.0
# (hours the true mapping time may be BEFORE the stamp, hours it may be AFTER), by time source (labels.npz
# time_sources). An edit-time stamp is typically the 04-06 upload of a 21-23 flight (data.EDIT_LAG_H = 8):
# allow up to 12 h earlier. A bare date: +-12 h. Unknown: +-8 h. ASSUMPTION, to be confirmed against the catalog.
TIME_SLACK = {"mapped": (0.0, 0.0), "edit": (12.0, 0.0), "date_only": (12.0, 12.0), "unknown": (8.0, 8.0)}


def perimeter_bounds(times_h, sources=None, slack=None):
    """Earliest and latest possible true time of every perimeter (hours, same origin as times_h)."""
    slack = TIME_SLACK if slack is None else slack
    t = np.asarray(times_h, np.float64)
    if sources is None:
        return t.copy(), t.copy()
    before = np.array([slack.get(str(s), slack["unknown"])[0] for s in sources])
    after = np.array([slack.get(str(s), slack["unknown"])[1] for s in sources])
    return t - before, t + after


def interval_labels(arrival, i, lo, hi, horizon_h=HORIZON_H, bad=None, exclude=None):
    """arrival: (h, w) int, index of the first perimeter covering each cell (-1 never).
    i: the start perimeter. lo, hi: earliest / latest true time of every perimeter, in hours from the START
    (perimeter_bounds(times - t_i)). bad: optional (h, w) bool, cells whose burn is not trusted (label QC):
    they keep only "not burned by t_lo". exclude: optional (h, w) bool, cells left out (another fire).
    Returns t_lo, t_hi (float32, t_hi = inf where right-censored) and valid (bool).

    Censoring is at ONE time per sample, the earliest time of L, the last perimeter that is surely inside the horizon:
    every cell not covered by L is "not burned by lo[L]". Censoring cells that burned just after L at lo[L] but
    cells that burned later at the horizon would depend on the burn time itself (informative censoring) and drags
    the late-bin hazard toward zero (tests/test_hazard.py found this)."""
    arrival = np.asarray(arrival)
    n = len(lo)
    lo = np.maximum(np.asarray(lo, np.float64), 0.0)          # nothing is known before the start
    hi = np.asarray(hi, np.float64)
    inside = [k for k in range(i + 1, n) if hi[k] <= horizon_h]
    L = max(inside) if inside else i
    t_end = lo[L] if L > i else 0.0                           # no perimeter inside the horizon: nothing to learn
    a = arrival.astype(np.int64)
    ev = (a > i) & (a <= L)
    rest = (a > L) | (a < 0)
    t_lo = np.zeros(a.shape, np.float64)
    t_hi = np.full(a.shape, np.inf)
    ai = np.clip(a, 1, n - 1)
    t_lo[ev] = lo[ai[ev] - 1]
    t_hi[ev] = np.maximum(hi[ai[ev]], t_lo[ev] + 0.5)         # never an empty interval
    t_lo[rest] = t_end
    # untrusted burns, and (with widened bounds) an interval that still reaches past the horizon: "not burned by t_lo"
    cens = rest | (ev & (t_hi > horizon_h))
    if bad is not None:
        cens |= ev & np.asarray(bad, bool)
    t_hi[cens] = np.inf
    valid = (ev | rest) & ~(np.isinf(t_hi) & (t_lo <= 0))     # "not burned by 0 h" says nothing
    if exclude is not None:
        valid &= ~np.asarray(exclude, bool)
    t_lo[~valid] = 0.0
    t_hi[~valid] = np.inf
    return t_lo.astype(np.float32), t_hi.astype(np.float32), valid
