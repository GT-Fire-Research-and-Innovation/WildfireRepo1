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
bounds by slack() on the side the stamp is uncertain on, so a label never claims more precision than it has.
"""
import numpy as np

HORIZON_H = 72.0
# How far a perimeter's true mapping time can be from its stamp, by time source (labels.npz time_sources). Settings
# (train_hazard.py --mapped-slack-h / --edit-slack-h); the edit value should come from pipeline/measure_edit_delay.py.
#   mapped     the mapping time itself: +-MAPPED_SLACK_H (rounding; 0 by default)
#   edit       a database edit time: the map was made AT OR BEFORE it, by an unknown amount. Only the earlier side
#              moves: [stamp - EDIT_SLACK_H, stamp]. Widening the later side would tell the model a cell might have
#              burned after a map already showed it burned.
#   date_only  mapped sometime that local day: [local midnight, next local midnight] (Fire.time_bounds handles the
#              two ways a bare date was stored; on its own the stamp is taken as local midnight)
#   unknown    no record of the source: earlier side as an edit time, later side UNKNOWN_AFTER_H
MAPPED_SLACK_H = 0.0
EDIT_SLACK_H = 16.0          # until measured: too little slack teaches wrong timing, too much costs a little precision
UNKNOWN_AFTER_H = 8.0


def slack(source, day_extra_h=0.0):
    """(hours the true time may be BEFORE the stamp, hours it may be AFTER) for one perimeter."""
    s = str(source)
    if s == "mapped":
        return MAPPED_SLACK_H, MAPPED_SLACK_H
    if s == "edit":
        return EDIT_SLACK_H, 0.0
    if s == "date_only":
        return 0.0, 24.0 + day_extra_h
    return EDIT_SLACK_H, UNKNOWN_AFTER_H


def perimeter_bounds(times_h, sources=None, day_extra_h=None):
    """Earliest and latest possible true time of every perimeter (hours, same origin as times_h).
    day_extra_h: per perimeter, hours to add to the end of a bare date's day (a date stored as 00:00 UTC whose local
    day ends later than 24 h after it)."""
    t = np.asarray(times_h, np.float64)
    if sources is None:
        return t.copy(), t.copy()
    extra = np.zeros(len(t)) if day_extra_h is None else np.asarray(day_extra_h, np.float64)
    sl = np.array([slack(s, e) for s, e in zip(sources, extra)]).reshape(len(t), 2)
    return t - sl[:, 0], t + sl[:, 1]


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
