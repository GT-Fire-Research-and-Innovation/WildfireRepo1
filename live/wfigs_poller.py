"""
Saves every version of every NIFC (WFIGS) fire perimeter. NIFC keeps one polygon per fire and edits it in
place, so an earlier outline is gone for good unless someone saved it while it was up. This is that archive.

  python live/wfigs_poller.py --out data/live            # poll every 5 minutes, forever
  python live/wfigs_poller.py --out data/live --once     # one poll (the GitHub job calls this in a loop)
  python live/wfigs_poller.py --out data/live --report   # how complete the archive is: gaps, versions, delay

How a poll works
  1. Sweep: list every perimeter in NIFC's Current and Year-to-Date services WITHOUT its outline (about
     3 MB for 8,000 perimeters). Each gets a signature from its edit time, mapping time, area and outline
     length, so any edit to the outline changes the signature.
  2. Download the outline only for signatures never seen before, with every attribute NIFC publishes.
  3. Save it as a new version unless the same outline + mapping time is already archived for that fire.
Because step 1 compares against everything seen so far (not against "the last few hours"), a poll that
comes late still picks up the newest outline of every fire that changed. What a late poll cannot get back
is an outline that was replaced again before we looked: only frequent polling avoids that, and polls.csv
records every poll so the gaps are known.

Archive layout (OUT/perimeters)
  <UniqueFireIdentifier>/<PolygonDateTime UTC>_<hash8>.geojson   one file per version, all NIFC attributes
  index.csv        one row per archived version (fire id, IRWIN id, name, times, acres, map method,
                   containment, file, first_seen = when we first saw it, an upper bound on when it went public)
  seen.csv         every signature seen so far (fire id, signature)
  polls/YYYY-MM.csv  one row per poll: time, ok, perimeters listed, changed, saved, seconds, error
The first poll of a new archive saves everything in Current plus everything mapped or edited in the last
30 days; older Year-to-Date perimeters are only noted, and saved from the moment they change.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
       "WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query")
YTD = URL.replace("WFIGS_Interagency_Perimeters_Current", "WFIGS_Interagency_Perimeters_YearToDate")
KEEP = ["attr_UniqueFireIdentifier", "attr_IrwinID", "attr_IncidentName", "poly_PolygonDateTime",
        "poly_DateCurrent", "poly_GISAcres", "poly_MapMethod", "poly_FeatureCategory",
        "attr_PercentContained", "attr_FireDiscoveryDateTime", "attr_IncidentSize"]
SIG_FIELDS = ("OBJECTID,attr_UniqueFireIdentifier,attr_IrwinID,poly_DateCurrent,poly_PolygonDateTime,"
              "Shape__Area,Shape__Length")
BASELINE_DAYS = 30      # first poll of a new archive: save what was mapped or edited this recently
CHUNK = 50              # outlines downloaded per request


def ms_to_iso(ms):
    return None if ms is None else datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(url, params, tries=4):
    """One request, retried. ArcGIS reports its own errors with HTTP 200 and an 'error' body: those are
    raised too, otherwise a failed query would read as 'no perimeters'."""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=180)
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise RuntimeError(f"ArcGIS error: {str(d['error'])[:200]}")
            return d
        except (requests.RequestException, ValueError, RuntimeError):
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def _paged(url, where, fields, geometry=True):
    """Every feature matching `where`. The service cuts a page short when it gets too large, and says so
    with exceededTransferLimit: that flag, not the page length, decides whether to ask for more."""
    feats, offset, size = [], 0, 1000 if geometry else 2000
    while True:
        p = {"where": where, "outFields": fields, "orderByFields": "OBJECTID", "resultOffset": offset,
             "resultRecordCount": size}
        p.update({"outSR": 4326, "f": "geojson"} if geometry else {"returnGeometry": "false", "f": "json"})
        d = _get(url, p)
        page = d.get("features", [])
        feats += page
        more = d.get("exceededTransferLimit") or (d.get("properties") or {}).get("exceededTransferLimit")
        if not page or not (more or len(page) >= size):
            return feats
        offset += len(page)


def fire_id(p):
    return p.get("attr_UniqueFireIdentifier") or (p.get("attr_IrwinID") or "unknown").strip("{}")


def signature(p):
    """Changes whenever NIFC edits a perimeter: edit time, mapping time, area, outline length."""
    s = "|".join(str(p.get(k)) for k in ("poly_DateCurrent", "poly_PolygonDateTime", "Shape__Area", "Shape__Length"))
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def fetch(days=90, fields=None):
    """The newest perimeter of every fire that is still going (used by the live forecast): NIFC's 'Current'
    service plus the fires it drops. 'Current' only lists incidents whose record NIFC still treats as
    current, so active fires with a stale record fall out of it (2026-11-11: Ptarmigan 26k ac, Border 2,
    Luna, Garda Falls, ... were missing while Watch Duty showed them). The Year-to-Date service still has
    their perimeters: we add every wildfire there that is not declared out, not 100% contained, mapped in
    the last `days` days."""
    fields = fields or ",".join(KEEP + ["attr_IncidentTypeCategory", "attr_POOState"])
    since = (datetime.now(timezone.utc) - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    cur = _paged(URL, "1=1", fields)
    ytd = _paged(YTD, f"attr_IncidentTypeCategory='WF' AND attr_FireOutDateTime IS NULL AND "
                      f"(attr_PercentContained IS NULL OR attr_PercentContained < 100) AND "
                      f"poly_PolygonDateTime >= TIMESTAMP '{since}'", fields)
    best = {}
    for src, feats in (("current", cur), ("year_to_date", ytd)):
        for f in feats:
            p = f["properties"]
            p["feed"] = src
            old = best.get(fire_id(p))
            if old is None or (p.get("poly_PolygonDateTime") or 0) > (old["properties"].get("poly_PolygonDateTime") or 0):
                best[fire_id(p)] = f
    return list(best.values())


def poll(out):
    """One poll. Returns counts: perimeters listed per service, changed (new signatures), saved (new versions)."""
    arch = out / "perimeters"
    arch.mkdir(parents=True, exist_ok=True)
    idx_path, seen_path = arch / "index.csv", arch / "seen.csv"
    idx = pd.read_csv(idx_path) if idx_path.exists() else pd.DataFrame(columns=["fire_id", "hash"])
    have = set(zip(idx["fire_id"].astype(str), idx["hash"].astype(str)))
    first = not seen_path.exists()
    seen = set() if first else set(map(tuple, pd.read_csv(seen_path, dtype=str)[["fire_id", "sig"]].values))
    now_dt = datetime.now(timezone.utc)
    now = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    recent_ms = (now_dt.timestamp() - BASELINE_DAYS * 86400) * 1000

    # 1. sweep both services without outlines; note which signatures are new
    todo, noted, st = {}, set(), {}
    for feed, url in (("current", URL), ("year_to_date", YTD)):
        rows = [f["attributes"] for f in _paged(url, "1=1", SIG_FIELDS, geometry=False)]
        st[feed] = len(rows)
        for a in rows:
            key = (fire_id(a), signature(a))
            if key in seen or key in todo or key in noted:
                continue
            old = max(a.get("poly_DateCurrent") or 0, a.get("poly_PolygonDateTime") or 0) < recent_ms
            if first and feed != "current" and old:
                noted.add(key)                   # a new archive starts here: saved once it changes
            else:
                todo[key] = (feed, a["OBJECTID"])
    st["changed"] = len(todo)

    # 2. download the changed outlines; 3. save the ones not archived yet
    rows = []
    for feed, url in (("current", URL), ("year_to_date", YTD)):
        ids = [oid for f_, oid in todo.values() if f_ == feed]
        for k in range(0, len(ids), CHUNK):
            for f in _paged(url, "OBJECTID IN (%s)" % ",".join(map(str, ids[k:k + CHUNK])), "*"):
                p = f["properties"]
                p["feed"] = feed
                fid = fire_id(p)
                noted.add((fid, signature(p)))   # by what came back, so an edit made mid-poll is picked up next time
                if not f.get("geometry"):
                    continue
                geom = json.dumps(f["geometry"], sort_keys=True)
                h = hashlib.sha1((geom + str(p.get("poly_PolygonDateTime"))).encode()).hexdigest()[:8]
                if (fid, h) in have:
                    continue
                have.add((fid, h))
                t = ms_to_iso(p.get("poly_PolygonDateTime")) or now
                d = arch / fid
                d.mkdir(exist_ok=True)
                fn = d / f"{t.replace(':', '')}_{h}.geojson"
                fn.write_text(json.dumps({"type": "FeatureCollection", "features": [f]}))
                rows.append({"fire_id": fid, "irwin_id": p.get("attr_IrwinID"), "name": p.get("attr_IncidentName"),
                             "polygon_time": t, "date_current": ms_to_iso(p.get("poly_DateCurrent")),
                             "discovered": ms_to_iso(p.get("attr_FireDiscoveryDateTime")), "acres": p.get("poly_GISAcres"),
                             "map_method": p.get("poly_MapMethod"), "category": p.get("poly_FeatureCategory"),
                             "pct_contained": p.get("attr_PercentContained"), "file": str(fn.relative_to(arch)),
                             "type": p.get("attr_IncidentTypeCategory"), "state": p.get("attr_POOState"), "feed": feed,
                             "hash": h, "first_seen": now})
    if rows:
        idx = pd.concat([idx, pd.DataFrame(rows)], ignore_index=True)
        idx.to_csv(idx_path, index=False)
    noted -= seen
    if noted or first:                           # last: a poll that dies before this is simply repeated
        pd.DataFrame(sorted(noted), columns=["fire_id", "sig"]).to_csv(seen_path, mode="a", header=first, index=False)
    st.update(saved=len(rows), fires=int(idx["fire_id"].nunique()))
    return st


def poll_logged(out):
    """poll(), with one row added to polls/YYYY-MM.csv whether it worked or not."""
    t0 = datetime.now(timezone.utc)
    row = {"time": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "ok": 1, "current": "", "year_to_date": "", "changed": "",
           "saved": "", "seconds": "", "error": ""}
    try:
        st = poll(out)
        row.update({k: st[k] for k in ("current", "year_to_date", "changed", "saved")})
        msg = f"{st['saved']} new perimeter versions ({st['changed']} changed); {st['fires']} fires archived"
    except Exception as ex:  # noqa: BLE001
        row.update(ok=0, error=f"{type(ex).__name__}: {ex}"[:200].replace("\n", " "))
        msg = f"poll failed: {row['error']}"
    row["seconds"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 1)
    log = out / "perimeters" / "polls" / f"{t0:%Y-%m}.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"{t0:%Y-%m-%d %H:%M UTC}: {msg}", flush=True)
    return bool(row["ok"])


def report(out, gap_min=20):
    """How complete the archive is. An outline that NIFC replaced between two polls is lost, so the time
    between successful polls is the number that matters."""
    arch = out / "perimeters"
    logs = sorted((arch / "polls").glob("*.csv"))
    if logs:
        p = pd.concat([pd.read_csv(f) for f in logs], ignore_index=True)
        p["time"] = pd.to_datetime(p["time"], utc=True)
        ok = p[p["ok"] == 1].sort_values("time")
        gap = ok["time"].diff().dt.total_seconds().div(60).dropna()
        span = (ok["time"].max() - ok["time"].min()).total_seconds() / 60 if len(ok) > 1 else 0
        print(f"polls: {len(ok)} ok, {int((p['ok'] == 0).sum())} failed, {ok['time'].min():%Y-%m-%d %H:%M} to "
              f"{ok['time'].max():%Y-%m-%d %H:%M} UTC")
        if len(gap):
            late = gap[gap > gap_min]
            print(f"time between polls: median {gap.median():.1f} min, longest {gap.max() / 60:.1f} h; "
                  f"{100 * (1 - late.sum() / span):.1f}% of the period polled at least every {gap_min} min")
            for i in late.sort_values(ascending=False).index[:10]:
                print(f"  gap of {gap[i] / 60:.1f} h ending {ok['time'][i]:%Y-%m-%d %H:%M} UTC")
    else:
        print("polls: no log yet")
    idx = pd.read_csv(arch / "index.csv")
    n = idx.groupby("fire_id").size()
    print(f"archive: {len(idx)} perimeter versions of {len(n)} fires; {int((n >= 2).sum())} fires with 2+ versions, "
          f"{int((n >= 5).sum())} with 5+, most {int(n.max())}")
    late = (pd.to_datetime(idx["first_seen"], utc=True) - pd.to_datetime(idx["date_current"], utc=True)).dt.total_seconds() / 60
    late = late[(late >= 0) & (late < 24 * 60)]              # baseline rows were edited long before we started
    if len(late):
        print(f"delay from NIFC's edit to our copy (versions caught within a day, n={len(late)}): "
              f"median {late.median():.0f} min, 90% within {late.quantile(0.9):.0f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/live")
    ap.add_argument("--interval", type=int, default=300, help="seconds between polls (NIFC refreshes about every 5 min)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true", help="print how complete the archive is, then stop")
    a = ap.parse_args()
    out = Path(a.out)
    if a.report:
        return report(out)
    while True:
        t0 = time.time()
        ok = poll_logged(out)
        if a.once:
            return 0 if ok else 1
        time.sleep(max(5.0, a.interval - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
