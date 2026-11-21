"""fires/<id>/links.json + LINKS.md for every fire, and catalog/raw_index.csv, from the manifests."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from fah.channels.registry import ChannelRegistry
from fah.commands.settings import RunSettings
from fah.core.errors import ExitCode
from fah.core.fire import Fire
from fah.core.result import Status
from fah.hub.catalog import FireCatalog
from fah.hub.manifest import ManifestBook
from fah.hub.repo import HubRepo

log = logging.getLogger(__name__)


class LinksBuilder:
    """Every channel of every fire: its status, its files in the repo and the upstream URLs they came from. Uploaded
    in commits of links_chunk fires."""

    def __init__(self, settings: RunSettings, registry: ChannelRegistry | None = None) -> None:
        self.s = settings
        self.channels = (registry or ChannelRegistry()).names()

    @staticmethod
    def fire_links(fire: Fire, rows: Sequence[dict], repo: str | None, channels: Sequence[str]) -> dict:
        """links.json of one fire: the latest manifest row per channel (channels not run yet are 'pending')."""
        fid = fire.fire_id
        base = f"https://huggingface.co/datasets/{repo}" if repo else "."
        latest = {}
        for r in sorted(rows, key=lambda r: str(r.get("fetched_utc", ""))):
            latest[r["channel"]] = r
        chans = {}
        for ch in channels:
            r = latest.get(ch)
            if r is None:
                chans[ch] = {"status": Status.PENDING}
                continue
            e = {"status": r["status"], "fetched_utc": r.get("fetched_utc", "")}
            files = [x for x in str(r.get("files") or "").split(";") if x]
            if files:
                e["bytes"] = int(float(r.get("bytes") or 0))
                e["files"] = [{"name": x, "url": f"{base}/resolve/main/fires/{fid}/{ch}/{x}"} for x in files]
            src = [s.strip() for s in str(r.get("sources") or "").split(ManifestBook.SEP) if s.strip()]
            if src:
                e["sources"] = src
            if r.get("error"):
                e["note"] = r["error"]
            chans[ch] = e
        return {
            "fire_id": fid,
            "name": fire.name,
            "year": fire.year,
            "irwin_incident": fire.irwin(),
            "first_perimeter_utc": fire.first,
            "last_perimeter_utc": fire.last,
            "time_window_utc": [t.isoformat() for t in fire.span()],
            "archive": fire.archive,
            "region_lonlat": [round(v, 5) for v in fire.region()],
            "perimeters_box_lonlat": [round(v, 5) for v in fire.box],
            "ignition": dict(fire.ignition._asdict()) if fire.ignition else None,
            "folder": f"{base}/tree/main/fires/{fid}",
            "channels": chans,
        }

    @staticmethod
    def markdown(links: dict) -> str:
        """LINKS.md: the same as links.json, readable on the Hugging Face file page."""
        w, s, e, n = links["region_lonlat"]
        t0, t1 = links["time_window_utc"]
        ig = links.get("ignition")
        out = [
            f"# {links['name']} ({links['year']}) · `{links['fire_id']}`",
            "",
            f"IRWIN `{links['irwin_incident']}` · {links.get('archive') or ''} perimeters "
            f"{str(links['first_perimeter_utc'])[:16]} → "
            f"{str(links['last_perimeter_utc'])[:16]} UTC · files cover {t0[:16]} → {t1[:16]} UTC over region "
            f"W {w:.4f} S {s:.4f} E {e:.4f} N {n:.4f}"
            + (f" · ignition {ig['lat']:.4f}, {ig['lon']:.4f} at {ig['time_utc']} ({ig['method']})" if ig else ""),
            "",
            f"[This fire's folder]({links['folder']})",
            "",
            "| channel | status | files | upstream sources |",
            "|---|---|---|---|",
        ]
        for ch, c in links["channels"].items():
            files = ", ".join(f"[{x['name']}]({x['url']})" for x in c.get("files", [])) or "—"
            src = c.get("sources", [])
            srct = f"[link]({src[0]})" if len(src) == 1 else (f"{len(src)} links (below)" if src else "—")
            status = c["status"] + (f" ({c['note'][:80]})" if c.get("note") and c["status"] != Status.OK else "")
            out.append(f"| {ch} | {status} | {files} | {srct} |")
        many = [(ch, c["sources"]) for ch, c in links["channels"].items() if len(c.get("sources", [])) > 1]
        if many:
            out += ["", "## Upstream sources"]
            for ch, src in many:
                out += (
                    ["", f"<details><summary>{ch}: {len(src)} links</summary>", ""]
                    + [f"- {u}" for u in src]
                    + ["", "</details>"]
                )
        if any("{MAP_KEY}" in u for c in links["channels"].values() for u in c.get("sources", [])):
            out += [
                "",
                "FIRMS links need your own map key in place of `{MAP_KEY}` "
                "(free: https://firms.modaps.eosdis.nasa.gov/api/map_key/).",
            ]
        return "\n".join(out) + "\n"

    def execute(self) -> int:
        s = self.s
        hub = HubRepo.connect(s.hf_repo)
        work = Path(s.work)
        if not s.dry_run:
            hub.check_write()
        fires = FireCatalog(hub, work / "target", s.catalog_file).fires(s.archive, list(s.ids) if s.ids else None)
        fires = FireCatalog.with_extents(fires, FireCatalog.load_extents(hub, work / "target", s.extents_file))
        m = ManifestBook.read_local(Path(s.local_manifests)) if s.local_manifests else ManifestBook.read(hub)
        by_fire = {fid: g.to_dict("records") for fid, g in m.groupby("fire_id")} if len(m) else {}
        log.info(f"links for {len(fires)} fires from {len(m)} manifest rows")
        root = work / "links"
        shutil.rmtree(root, ignore_errors=True)
        index = []
        for k in range(0, len(fires), s.links_chunk):
            d = root / f"chunk{k // s.links_chunk:03d}"
            for f in fires[k : k + s.links_chunk]:
                links = self.fire_links(f, by_fire.get(f.fire_id, []), s.hf_repo, self.channels)
                fd = d / "fires" / f.fire_id
                fd.mkdir(parents=True, exist_ok=True)
                (fd / "links.json").write_text(json.dumps(links, indent=1))
                (fd / "LINKS.md").write_text(self.markdown(links))
                index += [
                    dict(
                        fire_id=f.fire_id,
                        name=f.name,
                        year=f.year,
                        channel=ch,
                        status=c["status"],
                        n_files=len(c.get("files", [])),
                        bytes=c.get("bytes", 0),
                    )
                    for ch, c in links["channels"].items()
                ]
            if k + s.links_chunk >= len(fires):
                (d / "catalog").mkdir(parents=True, exist_ok=True)
                pd.DataFrame(index).to_csv(d / "catalog" / "raw_index.csv", index=False)
            if not s.dry_run:
                hub.upload_folder(d, f"Per-fire links ({min(k + s.links_chunk, len(fires))}/{len(fires)} fires)")
                shutil.rmtree(d, ignore_errors=True)
        log.info(("[dry-run] kept " + str(root)) if s.dry_run else "links uploaded")
        return ExitCode.OK
