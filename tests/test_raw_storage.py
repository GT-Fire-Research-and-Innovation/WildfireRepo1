"""Raw storage layout: gzipped GeoJSON with a fixed header, a fire fetched again replacing its older files on
the Hub, and --repack gzipping plain .geojson already in the repo. Offline."""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import raw_channels as RC  # noqa: E402
import raw_grab as RG  # noqa: E402

FEATS = [{"type": "Feature", "properties": {"name": "Camp", "acres": 153336}, "geometry": None}]


def test_geojson_is_gzipped_with_the_same_bytes_every_time(tmp_path):
    a, b = tmp_path / "a.geojson.gz", tmp_path / "b.geojson.gz"
    RC._write_geojson(a, FEATS)
    RC._write_geojson(b, FEATS)
    assert a.read_bytes() == b.read_bytes() and a.read_bytes()[:2] == b"\x1f\x8b"
    assert json.load(gzip.open(a)) == {"type": "FeatureCollection", "features": FEATS}
    plain = tmp_path / "c.geojson"
    plain.write_text(json.dumps({"type": "FeatureCollection", "features": FEATS}))
    RC.gzip_copy(plain, tmp_path / "c.geojson.gz")
    assert gzip.open(tmp_path / "c.geojson.gz").read() == plain.read_bytes()


def test_new_layout_names_are_required_for_a_fire_to_count_as_done():
    assert RC.expected_files("perimeters") == {"records.geojson.gz"} and RC.expected_files("viirs") == {"viirs.nc"}
    assert RC.expected_files("suppression") == {"event_lines.geojson.gz", "event_polygons.geojson.gz"}
    assert RC.expected_files("terrain") is None


class Api:
    def __init__(self):
        self.calls = []

    def upload_folder(self, repo_id, folder_path, repo_type, commit_message, delete_patterns=None):
        self.calls.append(delete_patterns)


def staged(tmp, fid, name="history.geojson.gz"):
    d = tmp / "stage" / fid / "history"
    d.mkdir(parents=True)
    (d / name).write_bytes(b"x")
    return d


def test_a_new_result_replaces_the_fires_older_files_but_an_error_keeps_them(tmp_path):
    api = Api()
    b = RG.Batcher(api, "me/raw", "history", tmp_path, batch_fires=10, min_free_gb=0)
    f = {"name": "n", "year": 2018, "bounds": [-121.6, 39.7, -121.4, 39.9]}
    b.add({**f, "fire_id": "A"}, "ok", staged(tmp_path, "A"))
    b.add({**f, "fire_id": "B"}, "empty")
    b.add({**f, "fire_id": "C"}, "error", error="timeout")
    b.add({**f, "fire_id": "D"}, "n/a", error="before 2018")
    b.upload()
    assert api.calls == [["fires/A/history/*", "fires/B/history/*"]]
    b.add({**f, "fire_id": "E"}, "error", error="timeout")
    b.upload()
    assert api.calls[-1] is None                                    # nothing replaced: no listing, no deletes


def test_repack_gzips_plain_geojson_in_place_and_keeps_the_old_rows_labels(tmp_path, monkeypatch):
    import huggingface_hub
    hub = tmp_path / "hub"
    for fid in ("A", "B"):
        p = hub / "fires" / fid / "perimeters" / "records.geojson"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"type": "FeatureCollection", "features": FEATS, "fid": fid}))
    rows = [dict(fire_id="A", name="Camp", year="2018", channel="perimeters", status="ok", files="records.geojson",
                 sources="https://nifc/a | https://nifc/b", fetched_utc="2026-11-16T10:00:00+00:00", window="", region=""),
            dict(fire_id="B", name="Old", year="2018", channel="perimeters", status="ok", files="records.geojson.gz",
                 sources="", fetched_utc="2026-11-16T10:00:00+00:00", window="", region=""),
            dict(fire_id="C", name="Err", year="2018", channel="perimeters", status="error", files="",
                 sources="", fetched_utc="2026-11-16T10:00:00+00:00", window="", region="")]
    monkeypatch.setattr(RG, "read_manifests", lambda api, repo, ch=None: pd.DataFrame(rows))
    got = []

    def download(repo, path, repo_type, token=None, local_dir=None):
        got.append(path)
        dst = Path(local_dir) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((hub / path).read_bytes())
        return str(dst)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert RG.main(["--repack", "--channel", "perimeters", "--hf-repo", "me/raw", "--dry-run", "--work", str(tmp_path / "w")]) == 0
    assert got == ["fires/A/perimeters/records.geojson"]             # B is already gzipped, C has nothing
    [batch] = (tmp_path / "w" / "batch").iterdir()
    gz = batch / "fires" / "A" / "perimeters" / "records.geojson.gz"
    assert gzip.open(gz).read() == (hub / "fires/A/perimeters/records.geojson").read_bytes()
    [m] = (batch / "manifest" / "perimeters").glob("*.csv")
    row = pd.read_csv(m, dtype=str, keep_default_na=False).iloc[0].to_dict()
    assert row["fire_id"] == "A" and row["files"] == "records.geojson.gz" and row["status"] == "ok"
    assert row["sources"] == "https://nifc/a | https://nifc/b" and "2026-11-16T10:00:00" in row["error"]
    assert RG.done_ids(pd.DataFrame([row]), "perimeters") == {"A"}
    assert not (tmp_path / "w" / "repack_download").exists()
