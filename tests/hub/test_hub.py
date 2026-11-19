"""The Hugging Face side, offline: the latest manifest row decides what is done, a batch is uploaded with its
manifest and then deleted (kept when the upload fails), secrets never reach a manifest, a new result replaces the
fire's older files, and setup problems stop with one clear line."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fah.core.errors import SetupError
from fah.hub.batch import BatchUploader
from fah.hub.manifest import ManifestBook
from fah.hub.repo import HubRepo


class FakeApi:
    """HfApi's upload_folder: records each commit's files and delete patterns, or fails."""

    def __init__(self, fail: bool = False) -> None:
        self.fail, self.commits, self.deletes = fail, [], []

    def upload_folder(self, repo_id, folder_path, repo_type, commit_message, delete_patterns=None):
        if self.fail:
            raise RuntimeError("503")
        files = sorted(str(p.relative_to(folder_path)) for p in Path(folder_path).rglob("*") if p.is_file())
        self.commits.append((commit_message, files))
        self.deletes.append(delete_patterns)


def uploader(tmp_path: Path, api: FakeApi, channel: str = "firms", **kw) -> BatchUploader:
    return BatchUploader(HubRepo(api, "me/raw"), channel, tmp_path, min_free_gb=0, **kw)


def staged(tmp: Path, fid: str, channel: str = "firms", name: str = "x.csv", size: int = 10) -> Path:
    d = tmp / "stage" / fid / channel
    d.mkdir(parents=True)
    (d / name).write_bytes(b"0" * size)
    return d


def test_latest_manifest_row_decides_done():
    m = pd.DataFrame([
        dict(fire_id="A", status="error", fetched_utc="2026-11-01"),
        dict(fire_id="A", status="ok", fetched_utc="2026-11-02"),
        dict(fire_id="B", status="ok", fetched_utc="2026-11-01"),
        dict(fire_id="B", status="error", fetched_utc="2026-11-02"),
        dict(fire_id="C", status="n/a", fetched_utc="2026-11-01"),
        dict(fire_id="D", status="empty", fetched_utc="2026-11-01"),
    ])  # fmt: skip
    assert ManifestBook.done_ids(m) == {"A", "C", "D"}
    assert ManifestBook.done_ids(None) == set()


def test_batch_uploads_with_manifest_then_deletes(tmp_path, camp):
    api = FakeApi()
    b = uploader(tmp_path, api, batch_fires=2)
    b.add(camp.replace(fire_id="A").ref(), "ok", staged(tmp_path, "A"), ["https://firms/x"])
    assert not api.commits
    b.add(camp.replace(fire_id="B").ref(), "n/a", error="no data")
    assert len(api.commits) == 1
    _, files = api.commits[0]
    assert "fires/A/firms/x.csv" in files and any(f.startswith("manifest/firms/") for f in files)
    assert not list((tmp_path / "batch").iterdir())  # deleted after upload
    assert b.totals["ok"] == 1 and b.totals["n/a"] == 1


def test_failed_upload_keeps_batch(tmp_path, monkeypatch, camp):
    monkeypatch.setattr("time.sleep", lambda s: None)
    b = uploader(tmp_path, FakeApi(fail=True), batch_fires=1)
    with pytest.raises(RuntimeError, match="batch kept"):
        b.add(camp.replace(fire_id="A").ref(), "ok", staged(tmp_path, "A"))
    assert list((tmp_path / "batch").rglob("x.csv"))


def test_error_status_uploads_no_files(tmp_path, camp):
    api = FakeApi()
    b = uploader(tmp_path, api, batch_fires=1)
    b.add(camp.replace(fire_id="A").ref(), "error", staged(tmp_path, "A"), error="HTTP 500")
    assert not any(f.startswith("fires/") for f in api.commits[0][1])


def test_shared_file_added_once(tmp_path, camp):
    src = tmp_path / "ics.zip"
    src.write_bytes(b"zip")
    api = FakeApi()
    b = uploader(tmp_path, api, "ics209", batch_fires=1, uploaded_shared={"shared/old.zip"})
    b.add_shared(src, "shared/ics209/ics209plus-wildfire.zip")
    b.add_shared(src, "shared/old.zip")
    b.add(camp.replace(fire_id="A").ref(), "empty")
    files = api.commits[0][1]
    assert "shared/ics209/ics209plus-wildfire.zip" in files and "shared/old.zip" not in files


def test_errors_in_manifest_never_hold_secrets(tmp_path, monkeypatch, camp):
    """A connection error quotes the request URL; the manifest is public, so the key and email must go."""
    monkeypatch.setenv("FIRMS_MAP_KEY", "k3yK3yK3y")
    monkeypatch.setenv("LANDFIRE_EMAIL", "crew@example.org")
    b = uploader(tmp_path, FakeApi(), batch_fires=2, keep=True)
    b.add(camp.replace(fire_id="A").ref(), "error",
          error="ConnectionError: HTTPSConnectionPool(host='firms.modaps.eosdis.nasa.gov', port=443): Max retries "
                "exceeded with url: /api/area/csv/k3yK3yK3y/VIIRS_SNPP_SP/x/5/2018-11-02")  # fmt: skip
    b.add(camp.replace(fire_id="B").ref(), "error",
          error="ReadTimeout for https://lfps.usgs.gov/api/job/submit?Email=crew%40example.org&Layer_List=x "
                "(contact crew@example.org)")  # fmt: skip
    text = "".join(p.read_text() for p in (tmp_path / "batch").rglob("*.csv"))
    assert "k3yK3yK3y" not in text and "crew@example.org" not in text and "crew%40example.org" not in text


def test_a_new_result_replaces_the_fires_older_files_but_an_error_keeps_them(tmp_path, camp):
    api = FakeApi()
    b = uploader(tmp_path, api, "history", batch_fires=10)
    b.add(camp.replace(fire_id="A").ref(), "ok", staged(tmp_path, "A", "history", "history.geojson.gz", 1))
    b.add(camp.replace(fire_id="B").ref(), "empty")
    b.add(camp.replace(fire_id="C").ref(), "error", error="timeout")
    b.add(camp.replace(fire_id="D").ref(), "n/a", error="before 2018")
    b.upload()
    assert api.deletes == [["fires/A/history/*", "fires/B/history/*"]]
    b.add(camp.replace(fire_id="E").ref(), "error", error="timeout")
    b.upload()
    assert api.deletes[-1] is None  # nothing replaced: no listing, no deletes


def test_batch_rows_record_the_labels_they_are_given(tmp_path, camp):
    b = BatchUploader(None, "gfs", tmp_path, dry_run=True)
    b.add(camp.ref(), "n/a", error="x", labels=("2018-11-05/2018-11-28", "a,b,c,d"))
    assert b.rows[-1]["window"] == "2018-11-05/2018-11-28" and b.rows[-1]["region"] == "a,b,c,d"
    assert {"window", "region"} <= set(ManifestBook.COLUMNS)


class WhoAmI:
    def __init__(self, token: str | None = "hf_x", me: dict | None = None, fail: bool = False) -> None:
        self.token, self.me, self.fail = token, me, fail

    def whoami(self):
        if self.fail:
            raise RuntimeError("401 Client Error: Unauthorized")
        return self.me


ME = {"name": "sam", "orgs": [{"name": "gt-fire"}], "auth": {"accessToken": {"role": "write"}}}


@pytest.mark.parametrize("api, repo, says", [
    (WhoAmI(token=None, fail=True), "sam/raw", "HF_TOKEN is not set"),
    (WhoAmI(fail=True), "sam/raw", "rejected HF_TOKEN"),
    (WhoAmI(me=ME), "your-name/wildfire-raw-2014-2025", "'your-name' is not your account"),
    (WhoAmI(me=dict(ME, auth={"accessToken": {"role": "read"}})), "sam/raw", "read-only"),
    (WhoAmI(me=ME), "no-slash", "account/repo-name"),
])  # fmt: skip
def test_setup_problems_stop_with_one_clear_line(api, repo, says):
    with pytest.raises(SetupError, match=says):
        HubRepo(api, repo).check_write()


def test_setup_ok_for_own_account_and_org():
    assert HubRepo(WhoAmI(me=ME), "sam/raw").check_write() == "sam"
    assert HubRepo(WhoAmI(me=ME), "gt-fire/raw").check_write() == "sam"
