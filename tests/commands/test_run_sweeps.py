"""ChannelRun, offline: with --max-hours, a pass that ends before the deadline checks the channel's whole manifest in
the repo and goes back to the fires not done (errors such as rate limits, fires a pass never reached) until all are
done or the deadline nears; the run ends with the list of fires it missed."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from fah.channels.base import Channel
from fah.commands.progress import ChannelProgress
from fah.commands.run import ChannelRun
from fah.commands.settings import RunSettings
from fah.core.errors import ExitCode
from fah.core.result import FetchResult, Status
from fah.hub.catalog import FireCatalog
from fah.hub.manifest import ManifestBook
from fah.hub.repo import HubRepo


class FakeHub:
    """A repo in a local folder: upload_folder copies a batch in, files / download read it back."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def check_write(self) -> str:
        return "me"

    def ensure(self, private: bool) -> None:
        pass

    def shared_paths(self) -> set[str]:
        return set()

    def files(self, folder: str | None = None, recursive: bool = True) -> list[str]:
        paths = (p.relative_to(self.root).as_posix() for p in sorted(self.root.rglob("*")) if p.is_file())
        return [p for p in paths if p.startswith(folder or "")]

    def download(self, path: str, local_dir: Path | None = None) -> str:
        return str(self.root / path)

    def upload_folder(self, folder: Path, message: str, delete_patterns: list[str] | None = None) -> None:
        shutil.copytree(folder, self.root, dirs_exist_ok=True)


class FlakyChannel(Channel):
    """fails: {fire_id: errors before it works}; unreached: fires the first pass never yields (as a shared download
    can lose a fire); empty: fires the source has nothing for; calls: every fire handed to fetch, in order."""

    name = "flaky"
    region_cut = False

    def __init__(
        self, fails: dict[str, int] | None = None, unreached: set[str] | None = None, empty: set[str] = frozenset()
    ) -> None:
        self.fails, self.unreached, self.calls = dict(fails or {}), set(unreached or ()), []
        self.empty = set(empty)

    def skip_reason(self, fire):
        return "not covered" if fire.fire_id == "NA" else None

    def fetch(self, fires, ctx):
        for f in fires:
            fid = f.fire_id
            self.calls.append(fid)
            if fid in self.unreached:
                self.unreached.discard(fid)
                continue
            if self.fails.get(fid, 0):
                self.fails[fid] -= 1
                yield FetchResult(f, Status.ERROR, (), "boom")
                continue
            if fid in self.empty:
                yield FetchResult(f, Status.EMPTY)
                continue
            (ctx.stage(fid) / "x.txt").write_text("x")
            yield FetchResult(f, Status.OK, ("https://example.org/x",))


class OneChannel:
    def __init__(self, channel: Channel) -> None:
        self.channel = channel

    def get(self, name: str) -> Channel:
        return self.channel


@pytest.fixture
def repo(tmp_path, monkeypatch, camp):
    """A fake repo and fires A..F (the Camp Fire under other ids) as its catalog; no pause between passes."""
    hub = FakeHub(tmp_path / "hub")
    monkeypatch.setattr(HubRepo, "connect", classmethod(lambda cls, repo_id: hub))
    fires = [camp.replace(fire_id=k) for k in "ABCDEF"]
    monkeypatch.setattr(FireCatalog, "fires", lambda self, archive="all", ids=None: fires)
    monkeypatch.setattr(ChannelRun, "RETRY_PAUSE_S", 0.0)
    return hub


def run(tmp_path, channel: Channel, **opts) -> int:
    opts.setdefault("max_hours", 1.0)
    s = RunSettings(channel=channel.name, hf_repo="me/raw", work=tmp_path / "w", **opts)
    return ChannelRun(s, OneChannel(channel)).execute()


def done(hub: FakeHub) -> set[str]:
    return ManifestBook.done_ids(ManifestBook.read(hub, "flaky"))


def test_fires_not_done_get_more_passes_until_all_are_done(tmp_path, repo):
    ch = FlakyChannel(fails={"B": 1, "D": 2, "F": 3}, unreached={"C"})  # e.g. rate-limited for a while
    assert run(tmp_path, ch) == ExitCode.OK
    # pass 1: all; pass 2: the errors and the unreached C; then whatever still fails
    assert ch.calls == [*"ABCDEF", *"BCDF", *"DF", "F"]
    assert done(repo) == set("ABCDEF")


def test_passes_end_at_the_deadline_and_a_run_without_one_makes_one_pass(tmp_path, repo):
    near = FlakyChannel(fails={"A": 99})
    assert run(tmp_path, near, max_hours=0.2) == ExitCode.OK  # 12 min: inside the 15 min kept for the upload
    assert near.calls == list("ABCDEF")
    shutil.rmtree(repo.root)
    open_ended = FlakyChannel(fails={"A": 99})
    assert run(tmp_path, open_ended, max_hours=None) == ExitCode.OK
    assert open_ended.calls == list("ABCDEF")


def test_a_limited_run_goes_back_only_to_its_own_fires(tmp_path, repo):
    ch = FlakyChannel(fails={"A": 1})
    assert run(tmp_path, ch, limit=2) == ExitCode.OK
    assert ch.calls == ["A", "B", "A"] and done(repo) == {"A", "B"}


def test_error_streak_stops_a_broken_channel_but_not_over_fires_failing_again(tmp_path, repo):
    broken = FlakyChannel(fails={k: 99 for k in "ABCDEF"})
    assert run(tmp_path, broken, max_errors_in_row=2) == ExitCode.CHANNEL_BROKEN  # nothing ever worked
    assert broken.calls == ["A", "B"]

    shutil.rmtree(repo.root)
    ch = FlakyChannel(fails={"B": 3, "D": 3})  # no two in a row in pass 1; later passes are only B, D
    assert run(tmp_path, ch, max_errors_in_row=2) == ExitCode.OK
    assert ch.calls == [*"ABCDEF", *"BD", *"BD", *"BD"] and done(repo) == set("ABCDEF")


def test_an_outage_after_fires_worked_pauses_the_run_then_it_goes_back(tmp_path, repo):
    ch = FlakyChannel(fails={"C": 1, "D": 1})  # the source fails for a while after A, B worked
    assert run(tmp_path, ch, max_errors_in_row=2) == ExitCode.OK
    assert ch.calls == [*"ABCD", *"CDEF"] and done(repo) == set("ABCDEF")

    shutil.rmtree(repo.root)
    down = FlakyChannel(fails={k: 99 for k in "CDEF"})  # still down after the pause: E, F fail too
    assert run(tmp_path, down, max_errors_in_row=2) == ExitCode.CHANNEL_BROKEN
    assert down.calls == [*"ABCD", *"CDEF"] and done(repo) == {"A", "B"}

    shutil.rmtree(repo.root)
    once = FlakyChannel(fails={"C": 1, "D": 1})
    assert run(tmp_path, once, max_errors_in_row=2, max_hours=None) == ExitCode.CHANNEL_BROKEN  # no later pass


def test_redo_counts_only_this_runs_rows_when_it_checks_the_repo(tmp_path, repo):
    old = repo.root / "manifest" / "flaky" / "old.csv"
    old.parent.mkdir(parents=True)
    pd.DataFrame([dict(fire_id=k, status="ok", fetched_utc="2026-01-01T00:00:00+00:00") for k in "ABCDEF"],
                 columns=ManifestBook.COLUMNS).to_csv(old, index=False)  # fmt: skip
    assert run(tmp_path, FlakyChannel(), limit=2) == ExitCode.OK  # everything done: nothing to fetch
    ch = FlakyChannel(unreached={"A"})
    assert run(tmp_path, ch, redo=True, limit=2) == ExitCode.OK
    assert ch.calls == ["A", "B", "A"]  # A's old 'ok' row does not stand in for this run's fetch


def test_dry_run_makes_one_pass(tmp_path, repo):
    ch = FlakyChannel(fails={"A": 1})
    assert run(tmp_path, ch, dry_run=True) == ExitCode.OK
    assert ch.calls == list("ABCDEF")


def test_the_run_ends_with_every_missed_fire_its_name_and_folder(tmp_path, repo, monkeypatch, camp, caplog):
    fires = [camp.replace(fire_id=k, name=f"Fire {k}") for k in ("A", "B", "C", "NA")]
    monkeypatch.setattr(FireCatalog, "fires", lambda self, archive="all", ids=None: fires)
    ch = FlakyChannel(fails={"A": 99}, empty={"B"})
    with caplog.at_level("INFO"):
        assert run(tmp_path, ch, max_hours=0.2) == ExitCode.OK  # one pass: the deadline is already near
    lines = caplog.messages
    i = next(k for k, m in enumerate(lines) if m.startswith("missed in flaky"))
    assert lines[i] == "missed in flaky: 1 not done, 1 empty (of 4 fires):"
    folder = "https://huggingface.co/datasets/me/raw/tree/main/fires"
    assert lines[i + 1] == f"  A  Fire A (2018)  error  {folder}/A  boom"
    assert lines[i + 2] == f"  B  Fire B (2018)  empty  {folder}/B"
    assert not any(" C " in m or "  NA " in m for m in lines[i + 3 :])  # ok and n/a fires are not missed


def test_missed_fires_name_why_they_are_not_done(camp, channels):
    a, b, c, d = (camp.replace(fire_id=k) for k in "ABCD")
    hrrr = channels.get("hrrr")
    m = pd.DataFrame([dict(fire_id="A", status="ok", window="old", region=hrrr.region_label(a), files="hrrr_ana.nc"),
                      dict(fire_id="B", status="error", error="HTTP 400 Exceeding allowed transaction limit."),
                      dict(fire_id="C", status="empty", window=hrrr.window_label(c), region=hrrr.region_label(c))],
                     columns=ManifestBook.COLUMNS).fillna("").assign(fetched_utc="2026-11-21")  # fmt: skip
    got = [(f.fire_id, st, note) for f, st, note in ChannelProgress.gaps(m, hrrr, [a, b, c, d])]
    assert got == [("A", "outdated", ""), ("B", "error", "HTTP 400 Exceeding allowed transaction limit."),
                   ("D", "not fetched", ""), ("C", "empty", "")]  # fmt: skip
