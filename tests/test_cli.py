"""The command line keeps the flags, defaults and exit codes the Kaggle notebook relies on."""

from __future__ import annotations

import logging

import pytest

from fah.__main__ import main
from fah.hub.repo import HubRepo

FLAGS = ["--channel", "--status", "--check", "--links", "--links-chunk", "--extents", "--repack", "--extents-file",
         "--local-manifests", "--hf-repo", "--work", "--archive", "--catalog-file", "--catalog", "--years",
         "--min-acres", "--min-days", "--partial-windows", "--ids", "--limit", "--batch-gb",
         "--batch-fires", "--min-free-gb", "--max-hours", "--threads", "--private", "--redo", "--dry-run", "--keep",
         "--keep-cache", "--max-errors-in-row", "--verbose"]  # fmt: skip


def test_help_lists_every_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and all(flag in out for flag in FLAGS)


def test_a_setup_error_exits_3_with_one_line(monkeypatch, capsys):
    class NoToken:
        token = None

        def whoami(self):
            raise RuntimeError("401")

    monkeypatch.setattr(HubRepo, "connect", classmethod(lambda cls, repo: cls(NoToken(), repo)))
    assert main(["--check", "--hf-repo", "sam/raw"]) == 3
    assert "SETUP ERROR: HF_TOKEN is not set" in capsys.readouterr().err
    # huggingface_hub's HTTP library logs every request at INFO: the command line keeps those lines out
    assert all(logging.getLogger(n).getEffectiveLevel() == logging.WARNING for n in ("httpx", "httpcore"))


def test_a_channel_that_cannot_run_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.setattr("fah.core.secrets.Secrets.ENV_FILES", ())
    assert main(["--channel", "firms", "--dry-run", "--work", str(tmp_path)]) == 1
    assert "FIRMS_MAP_KEY" in capsys.readouterr().err


def test_missing_arguments_are_usage_errors():
    for argv in (["--status"], ["--repack"], [], ["--channel", "history"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2


def test_catalog_needs_a_target_repo_or_dry_run(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--catalog", "build"])
    assert e.value.code == 2 and "--catalog needs --hf-repo" in capsys.readouterr().err
