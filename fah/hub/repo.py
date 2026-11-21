"""The target dataset repo on Hugging Face."""

from __future__ import annotations

import io
import logging
import os
import random
import time
from collections.abc import Callable
from pathlib import Path

from fah.core.errors import SetupError
from fah.core.secrets import Secrets

log = logging.getLogger(__name__)


class HubRepo:
    """One dataset repo, through a huggingface_hub HfApi (token from HF_TOKEN, else a cached `hf auth login`)."""

    CARD = Path(__file__).resolve().parents[2] / "docs" / "RAW_DATA_CARD.md"
    WRITE_TRIES = 6
    WRITE_WAIT_S = 30.0  # doubled each try (30 s to 8 min), times 0.5-1.5 so parallel copies do not retry in step

    def __init__(self, api, repo_id: str | None) -> None:
        self.api, self.repo_id = api, repo_id

    @classmethod
    def connect(cls, repo_id: str | None) -> HubRepo:
        from huggingface_hub import HfApi

        return cls(HfApi(token=os.environ.get("HF_TOKEN") or None), repo_id)

    @property
    def token(self) -> str | None:
        return self.api.token

    def check_write(self) -> str:
        """Fail fast, with one clear line, when the token cannot write to the repo. -> the account name."""
        repo = self.repo_id
        if not repo or "/" not in repo:
            raise SetupError(f"--hf-repo must look like 'account/repo-name', got {repo!r}")
        try:
            me = self.api.whoami()
        except Exception as ex:  # noqa: BLE001  (any failure means the token is missing or refused)
            if not self.api.token:
                raise SetupError(
                    "HF_TOKEN is not set. On Kaggle: Add-ons > Secrets > add a WRITE token from "
                    "https://huggingface.co/settings/tokens under the label SECRET_LABELS names (default "
                    "HF_TOKEN_ALL), switch it on for this notebook, then rerun the secrets cell."
                ) from ex
            raise SetupError(
                f"Hugging Face rejected HF_TOKEN ({type(ex).__name__}: {Secrets.redact(str(ex))[:120]}). "
                "Make a new WRITE token at https://huggingface.co/settings/tokens and update the Kaggle "
                "secret."
            ) from ex
        names = {me.get("name")} | {o.get("name") for o in me.get("orgs") or []}
        names.discard(None)
        ns = repo.split("/")[0]
        if ns not in names:
            raise SetupError(
                f"HF_REPO {repo!r}: '{ns}' is not your account or one of your organisations "
                f'({", ".join(sorted(names))}). Use e.g. HF_REPO = "{me.get("name")}/wildfire-raw-2014-2025".'
            )
        role = ((me.get("auth") or {}).get("accessToken") or {}).get("role")
        if role == "read":
            raise SetupError(
                "HF_TOKEN is read-only. Make a token with write access at https://huggingface.co/settings/tokens."
            )
        return me.get("name")

    def ensure(self, private: bool) -> None:
        """Create the repo; on first use add the dataset card (docs/RAW_DATA_CARD.md)."""
        from huggingface_hub import CommitOperationAdd

        self._write(
            "create repo",
            lambda: self.api.create_repo(self.repo_id, repo_type="dataset", private=private, exist_ok=True),
        )
        if "README.md" in {p.path for p in self.api.list_repo_tree(self.repo_id, repo_type="dataset")}:
            return
        ops = [CommitOperationAdd("README.md", io.BytesIO(self.CARD.read_bytes()))]
        self._write(
            "dataset card",
            lambda: self.api.create_commit(self.repo_id, ops, repo_type="dataset", commit_message="Dataset card"),
        )

    def files(self, folder: str | None = None, recursive: bool = True) -> list[str]:
        """Paths under a folder of the repo; [] when the repo or folder does not exist yet."""
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        try:
            return [
                p.path
                for p in self.api.list_repo_tree(
                    self.repo_id, path_in_repo=folder, recursive=recursive, repo_type="dataset"
                )
            ]
        except (EntryNotFoundError, RepositoryNotFoundError):
            return []
        except Exception as ex:  # noqa: BLE001  (folder not created yet)
            if "404" in str(ex):
                return []
            raise

    def shared_paths(self) -> set[str]:
        """Files already under shared/ (uploaded once per repo)."""
        try:
            return set(self.files("shared"))
        except Exception:  # noqa: BLE001  (treated as none: they are uploaded again)
            return set()

    def download(self, path: str, local_dir: Path | None = None) -> str:
        """A repo file, downloaded (into local_dir when given). -> its local path."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            self.repo_id, path, repo_type="dataset", token=self.token, local_dir=str(local_dir) if local_dir else None
        )

    def snapshot(self, patterns: list[str], local_dir: Path) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            self.repo_id, repo_type="dataset", local_dir=str(local_dir), token=self.token, allow_patterns=patterns
        )

    def upload_folder(self, folder: Path, message: str, delete_patterns: list[str] | None = None) -> None:
        self._write(
            message,
            lambda: self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(folder),
                repo_type="dataset",
                commit_message=message,
                delete_patterns=delete_patterns,
            ),
        )

    def upload_file(self, local: Path, path: str, message: str) -> None:
        self._write(
            message,
            lambda: self.api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=message,
            ),
        )

    def _write(self, what: str, call: Callable[[], object]) -> None:
        """One write to the repo, retried: a commit racing another notebook copy's commit to the same repo, rate
        limits and server errors all pass with time. A refused token or a missing repo (401 / 403 / 404) does not."""
        for attempt in range(self.WRITE_TRIES):
            try:
                call()
                return
            except Exception as ex:  # noqa: BLE001  (HfHubHTTPError, connection errors)
                code = getattr(getattr(ex, "response", None), "status_code", None)
                if code in (401, 403, 404) or attempt == self.WRITE_TRIES - 1:
                    raise
                wait = self.WRITE_WAIT_S * 2**attempt * random.uniform(0.5, 1.5)
                log.info(f"  {what[:60]}: retry {attempt + 1} in {wait:.0f}s ({Secrets.redact(str(ex))[:150]})")
                time.sleep(wait)
