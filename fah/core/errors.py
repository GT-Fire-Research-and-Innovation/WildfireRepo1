"""Errors that end a command, and the exit codes the Kaggle notebook acts on."""

from __future__ import annotations


class ExitCode:
    OK = 0
    FAILED = 1  # a channel cannot run here (missing key or tool) or a step had nothing to do
    SETUP = 3  # token or repo name wrong: every other channel would fail the same way
    CHANNEL_BROKEN = 4  # a channel failed for many fires in a row: other channels may still work


class SetupError(Exception):
    """The Hugging Face token or repo cannot be used (exit code 3)."""


class ChannelUnavailable(Exception):
    """A channel cannot run on this machine: a key, tool or package is missing (exit code 1)."""
