"""Tests for the first-run setup hook in `kestrel start`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign.cli import _maybe_first_run_setup


def test_first_run_returns_none_when_env_exists(tmp_path):
    (tmp_path / ".env").write_text("FOO=bar\n")
    assert _maybe_first_run_setup(tmp_path) is None


def test_first_run_returns_none_when_skip_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_SKIP_FIRST_RUN", "1")
    # No .env present, but skip env wins.
    assert _maybe_first_run_setup(tmp_path) is None


def test_first_run_non_tty_exits_with_hint(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("KESTREL_SKIP_FIRST_RUN", raising=False)
    monkeypatch.setenv("CI", "true")  # Forces is_tty() -> False
    rc = _maybe_first_run_setup(tmp_path)
    assert rc == 1
    captured = capsys.readouterr()
    assert "kestrel setup --quickstart" in captured.out


def test_first_run_tty_yes_runs_wizard(tmp_path, monkeypatch):
    """With TTY + 'y' answer, the hook runs the wizard and returns None on success."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("KESTREL_NONINTERACTIVE", raising=False)
    monkeypatch.delenv("KESTREL_SKIP_FIRST_RUN", raising=False)

    with (
        patch("kestrel_sovereign.setup.prompts.is_tty", return_value=True),
        patch("builtins.input", return_value="y"),
        patch("kestrel_sovereign.setup.wizard.run_wizard", return_value=0) as mock_run,
    ):
        rc = _maybe_first_run_setup(tmp_path)
    assert rc is None
    mock_run.assert_called_once()


def test_first_run_tty_no_returns_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("KESTREL_NONINTERACTIVE", raising=False)
    monkeypatch.delenv("KESTREL_SKIP_FIRST_RUN", raising=False)

    with (
        patch("kestrel_sovereign.setup.prompts.is_tty", return_value=True),
        patch("builtins.input", return_value="n"),
    ):
        rc = _maybe_first_run_setup(tmp_path)
    assert rc == 1


def test_first_run_tty_eof_returns_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("KESTREL_NONINTERACTIVE", raising=False)
    monkeypatch.delenv("KESTREL_SKIP_FIRST_RUN", raising=False)

    with (
        patch("kestrel_sovereign.setup.prompts.is_tty", return_value=True),
        patch("builtins.input", side_effect=EOFError),
    ):
        rc = _maybe_first_run_setup(tmp_path)
    assert rc == 1


def test_first_run_propagates_wizard_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("KESTREL_NONINTERACTIVE", raising=False)
    monkeypatch.delenv("KESTREL_SKIP_FIRST_RUN", raising=False)

    with (
        patch("kestrel_sovereign.setup.prompts.is_tty", return_value=True),
        patch("builtins.input", return_value=""),  # default = yes
        patch("kestrel_sovereign.setup.wizard.run_wizard", return_value=2),
    ):
        rc = _maybe_first_run_setup(tmp_path)
    assert rc == 2
