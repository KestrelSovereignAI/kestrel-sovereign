"""Tests for the first-run setup hook in `kestrel start`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign.cli import _maybe_first_run_setup


def test_first_run_returns_none_when_env_exists(tmp_path):
    (tmp_path / ".env").write_text("FOO=bar\n")
    assert _maybe_first_run_setup(tmp_path) is None


def test_first_run_returns_none_when_agent_already_registered(tmp_path):
    """Captures the CI clean-install scenario:

    A user (or CI) ran ``kestrel create`` first, which registered an
    agent and wrote ``multi_agent.toml`` but did NOT create ``.env``
    (inception falls back to plaintext key storage with a warning).
    Then they run ``kestrel start``. The first-run hook must NOT
    intercept — they've clearly done setup, even if not via the wizard.

    Pre-fix: the hook fired and exited 1, breaking the CI clean-install
    job that was already passing the inception + 3-pillar checks.
    """
    import toml as _toml

    multi_agent_path = tmp_path / "multi_agent.toml"
    multi_agent_path.write_text(_toml.dumps({
        "host": {"port": 8888, "bind": "0.0.0.0"},
        "agents": {
            "CITestAgent": {
                "data_dir": "agent_data/CITestAgent",
                "port": 8801,
                "autostart": True,
            }
        },
    }))
    # No .env on purpose.
    assert not (tmp_path / ".env").exists()
    assert _maybe_first_run_setup(tmp_path) is None


def test_first_run_fires_when_multi_agent_has_no_agents(tmp_path):
    """An empty multi_agent is the same as no multi_agent — fire the prompt."""
    import toml as _toml

    (tmp_path / "multi_agent.toml").write_text(_toml.dumps({
        "host": {"port": 8888, "bind": "0.0.0.0"},
        "agents": {},
    }))
    # No .env, multi_agent exists but is empty.
    with (
        patch("kestrel_sovereign.setup.prompts.is_tty", return_value=False),
    ):
        rc = _maybe_first_run_setup(tmp_path)
    # Non-TTY path → exit 1 with the hint.
    assert rc == 1


def test_first_run_tolerates_corrupt_multi_agent(tmp_path):
    """If multi_agent.toml fails to parse, we still want to prompt setup —
    don't silently let the user proceed with broken config."""
    (tmp_path / "multi_agent.toml").write_text("[broken\nthis = is not valid")
    with (
        patch("kestrel_sovereign.setup.prompts.is_tty", return_value=False),
    ):
        rc = _maybe_first_run_setup(tmp_path)
    assert rc == 1


def test_first_run_returns_none_when_skip_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_SKIP_FIRST_RUN", "1")
    # No .env present, but skip env wins.
    assert _maybe_first_run_setup(tmp_path) is None


def test_first_run_skipped_inside_git_worktree(tmp_path, monkeypatch):
    """Running any kestrel command from inside a git worktree must NOT
    fire the first-run hook. Worktrees never carry the user's gitignored
    state (.env, multi_agent.toml) — the user's real setup lives in the main
    checkout. Pre-fix, running from a worktree misfired the wizard on
    users who already had a perfectly valid setup at the main checkout.

    Worktree marker: ``.git`` is a FILE containing ``gitdir: ...`` rather
    than a directory.
    """
    monkeypatch.delenv("KESTREL_SKIP_FIRST_RUN", raising=False)
    # Simulate a worktree: .git is a file pointer, not a directory.
    (tmp_path / ".git").write_text(
        f"gitdir: {tmp_path}/main-checkout/.git/worktrees/foo\n"
    )
    # No .env, no multi_agent — would otherwise trigger the prompt path.
    assert _maybe_first_run_setup(tmp_path) is None


def test_first_run_still_fires_when_git_is_directory(tmp_path, monkeypatch):
    """Negative case: a regular checkout has ``.git`` as a directory.
    The worktree bypass must NOT short-circuit the genuine fresh-checkout
    case."""
    monkeypatch.delenv("KESTREL_SKIP_FIRST_RUN", raising=False)
    monkeypatch.setenv("CI", "true")  # Force is_tty() -> False, deterministic
    # Main-checkout shape: .git is a directory.
    (tmp_path / ".git").mkdir()
    rc = _maybe_first_run_setup(tmp_path)
    assert rc == 1  # Falls through to the non-tty hint path


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


def test_setup_check_and_reset_combo_is_rejected(tmp_path, capsys):
    """`kestrel setup --check --reset` must error before doing anything.

    The first defense is here at the CLI: --check is read-only by
    contract, --reset moves files; combining them is a contradiction.
    Reject with a clear message and exit 2 (usage error).
    """
    from kestrel_sovereign.cli import build_parser, cmd_setup

    # Stub a configured project so any wizard run that slipped through
    # would touch real files. We assert it doesn't.
    (tmp_path / ".env").write_text("KESTREL_DATA_KEY=stub\n")
    (tmp_path / "kestrel.toml").write_text("[llm]\nroute_priority = ['ollama:local']\n")
    env_text = (tmp_path / ".env").read_text()

    parser = build_parser()
    args = parser.parse_args(["setup", "--check", "--reset"])

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=tmp_path):
        rc = cmd_setup(args)

    assert rc == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err.lower()
    # Files untouched.
    assert (tmp_path / ".env").read_text() == env_text
    assert list(tmp_path.glob(".env.backup-*")) == []
    assert list(tmp_path.glob("kestrel.toml.backup-*")) == []


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
