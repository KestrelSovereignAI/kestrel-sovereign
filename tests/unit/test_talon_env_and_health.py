"""Tests for talon coordinator env handling and the new talon_health tool.

Past dispatch failures were quietly caused by passing the parent
process's full ``os.environ`` to kestrel-talon — including
``ANTHROPIC_API_KEY``, which forces Claude Agent SDK away from the
Claude Max OAuth path talon expects. The fix strips Anthropic
credentials before launching the subprocess; ``talon_health``
exposes the result of that without dispatching real work.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.features.talon.coordinator import TalonCoordinatorFeature


def _make_feature():
    agent = SimpleNamespace(_scheduler=None)
    return TalonCoordinatorFeature(agent)


def test_build_subprocess_env_strips_anthropic_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-leakedkey")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-bad")
    monkeypatch.setenv("CLAUDE_API_KEY", "ck-bad")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")

    env = TalonCoordinatorFeature._build_subprocess_env()

    for key in TalonCoordinatorFeature._ANTHROPIC_KEYS_TO_STRIP:
        assert key not in env, f"{key} leaked into talon subprocess env"
    assert env.get("GITHUB_TOKEN") == "ghp_test_token"
    assert env.get("GH_TOKEN") == "ghp_test_token"


def test_build_subprocess_env_promotes_gh_token_to_github_token(monkeypatch):
    """If only GH_TOKEN is set, mirror it into GITHUB_TOKEN — talon's
    downstream subprocesses (gh CLI, octokit) expect either name.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghp_only_gh_form")

    env = TalonCoordinatorFeature._build_subprocess_env()

    assert env.get("GH_TOKEN") == "ghp_only_gh_form"
    assert env.get("GITHUB_TOKEN") == "ghp_only_gh_form"


def test_build_subprocess_env_raises_when_no_github_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)

    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        TalonCoordinatorFeature._build_subprocess_env()


@pytest.mark.asyncio
async def test_dispatch_via_cli_returns_clear_error_when_no_token(monkeypatch):
    """The error must surface as a structured result, not raise — the
    agent has to be able to report it back to the user.
    """
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    feat = _make_feature()
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin",
        return_value="/fake/kestrel-talon",
    ):
        result = await feat._dispatch_via_cli(["claim", "--issue", "1"])

    assert result["dispatched"] is False
    assert "GITHUB_TOKEN" in result["error"]


@pytest.mark.asyncio
async def test_talon_health_returns_unhealthy_when_binary_missing(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    feat = _make_feature()
    with patch.object(TalonCoordinatorFeature, "_find_talon_bin", return_value=None):
        report = await feat.talon_health()

    assert report["healthy"] is False
    assert report["binary"]["found"] is False
    assert "execute" not in report  # short-circuited at binary stage


@pytest.mark.asyncio
async def test_talon_health_runs_help_and_reports_success(monkeypatch, tmp_path):
    """Mock the subprocess so we can assert health-check shape without
    requiring kestrel-talon installed on the test runner.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-be-passed")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    fake_bin = tmp_path / "kestrel-talon"
    fake_bin.write_text("#!/bin/sh\necho 'Usage: kestrel-talon ...'\n")
    fake_bin.chmod(0o755)

    captured_env: dict = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(
            return_value=(b"Usage: kestrel-talon [options]\n", b""),
        )
        return proc

    feat = _make_feature()
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ), patch(
        "asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec,
    ):
        report = await feat.talon_health()

    assert report["healthy"] is True
    assert report["binary"]["found"] is True
    assert report["binary"]["path"] == str(fake_bin)
    assert report["execute"]["ok"] is True
    assert report["execute"]["returncode"] == 0
    assert "Usage" in report["execute"]["first_line"]
    # Anthropic keys must have been stripped from what we passed to subprocess
    assert "ANTHROPIC_API_KEY" not in captured_env
    assert captured_env.get("GITHUB_TOKEN") == "ghp_test"
    # Stripped list reports what WAS in os.environ at health-check time
    assert "ANTHROPIC_API_KEY" in report["env"]["stripped_anthropic_keys"]
    assert report["env"]["github_token_source"] == "GITHUB_TOKEN"


@pytest.mark.asyncio
async def test_talon_health_reports_help_failure(monkeypatch, tmp_path):
    """If --help itself fails (binary segfaults, missing dep, etc.),
    health must surface that with returncode + stderr tail rather
    than silently succeeding.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon"
    fake_bin.write_text("#!/bin/sh\necho 'boom' >&2\nexit 7\n")
    fake_bin.chmod(0o755)

    async def fake_create_subprocess_exec(*args, **kwargs):
        proc = AsyncMock()
        proc.returncode = 7
        proc.communicate = AsyncMock(return_value=(b"", b"boom\n"))
        return proc

    feat = _make_feature()
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ), patch(
        "asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec,
    ):
        report = await feat.talon_health()

    assert report["healthy"] is False
    assert report["execute"]["ok"] is False
    assert report["execute"]["returncode"] == 7
    assert "boom" in report["execute"]["stderr_tail"]
