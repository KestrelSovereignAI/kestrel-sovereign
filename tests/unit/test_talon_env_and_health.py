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
import os
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


@pytest.mark.asyncio
async def test_dispatch_via_cli_background_returns_immediately_and_logs(
    tmp_path, monkeypatch,
):
    """Background dispatch must return BEFORE talon completes — that
    was the whole reason ``talon_claim`` was timing out at 300s.
    Use a script that sleeps 30s and assert dispatch returns in
    under a second with a usable job_id and log path.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    # Real subprocess: sleep ~30s. We assert dispatch returns fast,
    # then kill the child so the test exits cleanly.
    fake_bin = tmp_path / "kestrel-talon"
    fake_bin.write_text(
        "#!/bin/sh\n"
        "echo 'starting'\n"
        "for i in $(seq 1 30); do echo \"tick $i\"; sleep 1; done\n"
    )
    fake_bin.chmod(0o755)

    agent = SimpleNamespace(_scheduler=None, storage_path=str(tmp_path / "agent.db"))
    feat = TalonCoordinatorFeature(agent)

    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        t0 = asyncio.get_event_loop().time()
        result = await feat._dispatch_via_cli_background(
            ["claim", "--repo", "x/y", "--issue", "1"],
            label="claim:x/y#1",
            extra_meta={"repo": "x/y", "issue": 1},
        )
        elapsed = asyncio.get_event_loop().time() - t0

    try:
        assert elapsed < 2.0, (
            f"Background dispatch should return in < 2s, took {elapsed:.2f}s"
        )
        assert result["dispatched"] is True
        assert result["method"] == "cli_background"
        assert result["job_id"]
        assert result["pid"]
        assert result["log_path"]

        job = feat._jobs[result["job_id"]]
        assert job["status"] == "dispatched"
        assert job["repo"] == "x/y"
        assert job["issue"] == 1
        assert os.path.isfile(job["log_path"])
    finally:
        # Reap the long-running fake-talon child so pytest doesn't
        # hang on shutdown waiting for it.
        proc = feat._jobs[result["job_id"]]["process"]
        proc.kill()
        await proc.wait()


@pytest.mark.asyncio
async def test_status_reaps_finished_background_jobs(tmp_path, monkeypatch):
    """``talon_status`` must reap subprocesses that have exited and
    flip their status to ``complete`` / ``failed`` based on rc.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    fake_bin = tmp_path / "kestrel-talon-fast"
    fake_bin.write_text("#!/bin/sh\necho done\nexit 0\n")
    fake_bin.chmod(0o755)

    agent = SimpleNamespace(_scheduler=None, storage_path=str(tmp_path / "agent.db"))
    feat = TalonCoordinatorFeature(agent)

    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["dummy"], label="dummy", extra_meta={},
        )

    job_id = result["job_id"]
    # Wait for the child to actually exit before polling status.
    await feat._jobs[job_id]["process"].wait()

    status = await feat.talon_status()
    matching = [j for j in status["jobs"] if j["id"] == job_id]
    assert matching, f"Job {job_id} missing from status"
    assert matching[0]["status"] == "complete"
    assert matching[0]["returncode"] == 0
    assert "completed_at" in matching[0]
    # Process handle must NOT leak into the response (not JSON-safe).
    assert "process" not in matching[0]


@pytest.mark.asyncio
async def test_status_marks_failed_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon-fail"
    fake_bin.write_text("#!/bin/sh\nexit 3\n")
    fake_bin.chmod(0o755)

    agent = SimpleNamespace(_scheduler=None, storage_path=str(tmp_path / "agent.db"))
    feat = TalonCoordinatorFeature(agent)

    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["dummy"], label="dummy", extra_meta={},
        )

    await feat._jobs[result["job_id"]]["process"].wait()
    status = await feat.talon_status()
    matching = [j for j in status["jobs"] if j["id"] == result["job_id"]]
    assert matching[0]["status"] == "failed"
    assert matching[0]["returncode"] == 3


@pytest.mark.asyncio
async def test_talon_job_log_returns_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon-log"
    fake_bin.write_text(
        "#!/bin/sh\n"
        "for i in $(seq 1 20); do echo \"log line $i\"; done\n"
    )
    fake_bin.chmod(0o755)

    agent = SimpleNamespace(_scheduler=None, storage_path=str(tmp_path / "agent.db"))
    feat = TalonCoordinatorFeature(agent)

    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["dummy"], label="dummy", extra_meta={},
        )

    await feat._jobs[result["job_id"]]["process"].wait()

    log_result = await feat.talon_job_log(result["job_id"], lines=5)
    assert log_result["success"] is True
    assert log_result["lines"] == 5
    assert "log line 20" in log_result["content"]
    assert "log line 16" in log_result["content"]
    # Earlier lines should NOT be in the tail
    assert "log line 1\n" not in log_result["content"][:200]


@pytest.mark.asyncio
async def test_talon_job_log_unknown_id():
    feat = TalonCoordinatorFeature(SimpleNamespace(_scheduler=None))
    result = await feat.talon_job_log("not-a-real-id")
    assert result["success"] is False
    assert "Unknown" in result["error"]
