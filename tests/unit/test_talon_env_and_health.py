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
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.tools.result import ToolResultStatus

import pytest

from kestrel_sovereign._async_process import terminate_process_tree
from kestrel_sovereign._bounded_subprocess import BoundedProcessResult
from kestrel_sovereign.features.talon.coordinator import TalonCoordinatorFeature


def _make_feature():
    agent = SimpleNamespace(_scheduler=None)
    return TalonCoordinatorFeature(agent)


def test_build_subprocess_env_strips_anthropic_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-leakedkey")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-bad")
    monkeypatch.setenv("CLAUDE_API_KEY", "ck-bad")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    # Clear any pre-existing GitHub token vars to isolate the test
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)
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


def test_build_git_subprocess_env_keeps_only_github_credential(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("KESTREL_DATA_KEY", "data-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_git_only")

    env = TalonCoordinatorFeature._build_git_subprocess_env(
        require_github_token=True
    )

    assert env["GITHUB_TOKEN"] == "ghp_git_only"
    assert env["GH_TOKEN"] == "ghp_git_only"
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "KESTREL_DATA_KEY" not in env


@pytest.mark.asyncio
async def test_local_git_run_does_not_inherit_github_token(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_local_git_must_not_receive")
    feature = _make_feature()
    captured: dict = {}

    async def fake_run(args, **kwargs):
        captured["env"] = kwargs["env"]
        return BoundedProcessResult(
            argv=tuple(args),
            returncode=0,
            stdout=b"",
            stderr=b"",
            duration_ms=1,
        )

    with patch(
        "kestrel_sovereign.features.talon.coordinator.run_bounded_subprocess",
        side_effect=fake_run,
    ):
        result = await feature._git_run(["status"], cwd=tmp_path)

    assert result["ok"] is True
    assert "GITHUB_TOKEN" not in captured["env"]
    assert "GH_TOKEN" not in captured["env"]


@pytest.mark.asyncio
async def test_workspace_status_git_probe_uses_untrusted_env(monkeypatch, tmp_path):
    """A hostile workspace git config must not inherit agent credentials."""

    for key, value in {
        "ANTHROPIC_API_KEY": "sk-ant-secret",
        "OPENAI_API_KEY": "sk-openai-secret",
        "GITHUB_TOKEN": "ghp_secret",
        "KESTREL_DATA_KEY": "data-secret",
    }.items():
        monkeypatch.setenv(key, value)
    feature = _make_feature()
    captured: dict = {}

    async def fake_run(args, **kwargs):
        captured["args"] = list(args)
        captured["env"] = kwargs["env"]
        return BoundedProcessResult(
            argv=tuple(args),
            returncode=0,
            stdout=b"",
            stderr=b"",
            duration_ms=1,
        )

    structural = {
        "repo": "org/repo",
        "path": str(tmp_path),
        "exists": True,
        "is_git": True,
        "head": "main",
        "clean": None,
        "last_fetch_at": None,
        "safe": True,
    }
    with (
        patch.object(feature, "_workspace_state", return_value=structural),
        patch(
            "kestrel_sovereign.features.talon.coordinator.run_bounded_subprocess",
            side_effect=fake_run,
        ),
    ):
        state = await feature._workspace_state_with_status("org/repo")

    assert state["clean"] is True
    assert captured["args"][:2] == ["git", "-c"]
    assert captured["args"][2].startswith("core.hooksPath=")
    for secret in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "KESTREL_DATA_KEY",
    ):
        assert secret not in captured["env"]


@pytest.mark.asyncio
async def test_git_run_returns_redacted_mapping_on_runner_failure(tmp_path):
    feature = _make_feature()
    secret = "github_pat_AAAAAAAAAAAAAAAAAAAAA"

    with patch(
        "kestrel_sovereign.features.talon.coordinator.run_bounded_subprocess",
        side_effect=RuntimeError(f"pipe failed while handling {secret}"),
    ):
        result = await feature._git_run(["status"], cwd=tmp_path)

    assert result["ok"] is False
    assert "subprocess tooling error" in result["error"]
    assert secret not in result["error"]
    assert "[REDACTED]" in result["error"]


@pytest.mark.asyncio
async def test_talon_health_returns_unhealthy_when_binary_missing(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    feat = _make_feature()
    with patch.object(TalonCoordinatorFeature, "_find_talon_bin", return_value=None):
        envelope = await feat.talon_health()

    # talon_health now returns a ToolResult envelope (#1061 wave 15);
    # the legacy report dict lives under .data.
    report = envelope.data
    assert report["healthy"] is False
    assert report["binary"]["found"] is False
    assert "execute" not in report  # short-circuited at binary stage


@pytest.mark.asyncio
async def test_talon_health_runs_help_and_reports_success(monkeypatch, tmp_path):
    """Mock the subprocess so we can assert health-check shape without
    requiring kestrel-talon installed on the test runner.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-be-passed")
    # Clear any pre-existing GitHub token vars to isolate the test
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    fake_bin = tmp_path / "kestrel-talon"
    fake_bin.write_text("#!/bin/sh\necho 'Usage: kestrel-talon ...'\n")
    fake_bin.chmod(0o755)

    captured_env: dict = {}

    async def fake_run_bounded_subprocess(args, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return BoundedProcessResult(
            argv=tuple(args),
            returncode=0,
            stdout=b"Usage: kestrel-talon [options]\n",
            stderr=b"",
            duration_ms=1,
        )

    feat = _make_feature()
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ), patch(
        "kestrel_sovereign.features.talon.coordinator.run_bounded_subprocess",
        side_effect=fake_run_bounded_subprocess,
    ):
        envelope = await feat.talon_health()

    report = envelope.data
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

    async def fake_run_bounded_subprocess(args, **kwargs):
        return BoundedProcessResult(
            argv=tuple(args),
            returncode=7,
            stdout=b"",
            stderr=b"boom\n",
            duration_ms=1,
        )

    feat = _make_feature()
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ), patch(
        "kestrel_sovereign.features.talon.coordinator.run_bounded_subprocess",
        side_effect=fake_run_bounded_subprocess,
    ):
        envelope = await feat.talon_health()

    report = envelope.data
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
        await terminate_process_tree(proc, terminate_grace=0.1, reap_timeout=1.0)


@pytest.mark.asyncio
async def test_background_dispatch_cleans_group_if_ownership_cannot_persist(
    tmp_path, monkeypatch,
):
    """No durable registry row means the detached process was never handed off."""

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    feat = TalonCoordinatorFeature(
        SimpleNamespace(_scheduler=None, storage_path=str(tmp_path / "agent.db"))
    )
    proc = MagicMock(pid=4242, returncode=None)
    cleanup = AsyncMock()
    captured: dict = {}

    async def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return proc

    with (
        patch.object(
            TalonCoordinatorFeature,
            "_find_talon_bin",
            return_value=str(fake_bin),
        ),
        patch.object(feat, "_persist_jobs", return_value=False),
        patch(
            "kestrel_sovereign.features.talon.coordinator.terminate_process_tree",
            cleanup,
        ),
        patch("asyncio.create_subprocess_exec", side_effect=fake_create),
    ):
        result = await feat._dispatch_via_cli_background(
            ["claim", "--repo", "x/y", "--issue", "1"],
            label="claim:x/y#1",
        )

    assert result["dispatched"] is False
    assert "persist" in result["error"].lower()
    cleanup.assert_awaited_once_with(proc)
    assert feat._jobs == {}
    if os.name == "nt":
        assert captured["creationflags"]
    else:
        assert captured["start_new_session"] is True


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
    matching = [j for j in status.data["jobs"] if j["id"] == job_id]
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
    matching = [j for j in status.data["jobs"] if j["id"] == result["job_id"]]
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
    assert log_result.data["success"] is True
    assert log_result.data["lines"] == 5
    assert "log line 20" in log_result.data["content"]
    assert "log line 16" in log_result.data["content"]
    # Earlier lines should NOT be in the tail
    assert "log line 1\n" not in log_result.data["content"][:200]


@pytest.mark.asyncio
async def test_talon_job_log_unknown_id():
    feat = TalonCoordinatorFeature(SimpleNamespace(_scheduler=None))
    result = await feat.talon_job_log("not-a-real-id")
    assert result.status is ToolResultStatus.ERROR
    assert "Unknown" in result.error


@pytest.mark.asyncio
async def test_status_survives_feature_restart(tmp_path, monkeypatch):
    """A CLI-background job dispatched before a restart must still be
    visible in ``talon_status`` from a freshly-constructed feature
    (Kestrel restart) — its public metadata persists to jobs.json,
    and the exit-code sidecar makes status authoritative.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon-restart"
    fake_bin.write_text(
        "#!/bin/sh\necho 'restart-marker-output'\nexit 0\n"
    )
    fake_bin.chmod(0o755)

    agent = SimpleNamespace(
        _scheduler=None, storage_path=str(tmp_path / "agent.db")
    )
    feat = TalonCoordinatorFeature(agent)

    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["claim", "--repo", "x/y", "--issue", "7"],
            label="claim:x/y#7",
            extra_meta={"repo": "x/y", "issue": 7},
        )

    job_id = result["job_id"]
    # Wait for the wrapper to exit — sidecar is written before exit.
    await feat._jobs[job_id]["process"].wait()

    # Simulate a Kestrel restart: a fresh feature reloads from disk.
    fresh = TalonCoordinatorFeature(agent)
    assert job_id in fresh._jobs, "eager reload should populate _jobs"

    status = await fresh.talon_status()
    matching = [j for j in status.data["jobs"] if j["id"] == job_id]
    assert matching, f"Job {job_id} lost after restart"
    job = matching[0]
    assert job["repo"] == "x/y"
    assert job["issue"] == 7
    assert "process" not in job
    # Strict: exit-sidecar must yield a definitive complete + rc=0.
    assert job["status"] == "complete"
    assert job["returncode"] == 0


@pytest.mark.asyncio
async def test_failed_job_reported_as_failed_after_restart(tmp_path, monkeypatch):
    """A CLI-background job that exits non-zero must surface as
    ``failed`` after a Kestrel restart (not silently ``complete``).
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon-fail"
    fake_bin.write_text(
        "#!/bin/sh\necho 'failure-output'\nexit 17\n"
    )
    fake_bin.chmod(0o755)

    agent = SimpleNamespace(
        _scheduler=None, storage_path=str(tmp_path / "agent.db")
    )
    feat = TalonCoordinatorFeature(agent)

    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["bogus"], label="fail", extra_meta={"repo": "x/y", "issue": 13},
        )
    job_id = result["job_id"]
    await feat._jobs[job_id]["process"].wait()

    fresh = TalonCoordinatorFeature(agent)
    status = await fresh.talon_status()
    job = next(j for j in status.data["jobs"] if j["id"] == job_id)
    assert job["status"] == "failed"
    assert job["returncode"] == 17


@pytest.mark.asyncio
async def test_dispatch_after_restart_preserves_old_jobs(tmp_path, monkeypatch):
    """Dispatching a new job from a fresh feature must NOT truncate
    older persisted jobs out of jobs.json — the registry must be
    additive across restart-then-dispatch sequences.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon-mix"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    agent = SimpleNamespace(
        _scheduler=None, storage_path=str(tmp_path / "agent.db")
    )
    feat = TalonCoordinatorFeature(agent)

    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        first = await feat._dispatch_via_cli_background(
            ["a"], label="a", extra_meta={"repo": "x/y", "issue": 1},
        )
    await feat._jobs[first["job_id"]]["process"].wait()

    # Fresh feature dispatches a second job. The first must still be
    # persisted (not erased by _persist_jobs writing only in-memory).
    fresh = TalonCoordinatorFeature(agent)
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        second = await fresh._dispatch_via_cli_background(
            ["b"], label="b", extra_meta={"repo": "x/y", "issue": 2},
        )
    await fresh._jobs[second["job_id"]]["process"].wait()

    # Construct a third feature; it should see both jobs on disk.
    third = TalonCoordinatorFeature(agent)
    assert first["job_id"] in third._jobs
    assert second["job_id"] in third._jobs


@pytest.mark.asyncio
async def test_no_sidecar_dead_pid_reports_finished_unknown(tmp_path, monkeypatch):
    """When a persisted job has no exit sidecar and its pid is dead,
    status must be ``finished_unknown`` — never silently ``complete``.
    A job killed by SIGKILL or system shutdown before the wrapper
    could write its sidecar lands here.
    """
    agent = SimpleNamespace(
        _scheduler=None, storage_path=str(tmp_path / "agent.db")
    )
    feat = TalonCoordinatorFeature(agent)

    # Hand-write a registry entry for a job whose pid is guaranteed dead
    # (pid 1 is init; signal 0 is permitted but won't ProcessLookupError).
    # We pick a pid we know does not exist by spawning and reaping a
    # short subprocess and capturing its pid post-exit.
    import subprocess as _subp
    dead = _subp.Popen(["/bin/sh", "-c", "exit 0"])
    dead.wait()
    fake_pid = dead.pid

    registry_path = feat._jobs_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        '{"orphan": {"method": "cli_background", "status": "running",'
        ' "pid": ' + str(fake_pid) + ', "label": "orphan",'
        ' "exit_path": "' + str(tmp_path / "nope.exit") + '",'
        ' "log_path": "' + str(tmp_path / "nope.log") + '",'
        ' "command": "x", "started_at": "2026-06-03T00:00:00+00:00"}}'
    )

    fresh = TalonCoordinatorFeature(agent)
    status = await fresh.talon_status()
    job = next(j for j in status.data["jobs"] if j["id"] == "orphan")
    assert job["status"] == "finished_unknown"
    assert job["returncode"] is None


@pytest.mark.asyncio
async def test_job_log_tail_after_restart(tmp_path, monkeypatch):
    """``talon_job_log`` must tail a known durable log after a restart,
    even though the fresh feature never tracked the job in memory.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon-restart-log"
    fake_bin.write_text(
        "#!/bin/sh\necho 'persisted-log-line'\nexit 0\n"
    )
    fake_bin.chmod(0o755)

    agent = SimpleNamespace(
        _scheduler=None, storage_path=str(tmp_path / "agent.db")
    )
    feat = TalonCoordinatorFeature(agent)

    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["dummy"], label="dummy", extra_meta={"repo": "x/y", "issue": 9},
        )

    job_id = result["job_id"]
    await feat._jobs[job_id]["process"].wait()

    fresh = TalonCoordinatorFeature(agent)
    log_result = await fresh.talon_job_log(job_id, lines=50)
    assert log_result.data["success"] is True
    assert "persisted-log-line" in log_result.data["content"]


# ----- Self-modification safeguards -----------------------------------


def test_workspace_path_is_outside_running_source_by_default():
    """Default workspace root must NOT be inside the running agent's
    source tree, regardless of where the user runs from.
    """
    from kestrel_sovereign.features.talon.coordinator import (
        _RUNNING_AGENT_SOURCE_ROOT, _path_contains,
    )
    workspace = TalonCoordinatorFeature._workspace_path_for(
        "KestrelSovereignAI/kestrel-sovereign"
    )
    assert not _path_contains(_RUNNING_AGENT_SOURCE_ROOT, workspace), (
        f"Default workspace {workspace} is inside running source "
        f"{_RUNNING_AGENT_SOURCE_ROOT} — that's the bug we're fixing."
    )


def test_assert_workspace_safe_refuses_running_source(monkeypatch):
    """When KESTREL_TALON_WORKSPACE_ROOT is set to the running agent's
    source root, the resolved workspace path lies INSIDE that source —
    ``_assert_workspace_safe`` rejects, no flag override.
    """
    from kestrel_sovereign.features.talon.coordinator import (
        _RUNNING_AGENT_SOURCE_ROOT,
    )
    monkeypatch.setenv(
        "KESTREL_TALON_WORKSPACE_ROOT", str(_RUNNING_AGENT_SOURCE_ROOT),
    )
    workspace = TalonCoordinatorFeature._workspace_path_for("any/repo")
    reason = TalonCoordinatorFeature._assert_workspace_safe(workspace)
    assert reason is not None
    assert "running agent's source tree" in reason
    # Error message must name the env var so the user knows what to set.
    assert "KESTREL_TALON_WORKSPACE_ROOT" in reason


def test_assert_workspace_safe_refuses_workspace_equal_to_source(monkeypatch):
    """When the workspace path resolves to the running source root
    itself, the error names the constitutional alternatives.
    """
    from kestrel_sovereign.features.talon.coordinator import (
        _RUNNING_AGENT_SOURCE_ROOT,
    )
    reason = TalonCoordinatorFeature._assert_workspace_safe(
        _RUNNING_AGENT_SOURCE_ROOT
    )
    assert reason is not None
    assert "running agent's source tree" in reason
    assert "code_edit" in reason or "propose_improvement" in reason


def test_assert_workspace_safe_refuses_workspace_containing_source():
    """If the workspace path WOULD contain the running agent's source
    (e.g. user set workspace root to the projects parent), refuse.
    """
    from kestrel_sovereign.features.talon.coordinator import (
        _RUNNING_AGENT_SOURCE_ROOT,
    )
    parent = _RUNNING_AGENT_SOURCE_ROOT.parent
    reason = TalonCoordinatorFeature._assert_workspace_safe(parent)
    assert reason is not None


@pytest.mark.asyncio
async def test_claim_refuses_unsafe_workspace_root(monkeypatch):
    """End-to-end: ``talon_claim`` returns ``unsafe_workspace`` when
    the resolved workspace would touch the running source tree.
    """
    from kestrel_sovereign.features.talon.coordinator import (
        _RUNNING_AGENT_SOURCE_ROOT,
    )
    monkeypatch.setenv(
        "KESTREL_TALON_WORKSPACE_ROOT", str(_RUNNING_AGENT_SOURCE_ROOT),
    )
    feat = _make_feature()
    with patch.object(
        TalonCoordinatorFeature, "_dispatch_via_a2a",
        new_callable=AsyncMock,
        return_value={"dispatched": False},
    ):
        result = await feat.talon_claim(repo="self", issue=1)
    assert result.status is ToolResultStatus.ERROR
    assert result.data["dispatched"] is False
    assert result.data["state"] == "unsafe_workspace"


@pytest.mark.asyncio
async def test_setup_workspace_clones_outside_running_source(
    tmp_path, monkeypatch,
):
    """Provisioning a workspace creates a directory under the workspace
    root, not under the running source. Approves auto via stubbed
    SecurityFeature; stubs git clone so we don't hit the network.
    """
    from pathlib import Path as _Path
    from kestrel_sovereign.features.talon.coordinator import (
        _RUNNING_AGENT_SOURCE_ROOT, _path_contains,
    )
    monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    sec = SimpleNamespace(
        name="SecurityFeature",
        approval_queue=SimpleNamespace(
            request_approval=AsyncMock(return_value=(True, "user")),
        ),
    )
    agent = SimpleNamespace(
        _scheduler=None,
        features={"SecurityFeature": sec},
    )
    agent.get_feature = lambda name: sec if name == "SecurityFeature" else None
    feat = TalonCoordinatorFeature(agent)

    async def fake_git_clone(self, url, dest):
        _Path(dest).mkdir(parents=True, exist_ok=True)
        (_Path(dest) / ".git").mkdir(exist_ok=True)
        (_Path(dest) / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        return {"ok": True}

    with patch.object(
        TalonCoordinatorFeature, "_git_clone", new=fake_git_clone,
    ):
        result = await feat.talon_setup_workspace(repo="org/repo")

    assert result.status is ToolResultStatus.OK
    assert result.data["state"] == "created"
    workspace_path = _Path(result.data["workspace"]["path"])
    assert workspace_path.exists()
    assert (workspace_path / ".git").is_dir()
    assert not _path_contains(_RUNNING_AGENT_SOURCE_ROOT, workspace_path)


@pytest.mark.asyncio
async def test_setup_workspace_denied_without_approval(tmp_path, monkeypatch):
    """Workspace setup is approval-gated. Without approval, fails
    with ``approval_denied`` and creates nothing on disk.
    """
    monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    sec = SimpleNamespace(
        name="SecurityFeature",
        approval_queue=SimpleNamespace(
            request_approval=AsyncMock(return_value=(False, "user")),
        ),
    )
    agent = SimpleNamespace(
        _scheduler=None,
        features={"SecurityFeature": sec},
    )
    agent.get_feature = lambda name: sec if name == "SecurityFeature" else None
    feat = TalonCoordinatorFeature(agent)

    result = await feat.talon_setup_workspace(repo="org/repo")
    assert result.status is ToolResultStatus.ERROR
    assert result.data["state"] == "approval_denied"
    assert not (tmp_path / "org__repo" / ".git").exists()


@pytest.mark.asyncio
async def test_workspace_status_reports_missing_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
    feat = _make_feature()
    result = await feat.talon_workspace_status(repo="org/never-cloned")
    # Unprovisioned workspace is PARTIAL (read succeeded, but the
    # workspace isn't ready for talon_claim) per #1061 wave 15.
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["exists"] is False
    assert result.data["is_git"] is False
    assert result.data["safe"] is True


@pytest.mark.asyncio
async def test_workspace_status_reports_unsafe(monkeypatch):
    """If KESTREL_TALON_WORKSPACE_ROOT is unsafe, workspace_status
    surfaces that — agent can react instead of being silently blocked.
    """
    from kestrel_sovereign.features.talon.coordinator import (
        _RUNNING_AGENT_SOURCE_ROOT,
    )
    monkeypatch.setenv(
        "KESTREL_TALON_WORKSPACE_ROOT", str(_RUNNING_AGENT_SOURCE_ROOT),
    )
    feat = _make_feature()
    result = await feat.talon_workspace_status(repo="org/x")
    # Unsafe workspace path is PARTIAL — read succeeded, but the
    # state.safe flag reflects a refusal-to-dispatch condition.
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["safe"] is False
    assert "running agent's source tree" in result.data["unsafe_reason"]
