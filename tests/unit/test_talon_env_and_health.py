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

from kestrel_sdk.tools.result import ToolResultStatus

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


# ----- talon_monitor (#1510) -----------------------------------------


class _CapturingDispatcher:
    """Minimal SignalDispatcher stand-in — collects enqueued signals
    and returns SignalHandles whose task is already done with a
    configurable SignalResult.

    ``talon_monitor`` calls ``enqueue_signal`` (fire-and-forget) and
    harvests delivery outcomes via ``handle.task.result()`` on the
    NEXT poll (#1528 P1 — block-free cron). Default outcome is
    ``Status.OK``; per-instance overrides drive the delivery-
    classification tests.

    By default the handle's task is already complete, so a single
    monitor call sees the result immediately (Step 0 in the same
    poll picks up Step 2 from the same poll — works because the
    test ordering harvests pending tasks at top of poll AFTER they
    were created at the bottom of the prior poll). To simulate an
    in-flight task, pass ``pending=True``.
    """

    def __init__(
        self, status_override=None, error_override=None, pending=False,
    ):
        from kestrel_sdk.signals import Status
        self.signals = []
        self._status = status_override or Status.OK
        self._error = error_override
        self._pending = pending

    async def enqueue_signal(self, signal):
        from kestrel_sdk.signals.models import SignalHandle, SignalResult
        self.signals.append(signal)

        async def _coro():
            return SignalResult(
                signal_id=signal.id,
                status=self._status,
                mode=signal.mode,
                duration_ms=1,
                error=self._error,
            )

        task = asyncio.create_task(_coro())
        if not self._pending:
            # Drive the task to completion before returning so the
            # caller can immediately observe the result via
            # task.result(). Mirrors the synchronous-feeling fast-
            # path on a dispatcher whose work is purely in-memory.
            await task
        return SignalHandle(signal_id=signal.id, task=task)


@pytest.mark.asyncio
async def test_monitor_emits_one_signal_per_transition(tmp_path, monkeypatch):
    """When a CLI background job transitions from running to a
    terminal state, talon_monitor must enqueue exactly one signal —
    and a subsequent poll with no further state change must not
    enqueue a second one.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon-mon"
    fake_bin.write_text("#!/bin/sh\necho 'mon-out'\nexit 0\n")
    fake_bin.chmod(0o755)

    dispatcher = _CapturingDispatcher()
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=dispatcher,
    )
    feat = TalonCoordinatorFeature(agent)

    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["x"], label="x", extra_meta={"repo": "x/y", "issue": 11},
        )

    job_id = result["job_id"]
    await feat._jobs[job_id]["process"].wait()
    # Detach the live process handle so the monitor's terminal-state
    # path (sidecar / pid liveness) is exercised — same code path
    # that would run after a Kestrel restart.
    feat._jobs[job_id]["process"] = None

    # Two-phase semantics (#1528): poll 1 enqueues, poll 2 harvests
    # and counts the delivery.
    poll1 = await feat.talon_monitor()
    assert poll1.data["signals_enqueued"] == 1
    assert poll1.data["signals_emitted"] == 0
    assert len(dispatcher.signals) == 1
    sig = dispatcher.signals[0]
    assert sig.payload["job_id"] == job_id
    assert sig.payload["status"] == "complete"

    poll2 = await feat.talon_monitor()
    assert poll2.data["signals_emitted"] == 1
    assert poll2.data["signals_enqueued"] == 0

    # A third poll with no further state change must NOT re-emit.
    poll3 = await feat.talon_monitor()
    assert poll3.data["signals_emitted"] == 0
    assert poll3.data["signals_enqueued"] == 0
    assert len(dispatcher.signals) == 1


@pytest.mark.asyncio
async def test_monitor_no_signal_for_in_flight_jobs(tmp_path, monkeypatch):
    """A running job (process handle, no return code yet) must not
    trigger a signal. Only terminal transitions wake cognition.
    """
    dispatcher = _CapturingDispatcher()
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=dispatcher,
    )
    feat = TalonCoordinatorFeature(agent)

    # Hand-build a running-job record.
    feat._jobs["live"] = {
        "method": "cli_background",
        "status": "running",
        "pid": os.getpid(),
        "process": None,
        "exit_path": str(tmp_path / "live.exit"),  # doesn't exist
        "log_path": str(tmp_path / "live.log"),
        "started_at": "2026-06-03T00:00:00+00:00",
        "label": "live",
    }
    result = await feat.talon_monitor()
    assert result.data["signals_emitted"] == 0
    assert dispatcher.signals == []
    # Status must still classify as running so the next poll keeps
    # watching.
    assert feat._jobs["live"]["status"] == "running"


@pytest.mark.asyncio
async def test_monitor_signal_survives_simulated_restart(tmp_path, monkeypatch):
    """When the durable registry records last_signaled_status, a
    fresh feature constructed after a Kestrel restart must NOT
    re-emit the signal for jobs that already woke a prior cognition.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon-restart-mon"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    d1 = _CapturingDispatcher()
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=d1,
    )
    feat = TalonCoordinatorFeature(agent)
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["x"], label="x", extra_meta={"repo": "x/y", "issue": 12},
        )
    job_id = result["job_id"]
    await feat._jobs[job_id]["process"].wait()
    feat._jobs[job_id]["process"] = None

    # Two polls to enqueue + harvest the wake.
    await feat.talon_monitor()
    poll2 = await feat.talon_monitor()
    assert poll2.data["signals_emitted"] == 1

    # Simulated restart — a fresh feature reloads from disk. The
    # persisted last_signaled_status must prevent re-emission.
    d2 = _CapturingDispatcher()
    fresh_agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=d2,
    )
    fresh = TalonCoordinatorFeature(fresh_agent)
    poll3 = await fresh.talon_monitor()
    assert poll3.data["signals_emitted"] == 0
    assert poll3.data["signals_enqueued"] == 0
    assert d2.signals == []


@pytest.mark.asyncio
async def test_monitor_finished_unknown_emits_signal(tmp_path, monkeypatch):
    """A finished_unknown transition (dead pid, no sidecar) must
    still wake the agent. The status itself is the actionable
    information — "we don't know how it ended".
    """
    dispatcher = _CapturingDispatcher()
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=dispatcher,
    )
    feat = TalonCoordinatorFeature(agent)

    # Spawn-and-reap a real child so the pid we record is genuinely
    # dead at monitor time (pid reuse risk is tolerable here — the
    # window between reap and monitor is microseconds in this test).
    import subprocess as _subp
    dead = _subp.Popen(["/bin/sh", "-c", "exit 0"])
    dead.wait()

    feat._jobs["orphan"] = {
        "method": "cli_background",
        "status": "running",
        "pid": dead.pid,
        "process": None,
        "exit_path": str(tmp_path / "missing.exit"),
        "log_path": str(tmp_path / "missing.log"),
        "started_at": "2026-06-03T00:00:00+00:00",
        "label": "orphan",
    }

    # Two polls to enqueue then harvest.
    await feat.talon_monitor()
    result = await feat.talon_monitor()
    assert result.data["signals_emitted"] == 1
    sig = dispatcher.signals[0]
    assert sig.payload["status"] == "finished_unknown"


@pytest.mark.asyncio
async def test_monitor_no_dispatcher_does_not_mark_signaled(tmp_path, monkeypatch):
    """If the agent has no dispatcher, the monitor must NOT mark
    a transition as already-signaled — otherwise a delayed
    dispatcher registration would silently lose the wake. Next poll
    with a real dispatcher should still fire.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    fake_bin = tmp_path / "kestrel-talon-no-disp"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    # First feature has NO dispatcher at all.
    agent = SimpleNamespace(
        _scheduler=None, storage_path=str(tmp_path / "agent.db"),
    )
    feat = TalonCoordinatorFeature(agent)
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin", return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["x"], label="x", extra_meta={"repo": "x/y", "issue": 14},
        )
    job_id = result["job_id"]
    await feat._jobs[job_id]["process"].wait()
    feat._jobs[job_id]["process"] = None

    poll1 = await feat.talon_monitor()
    assert poll1.data["signals_emitted"] == 0
    assert poll1.data["signals_enqueued"] == 0
    assert poll1.data["signals_skipped_no_dispatcher"] == 1
    # last_signaled_status must NOT have been set.
    assert "last_signaled_status" not in feat._jobs[job_id]

    # Second pass with a real dispatcher: poll enqueues, next poll
    # harvests the delivery.
    dispatcher = _CapturingDispatcher()
    feat.agent.dispatcher = dispatcher
    poll2 = await feat.talon_monitor()
    assert poll2.data["signals_enqueued"] == 1
    poll3 = await feat.talon_monitor()
    assert poll3.data["signals_emitted"] == 1
    assert len(dispatcher.signals) == 1


# ----- talon_monitor delivery classification (#1528) -----------------


async def _spawn_completed_job(feat, tmp_path, repo="x/y", issue=10):
    """Helper: dispatch a fake CLI job, wait for the wrapper to exit,
    and detach the live process handle so the monitor exercises the
    same code path that runs after a Kestrel restart.
    """
    fake_bin = tmp_path / "kestrel-talon-classify"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    with patch.object(
        TalonCoordinatorFeature, "_find_talon_bin",
        return_value=str(fake_bin),
    ):
        result = await feat._dispatch_via_cli_background(
            ["x"], label="x",
            extra_meta={"repo": repo, "issue": issue},
        )
    job_id = result["job_id"]
    await feat._jobs[job_id]["process"].wait()
    feat._jobs[job_id]["process"] = None
    return job_id


@pytest.mark.asyncio
async def test_monitor_records_ok_as_delivered(tmp_path, monkeypatch):
    """Dispatcher returns Status.OK → cognition fired. Monitor must
    mark the job as both signaled AND delivered, expose the
    delivery_status and last_delivered_at on the job record.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    dispatcher = _CapturingDispatcher()
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=dispatcher,
    )
    feat = TalonCoordinatorFeature(agent)
    job_id = await _spawn_completed_job(feat, tmp_path)

    # Poll 1 enqueues. Poll 2 harvests the (already-complete) task.
    poll1 = await feat.talon_monitor()
    assert poll1.data["signals_enqueued"] == 1
    assert poll1.data["signals_emitted"] == 0
    info = feat._jobs[job_id]
    assert info["pending_signal_id"] == dispatcher.signals[0].id
    assert info["pending_signaled_target"] == "complete"

    poll = await feat.talon_monitor()
    assert poll.data["signals_emitted"] == 1
    assert poll.data["signals_hard_failed"] == 0
    assert poll.data["signals_soft_failed"] == 0
    info = feat._jobs[job_id]
    assert info["last_signaled_status"] == "complete"
    assert info["last_delivery_status"] == "ok"
    assert "last_delivered_at" in info
    assert info["last_delivery_attempts"] == 1
    assert "pending_signal_id" not in info
    # Transition payload includes the delivery_status for operators.
    assert poll.data["transitions"][0]["delivery_status"] == "ok"


@pytest.mark.asyncio
async def test_monitor_soft_fail_does_not_mark_signaled(tmp_path, monkeypatch):
    """DROPPED_RATE_LIMIT / DROPPED_QUIET_HOURS / FAILED are transient.
    Monitor must record the failure but NOT mark last_signaled_status,
    so the next poll retries.
    """
    from kestrel_sdk.signals import Status
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    quiet_dispatcher = _CapturingDispatcher(
        status_override=Status.DROPPED_QUIET_HOURS,
        error_override="inside quiet window",
    )
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=quiet_dispatcher,
    )
    feat = TalonCoordinatorFeature(agent)
    job_id = await _spawn_completed_job(feat, tmp_path)

    # Poll 1 enqueues (attempt 1). Poll 2 harvests the soft fail
    # AND its Step 2 re-enqueues (attempt 2). Each subsequent poll
    # harvests the prior attempt's failure and enqueues the next.
    await feat.talon_monitor()
    poll = await feat.talon_monitor()
    assert poll.data["signals_emitted"] == 0
    assert poll.data["signals_soft_failed"] == 1
    info = feat._jobs[job_id]
    assert "last_signaled_status" not in info, (
        "soft fail must NOT mark signaled — next poll should retry"
    )
    assert info["last_delivery_status"] == "dropped_quiet_hours"
    assert info["last_delivery_error"] == "inside quiet window"
    # Each retry uses a fresh attempt suffix so the dispatcher's
    # coalescing window does NOT swallow the retry as COALESCED
    # against a prior soft-failed attempt (codex round 1 P1).
    keys = [s.dedupe_key for s in quiet_dispatcher.signals]
    assert len(set(keys)) == len(keys), (
        f"every retry needs a unique dedupe_key; got {keys!r}"
    )
    assert all(k.endswith(f":attempt-{i+1}") for i, k in enumerate(keys))

    # Swap to a working dispatcher; the very next poll-pair should
    # finally deliver (with yet-another fresh attempt suffix).
    feat.agent.dispatcher = _CapturingDispatcher()
    await feat.talon_monitor()  # harvest last quiet fail, enqueue against fresh
    harvest = await feat.talon_monitor()
    assert harvest.data["signals_emitted"] == 1
    info = feat._jobs[job_id]
    assert info["last_signaled_status"] == "complete"
    assert info["last_delivery_status"] == "ok"


@pytest.mark.asyncio
async def test_monitor_hard_fail_marks_signaled(tmp_path, monkeypatch):
    """DROPPED_VALIDATION / DROPPED_CYCLE are permanent. Monitor must
    mark the job as signaled so we don't loop forever on the same
    broken signal, AND must surface the failure separately so a
    human can inspect.
    """
    from kestrel_sdk.signals import Status
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    bad_dispatcher = _CapturingDispatcher(
        status_override=Status.DROPPED_VALIDATION,
        error_override="schema mismatch",
    )
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=bad_dispatcher,
    )
    feat = TalonCoordinatorFeature(agent)
    job_id = await _spawn_completed_job(feat, tmp_path)

    # Poll 1 enqueues, poll 2 harvests + classifies as hard fail.
    await feat.talon_monitor()
    poll = await feat.talon_monitor()
    assert poll.data["signals_emitted"] == 0
    assert poll.data["signals_hard_failed"] == 1
    info = feat._jobs[job_id]
    # Hard fail still locks the signaled flag — retry would just
    # re-fail with the same validation error.
    assert info["last_signaled_status"] == "complete"
    assert info["last_delivery_status"] == "dropped_validation"
    assert info["last_delivery_error"] == "schema mismatch"
    # Transition record exposes the failure to the operator.
    t = poll.data["transitions"][0]
    assert t["delivery_status"] == "dropped_validation"
    assert t["delivery_error"] == "schema mismatch"


@pytest.mark.asyncio
async def test_monitor_coalesced_counts_as_delivered(tmp_path, monkeypatch):
    """COALESCED means the wake fired via an adjacent signal in the
    coalescing window. The agent did wake; the monitor must treat
    this as delivered.
    """
    from kestrel_sdk.signals import Status
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    dispatcher = _CapturingDispatcher(
        status_override=Status.COALESCED,
    )
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=dispatcher,
    )
    feat = TalonCoordinatorFeature(agent)
    job_id = await _spawn_completed_job(feat, tmp_path)

    await feat.talon_monitor()
    poll = await feat.talon_monitor()
    assert poll.data["signals_emitted"] == 1
    info = feat._jobs[job_id]
    assert info["last_signaled_status"] == "complete"
    assert info["last_delivery_status"] == "coalesced"


@pytest.mark.asyncio
async def test_monitor_dispatcher_raises_records_failure(tmp_path, monkeypatch):
    """If dispatch_signal raises (dispatcher bug, not a normal
    Status return), the monitor must record dispatcher_raised AND
    leave the job re-tryable.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    class _BrokenDispatcher:
        async def enqueue_signal(self, signal):
            raise RuntimeError("boom")

    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=_BrokenDispatcher(),
    )
    feat = TalonCoordinatorFeature(agent)
    job_id = await _spawn_completed_job(feat, tmp_path)

    # enqueue_signal raises synchronously → soft-fail recorded in
    # the SAME poll, no two-phase needed.
    poll = await feat.talon_monitor()
    assert poll.data["signals_emitted"] == 0
    assert poll.data["signals_soft_failed"] == 1
    info = feat._jobs[job_id]
    assert "last_signaled_status" not in info
    assert info["last_delivery_status"] == "dispatcher_raised"
    assert "RuntimeError" in info["last_delivery_error"]


@pytest.mark.asyncio
async def test_monitor_retry_attempt_counter_in_dedupe_key(tmp_path, monkeypatch):
    """Each retry must use a fresh ``:attempt-N`` suffix on the
    dedupe_key so the dispatcher's coalescing window does not
    swallow the retry as COALESCED against the prior soft-failed
    attempt (codex round 1 P1).
    """
    from kestrel_sdk.signals import Status
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    rate_limited = _CapturingDispatcher(
        status_override=Status.DROPPED_RATE_LIMIT,
        error_override="cap hit",
    )
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=rate_limited,
    )
    feat = TalonCoordinatorFeature(agent)
    await _spawn_completed_job(feat, tmp_path)

    # Each ``talon_monitor`` poll harvests the previous (failed)
    # attempt AND re-enqueues a fresh attempt, so 3 polls produce
    # 3 distinct enqueues with attempt-1, attempt-2, attempt-3
    # suffixes. Each retry MUST carry a unique dedupe_key.
    for _ in range(3):
        await feat.talon_monitor()
    keys = [s.dedupe_key for s in rate_limited.signals]
    assert len(keys) == 3
    assert len(set(keys)) == 3, (
        f"each retry needs a unique dedupe_key; got {keys!r}"
    )
    for i, key in enumerate(keys):
        assert key.endswith(f":attempt-{i+1}"), key


@pytest.mark.asyncio
async def test_monitor_retry_cap_locks_signaled_after_max_attempts(
    tmp_path, monkeypatch,
):
    """A deterministically broken signal would otherwise loop
    forever. After MAX_DELIVERY_ATTEMPTS soft failures the monitor
    must lock signaled and surface ``max_attempts_exceeded``
    (codex round 1 P2).
    """
    from kestrel_sdk.signals import Status
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    broken_dispatcher = _CapturingDispatcher(
        status_override=Status.FAILED,
        error_override="upstream dead",
    )
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=broken_dispatcher,
    )
    feat = TalonCoordinatorFeature(agent)
    job_id = await _spawn_completed_job(feat, tmp_path)

    # Each poll-pair burns one attempt. Run 11 enqueue/harvest pairs;
    # the 11th MUST detect attempts >= MAX and synthesize a hard
    # fail without re-enqueueing.
    for _ in range(11):
        await feat.talon_monitor()
        await feat.talon_monitor()

    info = feat._jobs[job_id]
    assert info["last_signaled_status"] == "complete"
    # Final state is either the harvested FAILED (if cap hit during
    # harvest) or max_attempts_exceeded (if cap hit before enqueue).
    assert info["last_delivery_status"] in (
        "failed", "max_attempts_exceeded",
    )
    # No more attempts beyond cap.
    assert info["last_delivery_attempts"] <= 10


@pytest.mark.asyncio
async def test_monitor_pending_lost_at_restart(tmp_path, monkeypatch):
    """If pending_signal_id is persisted but no in-memory handle
    exists (Kestrel restarted mid-flight), Step 0 of the next
    monitor poll must mark the row as ``lost_at_restart`` and
    leave ``last_signaled_status`` unset so the next poll re-emits.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=_CapturingDispatcher(),
    )
    feat = TalonCoordinatorFeature(agent)

    # Hand-seed a job row that looks like one that survived restart
    # mid-dispatch: terminal status persisted, pending_signal_id
    # present, but no in-memory handle.
    feat._jobs["lost"] = {
        "method": "cli_background",
        "status": "complete",
        "pid": 1,
        "process": None,
        "exit_path": str(tmp_path / "lost.exit"),
        "log_path": str(tmp_path / "lost.log"),
        "started_at": "2026-06-03T00:00:00+00:00",
        "label": "lost",
        "pending_signal_id": "00000000000000000000000000000000",
        "pending_signaled_target": "complete",
        "pending_signal_enqueued_at": "2026-06-03T00:00:00+00:00",
        "last_delivery_attempts": 1,
    }

    poll1 = await feat.talon_monitor()
    # Step 0 swept the stale pending row → soft_failed counter ticks,
    # last_delivery_status records the lost_at_restart cause. Then
    # Step 2 of the same poll re-detected the still-terminal job
    # and enqueued a fresh attempt (against the new in-memory
    # handle), so pending_signal_id is set again — this time it
    # IS live and the next poll will harvest it normally.
    assert poll1.data["signals_soft_failed"] == 1
    assert poll1.data["signals_enqueued"] == 1
    info = feat._jobs["lost"]
    assert info["last_delivery_status"] == "lost_at_restart"
    assert "last_signaled_status" not in info
    # Step 2's fresh enqueue set a NEW pending_signal_id (not the
    # original stale "000…" placeholder).
    assert info["pending_signal_id"] != "00000000000000000000000000000000"
    # And we have an in-memory handle this time.
    assert "lost" in feat._pending_signal_tasks


@pytest.mark.asyncio
async def test_monitor_does_not_re_enqueue_while_pending(tmp_path, monkeypatch):
    """When a signal is in-flight (pending), the monitor must NOT
    re-emit for the same job. Coverage for the case where the
    dispatcher's task hasn't completed yet.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    # ``pending=True`` keeps the SignalHandle's task incomplete.
    pending_dispatcher = _CapturingDispatcher(pending=True)
    agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=pending_dispatcher,
    )
    feat = TalonCoordinatorFeature(agent)
    job_id = await _spawn_completed_job(feat, tmp_path)

    poll1 = await feat.talon_monitor()
    assert poll1.data["signals_enqueued"] == 1

    poll2 = await feat.talon_monitor()
    # Task still pending → Step 0 doesn't harvest, Step 2 doesn't
    # re-enqueue (the same job is already in _pending_signal_tasks).
    assert poll2.data["signals_enqueued"] == 0
    assert poll2.data["signals_emitted"] == 0
    assert len(pending_dispatcher.signals) == 1
    assert feat._jobs[job_id]["pending_signal_id"] is not None
