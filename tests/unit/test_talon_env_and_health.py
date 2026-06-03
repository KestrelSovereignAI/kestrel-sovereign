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
    """Minimal SignalDispatcher stand-in — collects enqueued signals."""

    def __init__(self):
        self.signals = []

    def enqueue_signal(self, signal):
        self.signals.append(signal)
        return None


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

    poll1 = await feat.talon_monitor()
    assert poll1.data["signals_emitted"] == 1
    assert len(dispatcher.signals) == 1
    sig = dispatcher.signals[0]
    assert sig.payload["job_id"] == job_id
    assert sig.payload["status"] == "complete"

    # A second poll with no further state change must NOT re-emit.
    poll2 = await feat.talon_monitor()
    assert poll2.data["signals_emitted"] == 0
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

    poll1 = await feat.talon_monitor()
    assert poll1.data["signals_emitted"] == 1

    # Simulated restart — a fresh feature reloads from disk. The
    # persisted last_signaled_status must prevent re-emission.
    d2 = _CapturingDispatcher()
    fresh_agent = SimpleNamespace(
        _scheduler=None,
        storage_path=str(tmp_path / "agent.db"),
        dispatcher=d2,
    )
    fresh = TalonCoordinatorFeature(fresh_agent)
    poll2 = await fresh.talon_monitor()
    assert poll2.data["signals_emitted"] == 0
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
    assert poll1.data["signals_skipped_no_dispatcher"] == 1
    # last_signaled_status must NOT have been set.
    assert "last_signaled_status" not in feat._jobs[job_id]

    # Second pass with a real dispatcher must still fire.
    dispatcher = _CapturingDispatcher()
    feat.agent.dispatcher = dispatcher
    poll2 = await feat.talon_monitor()
    assert poll2.data["signals_emitted"] == 1
    assert len(dispatcher.signals) == 1
