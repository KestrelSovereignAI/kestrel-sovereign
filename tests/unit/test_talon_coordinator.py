"""Tests for TalonCoordinatorFeature."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.talon.coordinator import TalonCoordinatorFeature


def _make_agent():
    agent = MagicMock()
    agent.agent_name = "kestrel"
    agent._features = []
    agent._scheduler = MagicMock()
    agent._scheduler.remove_schedule = MagicMock()
    agent._scheduler.add_schedule = MagicMock()
    return agent


class TestTalonCoordinatorInit:
    def test_creates_with_agent(self):
        agent = _make_agent()
        feature = TalonCoordinatorFeature(agent)
        assert feature.agent is agent
        assert feature._jobs == {}

    @pytest.mark.asyncio
    async def test_initialize(self):
        feature = TalonCoordinatorFeature(_make_agent())
        await feature.initialize()  # should not raise


class TestTalonClaim:
    @pytest.mark.asyncio
    async def test_claim_mesh_success(self, tmp_path, monkeypatch):
        """When mesh dispatch works, returns dispatched=True."""
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh:
            mock_mesh.return_value = {
                "dispatched": True, "method": "a2a",
                "task_id": "abc", "repo": "org/repo", "issue": 42,
            }
            result = await feature.talon_claim(repo="org/repo", issue=42)
            assert result.status is ToolResultStatus.OK
            assert result.data["dispatched"] is True
            assert result.data["method"] == "a2a"
            mock_mesh.assert_awaited_once_with("org/repo", 42)

    @pytest.mark.asyncio
    async def test_claim_falls_back_to_cli_background(self, tmp_path, monkeypatch):
        """When mesh fails, falls back to background CLI dispatch.

        Requires a provisioned workspace — the safeguard refuses to
        point Talon at the running source tree. The test sets
        KESTREL_TALON_WORKSPACE_ROOT to tmp_path and stubs a ready
        workspace state.
        """
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        feature = TalonCoordinatorFeature(_make_agent())
        ready_state = {
            "repo": "org/repo",
            "path": str(tmp_path / "org__repo"),
            "exists": True,
            "is_git": True,
            "head": "main",
            "clean": True,
            "last_fetch_at": None,
            "safe": True,
        }
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh, \
             patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg, \
             patch.object(TalonCoordinatorFeature, "_workspace_state", return_value=ready_state):
            mock_mesh.return_value = {"dispatched": False, "reason": "no_a2a_host"}
            mock_bg.return_value = {
                "dispatched": True,
                "method": "cli_background",
                "job_id": "abc",
                "pid": 1234,
            }
            result = await feature.talon_claim(repo="org/repo", issue=42)
            assert result.status is ToolResultStatus.OK
            assert result.data["dispatched"] is True
            assert result.data["method"] == "cli_background"
            assert result.data["job_id"] == "abc"
            args = mock_bg.call_args[0][0]
            assert "--worktree" in args
            assert "--repo-dir" in args
            assert "--model" in args and "opus" in args
            assert "--skip-clarification" in args
            repo_dir_idx = args.index("--repo-dir") + 1
            assert str(tmp_path) in args[repo_dir_idx]
            assert mock_bg.call_args.kwargs["env"]["GITHUB_TOKEN"]

    @pytest.mark.asyncio
    async def test_claim_can_dispatch_codex_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        feature = TalonCoordinatorFeature(_make_agent())
        ready_state = {
            "repo": "org/repo",
            "path": str(tmp_path / "org__repo"),
            "exists": True,
            "is_git": True,
            "head": "main",
            "clean": True,
            "last_fetch_at": None,
            "safe": True,
        }
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh, \
             patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg, \
             patch.object(TalonCoordinatorFeature, "_workspace_state", return_value=ready_state):
            mock_mesh.return_value = {"dispatched": False, "reason": "no_a2a_host"}
            mock_bg.return_value = {
                "dispatched": True,
                "method": "cli_background",
                "job_id": "abc",
                "pid": 1234,
            }
            result = await feature.talon_claim(
                repo="org/repo",
                issue=42,
                backend="codex",
                model="gpt-5.4-mini",
                auth_lane="oauth",
            )

        assert result.status is ToolResultStatus.OK
        args = mock_bg.call_args[0][0]
        assert args[args.index("--backend") + 1] == "codex"
        assert args[args.index("--codex-model") + 1] == "gpt-5.4-mini"
        assert "--model" not in args
        meta = mock_bg.call_args.kwargs["extra_meta"]
        assert meta["backend"] == "codex"
        assert meta["model"] == "gpt-5.4-mini"
        assert meta["auth_lane"] == "oauth"
        mock_mesh.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_talon_set_config_updates_preference_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
        feature = TalonCoordinatorFeature(_make_agent())

        result = await feature.talon_set_config(
            default_backend="codex",
            default_model="gpt-5.4-mini",
            default_auth_lane="oauth",
            max_iterations=2,
        )
        loaded = await feature.talon_get_config()

        assert result.status is ToolResultStatus.OK
        assert loaded.data["preference"]["default_backend"] == "codex"
        assert loaded.data["preference"]["default_model"] == "gpt-5.4-mini"
        assert loaded.data["preference"]["max_iterations"] == 2
        assert loaded.data["policy"]["allow_api_billing"] is False

    @pytest.mark.asyncio
    async def test_talon_set_config_parses_string_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
        feature = TalonCoordinatorFeature(_make_agent())

        result = await feature.talon_set_config(
            default_backend="codex",
            default_model="gpt-5.4-mini",
            default_auth_lane="oauth",
            skip_clarification="false",
            self_review="no",
        )
        loaded = await feature.talon_get_config()

        assert result.status is ToolResultStatus.OK
        assert loaded.data["preference"]["skip_clarification"] is False
        assert loaded.data["preference"]["self_review"] is False

    @pytest.mark.asyncio
    async def test_claim_parses_string_false_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        feature = TalonCoordinatorFeature(_make_agent())
        ready_state = {
            "repo": "org/repo",
            "path": str(tmp_path / "org__repo"),
            "exists": True,
            "is_git": True,
            "head": "main",
            "clean": True,
            "last_fetch_at": None,
            "safe": True,
        }
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh, \
             patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg, \
             patch.object(TalonCoordinatorFeature, "_workspace_state", return_value=ready_state):
            mock_mesh.return_value = {"dispatched": False, "reason": "no_a2a_host"}
            mock_bg.return_value = {
                "dispatched": True,
                "method": "cli_background",
                "job_id": "abc",
                "pid": 1234,
            }
            result = await feature.talon_claim(
                repo="org/repo",
                issue=42,
                backend="codex",
                model="gpt-5.4-mini",
                auth_lane="oauth",
                skip_clarification="false",
                self_review="false",
            )

        assert result.status is ToolResultStatus.OK
        args = mock_bg.call_args[0][0]
        assert "--skip-clarification" not in args
        assert "--self-review" not in args

    @pytest.mark.asyncio
    async def test_claim_refuses_when_workspace_not_provisioned(self):
        """The structural safeguard: dispatch refuses when the target
        repo has no workspace clone. No flag overrides this — the
        agent must call ``talon_setup_workspace`` first.
        """
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh, \
             patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg, \
             patch.object(TalonCoordinatorFeature, "_workspace_state", return_value={
                 "exists": False, "is_git": False, "safe": True,
                 "path": "/tmp/no-workspace/org__repo",
             }):
            mock_mesh.return_value = {"dispatched": False}
            result = await feature.talon_claim(repo="org/repo", issue=42)
            assert result.status is ToolResultStatus.ERROR
            assert result.data["dispatched"] is False
            assert result.data["state"] == "workspace_not_provisioned"
            assert "talon_setup_workspace" in result.data["next_step"]
            mock_bg.assert_not_called()


class TestTalonBatch:
    @pytest.mark.asyncio
    async def test_batch_with_prd(self):
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg:
            mock_bg.return_value = {"dispatched": True, "method": "cli_background"}
            result = await feature.talon_batch(repo="org/repo", prd="prd.json")
            assert result.status is ToolResultStatus.OK
            assert result.data["dispatched"] is True
            mock_bg.assert_awaited_once()
            args = mock_bg.call_args[0][0]
            assert "batch" in args
            assert "--prd" in args

    @pytest.mark.asyncio
    async def test_batch_with_label(self):
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg:
            mock_bg.return_value = {"dispatched": True, "method": "cli_background"}
            result = await feature.talon_batch(repo="org/repo", label="P0")
            assert result.status is ToolResultStatus.OK
            assert result.data["dispatched"] is True
            args = mock_bg.call_args[0][0]
            assert "--label" in args
            assert "P0" in args

    @pytest.mark.asyncio
    async def test_batch_no_args_errors(self):
        feature = TalonCoordinatorFeature(_make_agent())
        result = await feature.talon_batch(repo="org/repo")
        assert result.status is ToolResultStatus.ERROR
        assert result.data["dispatched"] is False
        assert "error" in result.data


class TestTalonStatus:
    @pytest.mark.asyncio
    async def test_status_empty(self):
        feature = TalonCoordinatorFeature(_make_agent())
        result = await feature.talon_status()
        assert result.status is ToolResultStatus.OK
        assert result.data["running"] == 0
        assert result.data["completed"] == 0

    @pytest.mark.asyncio
    async def test_status_with_jobs(self):
        feature = TalonCoordinatorFeature(_make_agent())
        feature._jobs["job-1"] = {"repo": "a/b", "issue": 1, "status": "dispatched"}
        feature._jobs["job-2"] = {"repo": "a/b", "issue": 2, "status": "complete"}
        result = await feature.talon_status()
        assert result.status is ToolResultStatus.OK
        assert result.data["running"] == 1
        assert result.data["completed"] == 1


class TestTalonPauseResume:
    @pytest.mark.asyncio
    async def test_pause(self):
        agent = _make_agent()
        feature = TalonCoordinatorFeature(agent)
        result = await feature.talon_pause()
        assert result.status is ToolResultStatus.OK
        assert result.data["paused"] is True
        agent._scheduler.remove_schedule.assert_called_once_with("signal_dispatch")

    @pytest.mark.asyncio
    async def test_resume(self):
        agent = _make_agent()
        feature = TalonCoordinatorFeature(agent)
        result = await feature.talon_resume()
        assert result.status is ToolResultStatus.OK
        assert result.data["resumed"] is True
        agent._scheduler.add_schedule.assert_called_once()


class TestA2ADispatch:
    @pytest.mark.asyncio
    async def test_no_host_url(self):
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_discover_host_url", return_value=None):
            result = await feature._dispatch_via_a2a("org/repo", 42)
            assert result["dispatched"] is False
            assert result["reason"] == "no_a2a_host"


class TestCLIDispatch:
    @pytest.mark.asyncio
    async def test_no_binary(self):
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(
            TalonCoordinatorFeature, "_find_talon_bin", return_value=None
        ):
            result = await feature._dispatch_via_cli(["claim", "--repo", "a/b", "--issue", "1"])
            assert result["dispatched"] is False
            assert "not found" in result["error"]


class TestTalonWait:
    @pytest.mark.asyncio
    async def test_unknown_job(self):
        feature = TalonCoordinatorFeature(_make_agent())
        result = await feature.talon_wait(job_id="nope", timeout_seconds=0)
        assert result.status is ToolResultStatus.ERROR
        assert "Unknown job_id" in result.error

    @pytest.mark.asyncio
    async def test_terminal_before_timeout_complete(self, tmp_path):
        feature = TalonCoordinatorFeature(_make_agent())
        log = tmp_path / "job.log"
        log.write_text("line-1\nline-2\nall done\n")
        feature._jobs["job-x"] = {
            "method": "cli_background",
            "status": "complete",
            "returncode": 0,
            "log_path": str(log),
        }
        result = await feature.talon_wait(
            job_id="job-x", timeout_seconds=30, poll_interval_seconds=1,
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "complete"
        assert result.data["returncode"] == 0
        assert result.data["timed_out"] is False
        assert "all done" in result.data["log_tail"]

    @pytest.mark.asyncio
    async def test_terminal_before_timeout_failed(self):
        feature = TalonCoordinatorFeature(_make_agent())
        feature._jobs["job-f"] = {
            "method": "cli_background",
            "status": "failed",
            "returncode": 2,
        }
        result = await feature.talon_wait(
            job_id="job-f", timeout_seconds=30,
        )
        assert result.status is ToolResultStatus.ERROR
        assert result.data["status"] == "failed"
        assert result.data["returncode"] == 2
        assert result.data["timed_out"] is False

    @pytest.mark.asyncio
    async def test_timeout_while_running(self):
        feature = TalonCoordinatorFeature(_make_agent())
        # method=a2a so _reap_cli_job is a no-op and the job stays
        # 'running'; timeout_seconds=0 returns after a single poll with
        # no sleep.
        feature._jobs["job-r"] = {"method": "a2a", "status": "running"}
        result = await feature.talon_wait(
            job_id="job-r", timeout_seconds=0, poll_interval_seconds=1,
        )
        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["status"] == "running"
        assert result.data["timed_out"] is True
        assert result.data["timeout_seconds"] == 0

    @pytest.mark.asyncio
    async def test_max_duration_rejected(self):
        feature = TalonCoordinatorFeature(_make_agent())
        feature._jobs["job-r"] = {"method": "a2a", "status": "running"}
        too_long = TalonCoordinatorFeature._TALON_WAIT_MAX_SECONDS + 1
        result = await feature.talon_wait(
            job_id="job-r", timeout_seconds=too_long,
        )
        assert result.status is ToolResultStatus.ERROR
        assert "exceeds the maximum" in result.error
        assert result.data["max_seconds"] == (
            TalonCoordinatorFeature._TALON_WAIT_MAX_SECONDS
        )

    @pytest.mark.asyncio
    async def test_reaps_cli_job_via_sidecar(self, tmp_path):
        """A running cli_background job whose exit sidecar appears is
        reaped to terminal state during the wait (shared _reap_cli_job).
        """
        agent = _make_agent()
        # Isolate the durable registry under tmp_path so _persist_jobs
        # does not write to a shared relative path (storage_path drives
        # _job_log_dir, not KESTREL_HOME).
        agent.storage_path = str(tmp_path / "agent_data" / "kestrel_prime.db")
        feature = TalonCoordinatorFeature(agent)
        exit_file = tmp_path / "job.exit"
        exit_file.write_text("0")
        feature._jobs["job-s"] = {
            "method": "cli_background",
            "status": "running",
            "process": None,
            "exit_path": str(exit_file),
            "pid": 999999,
        }
        result = await feature.talon_wait(
            job_id="job-s", timeout_seconds=30, poll_interval_seconds=1,
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "complete"
        assert result.data["returncode"] == 0

    @pytest.mark.asyncio
    async def test_reconciles_a2a_job_completed(self, tmp_path):
        """An a2a job reaches terminal state during the wait when Talon's
        task endpoint reports COMPLETED — the default dispatch transport.

        Without the shared _reconcile_a2a_job call in the poll loop this
        would time out (a2a is a no-op for _reap_cli_job).
        """
        agent = _make_agent()
        agent.storage_path = str(tmp_path / "agent_data" / "kestrel_prime.db")
        feature = TalonCoordinatorFeature(agent)
        feature._jobs["job-a"] = {"method": "a2a", "status": "running"}

        completed_payload = json.dumps(
            {"status": {"state": "completed"}}
        ).encode()

        class _Resp:
            def read(self):
                return completed_payload

        with patch.object(
            feature, "_discover_host_url", return_value="http://localhost:9999"
        ), patch(
            "kestrel_sovereign.features.talon.coordinator.urllib.request.urlopen",
            return_value=_Resp(),
        ):
            result = await feature.talon_wait(
                job_id="job-a", timeout_seconds=30, poll_interval_seconds=1,
            )
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "complete"
        assert result.data["timed_out"] is False

    @pytest.mark.asyncio
    async def test_reconciles_a2a_job_failed(self, tmp_path):
        """An a2a job whose task endpoint reports FAILED ends the wait in
        the 'failed' terminal state with an ERROR result.
        """
        agent = _make_agent()
        agent.storage_path = str(tmp_path / "agent_data" / "kestrel_prime.db")
        feature = TalonCoordinatorFeature(agent)
        feature._jobs["job-b"] = {"method": "a2a", "status": "running"}

        failed_payload = json.dumps(
            {"status": {"state": "failed", "message": {"text": "boom"}}}
        ).encode()

        class _Resp:
            def read(self):
                return failed_payload

        with patch.object(
            feature, "_discover_host_url", return_value="http://localhost:9999"
        ), patch(
            "kestrel_sovereign.features.talon.coordinator.urllib.request.urlopen",
            return_value=_Resp(),
        ):
            result = await feature.talon_wait(
                job_id="job-b", timeout_seconds=30, poll_interval_seconds=1,
            )
        assert result.status is ToolResultStatus.ERROR
        assert result.data["status"] == "failed"
        assert result.data["timed_out"] is False
        assert feature._jobs["job-b"]["error"] == "boom"
