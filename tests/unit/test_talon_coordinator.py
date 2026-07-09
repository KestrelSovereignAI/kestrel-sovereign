"""Tests for TalonCoordinatorFeature."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.talon.coordinator import TalonCoordinatorFeature
from kestrel_sovereign.features.talon.verification import CommandExecution
from kestrel_sovereign.features.talon.wait_provider import TalonWaitable
from kestrel_sovereign.waits import run_wait_loop
from kestrel_sovereign.waits.engine import MAX_HANDLE_WAIT_SECONDS


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

    @staticmethod
    def _ready_state(tmp_path):
        return {
            "repo": "org/repo",
            "path": str(tmp_path / "org__repo"),
            "exists": True,
            "is_git": True,
            "head": "main",
            "clean": True,
            "last_fetch_at": None,
            "safe": True,
        }

    @pytest.mark.asyncio
    async def test_claim_explicit_self_review_forces_cli(
        self, tmp_path, monkeypatch
    ):
        """Codex P2: an explicitly-set self_review must never ride A2A.

        The A2A payload carries repo/issue only, so choosing that path
        would silently drop the flag and apply the daemon default. With
        A2A available, an explicit self_review must go CLI with the flag
        in argv.
        """
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh, \
             patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg, \
             patch.object(TalonCoordinatorFeature, "_workspace_state",
                          return_value=self._ready_state(tmp_path)):
            # A2A is AVAILABLE — the wrong path would succeed silently.
            mock_mesh.return_value = {
                "dispatched": True, "method": "a2a", "task_id": "t1",
            }
            mock_bg.return_value = {
                "dispatched": True, "method": "cli_background",
                "job_id": "abc", "pid": 1234,
            }
            result = await feature.talon_claim(
                repo="org/repo", issue=42, self_review=True,
            )
        assert result.status is ToolResultStatus.OK
        mock_mesh.assert_not_awaited()
        mock_bg.assert_awaited_once()
        argv = mock_bg.call_args[0][0]
        assert "--self-review" in argv
        assert mock_bg.call_args.kwargs["env"]["GITHUB_TOKEN"]

    @pytest.mark.asyncio
    async def test_claim_explicit_self_review_false_forces_cli(
        self, tmp_path, monkeypatch
    ):
        """Explicit False is still explicit: it must reach the CLI argv
        (as the ABSENCE of --self-review), not be dropped on A2A where the
        daemon default (possibly True) would silently apply."""
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh, \
             patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg, \
             patch.object(TalonCoordinatorFeature, "_workspace_state",
                          return_value=self._ready_state(tmp_path)):
            mock_mesh.return_value = {
                "dispatched": True, "method": "a2a", "task_id": "t1",
            }
            mock_bg.return_value = {
                "dispatched": True, "method": "cli_background",
                "job_id": "abc", "pid": 1234,
            }
            result = await feature.talon_claim(
                repo="org/repo", issue=42, self_review=False,
            )
        assert result.status is ToolResultStatus.OK
        mock_mesh.assert_not_awaited()
        mock_bg.assert_awaited_once()
        assert "--self-review" not in mock_bg.call_args[0][0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kwargs, argv_probe", [
        ({"skip_clarification": True}, "--skip-clarification"),
        ({"max_iterations": 5}, "5"),
        ({"max_turns": 99}, "99"),
        ({"demo_check": True}, "--demo-check"),
        ({"eye_check": True}, "--eye-check"),
    ])
    async def test_claim_any_explicit_per_run_flag_forces_cli(
        self, tmp_path, monkeypatch, kwargs, argv_probe
    ):
        """Every explicit per-run control has the same silent-drop hole on
        A2A; all of them must force the CLI path that carries them."""
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh, \
             patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg, \
             patch.object(TalonCoordinatorFeature, "_workspace_state",
                          return_value=self._ready_state(tmp_path)):
            mock_mesh.return_value = {
                "dispatched": True, "method": "a2a", "task_id": "t1",
            }
            mock_bg.return_value = {
                "dispatched": True, "method": "cli_background",
                "job_id": "abc", "pid": 1234,
            }
            result = await feature.talon_claim(
                repo="org/repo", issue=42, **kwargs,
            )
        assert result.status is ToolResultStatus.OK
        mock_mesh.assert_not_awaited()
        mock_bg.assert_awaited_once()
        assert argv_probe in mock_bg.call_args[0][0]

    @pytest.mark.asyncio
    async def test_claim_explicit_worktree_opt_out_never_rides_a2a(
        self, tmp_path, monkeypatch
    ):
        """worktree=False is an explicit per-run control too: it must not
        ride A2A (which would silently ignore it). Under the default
        require_worktree policy the CLI path then rejects it loudly."""
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh, \
             patch.object(TalonCoordinatorFeature, "_workspace_state",
                          return_value=self._ready_state(tmp_path)):
            mock_mesh.return_value = {
                "dispatched": True, "method": "a2a", "task_id": "t1",
            }
            result = await feature.talon_claim(
                repo="org/repo", issue=42, worktree=False,
            )
        mock_mesh.assert_not_awaited()
        assert result.data["dispatched"] is False
        assert result.data["state"] == "talon_policy_rejected"

    @pytest.mark.asyncio
    async def test_claim_bare_still_prefers_a2a(self, tmp_path, monkeypatch):
        """Control: a claim with NO explicit per-run flags keeps the
        A2A-preferred path — the force-CLI condition must not over-trigger."""
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_a2a", new_callable=AsyncMock) as mock_mesh, \
             patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg:
            mock_mesh.return_value = {
                "dispatched": True, "method": "a2a",
                "task_id": "t1", "repo": "org/repo", "issue": 42,
            }
            result = await feature.talon_claim(repo="org/repo", issue=42)
        assert result.status is ToolResultStatus.OK
        assert result.data["method"] == "a2a"
        mock_mesh.assert_awaited_once_with("org/repo", 42)
        mock_bg.assert_not_awaited()

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


def _ready_workspace_state(path):
    return {
        "repo": "org/repo",
        "path": str(path),
        "exists": True,
        "is_git": True,
        "head": "main",
        "clean": True,
        "last_fetch_at": None,
        "safe": True,
    }


class TestTalonBatch:
    @pytest.mark.asyncio
    async def test_batch_with_prd(self, tmp_path, monkeypatch):
        """F304: batch loads policy, requires a workspace, passes --repo-dir + abs prd."""
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        prd_file = tmp_path / "prd.json"
        prd_file.write_text("{}")
        feature = TalonCoordinatorFeature(_make_agent())
        workspace = tmp_path / "org__repo"
        with patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg, \
             patch.object(TalonCoordinatorFeature, "_workspace_state",
                          return_value=_ready_workspace_state(workspace)):
            mock_bg.return_value = {"dispatched": True, "method": "cli_background"}
            result = await feature.talon_batch(repo="org/repo", prd=str(prd_file))
            assert result.status is ToolResultStatus.OK
            assert result.data["dispatched"] is True
            mock_bg.assert_awaited_once()
            args = mock_bg.call_args[0][0]
            assert "batch" in args
            assert args[args.index("--prd") + 1] == str(prd_file)
            assert "--repo-dir" in args
            assert str(tmp_path) in args[args.index("--repo-dir") + 1]
            # env is credential-sanitized (still carries the GitHub token
            # because batch, like claim, runs in a sandbox and needs it).
            assert mock_bg.call_args.kwargs["env"]["GITHUB_TOKEN"] == "ghp_test"
            assert "ANTHROPIC_API_KEY" not in mock_bg.call_args.kwargs["env"]

    @pytest.mark.asyncio
    async def test_batch_rejects_relative_prd(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        feature = TalonCoordinatorFeature(_make_agent())
        workspace = tmp_path / "org__repo"
        with patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg, \
             patch.object(TalonCoordinatorFeature, "_workspace_state",
                          return_value=_ready_workspace_state(workspace)):
            result = await feature.talon_batch(repo="org/repo", prd="prd.json")
            assert result.status is ToolResultStatus.ERROR
            assert result.data["dispatched"] is False
            mock_bg.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_requires_provisioned_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        prd_file = tmp_path / "prd.json"
        prd_file.write_text("{}")
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_cli_background", new_callable=AsyncMock) as mock_bg:
            result = await feature.talon_batch(repo="org/repo", prd=str(prd_file))
            assert result.status is ToolResultStatus.ERROR
            assert result.data["state"] == "workspace_not_provisioned"
            assert "talon_setup_workspace" in result.data["next_step"]
            mock_bg.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_no_args_errors(self):
        feature = TalonCoordinatorFeature(_make_agent())
        result = await feature.talon_batch(repo="org/repo")
        assert result.status is ToolResultStatus.ERROR
        assert result.data["dispatched"] is False
        assert "error" in result.data


class TestTalonToolDocumentation:
    """Agent-facing tool self-documentation regressions (#1925 / #1923).

    ``talon_set_config`` and ``talon_batch`` must spell out their
    constrained vocabulary and contracts in the schema the agent sees,
    matching the ``talon_claim`` reference standard.
    """

    @staticmethod
    def _schema(method):
        # The @tool decorator parses ``description=`` + the Args: docstring
        # into ``_tool_schema`` — this is the agent-facing surface.
        return method._tool_schema

    @staticmethod
    def _full_doc(method):
        schema = method._tool_schema
        parts = [schema["description"]]
        parts.extend(p.description or "" for p in schema["parameters"])
        return "\n".join(parts)

    def test_set_config_documents_allowed_backends(self):
        doc = self._full_doc(TalonCoordinatorFeature.talon_set_config)
        for backend in ("claude", "codex", "opencode"):
            assert backend in doc

    def test_set_config_documents_model_vocabulary(self):
        doc = self._full_doc(TalonCoordinatorFeature.talon_set_config)
        for alias in ("opus", "sonnet", "haiku"):
            assert alias in doc

    def test_set_config_documents_auth_lanes_and_cross_field_rules(self):
        doc = self._full_doc(TalonCoordinatorFeature.talon_set_config)
        for lane in ("oauth", "api_key", "provider_config"):
            assert lane in doc
        # Cross-field rules from runtime.resolve_runtime.
        assert "codex" in doc and "provider_config" in doc
        assert "iteration" in doc.lower()
        assert "turn" in doc.lower()

    def test_set_config_has_args_block(self):
        # The decorator only populates per-parameter descriptions from an
        # Args: block, so a populated description proves the block exists.
        params = {
            p.name: p.description
            for p in self._schema(TalonCoordinatorFeature.talon_set_config)["parameters"]
        }
        assert params["default_backend"]
        assert params["default_model"]
        assert params["default_auth_lane"]

    def test_set_config_param_descriptions_not_truncated(self):
        # Regression: ``parse_docstring_params`` terminates a param description
        # at the next indented line starting with a word char, so a WRAPPED
        # Args entry silently drops its tail in the agent-facing schema (codex
        # P2). Assert the per-PARAM descriptions carry their full vocabulary,
        # not just the concatenated tool-level description.
        params = {
            p.name: (p.description or "")
            for p in self._schema(TalonCoordinatorFeature.talon_set_config)["parameters"]
        }
        # default_model must keep the codex/opencode tail (was cut at "When").
        assert "opencode" in params["default_model"]
        # codex/opencode model is OPTIONAL (omit for provider default), NOT
        # required — the schema must not overstate it (codex round-2 P2).
        assert "provider default" in params["default_model"].lower()
        assert "optional" in params["default_model"].lower()
        # default_auth_lane must keep the full cross-field rule tail.
        assert "allow_api_billing" in params["default_auth_lane"]

    def test_batch_param_descriptions_not_truncated(self):
        params = {
            p.name: (p.description or "")
            for p in self._schema(TalonCoordinatorFeature.talon_batch)["parameters"]
        }
        # There is no label mode anymore (F304); prd is required + absolute.
        assert "label" not in params
        assert "required" in params["prd"].lower()
        assert "absolute" in params["prd"].lower()

    def test_batch_documents_prd_required_and_workspace(self):
        doc = self._full_doc(TalonCoordinatorFeature.talon_batch)
        desc = self._schema(TalonCoordinatorFeature.talon_batch)["description"]
        # prd is required and there is no label mode.
        assert "prd" in desc.lower()
        assert "required" in desc.lower()
        # The sandbox-workspace guardrail (F304) is surfaced to the agent.
        assert "workspace" in doc.lower()
        assert "talon_setup_workspace" in doc

    def test_batch_documents_job_id_polling(self):
        doc = self._full_doc(TalonCoordinatorFeature.talon_batch)
        assert "job_id" in doc
        assert "talon_status" in doc


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
    """The TalonWaitable provider driven by the generic engine.

    There is no `talon_wait` tool any more — talon jobs are waited on via the
    single generic `wait("talon:<job_id>")` tool, which dispatches to this
    provider through the WaitRegistry. These tests exercise the same code path
    by calling ``run_wait_loop(TalonWaitable(feature), job_id, ...)`` directly
    (what the registry does), so the talon-specific reap/reconcile/terminal
    behavior stays covered.
    """

    @staticmethod
    async def _wait(feature, job_id, **kw):
        return await run_wait_loop(TalonWaitable(feature), job_id, **kw)

    @pytest.mark.asyncio
    async def test_unknown_job(self):
        feature = TalonCoordinatorFeature(_make_agent())
        result = await self._wait(feature, "nope", timeout_seconds=0)
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
        result = await self._wait(
            feature, "job-x", timeout_seconds=30, poll_interval_seconds=1,
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
        result = await self._wait(feature, "job-f", timeout_seconds=30)
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
        result = await self._wait(
            feature, "job-r", timeout_seconds=0, poll_interval_seconds=1,
        )
        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["status"] == "running"
        assert result.data["timed_out"] is True
        assert result.data["timeout_seconds"] == 0

    @pytest.mark.asyncio
    async def test_max_duration_rejected(self):
        feature = TalonCoordinatorFeature(_make_agent())
        feature._jobs["job-r"] = {"method": "a2a", "status": "running"}
        # The engine caps a held wait at MAX_HANDLE_WAIT_SECONDS.
        too_long = MAX_HANDLE_WAIT_SECONDS + 1
        result = await self._wait(feature, "job-r", timeout_seconds=too_long)
        assert result.status is ToolResultStatus.ERROR
        assert "exceeds the maximum" in result.error
        assert result.data["max_seconds"] == MAX_HANDLE_WAIT_SECONDS

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
        result = await self._wait(
            feature, "job-s", timeout_seconds=30, poll_interval_seconds=1,
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
            result = await self._wait(
                feature, "job-a", timeout_seconds=30, poll_interval_seconds=1,
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
            result = await self._wait(
                feature, "job-b", timeout_seconds=30, poll_interval_seconds=1,
            )
        assert result.status is ToolResultStatus.ERROR
        assert result.data["status"] == "failed"
        assert result.data["timed_out"] is False
        assert feature._jobs["job-b"]["error"] == "boom"

    @pytest.mark.asyncio
    async def test_active_handles_includes_terminal_jobs(self):
        """Regression (codex Wave 2 P1): a TERMINAL cli_background job must
        still be enumerated so a soft-dropped/restart-lost completion signal
        gets retried. Excluding terminal handles would silently lose the
        wake. The reconciler gates re-emits on the dedup ledger."""
        from kestrel_sovereign.features.talon.wait_provider import TalonWaitable

        feature = TalonCoordinatorFeature(_make_agent())
        feature._jobs = {
            "running-1": {"method": "cli_background", "status": "running"},
            "done-1": {"method": "cli_background", "status": "complete"},
            "failed-1": {"method": "cli_background", "status": "failed"},
            "a2a-1": {"method": "a2a", "status": "running"},
        }
        # _reload_persisted_jobs would clobber the in-memory dict from disk;
        # stub it so the test's fixture jobs are what active_handles sees.
        feature._reload_persisted_jobs = lambda: None
        handles = await TalonWaitable(feature).active_handles()
        # All cli_background jobs (terminal included); a2a excluded.
        assert set(handles) == {"running-1", "done-1", "failed-1"}

    @pytest.mark.asyncio
    async def test_unknown_job_poll_is_talon_schema_valid(self):
        """Regression (codex Wave 2 P2): a mode='signal' watch on a stale/
        unknown job id must still produce a talon.job_complete-schema-valid
        payload (job_id + status), else the dispatcher drops the wake and the
        caller is never woken."""
        from kestrel_sovereign.features.talon.wait_provider import TalonWaitable
        from kestrel_sovereign.signals.sources.talon import (
            _schema as talon_schema,
        )

        feature = TalonCoordinatorFeature(_make_agent())
        feature._reload_persisted_jobs = lambda: None
        feature._jobs = {}
        status = await TalonWaitable(feature).poll("ghost-1")
        assert status.outcome.value == "failed"
        # The talon.job_complete schema must accept the payload (it requires
        # job_id + status); this would raise ValueError if status were absent.
        talon_schema(dict(status.data))
        assert status.data["status"] == "finished_unknown"

    @pytest.mark.asyncio
    async def test_post_load_seeds_legacy_signal_ledger(self, tmp_path):
        """Regression (codex Wave 2 P2): on upgrade, jobs.json rows carrying
        legacy last_signaled_status must seed the generic ledger so the first
        wait_reconcile tick doesn't re-fire talon.job_complete for an
        already-delivered terminal job."""
        from types import SimpleNamespace

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.async_wait_signal_store import (
            WaitSignalStore,
        )
        from kestrel_sovereign.waits import WaitRegistry

        db = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
        agent = SimpleNamespace(
            did="did:test:agent",
            agent_id="did:test:agent",
            _raw_storage=SimpleNamespace(db=db),
            wait_registry=WaitRegistry(),
        )
        feature = TalonCoordinatorFeature(agent)
        feature._jobs = {
            "done-1": {"method": "cli_background", "status": "complete",
                       "last_signaled_status": "complete"},
            "failed-1": {"method": "cli_background", "status": "failed",
                         "last_signaled_status": "failed"},
            "unsig-1": {"method": "cli_background", "status": "complete"},
        }
        feature._reload_persisted_jobs = lambda: None

        await feature.post_all_features_loaded(agent)

        store = WaitSignalStore(db, "did:test:agent")
        # Seeded with the reconciler's dedup token shape "<outcome>:<status>"
        # (complete -> done, failed -> failed) so the first tick won't re-fire.
        assert (await store.get("talon", "done-1")).last_signaled_outcome == "done:complete"
        assert (await store.get("talon", "failed-1")).last_signaled_outcome == "failed:failed"
        # No legacy status => no seed row (the reconciler will signal it fresh).
        assert await store.get("talon", "unsig-1") is None


class TestTalonVerify:
    @pytest.mark.asyncio
    async def test_verify_no_commands_fails(self, tmp_path):
        feature = TalonCoordinatorFeature(_make_agent())
        result = await feature.talon_verify(commands="   \n\n", cwd=str(tmp_path))
        assert result.status is ToolResultStatus.ERROR
        assert result.data["overall_state"] == "not_run"

    @pytest.mark.asyncio
    async def test_verify_invalid_cwd_fails(self, tmp_path):
        feature = TalonCoordinatorFeature(_make_agent())
        result = await feature.talon_verify(
            commands="uv run pytest", cwd=str(tmp_path / "nope")
        )
        assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_verify_no_workspace_refuses_provisioned(self, tmp_path, monkeypatch):
        """F301: with no cwd and no provisioned workspace, refuse structurally.

        Never fall back to the running agent's source tree.
        """
        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path / "ws"))
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_make_verify_executor") as mock_exec:
            result = await feature.talon_verify(
                commands="uv run pytest tests/unit", repo="org/repo"
            )
        assert result.status is ToolResultStatus.ERROR
        assert result.data["state"] == "workspace_not_provisioned"
        assert "talon_setup_workspace" in result.data["next_step"]
        # No executor was ever built — nothing ran in the source tree.
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_executor_sanitizes_untrusted_env(self, tmp_path, monkeypatch):
        """F302: the verify subprocess gets no provider creds and no GH token."""
        import asyncio as _asyncio

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oauth-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
        monkeypatch.setenv("KESTREL_API_KEY", "kestrel-secret")
        monkeypatch.setenv("KESTREL_DATA_KEY", "data-secret")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GH_TOKEN", "ghp_secret2")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
        monkeypatch.setenv("PATH_SHOULD_SURVIVE", "keep-me")

        feature = TalonCoordinatorFeature(_make_agent())
        executor = feature._make_verify_executor(tmp_path)

        captured = {}

        async def fake_create(*argv, **kwargs):
            captured["env"] = kwargs["env"]
            raise FileNotFoundError("stop after env is built")

        with patch.object(_asyncio, "create_subprocess_exec", side_effect=fake_create):
            await executor("pytest -q", timeout=5)

        env = captured["env"]
        for leaked in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "OPENAI_API_KEY",
            "KESTREL_API_KEY",
            "KESTREL_DATA_KEY",
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "GOOGLE_API_KEY",
        ):
            assert leaked not in env, f"{leaked} must be stripped from verify env"
        # Non-secret vars are preserved so the command can still run.
        assert env.get("PATH_SHOULD_SURVIVE") == "keep-me"

    @pytest.mark.asyncio
    async def test_verify_allowlisted_pass(self, tmp_path):
        feature = TalonCoordinatorFeature(_make_agent())

        async def fake_exec(command, *, timeout=600):
            return CommandExecution(ran=True, returncode=0, stdout="2 passed")

        with patch.object(feature, "_make_verify_executor", return_value=fake_exec):
            result = await feature.talon_verify(
                commands="uv run pytest tests/unit", cwd=str(tmp_path)
            )
        assert result.status is ToolResultStatus.OK
        assert result.data["overall_state"] == "passed"
        assert result.data["all_passed"] is True
        assert "## Test Evidence" in result.confirmation

    @pytest.mark.asyncio
    async def test_verify_failed_is_partial(self, tmp_path):
        feature = TalonCoordinatorFeature(_make_agent())

        async def fake_exec(command, *, timeout=600):
            return CommandExecution(ran=True, returncode=1, stderr="1 failed")

        with patch.object(feature, "_make_verify_executor", return_value=fake_exec):
            result = await feature.talon_verify(
                commands="uv run pytest tests/unit", cwd=str(tmp_path)
            )
        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["overall_state"] == "failed"

    @staticmethod
    def _init_git_repo_with_pr_branch(root):
        """Create a git checkout whose ``pr-branch`` carries a file absent on main.

        Returns the path. ``main`` has no ``marker.txt``; ``pr-branch``
        adds it — the exact #1631 shape where a test file exists only on
        the PR branch and not on the checked-out (main) tree.
        """
        import subprocess as _sp

        def git(*args):
            _sp.run(
                ["git", *args], cwd=str(root), check=True,
                capture_output=True, text=True,
            )

        root.mkdir(parents=True, exist_ok=True)
        git("init", "-b", "main")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "Test")
        (root / "base.txt").write_text("base\n")
        git("add", "base.txt")
        git("commit", "-m", "base on main")
        git("checkout", "-b", "pr-branch")
        (root / "marker.txt").write_text("only on pr branch\n")
        git("add", "marker.txt")
        git("commit", "-m", "add marker on pr-branch")
        git("checkout", "main")
        return root

    @staticmethod
    def _init_git_repo_with_stale_local_branch(root):
        """Create a checkout where local feature is stale vs origin/feature."""
        import subprocess as _sp

        remote = root / "remote.git"
        seed = root / "seed"
        workspace = root / "workspace"

        def git(cwd, *args):
            _sp.run(
                ["git", *args], cwd=str(cwd), check=True,
                capture_output=True, text=True,
            )

        root.mkdir(parents=True, exist_ok=True)
        remote.mkdir()
        git(remote, "init", "--bare")

        seed.mkdir()
        git(seed, "init", "-b", "main")
        git(seed, "config", "user.email", "t@example.com")
        git(seed, "config", "user.name", "Test")
        (seed / "base.txt").write_text("base\n")
        git(seed, "add", "base.txt")
        git(seed, "commit", "-m", "base on main")
        git(seed, "remote", "add", "origin", str(remote))
        git(seed, "push", "origin", "main")
        git(seed, "checkout", "-b", "feature")
        (seed / "marker.txt").write_text("remote\n")
        git(seed, "add", "marker.txt")
        git(seed, "commit", "-m", "remote feature marker")
        git(seed, "push", "origin", "feature")
        remote_sha = _sp.run(
            ["git", "rev-parse", "HEAD"], cwd=str(seed), check=True,
            capture_output=True, text=True,
        ).stdout.strip()

        git(root, "clone", str(remote), str(workspace))
        git(workspace, "config", "user.email", "t@example.com")
        git(workspace, "config", "user.name", "Test")
        git(workspace, "checkout", "main")
        git(workspace, "checkout", "-b", "feature")
        (workspace / "marker.txt").write_text("stale-local\n")
        git(workspace, "add", "marker.txt")
        git(workspace, "commit", "-m", "stale local feature marker")
        git(workspace, "checkout", "main")
        return workspace, remote_sha

    @pytest.mark.asyncio
    async def test_verify_runs_against_pr_branch_not_main(self, tmp_path):
        """#1631: with ref set, the PR branch is checked out before running.

        The executor inspects the real working tree; ``marker.txt`` exists
        only on ``pr-branch``. Verification must run against that branch
        (exit 0), not against the checked-out ``main`` (where the file is
        absent and the issue's "file or directory not found" occurred).
        """
        from pathlib import Path

        workspace = self._init_git_repo_with_pr_branch(tmp_path / "ws")
        feature = TalonCoordinatorFeature(_make_agent())

        def make_exec(run_cwd):
            async def _exec(command, *, timeout=600):
                if (Path(run_cwd) / "marker.txt").exists():
                    return CommandExecution(ran=True, returncode=0, stdout="1 passed")
                return CommandExecution(
                    ran=True,
                    returncode=4,
                    stderr="file or directory not found: marker.txt",
                )

            return _exec

        with patch.object(
            feature, "_make_verify_executor", side_effect=make_exec
        ):
            result = await feature.talon_verify(
                commands="pytest marker.txt",
                cwd=str(workspace),
                ref="pr-branch",
            )

        assert result.status is ToolResultStatus.OK
        assert result.data["overall_state"] == "passed"
        assert result.data["requested_ref"] == "pr-branch"
        assert result.data["head_sha"]
        # The actual tree was switched to the requested branch commit.
        assert (workspace / "marker.txt").exists()

    @pytest.mark.asyncio
    async def test_verify_branch_ref_resets_stale_local_branch_to_remote(self, tmp_path):
        """Branch refs verify the fetched remote branch, not a stale local one."""
        import subprocess as _sp
        from pathlib import Path

        origin = tmp_path / "origin.git"
        seed = tmp_path / "seed"
        workspace = tmp_path / "workspace"

        def git(cwd, *args):
            _sp.run(
                ["git", *args],
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
            )

        _sp.run(["git", "init", "--bare", str(origin)], check=True)
        seed.mkdir()
        git(seed, "init", "-b", "main")
        git(seed, "config", "user.email", "t@example.com")
        git(seed, "config", "user.name", "Test")
        (seed / "base.txt").write_text("base\n")
        git(seed, "add", "base.txt")
        git(seed, "commit", "-m", "base")
        git(seed, "remote", "add", "origin", str(origin))
        git(seed, "push", "-u", "origin", "main")

        git(seed, "checkout", "-b", "pr-branch")
        (seed / "marker.txt").write_text("stale\n")
        git(seed, "add", "marker.txt")
        git(seed, "commit", "-m", "stale marker")
        git(seed, "push", "-u", "origin", "pr-branch")

        _sp.run(["git", "clone", str(origin), str(workspace)], check=True)
        git(workspace, "checkout", "pr-branch")
        git(workspace, "checkout", "main")

        (seed / "marker.txt").write_text("fresh\n")
        git(seed, "add", "marker.txt")
        git(seed, "commit", "-m", "fresh marker")
        git(seed, "push", "origin", "pr-branch")

        feature = TalonCoordinatorFeature(_make_agent())

        def make_exec(run_cwd):
            async def _exec(command, *, timeout=600):
                marker = Path(run_cwd) / "marker.txt"
                if marker.read_text() == "fresh\n":
                    return CommandExecution(ran=True, returncode=0, stdout="fresh")
                return CommandExecution(
                    ran=True,
                    returncode=4,
                    stderr=f"verified stale branch content: {marker.read_text()!r}",
                )

            return _exec

        with patch.object(
            feature, "_make_verify_executor", side_effect=make_exec
        ):
            result = await feature.talon_verify(
                commands="pytest marker.txt",
                cwd=str(workspace),
                ref="pr-branch",
            )

        assert result.status is ToolResultStatus.OK
        assert result.data["overall_state"] == "passed"
        assert result.data["checked_out_ref"] == "pr-branch"
        assert (workspace / "marker.txt").read_text() == "fresh\n"

    @pytest.mark.asyncio
    async def test_verify_unknown_ref_is_tooling_error_not_failure(self, tmp_path):
        """A ref that can't be checked out yields tooling_error, not a code failure.

        Crucially the commands must NOT run against the un-switched tree.
        """
        workspace = self._init_git_repo_with_pr_branch(tmp_path / "ws")
        feature = TalonCoordinatorFeature(_make_agent())

        ran = {"called": False}

        def make_exec(run_cwd):
            async def _exec(command, *, timeout=600):
                ran["called"] = True
                return CommandExecution(ran=True, returncode=0)

            return _exec

        with patch.object(
            feature, "_make_verify_executor", side_effect=make_exec
        ):
            result = await feature.talon_verify(
                commands="uv run pytest tests/unit",
                cwd=str(workspace),
                ref="does-not-exist-branch",
            )

        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["overall_state"] == "tooling_error"
        assert result.data["requested_ref"] == "does-not-exist-branch"
        assert result.data["checked_out_ref"] is None
        assert ran["called"] is False

    @pytest.mark.asyncio
    async def test_verify_ref_on_non_git_cwd_is_tooling_error(self, tmp_path):
        """Requesting a ref in a non-git dir is a tooling error, not a pass."""
        feature = TalonCoordinatorFeature(_make_agent())

        async def fake_exec(command, *, timeout=600):
            return CommandExecution(ran=True, returncode=0)

        with patch.object(feature, "_make_verify_executor", return_value=fake_exec):
            result = await feature.talon_verify(
                commands="uv run pytest", cwd=str(tmp_path), ref="pr/1630"
            )
        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["overall_state"] == "tooling_error"

    def test_parse_verify_ref_forms(self):
        parse = TalonCoordinatorFeature._parse_verify_ref
        assert parse("1630") == ("pr", "1630")
        assert parse("#1630") == ("pr", "1630")
        assert parse("pr/1630") == ("pr", "1630")
        assert parse("PR-1630") == ("pr", "1630")
        assert parse("issue-1626-restartcoordinator-busy-count") == (
            "ref",
            "issue-1626-restartcoordinator-busy-count",
        )


class TestTalonScheduleWorkRescue:
    """Safe recurring stalled_work_rescue scheduling (#2200)."""

    @pytest.mark.asyncio
    async def test_schedules_safe_recurring_loop(self):
        from kestrel_sdk.tools.result import ToolResult

        agent = _make_agent()
        scheduler = MagicMock()
        scheduler.schedule_add = AsyncMock(
            return_value=ToolResult.ok("added", data={"success": True})
        )
        agent.get_feature = MagicMock(return_value=scheduler)
        feature = TalonCoordinatorFeature(agent)

        result = await feature.talon_schedule_work_rescue()
        assert result.status is ToolResultStatus.OK
        scheduler.schedule_add.assert_awaited_once()
        kwargs = scheduler.schedule_add.await_args.kwargs
        assert kwargs["task_name"] == "workflow_run"
        assert kwargs["cron_expression"] == "0 */6 * * *"
        args = json.loads(kwargs["args_json"])
        assert args["name"] == "stalled_work_rescue"
        # Observation-only: never a pre-seeded target or standing approval.
        assert args["params"] == {"stale_days": 3, "recurring": True}
        assert result.data["schedule_request"]["task_name"] == "workflow_run"

    @pytest.mark.asyncio
    async def test_no_scheduler_returns_request_not_silent_noop(self):
        agent = _make_agent()
        agent.get_feature = MagicMock(return_value=None)
        feature = TalonCoordinatorFeature(agent)

        result = await feature.talon_schedule_work_rescue()
        assert result.status is ToolResultStatus.ERROR
        # The ready-to-use invocation is surfaced for a manual install.
        assert result.data["schedule_request"]["task_name"] == "workflow_run"

    @pytest.mark.asyncio
    async def test_scheduler_rejection_is_reported_failed(self):
        from kestrel_sdk.tools.result import ToolResult

        agent = _make_agent()
        scheduler = MagicMock()
        scheduler.schedule_add = AsyncMock(
            return_value=ToolResult.failed("Unknown scheduled task 'workflow_run'")
        )
        agent.get_feature = MagicMock(return_value=scheduler)
        feature = TalonCoordinatorFeature(agent)

        result = await feature.talon_schedule_work_rescue()
        assert result.status is ToolResultStatus.ERROR
        assert "Unknown scheduled task" in result.error


class TestSurveyStalledTalonJobs:
    """Live stalled-work discovery for fleet_stalled_sweep (#2200)."""

    def _feature(self):
        feature = TalonCoordinatorFeature(_make_agent())
        # Keep the survey purely in-memory (no persisted-registry reload).
        feature._reload_persisted_jobs = MagicMock()
        return feature

    @pytest.mark.asyncio
    async def test_surveys_stalled_running_job(self):
        from datetime import datetime, timedelta, timezone

        feature = self._feature()
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        feature._jobs = {
            "job-1": {
                "status": "running", "started_at": old,
                "label": "issue-1", "repo": "org/repo", "issue": 1,
            },
        }
        stalled = await feature._survey_stalled_talon_jobs(3)
        assert len(stalled) == 1
        assert stalled[0]["id"] == "job-1"
        assert stalled[0]["kind"] == "talon_job"
        assert stalled[0]["repo"] == "org/repo"

    @pytest.mark.asyncio
    async def test_recent_and_completed_jobs_are_not_stalled(self):
        from datetime import datetime, timedelta, timezone

        feature = self._feature()
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        feature._jobs = {
            "recent": {"status": "running", "started_at": recent},
            "done": {"status": "complete", "started_at": old},
        }
        assert await feature._survey_stalled_talon_jobs(3) == []

    @pytest.mark.asyncio
    async def test_missing_timestamp_is_treated_as_stalled(self):
        feature = self._feature()
        feature._jobs = {"job-x": {"status": "dispatched"}}
        stalled = await feature._survey_stalled_talon_jobs(3)
        assert [j["id"] for j in stalled] == ["job-x"]

    @pytest.mark.asyncio
    async def test_scan_stale_work_tool_wraps_live_survey(self):
        from datetime import datetime, timedelta, timezone

        feature = self._feature()
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        feature._jobs = {
            "job-1": {
                "status": "running",
                "started_at": old,
                "label": "issue-1",
                "repo": "org/repo",
                "issue": 1,
            },
            "job-2": {
                "status": "running",
                "started_at": old,
                "repo": "other/repo",
                "issue": 2,
            },
        }

        result = await feature.scan_stale_work(stale_days=3, repo="org/repo")

        assert result.status is ToolResultStatus.OK
        findings = result.data["findings"]
        assert len(findings) == 1
        assert findings[0]["id"] == "job-1"
        assert findings[0]["repo"] == "org/repo"
        assert findings[0]["kind"] == "talon_job"
        assert findings[0]["issue"] == 1
        assert findings[0]["suggested_gate"] == "govern_stalled_work_rescue"

    @pytest.mark.asyncio
    async def test_survey_is_registered_as_fleet_discover(self):
        # The coordinator binds its survey onto fleet_stalled_sweep so a
        # recurring tick observes real candidates without pre-seeding.
        from datetime import datetime, timedelta, timezone

        from kestrel_sovereign.signals import SourceRegistry

        feature = self._feature()
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        feature._jobs = {"job-1": {"status": "running", "started_at": old}}
        registry = SourceRegistry()
        feature.agent.signal_registry = registry
        feature.agent.get_feature = MagicMock(return_value=None)
        await feature.initialize()

        reg = registry.get("fleet_stalled_sweep")
        assert reg is not None
        result = await reg.handler({"stale_days": 3, "recurring": True})
        assert result["discovered"] is True
        assert [j["id"] for j in result["stalled_items"]] == ["job-1"]
