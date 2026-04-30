"""Tests for TalonCoordinatorFeature."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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
    async def test_claim_mesh_success(self):
        """When mesh dispatch works, returns dispatched=True."""
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_mesh", new_callable=AsyncMock) as mock_mesh:
            mock_mesh.return_value = {
                "dispatched": True, "method": "mesh",
                "message_id": "abc", "repo": "org/repo", "issue": 42,
            }
            result = await feature.talon_claim(repo="org/repo", issue=42)
            assert result["dispatched"] is True
            assert result["method"] == "mesh"
            mock_mesh.assert_awaited_once_with("org/repo", 42)

    @pytest.mark.asyncio
    async def test_claim_falls_back_to_cli(self):
        """When mesh fails, falls back to CLI."""
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_mesh", new_callable=AsyncMock) as mock_mesh, \
             patch.object(feature, "_dispatch_via_cli", new_callable=AsyncMock) as mock_cli:
            mock_mesh.return_value = {"dispatched": False, "reason": "no_mesh_host"}
            mock_cli.return_value = {"dispatched": True, "method": "cli"}
            result = await feature.talon_claim(repo="org/repo", issue=42)
            assert result["dispatched"] is True
            assert result["method"] == "cli"


class TestTalonBatch:
    @pytest.mark.asyncio
    async def test_batch_with_prd(self):
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = {"dispatched": True, "method": "cli"}
            result = await feature.talon_batch(repo="org/repo", prd="prd.json")
            assert result["dispatched"] is True
            mock_cli.assert_awaited_once()
            args = mock_cli.call_args[0][0]
            assert "batch" in args
            assert "--prd" in args

    @pytest.mark.asyncio
    async def test_batch_with_label(self):
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_dispatch_via_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = {"dispatched": True, "method": "cli"}
            result = await feature.talon_batch(repo="org/repo", label="P0")
            assert result["dispatched"] is True
            args = mock_cli.call_args[0][0]
            assert "--label" in args
            assert "P0" in args

    @pytest.mark.asyncio
    async def test_batch_no_args_errors(self):
        feature = TalonCoordinatorFeature(_make_agent())
        result = await feature.talon_batch(repo="org/repo")
        assert result["dispatched"] is False
        assert "error" in result


class TestTalonStatus:
    @pytest.mark.asyncio
    async def test_status_empty(self):
        feature = TalonCoordinatorFeature(_make_agent())
        result = await feature.talon_status()
        assert result["running"] == 0
        assert result["completed"] == 0

    @pytest.mark.asyncio
    async def test_status_with_jobs(self):
        feature = TalonCoordinatorFeature(_make_agent())
        feature._jobs["job-1"] = {"repo": "a/b", "issue": 1, "status": "dispatched"}
        feature._jobs["job-2"] = {"repo": "a/b", "issue": 2, "status": "complete"}
        result = await feature.talon_status()
        assert result["running"] == 1
        assert result["completed"] == 1


class TestTalonPauseResume:
    @pytest.mark.asyncio
    async def test_pause(self):
        agent = _make_agent()
        feature = TalonCoordinatorFeature(agent)
        result = await feature.talon_pause()
        assert result["paused"] is True
        agent._scheduler.remove_schedule.assert_called_once_with("signal_dispatch")

    @pytest.mark.asyncio
    async def test_resume(self):
        agent = _make_agent()
        feature = TalonCoordinatorFeature(agent)
        result = await feature.talon_resume()
        assert result["resumed"] is True
        agent._scheduler.add_schedule.assert_called_once()


class TestMeshDispatch:
    @pytest.mark.asyncio
    async def test_no_host_url(self):
        feature = TalonCoordinatorFeature(_make_agent())
        with patch.object(feature, "_discover_host_url", return_value=None):
            result = await feature._dispatch_via_mesh("org/repo", 42)
            assert result["dispatched"] is False
            assert result["reason"] == "no_mesh_host"


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
