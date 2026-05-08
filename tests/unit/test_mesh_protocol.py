"""Unit tests for the Agent Mesh Protocol.

Tests:
- MeshMessage creation and serialization
- Factory functions (make_assign_message, make_complete_message, etc.)
- MeshMessageType and MeshPriority enums
- PeersFeature mesh tools (send_mesh_message, mesh_inbox, receive_mesh_message)
- Talon handoff issue selection
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.peers.mesh import (
    MeshMessage,
    MeshMessageType,
    MeshPriority,
    make_assign_message,
    make_complete_message,
    make_reject_message,
    make_review_message,
)
from kestrel_sovereign.features.peers.feature import PeersFeature


# =========================================================================
# MeshMessage dataclass
# =========================================================================


class TestMeshMessage:
    def test_create_basic(self):
        msg = MeshMessage(
            type=MeshMessageType.ASSIGN,
            sender="kestrel",
            recipient="talon",
            payload={"issue": 42},
        )
        assert msg.type == MeshMessageType.ASSIGN
        assert msg.sender == "kestrel"
        assert msg.recipient == "talon"
        assert msg.payload == {"issue": 42}
        assert msg.priority == MeshPriority.NORMAL
        assert msg.id  # auto-generated UUID
        assert msg.timestamp  # auto-generated ISO timestamp

    def test_to_dict_roundtrip(self):
        msg = MeshMessage(
            type=MeshMessageType.REVIEW_NEEDED,
            sender="talon",
            recipient="kestrel",
            payload={"pr": 123},
            priority=MeshPriority.HIGH,
            repo="KestrelSovereignAI/kestrel-sovereign",
        )
        d = msg.to_dict()
        assert d["type"] == "review_needed"
        assert d["priority"] == "high"
        assert d["repo"] == "KestrelSovereignAI/kestrel-sovereign"

        restored = MeshMessage.from_dict(d)
        assert restored.type == MeshMessageType.REVIEW_NEEDED
        assert restored.priority == MeshPriority.HIGH
        assert restored.sender == "talon"
        assert restored.payload == {"pr": 123}

    def test_to_json_roundtrip(self):
        msg = MeshMessage(
            type=MeshMessageType.COMPLETE,
            sender="talon",
            recipient="kestrel",
            payload={"result": "done"},
        )
        j = msg.to_json()
        restored = MeshMessage.from_json(j)
        assert restored.type == MeshMessageType.COMPLETE
        assert restored.payload == {"result": "done"}
        assert restored.id == msg.id

    def test_from_dict_with_unknown_type_raises(self):
        with pytest.raises((KeyError, ValueError)):
            MeshMessage.from_dict({
                "type": "nonexistent",
                "sender": "a",
                "recipient": "b",
                "payload": {},
                "priority": "normal",
                "id": "x",
                "timestamp": "2026-01-01T00:00:00",
            })

    def test_correlation_id_preserved(self):
        msg = MeshMessage(
            type=MeshMessageType.STATUS_UPDATE,
            sender="a",
            recipient="b",
            payload={},
            correlation_id="corr-123",
        )
        d = msg.to_dict()
        assert d["correlation_id"] == "corr-123"
        restored = MeshMessage.from_dict(d)
        assert restored.correlation_id == "corr-123"


# =========================================================================
# Factory functions
# =========================================================================


class TestFactoryFunctions:
    def test_make_assign_message(self):
        msg = make_assign_message(
            sender="kestrel",
            recipient="talon",
            repo="owner/repo",
            issue_number=42,
            issue_title="Fix the bug",
            priority="high",
            context="Blocker for demo",
        )
        assert msg.type == MeshMessageType.ASSIGN
        assert msg.sender == "kestrel"
        assert msg.recipient == "talon"
        assert msg.priority == MeshPriority.HIGH
        assert msg.repo == "owner/repo"
        assert msg.payload["issue_number"] == 42
        assert msg.payload["issue_title"] == "Fix the bug"
        assert msg.payload["context"] == "Blocker for demo"

    def test_make_complete_message(self):
        msg = make_complete_message(
            sender="talon",
            recipient="kestrel",
            correlation_id="orig-123",
            repo="owner/repo",
            issue_number=42,
            pr_number=99,
            summary="Fixed the bug",
        )
        assert msg.type == MeshMessageType.COMPLETE
        assert msg.correlation_id == "orig-123"
        assert msg.payload["pr_number"] == 99
        assert msg.payload["issue_number"] == 42

    def test_make_review_message(self):
        msg = make_review_message(
            sender="talon",
            recipient="kestrel",
            repo="owner/repo",
            pr_number=50,
            pr_title="Ready for review",
        )
        assert msg.type == MeshMessageType.REVIEW_NEEDED
        assert msg.repo == "owner/repo"
        assert msg.payload["pr_number"] == 50
        assert msg.payload["pr_title"] == "Ready for review"

    def test_make_reject_message(self):
        msg = make_reject_message(
            sender="talon",
            recipient="kestrel",
            correlation_id="orig-456",
            reason="Too complex",
        )
        assert msg.type == MeshMessageType.REJECT
        assert msg.correlation_id == "orig-456"
        assert msg.payload["reason"] == "Too complex"

    def test_assign_default_priority_is_normal(self):
        msg = make_assign_message(
            sender="a", recipient="b", repo="r", issue_number=1, issue_title="t",
        )
        assert msg.priority == MeshPriority.NORMAL


# =========================================================================
# PeersFeature mesh tools
# =========================================================================


class TestPeersFeatureMesh:
    def _make_feature(self, name="kestrel"):
        agent = SimpleNamespace(_agent_name=name)
        feature = PeersFeature(agent)
        feature._host_url = "http://multi_agent:8888"
        feature._api_key = ""
        feature._own_name = name
        feature._mesh_inbox = []
        feature._mesh_log = []
        return feature

    def test_receive_mesh_message_stores_in_inbox(self):
        feature = self._make_feature()
        msg_dict = make_assign_message(
            sender="talon", recipient="kestrel",
            repo="r", issue_number=1, issue_title="t",
        ).to_dict()

        result = feature.receive_mesh_message(msg_dict)
        assert result["accepted"] is True
        assert len(feature._mesh_inbox) == 1
        assert feature._mesh_inbox[0]["type"] == "assign"

    def test_receive_mesh_message_rejects_invalid(self):
        feature = self._make_feature()
        result = feature.receive_mesh_message({})
        assert result["accepted"] is False
        assert len(feature._mesh_inbox) == 0

    def test_receive_mesh_message_accepts_any_recipient(self):
        """Messages are routed by the multi_agent — the agent accepts all delivered messages."""
        feature = self._make_feature("kestrel")
        msg_dict = make_assign_message(
            sender="talon", recipient="kestrel",
            repo="r", issue_number=1, issue_title="t",
        ).to_dict()

        result = feature.receive_mesh_message(msg_dict)
        assert result["accepted"] is True

    @pytest.mark.asyncio
    async def test_mesh_inbox_returns_recent(self):
        feature = self._make_feature()
        # Add a few messages
        for i in range(3):
            msg_dict = make_assign_message(
                sender="talon", recipient="kestrel",
                repo="r", issue_number=i, issue_title=f"Issue {i}",
            ).to_dict()
            feature.receive_mesh_message(msg_dict)

        result = await feature.mesh_inbox(limit=2)
        # mesh_inbox now returns a ToolResult envelope (#1061 wave 16);
        # the legacy {"messages", "count", "total"} dict lives under .data.
        assert result.status is ToolResultStatus.OK
        assert result.data["total"] == 3
        assert result.data["count"] == 2
        assert len(result.data["messages"]) == 2

    @pytest.mark.asyncio
    async def test_send_mesh_message_no_host(self):
        feature = self._make_feature()
        feature._host_url = None

        result = await feature.send_mesh_message(
            recipient="talon",
            message_type="assign",
            payload_json='{"issue_number": 1}',
        )
        assert result.status is ToolResultStatus.ERROR
        assert "multi_agent" in result.error.lower()


# =========================================================================
# Talon handoff — issue selection
# =========================================================================


class TestTalonHandoff:
    def test_select_best_candidate_prefers_unassigned(self):
        from kestrel_sovereign.features.strategic_memory.talon_handoff import _select_best_candidate

        issues = [
            {"number": 1, "title": "A", "assignees": [{"login": "user"}], "comments": 0, "labels": []},
            {"number": 2, "title": "B", "assignees": [], "comments": 0, "labels": []},
        ]
        pick = _select_best_candidate(issues)
        assert pick["number"] == 2

    def test_select_best_candidate_skips_blocked(self):
        from kestrel_sovereign.features.strategic_memory.talon_handoff import _select_best_candidate

        issues = [
            {"number": 1, "title": "A", "assignees": [], "comments": 0, "labels": [{"name": "blocked"}]},
            {"number": 2, "title": "B", "assignees": [], "comments": 0, "labels": [{"name": "wontfix"}]},
        ]
        assert _select_best_candidate(issues) is None

    def test_select_best_candidate_empty_list(self):
        from kestrel_sovereign.features.strategic_memory.talon_handoff import _select_best_candidate

        assert _select_best_candidate([]) is None

    @pytest.mark.asyncio
    async def test_pick_top_issue_no_token(self):
        from kestrel_sovereign.features.strategic_memory.talon_handoff import pick_top_issue

        with patch(
            "kestrel_sovereign.features.strategic_memory.talon_handoff.get_github_token",
            return_value=None,
        ):
            result = await pick_top_issue({"morning_signal_config": {"scan_repos": ["a/b"]}})
            assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_to_talon_no_issue(self):
        from kestrel_sovereign.features.strategic_memory.talon_handoff import dispatch_to_talon

        with patch(
            "kestrel_sovereign.features.strategic_memory.talon_handoff.pick_top_issue",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await dispatch_to_talon({})
            assert "No actionable issue" in result

    @pytest.mark.asyncio
    async def test_dispatch_to_talon_no_host(self):
        from kestrel_sovereign.features.strategic_memory.talon_handoff import dispatch_to_talon

        issue = {
            "repo": "owner/repo",
            "issue_number": 42,
            "issue_title": "Fix it",
            "priority": "high",
        }
        with patch(
            "kestrel_sovereign.features.strategic_memory.talon_handoff.pick_top_issue",
            new_callable=AsyncMock,
            return_value=issue,
        ), patch(
            "kestrel_sovereign.features.strategic_memory.talon_handoff._discover_host_url",
            return_value=None,
        ):
            result = await dispatch_to_talon({})
            assert "no multi_agent host url" in result.lower()
            assert "owner/repo#42" in result
