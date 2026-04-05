"""Tests for microcompact — zero-cost stale tool result clearing."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from kestrel_sovereign.agent.context_manager import ContextManager


def _make_tool_msg(tool_call_id: str, content: str, metadata: dict = None) -> dict:
    msg = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    if metadata:
        msg["metadata"] = metadata
    return msg


def _make_user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def _make_assistant_msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


@pytest.fixture
def context_manager():
    """Create a ContextManager with minimal mocking."""
    cm = object.__new__(ContextManager)
    cm.MICROCOMPACT_KEEP_RECENT = 3
    return cm


class TestMicrocompactBasic:
    def test_no_tool_messages_returns_zero(self, context_manager):
        history = [_make_user_msg("hi"), _make_assistant_msg("hello")]
        assert context_manager._microcompact_tool_results(history) == 0

    def test_fewer_than_keep_recent_not_cleared(self, context_manager):
        history = [
            _make_tool_msg("t1", "result1"),
            _make_tool_msg("t2", "result2"),
        ]
        assert context_manager._microcompact_tool_results(history) == 0
        assert history[0]["content"] == "result1"
        assert history[1]["content"] == "result2"

    def test_exactly_keep_recent_not_cleared(self, context_manager):
        history = [
            _make_tool_msg("t1", "result1"),
            _make_tool_msg("t2", "result2"),
            _make_tool_msg("t3", "result3"),
        ]
        assert context_manager._microcompact_tool_results(history) == 0

    def test_old_results_cleared(self, context_manager):
        history = [
            _make_tool_msg("t1", "old result 1"),
            _make_tool_msg("t2", "old result 2"),
            _make_tool_msg("t3", "recent 1"),
            _make_tool_msg("t4", "recent 2"),
            _make_tool_msg("t5", "recent 3"),
        ]
        cleared = context_manager._microcompact_tool_results(history)
        assert cleared == 2

        # First two should be markers
        marker1 = json.loads(history[0]["content"])
        assert marker1["cleared"] is True
        assert "old result 1" in marker1["summary"]

        marker2 = json.loads(history[1]["content"])
        assert marker2["cleared"] is True

        # Last three should be untouched
        assert history[2]["content"] == "recent 1"
        assert history[3]["content"] == "recent 2"
        assert history[4]["content"] == "recent 3"

    def test_tool_call_id_preserved(self, context_manager):
        history = [
            _make_tool_msg("call-abc", "old result"),
            _make_tool_msg("call-def", "recent 1"),
            _make_tool_msg("call-ghi", "recent 2"),
            _make_tool_msg("call-jkl", "recent 3"),
        ]
        context_manager._microcompact_tool_results(history)
        assert history[0]["tool_call_id"] == "call-abc"
        assert history[0]["role"] == "tool"


class TestMicrocompactProtection:
    def test_protected_messages_not_cleared(self, context_manager):
        history = [
            _make_tool_msg("t1", "protected result", {"context_priority": "protected"}),
            _make_tool_msg("t2", "old result"),
            _make_tool_msg("t3", "recent 1"),
            _make_tool_msg("t4", "recent 2"),
            _make_tool_msg("t5", "recent 3"),
        ]
        cleared = context_manager._microcompact_tool_results(history)
        # t1 is protected, t2 is the only clearable one
        assert cleared == 1
        assert history[0]["content"] == "protected result"
        assert json.loads(history[1]["content"])["cleared"] is True

    def test_excluded_messages_not_cleared(self, context_manager):
        history = [
            _make_tool_msg("t1", "excluded", {"excluded_from_context": True}),
            _make_tool_msg("t2", "old"),
            _make_tool_msg("t3", "r1"),
            _make_tool_msg("t4", "r2"),
            _make_tool_msg("t5", "r3"),
        ]
        cleared = context_manager._microcompact_tool_results(history)
        assert cleared == 1
        assert history[0]["content"] == "excluded"

    def test_decay_protected_not_cleared(self, context_manager):
        history = [
            _make_tool_msg("t1", "pinned", {"decay_protected": True}),
            _make_tool_msg("t2", "old"),
            _make_tool_msg("t3", "r1"),
            _make_tool_msg("t4", "r2"),
            _make_tool_msg("t5", "r3"),
        ]
        cleared = context_manager._microcompact_tool_results(history)
        assert cleared == 1
        assert history[0]["content"] == "pinned"


class TestMicrocompactMarkerFormat:
    def test_marker_is_valid_json(self, context_manager):
        history = [
            _make_tool_msg("t1", "old", {"tool_name": "Read"}),
            _make_tool_msg("t2", "r1"),
            _make_tool_msg("t3", "r2"),
            _make_tool_msg("t4", "r3"),
        ]
        context_manager._microcompact_tool_results(history)
        marker = json.loads(history[0]["content"])
        assert marker["cleared"] is True
        assert marker["tool_name"] == "Read"
        assert "cleared_at" in marker

    def test_already_cleared_not_double_cleared(self, context_manager):
        marker = json.dumps({"cleared": True, "tool_name": "Read", "summary": "", "cleared_at": "2024-01-01"})
        history = [
            _make_tool_msg("t1", marker),
            _make_tool_msg("t2", "r1"),
            _make_tool_msg("t3", "r2"),
            _make_tool_msg("t4", "r3"),
            _make_tool_msg("t5", "r4"),
        ]
        cleared = context_manager._microcompact_tool_results(history)
        # t1 is already cleared, should be skipped (not counted)
        assert cleared == 1


class TestMicrocompactMixedHistory:
    def test_interleaved_with_user_assistant(self, context_manager):
        history = [
            _make_user_msg("do thing 1"),
            _make_assistant_msg("calling tool"),
            _make_tool_msg("t1", "old tool result"),
            _make_user_msg("do thing 2"),
            _make_assistant_msg("calling another tool"),
            _make_tool_msg("t2", "another old result"),
            _make_user_msg("do thing 3"),
            _make_assistant_msg("calling tool 3"),
            _make_tool_msg("t3", "recent 1"),
            _make_tool_msg("t4", "recent 2"),
            _make_tool_msg("t5", "recent 3"),
        ]
        cleared = context_manager._microcompact_tool_results(history)
        assert cleared == 2

        # User/assistant messages untouched
        assert history[0]["content"] == "do thing 1"
        assert history[1]["content"] == "calling tool"

        # Old tool results cleared
        assert json.loads(history[2]["content"])["cleared"] is True
        assert json.loads(history[5]["content"])["cleared"] is True

        # Recent tool results intact
        assert history[8]["content"] == "recent 1"

    def test_metadata_as_json_string(self, context_manager):
        """Metadata may be stored as JSON string, not dict."""
        meta_str = json.dumps({"tool_name": "Bash", "context_priority": "droppable"})
        history = [
            _make_tool_msg("t1", "old result", meta_str),
            _make_tool_msg("t2", "r1"),
            _make_tool_msg("t3", "r2"),
            _make_tool_msg("t4", "r3"),
        ]
        cleared = context_manager._microcompact_tool_results(history)
        assert cleared == 1
        marker = json.loads(history[0]["content"])
        assert marker["tool_name"] == "Bash"
