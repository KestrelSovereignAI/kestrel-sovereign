"""
Tests for context analysis: duplicate detection and token attribution by source.

Covers:
- ContextStats accumulator class (record, reset, duplicate detection)
- Input normalization (path canonicalization, whitespace stripping)
- get_analysis() returns correct structure and values
- Integration with ToolContextManager.get_status() (nested analysis key)
- Integration with ContextManager.get_status() pass-through
- Reset on compaction
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.agent.orchestrator_engine import ContextStats


class TestContextStatsBasic:
    """Test the ContextStats accumulator in isolation."""

    def test_fresh_instance_has_empty_analysis(self):
        stats = ContextStats()
        analysis = stats.get_analysis()
        assert analysis["total_tool_calls"] == 0
        assert analysis["unique_tool_calls"] == 0
        assert analysis["duplicate_tool_calls"] == 0
        assert analysis["duplicate_percent"] == 0.0
        assert analysis["total_result_chars"] == 0
        assert analysis["attribution_by_tool"] == {}
        assert analysis["duplicates"] == []

    def test_record_single_call(self):
        stats = ContextStats()
        stats.record("read_file", {"path": "/tmp/foo.txt"}, 1500)

        analysis = stats.get_analysis()
        assert analysis["total_tool_calls"] == 1
        assert analysis["unique_tool_calls"] == 1
        assert analysis["duplicate_tool_calls"] == 0
        assert analysis["total_result_chars"] == 1500
        assert "read_file" in analysis["attribution_by_tool"]
        assert analysis["attribution_by_tool"]["read_file"]["calls"] == 1
        assert analysis["attribution_by_tool"]["read_file"]["result_chars"] == 1500
        assert analysis["attribution_by_tool"]["read_file"]["percent_of_total"] == 100.0

    def test_record_multiple_different_calls(self):
        stats = ContextStats()
        stats.record("read_file", {"path": "/a.txt"}, 1000)
        stats.record("web_search", {"query": "hello"}, 3000)
        stats.record("memory_recall", {"topic": "dogs"}, 500)

        analysis = stats.get_analysis()
        assert analysis["total_tool_calls"] == 3
        assert analysis["unique_tool_calls"] == 3
        assert analysis["duplicate_tool_calls"] == 0
        assert analysis["total_result_chars"] == 4500

        # Attribution sorted by result_chars descending
        tools = list(analysis["attribution_by_tool"].keys())
        assert tools[0] == "web_search"
        assert tools[1] == "read_file"
        assert tools[2] == "memory_recall"

    def test_duplicate_detection_exact_match(self):
        stats = ContextStats()
        stats.record("read_file", {"path": "/tmp/foo.txt"}, 1500)
        stats.record("read_file", {"path": "/tmp/foo.txt"}, 1500)

        analysis = stats.get_analysis()
        assert analysis["total_tool_calls"] == 2
        assert analysis["unique_tool_calls"] == 1
        assert analysis["duplicate_tool_calls"] == 1
        assert analysis["duplicate_percent"] == 50.0
        assert len(analysis["duplicates"]) == 1
        assert analysis["duplicates"][0]["tool_name"] == "read_file"
        assert analysis["duplicates"][0]["call_index"] == 1

    def test_duplicate_detection_whitespace_normalization(self):
        """Trailing/leading whitespace should be stripped for comparison."""
        stats = ContextStats()
        stats.record("run_command", {"cmd": "ls -la"}, 200)
        stats.record("run_command", {"cmd": "  ls -la  "}, 200)

        analysis = stats.get_analysis()
        assert analysis["duplicate_tool_calls"] == 1

    def test_duplicate_detection_path_normalization(self):
        """Relative vs absolute paths that resolve to the same file should match."""
        stats = ContextStats()
        abs_path = os.path.abspath("/tmp/./foo/../foo/bar.txt")
        stats.record("read_file", {"path": "/tmp/./foo/../foo/bar.txt"}, 500)
        stats.record("read_file", {"path": abs_path}, 500)

        analysis = stats.get_analysis()
        assert analysis["duplicate_tool_calls"] == 1

    def test_different_tools_same_args_not_duplicate(self):
        stats = ContextStats()
        stats.record("read_file", {"path": "/tmp/foo.txt"}, 500)
        stats.record("write_file", {"path": "/tmp/foo.txt"}, 500)

        analysis = stats.get_analysis()
        assert analysis["duplicate_tool_calls"] == 0

    def test_same_tool_different_args_not_duplicate(self):
        stats = ContextStats()
        stats.record("read_file", {"path": "/tmp/foo.txt"}, 500)
        stats.record("read_file", {"path": "/tmp/bar.txt"}, 500)

        analysis = stats.get_analysis()
        assert analysis["duplicate_tool_calls"] == 0

    def test_reset_clears_all_state(self):
        stats = ContextStats()
        stats.record("read_file", {"path": "/tmp/foo.txt"}, 1000)
        stats.record("read_file", {"path": "/tmp/foo.txt"}, 1000)

        stats.reset()
        analysis = stats.get_analysis()
        assert analysis["total_tool_calls"] == 0
        assert analysis["duplicate_tool_calls"] == 0
        assert analysis["total_result_chars"] == 0
        assert analysis["attribution_by_tool"] == {}
        assert analysis["duplicates"] == []

    def test_after_reset_same_call_is_not_duplicate(self):
        stats = ContextStats()
        stats.record("read_file", {"path": "/tmp/foo.txt"}, 1000)
        stats.reset()
        stats.record("read_file", {"path": "/tmp/foo.txt"}, 1000)

        analysis = stats.get_analysis()
        assert analysis["duplicate_tool_calls"] == 0

    def test_multiple_duplicates_of_same_call(self):
        stats = ContextStats()
        for _ in range(5):
            stats.record("read_file", {"path": "/a.txt"}, 100)

        analysis = stats.get_analysis()
        assert analysis["total_tool_calls"] == 5
        assert analysis["unique_tool_calls"] == 1
        assert analysis["duplicate_tool_calls"] == 4
        assert analysis["duplicate_percent"] == 80.0

    def test_attribution_percent(self):
        stats = ContextStats()
        stats.record("tool_a", {"x": "1"}, 750)
        stats.record("tool_b", {"x": "2"}, 250)

        analysis = stats.get_analysis()
        assert analysis["attribution_by_tool"]["tool_a"]["percent_of_total"] == 75.0
        assert analysis["attribution_by_tool"]["tool_b"]["percent_of_total"] == 25.0

    def test_empty_args(self):
        stats = ContextStats()
        stats.record("list_files", {}, 300)

        analysis = stats.get_analysis()
        assert analysis["total_tool_calls"] == 1


class TestContextStatsNormalization:
    """Test input normalization edge cases."""

    def test_normalize_non_path_strings(self):
        """Non-path strings should just be stripped, not path-resolved."""
        sig1 = ContextStats._normalize_input("search", {"query": "hello world"})
        sig2 = ContextStats._normalize_input("search", {"query": "hello world "})
        assert sig1 == sig2

    def test_normalize_preserves_numeric_args(self):
        sig1 = ContextStats._normalize_input("resize", {"width": 100, "height": 200})
        sig2 = ContextStats._normalize_input("resize", {"height": 200, "width": 100})
        # Keys are sorted, so order shouldn't matter
        assert sig1 == sig2

    def test_normalize_sorts_keys(self):
        sig1 = ContextStats._normalize_input("t", {"b": "2", "a": "1"})
        sig2 = ContextStats._normalize_input("t", {"a": "1", "b": "2"})
        assert sig1 == sig2

    def test_summarize_input_truncation(self):
        args = {"long_content": "x" * 200}
        summary = ContextStats._summarize_input(args, max_len=80)
        assert len(summary) <= 80
        assert summary.endswith("...")

    def test_summarize_empty_args(self):
        summary = ContextStats._summarize_input({})
        assert summary == "(no args)"


class TestContextStatsIntegrationWithToolContextManager:
    """Test that ContextStats integrates correctly with ToolContextManager.get_status()."""

    @pytest.mark.asyncio
    async def test_get_status_includes_analysis_when_stats_provided(self):
        from kestrel_sovereign.agent.tool_context_manager import ToolContextManager

        mock_storage = MagicMock()
        mock_storage.conversation = None
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "gpt-4"

        tcm = ToolContextManager(
            storage=mock_storage,
            model="gpt-4",
            llm_service=mock_llm,
        )

        # Create a counter mock
        counter = MagicMock()
        counter.count = MagicMock(return_value=10)

        stats = ContextStats()
        stats.record("read_file", {"path": "/tmp/a.txt"}, 1000)
        stats.record("web_search", {"query": "test"}, 2000)
        stats.record("read_file", {"path": "/tmp/a.txt"}, 1000)

        status = await tcm.get_status(
            counter=counter,
            history=[],
            context_stats=stats,
        )

        assert status["success"] is True
        assert "analysis" in status

        analysis = status["analysis"]
        assert analysis["total_tool_calls"] == 3
        assert analysis["duplicate_tool_calls"] == 1
        assert analysis["total_result_chars"] == 4000
        assert "read_file" in analysis["attribution_by_tool"]
        assert "web_search" in analysis["attribution_by_tool"]

    @pytest.mark.asyncio
    async def test_get_status_omits_analysis_when_no_stats(self):
        from kestrel_sovereign.agent.tool_context_manager import ToolContextManager

        mock_storage = MagicMock()
        mock_storage.conversation = None
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "gpt-4"

        tcm = ToolContextManager(
            storage=mock_storage,
            model="gpt-4",
            llm_service=mock_llm,
        )

        counter = MagicMock()
        counter.count = MagicMock(return_value=10)

        status = await tcm.get_status(
            counter=counter,
            history=[],
            context_stats=None,
        )

        assert status["success"] is True
        assert "analysis" not in status


class TestContextStatsIntegrationWithContextManager:
    """Test that ContextManager.get_status() passes context_stats through."""

    @pytest.mark.asyncio
    async def test_context_manager_passes_stats_through(self):
        from kestrel_sovereign.agent.context_manager import ContextManager

        mock_storage = MagicMock()
        mock_storage.conversation = None
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "gpt-4"

        cm = ContextManager(
            storage=mock_storage,
            llm_service=mock_llm,
        )

        stats = ContextStats()
        stats.record("tool_x", {"k": "v"}, 500)

        # Patch get_status on the tool_context_manager to verify args
        cm.tool_context_manager.get_status = AsyncMock(return_value={"success": True, "analysis": stats.get_analysis()})

        result = await cm.get_status(context_stats=stats)
        # Verify context_stats was passed through
        call_kwargs = cm.tool_context_manager.get_status.call_args
        assert call_kwargs.kwargs.get("context_stats") is stats


class TestContextStatsResetOnCompaction:
    """Test that context stats reset when compaction happens."""

    @pytest.mark.asyncio
    async def test_feature_compact_resets_stats(self):
        from kestrel_sovereign.features.context.feature import ContextFeature

        # Build a mock agent with context_stats
        mock_agent = MagicMock()
        stats = ContextStats()
        stats.record("read_file", {"path": "/a.txt"}, 1000)
        mock_agent.context_stats = stats

        feature = ContextFeature.__new__(ContextFeature)
        feature.agent = mock_agent
        feature.context_manager = AsyncMock()
        feature.context_manager.compact_session = AsyncMock(return_value={
            "success": True,
            "messages_compacted": 5,
            "messages_preserved": 10,
            "tokens_saved": 500,
            "tokens_before": 1500,
            "tokens_after": 1000,
        })
        feature.llm_service = MagicMock()

        # Before compaction, stats has data
        assert stats.get_analysis()["total_tool_calls"] == 1

        # Compact
        result = await feature.compact_context(keep_recent=10)

        from kestrel_sdk.tools.result import ToolResultStatus
        assert result.status is ToolResultStatus.OK
        # After compaction, stats should be reset
        assert stats.get_analysis()["total_tool_calls"] == 0


class TestContextFeatureLateBoundContextManager:
    """#1382 — ``ContextFeature.initialize()`` used to snapshot
    ``agent.context_manager`` into ``self.context_manager``, but the
    agent constructs its ``ContextManager`` AFTER registering features
    (kestrel_agent.py:819 vs 1034). The snapshot was always ``None``,
    so every @tool returned "Context manager not available" — including
    on multi-agent satellites where Nellie hit the bug.

    Fix: ``context_manager`` is now a property that reads
    ``self.agent.context_manager`` at call time. These tests pin that.
    """

    @pytest.mark.asyncio
    async def test_status_resolves_after_late_attach(self):
        """initialize() runs FIRST with no context_manager attached, the
        agent attaches one later, and the tool then succeeds — the
        registration-vs-init race that #1382 surfaced."""
        from types import SimpleNamespace

        from kestrel_sovereign.features.context.feature import ContextFeature
        from kestrel_sdk.tools.result import ToolResultStatus

        agent = SimpleNamespace()  # No context_manager attribute yet.
        feature = ContextFeature(agent)
        await feature.initialize()
        assert feature.context_manager is None

        # Agent attaches its real ContextManager — same ordering the
        # production init has in kestrel_agent.py:1034.
        cm = AsyncMock()
        cm.get_status = AsyncMock(return_value={
            "utilization_percent": 12,
            "message_count": 4,
        })
        agent.context_manager = cm
        agent.context_stats = None

        result = await feature.context_status()
        assert result.status is ToolResultStatus.OK
        assert result.data["utilization_percent"] == 12

    @pytest.mark.asyncio
    async def test_status_reports_missing_when_genuinely_unattached(self):
        """If the agent really never attaches a context_manager, the
        tool surfaces a precise error (not the historical opaque
        'Context manager not available')."""
        from types import SimpleNamespace

        from kestrel_sovereign.features.context.feature import ContextFeature
        from kestrel_sdk.tools.result import ToolResultStatus

        feature = ContextFeature(SimpleNamespace())
        await feature.initialize()

        result = await feature.context_status()
        assert result.status is ToolResultStatus.ERROR
        assert "context_manager is not attached" in (result.error or "")


class TestDispatchToolCallRecording:
    """Test that _dispatch_tool_call records into context_stats."""

    @pytest.mark.asyncio
    async def test_dispatch_records_into_stats(self):
        """Verify the recording path in _dispatch_tool_call."""
        from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin

        # Build a minimal mock agent with the mixin
        agent = MagicMock(spec=[])
        agent.did = "did:test:123"
        agent.context_stats = ContextStats()
        agent.observability_store = AsyncMock()
        agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
        agent.observability_store.log_tool_response = AsyncMock()
        agent.observability_store.log_error = AsyncMock()
        agent.features = {}
        agent._direct_tools = {}
        agent._tool_to_feature = {}
        agent.hooks_manager = AsyncMock()

        # Bind the mixin's method
        agent._dispatch_tool_call = OrchestratorEngineMixin._dispatch_tool_call.__get__(agent)

        # Create a mock tool_call — use a name that's in known_tools
        # so validation passes, but not in features_by_tool_name or _direct_tools
        # so it falls through to the "unknown feature tool" path and still records.
        tool_call = MagicMock()
        tool_call.name = "test_tool"
        tool_call.id = "tc-1"
        tool_call.arguments = {"param": "value"}

        messages = []
        features_by_tool_name = {}
        known_tools = {"test_tool"}

        # Dispatch — tool passes validation but is not in features or direct_tools,
        # so it produces an error result. The recording still happens after the result
        # is appended to messages.
        with patch("kestrel_sovereign.agent.orchestrator_engine._init_constants"):
            with patch("kestrel_sovereign.agent.orchestrator_engine.MAX_TOOL_RESULT_CHARS", 100000):
                    await agent._dispatch_tool_call(
                        tool_call, features_by_tool_name, known_tools,
                        messages, 0, "test message",
                    )

        analysis = agent.context_stats.get_analysis()
        assert analysis["total_tool_calls"] == 1
        assert "test_tool" in analysis["attribution_by_tool"]
        assert analysis["attribution_by_tool"]["test_tool"]["calls"] == 1
        assert analysis["attribution_by_tool"]["test_tool"]["result_chars"] > 0


class TestContextStatsSessionReset:
    """Test that context stats reset on session change."""

    def test_check_session_resets_on_change(self):
        stats = ContextStats()
        # Set initial session
        stats.check_session("session-1")

        stats.record("Read", {"file_path": "/a.txt"}, 100)
        stats.record("Read", {"file_path": "/b.txt"}, 200)
        assert stats.get_analysis()["total_tool_calls"] == 2

        # Same session — no reset
        stats.check_session("session-1")
        assert stats.get_analysis()["total_tool_calls"] == 2

        # Change session — should reset
        stats.check_session("session-2")
        assert stats.get_analysis()["total_tool_calls"] == 0

    def test_check_session_no_reset_same_session(self):
        stats = ContextStats()
        stats.check_session("session-1")
        stats.record("Bash", {"command": "ls"}, 50)
        stats.check_session("session-1")  # Same session
        assert stats.get_analysis()["total_tool_calls"] == 1  # Not reset

    def test_check_session_none_does_nothing(self):
        stats = ContextStats()
        stats.record("Read", {"file_path": "/a.txt"}, 100)
        stats.check_session(None)
        assert stats.get_analysis()["total_tool_calls"] == 1
