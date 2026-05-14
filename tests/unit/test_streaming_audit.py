"""
Unit tests for streaming response functionality.

Tests streaming behavior after per-response audit removal.
Only the local constitution hash check (_maybe_audit) remains.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestStreamingBasics:
    """Tests for basic streaming behavior."""

    @pytest.mark.asyncio
    async def test_command_input_uses_regular_processing(self):
        """Test that commands fall back to non-streaming processing."""
        from kestrel_sovereign.agent.streaming import StreamingMixin

        mock_agent = MagicMock()
        mock_agent._maybe_audit = AsyncMock()
        mock_agent.process_input = AsyncMock(return_value="Command executed successfully")
        mock_agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(mock_agent)

        chunks = []
        async for chunk in mock_agent.process_input_streaming("!help"):
            chunks.append(chunk)

        mock_agent._maybe_audit.assert_called_once()
        mock_agent.process_input.assert_called_once_with("!help", None, session_id=None, caller=None)
        assert "Command executed" in "".join(chunks)


class TestRealStreaming:
    """Tests for real LLM streaming (not fake chunking)."""

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_chunks(self):
        """Test that stream_with_messages yields chunks from the LLM."""
        from kestrel_sovereign.llm.service import LLMService

        mock_service = MagicMock(spec=LLMService)

        async def mock_stream(**kwargs):
            for word in ["Hello", " ", "World", "!"]:
                yield word

        mock_service.stream_with_messages = mock_stream

        chunks = []
        async for chunk in mock_service.stream_with_messages(
            messages=[{"role": "user", "content": "test"}],
            force_local_only=False,
            model_override=None
        ):
            chunks.append(chunk)

        assert len(chunks) == 4
        assert "".join(chunks) == "Hello World!"

    @pytest.mark.asyncio
    async def test_orchestrator_streaming_yields_real_chunks(self):
        """Test that _handle_orchestrator_response_streaming yields real chunks."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sovereign.llm.adapter import LLMResponse

        mock_agent = MagicMock()
        mock_agent.features = {}
        mock_agent.did = "test-did"

        mock_agent.observability_store = MagicMock()
        mock_agent.observability_store.log_tool_call = AsyncMock(return_value="event-1")
        mock_agent.observability_store.log_tool_response = AsyncMock()

        response = LLMResponse(content="Simple response", tool_calls=[])

        mock_agent._handle_orchestrator_response_streaming = (
            KestrelAgent._handle_orchestrator_response_streaming.__get__(mock_agent)
        )
        # Reflection phase (#1238) runs at every final-return path; bind it so
        # the streaming handler can call self._run_reflection_phase. With no
        # fact-save tools loaded it short-circuits.
        for refl_method in ("_run_reflection_phase", "_log_reflection_call", "_finalize_turn"):
            setattr(
                mock_agent,
                refl_method,
                getattr(KestrelAgent, refl_method).__get__(mock_agent),
            )
        mock_agent._build_all_tools = MagicMock(return_value=[])

        chunks = []
        async for chunk in mock_agent._handle_orchestrator_response_streaming(
            response=response,
            feature_tools=[],
            system_prompt="test",
            force_local_only=False,
            effective_model="test-model",
            user_message="test message"
        ):
            chunks.append(chunk)

        assert "".join(chunks) == "Simple response"

    @pytest.mark.asyncio
    async def test_orchestrator_streaming_with_tool_calls(self):
        """Test that tool calls are executed then response is streamed."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
        from kestrel_sovereign.hooks import HooksManager

        mock_agent = MagicMock()
        mock_agent.did = "test-did"

        mock_feature = MagicMock()
        mock_feature.tool_name = "test_tool"
        mock_feature.name = "test_feature"
        mock_feature.execute_as_subagent = AsyncMock(return_value={"success": True, "data": "result"})
        mock_feature.to_orchestrator_tool.return_value = {
            "type": "function",
            "function": {"name": "test_tool", "description": "test", "parameters": {}}
        }
        mock_agent.features = {"test_feature": mock_feature}

        mock_agent.hooks_manager = HooksManager()

        mock_agent.observability_store = MagicMock()
        mock_agent.observability_store.log_tool_call = AsyncMock(return_value="event-1")
        mock_agent.observability_store.log_tool_response = AsyncMock()

        mock_agent._direct_tools = {}
        mock_agent._tool_to_feature = {}

        mock_agent.llm_service = MagicMock()

        first_response = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={"task": "do something"})]
        )
        second_response = LLMResponse(content="Final response", tool_calls=[])

        mock_agent.llm_service.generate_with_messages = AsyncMock(return_value=second_response)

        async def mock_stream(**kwargs):
            for word in ["Final", " ", "streamed", " ", "response"]:
                yield word

        mock_agent.llm_service.stream_with_messages = mock_stream

        # Bind all orchestrator engine and tool registry mixin methods
        for method_name in (
            '_handle_orchestrator_response_streaming',
            '_execute_tool_with_hooks',
            '_execute_tool_batch',
            '_partition_tool_calls',
            '_dispatch_tool_call',
            '_dispatch_feature_tool',
            '_dispatch_direct_tool',
            '_get_denied_tools',
            '_handle_feature_error',
            '_prune_orchestrator_messages',
            '_build_all_tools',
            '_build_feature_tools',
            '_visible_features_by_tool_name',
            '_visible_known_tool_names',
            '_hidden_context_features',
            '_hidden_context_tools',
            '_feature_hidden_from_context',
            '_direct_tool_hidden_from_context',
            # Reflection phase (#1238) lives on OrchestratorEngineMixin and is
            # called from every final-return path of the streaming handler.
            '_run_reflection_phase',
            '_log_reflection_call',
            '_finalize_turn',
        ):
            setattr(mock_agent, method_name,
                    getattr(KestrelAgent, method_name).__get__(mock_agent))
        mock_agent._build_tool_calls_msg = KestrelAgent._build_tool_calls_msg
        mock_agent._explored_features = {}
        mock_agent._direct_tool_defs = []
        mock_agent._register_explored_feature_tools = MagicMock()

        chunks = []
        async for chunk in mock_agent._handle_orchestrator_response_streaming(
            response=first_response,
            feature_tools=[],
            system_prompt="test",
            force_local_only=False,
            effective_model="test-model",
            user_message="test message"
        ):
            chunks.append(chunk)

        mock_feature.execute_as_subagent.assert_called_once()

        full_output = "".join(chunks)
        assert "Final streamed response" in full_output
        assert "test_tool" in full_output
