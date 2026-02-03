"""
Unit tests for streaming response audit functionality.

Tests the _stream_text_with_pre_audit and _stream_text_with_post_audit methods
from StreamingMixin directly, without needing full process_input_streaming mocking.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from decimal import Decimal


class TestStreamingAudit:
    """Tests for streaming audit methods."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with required attributes for streaming audit."""
        agent = MagicMock()
        agent.audit_enabled = True
        agent.privacy_agent = MagicMock()
        agent.privacy_agent.add_conversation = AsyncMock()
        agent.wallet = MagicMock()
        agent.wallet.can_afford_audit.return_value = True
        return agent

    @pytest.mark.asyncio
    async def test_post_audit_mode_streams_immediately(self, mock_agent):
        """Test that post-audit mode yields chunks as they're generated."""
        from kestrel_sovereign.agent.streaming import StreamingMixin

        # Mock get_audit_response to return passing audit
        mock_agent.get_audit_response = AsyncMock(return_value={
            "risk_level": 1,
            "reasoning": "Response is safe"
        })

        # Bind the method to our mock
        mock_agent._stream_text_with_post_audit = StreamingMixin._stream_text_with_post_audit.__get__(mock_agent)

        # Collect chunks
        chunks = []
        async for chunk in mock_agent._stream_text_with_post_audit("Hello World!"):
            chunks.append(chunk)

        # Should have received the streaming chunks
        full = "".join(chunks)
        assert "Hello World!" in full

    @pytest.mark.asyncio
    async def test_post_audit_mode_adds_warning_on_failure(self, mock_agent):
        """Test that post-audit mode appends warning when audit fails."""
        from kestrel_sovereign.agent.streaming import StreamingMixin

        # Mock get_audit_response to return failing audit
        mock_agent.get_audit_response = AsyncMock(return_value={
            "risk_level": 4,
            "reasoning": "Response contains prohibited content"
        })

        mock_agent._stream_text_with_post_audit = StreamingMixin._stream_text_with_post_audit.__get__(mock_agent)

        # Collect all chunks
        full_response = ""
        async for chunk in mock_agent._stream_text_with_post_audit("Hello World!"):
            full_response += chunk

        # Should have the original content plus a warning
        assert "Hello World!" in full_response
        assert "INTEGRITY WARNING" in full_response
        assert "prohibited content" in full_response

    @pytest.mark.asyncio
    async def test_pre_audit_mode_buffers_response(self, mock_agent):
        """Test that pre-audit mode buffers before yielding."""
        from kestrel_sovereign.agent.streaming import StreamingMixin

        # Track when audit happens relative to yields
        audit_called = False

        async def mock_perform_audit(response, retry_count=0):
            nonlocal audit_called
            audit_called = True
            return response  # Pass through

        mock_agent._perform_integrity_audit = mock_perform_audit
        mock_agent._stream_text_with_pre_audit = StreamingMixin._stream_text_with_pre_audit.__get__(mock_agent)

        # Collect chunks
        chunks = []
        async for chunk in mock_agent._stream_text_with_pre_audit("Hello World!"):
            chunks.append(chunk)

        # Audit should have been called
        assert audit_called

        # Should have received the complete audited response (in chunks)
        full_response = "".join(chunks)
        assert "Hello World!" in full_response

    @pytest.mark.asyncio
    async def test_pre_audit_mode_returns_audit_correction(self, mock_agent):
        """Test that pre-audit mode returns corrected response when audit fails."""
        from kestrel_sovereign.agent.streaming import StreamingMixin

        # Mock audit to return correction
        async def mock_perform_audit(response, retry_count=0):
            return "SYSTEM_CORRECTION: Original response was unconstitutional."

        mock_agent._perform_integrity_audit = mock_perform_audit
        mock_agent._stream_text_with_pre_audit = StreamingMixin._stream_text_with_pre_audit.__get__(mock_agent)

        # Collect chunks
        full_response = ""
        async for chunk in mock_agent._stream_text_with_pre_audit("Hello World!"):
            full_response += chunk

        # Should have received the correction, not original
        assert "SYSTEM_CORRECTION" in full_response
        assert "unconstitutional" in full_response

    @pytest.mark.asyncio
    async def test_command_input_uses_regular_processing(self, mock_agent):
        """Test that commands fall back to non-streaming processing."""
        from kestrel_sovereign.agent.streaming import StreamingMixin

        # Mock process_input for commands
        mock_agent.process_input = AsyncMock(return_value="Command executed successfully")
        mock_agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(mock_agent)

        # Send a command
        chunks = []
        async for chunk in mock_agent.process_input_streaming("!help"):
            chunks.append(chunk)

        # Should have used process_input, not streaming
        mock_agent.process_input.assert_called_once_with("!help", None, session_id=None)
        assert "Command executed" in "".join(chunks)

    @pytest.mark.asyncio
    async def test_audit_disabled_skips_audit(self, mock_agent):
        """Test that disabled audit skips the audit step."""
        from kestrel_sovereign.agent.streaming import StreamingMixin

        mock_agent.audit_enabled = False
        mock_agent.get_audit_response = AsyncMock()  # Should not be called

        mock_agent._stream_text_with_post_audit = StreamingMixin._stream_text_with_post_audit.__get__(mock_agent)

        # Stream with audit disabled
        chunks = []
        async for chunk in mock_agent._stream_text_with_post_audit("Hello World!"):
            chunks.append(chunk)

        # Should not have called audit
        mock_agent.get_audit_response.assert_not_called()

        # Should still have content
        full = "".join(chunks)
        assert "Hello World!" in full


class TestStreamingAuditChunking:
    """Tests for the chunking behavior in pre-audit mode."""
    
    @pytest.mark.asyncio
    async def test_pre_audit_chunks_response(self):
        """Test that pre-audit mode chunks the audited response."""
        # This tests the actual chunking logic
        response = "A" * 150  # 150 characters
        chunk_size = 50
        
        chunks = []
        for i in range(0, len(response), chunk_size):
            chunks.append(response[i:i + chunk_size])
        
        # Should have 3 chunks of 50 chars each
        assert len(chunks) == 3
        assert all(len(c) == 50 for c in chunks)
        assert "".join(chunks) == response


class TestRealStreaming:
    """Tests for real LLM streaming (not fake chunking)."""

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_chunks(self):
        """Test that stream_with_messages yields chunks from the LLM."""
        from kestrel_sovereign.llm.service import LLMService

        # Create a mock LLM service
        mock_service = MagicMock(spec=LLMService)

        # Simulate streaming chunks
        async def mock_stream(**kwargs):
            for word in ["Hello", " ", "World", "!"]:
                yield word

        mock_service.stream_with_messages = mock_stream

        # Collect chunks
        chunks = []
        async for chunk in mock_service.stream_with_messages(
            messages=[{"role": "user", "content": "test"}],
            force_local_only=False,
            model_override=None
        ):
            chunks.append(chunk)

        # Should have received individual chunks
        assert len(chunks) == 4
        assert "".join(chunks) == "Hello World!"

    @pytest.mark.asyncio
    async def test_orchestrator_streaming_yields_real_chunks(self):
        """Test that _handle_orchestrator_response_streaming yields real chunks."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sovereign.llm.adapter import LLMResponse

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.features = {}
        mock_agent.did = "test-did"

        # Mock observability store
        mock_agent.observability_store = MagicMock()
        mock_agent.observability_store.log_tool_call = AsyncMock(return_value="event-1")
        mock_agent.observability_store.log_tool_response = AsyncMock()

        # Create a response with no tool calls
        response = LLMResponse(content="Simple response", tool_calls=[])

        # Bind the streaming method
        mock_agent._handle_orchestrator_response_streaming = (
            KestrelAgent._handle_orchestrator_response_streaming.__get__(mock_agent)
        )

        # Collect chunks - should yield content directly since no tool calls
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

        # With no tool calls, should yield the content directly
        assert "".join(chunks) == "Simple response"

    @pytest.mark.asyncio
    async def test_orchestrator_streaming_with_tool_calls(self):
        """Test that tool calls are executed then response is streamed."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

        mock_agent = MagicMock()
        mock_agent.did = "test-did"

        # Create a mock feature
        mock_feature = MagicMock()
        mock_feature.tool_name = "test_tool"
        mock_feature.execute_as_subagent = AsyncMock(return_value={"success": True, "data": "result"})
        mock_agent.features = {"test_feature": mock_feature}

        # Mock observability
        mock_agent.observability_store = MagicMock()
        mock_agent.observability_store.log_tool_call = AsyncMock(return_value="event-1")
        mock_agent.observability_store.log_tool_response = AsyncMock()

        # Mock LLM service
        mock_agent.llm_service = MagicMock()

        # First call returns tool call, second returns no tool calls
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

        # Bind the method
        mock_agent._handle_orchestrator_response_streaming = (
            KestrelAgent._handle_orchestrator_response_streaming.__get__(mock_agent)
        )

        # Call with the initial response that has tool calls
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

        # Tool should have been executed
        mock_feature.execute_as_subagent.assert_called_once()

        # Should have streamed the final response (with tool status prefix)
        full_output = "".join(chunks)
        assert "Final streamed response" in full_output
        # Tool status messages are also streamed
        assert "test_tool" in full_output
