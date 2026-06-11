"""
Streaming Tool Detection Tests

Comprehensive tests for the get_streaming_response_with_tools() method
across all LLM providers (OpenAI, Anthropic, Ollama, Vertex).

This tests the core feature: stream text as it arrives, detect tool calls
from the stream, and yield an LLMResponse at the end if tools were called.
"""
import asyncio
import json
import os
import pytest
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.llm import ToolCallStarted

from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter


# =============================================================================
# Test Tools Definition
# =============================================================================

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]


# =============================================================================
# OpenAI Adapter Tests (Unit - Mocked)
# =============================================================================

class TestOpenAIStreamingToolDetectionUnit:
    """Unit tests for OpenAI adapter streaming with tool detection."""

    @pytest.mark.asyncio
    async def test_text_only_streaming(self):
        """Test streaming text without tool calls."""
        adapter = OpenAIAdapter()

        # Create mock stream that yields text chunks
        async def mock_stream():
            chunks = ["Hello", " ", "world", "!"]
            for i, text in enumerate(chunks):
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = MagicMock()
                chunk.choices[0].delta.content = text
                chunk.choices[0].delta.tool_calls = None
                chunk.usage = None
                yield chunk
            # Final chunk with usage
            final_chunk = MagicMock()
            final_chunk.choices = []
            final_chunk.usage = MagicMock()
            final_chunk.usage.prompt_tokens = 10
            final_chunk.usage.completion_tokens = 5
            final_chunk.usage.total_tokens = 15
            yield final_chunk

        mock_response = mock_stream()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Collect results
        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=mock_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            tools=None
        ):
            results.append(item)

        # #1684: a text-only stream now ALSO emits a terminal LLMResponse
        # carrying token usage (previously dropped — a silent billing
        # undercount), so the service layer can meter the turn.
        text_chunks = [r for r in results if isinstance(r, str)]
        finals = [r for r in results if isinstance(r, LLMResponse)]
        assert "".join(text_chunks) == "Hello world!"
        assert len(finals) == 1
        assert not finals[0].tool_calls  # text-only: no tool calls
        assert finals[0].input_tokens == 10
        assert finals[0].output_tokens == 5

    @pytest.mark.asyncio
    async def test_single_tool_call_detection(self):
        """Test detecting a single tool call from stream."""
        adapter = OpenAIAdapter()

        # Create mock stream that yields tool call deltas
        async def mock_stream():
            # First chunk: tool call start with id and name
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta = MagicMock()
            chunk1.choices[0].delta.content = None
            tc1 = MagicMock()
            tc1.index = 0
            tc1.id = "call_abc123"
            tc1.function = MagicMock()
            tc1.function.name = "get_weather"
            tc1.function.arguments = '{"loc'
            chunk1.choices[0].delta.tool_calls = [tc1]
            chunk1.usage = None
            yield chunk1

            # Second chunk: more arguments
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta = MagicMock()
            chunk2.choices[0].delta.content = None
            tc2 = MagicMock()
            tc2.index = 0
            tc2.id = None
            tc2.function = MagicMock()
            tc2.function.name = None
            tc2.function.arguments = 'ation": "San Francisco"}'
            chunk2.choices[0].delta.tool_calls = [tc2]
            chunk2.usage = None
            yield chunk2

            # Final chunk with usage
            final_chunk = MagicMock()
            final_chunk.choices = []
            final_chunk.usage = MagicMock()
            final_chunk.usage.prompt_tokens = 25
            final_chunk.usage.completion_tokens = 12
            final_chunk.usage.total_tokens = 37
            yield final_chunk

        mock_response = mock_stream()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Collect results
        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=mock_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=SAMPLE_TOOLS
        ):
            results.append(item)

        # SDK 0.7.0+: stream now yields ToolCallStarted before the
        # final LLMResponse. ToolCallStarted carries id/name from the
        # first delta (when populated) and signals "tool call begun"
        # for the constitutional honesty layer.
        starts = [r for r in results if isinstance(r, ToolCallStarted)]
        finals = [r for r in results if isinstance(r, LLMResponse)]
        assert len(starts) == 1
        assert starts[0] == ToolCallStarted(
            index=0, id="call_abc123", name="get_weather"
        )
        assert len(finals) == 1
        final = finals[0]
        assert final.has_tool_calls
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0].id == "call_abc123"
        assert final.tool_calls[0].name == "get_weather"
        assert final.tool_calls[0].arguments == {"location": "San Francisco"}
        # Token counts
        assert final.input_tokens == 25
        assert final.output_tokens == 12
        assert final.total_tokens == 37

    @pytest.mark.asyncio
    async def test_multiple_parallel_tool_calls(self):
        """Test detecting multiple parallel tool calls."""
        adapter = OpenAIAdapter()

        async def mock_stream():
            # Tool call 1 - get_weather
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta = MagicMock()
            chunk1.choices[0].delta.content = None
            tc1 = MagicMock()
            tc1.index = 0
            tc1.id = "call_weather"
            tc1.function = MagicMock()
            tc1.function.name = "get_weather"
            tc1.function.arguments = '{"location": "NYC"}'
            chunk1.choices[0].delta.tool_calls = [tc1]
            chunk1.usage = None
            yield chunk1

            # Tool call 2 - search_web (different index)
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta = MagicMock()
            chunk2.choices[0].delta.content = None
            tc2 = MagicMock()
            tc2.index = 1
            tc2.id = "call_search"
            tc2.function = MagicMock()
            tc2.function.name = "search_web"
            tc2.function.arguments = '{"query": "news"}'
            chunk2.choices[0].delta.tool_calls = [tc2]
            chunk2.usage = None
            yield chunk2

            # Final chunk
            final_chunk = MagicMock()
            final_chunk.choices = []
            final_chunk.usage = MagicMock()
            final_chunk.usage.prompt_tokens = 30
            final_chunk.usage.completion_tokens = 20
            final_chunk.usage.total_tokens = 50
            yield final_chunk

        mock_response = mock_stream()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=mock_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Weather and news"}],
            tools=SAMPLE_TOOLS
        ):
            results.append(item)

        # SDK 0.7.0+: one ToolCallStarted per distinct tool-call index,
        # in arrival order, then a final LLMResponse with both calls.
        starts = [r for r in results if isinstance(r, ToolCallStarted)]
        finals = [r for r in results if isinstance(r, LLMResponse)]
        assert [s.index for s in starts] == [0, 1]
        assert [s.name for s in starts] == ["get_weather", "search_web"]
        assert len(finals) == 1
        final = finals[0]
        assert final.has_tool_calls
        assert len(final.tool_calls) == 2

        # Check tool calls are in order
        assert final.tool_calls[0].name == "get_weather"
        assert final.tool_calls[1].name == "search_web"

    @pytest.mark.asyncio
    async def test_text_then_tool_calls(self):
        """Test stream with text followed by tool calls."""
        adapter = OpenAIAdapter()

        async def mock_stream():
            # Text chunk first
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta = MagicMock()
            chunk1.choices[0].delta.content = "Let me check the weather. "
            chunk1.choices[0].delta.tool_calls = None
            chunk1.usage = None
            yield chunk1

            # Then tool call
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta = MagicMock()
            chunk2.choices[0].delta.content = None
            tc = MagicMock()
            tc.index = 0
            tc.id = "call_123"
            tc.function = MagicMock()
            tc.function.name = "get_weather"
            tc.function.arguments = '{"location": "LA"}'
            chunk2.choices[0].delta.tool_calls = [tc]
            chunk2.usage = None
            yield chunk2

            # Final
            final_chunk = MagicMock()
            final_chunk.choices = []
            final_chunk.usage = None
            yield final_chunk

        mock_response = mock_stream()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=mock_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Weather"}],
            tools=SAMPLE_TOOLS
        ):
            results.append(item)

        # SDK 0.7.0+: text chunk, then ToolCallStarted (when the tool
        # delta first arrives), then final LLMResponse.
        assert isinstance(results[0], str)
        assert results[0] == "Let me check the weather. "
        starts = [r for r in results if isinstance(r, ToolCallStarted)]
        finals = [r for r in results if isinstance(r, LLMResponse)]
        assert len(starts) == 1
        assert starts[0].index == 0
        assert len(finals) == 1
        final = finals[0]
        assert final.has_tool_calls
        assert final.content == "Let me check the weather. "
        # Marker arrives AFTER the leading text chunk.
        assert results.index(starts[0]) > 0

    @pytest.mark.asyncio
    async def test_malformed_json_arguments(self):
        """Test handling of malformed JSON in tool arguments."""
        adapter = OpenAIAdapter()

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = None
            tc = MagicMock()
            tc.index = 0
            tc.id = "call_bad"
            tc.function = MagicMock()
            tc.function.name = "get_weather"
            tc.function.arguments = 'not valid json {'  # Malformed
            chunk.choices[0].delta.tool_calls = [tc]
            chunk.usage = None
            yield chunk

            final = MagicMock()
            final.choices = []
            final.usage = None
            yield final

        mock_response = mock_stream()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=mock_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "test"}],
            tools=SAMPLE_TOOLS
        ):
            results.append(item)

        # SDK 0.7.0+: ToolCallStarted precedes the final LLMResponse.
        # Malformed JSON sentinel renamed from "raw" to "_raw" in
        # 0.7.0 to signal "sentinel, not real data".
        starts = [r for r in results if isinstance(r, ToolCallStarted)]
        finals = [r for r in results if isinstance(r, LLMResponse)]
        assert len(starts) == 1
        assert len(finals) == 1
        final = finals[0]
        assert final.has_tool_calls
        assert final.tool_calls[0].arguments == {"_raw": "not valid json {"}


# =============================================================================
# Anthropic Adapter Tests (Unit - Mocked)
# =============================================================================

class TestAnthropicStreamingToolDetectionUnit:
    """Unit tests for Anthropic adapter streaming with tool detection."""

    @pytest.mark.asyncio
    async def test_text_streaming(self):
        """Test streaming text from Anthropic."""
        adapter = AnthropicAdapter()

        # Mock stream events
        events = []

        # message_start with usage
        msg_start = MagicMock()
        msg_start.type = 'message_start'
        msg_start.message = MagicMock()
        msg_start.message.usage = MagicMock()
        msg_start.message.usage.input_tokens = 15
        events.append(msg_start)

        # content_block_start for text
        block_start = MagicMock()
        block_start.type = 'content_block_start'
        block_start.content_block = MagicMock()
        block_start.content_block.type = 'text'
        block_start.index = 0
        events.append(block_start)

        # text deltas
        for text in ["Hello", " from", " Claude"]:
            delta = MagicMock()
            delta.type = 'content_block_delta'
            delta.delta = MagicMock()
            delta.delta.type = 'text_delta'
            delta.delta.text = text
            events.append(delta)

        # message_delta with output tokens
        msg_delta = MagicMock()
        msg_delta.type = 'message_delta'
        msg_delta.usage = MagicMock()
        msg_delta.usage.output_tokens = 8
        events.append(msg_delta)

        async def mock_stream_iter(self):
            for event in events:
                yield event

        mock_stream = AsyncMock()
        mock_stream.__aiter__ = mock_stream_iter
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=mock_client,
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            tools=None
        ):
            results.append(item)

        # Should yield text chunks only
        text_results = [r for r in results if isinstance(r, str)]
        assert len(text_results) == 3
        assert "".join(text_results) == "Hello from Claude"

    @pytest.mark.asyncio
    async def test_tool_use_detection(self):
        """Test detecting tool_use from Anthropic stream."""
        adapter = AnthropicAdapter()

        events = []

        # message_start
        msg_start = MagicMock()
        msg_start.type = 'message_start'
        msg_start.message = MagicMock()
        msg_start.message.usage = MagicMock()
        msg_start.message.usage.input_tokens = 20
        events.append(msg_start)

        # content_block_start for tool_use
        tool_start = MagicMock()
        tool_start.type = 'content_block_start'
        tool_start.index = 0
        tool_start.content_block = MagicMock()
        tool_start.content_block.type = 'tool_use'
        tool_start.content_block.id = 'toolu_abc'
        tool_start.content_block.name = 'get_weather'
        events.append(tool_start)

        # input_json_delta chunks
        for chunk in ['{"loc', 'ation":', ' "Paris"}']:
            delta = MagicMock()
            delta.type = 'content_block_delta'
            delta.delta = MagicMock()
            delta.delta.type = 'input_json_delta'
            delta.delta.partial_json = chunk
            events.append(delta)

        # content_block_stop
        block_stop = MagicMock()
        block_stop.type = 'content_block_stop'
        events.append(block_stop)

        # message_delta
        msg_delta = MagicMock()
        msg_delta.type = 'message_delta'
        msg_delta.usage = MagicMock()
        msg_delta.usage.output_tokens = 15
        events.append(msg_delta)

        async def mock_stream_iter(self):
            for event in events:
                yield event

        mock_stream = AsyncMock()
        mock_stream.__aiter__ = mock_stream_iter
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=mock_client,
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}],
            tools=SAMPLE_TOOLS
        ):
            results.append(item)

        # SDK 0.7.0+: Anthropic emits ToolCallStarted at
        # content_block_start with type='tool_use' (id and name
        # populated), then the final LLMResponse. The marker's
        # ``index`` matches the event's ``index`` field — 0 here
        # because the mock stream has only one content block.
        starts = [r for r in results if isinstance(r, ToolCallStarted)]
        finals = [r for r in results if isinstance(r, LLMResponse)]
        assert len(starts) == 1
        assert starts[0] == ToolCallStarted(
            index=0, id="toolu_abc", name="get_weather"
        )
        assert len(finals) == 1
        final = finals[0]
        assert final.has_tool_calls
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0].id == 'toolu_abc'
        assert final.tool_calls[0].name == 'get_weather'
        assert final.tool_calls[0].arguments == {"location": "Paris"}


# =============================================================================
# Ollama Adapter Tests (Unit - Mocked Fallback)
# =============================================================================

class TestOllamaStreamingToolDetectionUnit:
    """Unit tests for Ollama adapter streaming with tool detection (fallback)."""

    @pytest.mark.asyncio
    async def test_fallback_with_tool_calls(self):
        """Test Ollama fallback: non-streaming when tools are detected."""
        adapter = OllamaAdapter()

        # Mock get_response to return tool calls
        mock_response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="ollama_call_0", name="get_weather", arguments={"location": "Tokyo"})
            ],
            input_tokens=10,
            output_tokens=5,
            total_tokens=15
        )

        with patch.object(adapter, 'get_response', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            mock_client = MagicMock()
            results = []
            async for item in adapter.get_streaming_response_with_tools(
                client=mock_client,
                model="llama3.2:3b",
                messages=[{"role": "user", "content": "Weather?"}],
                tools=SAMPLE_TOOLS
            ):
                results.append(item)

            # Should yield the LLMResponse immediately
            assert len(results) == 1
            assert isinstance(results[0], LLMResponse)
            assert results[0].has_tool_calls
            assert results[0].tool_calls[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_fallback_no_tool_calls(self):
        """Test Ollama fallback: yields content when no tools called."""
        adapter = OllamaAdapter()

        # Mock get_response to return text only
        mock_response = LLMResponse(
            content="The weather is sunny.",
            tool_calls=None,
            input_tokens=8,
            output_tokens=6,
            total_tokens=14
        )

        with patch.object(adapter, 'get_response', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            mock_client = MagicMock()
            results = []
            async for item in adapter.get_streaming_response_with_tools(
                client=mock_client,
                model="llama3.2:3b",
                messages=[{"role": "user", "content": "Weather?"}],
                tools=SAMPLE_TOOLS
            ):
                results.append(item)

            # #1684: the text-only fallback now also emits the terminal
            # LLMResponse (carrying usage) after the streamed content so the
            # service layer can meter the turn.
            text = [r for r in results if isinstance(r, str)]
            finals = [r for r in results if isinstance(r, LLMResponse)]
            assert "".join(text) == "The weather is sunny."
            assert len(finals) == 1
            assert finals[0].input_tokens == 8 and finals[0].output_tokens == 6

    @pytest.mark.asyncio
    async def test_no_tools_uses_streaming(self):
        """Test Ollama uses regular streaming when no tools provided."""
        adapter = OllamaAdapter()

        # #1684: the no-tools branch routes through _stream_with_usage (which
        # yields chunks + a terminal usage-bearing LLMResponse) so streamed
        # turns are metered. get_streaming_response keeps its text-only contract
        # by filtering that terminal out, but the tool-detection entry point
        # forwards it.
        async def mock_stream_with_usage(**kwargs):
            yield "Hello"
            yield " world"
            yield LLMResponse(content="Hello world", tool_calls=None,
                              input_tokens=3, output_tokens=2, total_tokens=5)

        with patch.object(adapter, '_stream_with_usage', return_value=mock_stream_with_usage()):
            mock_client = MagicMock()
            results = []
            async for item in adapter.get_streaming_response_with_tools(
                client=mock_client,
                model="llama3.2:3b",
                messages=[{"role": "user", "content": "Hi"}],
                tools=None  # No tools
            ):
                results.append(item)

            text = [r for r in results if isinstance(r, str)]
            finals = [r for r in results if isinstance(r, LLMResponse)]
            assert "".join(text) == "Hello world"
            assert len(finals) == 1
            assert finals[0].input_tokens == 3


# =============================================================================
# Vertex AI Adapter Tests (Unit - Mocked Fallback)
# =============================================================================

class TestVertexStreamingToolDetectionUnit:
    """Unit tests for Vertex AI adapter streaming with tool detection (fallback)."""

    @pytest.mark.asyncio
    async def test_fallback_with_tool_calls(self):
        """Test Vertex AI fallback: non-streaming when tools are detected."""
        adapter = VertexAIAdapter()

        mock_response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="vertex_call_0", name="search_web", arguments={"query": "AI news"})
            ],
            input_tokens=12,
            output_tokens=8,
            total_tokens=20
        )

        with patch.object(adapter, 'get_response', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            results = []
            async for item in adapter.get_streaming_response_with_tools(
                client=None,
                model="gemini-2.0-flash-001",
                messages=[{"role": "user", "parts": [{"text": "Search for AI news"}]}],
                tools=SAMPLE_TOOLS
            ):
                results.append(item)

            assert len(results) == 1
            assert isinstance(results[0], LLMResponse)
            assert results[0].has_tool_calls
            assert results[0].tool_calls[0].name == "search_web"


# =============================================================================
# Integration Tests (Real APIs)
# =============================================================================

class TestOpenAIStreamingToolDetectionIntegration:
    """Integration tests with real OpenAI API."""

    @pytest.mark.asyncio
    async def test_real_openai_text_streaming(self):
        """Test real OpenAI streaming without tools."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        import openai
        adapter = OpenAIAdapter()
        client = openai.AsyncOpenAI()

        messages = adapter.create_messages(
            user_prompt="Say 'hello' and nothing else.",
            system_prompt="Be brief."
        )

        text_chunks = []
        async for item in adapter.get_streaming_response_with_tools(
            client=client,
            model="gpt-5-mini",
            messages=messages,
            tools=None
        ):
            if isinstance(item, str):
                text_chunks.append(item)

        full_response = "".join(text_chunks)
        assert len(full_response) > 0
        assert "hello" in full_response.lower()

    @pytest.mark.asyncio
    async def test_real_openai_tool_detection(self):
        """Test real OpenAI tool detection from stream."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        import openai
        adapter = OpenAIAdapter()
        client = openai.AsyncOpenAI()

        messages = adapter.create_messages(
            user_prompt="What's the weather in Tokyo?",
            system_prompt="You must use the get_weather tool to answer weather questions."
        )

        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=client,
            model="gpt-5-mini",
            messages=messages,
            tools=SAMPLE_TOOLS
        ):
            results.append(item)

        # Should have at least one LLMResponse with tool calls
        llm_responses = [r for r in results if isinstance(r, LLMResponse)]
        assert len(llm_responses) >= 1

        response = llm_responses[-1]
        assert response.has_tool_calls
        assert any(tc.name == "get_weather" for tc in response.tool_calls)
        # Should have captured "Tokyo" in arguments
        weather_call = next(tc for tc in response.tool_calls if tc.name == "get_weather")
        assert "tokyo" in str(weather_call.arguments).lower()


class TestAnthropicStreamingToolDetectionIntegration:
    """Integration tests with real Anthropic API."""

    @pytest.mark.asyncio
    async def test_real_anthropic_text_streaming(self):
        """Test real Anthropic streaming without tools."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        import anthropic
        adapter = AnthropicAdapter()
        client = anthropic.AsyncAnthropic()

        messages = adapter.create_messages(
            user_prompt="Say 'greetings' and nothing else."
        )

        text_chunks = []
        async for item in adapter.get_streaming_response_with_tools(
            client=client,
            model="claude-haiku-4-5-20251001",
            messages=messages,
            system_prompt="Be very brief.",
            tools=None
        ):
            if isinstance(item, str):
                text_chunks.append(item)

        full_response = "".join(text_chunks)
        assert len(full_response) > 0
        assert "greetings" in full_response.lower()

    @pytest.mark.asyncio
    async def test_real_anthropic_tool_detection(self):
        """Test real Anthropic tool detection from stream."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        import anthropic
        adapter = AnthropicAdapter()
        client = anthropic.AsyncAnthropic()

        messages = adapter.create_messages(
            user_prompt="What's the weather in London?"
        )

        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=client,
            model="claude-haiku-4-5-20251001",
            messages=messages,
            system_prompt="You must use the get_weather tool to answer weather questions.",
            tools=SAMPLE_TOOLS
        ):
            results.append(item)

        # Should have at least one LLMResponse with tool calls
        llm_responses = [r for r in results if isinstance(r, LLMResponse)]
        assert len(llm_responses) >= 1

        response = llm_responses[-1]
        assert response.has_tool_calls
        assert any(tc.name == "get_weather" for tc in response.tool_calls)


class TestOllamaStreamingToolDetectionIntegration:
    """Integration tests with real Ollama."""

    @pytest.mark.asyncio
    async def test_real_ollama_text_streaming(self):
        """Test real Ollama streaming without tools."""
        try:
            import ollama
            client = ollama.AsyncClient()
            # Check if Ollama is running and get available models
            response = await client.list()
            if hasattr(response, 'models'):
                models = response.models
            elif isinstance(response, dict):
                models = response.get('models', [])
            else:
                models = []

            if not models:
                pytest.skip("No Ollama models available")

            # Get first available model name
            first_model = models[0]
            if hasattr(first_model, 'model'):
                model_name = first_model.model
            elif hasattr(first_model, 'name'):
                model_name = first_model.name
            elif isinstance(first_model, dict):
                model_name = first_model.get('model') or first_model.get('name')
            else:
                pytest.skip("Could not determine model name")

        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

        adapter = OllamaAdapter()

        messages = adapter.create_messages(
            user_prompt="Say 'test' and nothing else.",
            system_prompt="Be extremely brief."
        )

        text_chunks = []
        async for item in adapter.get_streaming_response_with_tools(
            client=client,
            model=model_name,
            messages=messages,
            tools=None
        ):
            if isinstance(item, str):
                text_chunks.append(item)

        full_response = "".join(text_chunks)
        assert len(full_response) > 0


# =============================================================================
# Service Layer Tests
# =============================================================================

class TestLLMServiceStreamWithToolDetection:
    """Tests for LLM service stream_with_tool_detection method."""

    @pytest.mark.asyncio
    async def test_service_streaming_text_only(self):
        """Test service-level streaming without tools."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from kestrel_sovereign.llm.service import LLMService

        service = LLMService()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Say hi"}]}]

        text_chunks = []
        async for item in service.stream_with_tool_detection(
            messages=messages,
            tools=None,
            model_override="openai/gpt-5-mini"
        ):
            if isinstance(item, str):
                text_chunks.append(item)

        full_response = "".join(text_chunks)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_service_streaming_with_tools(self):
        """Test service-level streaming with tool detection."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from kestrel_sovereign.llm.service import LLMService

        service = LLMService()
        messages = [
            {"role": "system", "content": "Use tools when asked about weather."},
            {"role": "user", "content": [{"type": "text", "text": "Weather in Berlin?"}]}
        ]

        results = []
        async for item in service.stream_with_tool_detection(
            messages=messages,
            tools=SAMPLE_TOOLS,
            model_override="openai/gpt-5-mini"
        ):
            results.append(item)

        # Should have captured tool calls
        llm_responses = [r for r in results if isinstance(r, LLMResponse)]
        assert len(llm_responses) >= 1
        assert llm_responses[-1].has_tool_calls


# =============================================================================
# Edge Cases
# =============================================================================

class TestStreamingToolDetectionEdgeCases:
    """Edge case tests for streaming tool detection."""

    @pytest.mark.asyncio
    async def test_empty_tool_arguments(self):
        """Test handling tool calls with empty arguments."""
        adapter = OpenAIAdapter()

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = None
            tc = MagicMock()
            tc.index = 0
            tc.id = "call_empty"
            tc.function = MagicMock()
            tc.function.name = "get_weather"
            tc.function.arguments = ''  # Empty
            chunk.choices[0].delta.tool_calls = [tc]
            chunk.usage = None
            yield chunk

            final = MagicMock()
            final.choices = []
            final.usage = None
            yield final

        mock_response = mock_stream()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=mock_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "test"}],
            tools=SAMPLE_TOOLS
        ):
            results.append(item)

        # SDK 0.7.0+: ToolCallStarted precedes the LLMResponse.
        finals = [r for r in results if isinstance(r, LLMResponse)]
        assert len(finals) == 1
        assert finals[0].tool_calls[0].arguments == {}

    @pytest.mark.asyncio
    async def test_tool_call_spanning_many_chunks(self):
        """Test tool call arguments split across many chunks."""
        adapter = OpenAIAdapter()

        # Simulate arguments arriving character by character
        argument_json = '{"location": "San Francisco, CA", "unit": "fahrenheit"}'

        async def mock_stream():
            # First chunk with id and name
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta = MagicMock()
            chunk1.choices[0].delta.content = None
            tc1 = MagicMock()
            tc1.index = 0
            tc1.id = "call_chunked"
            tc1.function = MagicMock()
            tc1.function.name = "get_weather"
            tc1.function.arguments = ''
            chunk1.choices[0].delta.tool_calls = [tc1]
            chunk1.usage = None
            yield chunk1

            # Arguments arrive in small chunks
            for char in argument_json:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = MagicMock()
                chunk.choices[0].delta.content = None
                tc = MagicMock()
                tc.index = 0
                tc.id = None
                tc.function = MagicMock()
                tc.function.name = None
                tc.function.arguments = char
                chunk.choices[0].delta.tool_calls = [tc]
                chunk.usage = None
                yield chunk

            # Final chunk
            final = MagicMock()
            final.choices = []
            final.usage = None
            yield final

        mock_response = mock_stream()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        results = []
        async for item in adapter.get_streaming_response_with_tools(
            client=mock_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Weather in SF?"}],
            tools=SAMPLE_TOOLS
        ):
            results.append(item)

        # SDK 0.7.0+: ToolCallStarted precedes the LLMResponse, even
        # for arguments spanning many chunks.
        finals = [r for r in results if isinstance(r, LLMResponse)]
        assert len(finals) == 1
        assert finals[0].tool_calls[0].arguments == {
            "location": "San Francisco, CA",
            "unit": "fahrenheit"
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
