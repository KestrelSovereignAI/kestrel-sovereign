"""
Token Tracking Tests

Verifies that LLMResponse.input_tokens, output_tokens, and total_tokens
are properly extracted from all provider responses.

This is critical for billing/metering - the vending machine system needs
real token counts to pass through costs.
"""
import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter


# =============================================================================
# Unit Tests (Mocked) - Verify Token Extraction Logic
# =============================================================================

class TestOpenAITokenExtraction:
    """Test OpenAI adapter extracts tokens from response.usage."""

    @pytest.mark.asyncio
    async def test_extracts_usage_from_response(self):
        """Test that OpenAI adapter extracts prompt_tokens and completion_tokens."""
        adapter = OpenAIAdapter()

        # Mock response with usage data
        mock_message = MagicMock()
        mock_message.content = "Hello world"
        mock_message.tool_calls = None

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await adapter.get_response(
            client=mock_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.total_tokens == 15

    @pytest.mark.asyncio
    async def test_handles_missing_usage(self):
        """Test graceful handling when usage is not present."""
        adapter = OpenAIAdapter()

        mock_message = MagicMock()
        mock_message.content = "Hello"
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await adapter.get_response(
            client=mock_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Hi"}]
        )

        # Should not crash, tokens should be None
        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens is None


class TestAnthropicTokenExtraction:
    """Test Anthropic adapter extracts tokens from response.usage."""

    @pytest.mark.asyncio
    async def test_extracts_usage_from_response(self):
        """Test that Anthropic adapter extracts input_tokens and output_tokens."""
        adapter = AnthropicAdapter()

        # Mock response with usage data
        mock_usage = MagicMock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 8

        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "Hello world"

        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.usage = mock_usage
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        response = await adapter.get_response(
            client=mock_client,
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="Be helpful"
        )

        assert response.input_tokens == 15
        assert response.output_tokens == 8
        assert response.total_tokens == 23  # input + output


class TestOllamaTokenExtraction:
    """Test Ollama adapter extracts tokens from response metrics."""

    @pytest.mark.asyncio
    async def test_extracts_eval_counts(self):
        """Test that Ollama adapter extracts prompt_eval_count and eval_count."""
        adapter = OllamaAdapter()

        # Mock response with Ollama-style metrics
        mock_response = {
            "message": {"role": "assistant", "content": "Hello!"},
            "prompt_eval_count": 12,
            "eval_count": 6,
            "done": True
        }

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=mock_response)

        response = await adapter.get_response(
            client=mock_client,
            model="llama3.2:3b",
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert response.input_tokens == 12
        assert response.output_tokens == 6
        assert response.total_tokens == 18


# =============================================================================
# Integration Tests (Real APIs) - Verify End-to-End Token Tracking
# =============================================================================

class TestRealOpenAITokenTracking:
    """Integration tests with real OpenAI API."""

    @pytest.mark.asyncio
    async def test_real_openai_returns_tokens(self):
        """Test that real OpenAI API returns token counts."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        import openai
        adapter = OpenAIAdapter()
        client = openai.AsyncOpenAI()

        messages = adapter.create_messages(
            user_prompt="What is 2+2? Answer briefly.",
            system_prompt="You are helpful."
        )

        response = await adapter.get_response(
            client=client,
            model="gpt-5-mini",
            messages=messages
        )

        # Real API should return token counts
        assert response.input_tokens is not None
        assert response.input_tokens > 0
        assert response.output_tokens is not None
        assert response.output_tokens > 0
        assert response.total_tokens == response.input_tokens + response.output_tokens


class TestRealAnthropicTokenTracking:
    """Integration tests with real Anthropic API."""

    @pytest.mark.asyncio
    async def test_real_anthropic_returns_tokens(self):
        """Test that real Anthropic API returns token counts."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        import anthropic
        adapter = AnthropicAdapter()
        client = anthropic.AsyncAnthropic()

        messages = adapter.create_messages(
            user_prompt="What is 3+3? Answer briefly.",
        )

        response = await adapter.get_response(
            client=client,
            model="claude-haiku-4-5-20251001",
            messages=messages,
            system_prompt="You are helpful."
        )

        # Real API should return token counts
        assert response.input_tokens is not None
        assert response.input_tokens > 0
        assert response.output_tokens is not None
        assert response.output_tokens > 0
        assert response.total_tokens == response.input_tokens + response.output_tokens


class TestRealOllamaTokenTracking:
    """Integration tests with real Ollama server."""

    @pytest.mark.asyncio
    async def test_real_ollama_returns_tokens(self):
        """Test that real Ollama returns token counts."""
        try:
            import ollama
            client = ollama.AsyncClient()
            # Check if Ollama is running
            await client.list()
        except Exception:
            pytest.skip("Ollama not available")

        from kestrel_sovereign.llm.model_metadata import ModelCategory

        adapter = OllamaAdapter()

        # Get a chat model (not embedding)
        models = await adapter.list_models()
        chat_models = [m for m in models if m.category == ModelCategory.CHAT]
        if not chat_models:
            pytest.skip("No Ollama chat models available")

        model_to_use = chat_models[0].id

        messages = adapter.create_messages(
            user_prompt="What is 4+4? Answer briefly.",
            system_prompt="Be concise."
        )

        response = await adapter.get_response(
            client=client,
            model=model_to_use,
            messages=messages
        )

        # Ollama should return token counts
        assert response.input_tokens is not None
        assert response.input_tokens > 0
        assert response.output_tokens is not None
        assert response.output_tokens > 0


# =============================================================================
# Service Layer Token Tracking
# =============================================================================

class TestLLMServiceTokenTracking:
    """Test token tracking through the LLM service layer."""

    @pytest.mark.asyncio
    async def test_service_passes_tokens_through_with_structured_output(self):
        """Test that LLM service preserves token counts when using structured output.

        Note: service.generate() returns str by default, but returns LLMResponse
        when tools or response_format is provided.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from pydantic import BaseModel, Field
        from kestrel_sovereign.llm.service import LLMService

        class SimpleAnswer(BaseModel):
            answer: int = Field(description="The numeric answer")

        service = LLMService()

        response = await service.generate(
            user_prompt="What is 5+5?",
            system_prompt="Answer with just the number.",
            model_override="gpt-5-mini",
            response_format=SimpleAnswer  # This makes it return LLMResponse
        )

        # With structured output, service returns LLMResponse with token counts
        assert hasattr(response, 'input_tokens'), "Response should be LLMResponse"
        assert response.input_tokens is not None
        assert response.input_tokens > 0
        assert response.output_tokens is not None
        assert response.output_tokens > 0

    @pytest.mark.asyncio
    async def test_service_passes_tokens_through_with_tools(self):
        """Test that LLM service preserves token counts when using tools."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from kestrel_sovereign.llm.service import LLMService

        service = LLMService()

        # Simple tool definition
        tools = [{
            "type": "function",
            "function": {
                "name": "get_answer",
                "description": "Get the answer to a math problem",
                "parameters": {
                    "type": "object",
                    "properties": {"result": {"type": "integer"}},
                    "required": ["result"]
                }
            }
        }]

        response = await service.generate(
            user_prompt="What is 5+5? Use the tool.",
            system_prompt="Use the get_answer tool.",
            model_override="gpt-5-mini",
            tools=tools  # This makes it return LLMResponse
        )

        # With tools, service returns LLMResponse with token counts
        assert hasattr(response, 'input_tokens'), "Response should be LLMResponse"
        assert response.input_tokens is not None
        assert response.input_tokens > 0
        assert response.output_tokens is not None
        assert response.output_tokens > 0


class TestUsageTrackingIntegration:
    """Test that _track_model_usage is called with token counts."""

    @pytest.mark.asyncio
    async def test_track_model_usage_receives_tokens(self):
        """Verify _track_model_usage is called with non-zero token count."""
        from kestrel_sovereign.llm.service import LLMService
        from kestrel_sovereign.llm.adapter import LLMResponse

        # Create service and mock the tracking method
        service = LLMService()
        tracked_calls = []
        original_track = service._track_model_usage

        async def mock_track(model_id, provider, tokens=0):
            tracked_calls.append({"model": model_id, "provider": provider, "tokens": tokens})
            return await original_track(model_id, provider, tokens)

        service._track_model_usage = mock_track

        # Mock a provider to avoid real API calls
        mock_response = LLMResponse(
            content="Hello",
            input_tokens=25,
            output_tokens=10,
        )

        mock_adapter = MagicMock()
        mock_adapter.create_messages = MagicMock(return_value=[{"role": "user", "content": "Hi"}])
        mock_adapter.get_response = AsyncMock(return_value=mock_response)

        # Replace first provider with our mock
        if service.providers:
            service.providers[0]["adapter"] = mock_adapter

            response = await service.get_response(
                system_prompt="Be helpful",
                user_prompt="Hello"
            )

            # Verify _track_model_usage was called with tokens
            assert len(tracked_calls) == 1, f"Expected 1 tracking call, got {len(tracked_calls)}"
            assert tracked_calls[0]["tokens"] == 35, f"Expected 35 tokens (25+10), got {tracked_calls[0]['tokens']}"
        else:
            pytest.skip("No providers available")

    @pytest.mark.asyncio
    async def test_track_model_usage_with_model_override(self):
        """Verify get_response_with_model also tracks tokens."""
        from kestrel_sovereign.llm.service import LLMService
        from kestrel_sovereign.llm.adapter import LLMResponse

        service = LLMService()
        tracked_calls = []

        async def mock_track(model_id, provider, tokens=0):
            tracked_calls.append({"model": model_id, "provider": provider, "tokens": tokens})

        service._track_model_usage = mock_track

        # Mock response with token counts
        mock_response = LLMResponse(
            content="World",
            input_tokens=15,
            output_tokens=5,
        )

        mock_adapter = MagicMock()
        mock_adapter.create_messages = MagicMock(return_value=[{"role": "user", "content": "Hi"}])
        mock_adapter.get_response = AsyncMock(return_value=mock_response)

        # Find any provider and replace adapter
        if service.providers:
            test_model = service.providers[0]["model"]
            service.providers[0]["adapter"] = mock_adapter

            await service.get_response_with_model(
                model_id=test_model,
                system_prompt="Be helpful",
                user_prompt="Hello"
            )

            assert len(tracked_calls) == 1
            assert tracked_calls[0]["tokens"] == 20  # 15 + 5
        else:
            pytest.skip("No providers available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
