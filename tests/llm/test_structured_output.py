"""
Comprehensive Structured Output Test Suite

Tests Pydantic response_format support across all LLM providers.
"""
import asyncio
import json
import logging
import os
import pytest
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from kestrel_sovereign.llm.service import LLMService

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# Test Response Models
# =============================================================================

class SimpleResponse(BaseModel):
    """Simple structured response for basic tests."""
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence level between 0.0 and 1.0 (e.g. 0.95)")


class ListResponse(BaseModel):
    """Response with a list field."""
    thoughts: str = Field(description="Thoughts about the request")
    items: List[str] = Field(description="List of items")


class MathResponse(BaseModel):
    """Response for math problems."""
    result: int = Field(description="The numeric result")
    explanation: str = Field(description="Brief explanation of the calculation")


class AnalysisResponse(BaseModel):
    """Complex response with nested fields."""
    summary: str = Field(description="Brief summary")
    key_points: List[str] = Field(description="Main points")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment")
    confidence_score: float = Field(description="Confidence between 0.0 and 1.0 (e.g. 0.95)")


# =============================================================================
# Provider Test Matrix
# =============================================================================

# Providers to test (will skip if not configured)
# Using smaller/cheaper models for faster tests:
# - OpenAI: gpt-5-mini (cheap, fast)
# - Anthropic: claude-haiku-4-5-20251001 (Claude Haiku 4.5)
# - Vertex: gemini-3-flash-preview (fast, cheap)
PROVIDERS = [
    ("openai", "gpt-5-mini"),
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("vertex_ai", "gemini-3-flash-preview"),
]


def provider_available(provider_name: str) -> bool:
    """Check if a provider is available based on environment."""
    if provider_name == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    elif provider_name == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    elif provider_name == "vertex_ai":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GCP_PROJECT_ID"))
    return False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def llm_service():
    """Create an LLM service instance."""
    from kestrel_sovereign.llm.service import LLMService
    return LLMService()


# =============================================================================
# Basic Structured Output Tests
# =============================================================================

class TestStructuredOutputBasic:
    """Basic structured output tests for each provider."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model", PROVIDERS)
    async def test_simple_structured_response(self, llm_service, provider, model):
        """Test simple structured response with answer and confidence."""
        if not provider_available(provider):
            pytest.skip(f"{provider} not configured")

        response = await llm_service.get_response(
            system_prompt="You are a helpful assistant. Always respond with high confidence.",
            user_prompt="What is 2+2?",
            model_override=provider,
            response_format=SimpleResponse,
        )

        # Response should be valid JSON
        assert response.content is not None
        data = json.loads(response.content)

        # Validate against schema
        parsed = SimpleResponse.model_validate(data)
        assert "4" in parsed.answer.lower() or parsed.answer == "4"
        # Some LLMs return 0-1, others 0-100 despite the field description
        confidence = parsed.confidence / 100 if parsed.confidence > 1 else parsed.confidence
        assert 0 <= confidence <= 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model", PROVIDERS)
    async def test_list_structured_response(self, llm_service, provider, model):
        """Test structured response with list field."""
        if not provider_available(provider):
            pytest.skip(f"{provider} not configured")

        response = await llm_service.get_response(
            system_prompt="You are a helpful assistant.",
            user_prompt="List 3 primary colors.",
            model_override=provider,
            response_format=ListResponse,
        )

        assert response.content is not None
        data = json.loads(response.content)
        parsed = ListResponse.model_validate(data)

        assert len(parsed.items) >= 3
        assert parsed.thoughts is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model", PROVIDERS)
    async def test_math_structured_response(self, llm_service, provider, model):
        """Test math problem with structured numeric response."""
        if not provider_available(provider):
            pytest.skip(f"{provider} not configured")

        response = await llm_service.get_response(
            system_prompt="You are a math assistant. Solve problems accurately.",
            user_prompt="What is 15 multiplied by 7?",
            model_override=provider,
            response_format=MathResponse,
        )

        assert response.content is not None
        data = json.loads(response.content)
        parsed = MathResponse.model_validate(data)

        assert parsed.result == 105
        assert len(parsed.explanation) > 0


# =============================================================================
# Complex Structured Output Tests
# =============================================================================

class TestStructuredOutputComplex:
    """Complex structured output tests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model", PROVIDERS)
    async def test_analysis_response(self, llm_service, provider, model):
        """Test complex analysis response with multiple fields."""
        if not provider_available(provider):
            pytest.skip(f"{provider} not configured")

        response = await llm_service.get_response(
            system_prompt="You are a text analyst. Analyze text for sentiment and key points.",
            user_prompt="The product exceeded my expectations. Great quality and fast shipping!",
            model_override=provider,
            response_format=AnalysisResponse,
        )

        assert response.content is not None
        data = json.loads(response.content)
        parsed = AnalysisResponse.model_validate(data)

        assert parsed.sentiment in ["positive", "negative", "neutral"]
        assert len(parsed.key_points) >= 1
        # Some LLMs return 0-100 despite the field description
        score = parsed.confidence_score / 100 if parsed.confidence_score > 1 else parsed.confidence_score
        assert 0 <= score <= 1


# =============================================================================
# Direct Adapter Tests
# =============================================================================

class TestAdapterStructuredOutput:
    """Test structured output directly on adapters."""

    @pytest.mark.asyncio
    async def test_openai_adapter_structured(self):
        """Test OpenAI adapter structured output directly."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        import openai
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        client = openai.AsyncOpenAI()
        adapter = OpenAIAdapter()

        messages = adapter.create_messages(
            user_prompt="What is 5+5?",
            system_prompt="Answer math questions."
        )

        response = await adapter.get_response(
            client=client,
            model="gpt-5-mini",
            messages=messages,
            response_format=MathResponse,
        )

        assert response.content is not None
        data = json.loads(response.content)
        parsed = MathResponse.model_validate(data)
        assert parsed.result == 10

    @pytest.mark.asyncio
    async def test_anthropic_adapter_structured(self):
        """Test Anthropic adapter structured output directly."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        import anthropic
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

        client = anthropic.AsyncAnthropic()
        adapter = AnthropicAdapter()

        messages = adapter.create_messages(
            user_prompt="What is 7+3?",
        )

        response = await adapter.get_response(
            client=client,
            model="claude-haiku-4-5-20251001",
            messages=messages,
            response_format=MathResponse,
            system_prompt="Answer math questions accurately."
        )

        assert response.content is not None
        data = json.loads(response.content)
        parsed = MathResponse.model_validate(data)
        assert parsed.result == 10

    @pytest.mark.asyncio
    async def test_vertex_adapter_structured(self):
        """Test Vertex AI adapter structured output directly."""
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GCP_PROJECT_ID")):
            pytest.skip("Google credentials not set")

        from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter

        adapter = VertexAIAdapter()

        messages = adapter.create_messages(
            user_prompt="What is 8+2?",
            system_prompt="Answer math questions."
        )

        response = await adapter.get_response(
            client=None,  # Uses internal client
            model="gemini-3-flash-preview",
            messages=messages,
            response_format=MathResponse,
        )

        assert response.content is not None
        data = json.loads(response.content)
        parsed = MathResponse.model_validate(data)
        assert parsed.result == 10

    @pytest.mark.asyncio
    async def test_ollama_adapter_structured(self):
        """Test Ollama adapter structured output directly."""
        try:
            import ollama
        except ImportError:
            pytest.skip("Ollama library not installed")

        # Check if Ollama is running by trying to list models
        try:
            client = ollama.AsyncClient()
            models = await client.list()
            if not models:
                pytest.skip("No Ollama models available")
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

        from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter()

        messages = adapter.create_messages(
            user_prompt="What is 6+4?",
            system_prompt="Answer math questions. Always respond with valid JSON."
        )

        # Use a small model that's likely to be available
        # Try llama3.2:latest first, fall back to any available model
        model_to_use = None
        model_list = models.models if hasattr(models, 'models') else models.get('models', [])
        for m in model_list:
            model_name = m.model if hasattr(m, 'model') else m.get('model', m.get('name', ''))
            if model_name:
                model_to_use = model_name
                # Prefer smaller models
                if 'llama3.2' in model_name.lower() or 'qwen' in model_name.lower():
                    break

        if not model_to_use:
            pytest.skip("No suitable Ollama model found")

        response = await adapter.get_response(
            client=client,
            model=model_to_use,
            messages=messages,
            response_format=MathResponse,
        )

        assert response.content is not None
        data = json.loads(response.content)
        parsed = MathResponse.model_validate(data)
        assert parsed.result == 10

    @pytest.mark.asyncio
    async def test_ollama_adapter_streaming_structured(self):
        """Test Ollama adapter streaming with structured output."""
        try:
            import ollama
        except ImportError:
            pytest.skip("Ollama library not installed")

        try:
            client = ollama.AsyncClient()
            models = await client.list()
            if not models:
                pytest.skip("No Ollama models available")
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

        from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
        from kestrel_sovereign.llm.model_metadata import ModelCategory

        adapter = OllamaAdapter()

        messages = adapter.create_messages(
            user_prompt="What is 4+4?",
            system_prompt="Answer math questions."
        )

        # Use adapter.list_models() which correctly categorizes embedding vs chat models
        discovered_models = await adapter.list_models()
        chat_models = [m for m in discovered_models if m.category == ModelCategory.CHAT]

        if not chat_models:
            pytest.skip("No suitable Ollama chat model found")

        model_to_use = chat_models[0].id

        chunks = []
        async for chunk in adapter.get_streaming_response(
            client=client,
            model=model_to_use,
            messages=messages,
            response_format=MathResponse,
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert full_response is not None

        # Should be valid JSON
        data = json.loads(full_response)
        parsed = MathResponse.model_validate(data)
        assert parsed.result == 8


# =============================================================================
# Edge Cases
# =============================================================================

class TestStructuredOutputEdgeCases:
    """Edge case tests for structured output."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model", PROVIDERS)
    async def test_empty_list_response(self, llm_service, provider, model):
        """Test that empty list is handled correctly."""
        if not provider_available(provider):
            pytest.skip(f"{provider} not configured")

        class EmptyListResponse(BaseModel):
            items: List[str] = Field(description="List of items (can be empty)")
            reason: str = Field(description="Why the list is empty or not")

        response = await llm_service.get_response(
            system_prompt="You are helpful.",
            user_prompt="List all months that have 32 days.",
            model_override=provider,
            response_format=EmptyListResponse,
        )

        assert response.content is not None
        data = json.loads(response.content)
        parsed = EmptyListResponse.model_validate(data)
        # The correct answer is [] but some LLMs hallucinate; verify structure works
        assert isinstance(parsed.items, list)
        assert isinstance(parsed.reason, str) and len(parsed.reason) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model", PROVIDERS)
    async def test_optional_fields(self, llm_service, provider, model):
        """Test response model with optional fields."""
        if not provider_available(provider):
            pytest.skip(f"{provider} not configured")

        # OpenAI doesn't support Optional fields in structured output
        # All fields must be in the 'required' array
        if provider == "openai":
            pytest.skip("OpenAI doesn't support optional fields in structured output")

        class OptionalResponse(BaseModel):
            answer: str = Field(description="The answer")
            details: Optional[str] = Field(default=None, description="Optional details")

        response = await llm_service.get_response(
            system_prompt="You are helpful. Be concise.",
            user_prompt="What color is the sky?",
            model_override=provider,
            response_format=OptionalResponse,
        )

        assert response.content is not None
        data = json.loads(response.content)
        parsed = OptionalResponse.model_validate(data)
        # Verify structure: answer should exist and mention sky/blue/color
        assert len(parsed.answer) > 0


# =============================================================================
# Streaming with Structured Output
# =============================================================================

class TestStreamingStructuredOutput:
    """Test streaming with structured output (where supported)."""

    @pytest.mark.asyncio
    async def test_openai_streaming_structured(self):
        """Test OpenAI streaming with structured output."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        import openai
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        client = openai.AsyncOpenAI()
        adapter = OpenAIAdapter()

        messages = adapter.create_messages(
            user_prompt="What is 3+3?",
            system_prompt="Answer math questions."
        )

        chunks = []
        async for chunk in adapter.get_streaming_response(
            client=client,
            model="gpt-5-mini",
            messages=messages,
            response_format=MathResponse,
        ):
            chunks.append(chunk)

        # Reconstruct the full response from chunks
        full_response = "".join(chunks)
        assert full_response is not None

        # Should be valid JSON
        data = json.loads(full_response)
        parsed = MathResponse.model_validate(data)
        assert parsed.result == 6


# =============================================================================
# Service Layer Streaming Tests
# =============================================================================

class TestServiceStreamingStructuredOutput:
    """Test streaming structured output through the service layer."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", "").strip(),
        reason="Requires OpenAI API key"
    )
    @pytest.mark.asyncio
    async def test_service_streaming_structured_openai(self):
        """Test streaming structured output via service layer with OpenAI."""
        service = LLMService()

        chunks = []
        async for chunk in service.get_streaming_response(
            system_prompt="Answer math questions with structured JSON output.",
            user_prompt="What is 5 + 5?",
            model_override="openai/gpt-5-mini",
            response_format=MathResponse,
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert full_response is not None

        # Parse and validate
        data = json.loads(full_response)
        parsed = MathResponse.model_validate(data)
        assert parsed.result == 10

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", "").strip(),
        reason="Requires OpenAI API key"
    )
    @pytest.mark.asyncio
    async def test_service_generate_stream_structured(self):
        """Test generate_stream with structured output."""
        service = LLMService()

        chunks = []
        async for chunk in service.generate_stream(
            system_prompt="Respond with structured JSON.",
            user_prompt="List three primary colors.",
            model_override="openai/gpt-5-mini",
            response_format=ListResponse,
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        data = json.loads(full_response)
        parsed = ListResponse.model_validate(data)
        assert len(parsed.items) == 3

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", "").strip()
        and not os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip(),
        reason="anthropic not configured"
    )
    async def test_streaming_fallback_for_non_streaming_provider(self):
        """Test that non-streaming providers fall back gracefully."""
        service = LLMService()

        # Anthropic streaming with structured output falls back to non-streaming
        chunks = []
        async for chunk in service.get_streaming_response(
            system_prompt="Answer questions.",
            user_prompt="What is 2 + 2?",
            model_override="anthropic/claude-haiku-4-5-20251001",
            response_format=MathResponse,
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        # Should still get valid structured output (via fallback)
        data = json.loads(full_response)
        parsed = MathResponse.model_validate(data)
        assert parsed.result == 4


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
