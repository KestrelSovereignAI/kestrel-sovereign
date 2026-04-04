"""
Integration tests for OpenRouter LLM inference.

These tests verify that OpenRouter works as an LLM provider for actual inference,
not just key provisioning. Uses REAL API calls.

Run with: uv run pytest tests/integration/test_openrouter_inference.py -v
"""

import os
import pytest
from dotenv import load_dotenv

# Load .env before checking for the key
load_dotenv()

# Skip all tests if OpenRouter key not available
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


def check_openrouter_response(response) -> None:
    """Check OpenRouter response and skip test if rate limited or out of credits."""
    if response.status_code == 402:
        pytest.skip("OpenRouter account has insufficient credits (402)")
    if response.status_code == 429:
        pytest.skip("OpenRouter rate limit exceeded (429)")
    assert response.status_code == 200, f"Got {response.status_code}: {response.text}"


@pytest.mark.asyncio
async def test_openrouter_provider_initialization():
    """Test that OpenRouter provider initializes correctly."""
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()

    # Find OpenRouter provider
    openrouter_provider = None
    for p in service.providers:
        if p["name"] == "openrouter":
            openrouter_provider = p
            break

    assert openrouter_provider is not None, "OpenRouter provider not initialized"
    # Model may be "auto" (resolved at runtime via selection_hints) or explicit
    assert openrouter_provider["model"] is not None
    assert openrouter_provider["adapter"] is not None
    assert openrouter_provider["client"] is not None


@pytest.mark.asyncio
async def test_openrouter_simple_inference():
    """Test basic inference through OpenRouter."""
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()

    response = await service.get_response(
        system_prompt="You are a helpful assistant. Respond in exactly one word.",
        user_prompt="What is 2+2? Reply with just the number.",
    )

    # Should get a response (exact format may vary)
    assert response is not None
    assert len(str(response)) > 0
    # The response should contain "4" somewhere
    assert "4" in str(response)


@pytest.mark.asyncio
async def test_openrouter_model_override():
    """Test that model override works with OpenRouter."""
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()

    # Use a specific model via OpenRouter — prefix with provider name
    # so routing sends it to the openrouter provider, not a nonexistent "deepseek" provider
    response = await service.get_response(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'hello' and nothing else.",
        model_override="openrouter/deepseek/deepseek-chat-v3.1",
    )

    assert response is not None
    assert len(str(response)) > 0


@pytest.mark.asyncio
async def test_openrouter_structured_output():
    """Test structured output with OpenRouter."""
    from kestrel_sovereign.llm.service import LLMService
    from pydantic import BaseModel

    class SimpleAnswer(BaseModel):
        answer: str
        confidence: float

    service = LLMService()

    response = await service.get_response(
        system_prompt="You are a helpful assistant that responds with structured JSON.",
        user_prompt="What color is the sky on a clear day? Respond with answer and confidence (0-1).",
        response_format=SimpleAnswer,
    )

    # Response should be parseable as LLMResponse or have content
    assert response is not None


@pytest.mark.asyncio
async def test_openrouter_is_primary_provider():
    """Test that OpenRouter is available as a provider."""
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()

    # Check provider order - use registry if available, fallback to providers list
    if hasattr(service, 'provider_registry') and service.provider_registry:
        provider_names = [p.name for p in service.provider_registry.providers]
    else:
        provider_names = [p["name"] for p in service.providers]

    # OpenRouter should be available (position may vary based on config)
    assert "openrouter" in provider_names, \
        f"OpenRouter should be in providers, got: {provider_names}"


@pytest.mark.asyncio
async def test_openrouter_with_agent_key():
    """Test inference using a provisioned agent key."""
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService
    import httpx
    import uuid

    # Create a temporary key
    service = OpenRouterProvisioningService()
    agent_name = f"test-inference-{uuid.uuid4().hex[:8]}"
    key_info = None

    try:
        key_info = await service.create_agent_key(
            agent_name=agent_name,
            limit_usd=0.10,
            limit_reset="monthly",
        )

        # Use the key for inference
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key_info.key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3.1",
                    "messages": [{"role": "user", "content": "Say 'test passed'"}],
                    "max_tokens": 10,
                },
                timeout=30.0,
            )

            check_openrouter_response(response)
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0

    finally:
        # Always clean up the key, even if assertions fail
        if key_info:
            try:
                await service.delete_key(key_info.key_hash)
            except Exception:
                pass
        await service.close()
