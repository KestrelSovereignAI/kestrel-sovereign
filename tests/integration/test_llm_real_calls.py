"""
Integration tests for LLMService.generate() with REAL API calls.

These tests call LLMService.generate() with REAL gpt-5-mini API calls via OpenAI.
NO MOCKS - these are real integration tests.

Run with: uv run pytest tests/integration/test_llm_real_calls.py -v
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Skip all tests if OpenAI key not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY"
)


@pytest.mark.asyncio
async def test_basic_generate():
    """Test basic generate() with gpt-5-mini - REAL API call."""
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()
    try:
        response = await service.generate(
            system_prompt="You are helpful",
            user_prompt="Say hello in exactly 3 words",
            model_override="gpt-5-mini"
        )

        # Assert response is non-empty string
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # Assert response length < 200 (sanity check)
        assert len(response) < 200
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_generate_with_tools_returns_llm_response():
    """Test generate() with tools returns LLMResponse - REAL API call."""
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.adapter import LLMResponse

    service = LLMService()
    try:
        # Simple tool definition
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name, e.g. Tokyo"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        response = await service.generate(
            system_prompt="Use tools when asked about weather",
            user_prompt="What is the weather in Tokyo?",
            tools=tools,
            model_override="gpt-5-mini"
        )

        # Assert returns LLMResponse (not plain string)
        assert isinstance(response, LLMResponse)

        # Assert response has tool calls OR content
        assert response.has_tool_calls or response.content
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_generate_with_messages():
    """Test generate_with_messages() with conversation context - REAL API call."""
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()
    try:
        # Build a 3-message conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "That's a nice color! Blue is very calming."},
            {"role": "user", "content": "What is my favorite color?"}
        ]

        response = await service.generate_with_messages(
            messages=messages,
            model_override="openai/gpt-5-mini"
        )

        # Assert response references prior context
        assert response is not None
        assert isinstance(response, str)
        assert "blue" in response.lower()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_generate_returns_coherent_response():
    """Test generate() returns coherent factual response - REAL API call."""
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()
    try:
        response = await service.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is the capital of France?",
            model_override="gpt-5-mini"
        )

        # Assert "Paris" appears in the response
        assert "Paris" in response
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_generate_handles_empty_system_prompt():
    """Test generate() works with empty system prompt - REAL API call."""
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()
    try:
        response = await service.generate(
            system_prompt="",
            user_prompt="Say 'hello world'",
            model_override="gpt-5-mini"
        )

        # Assert response is still non-empty
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
    finally:
        await service.close()
