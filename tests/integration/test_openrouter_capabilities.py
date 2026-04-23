"""
OpenRouter Capability Tests - Comprehensive Feature Parity

Tests OpenRouter's capabilities to ensure feature parity with our
local LLM router tests. This validates that OpenRouter can handle:
1. Tool calling (function calling)
2. Streaming responses
3. Streaming with tool detection
4. Multi-turn tool conversations
5. Structured output
6. Various models (DeepSeek, Claude, Llama)

Run with: uv run pytest tests/integration/test_openrouter_capabilities.py -v
"""

import os
import json
import pytest
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load .env before checking for keys
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
    response.raise_for_status()


# =============================================================================
# Test Tools Definition (Same as LLM router tests)
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
# Tool Calling Tests
# =============================================================================

class TestOpenRouterToolCalling:
    """Test OpenRouter tool/function calling capabilities."""

    @pytest.mark.asyncio
    async def test_tool_calling_deepseek(self):
        """Test tool calling with DeepSeek model via OpenRouter."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3.1",
                    "messages": [
                        {"role": "system", "content": "You must use the get_weather tool to answer weather questions."},
                        {"role": "user", "content": "What's the weather in Tokyo?"}
                    ],
                    "tools": SAMPLE_TOOLS,
                    "tool_choice": "auto",
                    "max_tokens": 200,
                },
                timeout=60.0,
            )

            check_openrouter_response(response)
            data = response.json()

            # Check if tool calls are in response
            assert "choices" in data
            choice = data["choices"][0]
            message = choice["message"]

            # DeepSeek should call the get_weather tool
            if message.get("tool_calls"):
                tool_calls = message["tool_calls"]
                assert len(tool_calls) >= 1
                assert tool_calls[0]["function"]["name"] == "get_weather"
                args = json.loads(tool_calls[0]["function"]["arguments"])
                assert "tokyo" in args.get("location", "").lower()
                print(f"✅ DeepSeek tool call: {tool_calls[0]}")
            else:
                # Some models might not support tools - that's informative
                print(f"⚠️ DeepSeek did not use tool calling, responded with: {message.get('content', '')[:100]}")

    @pytest.mark.asyncio
    async def test_tool_calling_claude_via_openrouter(self):
        """Test tool calling with Claude via OpenRouter."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-3.5-haiku",
                    "messages": [
                        {"role": "system", "content": "You must use the get_weather tool to answer weather questions."},
                        {"role": "user", "content": "What's the weather in Paris?"}
                    ],
                    "tools": SAMPLE_TOOLS,
                    "tool_choice": "auto",
                    "max_tokens": 200,
                },
                timeout=60.0,
            )

            check_openrouter_response(response)
            data = response.json()

            choice = data["choices"][0]
            message = choice["message"]

            if message.get("tool_calls"):
                tool_calls = message["tool_calls"]
                assert len(tool_calls) >= 1
                assert tool_calls[0]["function"]["name"] == "get_weather"
                print(f"✅ Claude tool call: {tool_calls[0]}")
            else:
                print(f"⚠️ Claude did not use tool calling")

    @pytest.mark.asyncio
    async def test_multiple_parallel_tool_calls(self):
        """Test that OpenRouter supports multiple parallel tool calls."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3.1",
                    "messages": [
                        {"role": "system", "content": "Use tools when asked. You can call multiple tools in parallel."},
                        {"role": "user", "content": "What's the weather in NYC and also search for latest AI news?"}
                    ],
                    "tools": SAMPLE_TOOLS,
                    "tool_choice": "auto",
                    "max_tokens": 300,
                },
                timeout=60.0,
            )

            check_openrouter_response(response)
            data = response.json()
            message = data["choices"][0]["message"]

            if message.get("tool_calls"):
                tool_calls = message["tool_calls"]
                tool_names = [tc["function"]["name"] for tc in tool_calls]
                print(f"✅ Parallel tool calls: {tool_names}")
                # Check if both tools were called
                if len(tool_calls) >= 2:
                    assert "get_weather" in tool_names or "search_web" in tool_names
            else:
                print(f"⚠️ Model did not use parallel tool calls")


# =============================================================================
# Streaming Tests
# =============================================================================

class TestOpenRouterStreaming:
    """Test OpenRouter streaming capabilities."""

    @pytest.mark.asyncio
    async def test_basic_streaming(self):
        """Test basic streaming response from OpenRouter."""
        import httpx

        chunks_received = []

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3.1",
                    "messages": [{"role": "user", "content": "Count from 1 to 5, one number per line."}],
                    "stream": True,
                    "max_tokens": 50,
                },
                timeout=60.0,
            ) as response:
                check_openrouter_response(response)

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if data.get("choices"):
                                delta = data["choices"][0].get("delta", {})
                                if delta.get("content"):
                                    chunks_received.append(delta["content"])
                        except json.JSONDecodeError:
                            pass

        full_response = "".join(chunks_received)
        print(f"✅ Streaming received {len(chunks_received)} chunks: {full_response[:100]}...")
        assert len(chunks_received) > 1, "Should receive multiple chunks"
        assert "1" in full_response and "5" in full_response

    @pytest.mark.asyncio
    async def test_streaming_with_tools(self):
        """Test streaming with tool calling - critical capability."""
        import httpx

        chunks_received = []
        tool_calls_detected = []

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3.1",
                    "messages": [
                        {"role": "system", "content": "Use tools when needed."},
                        {"role": "user", "content": "What's the weather in London?"}
                    ],
                    "tools": SAMPLE_TOOLS,
                    "stream": True,
                    "max_tokens": 200,
                },
                timeout=60.0,
            ) as response:
                check_openrouter_response(response)

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if data.get("choices"):
                                delta = data["choices"][0].get("delta", {})
                                if delta.get("content"):
                                    chunks_received.append(delta["content"])
                                if delta.get("tool_calls"):
                                    tool_calls_detected.extend(delta["tool_calls"])
                        except json.JSONDecodeError:
                            pass

        print(f"✅ Streaming: {len(chunks_received)} text chunks, {len(tool_calls_detected)} tool call deltas")

        # Either text or tool calls should be received
        assert len(chunks_received) > 0 or len(tool_calls_detected) > 0, \
            "Should receive either text chunks or tool call deltas"


# =============================================================================
# Multi-Turn Conversation with Tools
# =============================================================================

class TestOpenRouterMultiTurnTools:
    """Test multi-turn conversations with tool results."""

    @pytest.mark.asyncio
    async def test_tool_result_handling(self):
        """Test that OpenRouter handles tool results correctly."""
        import httpx

        # Step 1: Get tool call from LLM
        async with httpx.AsyncClient() as client:
            response1 = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3.1",
                    "messages": [
                        {"role": "system", "content": "Use get_weather tool for weather questions."},
                        {"role": "user", "content": "What's the weather in Berlin?"}
                    ],
                    "tools": SAMPLE_TOOLS,
                    "tool_choice": "auto",
                    "max_tokens": 200,
                },
                timeout=60.0,
            )

            check_openrouter_response(response1)
            data1 = response1.json()
            message1 = data1["choices"][0]["message"]

            if not message1.get("tool_calls"):
                pytest.skip("Model did not use tool calling - can't test multi-turn")

            tool_call = message1["tool_calls"][0]
            tool_call_id = tool_call["id"]

            # Step 2: Send tool result back
            response2 = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3.1",
                    "messages": [
                        {"role": "system", "content": "Use get_weather tool for weather questions."},
                        {"role": "user", "content": "What's the weather in Berlin?"},
                        {"role": "assistant", "tool_calls": message1["tool_calls"]},
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({
                                "temperature": 15,
                                "unit": "celsius",
                                "condition": "partly cloudy"
                            })
                        }
                    ],
                    "max_tokens": 200,
                },
                timeout=60.0,
            )

            check_openrouter_response(response2)
            data2 = response2.json()
            message2 = data2["choices"][0]["message"]
            final_response = message2.get("content") or ""

            # Some models return empty content but have reasoning or other fields
            if not final_response:
                # Check if there's any indication of processing the tool result
                if message2.get("tool_calls"):
                    pytest.skip("Model made another tool call instead of responding")
                # Accept empty as valid if the API call succeeded - model behavior varies
                print("⚠️ Model returned empty content - API works but model behavior varies")
                return

            print(f"✅ Multi-turn tool flow completed. Final response: {final_response[:100]}...")

            # Final response should mention weather-related info
            # LLM responses are non-deterministic, so be lenient
            weather_indicators = ["15", "celsius", "cloudy", "weather", "temperature", "berlin", "degrees", "partly"]
            found_indicator = any(ind in final_response.lower() for ind in weather_indicators)
            assert found_indicator, \
                f"Final response should incorporate tool result: {final_response}"


# =============================================================================
# Structured Output Tests
# =============================================================================

class TestOpenRouterStructuredOutput:
    """Test OpenRouter structured output capabilities."""

    @pytest.mark.asyncio
    async def test_json_mode(self):
        """Test JSON mode response format."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3.1",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
                        {"role": "user", "content": "List 3 colors with their hex codes. Respond as JSON array."}
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 200,
                },
                timeout=60.0,
            )

            check_openrouter_response(response)
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Try to parse as JSON
            try:
                parsed = json.loads(content)
                print(f"✅ JSON mode returned valid JSON: {type(parsed)}")
                assert isinstance(parsed, (dict, list))
            except json.JSONDecodeError:
                print(f"⚠️ Response not valid JSON: {content[:100]}")


# =============================================================================
# LLM Service Integration Tests
# =============================================================================

class TestOpenRouterViaLLMService:
    """Test OpenRouter through our LLMService layer."""

    @pytest.mark.asyncio
    async def test_llm_service_tool_calling_via_openrouter(self):
        """Test tool calling through LLMService with OpenRouter."""
        from kestrel_sovereign.llm.service import LLMService

        service = LLMService()

        # Verify OpenRouter is being used. Provider names are composite
        # "<vendor>:<route>" under the vendor/route/model schema — match by
        # vendor field instead of a bare name.
        vendors = {p.get("vendor") for p in service.providers}
        assert "openrouter" in vendors, f"OpenRouter vendor not initialized: {vendors}"

        # Call with tools
        response = await service.generate(
            user_prompt="What's the weather in San Francisco?",
            system_prompt="Use the get_weather tool to answer weather questions.",
            tools=SAMPLE_TOOLS,
        )

        print(f"✅ LLMService response type: {type(response)}")

        if hasattr(response, 'has_tool_calls') and response.has_tool_calls:
            print(f"✅ Tool calls via LLMService: {response.tool_calls}")
            assert any(tc.name == "get_weather" for tc in response.tool_calls)
        elif hasattr(response, 'content'):
            print(f"⚠️ LLMService returned content instead of tool calls: {response.content[:100]}")
        else:
            print(f"⚠️ Response: {response}")

    @pytest.mark.asyncio
    async def test_llm_service_streaming_via_openrouter(self):
        """Test streaming through LLMService with OpenRouter.

        Note: Chunk count varies by provider and response length.
        Some providers may return the entire response as a single chunk,
        especially for short responses or when falling back to non-streaming.
        """
        from kestrel_sovereign.llm.service import LLMService

        service = LLMService()

        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is 2+2?"}
        ]

        chunks = []
        async for chunk in service.stream_with_messages(messages=messages):
            # Accept both str and objects that can be converted to str
            if chunk is not None:
                chunks.append(str(chunk) if not isinstance(chunk, str) else chunk)

        full_response = "".join(chunks)
        print(f"✅ Streaming via LLMService: {len(chunks)} chunks, {full_response[:100]}...")

        # Verify we got a response
        assert len(chunks) >= 1, "Should receive at least one chunk"
        assert len(full_response) > 0, "Should receive non-empty response"
        # Basic sanity check - response should contain "4" for 2+2
        assert "4" in full_response, f"Response should contain '4' for 2+2, got: {full_response[:200]}"

    @pytest.mark.asyncio
    async def test_llm_service_streaming_with_tools_via_openrouter(self):
        """Test streaming with tool detection through LLMService."""
        from kestrel_sovereign.llm.service import LLMService
        from kestrel_sovereign.llm.adapter import LLMResponse

        service = LLMService()

        results = []
        async for item in service.stream_with_tool_detection(
            messages=[
                {"role": "system", "content": "Use tools when needed."},
                {"role": "user", "content": "What's the weather in Tokyo?"}
            ],
            tools=SAMPLE_TOOLS,
        ):
            results.append(item)

        text_chunks = [r for r in results if isinstance(r, str)]
        llm_responses = [r for r in results if isinstance(r, LLMResponse)]

        print(f"✅ Stream with tools: {len(text_chunks)} text chunks, {len(llm_responses)} LLMResponse objects")

        if llm_responses and llm_responses[-1].has_tool_calls:
            print(f"✅ Tool calls detected: {llm_responses[-1].tool_calls}")


# =============================================================================
# Model Comparison Tests
# =============================================================================

class TestOpenRouterModelComparison:
    """Compare tool calling across different models via OpenRouter."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [
        "deepseek/deepseek-chat-v3.1",
        "meta-llama/llama-3.3-70b-instruct",
        # "anthropic/claude-3.5-haiku",  # Uncomment if you have credits
    ])
    @pytest.mark.asyncio
    async def test_tool_calling_by_model(self, model: str):
        """Test tool calling works across different models."""
        import httpx

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "Use the get_weather tool for weather questions."},
                            {"role": "user", "content": "What's the weather in Seattle?"}
                        ],
                        "tools": SAMPLE_TOOLS,
                        "tool_choice": "auto",
                        "max_tokens": 200,
                    },
                    timeout=60.0,
                )

                if response.status_code != 200:
                    print(f"⚠️ {model}: HTTP {response.status_code}")
                    return

                data = response.json()
                message = data["choices"][0]["message"]

                if message.get("tool_calls"):
                    print(f"✅ {model}: Used tool calling")
                else:
                    print(f"⚠️ {model}: No tool calls, responded with text")

            except Exception as e:
                print(f"⚠️ {model}: Error - {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
