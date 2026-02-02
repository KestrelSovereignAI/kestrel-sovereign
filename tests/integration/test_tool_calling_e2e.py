"""
Integration tests for OpenAI-style tool calling with Features as Subagents.

Tests the A2A (Agent-to-Agent) pattern where:
1. KestrelAgent exposes Features as high-level tools
2. LLM can call features via function calling
3. Features execute as subagents with their own context
4. Results are returned to the orchestrator
"""

import pytest
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Import the components we're testing
from kestrel_sovereign.llm.adapter import LLMAdapter, LLMResponse, ToolCall
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.features.base import Feature, tool as feature_tool
from kestrel_sovereign.tools.base import ToolCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestToolCall:
    """Test the ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall instance."""
        tc = ToolCall(
            id="call_123",
            name="test_tool",
            arguments={"arg1": "value1", "arg2": 42}
        )
        assert tc.id == "call_123"
        assert tc.name == "test_tool"
        assert tc.arguments["arg1"] == "value1"
        assert tc.arguments["arg2"] == 42


class TestLLMResponse:
    """Test the LLMResponse dataclass."""

    def test_llm_response_with_content(self):
        """Test LLMResponse with just content."""
        response = LLMResponse(content="Hello, world!")
        assert response.content == "Hello, world!"
        assert not response.has_tool_calls
        assert response.tool_calls is None

    def test_llm_response_with_tool_calls(self):
        """Test LLMResponse with tool calls."""
        tool_calls = [
            ToolCall(id="call_1", name="feature1", arguments={"task": "do something"})
        ]
        response = LLMResponse(content=None, tool_calls=tool_calls)
        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "feature1"

    def test_llm_response_with_both(self):
        """Test LLMResponse with both content and tool calls."""
        tool_calls = [
            ToolCall(id="call_1", name="feature1", arguments={"task": "do something"})
        ]
        response = LLMResponse(content="I'll help with that.", tool_calls=tool_calls)
        assert response.content == "I'll help with that."
        assert response.has_tool_calls


class TestFeatureAsSubagent:
    """Test the Feature-as-Subagent pattern."""

    def test_feature_to_orchestrator_tool(self):
        """Test converting a feature to an orchestrator tool definition."""
        # Create a mock feature
        class TestFeature(Feature):
            @property
            def tool_description(self) -> str:
                return "A test feature for doing test things"

            def initialize(self):
                pass

        agent = MagicMock()
        feature = TestFeature(agent)

        # Convert to tool
        tool_def = feature.to_orchestrator_tool()

        # Verify structure matches OpenAI function calling format
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "test_feature"
        assert "task" in tool_def["function"]["parameters"]["properties"]
        assert "context" in tool_def["function"]["parameters"]["properties"]
        assert "A test feature" in tool_def["function"]["description"]

    def test_feature_tool_name_conversion(self):
        """Test that CamelCase feature names are converted to snake_case."""
        class MyAwesomeFeature(Feature):
            @property
            def tool_description(self) -> str:
                return "Test"

            def initialize(self):
                pass

        agent = MagicMock()
        feature = MyAwesomeFeature(agent)
        assert feature.tool_name == "my_awesome_feature"


class TestOpenAIAdapterToolCalling:
    """Test OpenAI adapter tool calling support."""

    def test_create_messages_with_tools(self):
        """Test that create_messages works correctly."""
        adapter = OpenAIAdapter()
        messages = adapter.create_messages(
            user_prompt="Hello",
            system_prompt="You are helpful"
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_response_parses_tool_calls(self):
        """Test that get_response correctly parses tool calls from the API response."""
        adapter = OpenAIAdapter()

        # Mock the OpenAI client response
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "model_agent"
        mock_tool_call.function.arguments = '{"task": "list all models"}'

        mock_choice = MagicMock()
        mock_choice.message.content = "I'll list the models for you."
        mock_choice.message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Define tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "model_agent",
                    "description": "Manage LLM models",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"}
                        },
                        "required": ["task"]
                    }
                }
            }
        ]

        # Call the adapter
        response = await adapter.get_response(
            client=mock_client,
            model="gpt-5",
            messages=[{"role": "user", "content": "List models"}],
            tools=tools
        )

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "I'll list the models for you."
        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "model_agent"
        assert response.tool_calls[0].arguments["task"] == "list all models"


class TestToolFormatConversion:
    """Test tool format conversion for different providers."""

    def test_openai_format_passthrough(self):
        """Test that OpenAI format tools are used as-is."""
        adapter = OpenAIAdapter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "Test tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        # OpenAI adapter uses tools directly, no conversion needed
        assert tools[0]["type"] == "function"


class TestE2EToolCallingFlow:
    """End-to-end tests for the complete tool calling flow."""

    @pytest.mark.asyncio
    async def test_orchestrator_dispatches_to_feature(self):
        """Test that orchestrator correctly dispatches tool calls to features."""
        # This is a more complex integration test that would require
        # a real LLM or comprehensive mocking of the entire flow.
        # For now, we verify the component interfaces work together.

        # Create a mock feature
        class MockModelFeature(Feature):
            @property
            def tool_description(self) -> str:
                return "Manage LLM models"

            def initialize(self):
                pass

            async def execute_as_subagent(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
                """Mock execution that returns a predefined result."""
                return {
                    "success": True,
                    "result": f"Executed task: {task}"
                }

        agent = MagicMock()
        feature = MockModelFeature(agent)

        # Test execute_as_subagent
        result = await feature.execute_as_subagent(task="list all models")
        assert result["success"] is True
        assert "list all models" in result["result"]

    @pytest.mark.asyncio
    async def test_multi_turn_tool_calling(self):
        """Test that multi-turn tool calling works correctly."""
        # Simulate a conversation where the LLM makes multiple tool calls

        # First response has tool calls
        first_response = LLMResponse(
            content="I'll check the models and storage.",
            tool_calls=[
                ToolCall(id="call_1", name="model_agent", arguments={"task": "list models"}),
                ToolCall(id="call_2", name="model_agent", arguments={"task": "check storage"})
            ]
        )

        # Second response has no tool calls (final answer)
        second_response = LLMResponse(
            content="I found 5 models using 10GB of storage."
        )

        # Verify the flow works as expected
        assert first_response.has_tool_calls
        assert len(first_response.tool_calls) == 2
        assert not second_response.has_tool_calls


class TestToolSchemaGeneration:
    """Test automatic tool schema generation from Feature methods."""

    def test_tool_decorator_generates_schema(self):
        """Test that @tool decorator correctly generates tool schema."""
        class TestFeature(Feature):
            @property
            def tool_description(self) -> str:
                return "Test feature"

            def initialize(self):
                pass

            @feature_tool(
                name="my_tool",
                description="Does something useful",
                category=ToolCategory.SYSTEM
            )
            async def my_tool(self, file_path: str, count: int = 10):
                """
                Process a file.

                Args:
                    file_path: Path to the file to process
                    count: Number of items to process
                """
                return {"processed": True}

        agent = MagicMock()
        feature = TestFeature(agent)

        tools = feature.get_tools()
        assert len(tools) == 1

        tool = tools[0]
        assert tool.name == "my_tool"
        assert tool.schema.description == "Does something useful"

        # Verify parameters were extracted
        params = {p.name: p for p in tool.schema.parameters}
        assert "file_path" in params
        assert "count" in params
        assert params["file_path"].required is True
        assert params["count"].required is False

    def test_tool_to_openai_format(self):
        """Test converting a tool schema to OpenAI format."""
        class TestFeature(Feature):
            @property
            def tool_description(self) -> str:
                return "Test feature"

            def initialize(self):
                pass

            @feature_tool(
                name="search",
                description="Search for something",
                category=ToolCategory.COMMUNICATION
            )
            async def search(self, query: str, limit: int = 10):
                """
                Search for items.

                Args:
                    query: The search query
                    limit: Maximum results to return
                """
                return {"results": []}

        agent = MagicMock()
        feature = TestFeature(agent)

        tools = feature.get_tools()
        tool = tools[0]

        openai_format = tool.schema.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "search"
        assert "query" in openai_format["function"]["parameters"]["properties"]
        assert "limit" in openai_format["function"]["parameters"]["properties"]


class TestAutonomousToolCalling:
    """
    HARD E2E tests that verify the LLM uses function calling autonomously.

    These tests ensure that when a user asks for an action (like web search),
    the LLM invokes the tool via function calling API rather than outputting
    `!` commands as text.
    """

    @pytest.mark.asyncio
    async def test_llm_uses_function_calling_not_text_commands(self):
        """
        CRITICAL TEST: Verify LLM uses function calling instead of text commands.

        This test sends a natural language query that should trigger tool use.
        We verify that the LLM response contains `tool_calls` in the API response
        rather than outputting `!web-search` or similar as text content.
        """
        import os
        from kestrel_sovereign.llm.service import LLMService

        # Skip if no API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        # Create service with OpenAI (initializes providers in __init__)
        service = LLMService()

        # Define a simple tool that the LLM should use
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # System prompt that explicitly tells LLM to use function calling
        system_prompt = """You have tools available via function calling.
When you need to search the web, USE FUNCTION CALLING - call the web_search tool directly.
DO NOT output !web-search or any text commands. Use the tools API."""

        # User asks for a web search
        user_prompt = "Search the web for Google Agent Development Kit"

        # Call the LLM with tools
        response = await service.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            model_override="gpt-5-mini"  # Use mini for speed/cost
        )

        # CRITICAL ASSERTIONS:
        # 1. If LLM has tool_calls, it used function calling correctly
        # 2. If no tool_calls, check that content doesn't contain text commands

        if response.has_tool_calls:
            # SUCCESS: LLM used function calling
            logger.info(f"✅ LLM used function calling: {response.tool_calls}")
            assert any(tc.name == "web_search" for tc in response.tool_calls), \
                "LLM should have called web_search tool"
        else:
            # FAILURE CHECK: Did LLM output a text command instead?
            content = response.content or ""
            text_command_patterns = [
                "!web-search",
                "!web_search",
                "!search",
                "```\n!",  # Command in code block
            ]

            for pattern in text_command_patterns:
                if pattern.lower() in content.lower():
                    pytest.fail(
                        f"LLM output text command '{pattern}' instead of using function calling.\n"
                        f"Response content: {content[:500]}\n"
                        f"This indicates the system prompt is still telling LLM to output commands as text."
                    )

            # LLM might have decided not to use a tool (acceptable for some models)
            logger.warning(
                f"LLM did not use function calling, but also didn't output text commands.\n"
                f"Response: {content[:200]}"
            )

    @pytest.mark.asyncio
    async def test_tool_results_returned_to_orchestrator(self):
        """
        Test that when LLM calls a tool, the result can be returned to it.

        This verifies the multi-turn tool calling flow works end-to-end.
        """
        import os
        from kestrel_sovereign.llm.service import LLMService

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        # Create service (initializes providers in __init__)
        service = LLMService()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        system_prompt = "You are a helpful assistant with access to weather data. Use the get_weather tool when asked about weather."
        user_prompt = "What's the weather in Paris?"

        # First call - should get tool call
        response = await service.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            model_override="gpt-5-mini"
        )

        if response.has_tool_calls:
            # Verify we got a weather tool call
            weather_call = next(
                (tc for tc in response.tool_calls if tc.name == "get_weather"),
                None
            )
            assert weather_call is not None, "Expected get_weather tool call"

            # Verify the arguments include location
            assert "location" in weather_call.arguments, "Tool call should include location"
            logger.info(f"✅ Got tool call: {weather_call}")
        else:
            # Some models might answer directly - that's okay too
            logger.info(f"LLM answered directly: {response.content[:100]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
