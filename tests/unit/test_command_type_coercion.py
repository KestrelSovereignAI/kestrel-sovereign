"""
Unit tests for command argument type coercion in tools/base.py.

Tests that parse_command_args properly converts string arguments to
their expected types based on the parameter schema.
"""
import pytest
from kestrel_sdk.tools.base import AgentTool, ToolSchema, ToolParameter, ToolCategory


class MockToolWithTypes(AgentTool):
    """Mock tool with various parameter types for testing coercion."""

    @property
    def name(self) -> str:
        return "test_tool"

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="test_tool",
            description="Test tool for type coercion",
            category=ToolCategory.SYSTEM,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max results",
                    required=False,
                    default=10
                ),
                ToolParameter(
                    name="threshold",
                    type="number",
                    description="Score threshold",
                    required=False,
                    default=0.5
                ),
                ToolParameter(
                    name="verbose",
                    type="boolean",
                    description="Verbose output",
                    required=False,
                    default=False
                ),
            ],
            command_prefix="!test"
        )

    async def execute(self, **kwargs):
        return {"success": True, "args": kwargs}


class TestTypeCoercion:
    """Tests for _coerce_type method."""

    @pytest.fixture
    def tool(self):
        return MockToolWithTypes()

    def test_coerce_integer(self, tool):
        """Should convert string to integer."""
        assert tool._coerce_type("42", "integer") == 42
        assert tool._coerce_type("0", "integer") == 0
        assert tool._coerce_type("-5", "integer") == -5

    def test_coerce_integer_invalid(self, tool):
        """Invalid integers should return original string."""
        assert tool._coerce_type("abc", "integer") == "abc"
        assert tool._coerce_type("3.14", "integer") == "3.14"

    def test_coerce_number(self, tool):
        """Should convert string to float."""
        assert tool._coerce_type("3.14", "number") == 3.14
        assert tool._coerce_type("0.0", "number") == 0.0
        assert tool._coerce_type("-2.5", "number") == -2.5
        assert tool._coerce_type("42", "number") == 42.0

    def test_coerce_number_invalid(self, tool):
        """Invalid numbers should return original string."""
        assert tool._coerce_type("abc", "number") == "abc"

    def test_coerce_boolean_true(self, tool):
        """Should convert truthy strings to True."""
        for val in ["true", "True", "TRUE", "1", "yes", "Yes", "on", "ON"]:
            assert tool._coerce_type(val, "boolean") is True

    def test_coerce_boolean_false(self, tool):
        """Should convert falsy strings to False."""
        for val in ["false", "False", "FALSE", "0", "no", "No", "off", "OFF"]:
            assert tool._coerce_type(val, "boolean") is False

    def test_coerce_boolean_invalid(self, tool):
        """Invalid booleans should return original string."""
        assert tool._coerce_type("maybe", "boolean") == "maybe"

    def test_coerce_string(self, tool):
        """Strings should pass through unchanged."""
        assert tool._coerce_type("hello", "string") == "hello"
        assert tool._coerce_type("42", "string") == "42"


class TestParseCommandArgs:
    """Tests for parse_command_args with type coercion."""

    @pytest.fixture
    def tool(self):
        return MockToolWithTypes()

    def test_parse_with_integer_coercion(self, tool):
        """Integer parameters should be coerced from strings."""
        args = tool.parse_command_args("!test foo 20")
        assert args["query"] == "foo"
        assert args["limit"] == 20
        assert isinstance(args["limit"], int)

    def test_parse_with_default_integer(self, tool):
        """Default integer values should remain integers."""
        args = tool.parse_command_args("!test foo")
        assert args["query"] == "foo"
        assert args["limit"] == 10
        assert isinstance(args["limit"], int)

    def test_parse_with_number_coercion(self, tool):
        """Number parameters should be coerced from strings."""
        args = tool.parse_command_args("!test foo 5 0.75")
        assert args["threshold"] == 0.75
        assert isinstance(args["threshold"], float)

    def test_parse_with_boolean_coercion(self, tool):
        """Boolean parameters should be coerced from strings."""
        args = tool.parse_command_args("!test foo 5 0.5 true")
        assert args["verbose"] is True
        assert isinstance(args["verbose"], bool)

    def test_empty_command(self, tool):
        """Empty or non-command input returns empty dict."""
        assert tool.parse_command_args("") == {}
        assert tool.parse_command_args("hello") == {}


class TestMemoryCommandSimulation:
    """Simulate the !memory search command that was failing."""

    @pytest.fixture
    def memory_search_tool(self):
        """Create a tool similar to memory search."""
        class MemorySearchTool(AgentTool):
            @property
            def name(self) -> str:
                return "search_memory"

            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="search_memory",
                    description="Search memory",
                    category=ToolCategory.MEMORY,
                    parameters=[
                        ToolParameter(
                            name="query",
                            type="string",
                            description="Search query",
                            required=True
                        ),
                        ToolParameter(
                            name="limit",
                            type="integer",
                            description="Max results",
                            required=False,
                            default=10
                        ),
                    ],
                    command_prefix="!memory search"
                )

            async def execute(self, **kwargs):
                # Simulate database query that needs integer limit
                limit = kwargs.get("limit", 10)
                # This would fail if limit is a string
                if not isinstance(limit, int):
                    raise TypeError(f"LIMIT must be integer, got {type(limit)}")
                return {"success": True, "limit": limit}

        return MemorySearchTool()

    def test_memory_search_limit_is_integer(self, memory_search_tool):
        """The limit parameter should be coerced to integer."""
        # This simulates: !memory search reflection 5
        args = memory_search_tool.parse_command_args("!memory search reflection 5")
        assert args["query"] == "reflection"
        assert args["limit"] == 5
        assert isinstance(args["limit"], int), "limit must be int for SQL LIMIT clause"

    def test_memory_search_default_limit(self, memory_search_tool):
        """Default limit should already be an integer."""
        args = memory_search_tool.parse_command_args("!memory search reflection")
        assert args["limit"] == 10
        assert isinstance(args["limit"], int)

    @pytest.mark.asyncio
    async def test_execute_with_coerced_limit(self, memory_search_tool):
        """Execute should work with coerced integer limit."""
        args = memory_search_tool.parse_command_args("!memory search test 15")
        result = await memory_search_tool.execute(**args)
        assert result["success"] is True
        assert result["limit"] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
