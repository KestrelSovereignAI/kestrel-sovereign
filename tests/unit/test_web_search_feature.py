"""
Unit tests for WebSearchFeature and WebSearchTool.

Tests the web search functionality:
1. WebSearchTool - API integration with Tavily
2. WebSearchFeature - Feature wrapper with @tool decorator
3. Disabled state handling when API key missing
4. Result formatting for LLM consumption
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from kestrel_sovereign.features.web_search.feature import WebSearchFeature
from kestrel_sovereign.features.web_search.tool import WebSearchTool


# =============================================================================
# WebSearchTool Tests
# =============================================================================

class TestWebSearchTool:
    """Tests for the WebSearchTool class."""

    def test_init_without_api_key(self):
        """Tool should be disabled without API key."""
        with patch.dict("os.environ", {}, clear=True):
            tool = WebSearchTool(api_key=None)
            assert tool.enabled is False

    def test_init_with_api_key(self):
        """Tool should be enabled with API key."""
        tool = WebSearchTool(api_key="test-api-key")
        assert tool.enabled is True
        assert tool.api_key == "test-api-key"

    def test_init_from_env(self):
        """Tool should read API key from environment."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "env-api-key"}):
            tool = WebSearchTool()
            assert tool.enabled is True
            assert tool.api_key == "env-api-key"

    @pytest.mark.asyncio
    async def test_search_when_disabled(self):
        """Search should return error when disabled."""
        tool = WebSearchTool(api_key=None)
        tool.enabled = False

        result = await tool.search("test query")

        assert result["success"] is False
        assert "not enabled" in result["error"]
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful search with mocked API."""
        tool = WebSearchTool(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "Test answer summary",
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "content": "Result content 1"
                },
                {
                    "title": "Result 2",
                    "url": "https://example.com/2",
                    "content": "Result content 2"
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            result = await tool.search("test query", max_results=5)

        assert result["success"] is True
        assert result["query"] == "test query"
        assert result["answer"] == "Test answer summary"
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Result 1"

    @pytest.mark.asyncio
    async def test_search_api_error(self):
        """Test handling of API errors."""
        tool = WebSearchTool(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            result = await tool.search("test query")

        assert result["success"] is False
        assert "401" in result["error"]

    @pytest.mark.asyncio
    async def test_search_timeout(self):
        """Test handling of timeout errors."""
        import httpx

        tool = WebSearchTool(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.side_effect = httpx.TimeoutException("Connection timed out")
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            result = await tool.search("test query")

        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_format_results_for_llm_success(self):
        """Test formatting successful results for LLM."""
        tool = WebSearchTool(api_key="test-key")

        search_result = {
            "success": True,
            "answer": "This is the AI summary",
            "results": [
                {
                    "title": "First Result",
                    "url": "https://example.com/1",
                    "content": "Content of first result"
                },
                {
                    "title": "Second Result",
                    "url": "https://example.com/2",
                    "content": "Content of second result"
                }
            ]
        }

        formatted = tool.format_results_for_llm(search_result)

        assert "Summary: This is the AI summary" in formatted
        assert "Found 2 results" in formatted
        assert "First Result" in formatted
        assert "https://example.com/1" in formatted
        assert "Second Result" in formatted

    def test_format_results_for_llm_failure(self):
        """Test formatting failed results."""
        tool = WebSearchTool(api_key="test-key")

        search_result = {
            "success": False,
            "error": "API key invalid",
            "results": []
        }

        formatted = tool.format_results_for_llm(search_result)

        assert "Search failed" in formatted
        assert "API key invalid" in formatted

    def test_format_results_truncates_long_content(self):
        """Test that long content is truncated."""
        tool = WebSearchTool(api_key="test-key")

        long_content = "A" * 500  # 500 characters
        search_result = {
            "success": True,
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com",
                    "content": long_content
                }
            ]
        }

        formatted = tool.format_results_for_llm(search_result)

        # Should be truncated to ~300 chars + "..."
        assert "..." in formatted
        assert "A" * 301 not in formatted


# =============================================================================
# WebSearchFeature Tests
# =============================================================================

class TestWebSearchFeature:
    """Tests for the WebSearchFeature class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.agent_id = "test-agent"
        return agent

    @pytest_asyncio.fixture
    async def feature_disabled(self, mock_agent):
        """Create a feature with disabled search (no API key)."""
        with patch.dict("os.environ", {}, clear=True):
            feature = WebSearchFeature(mock_agent)
            await feature.initialize()
            return feature

    @pytest_asyncio.fixture
    async def feature_enabled(self, mock_agent):
        """Create a feature with enabled search."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
            feature = WebSearchFeature(mock_agent)
            await feature.initialize()
            return feature

    def test_tool_description(self, mock_agent):
        """Test the feature has a description."""
        feature = WebSearchFeature(mock_agent)
        desc = feature.tool_description
        assert "search" in desc.lower()
        assert "web" in desc.lower()

    @pytest.mark.asyncio
    async def test_search_when_disabled(self, feature_disabled):
        """Search should return error when API key missing."""
        result = await feature_disabled.search("test query")

        assert result["success"] is False
        assert "disabled" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_when_enabled(self, feature_enabled):
        """Search should work when API key present."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "Test answer",
            "results": [{"title": "Test", "url": "https://test.com", "content": "Test content"}]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            result = await feature_enabled.search("test query", max_results=3)

        assert result["success"] is True
        assert result["query"] == "test query"

    def test_has_tool_decorator(self, mock_agent):
        """Verify the search method has the @tool decorator."""
        feature = WebSearchFeature(mock_agent)

        # Check that the method has tool schema (set by @tool decorator)
        search_method = feature.search
        assert hasattr(search_method, "_tool_schema") or hasattr(feature.__class__.search, "_tool_schema")

    def test_feature_name(self, mock_agent):
        """Test the feature name is set correctly."""
        feature = WebSearchFeature(mock_agent)
        # Feature name is derived from class name
        assert "web" in feature.__class__.__name__.lower() or "search" in feature.__class__.__name__.lower()


# =============================================================================
# Integration-style Tests (still unit, but more comprehensive)
# =============================================================================

class TestWebSearchIntegration:
    """More comprehensive tests combining tool and feature."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.agent_id = "test-agent"
        return agent

    @pytest.mark.asyncio
    async def test_search_and_format_flow(self, mock_agent):
        """Test the complete search and format flow."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
            feature = WebSearchFeature(mock_agent)
            await feature.initialize()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "answer": "Python is a programming language created by Guido van Rossum.",
                "results": [
                    {
                        "title": "Python Official Website",
                        "url": "https://python.org",
                        "content": "Python is a programming language that lets you work quickly."
                    },
                    {
                        "title": "Python Tutorial",
                        "url": "https://docs.python.org/tutorial",
                        "content": "Python is an easy to learn, powerful programming language."
                    }
                ]
            }

            with patch("httpx.AsyncClient") as mock_client:
                mock_instance = AsyncMock()
                mock_instance.post.return_value = mock_response
                mock_instance.__aenter__.return_value = mock_instance
                mock_instance.__aexit__.return_value = None
                mock_client.return_value = mock_instance

                result = await feature.search("What is Python?")

            assert result["success"] is True
            assert "Python" in result["answer"]
            assert len(result["results"]) == 2

            # Format for LLM
            formatted = feature.tool.format_results_for_llm(result)
            assert "Python" in formatted
            assert "python.org" in formatted

    @pytest.mark.asyncio
    async def test_max_results_clamped(self, mock_agent):
        """Test that max_results is clamped to valid range."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
            feature = WebSearchFeature(mock_agent)
            await feature.initialize()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": []}

            with patch("httpx.AsyncClient") as mock_client:
                mock_instance = AsyncMock()
                mock_instance.post.return_value = mock_response
                mock_instance.__aenter__.return_value = mock_instance
                mock_instance.__aexit__.return_value = None
                mock_client.return_value = mock_instance

                # Request 100 results (should be clamped to 10)
                await feature.search("test", max_results=100)

                # Verify the API was called with clamped value
                call_args = mock_instance.post.call_args
                assert call_args is not None
                json_arg = call_args.kwargs.get("json") or call_args[1].get("json")
                assert json_arg["max_results"] <= 10
