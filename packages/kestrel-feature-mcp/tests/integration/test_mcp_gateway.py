"""
Integration tests for Docker MCP Gateway.

These tests verify the gateway integration with real MCP servers from Docker's catalog.
Tests require Docker Desktop 29+ with MCP Toolkit enabled.

Run with: uv run pytest tests/integration/test_mcp_gateway.py -v -x
"""

import pytest
import pytest_asyncio

# Mark all tests in this module as requiring Docker MCP
pytestmark = pytest.mark.docker_mcp
import asyncio
import logging
from pathlib import Path

from kestrel_feature_mcp.gateway import (
    DockerMCPGateway,
    DockerMCPGatewayError,
    DockerMCPNotInstalledError,
    list_available_servers,
    list_enabled_servers,
)
from kestrel_feature_mcp.manager import MCPGatewayManager
from kestrel_feature_mcp.registry import (
    check_docker_mcp_available,
    list_docker_catalog_servers,
    search_docker_catalog,
    format_docker_catalog_summary,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test timeout
GATEWAY_TIMEOUT = 60  # seconds


def check_docker_mcp():
    """Check if Docker MCP is available."""
    return check_docker_mcp_available()


@pytest.fixture(scope="module")
def docker_mcp_available():
    """Skip all tests in module if Docker MCP is not available."""
    if not check_docker_mcp():
        pytest.skip("Docker MCP Toolkit not available")


class TestDockerMCPAvailability:
    """Test Docker MCP Toolkit availability checks."""

    def test_check_docker_mcp_available(self):
        """Test that we can detect Docker MCP availability."""
        result = check_docker_mcp_available()
        # This should return True or False, not raise
        assert isinstance(result, bool)

    def test_format_docker_catalog_summary(self):
        """Test catalog summary formatting."""
        summary = format_docker_catalog_summary()
        assert isinstance(summary, str)
        # Should either show server count or indicate not installed
        assert "Docker MCP" in summary or "not installed" in summary.lower()


class TestDockerMCPCatalog:
    """Test Docker MCP Catalog integration."""

    @pytest.mark.asyncio
    async def test_list_docker_catalog_servers(self, docker_mcp_available):
        """Test listing servers from Docker MCP catalog."""
        servers = await list_docker_catalog_servers()

        assert isinstance(servers, list)
        # Docker MCP catalog should have 300+ servers
        assert len(servers) > 100, f"Expected 100+ servers, got {len(servers)}"

        # Each server should have name and description
        for server in servers[:5]:
            assert "name" in server
            assert "description" in server
            assert isinstance(server["name"], str)
            assert len(server["name"]) > 0

    @pytest.mark.asyncio
    async def test_search_docker_catalog_fetch(self, docker_mcp_available):
        """Test searching for fetch server in catalog."""
        results = await search_docker_catalog("fetch")

        assert len(results) > 0
        names = [s["name"] for s in results]
        assert "fetch" in names, f"Expected 'fetch' in results, got {names}"

    @pytest.mark.asyncio
    async def test_search_docker_catalog_database(self, docker_mcp_available):
        """Test searching for database servers."""
        results = await search_docker_catalog("database")

        assert len(results) > 0
        # Should find some database-related servers
        logger.info(f"Found {len(results)} database-related servers")

    @pytest.mark.asyncio
    async def test_search_docker_catalog_nonexistent(self, docker_mcp_available):
        """Test searching for non-existent server."""
        results = await search_docker_catalog("xyznonexistent123")

        assert len(results) == 0


class TestDockerMCPGateway:
    """Test Docker MCP Gateway process management."""

    @pytest_asyncio.fixture
    async def gateway(self, docker_mcp_available):
        """Create a gateway instance with cleanup."""
        gw = DockerMCPGateway(port=9100)  # Use different port to avoid conflicts
        yield gw
        # Cleanup
        if gw.is_running:
            await gw.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_gateway_check_available(self, docker_mcp_available):
        """Test gateway availability check."""
        assert DockerMCPGateway.check_docker_mcp_available()

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_gateway_start_fetch(self, gateway):
        """Test starting gateway with fetch server."""
        auth_token = await gateway.start(["fetch"])

        assert auth_token is not None
        assert len(auth_token) > 10
        assert gateway.is_running
        assert "fetch" in gateway.enabled_servers

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_gateway_stop(self, gateway):
        """Test stopping gateway."""
        await gateway.start(["fetch"])
        assert gateway.is_running

        await gateway.stop()
        assert not gateway.is_running
        assert gateway.auth_token is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_gateway_restart(self, gateway):
        """Test restarting gateway with different servers."""
        # Start with fetch
        token1 = await gateway.start(["fetch"])
        assert "fetch" in gateway.enabled_servers

        # Restart with different servers
        token2 = await gateway.restart(["fetch", "time"])
        assert gateway.is_running
        # Token should be different after restart
        # (actually might be the same, depends on implementation)

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_gateway_add_server(self, gateway):
        """Test adding a server to running gateway."""
        await gateway.start(["fetch"])
        initial_servers = set(gateway.enabled_servers)

        # Add time server
        await gateway.add_server("time")
        assert "time" in gateway.enabled_servers
        assert "fetch" in gateway.enabled_servers


class TestMCPGatewayManager:
    """Test the high-level MCPGatewayManager class."""

    @pytest_asyncio.fixture
    async def manager(self, docker_mcp_available):
        """Create a manager instance with cleanup."""
        mgr = MCPGatewayManager(port=9200)  # Different port
        yield mgr
        # Cleanup
        if mgr.is_connected:
            await mgr.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_manager_start_and_list_tools(self, manager):
        """Test starting manager and listing tools."""
        tools = await manager.start(["fetch"])

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "fetch" in tool_names
        assert manager.is_connected

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_manager_call_fetch_tool(self, manager):
        """Test calling the fetch tool."""
        await manager.start(["fetch"])

        # Call fetch with a simple URL
        result = await manager.call_tool("fetch", {"url": "https://httpbin.org/get"})

        # Check result has content
        assert result is not None
        assert hasattr(result, "content")
        assert len(result.content) > 0

        # Content should contain httpbin response
        text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
        assert "httpbin" in text.lower() or "origin" in text.lower()

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_manager_get_all_tools(self, manager):
        """Test getting all tools as dicts."""
        await manager.start(["fetch"])
        tools = manager.get_all_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_manager_stop(self, manager):
        """Test stopping manager."""
        await manager.start(["fetch"])
        assert manager.is_connected

        await manager.stop()
        assert not manager.is_connected

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT * 2)
    async def test_manager_enable_additional_server(self, manager):
        """Test enabling additional server."""
        tools1 = await manager.start(["fetch"])
        initial_count = len(tools1)

        # Enable time server
        tools2 = await manager.enable_server("time")

        # Should have more tools now
        assert len(tools2) > initial_count


class TestMultipleServers:
    """Test loading multiple MCP servers."""

    @pytest_asyncio.fixture
    async def manager(self, docker_mcp_available):
        """Create a manager instance with cleanup."""
        mgr = MCPGatewayManager(port=9300)
        yield mgr
        if mgr.is_connected:
            await mgr.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT * 2)
    async def test_start_multiple_servers(self, manager):
        """Test starting with multiple servers at once."""
        tools = await manager.start(["fetch", "time"])

        tool_names = [t.name for t in tools]
        logger.info(f"Tools from fetch + time: {tool_names}")

        # Should have tools from both servers
        assert "fetch" in tool_names
        # Time server should provide time-related tools
        assert any("time" in name.lower() or "timezone" in name.lower() for name in tool_names)


class TestRealWorldUsage:
    """Test real-world usage scenarios."""

    @pytest_asyncio.fixture
    async def manager(self, docker_mcp_available):
        """Create a manager instance with cleanup."""
        mgr = MCPGatewayManager(port=9400)
        yield mgr
        if mgr.is_connected:
            await mgr.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(GATEWAY_TIMEOUT)
    async def test_fetch_github_readme(self, manager):
        """Test fetching a real URL (GitHub README)."""
        await manager.start(["fetch"])

        # Fetch the MCP SDK README
        result = await manager.call_tool("fetch", {
            "url": "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/main/README.md"
        })

        text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
        # Should contain some expected content
        assert len(text) > 100  # Should have substantial content


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, docker_mcp_available):
        """Test calling a non-existent tool."""
        manager = MCPGatewayManager(port=9500)
        try:
            await manager.start(["fetch"])

            with pytest.raises(ValueError) as exc_info:
                await manager.call_tool("nonexistent_tool_xyz", {})

            assert "nonexistent_tool_xyz" in str(exc_info.value)
        finally:
            if manager.is_connected:
                await manager.stop()

    @pytest.mark.asyncio
    async def test_call_without_connection(self, docker_mcp_available):
        """Test calling tool before starting gateway."""
        manager = MCPGatewayManager(port=9600)

        with pytest.raises(RuntimeError) as exc_info:
            await manager.call_tool("fetch", {"url": "https://example.com"})

        assert "Not connected" in str(exc_info.value)
