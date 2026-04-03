"""
E2E tests for agent-controlled MCP server workflow.

These tests simulate how an AI agent would:
1. Discover available MCP servers from the catalog
2. Start a gateway with needed servers (session-scoped)
3. Use tools to complete a task
4. Shut down the gateway when done

This validates the "vending machine" model where agents spin up
servers on-demand rather than having always-running containers.

Run with: uv run pytest tests/integration/test_mcp_agent_workflow.py -v -x
"""

import pytest
import pytest_asyncio
import asyncio
import logging

from kestrel_feature_mcp.gateway import DockerMCPGateway
from kestrel_feature_mcp.manager import MCPGatewayManager
from kestrel_feature_mcp.registry import (
    check_docker_mcp_available,
    search_docker_catalog,
    list_docker_catalog_servers,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test timeout
WORKFLOW_TIMEOUT = 120  # seconds for full workflow


def check_docker_mcp():
    """Check if Docker MCP is available."""
    return check_docker_mcp_available()


# Mark all tests in this module as requiring Docker MCP
# CI can exclude these with: pytest -m "not docker_mcp"
pytestmark = [
    pytest.mark.docker_mcp,
    pytest.mark.skipif(
        not check_docker_mcp_available(),
        reason="Docker MCP Toolkit not available"
    )
]


@pytest.fixture(autouse=True)
def docker_mcp_available():
    """Skip all tests if Docker MCP is not available.

    Using autouse=True and function scope to ensure every test checks this.
    The module-level skipif marker doesn't work reliably in all environments.
    """
    if not check_docker_mcp():
        pytest.skip("Docker MCP Toolkit not available")


class TestAgentDiscoveryWorkflow:
    """
    Test the agent's ability to discover MCP servers.

    An agent needs to:
    1. Search the catalog for servers matching a capability
    2. Understand what tools each server provides
    3. Choose the right server for the task
    """

    @pytest.mark.asyncio
    async def test_agent_discovers_fetch_for_http_task(self):
        """
        Scenario: Agent needs to make HTTP requests.

        Agent should discover "fetch" server provides this capability.
        """
        # Agent searches for HTTP/web capabilities
        results = await search_docker_catalog("fetch")

        # Should find fetch server
        assert len(results) > 0
        names = [s["name"] for s in results]
        assert "fetch" in names

        # Agent can read description to understand capability
        fetch_server = next(s for s in results if s["name"] == "fetch")
        assert "description" in fetch_server
        logger.info(f"Agent discovered: {fetch_server['name']} - {fetch_server['description']}")

    @pytest.mark.asyncio
    async def test_agent_discovers_database_servers(self):
        """
        Scenario: Agent needs to query a database.

        Agent should find sqlite, postgres, or other database servers.
        """
        # Agent searches for database capabilities
        results = await search_docker_catalog("database")

        # Should find database-related servers
        assert len(results) > 0
        names = [s["name"].lower() for s in results]

        # Should find at least one database server
        database_keywords = ["sqlite", "postgres", "mysql", "database", "sql"]
        found_db = any(
            any(kw in name for kw in database_keywords)
            for name in names
        )
        assert found_db, f"No database server found. Results: {names[:10]}"
        logger.info(f"Agent discovered {len(results)} database-related servers")

    @pytest.mark.asyncio
    async def test_agent_browses_full_catalog(self):
        """
        Scenario: Agent wants to see all available capabilities.
        """
        servers = await list_docker_catalog_servers()

        assert len(servers) > 100  # Docker catalog has 311+ servers
        logger.info(f"Agent can browse {len(servers)} available MCP servers")

        # Sample some categories
        categories = set()
        for s in servers[:50]:
            desc = s.get("description", "").lower()
            if "database" in desc or "sql" in desc:
                categories.add("database")
            elif "api" in desc or "http" in desc:
                categories.add("web")
            elif "file" in desc or "storage" in desc:
                categories.add("storage")

        logger.info(f"Sample categories found: {categories}")


class TestAgentSessionWorkflow:
    """
    Test the full session workflow where an agent:
    1. Starts a gateway for a task
    2. Uses tools
    3. Shuts down when done

    This is the "vending machine" model - spin up on demand.
    """

    @pytest_asyncio.fixture
    async def manager(self, docker_mcp_available):
        """Create a manager with cleanup - simulates session lifecycle."""
        mgr = MCPGatewayManager(port=9700)  # Unique port for this test
        yield mgr
        # Cleanup - this simulates end of agent session
        if mgr.is_connected:
            await mgr.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(WORKFLOW_TIMEOUT)
    async def test_agent_session_fetch_url(self, manager):
        """
        Full session: Agent needs to fetch a URL.

        1. Agent determines it needs fetch capability
        2. Starts gateway with fetch server
        3. Calls fetch tool
        4. Gets result
        5. Session ends, gateway stops
        """
        # Step 1: Agent starts session with needed server
        logger.info("Agent starting session with fetch server...")
        tools = await manager.start(["fetch"])

        # Verify tools are available
        tool_names = [t.name for t in tools]
        assert "fetch" in tool_names
        logger.info(f"Session started with tools: {tool_names}")

        # Step 2: Agent uses the tool
        logger.info("Agent calling fetch tool...")
        result = await manager.call_tool("fetch", {
            "url": "https://httpbin.org/json"
        })

        # Step 3: Verify result
        assert result is not None
        text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
        assert "slideshow" in text.lower() or "httpbin" in text.lower()
        logger.info("Agent successfully fetched URL content")

        # Step 4: Session ends (fixture cleanup stops gateway)
        # This simulates the natural end of an agent task

    @pytest.mark.asyncio
    @pytest.mark.timeout(WORKFLOW_TIMEOUT)
    async def test_agent_session_multiple_tools(self, manager):
        """
        Session with multiple servers: Agent needs web + time capabilities.
        """
        # Start with multiple servers
        logger.info("Agent starting session with fetch + time servers...")
        tools = await manager.start(["fetch", "time"])

        tool_names = [t.name for t in tools]
        logger.info(f"Session started with {len(tool_names)} tools: {tool_names}")

        # Should have tools from both servers
        assert "fetch" in tool_names
        # Time server provides time-related tools
        has_time_tool = any("time" in name.lower() or "timezone" in name.lower() for name in tool_names)
        assert has_time_tool, f"Expected time tools, got: {tool_names}"

        # Use fetch tool
        result = await manager.call_tool("fetch", {"url": "https://httpbin.org/get"})
        assert result is not None
        logger.info("Agent used fetch tool successfully")

    @pytest.mark.asyncio
    @pytest.mark.timeout(WORKFLOW_TIMEOUT)
    async def test_agent_adds_capability_mid_session(self, manager):
        """
        Agent discovers it needs additional capability mid-task.

        1. Starts with fetch
        2. Realizes it needs time
        3. Adds time server to session
        4. Continues with both
        """
        # Start with just fetch
        tools = await manager.start(["fetch"])
        initial_count = len(tools)
        logger.info(f"Session started with {initial_count} tools")

        # Agent realizes it needs time capability
        logger.info("Agent enabling additional server mid-session...")
        tools = await manager.enable_server("time")

        # Should now have more tools
        assert len(tools) > initial_count
        logger.info(f"Session now has {len(tools)} tools (added {len(tools) - initial_count})")


class TestAgentErrorRecovery:
    """
    Test how agent handles errors in MCP workflow.
    """

    @pytest.mark.asyncio
    async def test_agent_handles_unknown_server(self):
        """Agent tries to start with non-existent server."""
        manager = MCPGatewayManager(port=9800)

        try:
            # Try to start with fake server
            with pytest.raises(Exception):  # Should fail
                await manager.start(["nonexistent_server_xyz"])
        finally:
            if manager.is_connected:
                await manager.stop()

    @pytest.mark.asyncio
    async def test_agent_handles_tool_not_found(self):
        """Agent calls a tool that doesn't exist."""
        manager = MCPGatewayManager(port=9801)

        try:
            await manager.start(["fetch"])

            # Try to call non-existent tool
            with pytest.raises(ValueError) as exc_info:
                await manager.call_tool("nonexistent_tool", {})

            assert "nonexistent_tool" in str(exc_info.value)
            logger.info("Agent properly handled unknown tool error")
        finally:
            if manager.is_connected:
                await manager.stop()

    @pytest.mark.asyncio
    async def test_agent_handles_disconnection(self):
        """Agent handles gateway disconnection gracefully."""
        manager = MCPGatewayManager(port=9802)

        await manager.start(["fetch"])
        assert manager.is_connected

        # Simulate disconnection (stop gateway)
        await manager.stop()
        assert not manager.is_connected

        # Agent should get clear error if trying to use disconnected gateway
        with pytest.raises(RuntimeError) as exc_info:
            await manager.call_tool("fetch", {"url": "https://example.com"})

        assert "Not connected" in str(exc_info.value)
        logger.info("Agent properly handled disconnection")


class TestAgentResourceManagement:
    """
    Test that agent properly manages resources (no leaks).
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(WORKFLOW_TIMEOUT * 2)
    async def test_multiple_sessions_no_resource_leak(self):
        """
        Multiple agent sessions should not leak resources.

        Each session starts fresh, uses tools, stops cleanly.
        """
        port = 9900

        for i in range(3):  # Run 3 sessions
            logger.info(f"Starting session {i + 1}/3...")
            manager = MCPGatewayManager(port=port)

            try:
                # Start session
                tools = await manager.start(["fetch"])
                assert len(tools) > 0

                # Use tool
                result = await manager.call_tool("fetch", {"url": "https://httpbin.org/get"})
                assert result is not None

                # End session
                await manager.stop()
                assert not manager.is_connected

                logger.info(f"Session {i + 1} completed cleanly")

            except Exception as e:
                # Cleanup on error
                if manager.is_connected:
                    await manager.stop()
                raise

            # Small delay between sessions
            await asyncio.sleep(0.5)

        logger.info("All sessions completed without resource leaks")

    @pytest.mark.asyncio
    async def test_gateway_stops_on_exception(self):
        """Gateway should be stoppable even after errors."""
        manager = MCPGatewayManager(port=9901)

        try:
            await manager.start(["fetch"])

            # Cause an error
            try:
                await manager.call_tool("bad_tool", {})
            except ValueError:
                pass  # Expected

            # Should still be able to stop cleanly
            await manager.stop()
            assert not manager.is_connected

        finally:
            if manager.is_connected:
                await manager.stop()


class TestSessionTimeout:
    """
    Test session timeout auto-cleanup functionality.

    This ensures gateways don't run forever if an agent crashes
    or forgets to stop the gateway.
    """

    @pytest.mark.asyncio
    async def test_gateway_timeout_properties(self):
        """Test timeout configuration and time_remaining property."""
        # Gateway with 5 minute timeout
        gateway = DockerMCPGateway(port=9850, session_timeout=300.0)
        assert gateway.session_timeout == 300.0
        assert gateway.time_remaining is None  # Not started yet

        try:
            await gateway.start(["fetch"])
            assert gateway.is_running

            # Check time remaining
            remaining = gateway.time_remaining
            assert remaining is not None
            assert 295 < remaining <= 300  # Should be close to 300s
            logger.info(f"Time remaining: {remaining:.1f}s")

        finally:
            await gateway.stop()
            assert gateway.time_remaining is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(20)  # Test should complete in 20s
    async def test_gateway_auto_stops_on_timeout(self):
        """Test that gateway auto-stops after timeout."""
        # Use a very short timeout for testing (3 seconds)
        gateway = DockerMCPGateway(port=9851, session_timeout=3.0)

        try:
            await gateway.start(["fetch"])
            assert gateway.is_running
            logger.info("Gateway started, waiting for timeout...")

            # Wait for timeout + buffer
            await asyncio.sleep(4.0)

            # Gateway should have auto-stopped
            assert not gateway.is_running
            logger.info("Gateway auto-stopped after timeout")

        finally:
            # Cleanup in case test fails
            if gateway.is_running:
                await gateway.stop()

    @pytest.mark.asyncio
    async def test_timeout_reset_on_activity(self):
        """Test that timeout can be reset (keep-alive pattern)."""
        gateway = DockerMCPGateway(port=9852, session_timeout=10.0)

        try:
            await gateway.start(["fetch"])
            initial_remaining = gateway.time_remaining
            assert initial_remaining is not None

            # Wait a bit
            await asyncio.sleep(2.0)
            remaining_after_wait = gateway.time_remaining
            assert remaining_after_wait < initial_remaining

            # Reset timeout (simulates activity)
            gateway.reset_timeout()

            # Should be back to full timeout
            remaining_after_reset = gateway.time_remaining
            assert remaining_after_reset > remaining_after_wait
            logger.info(f"Timeout reset: {remaining_after_wait:.1f}s -> {remaining_after_reset:.1f}s")

        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_no_timeout_by_default(self):
        """Test that timeout is disabled by default."""
        gateway = DockerMCPGateway(port=9853)  # No timeout specified
        assert gateway.session_timeout is None

        try:
            await gateway.start(["fetch"])
            assert gateway.time_remaining is None  # No timeout = None
            logger.info("Gateway running without timeout (as expected)")

        finally:
            await gateway.stop()


class TestRealWorldAgentScenarios:
    """
    Real-world scenarios an agent might encounter.
    """

    @pytest_asyncio.fixture
    async def manager(self, docker_mcp_available):
        """Create manager with cleanup."""
        mgr = MCPGatewayManager(port=9950)
        yield mgr
        if mgr.is_connected:
            await mgr.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(WORKFLOW_TIMEOUT)
    async def test_scenario_check_api_status(self, manager):
        """
        Scenario: User asks "Is the GitHub API up?"

        Agent should:
        1. Start fetch server
        2. Fetch GitHub API status endpoint
        3. Parse and report result
        """
        await manager.start(["fetch"])

        # Check GitHub API status
        result = await manager.call_tool("fetch", {
            "url": "https://www.githubstatus.com/api/v2/status.json"
        })

        text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])

        # Should contain status information
        assert "status" in text.lower() or "indicator" in text.lower()
        logger.info("Agent successfully checked API status")

    @pytest.mark.asyncio
    @pytest.mark.timeout(WORKFLOW_TIMEOUT)
    async def test_scenario_fetch_documentation(self, manager):
        """
        Scenario: User asks "Show me the MCP SDK README"

        Agent fetches documentation from GitHub raw URL.
        """
        await manager.start(["fetch"])

        result = await manager.call_tool("fetch", {
            "url": "https://raw.githubusercontent.com/modelcontextprotocol/python-sdk/main/README.md"
        })

        text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])

        # Should contain MCP-related content
        assert len(text) > 100
        logger.info(f"Agent fetched documentation ({len(text)} chars)")
