"""
Integration tests for REAL MCP servers.

These tests run actual MCP server containers, not mocks.
Tests are marked with @pytest.mark.docker and will be skipped if Docker is unavailable.

Transport Types:
- SSE servers can be connected to directly (test-server, private servers)
- STDIO servers need a wrapper (official mcp/* images) - tested separately
"""

import pytest
import pytest_asyncio
import asyncio
import logging
from pathlib import Path

from kestrel_feature_mcp.manager import MCPToolManager
from kestrel_feature_mcp.registry import get_registry, TransportType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timeout for container operations
CONTAINER_TIMEOUT = 60  # seconds

# Test image (built from tests/integration/mcp_test_server/)
TEST_IMAGE = "kestrel-mcp-test-server"


def check_docker_available():
    """Check if Docker is available and running."""
    try:
        import docker
        from docker.credentials.errors import StoreError
        client = docker.from_env()
        client.ping()
        return True
    except ImportError:
        return False
    except docker.credentials.errors.StoreError:
        return False
    except Exception:
        return False


@pytest.fixture(scope="module")
def docker_available():
    """Skip all tests in module if Docker is not available."""
    if not check_docker_available():
        pytest.skip("Docker not available")


@pytest_asyncio.fixture
async def build_test_image(docker_available):
    """Build the test MCP server image."""
    try:
        import docker
        from docker.credentials.errors import StoreError
        client = docker.from_env()
        dockerfile_path = Path(__file__).parent / "mcp_test_server"

        logger.info(f"Building test image {TEST_IMAGE} from {dockerfile_path}...")
        client.images.build(
            path=str(dockerfile_path),
            tag=TEST_IMAGE,
            rm=True
        )
        yield TEST_IMAGE
    except docker.credentials.errors.StoreError as e:
        pytest.skip(f"Docker credential store not available: {e}")


@pytest_asyncio.fixture
async def mcp_manager(docker_available):
    """MCPToolManager instance with cleanup."""
    manager = MCPToolManager()
    yield manager

    # Cleanup all active tools
    active_containers = list(manager.active_tools.keys())
    for container in active_containers:
        try:
            await manager.stop_tool(container)
        except Exception as e:
            logger.warning(f"Cleanup failed for {container}: {e}")

    # Close Docker client
    manager.close()


class TestSSEServerLifecycle:
    """
    Test the full lifecycle of SSE-native MCP servers.
    Uses the kestrel-mcp-test-server which natively supports SSE.
    """

    @pytest.mark.asyncio
    @pytest.mark.docker
    @pytest.mark.timeout(CONTAINER_TIMEOUT)
    async def test_start_container(self, mcp_manager, build_test_image):
        """Test starting an MCP server container."""
        container_name = "test-mcp-lifecycle-start"

        await mcp_manager.start_tool_container(
            TEST_IMAGE,
            container_name=container_name,
            environment={"PORT": "8000"}
        )

        # Verify container is tracked
        import docker
        client = docker.from_env()
        container = client.containers.get(container_name)
        assert container.status == "running"

    @pytest.mark.asyncio
    @pytest.mark.docker
    @pytest.mark.timeout(CONTAINER_TIMEOUT)
    async def test_connect_and_discover_tools(self, mcp_manager, build_test_image):
        """Test connecting to an MCP server and discovering tools."""
        container_name = "test-mcp-lifecycle-connect"

        await mcp_manager.start_tool_container(
            TEST_IMAGE,
            container_name=container_name
        )

        tools = await mcp_manager.connect_to_tool(container_name)
        tool_names = [t.name for t in tools]

        logger.info(f"Discovered tools: {tool_names}")
        assert len(tools) >= 2
        assert "echo" in tool_names
        assert "add" in tool_names

    @pytest.mark.asyncio
    @pytest.mark.docker
    @pytest.mark.timeout(CONTAINER_TIMEOUT)
    async def test_execute_echo_tool(self, mcp_manager, build_test_image):
        """Test executing the echo tool."""
        container_name = "test-mcp-lifecycle-echo"

        await mcp_manager.start_tool_container(TEST_IMAGE, container_name=container_name)
        await mcp_manager.connect_to_tool(container_name)

        test_text = "Hello from integration test!"
        result = await mcp_manager.call_tool(
            container_name,
            "echo",
            {"text": test_text}
        )

        logger.info(f"Echo result: {result}")
        assert hasattr(result, "content")
        assert result.content[0].text == f"Echo: {test_text}"

    @pytest.mark.asyncio
    @pytest.mark.docker
    @pytest.mark.timeout(CONTAINER_TIMEOUT)
    async def test_execute_add_tool(self, mcp_manager, build_test_image):
        """Test executing the add tool."""
        container_name = "test-mcp-lifecycle-add"

        await mcp_manager.start_tool_container(TEST_IMAGE, container_name=container_name)
        await mcp_manager.connect_to_tool(container_name)

        result = await mcp_manager.call_tool(
            container_name,
            "add",
            {"a": 42, "b": 58}
        )

        logger.info(f"Add result: {result}")
        assert hasattr(result, "content")
        assert int(result.content[0].text) == 100

    @pytest.mark.asyncio
    @pytest.mark.docker
    @pytest.mark.timeout(CONTAINER_TIMEOUT)
    async def test_stop_tool(self, mcp_manager, build_test_image):
        """Test stopping an MCP server."""
        container_name = "test-mcp-lifecycle-stop"

        await mcp_manager.start_tool_container(TEST_IMAGE, container_name=container_name)
        await mcp_manager.connect_to_tool(container_name)

        # Stop the tool
        await mcp_manager.stop_tool(container_name)

        # Verify container is removed
        import docker
        client = docker.from_env()
        try:
            container = client.containers.get(container_name)
            assert container.status != "running"
        except docker.errors.NotFound:
            pass  # Container was removed, which is expected


class TestRegistryIntegration:
    """
    Test loading servers using registry metadata.
    Verifies that the catalog entries work with the manager.
    """

    @pytest.mark.asyncio
    @pytest.mark.docker
    @pytest.mark.timeout(CONTAINER_TIMEOUT)
    async def test_load_from_registry(self, mcp_manager, build_test_image):
        """Test loading a server using registry lookup."""
        registry = get_registry()
        server = registry.get("test-server")

        assert server is not None, "test-server not in registry"
        assert server.transport == TransportType.SSE, "test-server should use SSE"

        container_name = "test-registry-lookup"
        await mcp_manager.start_tool_container(
            server.image,
            container_name=container_name
        )

        tools = await mcp_manager.connect_to_tool(container_name)
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_registry_sse_servers(self):
        """Verify registry correctly identifies SSE-ready servers."""
        registry = get_registry()
        sse_servers = registry.list_sse_ready()

        # Should have at least test-server as SSE-native
        names = [s.name for s in sse_servers]
        assert "test-server" in names

        # All should have SSE transport
        for server in sse_servers:
            assert server.transport == TransportType.SSE

    @pytest.mark.asyncio
    async def test_registry_stdio_servers(self):
        """Verify registry correctly identifies servers needing wrappers."""
        registry = get_registry()
        stdio_servers = registry.list_requires_wrapper()

        # Official mcp/* servers use stdio
        names = [s.name for s in stdio_servers]
        assert "fetch" in names
        assert "sqlite" in names

        # All should require wrapper
        for server in stdio_servers:
            assert server.requires_wrapper is True

    @pytest.mark.asyncio
    async def test_registry_docker_images_valid(self):
        """Verify all Docker images in registry have valid format."""
        registry = get_registry()
        docker_servers = registry.list_docker()

        for server in docker_servers:
            image = server.full_image
            assert image is not None, f"Server {server.name} has no image"
            # Image should have format like "mcp/name:tag" or "registry/image:tag"
            assert "/" in image or ":" in image, \
                f"Invalid image format for {server.name}: {image}"


class TestMultipleServers:
    """
    Test loading multiple MCP servers simultaneously.
    """

    @pytest.mark.asyncio
    @pytest.mark.docker
    @pytest.mark.timeout(CONTAINER_TIMEOUT * 2)
    async def test_load_multiple_servers(self, mcp_manager, build_test_image):
        """Test loading multiple instances of the same server."""
        containers = ["test-multi-1", "test-multi-2"]

        # Start containers
        for name in containers:
            await mcp_manager.start_tool_container(
                TEST_IMAGE,
                container_name=name
            )

        # Connect to all
        for name in containers:
            tools = await mcp_manager.connect_to_tool(name)
            assert len(tools) >= 2

        # Verify all tracked
        assert len(mcp_manager.active_tools) == len(containers)

        # Execute on each
        for i, name in enumerate(containers):
            result = await mcp_manager.call_tool(
                name,
                "add",
                {"a": i, "b": 100}
            )
            assert int(result.content[0].text) == i + 100


class TestContainerCleanup:
    """
    Test proper container cleanup on errors and shutdown.
    """

    @pytest.mark.asyncio
    @pytest.mark.docker
    @pytest.mark.timeout(CONTAINER_TIMEOUT)
    async def test_cleanup_removes_container(self, docker_available, build_test_image):
        """Test that cleanup properly removes containers."""
        import docker
        client = docker.from_env()

        manager = MCPToolManager()
        container_name = "test-cleanup-verify"

        try:
            await manager.start_tool_container(
                TEST_IMAGE,
                container_name=container_name
            )
            await manager.connect_to_tool(container_name)

            # Verify running
            container = client.containers.get(container_name)
            assert container.status == "running"

        finally:
            # Cleanup
            await manager.stop_tool(container_name)
            manager.close()

            # Verify removed
            try:
                container = client.containers.get(container_name)
                assert container.status != "running"
            except docker.errors.NotFound:
                pass  # Expected

    @pytest.mark.asyncio
    @pytest.mark.docker
    @pytest.mark.timeout(CONTAINER_TIMEOUT)
    async def test_restart_existing_container(self, mcp_manager, build_test_image):
        """Test that existing containers are restarted cleanly."""
        container_name = "test-restart"

        # Start first time
        await mcp_manager.start_tool_container(
            TEST_IMAGE,
            container_name=container_name
        )
        await mcp_manager.connect_to_tool(container_name)

        # Stop but leave container
        await mcp_manager.stop_tool(container_name)

        # Start again - should work
        await mcp_manager.start_tool_container(
            TEST_IMAGE,
            container_name=container_name
        )
        tools = await mcp_manager.connect_to_tool(container_name)
        assert len(tools) >= 2
