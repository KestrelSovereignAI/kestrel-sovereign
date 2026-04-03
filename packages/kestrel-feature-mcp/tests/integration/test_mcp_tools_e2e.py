"""
Integration tests for MCP Tool Manager.

Tests the dynamic loading and execution of MCP tools via Docker.
Uses REAL Docker containers and REAL MCP SDK connections.
"""

import pytest
import pytest_asyncio
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
import logging

from kestrel_feature_mcp.manager import MCPToolManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test image
TEST_IMAGE = "kestrel-mcp-test-server"

@pytest.fixture
def check_docker():
    """Skip test if Docker is not available."""
    try:
        import docker
        from docker.credentials.errors import StoreError
        client = docker.from_env()
        client.ping()
    except ImportError as e:
        pytest.skip(f"Docker SDK not installed: {e}")
    except docker.credentials.errors.StoreError as e:
        pytest.skip(f"Docker credential store not available: {e}")
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")

@pytest_asyncio.fixture
async def build_test_image(check_docker):
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
        # Optional: cleanup image
        # client.images.remove(TEST_IMAGE)
    except docker.credentials.errors.StoreError as e:
        pytest.skip(f"Docker credential store not available: {e}")

@pytest_asyncio.fixture
async def mcp_manager(check_docker):
    """MCPToolManager instance for testing."""
    manager = MCPToolManager()
    yield manager
    # Cleanup all active tools
    active_containers = list(manager.active_tools.keys())
    for container in active_containers:
        await manager.stop_tool(container)

@pytest.mark.asyncio
async def test_mcp_server_lifecycle(mcp_manager, build_test_image):
    """
    Test the full lifecycle of a custom MCP server:
    1. Start container
    2. Connect via SSE
    3. List tools
    4. Execute tool (echo)
    5. Verify output
    6. Execute tool (add)
    7. Verify output
    8. Stop tool
    """
    container_name = "test-mcp-server"
    
    # 1. Start container
    logger.info(f"Starting {TEST_IMAGE}...")
    await mcp_manager.start_tool_container(
        TEST_IMAGE,
        container_name=container_name,
        environment={"PORT": "8000"}
    )
    
    # 2. Connect
    logger.info("Connecting to tool...")
    # Retry connection a few times as container startup might take a moment
    for i in range(5):
        try:
            tools = await mcp_manager.connect_to_tool(container_name)
            break
        except Exception as e:
            if i == 4:
                raise
            logger.info(f"Connection attempt {i+1} failed, retrying... ({e})")
            await asyncio.sleep(2)
    
    # 3. List tools
    logger.info(f"Discovered tools: {[t.name for t in tools]}")
    tool_names = [t.name for t in tools]
    assert "echo" in tool_names
    assert "add" in tool_names
    
    # 4. Execute tool (echo)
    test_text = "Hello MCP"
    logger.info(f"Calling echo with '{test_text}'...")
    
    result = await mcp_manager.call_tool(
        container_name, 
        "echo", 
        {"text": test_text}
    )
    
    # 5. Verify output
    logger.info(f"Echo result: {result}")
    # Result is likely a CallToolResult
    assert hasattr(result, "content")
    assert result.content[0].text == f"Echo: {test_text}"
    
    # 6. Execute tool (add)
    logger.info("Calling add with 10 + 20...")
    result = await mcp_manager.call_tool(
        container_name,
        "add",
        {"a": 10, "b": 20}
    )
    
    # 7. Verify output
    logger.info(f"Add result: {result}")
    assert int(result.content[0].text) == 30
    
    # 8. Stop tool
    logger.info("Stopping tool...")
    await mcp_manager.stop_tool(container_name)
    
    # Verify container is gone or stopped
    import docker
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        assert container.status != "running"
    except docker.errors.NotFound:
        pass # Container removed, which is also fine

