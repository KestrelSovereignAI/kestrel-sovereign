"""
Integration tests for Agent MCP Workflow.

Tests the full KestrelAgent workflow with MCP tools:
1. Agent receives command to load tool (!mcp-load)
2. Agent loads tool via MCPToolManager
3. Agent lists available tools (!mcp-list)
4. Agent executes tool (!mcp-call)
5. Agent unloads tool (!mcp-unload)

Uses REAL services (Docker, MCP SDK, KestrelAgent) - NO MOCKS.
"""

import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
import logging
import json

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    except docker.credentials.errors.StoreError as e:
        pytest.skip(f"Docker credential store not available: {e}")

@pytest_asyncio.fixture
async def llm_service():
    """Real LLMService instance for testing with proper async cleanup."""
    service = LLMService()
    yield service
    await service.close()

@pytest_asyncio.fixture
async def kestrel_agent(llm_service, check_docker, build_test_image, temp_db):
    """
    Real KestrelAgent instance for testing.

    Uses the temp_db fixture from conftest.py for automatic cleanup.
    """
    # Create agent with storage_path (new API)
    agent = KestrelAgent(
        did="did:test:agent_mcp_test",
        storage_path=str(temp_db),
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )

    # Initialize async storage
    await agent.initialize()

    yield agent

    # Async cleanup for MCP and other resources
    await agent.shutdown()

@pytest.mark.asyncio
async def test_agent_mcp_workflow(kestrel_agent):
    """
    Test the agent's ability to manage and use MCP tools via commands.
    """
    # 1. Load Tool
    # We use our custom test server which exposes 'echo' and 'add'
    logger.info("Step 1: Loading MCP tool...")
    response = await kestrel_agent.process_input(f"!mcp-load {TEST_IMAGE}")
    logger.info(f"Load Response: {response}")
    
    assert "✅ Loaded" in response
    assert "mcp-kestrel-mcp-test-server" in response
    assert "echo" in response
    
    # 2. List Tools
    logger.info("Step 2: Listing tools...")
    response = await kestrel_agent.process_input("!mcp-list")
    logger.info(f"List Response: {response}")
    
    assert "Available MCP Tools:" in response
    assert "echo" in response
    assert "add" in response
    
    # 3. Use Tool (Echo)
    test_content = "test"
    
    echo_args_json = '{"text":"test"}'
    
    logger.info("Step 3: Calling echo...")
    container_name = "mcp-kestrel-mcp-test-server"
    
    response = await kestrel_agent.process_input(f"!mcp-call {container_name} echo {echo_args_json}")
    logger.info(f"Echo Response: {response}")
    
    assert "Result:" in response
    assert f"Echo: {test_content}" in response
    
    # 4. Use Tool (Add)
    add_args_json = '{"a":10,"b":20}'
    
    logger.info("Step 4: Calling add...")
    response = await kestrel_agent.process_input(f"!mcp-call {container_name} add {add_args_json}")
    logger.info(f"Add Response: {response}")
    
    assert "Result:" in response
    assert "30" in response
    
    # 5. Unload Tool
    logger.info("Step 5: Unloading tool...")
    response = await kestrel_agent.process_input(f"!mcp-unload {container_name}")
    logger.info(f"Unload Response: {response}")
    
    assert "✅ Unloaded" in response
    
    # Verify it's gone
    response = await kestrel_agent.process_input("!mcp-list")
    assert "No MCP tools loaded" in response

