import pytest
import pytest_asyncio
import logging
from pathlib import Path
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.features.sovereignty import SovereigntyFeature
from kestrel_sovereign.features.mcp import MCPAgent
from kestrel_sovereign.features.model import ModelAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest_asyncio.fixture
async def kestrel_agent(temp_db):
    """
    Creates a full KestrelAgent instance with real dependencies.

    Uses the temp_db fixture from conftest.py for automatic cleanup.
    """
    llm_service = LLMService()

    agent = KestrelAgent(
        did="did:test:dynamic",
        storage_path=str(temp_db),
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )

    # Initialize async storage
    await agent.initialize()

    yield agent

    # Cleanup agent resources (temp_db cleanup handled by fixture)
    await agent.shutdown()
    # Close LLM service to cleanup aiosqlite connections
    await llm_service.close()

@pytest.mark.asyncio
async def test_feature_registration(kestrel_agent):
    """
    Test that features are correctly registered and their tools are available.
    """
    logger.info("Testing Feature Registration...")

    # Check if features are in the dictionary
    assert "MCPAgent" in kestrel_agent.features
    assert "ModelAgent" in kestrel_agent.features
    assert "SovereigntyFeature" in kestrel_agent.features

    # Check if tools are registered via TaskManager
    task_manager = kestrel_agent.task_manager

    # MCP Tools - check via command routing
    assert task_manager.get_agent_for_command("!mcp-list") is not None

    # Model Tools (actual command is !model-list)
    assert task_manager.get_agent_for_command("!model-list") is not None

    # Sovereignty Tools
    assert task_manager.get_agent_for_command("!export-sovereignty") is not None
    assert task_manager.get_agent_for_command("!check-sovereignty-status") is not None

@pytest.mark.asyncio
async def test_sovereignty_feature_execution(kestrel_agent):
    """
    Test executing sovereignty tools via the agent.
    """
    logger.info("Testing Sovereignty Feature Execution...")

    # Execute check status via feature directly
    sovereignty_feature = kestrel_agent.features.get("SovereigntyFeature")
    assert sovereignty_feature is not None

    tool = None
    for t in sovereignty_feature.get_tools():
        if t.name == "check_sovereignty_status":
            tool = t
            break

    assert tool is not None
    result = await tool.execute()

    assert result["success"] is True
    assert "No sovereignty exports found" in result["result"]

@pytest.mark.asyncio
async def test_mcp_feature_execution(kestrel_agent):
    """
    Test executing MCP tools via the agent.
    """
    logger.info("Testing MCP Feature Execution...")

    # List servers via feature directly
    mcp_feature = kestrel_agent.features.get("MCPAgent")
    assert mcp_feature is not None

    tool = None
    for t in mcp_feature.get_tools():
        if t.name == "mcp_list_servers":
            tool = t
            break

    assert tool is not None
    result = await tool.execute()

    assert result["success"] is True
    assert "No MCP tools loaded" in result["result"]

@pytest.mark.asyncio
async def test_model_feature_execution(kestrel_agent):
    """
    Test executing Model tools via the agent.
    """
    logger.info("Testing Model Feature Execution...")

    # List models via feature directly
    model_feature = kestrel_agent.features.get("ModelAgent")
    assert model_feature is not None

    tool = None
    for t in model_feature.get_tools():
        if t.name == "list_models":
            tool = t
            break

    assert tool is not None
    result = await tool.execute()

    assert result["success"] is True
    assert isinstance(result["result"], list)  # discover_all_models returns a list
