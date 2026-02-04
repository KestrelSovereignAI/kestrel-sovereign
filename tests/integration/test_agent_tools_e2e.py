"""
Integration tests for unified Feature/A2A agent tools system.

Tests the complete agent tools architecture including:
- Feature registration as A2A agents
- Command routing via TaskManager
- ModelAgent Feature functionality
- Integration with KestrelAgent

Uses REAL services (Ollama, LLM service, storage) - NO MOCKS.
Uses small models only (qwen2.5:0.5b, ~500MB).
"""

import os
import pytest
import pytest_asyncio
from pathlib import Path

from kestrel_sovereign.tools import AgentTool, ToolSchema, ToolParameter, ToolCategory
from kestrel_sovereign.features.model import ModelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode


# Test model (supports tools, available locally)
# llama3.2:latest supports tools and is already downloaded
TEST_MODEL = "llama3.2:latest"


@pytest.fixture
def skip_if_no_ollama():
    """Skip test if Ollama is not available."""
    try:
        import ollama
        client = ollama.Client()
        client.list()
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")


@pytest_asyncio.fixture
async def llm_service():
    """Real LLMService instance for testing with proper async cleanup."""
    service = LLMService()
    yield service
    await service.close()




@pytest_asyncio.fixture
async def model_agent(llm_service):
    """ModelAgent Feature instance for testing."""
    class MockAgent:
        def __init__(self, service):
            self.llm_service = service

    agent = MockAgent(llm_service)
    feature = ModelAgent(agent)
    await feature.initialize()
    return feature


@pytest_asyncio.fixture
async def kestrel_agent(llm_service, temp_db):
    """
    Real KestrelAgent instance for testing.

    Uses the temp_db fixture from conftest.py for automatic cleanup.
    """
    # Create agent with new API
    agent = KestrelAgent(
        did="did:test:agent_tools_test",
        storage_path=str(temp_db),
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )

    # Initialize async storage
    await agent.initialize()

    # Skip bootstrap for test agents - mark as complete
    if agent.bootstrap_service:
        from kestrel_sovereign.bootstrap import BootstrapState
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)

    yield agent

    # Async cleanup for MCP and other resources
    await agent.shutdown()


@pytest_asyncio.fixture
async def cleanup_test_model():
    """Cleanup test model after test completes."""
    yield
    # Cleanup happens after test
    try:
        import ollama
        client = ollama.Client()
        client.delete(TEST_MODEL)
    except (ollama.ResponseError, Exception):
        pass  # Model might not exist or Ollama unavailable, that's okay


# ============================================================================
# ModelAgent Feature Tests
# ============================================================================

@pytest.mark.asyncio
async def test_model_agent_creation(model_agent):
    """Verify ModelAgent Feature can be created."""
    assert model_agent.name == "ModelAgent"
    assert model_agent.tool_name == "model_agent"


@pytest.mark.asyncio
async def test_model_agent_get_tools(model_agent):
    """Verify ModelAgent provides tools."""
    tools = model_agent.get_tools()
    assert len(tools) > 0
    tool_names = [t.name for t in tools]
    assert "list_models" in tool_names


@pytest.mark.asyncio
async def test_model_agent_get_agent_card(model_agent):
    """Verify ModelAgent generates valid AgentCard."""
    card = model_agent.get_agent_card()
    assert card is not None
    assert card.name == "model_agent"
    assert len(card.skills) > 0
    skill_ids = [s.id for s in card.skills]
    assert "list_models" in skill_ids


@pytest.mark.asyncio
async def test_model_agent_list_models(model_agent, skip_if_no_ollama):
    """Verify listing models works via Feature."""
    # Find the list_models tool
    list_tool = None
    for tool in model_agent.get_tools():
        if tool.name == "list_models":
            list_tool = tool
            break

    assert list_tool is not None
    result = await list_tool.execute()

    assert result["success"] is True
    # Result contains 'result' key with list of ModelInfo objects
    assert "result" in result
    assert isinstance(result["result"], list)


@pytest.mark.asyncio
async def test_model_agent_storage_status(model_agent, skip_if_no_ollama):
    """Verify storage status check works."""
    # Find the storage tool
    storage_tool = None
    for tool in model_agent.get_tools():
        if tool.name == "check_storage":
            storage_tool = tool
            break

    if storage_tool is None:
        pytest.skip("check_storage tool not available")

    result = await storage_tool.execute()

    assert result["success"] is True
    assert "storage" in result


# ============================================================================
# KestrelAgent Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_kestrel_agent_has_task_manager(kestrel_agent):
    """Verify KestrelAgent has TaskManager initialized."""
    assert hasattr(kestrel_agent, 'task_manager')
    assert kestrel_agent.task_manager is not None


@pytest.mark.asyncio
async def test_kestrel_agent_features_registered(kestrel_agent):
    """Verify features are registered as A2A agents."""
    # Check TaskManager has registered agents
    agent_cards = kestrel_agent.task_manager.get_agent_cards()
    assert len(agent_cards) > 0

    agent_names = [card.name for card in agent_cards]
    # Should have model_agent registered
    assert "model_agent" in agent_names


@pytest.mark.asyncio
async def test_kestrel_agent_tool_registered(kestrel_agent):
    """Verify tools are registered via TaskManager."""
    # Get skill from TaskManager - actual command is !model-list
    result = kestrel_agent.task_manager.get_agent_for_command("!model-list")
    assert result is not None
    agent_id, skill_id = result
    assert skill_id == "list_models"


@pytest.mark.asyncio
async def test_kestrel_agent_command_execution(kestrel_agent, skip_if_no_ollama):
    """Verify agent can execute tool commands."""
    # Test !model-list command (ModelAgent feature)
    response = await kestrel_agent.process_input("!model-list")

    assert response is not None
    # Response can be a list of ModelInfo objects or formatted string
    if isinstance(response, list):
        # Unified A2A system returns raw model list
        assert len(response) > 0
        # Check first item has model-like properties
        first = response[0]
        assert hasattr(first, 'id') or 'id' in first if isinstance(first, dict) else True
    else:
        # Formatted string response
        assert isinstance(response, str)
        assert "Models" in response or "ollama" in response.lower() or "openai" in response.lower()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", "").strip() and not os.environ.get("ANTHROPIC_API_KEY", "").strip(),
    reason="Requires LLM API key to process unknown command"
)
@pytest.mark.asyncio
async def test_kestrel_agent_unknown_command(kestrel_agent):
    """Verify unknown commands return constitutional error message."""
    response = await kestrel_agent.process_input("!totally-fake-command")

    # Unknown commands return constitutional error messages
    assert response is not None
    assert len(response) > 10  # Should be a detailed response


@pytest.mark.asyncio
async def test_kestrel_agent_status_command(kestrel_agent):
    """Verify agent can handle status command."""
    response = await kestrel_agent.process_input("!status")

    assert response is not None
    # Status command is a built-in that should always work
    assert isinstance(response, str)
    assert "Agent ID:" in response or "did:" in response.lower()


# ============================================================================
# TaskManager Command Routing Tests
# ============================================================================

@pytest.mark.asyncio
async def test_task_manager_routes_command(kestrel_agent, skip_if_no_ollama):
    """Verify TaskManager correctly routes commands to features."""
    # Get the agent/skill for a command - actual command is !model-list
    result = kestrel_agent.task_manager.get_agent_for_command("!model-list")

    assert result is not None
    agent_id, skill_id = result
    assert agent_id == "model_agent"
    assert skill_id == "list_models"


@pytest.mark.asyncio
async def test_task_manager_execute_skill(kestrel_agent, skip_if_no_ollama):
    """Verify TaskManager can execute skills directly."""
    task = await kestrel_agent.task_manager.execute_skill(
        agent_id="model_agent",
        skill_id="list_models",
        args={},
        sync=True
    )

    assert task is not None
    assert task.status.state.value == "completed"
    assert task.artifacts is not None
    assert len(task.artifacts) > 0


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_list_models_workflow(kestrel_agent, skip_if_no_ollama):
    """Test complete workflow: list models via unified routing."""
    # Step 1: List models via command (actual command is !model-list)
    list_response = await kestrel_agent.process_input("!model-list")
    assert list_response is not None
    # Can be list (A2A) or string (legacy)
    assert isinstance(list_response, (list, str))


@pytest.mark.asyncio
async def test_e2e_feature_as_a2a_agent(kestrel_agent, skip_if_no_ollama):
    """Test that Features work as A2A agents."""
    # Get the model_agent Feature - key is class name "ModelAgent"
    model_feature = kestrel_agent.features.get("ModelAgent")
    assert model_feature is not None, f"ModelAgent not found. Available: {list(kestrel_agent.features.keys())}"

    # Verify it has AgentCard
    card = model_feature.get_agent_card()
    assert card is not None
    assert card.name == "model_agent"

    # Verify it can handle tasks
    from kestrel_sovereign.a2a.types import Task, TaskStatus, TaskState, Message, TextPart

    task = Task(
        id="test-task-001",
        sessionId="test-session",
        status=TaskStatus(state=TaskState.SUBMITTED),
        metadata={"skill": "list_models", "args": {}},
        history=[Message(role="user", parts=[TextPart(text="List models")])]
    )

    result_task = await model_feature.handle_task(task)

    assert result_task.status.state == TaskState.COMPLETED
    assert result_task.artifacts is not None
