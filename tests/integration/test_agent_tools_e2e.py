"""
Integration tests for unified Feature/A2A agent tools system.

Tests the complete agent tools architecture including:
- Feature registration as A2A agents
- Command routing via TaskManager
- ModelAgent Feature functionality
- Integration with KestrelAgent

Uses REAL services (Ollama, LLM service, storage) - NO MOCKS.

Every call that reaches a live route runs under a deadline, and the tests
that make one skip unless that route answers within a few seconds (#3269).
A test's probe and calls share its ``live_call_budget``, so their deadlines
together fit inside the test's timeout.
"""

import os
import pytest
import pytest_asyncio
from pathlib import Path

from kestrel_sovereign._vendor.a2a.types import TaskState

from kestrel_sdk.tools.base import AgentTool, ToolSchema, ToolParameter, ToolCategory
from kestrel_sovereign.features.model import ModelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode
from tests.integration.live_routes import (
    TEST_TIMEOUT_SECONDS,
    bounded_live_call,
    describe_chat_routes,
    describe_discovery_routes,
    require_answering_chat_route,
    require_answering_discovery,
    require_answering_ollama,
)
from tests.shared.genesis_audit import complete_deterministic_genesis_audit

pytestmark = pytest.mark.timeout(TEST_TIMEOUT_SECONDS)


@pytest.fixture
def skip_if_no_ollama():
    """Skip test if the Ollama daemon does not list its models within seconds."""
    require_answering_ollama()


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
    await complete_deterministic_genesis_audit(
        agent,
        provenance="test:agent_tools_e2e",
    )

    # Skip bootstrap for test agents - mark as complete
    if agent.bootstrap_service:
        from kestrel_sovereign.bootstrap import BootstrapState
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)

    # test_kestrel_agent_command_execution dispatches
    # ``!model-list`` which routes to ModelAgent.list_models
    # through the orchestrator, firing the SecurityHook
    # PRE_TOOL_USE chain.  Grant ALLOW for that single pair — every
    # other tool stays at default ASK so an unexpected dispatch
    # hangs and surfaces the regression.  See
    # conftest.grant_permissions for the full background (#879).
    from tests.integration.conftest import grant_permissions
    await grant_permissions(
        agent,
        ("ModelAgent", "list_models"),
        reason="agent-tools-e2e",
    )

    yield agent

    # Async cleanup for MCP and other resources
    await agent.shutdown()


@pytest_asyncio.fixture
async def usable_model_list(skip_if_no_ollama, llm_service, live_call_budget):
    """Skip unless the discovery behind ``!model-list`` finishes within seconds.

    A live daemon is not enough (#3269): discovery contacts every configured
    route, so one that accepts the connection and never answers stalls the
    listing. Request it after the agent or feature fixture, so the probe sees
    the routes those finished setting up.
    """
    await require_answering_discovery(llm_service, budget=live_call_budget)


@pytest_asyncio.fixture
async def usable_chat_route(kestrel_agent, live_call_budget):
    """Skip unless the agent's routes answer a one-word prompt within seconds."""
    await require_answering_chat_route(kestrel_agent.llm_service, budget=live_call_budget)


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
async def test_model_agent_contributes_native_skills(model_agent):
    """A feature contributes skills to its host's signed Agent Card."""
    skills = model_agent.get_agent_skills()
    assert skills
    assert "list_models" in {skill.id for skill in skills}
    assert all(skill.name and skill.description for skill in skills)


@pytest.mark.asyncio
async def test_model_agent_list_models(model_agent, usable_model_list, live_call_budget):
    """Verify listing models works via Feature."""
    # Find the list_models tool
    list_tool = None
    for tool in model_agent.get_tools():
        if tool.name == "list_models":
            list_tool = tool
            break

    assert list_tool is not None
    result = await bounded_live_call(
        list_tool.execute(),
        budget=live_call_budget,
        what="ModelAgent list_models",
        waiting_on=describe_discovery_routes(model_agent.llm_service),
    )

    # Unified flat ToolResult wire shape (#F025): envelope spread TOP-LEVEL.
    assert result["success"] is True
    assert result["status"] == "ok"
    assert "data" in result
    assert "models" in result["data"]
    assert isinstance(result["data"]["models"], list)
    assert result["data"]["count"] > 0


@pytest.mark.asyncio
async def test_model_agent_storage_status(model_agent, skip_if_no_ollama, live_call_budget):
    """Verify storage status check works."""
    # Find the storage tool
    storage_tool = None
    for tool in model_agent.get_tools():
        if tool.name == "check_storage":
            storage_tool = tool
            break

    if storage_tool is None:
        pytest.skip("check_storage tool not available")

    result = await bounded_live_call(
        storage_tool.execute(),
        budget=live_call_budget,
        what="ModelAgent check_storage",
        waiting_on="the Ollama daemon",
    )

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
    """The model feature contributes the same native skills as its host catalog."""
    providers = kestrel_agent.task_manager.get_provider_skills()
    feature = kestrel_agent.features["ModelAgent"]
    assert "ModelAgent" in providers
    assert providers["ModelAgent"] == feature.get_agent_skills()
    assert "list_models" in {
        skill.id for skill in kestrel_agent.task_manager.get_agent_skills()
    }


@pytest.mark.asyncio
async def test_kestrel_agent_tool_registered(kestrel_agent):
    """Verify tools are registered via TaskManager."""
    # Get skill from TaskManager - actual command is !model-list
    result = kestrel_agent.task_manager.get_agent_for_command("!model-list")
    assert result is not None
    agent_id, skill_id = result
    assert skill_id == "list_models"


@pytest.mark.asyncio
async def test_kestrel_agent_command_execution(
    kestrel_agent, usable_model_list, live_call_budget
):
    """Verify agent can execute tool commands."""
    # Test !model-list command (ModelAgent feature)
    response = await bounded_live_call(
        kestrel_agent.process_input("!model-list"),
        budget=live_call_budget,
        what="process_input('!model-list')",
        waiting_on=describe_discovery_routes(kestrel_agent.llm_service),
    )

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
async def test_kestrel_agent_unknown_command(
    kestrel_agent, usable_chat_route, live_call_budget
):
    """Verify unknown commands return constitutional error message."""
    # No command handles it, so it becomes an ordinary turn on a live route.
    response = await bounded_live_call(
        kestrel_agent.process_input("!totally-fake-command"),
        budget=live_call_budget,
        what="process_input('!totally-fake-command')",
        waiting_on=describe_chat_routes(kestrel_agent.llm_service),
    )

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
async def test_task_manager_routes_command(kestrel_agent):
    """Verify TaskManager correctly routes commands to features."""
    # Get the agent/skill for a command - actual command is !model-list
    result = kestrel_agent.task_manager.get_agent_for_command("!model-list")

    assert result is not None
    agent_id, skill_id = result
    assert agent_id == "ModelAgent"
    assert skill_id == "list_models"


@pytest.mark.asyncio
async def test_task_manager_execute_skill(kestrel_agent, usable_model_list, live_call_budget):
    """Verify TaskManager can execute skills directly."""
    task = await bounded_live_call(
        kestrel_agent.task_manager.execute_skill(
            agent_id="ModelAgent",
            skill_id="list_models",
            args={},
            sync=True
        ),
        budget=live_call_budget,
        what="TaskManager execute_skill(ModelAgent, list_models)",
        waiting_on=describe_discovery_routes(kestrel_agent.llm_service),
    )

    assert task is not None
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert task.artifacts is not None
    assert len(task.artifacts) > 0


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_list_models_workflow(kestrel_agent, usable_model_list, live_call_budget):
    """Test complete workflow: list models via unified routing."""
    # Step 1: List models via command (actual command is !model-list)
    list_response = await bounded_live_call(
        kestrel_agent.process_input("!model-list"),
        budget=live_call_budget,
        what="process_input('!model-list')",
        waiting_on=describe_discovery_routes(kestrel_agent.llm_service),
    )
    assert list_response is not None
    # Can be list (A2A) or string (legacy)
    assert isinstance(list_response, (list, str))


@pytest.mark.asyncio
async def test_e2e_feature_as_a2a_agent(kestrel_agent, usable_model_list, live_call_budget):
    """Test that Features work as A2A agents."""
    # Get the model_agent Feature - key is class name "ModelAgent"
    model_feature = kestrel_agent.features.get("ModelAgent")
    assert model_feature is not None, f"ModelAgent not found. Available: {list(kestrel_agent.features.keys())}"

    # The feature contributes native skills; the host publishes its signed card.
    assert "list_models" in {skill.id for skill in model_feature.get_agent_skills()}
    from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_task, new_text_message
    from kestrel_sovereign._vendor.a2a.types import Role

    task = new_task(
        "test-task-001", "test-context", TaskState.TASK_STATE_SUBMITTED,
        history=[new_text_message("List models", role=Role.ROLE_USER)],
    )
    task.metadata.update({"skill": "list_models", "args": {}})

    result_task = await bounded_live_call(
        model_feature.handle_task(task),
        budget=live_call_budget,
        what="ModelAgent handle_task(list_models)",
        waiting_on=describe_discovery_routes(kestrel_agent.llm_service),
    )

    assert result_task.status.state == TaskState.TASK_STATE_COMPLETED
    assert result_task.artifacts is not None
