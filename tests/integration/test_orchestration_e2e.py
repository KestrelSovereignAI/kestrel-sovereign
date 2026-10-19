"""
End-to-End Orchestration Tests for Kestrel Agent.

This test suite verifies that the KestrelAgent correctly orchestrates its sub-agents
(ModelAgent) to perform complex tasks.

It uses a real LLMService (which may use local Ollama or cloud providers) but focuses
on the agent's internal wiring and command delegation.

Every call that reaches a live route runs under a deadline, and the tests
that make one skip unless that route answers within a few seconds (#3269).
A test's probe and calls share its ``live_call_budget``, so the deadlines of
its sequential calls together fit inside the test's timeout.
"""

import pytest
import pytest_asyncio
import logging
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.generation_gate import failure_summary
from kestrel_sovereign.llm.output_ceiling import incomplete_generation
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode
from tests.integration.live_routes import (
    TEST_TIMEOUT_SECONDS,
    bounded_live_call,
    describe_chat_routes,
    describe_discovery_routes,
    require_answering_chat_route,
    require_answering_discovery,
)
from tests.shared import no_llm_credentials
from tests.shared.genesis_audit import complete_deterministic_genesis_audit

pytestmark = pytest.mark.timeout(TEST_TIMEOUT_SECONDS)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest_asyncio.fixture
async def kestrel_agent(temp_db):
    """
    Creates a full KestrelAgent instance with real dependencies.

    Uses the temp_db fixture from conftest.py for automatic cleanup.
    """
    llm_service = LLMService()  # Will pick up env vars or config

    agent = KestrelAgent(
        did="did:test:orchestrator",
        storage_path=str(temp_db),
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )

    # Initialize async storage
    await agent.initialize()
    await complete_deterministic_genesis_audit(
        agent,
        provenance="test:orchestration_e2e",
    )

    # Skip bootstrap for test agents
    from tests.integration.conftest import complete_bootstrap
    await complete_bootstrap(agent)

    yield agent

    # Async cleanup for MCP and other resources
    await agent.shutdown()
    # Close LLM service to cleanup aiosqlite connections
    await llm_service.close()


@pytest_asyncio.fixture
async def usable_model_list(kestrel_agent, live_call_budget):
    """Skip unless the discovery behind ``!model-list`` finishes within seconds."""
    await require_answering_discovery(kestrel_agent.llm_service, budget=live_call_budget)


@pytest_asyncio.fixture
async def usable_chat_route(kestrel_agent, live_call_budget):
    """Skip unless the agent's routes answer a one-word prompt within seconds."""
    await require_answering_chat_route(kestrel_agent.llm_service, budget=live_call_budget)


async def _bounded_input(agent, user_input: str, waiting_on: str, budget):
    """Run one ``process_input`` under the live-route deadline ``budget`` allows."""
    return await bounded_live_call(
        agent.process_input(user_input),
        budget=budget,
        what=f"process_input({user_input!r})",
        waiting_on=waiting_on,
    )


@pytest.mark.asyncio
@pytest.mark.skipif(no_llm_credentials(), reason="Requires LLM credentials for orchestration tests")
async def test_orchestrator_model_management(kestrel_agent, usable_model_list, live_call_budget):
    """
    Test that the orchestrator correctly delegates model management commands
    to the ModelAgent.
    """
    logger.info("Testing Model Management Orchestration...")
    discovery_routes = describe_discovery_routes(kestrel_agent.llm_service)

    # 1. List Models - returns formatted string through command routing
    response = await _bounded_input(
        kestrel_agent, "!model-list", discovery_routes, live_call_budget
    )
    logger.info(f"List Models Response: {response}")

    # Command routing returns formatted string, not raw list
    assert isinstance(response, str), f"Expected formatted string, got {type(response)}"
    assert len(response) > 0, "Expected non-empty response"

    # Verify response contains model information
    assert "models" in response.lower() or "ollama" in response.lower() or "openai" in response.lower(), \
        f"Expected model provider info in response"
    
    # 2. Show Current Model
    # The !model command shows the currently configured model
    response = await _bounded_input(kestrel_agent, "!model", discovery_routes, live_call_budget)
    logger.info(f"Current Model Response: {response}")
    assert isinstance(response, str), f"Expected string response, got {type(response)}"
    # Response should mention current model or provide instructions
    assert len(response) > 0, "Expected non-empty response from !model"
    
    # 3. Model Info
    # Query info about a non-existent model to verify error handling
    response = await _bounded_input(
        kestrel_agent, "!model-info non-existent-model", discovery_routes, live_call_budget
    )
    assert "Model not found" in response or "Error" in response or "not found" in response.lower()

# Native MCP dispatch and approval behavior is exercised without an LLM in
# test_mcp_permission_boundary.py, using its own installed feature and peer.

@pytest.mark.asyncio
@pytest.mark.skipif(no_llm_credentials(), reason="Requires LLM credentials for natural language tool use")
async def test_orchestrator_natural_language_tool_use(
    kestrel_agent, usable_chat_route, live_call_budget
):
    """
    The agent must respond to a natural-language tool-use prompt without
    crashing or entering safe mode, and ideally invoke ModelManagerTool.

    Hard failures (these always fail the test):
      - The call raises, other than as below.
      - Response is missing / empty / not a string.
      - Agent enters safe mode (constitution audit failure or similar).

    Soft failures (xfail, never green-lit silently):
      - The local LLM does not produce any of the keywords we expect from
        the tool's formatted output. This is recorded as ``xfail`` so it
        stays visible in test reports rather than disappearing into a log
        line, but does not break CI on flaky local models.
      - The local LLM does not finish its response: a small model caught in
        a repetition loop stops at its route's output cap, and the turn fails
        as designed rather than answering (#3552).
    """
    logger.info("Testing Natural Language Tool Use...")

    query = "Please list all available AI models."
    try:
        response = await _bounded_input(
            kestrel_agent,
            query,
            describe_chat_routes(kestrel_agent.llm_service),
            live_call_budget,
        )
    except Exception as exc:
        unfinished = incomplete_generation(exc)
        if unfinished is None:
            raise
        pytest.xfail(
            f"Local LLM did not finish its response: it {failure_summary(unfinished)}."
        )
    logger.info(f"NL Response: {response}")

    assert isinstance(response, str), f"Expected str response, got {type(response)}"
    assert response.strip(), "Agent returned an empty response"
    assert "SAFE MODE" not in response, (
        f"Agent entered safe mode unexpectedly: {response[:300]}"
    )

    tool_output_indicators = (
        "Available Models",
        "Ollama",
        "OpenAI",
        "Anthropic",
        "Gemini",
        "Claude",
    )
    matched = [ind for ind in tool_output_indicators if ind.lower() in response.lower()]
    if not matched:
        pytest.xfail(
            "Local LLM did not invoke ModelManagerTool via NL. "
            f"Response head: {response[:300]!r}"
        )
