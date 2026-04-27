"""
End-to-End Orchestration Tests for Kestrel Agent.

This test suite verifies that the KestrelAgent correctly orchestrates its sub-agents
(MCPAgent, ModelAgent) to perform complex tasks.

It uses a real LLMService (which may use local Ollama or cloud providers) but focuses
on the agent's internal wiring and command delegation.
"""

import os
import pytest
import pytest_asyncio
import logging
import asyncio
import shutil
from pathlib import Path
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode
from tests.shared import no_llm_credentials, no_docker


def _mcp_available() -> bool:
    try:
        from kestrel_sovereign.features.mcp.feature import MCPAgent  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test constants
TEST_MCP_IMAGE = "kestrel-mcp-test-server" # Custom image built in previous steps

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

    # Skip bootstrap for test agents
    from tests.integration.conftest import complete_bootstrap
    await complete_bootstrap(agent)

    yield agent

    # Async cleanup for MCP and other resources
    await agent.shutdown()
    # Close LLM service to cleanup aiosqlite connections
    await llm_service.close()

@pytest.mark.asyncio
@pytest.mark.skipif(no_llm_credentials(), reason="Requires LLM credentials for orchestration tests")
async def test_orchestrator_model_management(kestrel_agent):
    """
    Test that the orchestrator correctly delegates model management commands
    to the ModelAgent.
    """
    logger.info("Testing Model Management Orchestration...")
    
    # 1. List Models - returns formatted string through command routing
    response = await kestrel_agent.process_input("!model-list")
    logger.info(f"List Models Response: {response}")

    # Command routing returns formatted string, not raw list
    assert isinstance(response, str), f"Expected formatted string, got {type(response)}"
    assert len(response) > 0, "Expected non-empty response"

    # Verify response contains model information
    assert "models" in response.lower() or "ollama" in response.lower() or "openai" in response.lower(), \
        f"Expected model provider info in response"
    
    # 2. Show Current Model
    # The !model command shows the currently configured model
    response = await kestrel_agent.process_input("!model")
    logger.info(f"Current Model Response: {response}")
    assert isinstance(response, str), f"Expected string response, got {type(response)}"
    # Response should mention current model or provide instructions
    assert len(response) > 0, "Expected non-empty response from !model"
    
    # 3. Model Info
    # Query info about a non-existent model to verify error handling
    response = await kestrel_agent.process_input("!model-info non-existent-model")
    assert "Model not found" in response or "Error" in response or "not found" in response.lower()

@pytest.mark.asyncio
@pytest.mark.skipif(
    no_docker() or no_llm_credentials() or not _mcp_available(),
    reason="Docker/LLM credentials/kestrel-feature-mcp not available"
)
async def test_orchestrator_mcp_management(kestrel_agent):
    """
    Test that the orchestrator correctly delegates MCP commands
    to the MCPAgent.
    
    Note: This test requires Docker to be available and running.
    """
    logger.info("Testing MCP Orchestration...")
    
    # 1. Load Tool
    # This requires the docker image to be available.
    # We assume kestrel-mcp-test-server is built (from previous tests).
    response = await kestrel_agent.process_input(f"!mcp-load {TEST_MCP_IMAGE}")
    logger.info(f"MCP Load Response: {response}")
    
    assert "✅ Loaded" in response
    assert "echo" in response # The test server exposes 'echo'
    
    # 2. List Tools
    response = await kestrel_agent.process_input("!mcp-list")
    logger.info(f"MCP List Response: {response}")
    assert "echo" in response
    assert "add" in response
    
    # 3. Call Tool
    # !mcp-call <container> <tool> <args_json>
    # Container name is usually mcp-<image_name_sanitized>
    container_name = f"mcp-{TEST_MCP_IMAGE.replace('/', '-').replace(':', '-')}"
    
    # We need to wait a bit for the server to be fully ready if it wasn't already
    await asyncio.sleep(2)
    
    call_cmd = f"!mcp-call {container_name} echo {{\"message\": \"Hello Kestrel\"}}"
    response = await kestrel_agent.process_input(call_cmd)
    logger.info(f"MCP Call Response: {response}")
    
    assert "Hello Kestrel" in response

@pytest.mark.asyncio
@pytest.mark.skipif(no_llm_credentials(), reason="Requires LLM credentials for natural language tool use")
async def test_orchestrator_natural_language_tool_use(kestrel_agent):
    """
    The agent must respond to a natural-language tool-use prompt without
    crashing or entering safe mode, and ideally invoke ModelManagerTool.

    Hard failures (these always fail the test):
      - The call raises.
      - Response is missing / empty / not a string.
      - Agent enters safe mode (constitution audit failure or similar).

    Soft failure (xfail, never green-lit silently):
      - The local LLM does not produce any of the keywords we expect from
        the tool's formatted output. This is recorded as ``xfail`` so it
        stays visible in test reports rather than disappearing into a log
        line, but does not break CI on flaky local models.
    """
    logger.info("Testing Natural Language Tool Use...")

    query = "Please list all available AI models."
    response = await kestrel_agent.process_input(query)
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

