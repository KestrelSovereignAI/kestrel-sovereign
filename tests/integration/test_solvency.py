import pytest
import pytest_asyncio
from decimal import Decimal
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode
from unittest.mock import patch, MagicMock
import shutil
import tempfile
import os

@pytest.fixture
def temp_db():
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_agent.db")
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest_asyncio.fixture
async def agent(temp_db):
    """Create a KestrelAgent instance with mocked LLM config for deterministic tests."""
    llm_service = LLMService()

    # Override the LLM service config AFTER initialization to inject test economy model
    # This ensures tests pass regardless of what LLM providers are actually available
    if "ollama" not in llm_service.config:
        llm_service.config["ollama"] = {}
    llm_service.config["ollama"]["model"] = "test-economy-model"
    llm_service.config["ollama"]["enabled"] = True

    # Use a dummy DID
    did = "did:pkh:eip155:1:0x1234567890123456789012345678901234567890"
    # Use new API: storage_path instead of storage object
    agent = KestrelAgent(did, storage_path=temp_db, llm_service=llm_service, privacy_mode=PrivacyMode.NORMAL)
    await agent.initialize()
    yield agent
    # Cleanup both agent and LLM service
    await agent.shutdown()
    await llm_service.close()

@pytest.mark.asyncio
async def test_solvency_green_zone(agent):
    """Test solvency check when balance is high."""
    # Set balance to 100 FIL (directly set the internal balance)
    from kestrel_sovereign.features.wallet.feature import Currency
    agent.wallet._balances[Currency.FIL]["main"] = Decimal("100.0")

    override = await agent.check_solvency()

    assert override is None
    assert agent._current_model_preference == "NORMAL"

@pytest.mark.asyncio
async def test_solvency_yellow_zone(agent):
    """Test solvency check when balance is low (Economy Mode)."""
    # Set balance to 5 FIL (directly set the internal balance)
    from kestrel_sovereign.features.wallet.feature import Currency
    agent.wallet._balances[Currency.FIL]["main"] = Decimal("5.0")

    override = await agent.check_solvency()

    # Expect the mocked economy model
    assert override == "test-economy-model", f"Expected 'test-economy-model' but got {override}"
    assert agent._current_model_preference == "ECONOMY"

@pytest.mark.asyncio
async def test_solvency_red_zone(agent):
    """Test solvency check when balance is critical."""
    # Set balance to 0.5 FIL (directly set the internal balance)
    from kestrel_sovereign.features.wallet.feature import Currency
    agent.wallet._balances[Currency.FIL]["main"] = Decimal("0.5")

    override = await agent.check_solvency()

    assert override == "test-economy-model"
    assert agent._current_model_preference == "CRITICAL"

@pytest.mark.asyncio
async def test_solvency_transitions(agent):
    """Test transitions between solvency states."""
    from kestrel_sovereign.features.wallet.feature import Currency

    # Start High
    agent.wallet._balances[Currency.FIL]["main"] = Decimal("100.0")
    await agent.check_solvency()
    assert agent._current_model_preference == "NORMAL"

    # Drop to Economy
    agent.wallet._balances[Currency.FIL]["main"] = Decimal("5.0")
    override = await agent.check_solvency()
    assert override == "test-economy-model"
    assert agent._current_model_preference == "ECONOMY"

    # Drop to Critical
    agent.wallet._balances[Currency.FIL]["main"] = Decimal("0.5")
    override = await agent.check_solvency()
    assert override == "test-economy-model"
    assert agent._current_model_preference == "CRITICAL"

    # Recover to High
    agent.wallet._balances[Currency.FIL]["main"] = Decimal("50.0")
    override = await agent.check_solvency()
    assert override is None
    assert agent._current_model_preference == "NORMAL"
