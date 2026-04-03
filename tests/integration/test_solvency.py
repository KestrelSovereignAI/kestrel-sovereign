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

    # Also update the already-built providers list so _get_local_model_fallback()
    # returns our test model (it checks providers before config)
    for provider in llm_service.providers:
        if provider.get("name") == "ollama":
            provider["model"] = "test-economy-model"
            break

    # Use a dummy DID
    did = "did:pkh:eip155:1:0x1234567890123456789012345678901234567890"
    # Use new API: storage_path instead of storage object
    agent = KestrelAgent(did, storage_path=temp_db, llm_service=llm_service, privacy_mode=PrivacyMode.NORMAL)
    await agent.initialize()
    yield agent
    # Cleanup both agent and LLM service
    await agent.shutdown()
    await llm_service.close()

def _set_fil_balance(agent, amount: Decimal):
    """Set FIL balance for solvency tests, zeroing audit to avoid interference."""
    from kestrel_sovereign.features.wallet.feature import Currency
    agent.wallet._balances[Currency.FIL]["main"] = amount
    agent.wallet._balances[Currency.FIL]["audit"] = Decimal("0")

@pytest.mark.asyncio
async def test_solvency_green_zone(agent):
    """Test solvency check when balance is high."""
    _set_fil_balance(agent, Decimal("100.0"))

    override = await agent.check_solvency()

    assert override is None
    assert agent._current_model_preference == "NORMAL"

@pytest.mark.asyncio
async def test_solvency_yellow_zone(agent):
    """Test solvency check when balance is low (Economy Mode)."""
    # 0.5 FIL → $2.75 USD at default $5.50/FIL rate (Yellow Zone: $0.50-$5.00)
    _set_fil_balance(agent, Decimal("0.5"))

    override = await agent.check_solvency()

    # Expect the mocked economy model
    assert override == "test-economy-model", f"Expected 'test-economy-model' but got {override}"
    assert agent._current_model_preference == "ECONOMY"

@pytest.mark.asyncio
async def test_solvency_red_zone(agent):
    """Test solvency check when balance is critical."""
    # 0.05 FIL → $0.275 USD at default $5.50/FIL rate (Red Zone: < $0.50)
    from kestrel_sovereign.features.wallet.feature import Currency
    _set_fil_balance(agent, Decimal("0.05"))

    override = await agent.check_solvency()

    assert override == "test-economy-model"
    assert agent._current_model_preference == "CRITICAL"

@pytest.mark.asyncio
async def test_solvency_transitions(agent):
    """Test transitions between solvency states."""
    # Start High
    _set_fil_balance(agent, Decimal("100.0"))
    await agent.check_solvency()
    assert agent._current_model_preference == "NORMAL"

    # Drop to Economy (0.5 FIL → $2.75 USD, Yellow Zone)
    _set_fil_balance(agent, Decimal("0.5"))
    override = await agent.check_solvency()
    assert override == "test-economy-model"
    assert agent._current_model_preference == "ECONOMY"

    # Drop to Critical (0.05 FIL → $0.275 USD, Red Zone)
    _set_fil_balance(agent, Decimal("0.05"))
    override = await agent.check_solvency()
    assert override == "test-economy-model"
    assert agent._current_model_preference == "CRITICAL"

    # Recover to High
    _set_fil_balance(agent, Decimal("50.0"))
    override = await agent.check_solvency()
    assert override is None
    assert agent._current_model_preference == "NORMAL"
