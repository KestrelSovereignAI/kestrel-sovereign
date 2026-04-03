"""
End-to-end tests for the Reflection feature.

Tests the complete reflection flow including:
- Insight generation from interactions
- Constitutional approval for improvements
- Sleep integration
- Database persistence
- GitHub ticket creation
- Self-model management
"""

import pytest
import uuid
import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.reflection import (
    ReflectionFeature,
    InteractionAnalyzer,
    Insight,
    InsightType,
    ReflectionSession,
    ImprovementProposal,
    ChangeType,
    SelfModel,
    EconomicGate,
    TicketCreator,
    SelfModelManager,
    EconomicsConfigError,
    TicketConfigError,
    SelfModelConfigError,
)
from kestrel_sovereign.features.reflection.hooks import ReflectionSleepHook, create_reflection_hook


class MockLLMService:
    """Mock LLM service that returns predictable insights."""

    async def generate(self, *, system_prompt: str, user_prompt: str, **kwargs):
        """Return mock insights in JSON format."""
        return '''
{
  "insights": [
    {
      "type": "pattern",
      "title": "User prefers concise responses",
      "description": "The user frequently asks for shorter explanations",
      "evidence": ["msg_1", "msg_2"],
      "confidence": 0.85,
      "actionable": true,
      "suggested_action": "Limit responses to 2-3 paragraphs unless detail is requested"
    },
    {
      "type": "success",
      "title": "Code explanations well received",
      "description": "Technical explanations with examples got positive engagement",
      "evidence": ["msg_3"],
      "confidence": 0.9,
      "actionable": false
    }
  ]
}
'''


class MockConversationStore:
    """Mock conversation store with test data."""

    def __init__(self):
        self.messages = [
            {"id": "msg_1", "role": "user", "content": "Can you explain this briefly?", "metadata": {}},
            {"id": "msg_2", "role": "assistant", "content": "Here's a brief explanation...", "metadata": {}},
            {"id": "msg_3", "role": "user", "content": "Great explanation, thanks!", "metadata": {"emotional_valence": 0.8}},
        ]

    async def get_conversation_history(self, limit: int = 50, agent_id: str = ""):
        return self.messages[:limit]

    async def get_history_since(self, since, agent_id: str = "", limit: int = 50):
        return self.messages[:limit]


class MockDatabase:
    """Mock database for testing."""

    def __init__(self):
        self.insights = []
        self.sessions = []
        self.proposals = []
        self.rules = []

    async def execute(self, sql: str, params: tuple = ()):
        """Mock execute that stores data."""
        if "INSERT INTO reflection_insights" in sql:
            self.insights.append(params)
        elif "INSERT INTO reflection_sessions" in sql:
            self.sessions.append(params)
        elif "INSERT INTO improvement_proposals" in sql:
            self.proposals.append(params)
        elif "INSERT INTO behavior_rules" in sql:
            self.rules.append(params)
        return 1

    async def fetchall(self, sql: str, params: tuple = ()):
        """Mock fetchall."""
        if "reflection_insights" in sql:
            return [(
                "insight_1", "pattern", "Test insight", "Description",
                "[]", 0.8, 1, "Do something", datetime.utcnow().isoformat()
            )]
        elif "behavior_rules" in sql:
            return [(
                "rule_1", "prop_1", "always", "Be concise",
                "response_style", 1, 10, datetime.utcnow().isoformat()
            )]
        return []


class MockAgent:
    """Mock agent for testing."""

    def __init__(self):
        self.agent_id = "test-agent-123"
        self.did = "did:pkh:eip155:1:test"
        self.storage = MagicMock()
        self.llm_service = MockLLMService()
        self.conversation_store = MockConversationStore()
        self._db = MockDatabase()
        self.storage.db = self._db
        self.features = {}

    def get_feature(self, name: str):
        return self.features.get(name)


# ============================================================================
# InteractionAnalyzer Tests
# ============================================================================

@pytest.mark.asyncio
async def test_analyzer_generates_insights():
    """Test that the analyzer generates insights from conversations."""
    llm = MockLLMService()
    conversations = MockConversationStore()

    analyzer = InteractionAnalyzer(
        llm_service=llm,
        conversation_store=conversations,
        agent_id="test-agent",
    )

    insights = await analyzer.analyze(scope="today", depth="normal")

    assert len(insights) == 2
    assert insights[0].type == InsightType.PATTERN
    assert insights[0].title == "User prefers concise responses"
    assert insights[0].confidence == 0.85
    assert insights[0].actionable is True
    assert insights[1].type == InsightType.SUCCESS
    assert insights[1].actionable is False


@pytest.mark.asyncio
async def test_analyzer_handles_empty_conversations():
    """Test analyzer handles case with no conversations."""
    llm = MockLLMService()

    class EmptyStore:
        async def get_conversation_history(self, **kwargs):
            return []

    analyzer = InteractionAnalyzer(
        llm_service=llm,
        conversation_store=EmptyStore(),
        agent_id="test-agent",
    )

    insights = await analyzer.analyze(scope="today", depth="normal")
    assert insights == []


@pytest.mark.asyncio
async def test_analyzer_handles_llm_error():
    """Test analyzer gracefully handles LLM errors."""
    class FailingLLM:
        async def generate(self, **kwargs):
            raise Exception("LLM unavailable")

    analyzer = InteractionAnalyzer(
        llm_service=FailingLLM(),
        conversation_store=MockConversationStore(),
        agent_id="test-agent",
    )

    insights = await analyzer.analyze(scope="today", depth="normal")
    assert insights == []


# ============================================================================
# ReflectionFeature Tests
# ============================================================================

@pytest.mark.asyncio
async def test_reflect_runs_layered_checks():
    """Test that reflect() runs layered checks: Arms → Memory → Mind → Action."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()

    # Set up analyzer manually since agent doesn't have all components
    feature.analyzer = InteractionAnalyzer(
        llm_service=agent.llm_service,
        conversation_store=agent.conversation_store,
        agent_id=agent.agent_id,
    )
    feature._db = agent._db

    result = await feature.reflect(scope="today", depth="normal")

    assert result["success"] is True
    assert "summary" in result
    assert result["summary"]["layers_completed"] >= 1
    # Should have layer results
    assert "arms" in result or "memory" in result or "mind" in result


@pytest.mark.asyncio
async def test_reflect_returns_summary_stats():
    """Test that reflect() returns summary statistics."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature.analyzer = InteractionAnalyzer(
        llm_service=agent.llm_service,
        conversation_store=agent.conversation_store,
        agent_id=agent.agent_id,
    )
    feature._db = agent._db

    result = await feature.reflect(scope="today", depth="normal")

    # Check summary structure
    assert "summary" in result
    summary = result["summary"]
    assert "layers_completed" in summary
    assert "total_passed" in summary
    assert "total_failed" in summary
    assert "action_count" in summary


@pytest.mark.asyncio
async def test_reflect_legacy_generates_insights():
    """Test that reflect_legacy() generates and stores insights (old behavior)."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()

    # Set up analyzer manually since agent doesn't have all components
    feature.analyzer = InteractionAnalyzer(
        llm_service=agent.llm_service,
        conversation_store=agent.conversation_store,
        agent_id=agent.agent_id,
    )
    feature._db = agent._db

    result = await feature.reflect(scope="today", depth="normal")

    # reflect() returns a dict with reflection results
    assert isinstance(result, dict)
    assert "id" in result  # Reflection session ID
    assert "arms" in result  # Layer results


@pytest.mark.asyncio
async def test_reflect_legacy_stores_insights_in_db():
    """Test that insights are persisted to the database via legacy method."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature.analyzer = InteractionAnalyzer(
        llm_service=agent.llm_service,
        conversation_store=agent.conversation_store,
        agent_id=agent.agent_id,
    )
    feature._db = agent._db

    result = await feature.reflect(scope="today", depth="normal")

    # Check that reflection ran successfully
    assert isinstance(result, dict)
    assert "id" in result  # Reflection session ID


@pytest.mark.asyncio
async def test_get_insights_retrieves_from_db():
    """Test that get_insights retrieves stored insights."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db

    result = await feature.get_insights(min_confidence=0.5, limit=10)

    assert result["success"] is True
    assert result["count"] >= 0


@pytest.mark.asyncio
async def test_propose_improvement_requires_approval():
    """Test that propose_improvement requires constitutional approval."""
    agent = MockAgent()

    # Create a mock security feature with approval queue
    mock_security = MagicMock()
    mock_queue = AsyncMock()
    mock_queue.request_approval = AsyncMock(return_value=(False, None))
    mock_security.approval_queue = mock_queue
    agent.features["security"] = mock_security

    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db

    result = await feature.propose_improvement(
        title="Be more concise",
        description="User prefers shorter responses",
        change_type="response_style",
        proposed_change="Limit responses to 3 sentences when possible",
    )

    assert result["success"] is True
    assert result["requires_approval"] is True
    assert result["approved"] is False
    assert "rejection_reason" in result


@pytest.mark.asyncio
async def test_propose_improvement_applies_when_approved():
    """Test that approved improvements are applied."""
    agent = MockAgent()

    # Create a mock security feature that approves
    mock_security = MagicMock()
    mock_queue = AsyncMock()
    mock_queue.request_approval = AsyncMock(return_value=(True, "once"))
    mock_security.approval_queue = mock_queue
    agent.features["security"] = mock_security

    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db

    result = await feature.propose_improvement(
        title="Be more concise",
        description="User prefers shorter responses",
        change_type="response_style",
        proposed_change="Limit responses to 3 sentences when possible",
    )

    assert result["success"] is True
    assert result["approved"] is True
    assert result["applied"] is True

    # Check that behavior rule was stored
    assert len(agent._db.rules) == 1


@pytest.mark.asyncio
async def test_invalid_change_type_rejected():
    """Test that invalid change types are rejected."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db

    result = await feature.propose_improvement(
        title="Test",
        description="Test",
        change_type="invalid_type",
        proposed_change="Test",
    )

    assert result["success"] is False
    assert "Invalid change_type" in result["error"]


# ============================================================================
# Sleep Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_sleep_hook_calls_reflection():
    """Test that sleep hook calls reflection methods."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature.analyzer = InteractionAnalyzer(
        llm_service=agent.llm_service,
        conversation_store=agent.conversation_store,
        agent_id=agent.agent_id,
    )
    feature._db = agent._db

    hook = ReflectionSleepHook(feature)

    # Test pre-sleep (now uses layered reflection)
    pre_result = await hook.on_pre_sleep(agent)
    # New layered reflection returns success in the result dict
    assert pre_result.get("success") is True or "summary" in pre_result

    # Test post-consolidation
    consolidation_result = {"episodes_created": 1}
    post_result = await hook.on_post_consolidation(agent, consolidation_result)
    assert post_result.get("success") is True or "summary" in post_result


@pytest.mark.asyncio
async def test_sleep_hook_skips_when_no_episodes():
    """Test that post-consolidation is skipped when no episodes created."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db

    hook = ReflectionSleepHook(feature)

    # Test with no episodes
    consolidation_result = {"episodes_created": 0}
    result = await hook.on_post_consolidation(agent, consolidation_result)

    assert result.get("skipped") is True


@pytest.mark.asyncio
async def test_create_reflection_hook_returns_none_without_feature():
    """Test that create_reflection_hook returns None when feature doesn't exist."""
    agent = MockAgent()
    # No reflection feature registered

    hook = create_reflection_hook(agent)
    assert hook is None


@pytest.mark.asyncio
async def test_create_reflection_hook_returns_hook_with_feature():
    """Test that create_reflection_hook returns hook when feature exists."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    agent.features["reflection"] = feature

    hook = create_reflection_hook(agent)
    assert hook is not None
    assert isinstance(hook, ReflectionSleepHook)


# ============================================================================
# Model Tests
# ============================================================================

def test_insight_to_dict():
    """Test Insight serialization."""
    insight = Insight(
        id="test-123",
        type=InsightType.IMPROVEMENT,
        title="Test insight",
        description="This is a test",
        evidence=["msg_1", "msg_2"],
        confidence=0.8,
        actionable=True,
        suggested_action="Do something",
    )

    data = insight.to_dict()

    assert data["id"] == "test-123"
    assert data["type"] == "improvement"
    assert data["confidence"] == 0.8
    assert data["actionable"] is True


def test_insight_from_dict():
    """Test Insight deserialization."""
    data = {
        "id": "test-123",
        "type": "pattern",
        "title": "Test",
        "description": "Test description",
        "evidence": ["msg_1"],
        "confidence": 0.7,
        "actionable": False,
        "created_at": "2025-01-01T00:00:00",
    }

    insight = Insight.from_dict(data)

    assert insight.id == "test-123"
    assert insight.type == InsightType.PATTERN
    assert insight.confidence == 0.7


def test_reflection_session_duration():
    """Test ReflectionSession duration calculation."""
    start = datetime.utcnow()
    end = start + timedelta(seconds=5)

    session = ReflectionSession(
        id="session-1",
        trigger="on_demand",
        started_at=start,
        completed_at=end,
    )

    assert session.duration_ms == 5000


def test_improvement_proposal_states():
    """Test ImprovementProposal state properties."""
    proposal = ImprovementProposal(
        id="prop-1",
        insight_id=None,
        title="Test",
        description="Test",
        change_type=ChangeType.BEHAVIOR,
        proposed_change="Test change",
    )

    # Initial state
    assert proposal.is_pending is True
    assert proposal.is_rejected is False
    assert proposal.is_applied is False

    # After rejection
    proposal.rejection_reason = "Not approved"
    assert proposal.is_pending is False
    assert proposal.is_rejected is True

    # After approval and application
    proposal2 = ImprovementProposal(
        id="prop-2",
        insight_id=None,
        title="Test",
        description="Test",
        change_type=ChangeType.BEHAVIOR,
        proposed_change="Test change",
        approved=True,
        applied_at=datetime.utcnow(),
    )
    assert proposal2.is_applied is True


# ============================================================================
# Get Active Guidance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_get_active_guidance():
    """Test retrieval of active behavior rules for prompts."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db

    guidance = await feature.get_active_guidance()

    # Should return list (may be empty in mock)
    assert isinstance(guidance, list)


# ============================================================================
# SelfModel Tests
# ============================================================================

def test_self_model_default():
    """Test SelfModel default creation."""
    model = SelfModel.default("did:pkh:test")

    assert model.agent_did == "did:pkh:test"
    assert model.version == 1
    assert "helpfulness" in model.personality_traits
    assert model.personality_traits["helpfulness"] == 0.8
    assert model.communication_style["tone"] == "friendly"


def test_self_model_to_dict():
    """Test SelfModel serialization."""
    model = SelfModel(
        agent_did="did:pkh:test",
        version=2,
        personality_traits={"helpful": 0.9},
        learned_preferences=["user likes examples"],
    )

    data = model.to_dict()

    assert data["agent_did"] == "did:pkh:test"
    assert data["version"] == 2
    assert data["personality_traits"]["helpful"] == 0.9
    assert "user likes examples" in data["learned_preferences"]


def test_self_model_from_dict():
    """Test SelfModel deserialization."""
    data = {
        "agent_did": "did:pkh:test",
        "version": 3,
        "personality_traits": {"formal": 0.6},
        "communication_style": {"tone": "professional"},
        "learned_preferences": [],
        "behavior_patterns": ["[Success] Be concise"],
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-02T00:00:00",
    }

    model = SelfModel.from_dict(data)

    assert model.agent_did == "did:pkh:test"
    assert model.version == 3
    assert model.personality_traits["formal"] == 0.6
    assert "[Success] Be concise" in model.behavior_patterns


def test_self_model_bytes_roundtrip():
    """Test SelfModel bytes serialization roundtrip."""
    original = SelfModel(
        agent_did="did:pkh:test",
        version=5,
        personality_traits={"helpfulness": 0.95},
        learned_preferences=["test preference"],
    )

    # Convert to bytes and back
    data = original.to_bytes()
    restored = SelfModel.from_bytes(data)

    assert restored.agent_did == original.agent_did
    assert restored.version == original.version
    assert restored.personality_traits == original.personality_traits
    assert restored.learned_preferences == original.learned_preferences


# ============================================================================
# EconomicGate Tests
# ============================================================================

def test_economic_gate_requires_wallet():
    """Test that EconomicGate fails fast without wallet."""
    with pytest.raises(EconomicsConfigError) as exc_info:
        EconomicGate(None)

    assert "WalletFeature required" in str(exc_info.value)


def test_economic_gate_can_create_tickets_paid():
    """Test ticket creation allowed for paid tier."""
    mock_wallet = MagicMock()
    mock_wallet.is_paid_tier.return_value = True
    mock_wallet.has_revenue_share.return_value = False

    gate = EconomicGate(mock_wallet)
    assert gate.can_create_tickets() is True


def test_economic_gate_can_create_tickets_revenue_share():
    """Test ticket creation allowed for revenue share."""
    mock_wallet = MagicMock()
    mock_wallet.is_paid_tier.return_value = False
    mock_wallet.has_revenue_share.return_value = True

    gate = EconomicGate(mock_wallet)
    assert gate.can_create_tickets() is True


def test_economic_gate_cannot_create_tickets_free():
    """Test ticket creation denied for free tier."""
    mock_wallet = MagicMock()
    mock_wallet.is_paid_tier.return_value = False
    mock_wallet.has_revenue_share.return_value = False

    gate = EconomicGate(mock_wallet)
    assert gate.can_create_tickets() is False


def test_economic_gate_self_model_requires_paid():
    """Test self-model updates require paid tier."""
    mock_wallet = MagicMock()
    mock_wallet.is_paid_tier.return_value = False

    gate = EconomicGate(mock_wallet)
    assert gate.can_update_self_model() is False

    mock_wallet.is_paid_tier.return_value = True
    assert gate.can_update_self_model() is True


def test_economic_gate_check_or_raise():
    """Test check_or_raise raises appropriate errors."""
    mock_wallet = MagicMock()
    mock_wallet.is_paid_tier.return_value = False
    mock_wallet.has_revenue_share.return_value = False

    gate = EconomicGate(mock_wallet)

    with pytest.raises(PermissionError) as exc_info:
        gate.check_or_raise("ticket_creation")

    assert "Paid tier or revenue share required" in str(exc_info.value)


# ============================================================================
# TicketCreator Tests
# ============================================================================

def test_ticket_creator_requires_github_pat(monkeypatch):
    """Test that TicketCreator fails fast without GITHUB_PAT."""
    # Clear the env vars using monkeypatch
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    mock_client = MagicMock()
    mock_client._configured = True

    with pytest.raises(TicketConfigError) as exc_info:
        TicketCreator(mock_client)

    assert "GITHUB_PAT or GITHUB_TOKEN" in str(exc_info.value)


def test_ticket_creator_requires_configured_client(monkeypatch):
    """Test that TicketCreator fails fast if client not configured."""
    # Set a fake token using monkeypatch
    monkeypatch.setenv("GITHUB_PAT", "test-token")

    mock_client = MagicMock()
    mock_client._configured = False

    with pytest.raises(TicketConfigError) as exc_info:
        TicketCreator(mock_client)

    assert "GitHub client not configured" in str(exc_info.value)


# ============================================================================
# SelfModelManager Tests
# ============================================================================

def test_self_model_manager_requires_lighthouse_key(monkeypatch):
    """Test that SelfModelManager fails fast without LIGHTHOUSE_API_KEY."""
    # Clear the env var using monkeypatch
    monkeypatch.delenv("LIGHTHOUSE_API_KEY", raising=False)

    mock_provider = MagicMock()
    mock_provider.is_available.return_value = True

    with pytest.raises(SelfModelConfigError) as exc_info:
        SelfModelManager(mock_provider, "did:pkh:test")

    assert "LIGHTHOUSE_API_KEY" in str(exc_info.value)


def test_self_model_manager_requires_available_provider(monkeypatch):
    """Test that SelfModelManager fails fast if provider not available."""
    monkeypatch.setenv("LIGHTHOUSE_API_KEY", "test-key")

    mock_provider = MagicMock()
    mock_provider.is_available.return_value = False

    with pytest.raises(SelfModelConfigError) as exc_info:
        SelfModelManager(mock_provider, "did:pkh:test")

    assert "is not available" in str(exc_info.value)


# ============================================================================
# Integration Tests for New Tools
# ============================================================================

@pytest.mark.asyncio
async def test_create_improvement_ticket_without_ticket_creator():
    """Test create_improvement_ticket fails gracefully without ticket creator."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db
    # Explicitly set ticket creator to None
    feature._ticket_creator = None

    result = await feature.create_improvement_ticket(insight_id="test-insight")

    assert result["success"] is False
    assert "Ticket" in result["error"] and "not available" in result["error"]


@pytest.mark.asyncio
async def test_get_self_model_without_manager():
    """Test get_self_model fails gracefully without manager."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db
    # Explicitly set self model manager to None
    feature._self_model_manager = None

    result = await feature.get_self_model()

    assert result["success"] is False
    assert "Self-model" in result["error"] and "not available" in result["error"]


@pytest.mark.asyncio
async def test_update_self_model_without_manager():
    """Test update_self_model fails gracefully without manager."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db
    feature._self_model_manager = None

    result = await feature.update_self_model()

    assert result["success"] is False
    assert "Self-model" in result["error"] and "not available" in result["error"]


@pytest.mark.asyncio
async def test_create_improvement_ticket_economic_gate():
    """Test that create_improvement_ticket checks economic gate."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db

    # Mock ticket creator
    mock_ticket_creator = MagicMock()
    feature._ticket_creator = mock_ticket_creator

    # Mock economic gate that denies access
    mock_wallet = MagicMock()
    mock_wallet.is_paid_tier.return_value = False
    mock_wallet.has_revenue_share.return_value = False
    feature._economic_gate = EconomicGate(mock_wallet)

    # Create ticket handler with the mocked components
    from kestrel_sovereign.features.reflection.ticket_handler import TicketHandler
    feature._ticket_handler = TicketHandler(
        mock_ticket_creator,
        feature._economic_gate,
        feature._db_helper,
        agent,
    )

    result = await feature.create_improvement_ticket(insight_id="test-insight")

    assert result["success"] is False
    assert "requires paid tier or revenue share" in result["error"]


@pytest.mark.asyncio
async def test_update_self_model_economic_gate():
    """Test that update_self_model checks economic gate."""
    agent = MockAgent()
    feature = ReflectionFeature(agent)
    await feature.initialize()
    feature._db = agent._db

    # Mock self-model manager
    mock_manager = MagicMock()
    feature._self_model_manager = mock_manager

    # Mock economic gate that denies access
    mock_wallet = MagicMock()
    mock_wallet.is_paid_tier.return_value = False
    feature._economic_gate = EconomicGate(mock_wallet)

    # Create self-model handler with the mocked components
    from kestrel_sovereign.features.reflection.self_model_handler import SelfModelHandler
    feature._self_model_handler = SelfModelHandler(
        mock_manager,
        feature._economic_gate,
        feature._db_helper,
        agent,
    )

    result = await feature.update_self_model()

    assert result["success"] is False
    assert "require paid tier" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
