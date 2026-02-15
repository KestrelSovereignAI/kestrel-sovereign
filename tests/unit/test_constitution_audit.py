"""Unit tests for periodic constitution audit enforcement."""
import pytest
import asyncio
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode


@pytest.fixture
async def mock_agent():
    """Create a mock agent with minimal initialization for testing."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False

    # Mock the methods we'll be testing
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method from the mixin
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    return agent


@pytest.mark.asyncio
async def test_audit_triggers_after_exactly_audit_interval():
    """Test that audit triggers after exactly AUDIT_INTERVAL interactions."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call _maybe_audit 99 times - should not trigger
    for i in range(99):
        await agent._maybe_audit()
        # Verify audit was NOT called
        if i < 98:
            agent._verify_constitution_integrity.assert_not_called()

    # On the 100th interaction, audit should trigger
    await agent._maybe_audit()

    # Verify audit was called exactly once
    assert agent._verify_constitution_integrity.call_count == 1

    # Verify counter was reset
    assert agent._interaction_count == 0


@pytest.mark.asyncio
async def test_audit_triggers_after_24_hours():
    """Test that audit triggers after 24 hours elapsed."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Set last audit time to 25 hours ago
    agent._last_audit_time = datetime.now(timezone.utc) - timedelta(hours=25)

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call _maybe_audit once - should trigger due to time
    await agent._maybe_audit()

    # Verify audit was called
    agent._verify_constitution_integrity.assert_called_once()

    # Verify counters were reset
    assert agent._interaction_count == 0
    assert (datetime.now(timezone.utc) - agent._last_audit_time).total_seconds() < 1


@pytest.mark.asyncio
async def test_counter_resets_after_audit():
    """Test that interaction counter and timestamp reset after audit."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 95
    agent._last_audit_time = datetime.now(timezone.utc) - timedelta(hours=1)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Record initial timestamp
    initial_time = agent._last_audit_time

    # Trigger audit by reaching AUDIT_INTERVAL
    for _ in range(5):
        await agent._maybe_audit()

    # Verify counter was reset to 0 (then incremented by subsequent calls)
    assert agent._interaction_count < 10

    # Verify timestamp was updated
    assert agent._last_audit_time > initial_time


@pytest.mark.asyncio
async def test_safe_mode_activates_on_integrity_failure():
    """Test that safe mode activates when integrity check fails."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False

    # Mock integrity check to fail
    agent._verify_constitution_integrity = AsyncMock(
        return_value=(False, "Constitution file modified")
    )
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Trigger audit by reaching AUDIT_INTERVAL
    for _ in range(100):
        await agent._maybe_audit()

    # Verify safe mode was entered
    agent.enter_safe_mode.assert_called_once()

    # Verify the error message was passed
    call_args = agent.enter_safe_mode.call_args
    assert "Constitution audit failed" in call_args[0][0]
    assert "Constitution file modified" in call_args[0][0]


@pytest.mark.asyncio
async def test_audit_respects_custom_interval():
    """Test that custom AUDIT_INTERVAL is respected."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 50  # Custom interval
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call 49 times - should not trigger
    for _ in range(49):
        await agent._maybe_audit()

    agent._verify_constitution_integrity.assert_not_called()

    # 50th call should trigger
    await agent._maybe_audit()

    agent._verify_constitution_integrity.assert_called_once()


@pytest.mark.asyncio
async def test_audit_does_not_trigger_before_interval():
    """Test that audit does not trigger before reaching interval or 24h."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call 50 times (half the interval)
    for _ in range(50):
        await agent._maybe_audit()

    # Verify audit was NOT called
    agent._verify_constitution_integrity.assert_not_called()

    # Verify counter incremented correctly
    assert agent._interaction_count == 50


@pytest.mark.asyncio
async def test_multiple_audits_over_time():
    """Test that multiple audits can be triggered over time."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 10  # Small interval for testing
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Trigger 3 audits
    for round_num in range(3):
        for _ in range(10):
            await agent._maybe_audit()

        # Verify audit was called (round_num + 1) times
        assert agent._verify_constitution_integrity.call_count == round_num + 1

        # Verify counter was reset
        assert agent._interaction_count == 0


@pytest.mark.asyncio
async def test_audit_lazy_initialization():
    """Test that audit tracking initializes on first call if not already initialized."""
    agent = MagicMock(spec=KestrelAgent)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Don't initialize _interaction_count or _last_audit_time
    # (simulating an agent created before this feature was added)

    # Add the initialization method and _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._init_constitution_audit_tracking = ConstitutionMixin._init_constitution_audit_tracking.__get__(agent, KestrelAgent)
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call _maybe_audit - should auto-initialize
    await agent._maybe_audit()

    # Verify attributes were created
    assert hasattr(agent, '_interaction_count')
    assert hasattr(agent, '_last_audit_time')
    assert agent._interaction_count == 1
    assert isinstance(agent._last_audit_time, datetime)
