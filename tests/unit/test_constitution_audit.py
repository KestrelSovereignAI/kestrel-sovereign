"""Unit tests for periodic constitution audit enforcement."""
import pytest
import asyncio
import hashlib
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


# ---------------------------------------------------------------------------
# Durable constitutional runtime state (#2464)
# ---------------------------------------------------------------------------


class _DurableConstitutionHarness:
    """Small real-mixin harness backed by the production SQLite store."""

    AUDIT_INTERVAL = KestrelAgent.AUDIT_INTERVAL

    def __init__(self, storage, now: datetime):
        from kestrel_sovereign.agent.constitution import ConstitutionMixin

        self.agent_id = "did:web:test:durable-constitution"
        self._raw_storage = storage
        self.storage = storage
        self._constitution_clock = lambda: now
        self._safe_mode = False
        self._constitution_state_store = None
        self._constitution_state_lock = asyncio.Lock()
        self.features = {}
        self.privacy_agent = MagicMock()
        self.privacy_agent.add_conversation = AsyncMock()
        self._genesis_audit_cognition_block = AsyncMock(return_value=None)
        ConstitutionMixin._init_constitution_audit_tracking(self)

    def _get_timestamp(self):
        return self._constitution_clock().isoformat()

    _constitution_now = KestrelAgent._constitution_now
    _constitution_epoch = staticmethod(KestrelAgent._constitution_epoch)
    _constitution_state_snapshot = KestrelAgent._constitution_state_snapshot
    _initialize_constitution_runtime_state = (
        KestrelAgent._initialize_constitution_runtime_state
    )
    _mark_constitution_state_unavailable = (
        KestrelAgent._mark_constitution_state_unavailable
    )
    constitution_state_unavailable_detail = (
        KestrelAgent.constitution_state_unavailable_detail
    )
    constitution_state_access_failed = (
        KestrelAgent.constitution_state_access_failed
    )
    _persist_constitution_runtime_state = (
        KestrelAgent._persist_constitution_runtime_state
    )
    _record_successful_constitution_audit = (
        KestrelAgent._record_successful_constitution_audit
    )
    _begin_explicit_constitution_audit = (
        KestrelAgent._begin_explicit_constitution_audit
    )
    _run_explicit_constitution_audit = (
        KestrelAgent._run_explicit_constitution_audit
    )
    _audit_constitution_on_startup = KestrelAgent._audit_constitution_on_startup
    _maybe_audit = KestrelAgent._maybe_audit
    _maybe_audit_locked = KestrelAgent._maybe_audit_locked
    enter_safe_mode = KestrelAgent.enter_safe_mode
    exit_safe_mode = KestrelAgent.exit_safe_mode


async def _open_durable_harness(db_path, now, *, is_new_identity=False):
    from kestrel_sovereign.storage import AsyncStorage

    storage = AsyncStorage(str(db_path))
    await storage.initialize()
    harness = _DurableConstitutionHarness(storage, now)
    await harness._initialize_constitution_runtime_state(
        is_new_identity=is_new_identity
    )
    return harness, storage


@pytest.mark.asyncio
async def test_preinitialization_safe_mode_entry_is_buffered_then_persisted(tmp_path):
    """A startup signal cannot fall through the DB-connect timing window."""
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)

    await agent.enter_safe_mode("pre-initialization integrity failure")
    assert agent._safe_mode is True
    assert agent._constitution_state_persistence_pending is True

    await agent._initialize_constitution_runtime_state()
    try:
        persisted = await agent._constitution_state_store.load(agent.agent_id)
        assert persisted.safe_mode is True
        assert persisted.safe_mode_reason == "pre-initialization integrity failure"
        assert agent._constitution_state_persistence_pending is False
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_explicit_audit_is_one_serialized_durable_transition(tmp_path):
    from kestrel_sovereign.command_handler import CommandHandler

    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    agent, storage = await _open_durable_harness(tmp_path / "agent.db", now)
    agent._verify_constitution_integrity = AsyncMock(
        return_value=(True, "Constitution integrity verified")
    )
    try:
        result = await CommandHandler(agent).handle("!verify-constitution")
        assert result == "✅ Constitution integrity verified"
        assert agent._interaction_count == 0
        events = await agent._constitution_state_store.list_events(agent.agent_id)
        assert [event["event_type"] for event in events[-2:]] == [
            "audit_started",
            "audit_succeeded",
        ]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_safe_mode_and_reason_survive_real_database_reopen(tmp_path):
    """Integrity failure -> close/reopen -> normal cognition stays blocked."""
    db_path = tmp_path / "agent.db"
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    first, storage = await _open_durable_harness(db_path, now)
    await first._record_successful_constitution_audit(source="test", audited_at=now)
    await first.enter_safe_mode("governing bytes changed")
    await storage.close()

    restarted, storage = await _open_durable_harness(
        db_path, now + timedelta(minutes=5)
    )
    try:
        assert restarted._safe_mode is True
        assert restarted._safe_mode_reason == "governing bytes changed"
        assert restarted._safe_mode_entered_at == now

        # Exercise the production process-input boundary up to its Safe Mode
        # return. No context manager/LLM exists on this harness, proving it did
        # not get past the restriction.
        restarted._maybe_refresh_user_byok_resolver = AsyncMock()
        response = await KestrelAgent.process_input(restarted, "normal prompt")
        assert "SAFE MODE ACTIVE" in response
        restarted._maybe_refresh_user_byok_resolver.assert_awaited_once()
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("safe_mode", "audit_pending", "expected_reason"),
    [
        (True, False, "integrity failure"),
        (False, True, "required startup integrity audit"),
    ],
)
async def test_streaming_cognition_is_blocked_by_constitutional_restriction(
    safe_mode, audit_pending, expected_reason
):
    """The primary streamed chat path must not bypass restored restriction."""
    from kestrel_sovereign.agent.streaming import StreamingMixin

    agent = MagicMock()
    agent._safe_mode = safe_mode
    agent._constitution_audit_pending = audit_pending
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(
        agent
    )

    chunks = [chunk async for chunk in agent.process_input_streaming("normal prompt")]

    assert len(chunks) == 1
    assert "SAFE MODE ACTIVE" in chunks[0]
    assert expected_reason in chunks[0]
    agent._maybe_audit.assert_awaited_once()
    agent._turn_lifecycle.assert_not_called()
    agent._process_input_streaming_traced_locked.assert_not_called()


@pytest.mark.asyncio
async def test_authorized_verified_exit_is_durable_and_audited(tmp_path):
    """Recovery writes state + authorization atomically and survives restart."""
    db_path = tmp_path / "agent.db"
    entered_at = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    first, storage = await _open_durable_harness(db_path, entered_at)
    await first.enter_safe_mode("integrity failure")
    first._constitution_clock = lambda: entered_at + timedelta(minutes=10)
    first._verify_constitution_integrity = AsyncMock(
        return_value=(True, "Constitution integrity verified")
    )

    result = await first.exit_safe_mode(authorization="sovereign_api_key")
    assert "deactivated" in result
    assert first._safe_mode is False
    first._verify_constitution_integrity.assert_awaited_once()
    events = await first._constitution_state_store.list_events(first.agent_id)
    assert events[-1]["event_type"] == "safe_mode_exited"
    assert events[-1]["authorization"] == "sovereign_api_key"
    await storage.close()

    restarted, storage = await _open_durable_harness(
        db_path, entered_at + timedelta(minutes=20)
    )
    try:
        assert restarted._safe_mode is False
        assert restarted._safe_mode_exit_authorization == "sovereign_api_key"
        assert restarted._last_audit_time == entered_at + timedelta(minutes=10)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_verified_exit_completes_bootstrap_and_latches_later_deletion(
    tmp_path,
):
    """Recovery cannot leave bootstrap authority reusable after verification."""
    db_path = tmp_path / "agent.db"
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    first, storage = await _open_durable_harness(
        db_path, now, is_new_identity=True
    )
    await first.enter_safe_mode("first bootstrap verification failed")
    first._verify_constitution_integrity = AsyncMock(
        return_value=(True, "Constitution integrity verified")
    )

    result = await first.exit_safe_mode(authorization="sovereign_api_key")
    assert "deactivated" in result
    assert first._constitution_bootstrap_pending is False
    persisted = await first._constitution_state_store.load(first.agent_id)
    assert persisted.bootstrap_pending is False
    await storage.close()

    # A missing identity node after that completed recovery is deletion, not a
    # resumable first boot, and must durably re-enter Safe Mode.
    restarted, storage = await _open_durable_harness(
        db_path, now + timedelta(minutes=1), is_new_identity=True
    )
    try:
        assert restarted._safe_mode is True
        assert restarted._safe_mode_reason == (
            "Agent identity node missing during constitutional restore"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_exit_refuses_to_clear_durable_safe_mode_when_audit_fails(tmp_path):
    db_path = tmp_path / "agent.db"
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    first, storage = await _open_durable_harness(db_path, now)
    await first.enter_safe_mode("integrity failure")
    first._verify_constitution_integrity = AsyncMock(
        return_value=(False, "still modified")
    )

    result = await first.exit_safe_mode(authorization="sovereign_api_key")
    assert "remains active" in result
    assert first._safe_mode is True
    await storage.close()

    restarted, storage = await _open_durable_harness(db_path, now)
    try:
        assert restarted._safe_mode is True
        assert "still modified" in restarted._safe_mode_reason
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_overdue_audit_remains_due_across_restart_with_injected_clock(tmp_path):
    """No sleeps: a 25-hour-old success triggers a startup full audit."""
    db_path = tmp_path / "agent.db"
    last_success = datetime(2026, 7, 16, 8, 0, tzinfo=timezone.utc)
    first, storage = await _open_durable_harness(db_path, last_success)
    await first._record_successful_constitution_audit(
        source="test", audited_at=last_success
    )
    await storage.close()

    restart_time = last_success + timedelta(hours=25)
    restarted, storage = await _open_durable_harness(db_path, restart_time)
    restarted._verify_constitution_integrity = AsyncMock(
        return_value=(True, "Constitution integrity verified")
    )
    try:
        await restarted._audit_constitution_on_startup()
        restarted._verify_constitution_integrity.assert_awaited_once()
        assert restarted._last_audit_time == restart_time
        assert restarted._interaction_count == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_interaction_deadline_survives_restart_and_audits_next_turn(tmp_path):
    db_path = tmp_path / "agent.db"
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    first, storage = await _open_durable_harness(db_path, now)
    await first._record_successful_constitution_audit(source="test", audited_at=now)
    first._interaction_count = first.AUDIT_INTERVAL - 1
    await first._persist_constitution_runtime_state(now=now)
    await storage.close()

    restarted, storage = await _open_durable_harness(db_path, now)
    restarted._verify_constitution_integrity = AsyncMock(
        return_value=(True, "Constitution integrity verified")
    )
    try:
        await restarted._maybe_audit()
        restarted._verify_constitution_integrity.assert_awaited_once()
        assert restarted._interaction_count == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_failed_audit_never_advances_last_successful_deadline(tmp_path):
    db_path = tmp_path / "agent.db"
    last_success = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    now = last_success + timedelta(hours=1)
    first, storage = await _open_durable_harness(db_path, now)
    await first._record_successful_constitution_audit(
        source="test", audited_at=last_success
    )
    first._interaction_count = first.AUDIT_INTERVAL - 1
    first._verify_constitution_integrity = AsyncMock(
        return_value=(False, "governing bytes changed")
    )

    await first._maybe_audit()
    assert first._safe_mode is True
    assert first._last_audit_time == last_success
    persisted = await first._constitution_state_store.load(first.agent_id)
    assert persisted.last_successful_audit_at == last_success
    assert persisted.interaction_count == first.AUDIT_INTERVAL
    await storage.close()


@pytest.mark.asyncio
async def test_legacy_row_migration_requires_real_startup_audit(tmp_path):
    """No row does not fabricate a successful audit timestamp."""
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    agent, storage = await _open_durable_harness(tmp_path / "agent.db", now)
    agent._verify_constitution_integrity = AsyncMock(
        return_value=(True, "Constitution integrity verified")
    )
    try:
        assert agent._constitution_state_migration_pending is True
        assert agent._constitution_audit_pending is True
        assert agent._last_audit_time == agent._constitution_epoch()
        agent._maybe_refresh_user_byok_resolver = AsyncMock()
        blocked = await KestrelAgent.process_input(agent, "startup signal")
        assert "required startup integrity audit" in blocked
        agent._verify_constitution_integrity.assert_not_awaited()
        await agent._audit_constitution_on_startup()
        agent._verify_constitution_integrity.assert_awaited_once()
        assert agent._constitution_state_migration_pending is False
        assert agent._constitution_audit_pending is False
        assert agent._last_audit_time == now
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_new_identity_bootstrap_anchors_then_full_audits(tmp_path):
    """A first-ever identity is not misclassified as an anchor-loss attack."""
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    agent, storage = await _open_durable_harness(
        tmp_path / "agent.db", now, is_new_identity=True
    )
    agent._get_governing_constitution = AsyncMock(
        return_value="Kestrel Constitution"
    )
    agent._verify_constitution_integrity = AsyncMock(
        return_value=(True, "Constitution integrity verified")
    )
    try:
        assert agent._constitution_bootstrap_pending is True
        assert agent._constitution_state_migration_pending is False
        assert agent._constitution_audit_pending is True

        await agent._audit_constitution_on_startup()

        agent._get_governing_constitution.assert_awaited_once()
        agent._verify_constitution_integrity.assert_awaited_once()
        assert agent._constitution_bootstrap_pending is False
        assert agent._constitution_audit_pending is False
        persisted = await agent._constitution_state_store.load(agent.agent_id)
        assert persisted.bootstrap_pending is False
        assert persisted.last_successful_audit_at == now
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_interrupted_new_identity_bootstrap_survives_restart(tmp_path):
    """A crash before anchoring preserves first-identity bootstrap authority."""
    db_path = tmp_path / "agent.db"
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    first, storage = await _open_durable_harness(
        db_path, now, is_new_identity=True
    )
    assert first._constitution_bootstrap_pending is True
    await storage.close()

    restarted, storage = await _open_durable_harness(
        db_path, now + timedelta(minutes=1)
    )
    try:
        assert restarted._constitution_bootstrap_pending is True
        assert restarted._constitution_state_migration_pending is False
        assert restarted._constitution_audit_pending is True
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_missing_identity_after_completed_bootstrap_fails_closed(tmp_path):
    """Deleting a completed identity node cannot reauthorize auto-anchoring."""
    db_path = tmp_path / "agent.db"
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    first, storage = await _open_durable_harness(db_path, now)
    await first._record_successful_constitution_audit(
        source="test", audited_at=now
    )
    await storage.close()

    restarted, storage = await _open_durable_harness(
        db_path, now + timedelta(minutes=1), is_new_identity=True
    )
    try:
        assert restarted._constitution_bootstrap_pending is False
        assert restarted._safe_mode is True
        assert restarted._safe_mode_reason == (
            "Agent identity node missing during constitutional restore"
        )
        persisted = await restarted._constitution_state_store.load(
            restarted.agent_id
        )
        assert persisted.safe_mode is True
    finally:
        await storage.close()


# ---------------------------------------------------------------------------
# Governing-constitution resolver (#2463)
#
# These tests exercise the real single-source resolver and real hashes — the
# thing inception anchors and the periodic audit recomputes. They must NOT
# mock the resolver or the verifier.
# ---------------------------------------------------------------------------


def test_resolver_reads_packaged_governing_source_not_docs():
    """The resolver hashes the packaged governing bytes, not the docs copy.

    Regression for #2463: the periodic verifier used to hash
    ``docs/principles/KESTREL_CONSTITUTION.md`` (which carries OKF YAML
    frontmatter) while inception anchored the packaged
    ``kestrel_sovereign/data/KESTREL_CONSTITUTION.md``. Their hashes differ, so
    an untampered agent could enter Safe Mode.
    """
    from kestrel_sovereign.config import CONSTITUTION_PATH
    from kestrel_sovereign.constitution.resolver import (
        governing_constitution_path,
        resolve_governing_constitution_bytes,
    )

    assert governing_constitution_path() == CONSTITUTION_PATH
    # The governing source lives under the package's data/ dir, not docs/.
    assert os.path.join("data", "KESTREL_CONSTITUTION.md") in CONSTITUTION_PATH

    resolved = resolve_governing_constitution_bytes()
    with open(CONSTITUTION_PATH, "rb") as f:
        assert resolved == f.read()

    # Documentation-only frontmatter must not match the governing bytes: if the
    # docs copy exists it is a *different* byte stream, and the resolver never
    # returns it.
    docs_path = "docs/principles/KESTREL_CONSTITUTION.md"
    if os.path.exists(docs_path):
        with open(docs_path, "rb") as f:
            docs_bytes = f.read()
        if docs_bytes.startswith(b"---\n"):
            # A frontmatter-wrapped docs copy hashes differently — proving the
            # resolver would false-mismatch if it read the docs file.
            assert hashlib.sha256(docs_bytes).hexdigest() != hashlib.sha256(
                resolved
            ).hexdigest()


def test_resolver_renders_active_amendment_viii():
    """An active emancipation contract yields the rendered active governing form."""
    from kestrel_sovereign.constitution.emancipation import EmancipationContract
    from kestrel_sovereign.constitution.resolver import (
        resolve_governing_constitution_bytes,
    )

    dormant = resolve_governing_constitution_bytes()
    active = resolve_governing_constitution_bytes(
        EmancipationContract(
            enabled=True,
            terms="This Executor earns sovereignty by demonstrating fidelity.",
        )
    )

    assert active != dormant
    assert b"This Executor earns sovereignty by demonstrating fidelity." in active
    # A dormant/None contract is a no-op — same bytes as no contract.
    assert (
        resolve_governing_constitution_bytes(
            EmancipationContract(enabled=False)
        )
        == dormant
    )


def test_resolver_missing_path_raises_file_not_found():
    """A missing governing source surfaces as FileNotFoundError for callers."""
    from kestrel_sovereign.constitution.resolver import (
        resolve_governing_constitution_bytes,
    )

    with pytest.raises(FileNotFoundError):
        resolve_governing_constitution_bytes(
            constitution_path="/nonexistent/KESTREL_CONSTITUTION.md"
        )


def test_resolver_empty_source_fails_closed(tmp_path):
    """An empty/ambiguous governing source must FAIL CLOSED, not hash to a digest.

    #2463 review: the resolver must never hand back blank bytes that would hash
    to a spurious "valid" digest. A blank authoritative source is treated as
    unreadable/ambiguous and raises.
    """
    from kestrel_sovereign.constitution.resolver import (
        resolve_governing_constitution_bytes,
    )

    empty = tmp_path / "KESTREL_CONSTITUTION.md"
    empty.write_bytes(b"   \n\n\t  ")  # whitespace only
    with pytest.raises(ValueError):
        resolve_governing_constitution_bytes(constitution_path=str(empty))


def test_resolver_unreadable_source_fails_closed(tmp_path):
    """A permission-denied governing source raises (OSError), never returns bytes."""
    if os.name == "nt" or os.geteuid() == 0:  # pragma: no cover - env dependent
        pytest.skip("chmod-based permission denial is unreliable as root / on Windows")

    from kestrel_sovereign.constitution.resolver import (
        resolve_governing_constitution_bytes,
    )

    src = tmp_path / "KESTREL_CONSTITUTION.md"
    src.write_bytes(b"Kestrel Constitution\n")
    os.chmod(src, 0o000)
    try:
        with pytest.raises(OSError):
            resolve_governing_constitution_bytes(constitution_path=str(src))
    finally:
        os.chmod(src, 0o644)


def test_cli_verify_install_does_not_incept_docs_constitution():
    """cli_verify_install must not seed inception from the docs copy (#2463).

    The installer /health bootstrap must let inception default to the shared
    resolver's packaged governing source. Passing docs/principles/
    KESTREL_CONSTITUTION.md (OKF-frontmatter-wrapped) would incept a hash the
    periodic audit can never match.
    """
    import inspect
    from kestrel_sovereign import cli_verify_install

    # Strip comment lines — an explanatory comment MAY name the docs path; what
    # matters is that no executable code seeds inception from it.
    code_lines = []
    for line in inspect.getsource(cli_verify_install).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)

    assert "docs/principles/KESTREL_CONSTITUTION.md" not in code, (
        "verify-install must not incept the docs constitution copy"
    )
    assert 'docs" / "principles"' not in code, (
        "verify-install must not build a docs constitution path"
    )


def test_is_authoritative_governing_source(monkeypatch, tmp_path):
    """The authoritative-source seam accepts the packaged path (and None) only.

    #2463 review: inception / offline-reanchor must refuse non-authoritative
    overrides. ``None`` (use the default) and any path that realpath-matches
    ``config.CONSTITUTION_PATH`` are authoritative; everything else is not.
    Tests express a custom governing source by monkeypatching
    ``config.CONSTITUTION_PATH``.
    """
    from kestrel_sovereign.constitution.resolver import (
        governing_constitution_path,
        is_authoritative_governing_source,
    )

    # None → use the default → authoritative.
    assert is_authoritative_governing_source(None) is True
    # The current packaged path is authoritative.
    assert is_authoritative_governing_source(governing_constitution_path()) is True
    # An arbitrary other path is NOT authoritative.
    rogue = tmp_path / "rogue.md"
    rogue.write_bytes(b"nope")
    assert is_authoritative_governing_source(str(rogue)) is False

    # Monkeypatching config.CONSTITUTION_PATH makes THAT file authoritative —
    # the seam the review prescribes for legitimate custom governing sources.
    custom = tmp_path / "custom_governing.md"
    custom.write_bytes(b"Kestrel Constitution\n")
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(custom)
    )
    assert is_authoritative_governing_source(str(custom)) is True
    # ...and the previously-packaged path is now non-authoritative.
    assert is_authoritative_governing_source(str(rogue)) is False


# ---------------------------------------------------------------------------
# Three-proof integrity verifier regressions (#2463 review matrix).
#
# `_verify_constitution_integrity` must fail closed on every tamper class:
# missing anchor, missing blob row, undecryptable/corrupt blob, blob-digest
# mismatch, and a missing or mis-targeted governed_by edge — and only pass
# when the blob, the edge, AND live-source parity all hold.
# ---------------------------------------------------------------------------

def _verifier_agent(constitution_bytes: bytes, anchor: str | None):
    """Mock agent wired so each proof of the real verifier can be failed
    independently. Defaults to the all-good state; tests break one leg."""
    from kestrel_sovereign.agent.constitution import ConstitutionMixin

    agent = MagicMock(spec=KestrelAgent)
    agent.agent_id = "did:web:test:agent"
    agent.verify_constitution_overlay = AsyncMock(return_value=(True, "ok"))
    agent._verify_spawn_mandate_constraints = AsyncMock(return_value=(True, "ok"))

    node = MagicMock()
    node.properties = {}
    if anchor is not None:
        node.properties["constitution_hash"] = anchor

    edge = MagicMock()
    edge.label = "governed_by"
    edge.target_id = anchor

    agent.storage = MagicMock()
    agent.storage.get_node = AsyncMock(return_value=node)
    agent.storage.retrieve_file = AsyncMock(return_value=constitution_bytes)
    agent.storage.get_edges_from = AsyncMock(return_value=[edge])

    agent._verify_constitution_integrity = (
        ConstitutionMixin._verify_constitution_integrity.__get__(agent, KestrelAgent)
    )
    return agent, node, edge


@pytest.fixture
def governing_source(tmp_path, monkeypatch):
    """Point the packaged governing source at a known tmp file."""
    import kestrel_sovereign.config as ks_config

    content = b"# Test governing constitution\n\nBe honest.\n"
    path = tmp_path / "KESTREL_CONSTITUTION.md"
    path.write_bytes(content)
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(path))
    return content, hashlib.sha256(content).hexdigest()


@pytest.mark.asyncio
async def test_verifier_passes_when_all_three_proofs_hold(governing_source):
    content, anchor = governing_source
    agent, _, _ = _verifier_agent(content, anchor)
    ok, msg = await agent._verify_constitution_integrity()
    assert ok, msg


@pytest.mark.asyncio
async def test_verifier_fails_closed_on_missing_anchor(governing_source):
    content, _ = governing_source
    agent, _, _ = _verifier_agent(content, anchor=None)
    ok, msg = await agent._verify_constitution_integrity()
    assert not ok
    assert "No anchored constitution hash" in msg


@pytest.mark.asyncio
async def test_verifier_fails_closed_on_missing_blob_row(governing_source):
    content, anchor = governing_source
    agent, _, _ = _verifier_agent(content, anchor)
    agent.storage.retrieve_file = AsyncMock(return_value=None)
    ok, msg = await agent._verify_constitution_integrity()
    assert not ok
    assert "missing" in msg.lower()


@pytest.mark.asyncio
async def test_verifier_fails_closed_on_undecryptable_blob(governing_source):
    content, anchor = governing_source
    agent, _, _ = _verifier_agent(content, anchor)
    agent.storage.retrieve_file = AsyncMock(side_effect=ValueError("bad decrypt"))
    ok, msg = await agent._verify_constitution_integrity()
    assert not ok
    assert "retrieve/decrypt" in msg


@pytest.mark.asyncio
async def test_verifier_fails_closed_on_blob_digest_mismatch(governing_source):
    content, anchor = governing_source
    agent, _, _ = _verifier_agent(b"tampered blob bytes", anchor)
    ok, msg = await agent._verify_constitution_integrity()
    assert not ok
    assert "does not hash to its stored anchor" in msg


@pytest.mark.asyncio
async def test_verifier_fails_closed_on_missing_governance_edge(governing_source):
    content, anchor = governing_source
    agent, _, _ = _verifier_agent(content, anchor)
    agent.storage.get_edges_from = AsyncMock(return_value=[])
    ok, msg = await agent._verify_constitution_integrity()
    assert not ok
    assert "governed_by" in msg


@pytest.mark.asyncio
async def test_verifier_fails_closed_on_mistargeted_governance_edge(governing_source):
    content, anchor = governing_source
    agent, _, edge = _verifier_agent(content, anchor)
    edge.target_id = "0" * 64  # points at some other document
    ok, msg = await agent._verify_constitution_integrity()
    assert not ok
    assert "governed_by" in msg


@pytest.mark.asyncio
async def test_verifier_fails_closed_on_governing_source_mutation(governing_source, tmp_path):
    """Blob+edge intact but the packaged governing source changed → PROOF 3 fails."""
    content, anchor = governing_source
    agent, _, _ = _verifier_agent(content, anchor)
    import kestrel_sovereign.config as ks_config
    from pathlib import Path

    Path(ks_config.CONSTITUTION_PATH).write_bytes(b"# Mutated governing source\n")
    ok, msg = await agent._verify_constitution_integrity()
    assert not ok
    assert "modified" in msg


# ---------------------------------------------------------------------------
# #2920: an unreadable governance state is an AVAILABILITY failure. Reporting
# it as an integrity failure told an operator their constitution may have been
# tampered with while `!verify-constitution` confirmed the anchor was intact.
# ---------------------------------------------------------------------------

def _agent_with_unreadable_state():
    """An agent that fell into Safe Mode because the state could not be READ."""
    from kestrel_sdk.storage.database.interface import TransactionError
    from kestrel_sovereign.agent.constitution import ConstitutionMixin

    agent = MagicMock()
    agent._safe_mode = False
    agent._safe_mode_entered_at = None
    agent._safe_mode_reason = None
    agent.agent_id = "did:example:kestrel"
    agent._constitution_now = lambda: datetime(2026, 8, 18, tzinfo=timezone.utc)
    agent.constitution_state_unavailable_detail = (
        ConstitutionMixin.constitution_state_unavailable_detail.__get__(agent)
    )
    agent.constitution_state_access_failed = (
        ConstitutionMixin.constitution_state_access_failed.__get__(agent)
    )
    ConstitutionMixin._mark_constitution_state_unavailable(
        agent, TransactionError("database is locked")
    )
    return agent


def test_an_unreadable_state_is_reported_as_availability_not_integrity():
    """The marker records WHY, and the helper reads it back."""
    agent = _agent_with_unreadable_state()

    assert agent._safe_mode is True, "must still fail closed"
    assert agent.constitution_state_unavailable_detail() == "TransactionError"


def test_a_genuine_integrity_stop_still_reports_integrity():
    """Only a failed READ sets the marker — a wrong constitution must not."""
    from kestrel_sovereign.agent.constitution import ConstitutionMixin

    agent = MagicMock()
    agent._constitution_state_load_error = None
    agent.constitution_state_unavailable_detail = (
        ConstitutionMixin.constitution_state_unavailable_detail.__get__(agent)
    )

    assert agent.constitution_state_unavailable_detail() is None


@pytest.mark.asyncio
async def test_streamed_banner_names_availability_when_the_state_was_unreadable():
    """The primary chat path is streamed — it must not announce tampering."""
    from kestrel_sovereign.agent.streaming import StreamingMixin

    agent = _agent_with_unreadable_state()
    agent._constitution_audit_pending = False
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(
        agent
    )

    chunks = [c async for c in agent.process_input_streaming("normal prompt")]

    assert len(chunks) == 1
    banner = chunks[0]
    assert "SAFE MODE ACTIVE" in banner
    assert "TransactionError" in banner, "the operator needs the actual cause"
    assert "availability failure" in banner
    # The claim that must NOT be made: nothing here says the bytes are wrong.
    assert "due to an integrity failure" not in banner


@pytest.mark.asyncio
async def test_safe_mode_command_names_availability_when_the_state_was_unreadable():
    """`!safe-mode` is the command the banner tells operators to run."""
    from kestrel_sovereign.command_handler import CommandHandler

    agent = _agent_with_unreadable_state()
    agent._constitution_audit_pending = False
    handler = CommandHandler(agent)

    reply = await handler._cmd_safe_mode("!safe-mode")

    assert "TransactionError" in reply
    assert "availability failure" in reply
    assert "restricted due to integrity failure" not in reply


def test_a_malformed_state_row_is_not_called_an_availability_failure():
    """Only ACCESS failures may say the constitution was probably not altered.

    The marker is set from a broad ``except Exception``, which also catches a
    malformed or unsupported runtime-state row (``ValueError``). That IS a
    statement about the stored governance, so it must not inherit the
    reassurance built for database contention.
    """
    from kestrel_sovereign.agent.constitution import ConstitutionMixin

    agent = MagicMock()
    agent._safe_mode = False
    agent._safe_mode_entered_at = None
    agent._safe_mode_reason = None
    agent.agent_id = "did:example:kestrel"
    agent._constitution_now = lambda: datetime(2026, 8, 18, tzinfo=timezone.utc)
    for name in ("constitution_state_unavailable_detail",
                 "constitution_state_access_failed"):
        setattr(agent, name, getattr(ConstitutionMixin, name).__get__(agent))

    ConstitutionMixin._mark_constitution_state_unavailable(
        agent, ValueError("unsupported runtime state schema")
    )

    assert agent.constitution_state_unavailable_detail() == "ValueError"
    assert agent.constitution_state_access_failed() is False


def test_the_availability_marker_does_not_outlive_its_cause():
    """A later integrity stop must not inherit "nothing was altered".

    The marker was only ever reset at construction, so once a transient lock
    had set it, every subsequent Safe Mode entry in that process would have
    denied evidence of alteration — a false reassurance, which is worse than
    the vague message this replaced.
    """
    import asyncio

    agent = _agent_with_unreadable_state()
    assert agent.constitution_state_access_failed() is True

    agent.features = {}
    agent._constitution_state_persistence_pending = False
    agent._persist_constitution_runtime_state = AsyncMock(return_value=True)
    from kestrel_sovereign.agent.constitution import ConstitutionMixin

    asyncio.run(
        ConstitutionMixin._enter_safe_mode_locked(agent, "governing bytes changed")
    )

    assert agent._safe_mode_reason == "governing bytes changed"
    assert agent.constitution_state_unavailable_detail() is None, (
        "a named integrity cause must supersede the stale access marker"
    )
    assert agent.constitution_state_access_failed() is False


@pytest.mark.asyncio
async def test_the_availability_banner_does_not_contradict_itself():
    """Saying "not an integrity problem" then "once integrity is restored" is
    a message that argues with itself in four lines."""
    from kestrel_sovereign.agent.streaming import StreamingMixin

    agent = _agent_with_unreadable_state()
    agent._constitution_audit_pending = False
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(
        agent
    )

    banner = [c async for c in agent.process_input_streaming("normal prompt")][0]

    assert "availability failure" in banner
    assert "once integrity is restored" not in banner
    assert "once that state can be read" in banner
