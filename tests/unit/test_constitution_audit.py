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
    ("safe_mode", "audit_pending", "cause", "expected_reason"),
    [
        (True, False, "integrity", "integrity failure"),
        (False, True, "integrity", "required startup integrity audit"),
        # The cases the Sovereign was previously told were integrity failures.
        (True, False, "state_unavailable", "could not be read"),
        (True, False, "state_not_persisted", "could not be saved"),
        (True, False, "identity_missing", "missing agent identity record"),
        (True, False, None, "cause was not recorded"),
    ],
)
async def test_streaming_cognition_is_blocked_by_constitutional_restriction(
    safe_mode, audit_pending, cause, expected_reason
):
    """The primary streamed chat path must not bypass restored restriction.

    It must also not misdescribe it. This is the surface the Sovereign
    actually reads, and it claimed an integrity failure unconditionally —
    reporting a constitutional violation for a missing decryption key.
    """
    from kestrel_sovereign.agent.streaming import StreamingMixin

    agent = MagicMock()
    agent._safe_mode = safe_mode
    agent._constitution_audit_pending = audit_pending
    # Stated, because an unset attribute on a MagicMock is a truthy Mock and
    # would match no cause at all.
    agent._safe_mode_cause = cause
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(
        agent
    )

    chunks = [chunk async for chunk in agent.process_input_streaming("normal prompt")]

    assert len(chunks) == 1
    assert "SAFE MODE ACTIVE" in chunks[0]
    assert expected_reason in chunks[0], chunks[0]
    if cause not in ("integrity", None) and not audit_pending:
        assert "integrity failure" not in chunks[0], (
            "a non-integrity restriction was reported as a constitutional "
            "violation"
        )
        # The remediation line is part of the claim. Saying operation resumes
        # "once integrity is restored" tells the operator to go fix an
        # integrity problem that does not exist — and that line escaped two
        # separate edits unnoticed because nothing asserted it.
        assert "once integrity is restored" not in chunks[0], chunks[0]
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


@pytest.mark.asyncio
async def test_a_store_outage_does_not_overwrite_an_integrity_finding(tmp_path):
    """The clobber that made the previous attempt at this defect fail.

    An integrity failure followed by an unreadable store used to report
    "Constitution runtime state unavailable" — the availability fact taking
    the slot the integrity finding was in. Amendment III requires the
    discrepancy be reported to the Sovereign, and it had been overwritten.

    The availability fact has its own home (``_constitution_state_load_error``)
    and does not need this slot.
    """
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    try:
        await agent.enter_safe_mode("governing bytes changed")
        assert agent._safe_mode_cause == SafeModeCause.INTEGRITY.value

        agent._mark_constitution_state_unavailable(RuntimeError("disk is full"))

        assert agent._safe_mode_reason == "governing bytes changed", (
            "the store outage overwrote the integrity finding"
        )
        assert agent._safe_mode_cause == SafeModeCause.INTEGRITY.value
        # ...and the availability fact is still recorded, in its own field.
        assert agent._constitution_state_load_error == "RuntimeError"
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_a_store_outage_alone_is_not_recorded_as_an_integrity_cause(tmp_path):
    """With nothing else wrong, the cause is availability, not integrity."""
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    try:
        agent._mark_constitution_state_unavailable(RuntimeError("disk is full"))

        assert agent._safe_mode is True
        assert agent._safe_mode_cause == SafeModeCause.STATE_UNAVAILABLE.value
        assert agent._safe_mode_cause != SafeModeCause.INTEGRITY.value
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_a_recorded_cause_survives_a_restart(tmp_path):
    """Persisted, not re-derived. A restart must not lose why.

    Without a stored cause every restored Safe Mode read as UNRECORDED, so a
    known integrity finding became "cause unrecorded" after a routine
    restart — the report losing the very thing Amendment III requires be
    reported.
    """
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    db = str(tmp_path / "agent.db")

    storage = AsyncStorage(db)
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    await agent._initialize_constitution_runtime_state()
    try:
        await agent.enter_safe_mode("governing bytes changed")
        assert agent._safe_mode_cause == SafeModeCause.INTEGRITY.value
    finally:
        await storage.close()

    # A second process, same database.
    storage2 = AsyncStorage(db)
    await storage2.initialize()
    restored = _DurableConstitutionHarness(storage2, now)
    try:
        await restored._initialize_constitution_runtime_state()
        assert restored._safe_mode is True
        assert restored._safe_mode_cause == SafeModeCause.INTEGRITY.value, (
            "the cause did not survive the restart"
        )
    finally:
        await storage2.close()


@pytest.mark.asyncio
async def test_a_row_written_before_causes_existed_reads_as_unrecorded(tmp_path):
    """A NULL column is not an integrity finding."""
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    db = str(tmp_path / "agent.db")

    storage = AsyncStorage(db)
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    await agent._initialize_constitution_runtime_state()
    try:
        await agent.enter_safe_mode("governing bytes changed")
        # Blank the column the way a pre-#2920 row has it.
        await storage.db.execute(
            "UPDATE constitution_runtime_state SET safe_mode_cause = NULL"
        )
    finally:
        await storage.close()

    storage2 = AsyncStorage(db)
    await storage2.initialize()
    restored = _DurableConstitutionHarness(storage2, now)
    try:
        await restored._initialize_constitution_runtime_state()
        assert restored._safe_mode is True
        assert restored._safe_mode_cause == SafeModeCause.UNRECORDED.value
    finally:
        await storage2.close()


@pytest.mark.asyncio
async def test_an_exited_restrictions_cause_is_not_revived_by_a_later_outage(tmp_path):
    """History must not be reported as a live violation.

    After an authorized exit the reason and cause linger. A later failed
    write used to carry them forward, reporting an integrity violation that
    had already been verified and cleared.
    """
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    try:
        await agent.enter_safe_mode("governing bytes changed")
        # The exit itself is exercised elsewhere; this is the state it leaves.
        agent._safe_mode = False

        agent._mark_constitution_state_unavailable(RuntimeError("disk is full"))

        assert agent._safe_mode_cause == SafeModeCause.STATE_UNAVAILABLE.value, (
            "an exited integrity finding was revived as a live cause"
        )
        assert agent._safe_mode_reason == "Constitution runtime state unavailable"
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_a_failed_write_marks_the_latch_as_not_durable(tmp_path):
    """A Safe Mode that could not be written down is not durable.

    Only a MISSING store was treated as not-persisted. When ``store.write``
    itself raised — a full disk, a dropped connection — the restriction
    existed in memory alone and would vanish on restart, while the report
    said nothing and the durability promise ("clears only with an authorized
    exit") stood.
    """
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    await agent._initialize_constitution_runtime_state()
    try:
        assert agent._constitution_state_persistence_pending is False

        # The store is present and the write fails — the case the old guard
        # could not distinguish from a healthy one.
        agent._constitution_state_store.write = AsyncMock(
            side_effect=RuntimeError("database is locked")
        )
        persisted = await agent.enter_safe_mode("governing bytes changed")

        assert persisted is False
        assert agent._safe_mode is True
        assert agent._constitution_state_persistence_pending is True, (
            "an in-memory-only restriction was reported as durable"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_a_buffered_cause_survives_the_restore_that_persists_it(tmp_path):
    """A pre-initialization entry must not be written with the old row's cause.

    Restoring the prior state overwrites the buffered cause before the
    pending entry is persisted, so the new restriction landed on disk
    carrying the previous cause or none — and reported the wrong thing after
    the next restart.
    """
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    db = str(tmp_path / "agent.db")

    # An agent with an existing, non-restricted durable row.
    storage = AsyncStorage(db)
    await storage.initialize()
    first = _DurableConstitutionHarness(storage, now)
    await first._initialize_constitution_runtime_state()
    try:
        await first._record_successful_constitution_audit(source="startup")
    finally:
        await storage.close()

    # A new process restricts cognition BEFORE its store is connected.
    storage2 = AsyncStorage(db)
    await storage2.initialize()
    agent = _DurableConstitutionHarness(storage2, now)
    try:
        await agent.enter_safe_mode(
            "decryption failures",
            cause=SafeModeCause.STATE_UNAVAILABLE.value,
        )
        assert agent._constitution_state_persistence_pending is True

        await agent._initialize_constitution_runtime_state()

        persisted = await agent._constitution_state_store.load(agent.agent_id)
        assert persisted.safe_mode is True
        assert persisted.safe_mode_cause == SafeModeCause.STATE_UNAVAILABLE.value, (
            "the buffered cause was replaced by the restored row's"
        )
    finally:
        await storage2.close()


@pytest.mark.asyncio
async def test_a_durable_write_clears_the_not_persisted_warning(tmp_path):
    """One transient failure must not mark the process forever."""
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    await agent._initialize_constitution_runtime_state()
    try:
        real_write = agent._constitution_state_store.write
        agent._constitution_state_store.write = AsyncMock(
            side_effect=RuntimeError("database is locked")
        )
        await agent.enter_safe_mode("governing bytes changed")
        assert agent._constitution_state_persistence_pending is True

        # The store recovers and a later entry writes through.
        agent._constitution_state_store.write = real_write
        await agent.enter_safe_mode("governing bytes changed")

        assert agent._constitution_state_persistence_pending is False, (
            "a recovered store still reported the latch as non-durable"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_a_missing_identity_node_is_not_reported_as_a_read_outage(tmp_path):
    """The state was read successfully; the identity it describes is gone.

    Labelling that STATE_UNAVAILABLE makes health report an outage that did
    not happen. It is a discrepancy — which Amendment III requires be
    reported — so it gets its own name rather than a neighbour's.
    """
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    db = str(tmp_path / "agent.db")

    # A completed durable row: audited, not awaiting bootstrap.
    storage = AsyncStorage(db)
    await storage.initialize()
    first = _DurableConstitutionHarness(storage, now)
    await first._initialize_constitution_runtime_state()
    try:
        await first._record_successful_constitution_audit(source="startup")
    finally:
        await storage.close()

    # Same row, but the identity node it describes is no longer there.
    storage2 = AsyncStorage(db)
    await storage2.initialize()
    agent = _DurableConstitutionHarness(storage2, now)
    try:
        await agent._initialize_constitution_runtime_state(is_new_identity=True)

        assert agent._safe_mode is True
        assert agent._safe_mode_cause == SafeModeCause.IDENTITY_MISSING.value
        assert agent._safe_mode_cause != SafeModeCause.STATE_UNAVAILABLE.value
        # The read itself succeeded, so nothing may claim otherwise.
        assert agent._constitution_state_load_error is None
    finally:
        await storage2.close()


@pytest.mark.asyncio
async def test_a_failed_write_is_not_recorded_as_a_read_outage(tmp_path):
    """Nothing failed to READ, so nothing may say the state was unreadable."""
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    await agent._initialize_constitution_runtime_state()
    try:
        agent._constitution_state_store.write = AsyncMock(
            side_effect=RuntimeError("disk is full")
        )
        # A normal-mode checkpoint, with the agent NOT restricted.
        await agent._record_successful_constitution_audit(source="startup")

        assert agent._safe_mode_cause == SafeModeCause.STATE_NOT_PERSISTED.value
        assert agent._constitution_state_load_error is None, (
            "a write failure was recorded as a read outage"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_recovery_clears_the_durability_flag_but_keeps_the_cause(tmp_path):
    """Recovery changes one of the two facts, not both.

    The flag says whether state is durable NOW; the cause says why cognition
    is restricted, and a write that failed really is that trigger. Erasing it
    to "unrecorded" on recovery lost the reason across a restart, so health
    and `!safe-mode` then claimed no cause had been recorded.
    """
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    await agent._initialize_constitution_runtime_state()
    try:
        real_write = agent._constitution_state_store.write
        agent._constitution_state_store.write = AsyncMock(
            side_effect=RuntimeError("disk is full")
        )
        await agent._record_successful_constitution_audit(source="startup")
        assert agent._safe_mode_cause == SafeModeCause.STATE_NOT_PERSISTED.value

        agent._constitution_state_store.write = real_write
        await agent._record_successful_constitution_audit(source="startup")

        assert agent._constitution_state_persistence_pending is False, (
            "the durability flag survived a successful write"
        )
        assert agent._safe_mode_cause == SafeModeCause.STATE_NOT_PERSISTED.value, (
            "the trigger was erased, so nothing records why it is restricted"
        )
    finally:
        await storage.close()


def test_the_safe_mode_command_reports_the_recorded_cause():
    """`!safe-mode` is what a Sovereign types to ask why. It must not guess."""
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.command_handler import CommandHandler

    agent = MagicMock()
    agent._safe_mode = True
    agent._constitution_audit_pending = False
    agent._safe_mode_cause = SafeModeCause.STATE_UNAVAILABLE.value

    import inspect as _inspect

    handler = CommandHandler.__new__(CommandHandler)
    handler.agent = agent
    reply = CommandHandler._cmd_safe_mode(handler, "")
    if _inspect.isawaitable(reply):
        reply = asyncio.run(reply)

    assert "SAFE MODE ACTIVE" in reply
    assert "could not be read" in reply
    assert "integrity failure" not in reply


@pytest.mark.asyncio
async def test_a_recovered_write_persists_the_trigger_it_recorded(tmp_path):
    """The durable row keeps why the agent is restricted.

    A restart reads this row, so a cause dropped here is a cause the operator
    never sees again.
    """
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    await agent._initialize_constitution_runtime_state()
    try:
        real_write = agent._constitution_state_store.write
        agent._constitution_state_store.write = AsyncMock(
            side_effect=RuntimeError("disk is full")
        )
        await agent._record_successful_constitution_audit(source="startup")
        assert agent._safe_mode_cause == SafeModeCause.STATE_NOT_PERSISTED.value

        agent._constitution_state_store.write = real_write
        await agent._record_successful_constitution_audit(source="startup")

        persisted = await agent._constitution_state_store.load(agent.agent_id)
        assert persisted.safe_mode_cause == SafeModeCause.STATE_NOT_PERSISTED.value, (
            "the durable row lost the trigger, so a restart cannot report it"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_a_blocked_command_reports_the_recorded_cause():
    """The branch an operator hits while trying to diagnose must not misreport."""
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    agent = MagicMock()
    agent._safe_mode = True
    agent._constitution_audit_pending = False
    agent._constitution_state_persistence_pending = False
    agent._safe_mode_cause = SafeModeCause.MEMORY_UNREADABLE.value
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._maybe_refresh_user_byok_resolver = AsyncMock()
    agent.process_input = KestrelAgent.process_input.__get__(agent)

    reply = await agent.process_input("!privacy")

    assert "SAFE MODE ACTIVE" in reply
    assert "could not be decrypted" in reply, reply
    assert "integrity issue" not in reply
    assert "once integrity is restored" not in reply

    # The non-command path carries the remediation suffix; it must not send
    # the operator after an integrity failure that did not happen either.
    agent2 = MagicMock()
    agent2._safe_mode = True
    agent2._constitution_audit_pending = False
    agent2._constitution_state_persistence_pending = False
    agent2._safe_mode_cause = SafeModeCause.MEMORY_UNREADABLE.value
    agent2._maybe_audit = AsyncMock()
    agent2._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent2._maybe_refresh_user_byok_resolver = AsyncMock()
    agent2.process_input = KestrelAgent.process_input.__get__(agent2)

    plain = await agent2.process_input("what is the weather")
    assert "could not be decrypted" in plain, plain
    assert "once integrity is restored" not in plain, plain


def test_a_non_durable_latch_is_disclosed_to_the_sovereign():
    """The cause slot keeps the stronger fact; the reader still needs both.

    Entering Safe Mode for an integrity failure and failing to persist it
    leaves a restriction that will not survive a restart — which health
    reported and the Sovereign-facing text did not.
    """
    from kestrel_sovereign.agent.constitution import (
        SafeModeCause,
        describe_safe_mode_restriction,
    )

    agent = MagicMock()
    agent._safe_mode_cause = SafeModeCause.INTEGRITY.value
    agent._constitution_state_persistence_pending = True

    phrase = describe_safe_mode_restriction(agent)

    assert "integrity failure" in phrase
    assert "will not survive a restart" in phrase


@pytest.mark.asyncio
async def test_a_retry_that_also_fails_still_says_not_persisted(tmp_path):
    """The replacement cause must not be applied by a write that failed."""
    from kestrel_sovereign.agent.constitution import SafeModeCause
    from kestrel_sovereign.storage import AsyncStorage

    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    agent = _DurableConstitutionHarness(storage, now)
    await agent._initialize_constitution_runtime_state()
    try:
        agent._constitution_state_store.write = AsyncMock(
            side_effect=RuntimeError("disk is full")
        )
        await agent._record_successful_constitution_audit(source="startup")
        assert agent._safe_mode_cause == SafeModeCause.STATE_NOT_PERSISTED.value

        # Still unwritable.
        await agent._record_successful_constitution_audit(source="startup")

        assert agent._safe_mode_cause == SafeModeCause.STATE_NOT_PERSISTED.value, (
            "a failed retry downgraded the cause to 'not recorded' while the "
            "store was still unwritable"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_the_cause_column_migration_runs_only_when_needed(tmp_path):
    """A fresh database already has the column; the ALTER is pure cost there.

    On hosted PostgreSQL it takes an ACCESS EXCLUSIVE lock, so agents
    starting concurrently against one database serialized on it. The metadata
    is inspected first, and a real migration failure is no longer swallowed
    by the same except that meant "already present".
    """
    from unittest.mock import patch

    from kestrel_sovereign.constitution.runtime_state import (
        ConstitutionRuntimeStateStore,
    )
    from kestrel_sovereign.storage import AsyncStorage

    storage = AsyncStorage(str(tmp_path / "agent.db"))
    await storage.initialize()
    try:
        store = ConstitutionRuntimeStateStore(storage._backend)
        await store.initialize()

        # Second startup against the same database: nothing left to add.
        store2 = ConstitutionRuntimeStateStore(storage._backend)
        with patch.object(
            store2._backend, "execute", wraps=store2._backend.execute
        ) as spy:
            await store2.initialize()
        altered = [
            c for c in spy.call_args_list
            if "ADD COLUMN safe_mode_cause" in str(c)
        ]
        assert not altered, "an unnecessary ALTER was issued on every startup"
    finally:
        await storage.close()
