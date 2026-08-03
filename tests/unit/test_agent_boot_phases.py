"""Integration-level tests for ``KestrelAgent.initialize()`` as a state machine.

These drive the REAL ``initialize()`` boundary (with the same storage /
memory / task-manager / feature doubles the existing ``TestInitialize`` suite
uses) and assert the #2522 boot-state-machine contract end to end:

* the public phase order IS the documented dependency sequence;
* a clean boot reaches ``READY`` and is idempotent;
* an injected failure at each phase boundary rolls back every resource the
  earlier phases opened (connection close / task-manager close / signal-source
  unregister / memory shutdown) and lands in the terminal ``FAILED`` state;
* a second ``initialize()`` after a failure is refused with
  ``AgentBootError`` — it never runs readiness on partial state;
* a boot cancelled mid-phase still unwinds every acquired resource;
* both the SQLite and the shared-pool PostgreSQL storage paths are exercised.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.agent.boot import AgentBootError, BootContext, BootPhaseState
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.async_database import AsyncDatabase


# Phase method names in boot order — the injected-failure matrix patches one
# of these to raise so the earlier phases run their real bodies first.
PHASE_METHODS = [
    "_boot_phase_storage_privacy",
    "_boot_phase_a2a_observability_signals",
    "_boot_phase_providers_payer_sync",
    "_boot_phase_identity_constitution_features",
    "_boot_phase_memory_bootstrap_context",
    "_boot_phase_periodic_services_readiness",
]

PHASE_NAMES = [
    "storage_privacy",
    "a2a_observability_signals",
    "providers_payer_sync",
    "identity_constitution_features",
    "memory_bootstrap_context",
    "periodic_services_readiness",
]


def _durable_backend_double() -> MagicMock:
    """Provide the transactional backend contract used during signal boot."""
    backend = MagicMock()
    backend.backend_type = "sqlite"
    backend.execute_script = AsyncMock()
    backend.execute = AsyncMock(return_value=1)
    backend.fetch_one = AsyncMock(return_value=None)
    backend.fetch_all = AsyncMock(return_value=[])

    @contextlib.asynccontextmanager
    async def transaction():
        yield

    backend.transaction = transaction
    return backend


@contextlib.contextmanager
def _boot_mocks():
    """Patch the heavy boot collaborators; yield handles for leak assertions.

    Mirrors the doubles the existing ``TestInitialize`` tests rely on: real
    ``initialize()`` runs, but storage / memory / task-manager are mocks so
    the whole sequence completes without a live DB or LLM. ``close`` /
    ``shutdown`` are ``AsyncMock``s so a rollback's teardown calls are
    observable.
    """
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage, patch(
        "kestrel_sovereign.kestrel_agent.discover_features", return_value=[]
    ), patch("kestrel_sovereign.kestrel_agent.verify_mandatory_feature_set"), patch(
        "kestrel_sovereign.kestrel_agent.MemorySystem"
    ) as MockMemorySystem, patch(
        "kestrel_sovereign.kestrel_agent.TaskManager"
    ) as MockTaskManager:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.add_node = AsyncMock()
        storage.db = MagicMock()
        storage.close = AsyncMock()
        storage._backend = _durable_backend_double()
        MockStorage.return_value = storage

        memory = AsyncMock()
        memory.initialize = AsyncMock()
        memory.retriever = MagicMock()
        memory.consolidator = MagicMock()
        memory.shutdown = AsyncMock()
        MockMemorySystem.return_value = memory

        task_manager = AsyncMock()
        task_manager.initialize = AsyncMock()
        task_manager.register_agent = MagicMock()
        # unregister_agent is synchronous on the real TaskManager; make the mock
        # sync too so feature teardown doesn't leave an un-awaited coroutine.
        task_manager.unregister_agent = MagicMock()
        task_manager.close = AsyncMock()
        MockTaskManager.return_value = task_manager

        yield SimpleNamespace(
            storage=storage, memory=memory, task_manager=task_manager
        )


async def _cleanup(agent: KestrelAgent) -> None:
    """Best-effort teardown of anything a mocked boot left open.

    The mocked ``TaskManager.close`` is a no-op, so the real SQLite
    observability backend opened in phase 2 must be closed explicitly (same as
    the existing feature-init suite does).
    """
    with contextlib.suppress(Exception):
        await agent.shutdown()
    obs = getattr(agent, "observability_store", None)
    backend = getattr(obs, "backend", None)
    if backend is not None:
        with contextlib.suppress(Exception):
            await backend.close()


def _make_agent(tmp_path) -> KestrelAgent:
    # No llm_service → the default LLMService is created, which passes the
    # phase-6 provider-reachability check (same as the existing init tests).
    return KestrelAgent(did="did:test:boot", storage_path=str(tmp_path / "boot.db"))


# ---------------------------------------------------------------------------
# Phase-order contract
# ---------------------------------------------------------------------------


def test_boot_phase_order_is_the_documented_dependency_sequence(tmp_path):
    agent = _make_agent(tmp_path)
    phases = agent._boot_phases()
    assert [p.name for p in phases] == PHASE_NAMES
    # The identity phase declares its durable retained resource explicitly.
    identity = next(p for p in phases if p.name == "identity_constitution_features")
    assert identity.retained  # non-empty: the durable identity graph node


# ---------------------------------------------------------------------------
# Clean boot — READY + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_boot_reaches_ready(tmp_path):
    agent = _make_agent(tmp_path)
    try:
        with _boot_mocks():
            await agent.initialize()
        assert agent._boot_state is BootPhaseState.READY
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_second_initialize_when_ready_is_a_noop(tmp_path):
    agent = _make_agent(tmp_path)
    try:
        with _boot_mocks():
            await agent.initialize()
        assert agent._boot_state is BootPhaseState.READY
        # Called again with NO mocks in scope: it must short-circuit on READY
        # before touching AsyncStorage, so this neither raises nor re-runs.
        await agent.initialize()
        assert agent._boot_state is BootPhaseState.READY
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_resume_callback_reconciles_sidecars_when_audit_persistence_fails(tmp_path):
    """Volatile handoff expiry cannot depend on `system.resumed` persistence."""
    from kestrel_sdk.signals import Status
    from kestrel_sovereign.signals.dispatcher import _TransientDurableHandoff

    agent = _make_agent(tmp_path)
    try:
        with _boot_mocks():
            await agent.initialize()

        dispatcher = agent.dispatcher
        now = datetime.now(timezone.utc)
        dispatcher._transient_durable_handoffs["expired-resume-handoff"] = (
            _TransientDurableHandoff(
                payload={"raw": "must-not-survive-resume"},
                consumer_id="workflow-wait",
                created_at=now - timedelta(minutes=2),
                retention_until=now + timedelta(days=1),
                expires_at=now - timedelta(seconds=1),
                initial_lease_token="live-only-capability",
            )
        )
        dispatcher._durable_store.persist_signal = AsyncMock(
            side_effect=RuntimeError("forced resumed-signal persistence failure")
        )
        original_dispatch_signal = dispatcher.dispatch_signal
        results = []

        async def capture_failed_dispatch(*args, **kwargs):
            result = await original_dispatch_signal(*args, **kwargs)
            results.append(result)
            return result

        dispatcher.dispatch_signal = capture_failed_dispatch

        # `dispatch_signal` encodes this persistence error as Status.FAILED;
        # it does not raise to the resume callback. The sidecar must already
        # be reconciled by the direct callback path.
        await agent.resume_monitor._on_resume(3600.0)

        assert "expired-resume-handoff" not in dispatcher._transient_durable_handoffs
        dispatcher._durable_store.persist_signal.assert_awaited_once()
        assert [result.status for result in results] == [Status.FAILED]
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Injected failure at each phase boundary — rollback + terminal FAILED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_index", list(range(len(PHASE_METHODS))))
async def test_injected_phase_failure_rolls_back_and_fails_terminally(
    tmp_path, fail_index
):
    agent = _make_agent(tmp_path)
    boom = AsyncMock(side_effect=RuntimeError(f"injected@{PHASE_NAMES[fail_index]}"))
    try:
        with _boot_mocks() as mocks:
            with patch.object(agent, PHASE_METHODS[fail_index], boom):
                with pytest.raises(RuntimeError, match="injected@"):
                    await agent.initialize()

            # Terminal state, regardless of which phase failed.
            assert agent._boot_state is BootPhaseState.FAILED

            # Storage (phase 1) committed for every failure at index >= 1, so
            # its connection must have been closed and the handle dropped.
            if fail_index >= 1:
                mocks.storage.close.assert_awaited()
                assert agent._raw_storage is None
                assert agent.storage is None
                assert agent.privacy_agent is None
            else:
                # Storage phase itself failed before opening anything.
                assert agent._raw_storage is None

            # A2A task manager (phase 2) + core signal sources.
            if fail_index >= 2:
                mocks.task_manager.close.assert_awaited()
                assert agent.task_manager is None
                # Core signal sources were unregistered on rollback.
                assert "a2a.task_complete" not in agent.signal_registry

            # Memory system (phase 5).
            if fail_index >= 5:
                mocks.memory.shutdown.assert_awaited()
                assert getattr(agent, "memory_system", None) is None

            # A retry over the rolled-back partial state is refused — readiness
            # can never run on it.
            with pytest.raises(AgentBootError, match="previously failed"):
                await agent.initialize()
            assert agent._boot_state is BootPhaseState.FAILED
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_failed_boot_never_started_periodic_services(tmp_path):
    """A failure before the readiness phase leaves no heartbeat/resume runner."""
    agent = _make_agent(tmp_path)
    boom = AsyncMock(side_effect=RuntimeError("injected@memory"))
    try:
        with _boot_mocks():
            with patch.object(
                agent, "_boot_phase_memory_bootstrap_context", boom
            ):
                with pytest.raises(RuntimeError):
                    await agent.initialize()
        # Phase 6 never ran → no periodic services exist.
        assert getattr(agent, "heartbeat_runner", None) is None
        assert getattr(agent, "resume_monitor", None) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_later_boot_failure_tears_down_durable_dispatcher_before_storage(tmp_path):
    """Phase-2 owner liveness cannot outlive a phase-3 boot failure."""
    agent = _make_agent(tmp_path)
    captured = {}

    async def fail_after_dispatcher(_ctx: BootContext) -> None:
        captured["dispatcher"] = agent.dispatcher
        raise RuntimeError("injected after dispatcher initialization")

    try:
        with _boot_mocks() as mocks:
            with patch.object(
                agent, "_boot_phase_providers_payer_sync", fail_after_dispatcher
            ):
                with pytest.raises(RuntimeError, match="after dispatcher"):
                    await agent.initialize()

            dispatcher = captured["dispatcher"]
            assert agent._boot_state is BootPhaseState.FAILED
            assert agent.dispatcher is None
            assert dispatcher._runtime_owner_heartbeat_timer is None
            assert dispatcher._durable_runtime_owner_registered is False
            # The owner release happens before the phase-1 storage close in
            # LIFO rollback order, rather than leaving timer work on a closed
            # database backend.
            release_calls = [
                call
                for call in mocks.storage._backend.execute.await_args_list
                if "stopped_at" in str(call)
            ]
            assert release_calls
            mocks.storage.close.assert_awaited()
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_boot_rollback_stops_owner_registered_at_sqlite_commit_cancellation(
    tmp_path,
):
    """Boot teardown releases an owner whose registration await never returned.

    ``aiosqlite`` can complete ``commit`` on its worker before cancellation is
    raised back into the caller.  Exercise that concrete boundary, then drive
    the real boot rollback seam (rather than only dispatcher shutdown) to
    prove an ambiguous registration cannot leave a live runtime owner behind.
    """
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from kestrel_sovereign.storage.db import SQLiteBackend

    agent = _make_agent(tmp_path)
    backend = SQLiteBackend(str(tmp_path / "boot-owner-commit-boundary.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=SourceRegistry(),
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    # Keep schema setup outside the armed registration transaction.
    await dispatcher._durable_store.initialize()
    connection = backend._connection
    assert connection is not None
    original_commit = connection.commit
    original_register = dispatcher._durable_store.register_runtime_owner
    registration_in_flight = False

    async def cancel_after_committed_owner_registration():
        await original_commit()
        if registration_in_flight:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)

    async def register_with_armed_commit(*args, **kwargs):
        nonlocal registration_in_flight
        registration_in_flight = True
        try:
            return await original_register(*args, **kwargs)
        finally:
            registration_in_flight = False

    try:
        connection.commit = cancel_after_committed_owner_registration
        dispatcher._durable_store.register_runtime_owner = register_with_armed_commit
        with pytest.raises(asyncio.CancelledError):
            await dispatcher.initialize_durable_delivery()

        assert dispatcher._durable_initialized is False
        assert dispatcher._durable_runtime_owner_registered is False
        assert dispatcher._durable_runtime_owner_registration_started is True

        # This is the callback registered before durable initialization's first
        # await. It must release the ambiguous owner before boot storage closes.
        connection.commit = original_commit
        dispatcher._durable_store.register_runtime_owner = original_register
        agent.dispatcher = dispatcher
        await agent._boot_teardown_dispatcher()

        assert agent.dispatcher is None
        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None
        assert dispatcher._durable_runtime_owner_registration_started is False
    finally:
        connection.commit = original_commit
        dispatcher._durable_store.register_runtime_owner = original_register
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await backend.close()


# ---------------------------------------------------------------------------
# Failure DURING an initializer's own await — the resource the initializer
# opened before raising is still torn down (#2522 P1). Teardown is registered
# BEFORE ``initialize()``'s first await, not after it returns, so a
# TaskManager whose 3rd store fails (leaking the first two) or a storage whose
# migration fails mid-connection does not leak.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["storage", "task_manager", "memory"])
async def test_mid_initialize_failure_still_tears_down_the_resource(
    tmp_path, resource
):
    agent = _make_agent(tmp_path)
    try:
        with _boot_mocks() as mocks:
            # The resource's OWN initialize() raises partway (it may have opened
            # a connection / started a worker before raising).
            getattr(mocks, resource).initialize.side_effect = RuntimeError(
                f"injected mid-{resource}-initialize"
            )
            with pytest.raises(RuntimeError, match="injected mid-"):
                await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            # The teardown registered before the failing await ran on rollback.
            if resource == "storage":
                mocks.storage.close.assert_awaited()
                assert agent._raw_storage is None
            elif resource == "task_manager":
                # Storage (phase 1) committed and the task manager's teardown
                # fired even though its OWN initialize is what raised.
                mocks.storage.close.assert_awaited()
                mocks.task_manager.close.assert_awaited()
                assert agent.task_manager is None
            else:  # memory
                mocks.storage.close.assert_awaited()
                mocks.task_manager.close.assert_awaited()
                mocks.memory.shutdown.assert_awaited()
                assert getattr(agent, "memory_system", None) is None
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Feature-owned signal sources are unregistered when a LATER phase fails
# (#2522 P2). A feature that successfully registers a dispatcher source and is
# then rolled back must not leave its feature-bound handler in the registry.
# ---------------------------------------------------------------------------


def _fake_source_registration(name: str):
    from kestrel_sdk.signals import (
        RedactionPolicy,
        SignalMode,
        SourceRegistration,
        Trust,
    )

    async def handler(payload):
        return None

    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda p: ""),
    )


@pytest.mark.asyncio
async def test_feature_owned_signal_sources_absent_after_rollback(tmp_path):
    from types import SimpleNamespace as _NS

    from kestrel_sovereign.features.base import Feature as _SovereignFeature

    class _SourceRegisteringFeature(_SovereignFeature):
        FAKE_SOURCE = "fake.feature_source"

        tool_name = "fake_source_feature"
        tool_description = "fake source-registering feature"

        def __init__(self, agent):
            super().__init__(agent)
            self.shutdown_calls = 0

        async def initialize(self):
            from kestrel_sovereign.signals import RegistrationPolicy

            outcome = self.agent.signal_registry.register_with_policy(
                _fake_source_registration(self.FAKE_SOURCE),
                RegistrationPolicy.OPTIONAL,
            )
            # Record the newly-owned source so base shutdown unregisters it.
            self._own_signal_sources(outcome)

        async def shutdown(self):
            self.shutdown_calls += 1
            await super().shutdown()

        def get_agent_card(self):
            return _NS(name=self.name, skills=[])

    agent = _make_agent(tmp_path)
    feature_ref = {}

    def _discover(a, **_kw):
        feature_ref["feature"] = _SourceRegisteringFeature(a)
        return [feature_ref["feature"]]

    boom = AsyncMock(side_effect=RuntimeError("injected@memory"))
    try:
        with _boot_mocks():
            with patch(
                "kestrel_sovereign.kestrel_agent.discover_features",
                side_effect=_discover,
            ):
                with patch.object(
                    agent, "_boot_phase_memory_bootstrap_context", boom
                ):
                    with pytest.raises(RuntimeError, match="injected@memory"):
                        await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            feature = feature_ref["feature"]
            # The feature registered its source in phase 4, then phase 5 failed →
            # boot rollback shut the feature down → its source is unregistered.
            assert feature.shutdown_calls >= 1
            assert (
                _SourceRegisteringFeature.FAKE_SOURCE not in agent.signal_registry
            )
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_feature_hooks_and_wait_providers_absent_after_rollback(tmp_path):
    """A LATER phase failure rolls back a feature's hooks AND wait providers —
    not just its ``shutdown()`` (#2522).

    A feature registers a hook (via ``get_hooks()``, auto-registered in
    ``_register_feature``) and a ``Waitable`` provider (in
    ``post_all_features_loaded``). The old boot rollback called only
    ``feature.shutdown()``, so the hook and the ``task:``/``talon:``-style wait
    provider were left registered on a dead agent. Rollback now runs the
    canonical per-feature teardown, so the hook registry AND the wait-provider
    registry are empty afterwards.
    """
    from types import SimpleNamespace as _NS

    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
    from kestrel_sovereign.features.base import Feature as _SovereignFeature

    class _NoopHook(Hook):
        def __init__(self) -> None:
            super().__init__(
                name="fake_feature_hook", events=[HookEvent.SESSION_START]
            )

        async def execute(self, input):  # noqa: A002 - SDK signature
            return HookOutput()

    class _FakeWaitable:
        kind = "fakewait"
        signal = None

        async def poll(self, handle):  # pragma: no cover - never polled
            raise NotImplementedError

    class _HookWaitFeature(_SovereignFeature):
        FAKE_SOURCE = "fake.hookwait_source"

        tool_name = "hook_wait_feature"
        tool_description = "feature registering a hook + wait provider"

        def __init__(self, agent):
            super().__init__(agent)
            self._hook = _NoopHook()

        async def initialize(self):
            from kestrel_sovereign.signals import RegistrationPolicy

            outcome = self.agent.signal_registry.register_with_policy(
                _fake_source_registration(self.FAKE_SOURCE),
                RegistrationPolicy.OPTIONAL,
            )
            self._own_signal_sources(outcome)

        def get_hooks(self):
            return [self._hook]

        async def post_all_features_loaded(self, agent):
            # Same canonical call the real task:/talon: features use — records
            # ownership so shutdown()/rollback unregisters it.
            self._register_wait_provider(
                agent.wait_registry, _FakeWaitable(), replace=True
            )

        def get_agent_card(self):
            return _NS(name=self.name, skills=[])

    agent = _make_agent(tmp_path)
    holder: dict = {}

    def _discover(a, **_kw):
        holder["feature"] = _HookWaitFeature(a)
        return [holder["feature"]]

    boom = AsyncMock(side_effect=RuntimeError("injected@memory"))
    try:
        with _boot_mocks():
            with patch(
                "kestrel_sovereign.kestrel_agent.discover_features",
                side_effect=_discover,
            ):
                with patch.object(
                    agent, "_boot_phase_memory_bootstrap_context", boom
                ):
                    with pytest.raises(RuntimeError, match="injected@memory"):
                        await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            feature = holder["feature"]

            # The hook registered in phase 4 is gone — every event bucket empty.
            assert all(
                not hooks for hooks in agent.hooks_manager._hooks.values()
            ), "feature hook left registered after boot rollback"
            assert feature._hook not in agent.hooks_manager.get_hooks(
                HookEvent.SESSION_START
            )

            # The wait provider registered in post_all_features_loaded is gone.
            assert agent.wait_registry.kinds() == []
            assert agent.wait_registry.get("fakewait") is None

            # And its dispatcher signal source too (base shutdown path).
            assert _HookWaitFeature.FAKE_SOURCE not in agent.signal_registry
            # The feature itself was dropped from the agent.
            assert feature.name not in agent.features
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_feature_source_unregistered_when_its_own_init_fails_after_registering(
    tmp_path,
):
    """A feature that registers a source and THEN raises in initialize() is not
    left in the registry — ``_register_feature`` shuts the failing feature down
    even though it never entered ``self.features`` (#2522 P1 + P2)."""
    from kestrel_sovereign.features.base import Feature as _SovereignFeature

    class _FailAfterRegisterFeature(_SovereignFeature):
        FAKE_SOURCE = "fake.fail_after_register"

        tool_name = "fail_after_register_feature"
        tool_description = "feature that fails after registering a source"

        async def initialize(self):
            from kestrel_sovereign.signals import RegistrationPolicy

            outcome = self.agent.signal_registry.register_with_policy(
                _fake_source_registration(self.FAKE_SOURCE),
                RegistrationPolicy.OPTIONAL,
            )
            self._own_signal_sources(outcome)
            raise RuntimeError("feature init boom after registering source")

    agent = _make_agent(tmp_path)

    def _discover(a, **_kw):
        return [_FailAfterRegisterFeature(a)]

    try:
        with _boot_mocks():
            with patch(
                "kestrel_sovereign.kestrel_agent.discover_features",
                side_effect=_discover,
            ):
                with pytest.raises(RuntimeError, match="feature init boom"):
                    await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            # Non-mandatory feature: its init error propagates and rolls the boot
            # back, but its source must not survive in the registry.
            assert _FailAfterRegisterFeature.FAKE_SOURCE not in agent.signal_registry
            # It never entered self.features (init failed before assignment).
            assert "_FailAfterRegisterFeature" not in {
                type(f).__name__ for f in getattr(agent, "features", {}).values()
            }
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Cancellation mid-boot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boot_cancelled_mid_phase_unwinds_all_resources(tmp_path):
    agent = _make_agent(tmp_path)
    started = asyncio.Event()

    async def hang(ctx: BootContext) -> None:
        started.set()
        await asyncio.Event().wait()  # owns nothing new; awaits cancellation

    try:
        with _boot_mocks() as mocks:
            # Hang in the final phase, AFTER storage/a2a/memory committed.
            with patch.object(
                agent, "_boot_phase_periodic_services_readiness", hang
            ):
                task = asyncio.ensure_future(agent.initialize())
                await asyncio.wait_for(started.wait(), timeout=3.0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            assert agent._boot_state is BootPhaseState.FAILED
            # Every acquired resource released despite cancellation mid-boot.
            mocks.storage.close.assert_awaited()
            mocks.task_manager.close.assert_awaited()
            mocks.memory.shutdown.assert_awaited()
            assert agent._raw_storage is None
            assert agent.task_manager is None

            # Retry refused after a cancelled/partial boot.
            with pytest.raises(AgentBootError):
                await agent.initialize()
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Re-entrancy guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_initialize_is_refused(tmp_path):
    agent = _make_agent(tmp_path)
    entered = asyncio.Event()

    async def hang_storage(ctx: BootContext) -> None:
        entered.set()
        await asyncio.Event().wait()  # hold the boot IN_PROGRESS

    first = None
    try:
        with patch.object(agent, "_boot_phase_storage_privacy", hang_storage):
            first = asyncio.ensure_future(agent.initialize())
            await asyncio.wait_for(entered.wait(), timeout=3.0)
            assert agent._boot_state is BootPhaseState.IN_PROGRESS
            # A second concurrent call while the first is IN_PROGRESS is refused.
            with pytest.raises(AgentBootError, match="already in progress"):
                await agent.initialize()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        assert agent._boot_state is BootPhaseState.FAILED
    finally:
        if first is not None and not first.done():
            first.cancel()
            with contextlib.suppress(Exception):
                await first
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Shared-pool PostgreSQL storage path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_storage_phase_uses_shared_postgres_pool(tmp_path):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async

    credentials = await create_kestrel_identity_async(
        str(tmp_path),
        identity_method="did:pkh",
        agent_name="Shared pool semantic authority test",
    )
    pool = MagicMock()
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=str(tmp_path / "kestrel_prime.db"),
        db_backend="postgres",
        pg_pool=pool,
        database_url="postgresql://scheduler-test/kestrel",
        llm_service=MagicMock(),
    )
    assert agent.identity is not None
    ctx = BootContext()
    # ``storage.db`` is a REAL database, not a MagicMock: this agent has an
    # on-disk inception whose birth record lives in a different database, so
    # the phase now reconciles it (#2871) and a duck-typed double would fail on
    # the first await. The runtime database is deliberately a second file so
    # the reconciliation path is the one production takes.
    runtime_db = await AsyncDatabase.sqlite(str(tmp_path / "runtime.db"))
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage, patch(
        "kestrel_sovereign.storage.db.postgres.PostgresBackend"
    ) as MockPGBackend:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.db = runtime_db
        storage.close = AsyncMock()
        MockStorage.return_value = storage
        pg_backend = MagicMock()
        MockPGBackend.from_pool.return_value = pg_backend

        try:
            await agent._boot_phase_storage_privacy(ctx)
        finally:
            await runtime_db.close()

        # The shared pool was adopted (not a fresh DSN connection).
        MockPGBackend.from_pool.assert_called_once_with(
            pool,
            advisory_dsn="postgresql://scheduler-test/kestrel",
        )
        _, kwargs = MockStorage.call_args
        assert kwargs.get("backend") is pg_backend
        capability = kwargs.get("_assertion_tenant_capability")
        assert capability is not None and capability.tenant_id == agent.did
    assert agent._raw_storage is storage
    # Storage teardown was registered for reverse-order rollback.
    assert "storage" in ctx.rollback_labels


@pytest.mark.asyncio
async def test_storage_phase_keeps_pool_recipe_when_database_url_is_ambient(
    monkeypatch,
):
    """An ambient URL must not replace a custom pool's advisory connector."""

    async def custom_connector(*_args, **_kwargs):
        return None

    ssl_context = object()

    class CustomConnectorPool:
        _connect_args = ()
        _connect_kwargs = {"ssl": ssl_context}
        _connect = staticmethod(custom_connector)
        _connection_class = object
        _record_class = object

        def get_max_size(self):
            return 3

    pool = CustomConnectorPool()
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://ambient/kestrel")
    agent = KestrelAgent(
        did="did:test:pg-ambient-url",
        db_backend="postgres",
        pg_pool=pool,
        llm_service=MagicMock(),
    )
    ctx = BootContext()
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.db = MagicMock()
        storage.close = AsyncMock()
        MockStorage.return_value = storage

        await agent._boot_phase_storage_privacy(ctx)

    backend = MockStorage.call_args.kwargs["backend"]
    assert agent._database_url == "postgresql://ambient/kestrel"
    assert backend._advisory_dsn is None
    assert backend._advisory_connect_args == ()
    assert backend._advisory_connect_kwargs["connect"] is custom_connector
    assert backend._advisory_connect_kwargs["ssl"] is ssl_context


@pytest.mark.asyncio
async def test_storage_phase_supports_pool_only_postgres_embedding():
    """A pool-only embedder retains an independent scheduler gate recipe."""

    class PoolOnlyAsyncpgDouble:
        _connect_args = ("postgresql://pool-only/kestrel",)
        _connect_kwargs = {"server_settings": {"application_name": "host"}}
        _connect = staticmethod(lambda *_args, **_kwargs: None)
        _connection_class = object
        _record_class = object

        def get_max_size(self):
            return 3

    pool = PoolOnlyAsyncpgDouble()
    agent = KestrelAgent(
        did="did:test:pg-pool-only",
        db_backend="postgres",
        pg_pool=pool,
        llm_service=MagicMock(),
    )
    ctx = BootContext()
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.db = MagicMock()
        storage.close = AsyncMock()
        MockStorage.return_value = storage

        await agent._boot_phase_storage_privacy(ctx)

    backend = MockStorage.call_args.kwargs["backend"]
    assert backend._pool is pool
    assert backend._advisory_connect_args == ("postgresql://pool-only/kestrel",)
    assert backend._advisory_connect_kwargs["server_settings"] == {
        "application_name": "host"
    }
    assert backend._advisory_max_pool_size == 3


@pytest.mark.asyncio
async def test_storage_phase_uses_sqlite_by_default(tmp_path):
    agent = _make_agent(tmp_path)
    ctx = BootContext()
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.db = MagicMock()
        storage.close = AsyncMock()
        MockStorage.return_value = storage

        await agent._boot_phase_storage_privacy(ctx)

        # SQLite path: first positional arg is the storage path, no backend kw.
        args, kwargs = MockStorage.call_args
        assert args and args[0] == str(tmp_path / "boot.db")
        assert "backend" not in kwargs
    assert "storage" in ctx.rollback_labels
