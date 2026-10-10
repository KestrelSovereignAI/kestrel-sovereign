"""Production-consumer regressions for review10; no providers or stress load."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionCustody,
    bind_execution_custody, require_execution_work,
)
from tests.unit.test_execution_custody import Authority


def control(outcome, wrapped):
    error = ExecutionCommitOutcomeError(outcome)
    if wrapped:
        wrapper = RuntimeError("native wrapper")
        wrapper.__cause__ = error
        return wrapper
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("require_success", [False, True])
async def test_assistant_persist_control_wins_outer_cancellation(outcome, wrapped, require_success):
    from tests.unit.test_streaming_persist_cancellation import _make_agent_with_persist
    entered, release = asyncio.Event(), asyncio.Event()
    error = control(outcome, wrapped)

    async def persist(*args, **kwargs):
        entered.set()
        await release.wait()
        raise error

    agent = _make_agent_with_persist(persist)
    task = asyncio.create_task(agent._persist_assistant_turn_safely(
        "answer", metadata=None, session_id="session", require_success=require_success,
    ))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        release.set()
        with pytest.raises(RuntimeError) as caught:
            await task
        assert caught.value is error
        agent.observability_store.log_metric.assert_not_awaited()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["strict", "warn"])
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_response_audit_manager_never_normalizes_control(mode, outcome, wrapped):
    from kestrel_sdk.hooks.base import HookEvent
    from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook
    from kestrel_sovereign.hooks.manager import HooksManager
    from tests.unit.test_response_audit import _make_agent, _make_hook_input
    agent = _make_agent()
    error = control(outcome, wrapped)
    agent.llm_service.get_audit_response = AsyncMock(side_effect=error)
    manager = HooksManager()
    manager.register(ResponseAuditHook(agent=agent, mode=mode))
    with pytest.raises(RuntimeError) as caught:
        await manager.execute_hooks(HookEvent.POST_RESPONSE, _make_hook_input("a" * 80))
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_conversation_embedding_control_is_not_keyword_fallback(outcome, wrapped):
    from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
    store = AsyncConversationStore.__new__(AsyncConversationStore)
    error = control(outcome, wrapped)
    store._lazy_embedding_service = lambda: SimpleNamespace(aembed=AsyncMock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        await store._maybe_embed("answer")
    assert caught.value is error
    store.agent_id = "did:test:embedding"
    store._lazy_embedding_service = lambda: SimpleNamespace(aembed=AsyncMock(side_effect=ValueError("offline")))
    assert await store._maybe_embed("answer") is None


def runtime_owner():
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    owner = KestrelAgent.__new__(KestrelAgent)
    owner._execution_custody = ExecutionCustody(Authority())
    owner._runtime_publication_ready = asyncio.Event()
    owner._background_tasks = set()
    return owner


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True])
async def test_native_stream_close_distinguishes_eof_from_abort_for_queued_child(abort):
    from kestrel_sovereign.execution_custody import _ExecutionForwarder
    owner = SimpleNamespace(_execution_custody=ExecutionCustody(Authority()))
    conversation = asyncio.Lock()
    await conversation.acquire()
    child = None

    async def queued_turn():
        async with conversation:
            require_execution_work(owner)
            return "independent child turn"

    async def source():
        nonlocal child
        child = asyncio.create_task(queued_turn())
        yield "parent answer"
        if abort:
            await asyncio.Event().wait()

    stream = _ExecutionForwarder(owner, source())
    try:
        assert await anext(stream) == "parent answer"
        if not abort:
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        await stream.aclose()
        conversation.release()
        if abort:
            with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
                await child
        else:
            assert await child == "independent child turn"
    finally:
        await stream.aclose()
        if conversation.locked():
            conversation.release()
        if child is not None:
            child.cancel()
            await asyncio.gather(child, return_exceptions=True)


@pytest.mark.asyncio
async def test_actual_isolated_supervisor_survives_claim_retirement_not_generation_loss(monkeypatch, tmp_path):
    from tests.unit.test_isolated_feature_runtime import (
        FakeIsolatedClient, _initialized_host_ingress_proxy,
    )
    owner = runtime_owner()
    from kestrel_sovereign.agent.boot import BootPhase, BootPhaseState
    from kestrel_sovereign.agent.custody import ResourceCustody
    owner._boot_state = BootPhaseState.NOT_STARTED
    owner._custody = ResourceCustody()
    owner._serving_record = None
    owner._host_authority_boot_expired = False
    owner.did = "did:test:cold-runtime"
    owner.features = {}
    health = asyncio.Event()

    class Client(FakeIsolatedClient):
        async def health(self):
            health.set()
            return True

    client = Client()
    feature = None

    async def phase(ctx):
        nonlocal feature
        feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch, tmp_path, lambda **kwargs: client, agent=owner,
        )
        ctx.on_rollback("resident-feature", feature.shutdown)
        assert not health.is_set()

    owner._boot_phases = lambda: [BootPhase("resident-publication", phase)]
    try:
        with bind_execution_custody(Authority()):
            await owner.initialize()
            assert owner._boot_state is BootPhaseState.READY
            assert owner._runtime_publication_ready.is_set()
        await asyncio.wait_for(health.wait(), 3)
        assert not feature._supervision_task.done()
        assert not client.stopped
        owner._execution_custody.revoke("cold runtime replaced")
        with pytest.raises(ExecutionAuthorityError, match="cold runtime replaced"):
            await asyncio.wait_for(asyncio.shield(feature._supervision_task), 3)
        assert client.stopped
        assert feature._traffic_gate.sealed
    finally:
        if feature is not None:
            await feature.shutdown()
        for task in tuple(owner._background_tasks):
            task.cancel()
        await asyncio.gather(*tuple(owner._background_tasks), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_supplied", [False, True])
@pytest.mark.parametrize("service_supplied", [False, True])
async def test_default_llm_boot_binds_storage_native_usage_before_first_consumer(monkeypatch, tmp_path, pool_supplied, service_supplied):
    import kestrel_sovereign.kestrel_agent as agent_module
    from kestrel_sovereign.agent.boot import BootContext
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.storage.db.postgres import PostgresBackend
    scope = ExecutionCustody(Authority())
    monkeypatch.setattr(agent_module, "_resolve_authenticated_agent_assertion_capability", lambda *args: None)
    from kestrel_sovereign.llm.service import LLMService
    service = LLMService(agent_data_dir=tmp_path) if service_supplied else None
    agent = KestrelAgent(
        did="did:test:default-native-usage", storage_path=str(tmp_path / "agent.db"), db_backend="postgres",
        database_url="postgresql://disposable.invalid/local",
        pg_pool=object() if pool_supplied else None, execution_custody=scope, llm_service=service,
    )
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._execution_custody = scope
    db = SimpleNamespace(backend=backend, backend_type="postgres")

    class Storage:
        def __init__(self, *args, **kwargs):
            self.db = db

        async def initialize(self):
            pass

    class ReachedFirstConsumer(Exception):
        pass

    def privacy_consumer(*args):
        assert agent.llm_service._usage_db is db
        assert agent.llm_service._db_initialized
        assert not agent.llm_service._usage_db_owned
        raise ReachedFirstConsumer

    monkeypatch.setattr(agent_module, "AsyncStorage", Storage)
    monkeypatch.setattr(agent_module, "PrivacyEnforcingStorage", privacy_consumer)
    agent._record_serving = Mock()
    agent._build_shared_pool_postgres_backend = Mock(return_value=backend)
    with pytest.raises(ReachedFirstConsumer):
        await agent._boot_phase_storage_privacy(BootContext())


@pytest.mark.asyncio
async def test_published_source_callback_is_new_runtime_ingress_not_retired_boot_turn(monkeypatch, tmp_path):
    from tests.unit.test_isolated_feature_runtime import FakeIsolatedClient, _initialized_host_ingress_proxy
    owner = runtime_owner()
    owner.did = "did:test:source-runtime"
    owner.features = {}
    owner._runtime_publication_ready.set()
    client = FakeIsolatedClient()
    proceed = asyncio.Event()
    feature = task = None

    async def reader():
        await proceed.wait()
        await client.event_handler({"type": "channel.link_cleared", "payload": {}})

    try:
        with bind_execution_custody(Authority()):
            feature, _ = await _initialized_host_ingress_proxy(
                monkeypatch, tmp_path, lambda **kwargs: client, agent=owner,
            )
            feature._route_link_cleared = AsyncMock(side_effect=lambda *_: require_execution_work(owner))
            task = asyncio.create_task(reader())
        proceed.set()
        await asyncio.wait_for(task, 2)
        feature._route_link_cleared.assert_awaited_once()
        owner._execution_custody.revoke("source generation replaced")
        with pytest.raises(ExecutionAuthorityError, match="source generation replaced"):
            await client.event_handler({"type": "channel.link_cleared", "payload": {}})
        assert feature._route_link_cleared.await_count == 1
    finally:
        proceed.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if feature is not None:
            await feature.shutdown()
        for owned in tuple(owner._background_tasks):
            owned.cancel()
        await asyncio.gather(*tuple(owner._background_tasks), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_control_terminalizer_failure_or_cancel_retains_exact_cleanup_debt(cancel):
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher
    dispatcher = SignalDispatcher.__new__(SignalDispatcher)
    dispatcher._retained_cognition_control_debt = {}
    dispatcher._retained_durable_cognition_tasks = set()
    dispatcher._retained_durable_cognition_cleanup_tasks = set()
    dispatcher._durable_delivery_owner = "dispatcher:original"
    dispatcher._agent = SimpleNamespace(did="original")
    dispatcher._discard_transient_durable_handoff = Mock()
    delivery = SimpleNamespace(
        delivery_id="delivery", consumer_id="consumer", lease_token="token",
    )
    entered, release = asyncio.Event(), asyncio.Event()
    error = control("unknown", True)

    async def fail(**kwargs):
        entered.set()
        await release.wait()
        raise OSError("terminal receipt unavailable")

    backend = SimpleNamespace(backend_type="postgres", fail_cognition_delivery=AsyncMock(side_effect=fail))
    dispatcher._durable_store = SimpleNamespace(backend=backend)
    task = asyncio.create_task(dispatcher._terminalize_failed_cognition(delivery, error))
    try:
        await entered.wait()
        if cancel:
            task.cancel()
        release.set()
        with pytest.raises(RuntimeError) as caught:
            await task
        assert caught.value is error
        assert dispatcher._retained_cognition_control_debt == {"delivery": (delivery, error)}
        first_identity = backend.fail_cognition_delivery.await_args
        backend.fail_cognition_delivery = AsyncMock(return_value=True)
        await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        assert not dispatcher._retained_cognition_control_debt
        assert backend.fail_cognition_delivery.await_args == first_identity
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_cleanup_failure_keeps_owner_fenced_until_exact_retry(tmp_path):
    from tests.unit.test_durable_signal_delivery import _dispatcher, _close

    backend, agent, dispatcher = await _dispatcher(tmp_path / "control-debt.db", "did:test:debt")
    delivery = SimpleNamespace(delivery_id="original")
    error = control("unknown", True)
    dispatcher._retained_cognition_control_debt[delivery.delivery_id] = (delivery, error)
    original_apply = dispatcher._apply_cognition_control_terminal
    original_release = dispatcher._release_runtime_owner_after_shutdown
    dispatcher._apply_cognition_control_terminal = AsyncMock(side_effect=OSError("receipt unavailable"))
    dispatcher._release_runtime_owner_after_shutdown = AsyncMock(wraps=original_release)
    try:
        with pytest.raises(RuntimeError) as caught:
            await dispatcher.shutdown_durable_delivery()
        assert caught.value is error
        assert dispatcher.durable_shutdown_owner_fenced
        assert dispatcher._runtime_owner_heartbeat_timer is not None
        dispatcher._release_runtime_owner_after_shutdown.assert_not_awaited()
        assert dispatcher._retained_cognition_control_debt == {"original": (delivery, error)}

        dispatcher._apply_cognition_control_terminal = AsyncMock()
        await dispatcher.shutdown_durable_delivery()
        assert not dispatcher.durable_shutdown_owner_fenced
        assert not dispatcher._retained_cognition_control_debt
        assert dispatcher._runtime_owner_heartbeat_timer is None
        dispatcher._release_runtime_owner_after_shutdown.assert_awaited_once_with(mark_owner_stopped=True)
    finally:
        dispatcher._apply_cognition_control_terminal = AsyncMock()
        dispatcher._release_runtime_owner_after_shutdown = original_release
        await dispatcher.shutdown_durable_delivery()
        dispatcher._apply_cognition_control_terminal = original_apply
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_cleanup_heartbeat_never_reopens_ordinary_denied_work():
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher

    owner = runtime_owner()
    owner._execution_custody.revoke("uncertain original generation")
    dispatcher = SignalDispatcher.__new__(SignalDispatcher)
    dispatcher._agent = owner
    owner.did = "did:test:cleanup"
    dispatcher._durable_delivery_owner = "dispatcher:original"
    dispatcher._retained_cognition_control_debt = {"original": object()}
    dispatcher._runtime_owner_fence_lock = asyncio.Lock()
    native = SimpleNamespace(backend_type="postgres", retain_cognition_cleanup_owner=AsyncMock(return_value=True))
    dispatcher._durable_store = SimpleNamespace(backend=native, heartbeat_runtime_owner=AsyncMock())
    # Capturing cleanup metadata context cannot grant a new resident admission.
    context = dispatcher._resident_timer_context()
    with pytest.raises(ExecutionAuthorityError):
        context.run(require_execution_work, owner)
    await dispatcher._heartbeat_runtime_owner()
    native.retain_cognition_cleanup_owner.assert_awaited_once_with(
        agent_id=owner.did, owner_id="dispatcher:original",
    )
    dispatcher._durable_store.heartbeat_runtime_owner.assert_not_awaited()
    with pytest.raises(ExecutionAuthorityError):
        require_execution_work(owner)


@pytest.mark.asyncio
async def test_resume_observer_preserves_cause_chained_control():
    from kestrel_sovereign.resume_monitor import ResumeMonitor

    error = control("unknown", True)
    monitor = ResumeMonitor(on_resume=AsyncMock(side_effect=error),
                            wall_clock=lambda: 500, mono_clock=lambda: 1)
    monitor._prev_wall, monitor._prev_mono = 0, 0
    with pytest.raises(RuntimeError) as caught:
        await monitor.poll_once()
    assert caught.value is error


@pytest.mark.asyncio
async def test_enabled_resident_heartbeat_and_scheduler_loops_are_not_permanent_busy(tmp_path):
    from tests.unit.test_restart_idle_gate_booted_agent import _booted_idle_agent
    from kestrel_sovereign.features.restart_coordinator.feature import RestartCoordinatorFeature
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
    from kestrel_sovereign.heartbeat import HeartbeatConfig, HeartbeatRunner

    async with _booted_idle_agent(tmp_path) as owner:
        heartbeat = HeartbeatRunner(owner, HeartbeatConfig(
            enabled=True, interval_seconds=3600, heartbeat_file=str(tmp_path / "heartbeat.md"),
        ))
        residents = []
        entered, release = asyncio.Event(), asyncio.Event()
        runner = SchedulerRunner.__new__(SchedulerRunner)
        runner._occurrence_task_factory = owner._track_background_task
        runner._occurrences = {}
        runner._tick_started_at = None
        runner._refresh_unclaimed_admission_snapshot = Mock()
        runner._occurrence_finished = lambda task_id, _: runner._occurrences.pop(task_id)

        async def occurrence(_):
            entered.set()
            await release.wait()

        runner._run_occurrence = occurrence
        try:
            await heartbeat.start()
            for name in ("scheduler-supervisor", "scheduler-runtime-telemetry"):
                residents.append(owner._track_runtime_task(asyncio.Event().wait(), name=name))
            await asyncio.sleep(0)
            coordinator = RestartCoordinatorFeature(owner)
            assert coordinator._agent_appears_idle()["idle"]
            runner._admit_occurrence(SimpleNamespace(id="actual-work", agent_id=owner.did))
            task = runner._occurrences["actual-work"].task
            await entered.wait()
            state = coordinator._agent_appears_idle()
            assert not state["idle"]
            assert "scheduler-occurrence:actual-work" in state["reason"]
            release.set()
            await task
            await asyncio.sleep(0)
            assert coordinator._agent_appears_idle()["idle"]
        finally:
            release.set()
            for admitted in tuple(runner._occurrences.values()):
                await asyncio.gather(admitted.task, return_exceptions=True)
            for task in residents:
                task.cancel()
            await asyncio.gather(*residents, return_exceptions=True)
            await heartbeat.stop()


@pytest.mark.asyncio
async def test_reboot_resident_publication_does_not_reuse_old_ready_barrier():
    from kestrel_sovereign.agent.boot import BootPhase, BootPhaseState
    from kestrel_sovereign.agent.custody import ResourceCustody

    owner = runtime_owner()
    owner._runtime_publication_ready.set()
    owner._boot_state = BootPhaseState.NOT_STARTED
    owner._custody = ResourceCustody()
    owner._serving_record = None
    owner._host_authority_boot_expired = False
    reached = asyncio.Event()
    task = None

    async def phase(ctx):
        nonlocal task
        assert not owner._runtime_publication_ready.is_set()
        task = owner._track_runtime_task(reached.wait(), name="reboot-resident")
        await asyncio.sleep(0)
        assert not task.done()

    owner._boot_phases = lambda: [BootPhase("new-publication", phase)]
    try:
        await owner.initialize()
        assert owner._runtime_publication_ready.is_set()
        reached.set()
        await task
    finally:
        reached.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_resident_handoff_waits_publication_then_keeps_original_runtime():
    owner = runtime_owner()
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def resident():
        require_execution_work(owner)
        entered.set()
        await proceed.wait()
        require_execution_work(owner)

    task = None
    try:
        with bind_execution_custody(Authority()):
            task = owner._track_runtime_task(resident(), name="resident")
            await asyncio.sleep(0)
            assert not entered.is_set()
            owner._runtime_publication_ready.set()
        await asyncio.wait_for(entered.wait(), 2)
        owner._execution_custody.revoke("runtime replaced")
        proceed.set()
        with pytest.raises(ExecutionAuthorityError, match="runtime replaced"):
            await task
    finally:
        proceed.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_denied_parent_cannot_promote_effect_to_resident():
    owner = runtime_owner()
    reached = Mock()

    async def resident():
        reached()

    with bind_execution_custody(Authority()) as scope:
        scope.revoke("scheduler claim expired before publication")
        with pytest.raises(ExecutionAuthorityError, match="claim expired"):
            owner._track_runtime_task(resident(), name="resident")
    assert not owner._background_tasks
    reached.assert_not_called()


@pytest.mark.asyncio
async def test_unpublished_resident_is_joinable_without_running_effects():
    owner = runtime_owner()
    reached = Mock()

    async def resident():
        reached()

    task = owner._track_runtime_task(resident(), name="unpublished")
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert not owner._background_tasks
    reached.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_durable_completed_route_keeps_control_object(outcome, wrapped):
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher
    error = control(outcome, wrapped)

    async def route():
        raise error

    task = asyncio.create_task(route())
    await asyncio.gather(task, return_exceptions=True)
    with pytest.raises(RuntimeError) as caught:
        SignalDispatcher._completed_durable_cognition_result(
            task, signal=SimpleNamespace(id="signal", mode="cognition"), start=0,
        )
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_retained_cognition_control_never_requeues_even_during_shutdown(shutdown):
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher
    dispatcher = SignalDispatcher.__new__(SignalDispatcher)
    dispatcher._retained_durable_cognition_tasks = set()
    dispatcher._retained_durable_cognition_cleanup_tasks = set()
    dispatcher._durable_shutdown = shutdown
    dispatcher._terminalize_failed_cognition = AsyncMock()
    dispatcher.release_durable_delivery_after_task = AsyncMock()
    error = control("unknown", True)
    delivery = SimpleNamespace(delivery_id="delivery")
    entered, release = asyncio.Event(), asyncio.Event()

    async def route():
        entered.set()
        await release.wait()
        raise error

    task = asyncio.create_task(route())
    try:
        await entered.wait()
        dispatcher._retain_durable_cognition_task(task, delivery=delivery)
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        dispatcher._terminalize_failed_cognition.assert_awaited_once_with(delivery, error)
        dispatcher.release_durable_delivery_after_task.assert_not_awaited()
        assert not dispatcher._retained_durable_cognition_tasks
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
