"""Boot/tool consumers preserve native receipts and join borrowed DB owners."""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.agent.boot import BootContext, BootPhase, BootPhaseState, run_boot_sequence
from kestrel_sovereign.agent.custody import ReleaseOutcome, ResourceCustody
from kestrel_sovereign.execution_custody import execution_commit_outcome
from kestrel_sovereign.features.inference_lease.feature import InferenceLeaseFeature
from kestrel_sovereign.features.model.feature import ModelAgent
from kestrel_sovereign.llm.inference_leases import InferenceLeaseCoordinator
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review30 import service_owner
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_inference_lease_coordinator import _Provider, _LLMService


KINDS = ["unknown", "cancel_unknown", "group"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_boot_unwinds_independent_native_effects_and_publishes_failed(tmp_path, kind):
    agent, db = await native_owner(tmp_path)
    ctx = BootContext(custody=ResourceCustody(truthful_owners={"independent"}))
    error = receipt(kind)
    states = []

    async def inverse(value, fail=False):
        await db.execute("INSERT INTO effects VALUES(?)", (value,))
        if fail:
            raise error
        return ReleaseOutcome.RELEASED

    async def phase(context):
        context.on_rollback("independent", lambda: inverse(1))
        context.on_rollback("failed-native", lambda: inverse(2, True))
        raise ValueError("later initialization failure")

    try:
        outcome = await await_owned_task(asyncio.create_task(run_boot_sequence([BootPhase("late", phase)], ctx, states.append)))
        assert outcome.error is error, "boot replaced the original native rollback receipt"
        assert states[-1] is BootPhaseState.FAILED
        with sqlite3.connect(tmp_path / "native.db") as observer:
            assert observer.execute("SELECT id FROM effects ORDER BY id").fetchall() == [(1,), (2,)]
        assert ctx.custody.is_held("failed-native")
        assert not ctx.custody.is_held("independent")
    finally:
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_feature_rollback_continues_after_native_control(tmp_path, monkeypatch, kind):
    agent, db = await native_owner(tmp_path)
    first, second = object(), object()
    agent.features = {"first": first, "second": second}
    error = receipt(kind)

    async def unregister(feature):
        await db.execute("INSERT INTO effects VALUES(?)", (1 if feature is first else 2,))
        if feature is second:
            raise error
        return ReleaseOutcome.RELEASED

    monkeypatch.setattr(agent, "_unregister_feature_runtime", unregister)
    try:
        outcome = await await_owned_task(asyncio.create_task(agent._boot_teardown_features()))
        assert outcome.error is error
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 2
        assert agent.features == {"second": second}
    finally:
        agent.features = {}
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_dispatcher_native_receipt_is_not_a_retry_request(tmp_path, kind):
    agent, db = await native_owner(tmp_path)
    error = receipt(kind)

    class Dispatcher:
        calls = 0

        async def shutdown_durable_delivery(self):
            self.calls += 1
            if self.calls == 1:
                await db.execute("INSERT INTO effects VALUES(1)")
                raise error
            return ReleaseOutcome.RELEASED

    dispatcher = agent.dispatcher = Dispatcher()
    try:
        outcome = await await_owned_task(asyncio.create_task(agent._boot_teardown_dispatcher()))
        assert outcome.error is error
        assert dispatcher.calls == 1, "native uncertainty was retried as caller cancellation"
        assert agent.dispatcher is dispatcher
        assert await agent._boot_teardown_storage() is ReleaseOutcome.RETAINED
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        agent.dispatcher = None
        await retire(agent)


@pytest.mark.asyncio
async def test_late_boot_failure_joins_actual_llm_writer_before_borrowed_db(tmp_path, monkeypatch):
    from kestrel_sovereign.llm import service as module

    agent, db = await native_owner(tmp_path)
    service = agent.llm_service = service_owner()
    service._usage_db = db
    ctx = BootContext(custody=agent._resource_custody())
    ctx.on_rollback("llm_service", agent._boot_teardown_llm)
    ctx.on_rollback("storage", agent._boot_teardown_storage)
    monkeypatch.setattr(module, "CLIENT_CLOSE_TIMEOUT", 0.01)
    started, release, closed = (asyncio.Event() for _ in range(3))
    original_close = agent._raw_storage.close

    async def writer():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        await db.execute("INSERT INTO effects VALUES(1)")

    task = asyncio.create_task(writer())
    service._preference_persistence_tasks.add(task)

    async def close():
        assert task.done(), "borrowed DB retired ahead of LLM-owned writer"
        await original_close()
        closed.set()

    monkeypatch.setattr(agent._raw_storage, "close", close)

    async def fail(_ctx):
        raise ValueError("late failure")

    boot = None
    try:
        await started.wait()
        boot = asyncio.create_task(run_boot_sequence([BootPhase("late", fail)], ctx, lambda _state: None))
        await asyncio.sleep(0.03)
        assert not closed.is_set()
        release.set()
        outcome = await await_owned_task(boot)
        assert isinstance(outcome.error, ValueError)
        assert task.done() and closed.is_set()
        assert not service.has_pending_close_owners()
        with sqlite3.connect(tmp_path / "native.db") as observer:
            assert observer.execute("SELECT COUNT(*) FROM effects").fetchone()[0] == 1
    finally:
        release.set()
        await await_owned_task(task)
        if boot is not None:
            await await_owned_task(boot)
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ["acquire", "status", "release"])
async def test_inference_tools_preserve_native_persistence_receipts(tmp_path, monkeypatch, kind, route):
    agent, db = await native_owner(tmp_path)
    await db.execute("CREATE TABLE IF NOT EXISTS agent_metadata(agent_id TEXT,key TEXT,value TEXT,updated_at TEXT,PRIMARY KEY(agent_id,key))")
    agent.privacy_agent = SimpleNamespace(can_use_cloud=lambda: True)
    provider = _Provider("offline", total="0.20", ready_seconds=1)
    feature = InferenceLeaseFeature(agent)
    feature._owner_id = "agent-1"
    feature._db = db
    coordinator = feature._coordinator = InferenceLeaseCoordinator(llm_service=_LLMService(), owner_id="agent-1", providers={"offline": provider}, persist_state=feature._persist_state)
    if route != "acquire":
        result = await feature.acquire("qwen3:8b", "1.00", "0.50")
        assert result.error is None
    original_execute = db.execute
    error = receipt(kind)

    async def committed_receipt(*args, **kwargs):
        await original_execute(*args, **kwargs)
        raise error

    monkeypatch.setattr(db, "execute", committed_receipt)
    try:
        operation = (feature.acquire("qwen3:8b", "1.00", "0.50") if route == "acquire" else getattr(feature, route)(coordinator.current_record.lease.lease_id))
        outcome = await await_owned_task(asyncio.create_task(operation))
        assert outcome.error is not None and execution_commit_outcome(outcome.error) == "unknown", "tool normalized native persistence to a failed string"
        # The coordinator now classifies before ordinary provisioning wraps.
        # Preserve the original receipt itself, not merely an explicit cause.
        assert outcome.error is error
        with sqlite3.connect(tmp_path / "native.db") as observer:
            assert observer.execute("SELECT COUNT(*) FROM agent_metadata").fetchone()[0] == 1
        if route == "acquire":
            assert provider.acquire_calls == 0
        if route == "release":
            assert provider.release_calls == 0
    finally:
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_model_tool_keeps_completed_native_preference_receipt(tmp_path, kind):
    agent, db = await native_owner(tmp_path)
    service = service_owner()
    # Reach the native writer-drain contract under an admitted selection;
    # an empty chain now correctly refuses an agent-written override (#2652).
    service.config = {"route_priority": ["local:native"]}
    service._mandate_preference = {"model": "original", "vendor": "local"}
    agent.llm_service = service
    error = receipt(kind)

    async def writer():
        await db.execute("INSERT INTO effects VALUES(1)")
        raise error

    task = asyncio.create_task(writer())
    service._preference_persistence_tasks.add(task)
    # Leave the original task in the canonical persistence registry, as a
    # late prior writer encountered by mandate_change, not a mocked service.
    model = ModelAgent(SimpleNamespace(llm_service=service, features={}))
    await model.initialize()
    try:
        outcome = await await_owned_task(asyncio.create_task(model.set_model("local", "new")))
        assert outcome.error is error, "model-selection tool flattened native control"
        assert service._mandate_preference["model"] == "original"
        assert task.done()
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        await await_owned_task(task)
        service._preference_persistence_tasks.clear()
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_optional_enablement_ddl_original_receipt_reaches_boot(tmp_path, monkeypatch, kind):
    from kestrel_sovereign.a2a.stores.unified.feature_enablement_store import FeatureEnablementStore
    from tests.unit.test_agent_boot_phases import _make_agent, _boot_mocks, _cleanup

    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    agent = _make_agent(tmp_path)
    backend = SQLiteBackend(str(tmp_path / "boot.db"))
    await backend.connect()
    error = receipt(kind)
    original = FeatureEnablementStore.initialize

    async def actual_ddl_then_receipt(store):
        await original(store)
        raise error

    monkeypatch.setattr(FeatureEnablementStore, "initialize", actual_ddl_then_receipt)
    try:
        with _boot_mocks(real_task_manager=True, primary_backend=backend):
            await agent._boot_phase_storage_privacy(BootContext(custody=agent._resource_custody()))
            outcome = await await_owned_task(asyncio.create_task(agent._boot_phase_a2a_observability_signals(BootContext(custody=agent._resource_custody()))))
            assert outcome.error is error, "optional boot initialization discarded its DDL receipt"
            with sqlite3.connect(tmp_path / "boot.db") as observer:
                assert observer.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='feature_enablement'").fetchone()[0] == 1
    finally:
        await _cleanup(agent)
        await backend.close()
