"""Review48 original scopes/receipts at actual consumer boundaries, offline."""
import asyncio
from contextlib import asynccontextmanager
from contextvars import Context
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    _explicit_control_chain,
    bind_execution_custody,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.delivery.feature import DeliveryFeature
from kestrel_sovereign.features.delivery.queue import DeliveryQueue
from kestrel_sovereign.features.health.feature import HealthFeature
from kestrel_sovereign.features.scheduler.runner import (
    SchedulerRunner,
    _LeaseRenewalState,
)
from kestrel_sovereign.llm.inference_leases import InferenceLeaseRecord
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_inference_lease_coordinator import (
    _coordinator,
    _Provider,
    _quote,
    _request,
)


def terminal(kind):
    return ExecutionCommitOutcomeError("committed") if kind == "committed" else receipt(kind)


class Fence:
    backend_type = "postgres"
    def require_work(self):
        pass

    async def lock_and_validate(self, connection):
        self.require_work()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["retry", "restore"])
@pytest.mark.parametrize("source", ["ambient", "retained_runtime"])
async def test_pending_inference_rechecks_original_admission_after_lock_wait(route, source):
    provider = _Provider("runpod", total="0.2", ready_seconds=1)
    coordinator, _ = _coordinator({"runpod": provider})
    request = _request()
    record = InferenceLeaseRecord("runpod", request, _quote(request))
    coordinator._record = record
    await coordinator._lock.acquire()
    entered = asyncio.Event()
    async def acquire():
        entered.set()
        if route == "retry":
            return await coordinator.acquire(request)
        return await coordinator.restore(record.to_persisted_dict())
    with bind_execution_custody(Fence()) as original:
        if source == "retained_runtime":
            coordinator._llm_service._execution_custody = original
        task = asyncio.create_task(acquire(), context=Context()) if source == "retained_runtime" else asyncio.create_task(acquire())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            with bind_execution_custody(Fence()) as replacement:
                original.revoke("original caller retired while waiting for coordinator")
                # Rebinding the mutable service must never rescue the caller
                # already waiting with its original runtime admission.
                coordinator._llm_service._execution_custody = replacement
                coordinator._lock.release()
                outcome = await await_owned_task(task)
            assert provider.acquire_calls == 0, "billable mutation preceded original authority check"
            assert is_execution_control_error(outcome.error)
        finally:
            if coordinator._lock.locked():
                coordinator._lock.release()
            if not task.done():
                task.cancel()
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "committed", "cancel_unknown", "group"])
@pytest.mark.parametrize("route", ["cancel", "lease_loss"])
async def test_scheduler_join_retains_original_native_effect_receipt(tmp_path, kind, route):
    agent, db = await native_owner(tmp_path)
    original = terminal(kind)
    entered = asyncio.Event()
    children = []
    runner = SchedulerRunner(db, agent.did, lambda *args: None)
    state = _LeaseRenewalState()
    async def dispatch():
        children.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await db.execute("INSERT INTO effects VALUES (1)")
            raise original
    parent = asyncio.create_task(runner._run_dispatch_while_lease_live(
        dispatch, SimpleNamespace(id="original-occurrence"), state,
    ))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if route == "cancel":
            parent.cancel()
        else:
            state.mark_lost()
        outcome = await asyncio.wait_for(await_owned_task(parent), 2)
        assert outcome.error is original
        assert children and all(child.done() for child in children)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        for _ in range(2):
            stopped = await await_owned_task(asyncio.create_task(runner.stop()))
            assert stopped.error is original
    finally:
        if not parent.done():
            parent.cancel()
        await await_owned_task(parent)
        for child in children:
            if not child.done():
                child.cancel()
            await await_owned_task(child)
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "committed", "cancel_unknown", "group"])
@pytest.mark.parametrize("route", ["run_once", "background"])
async def test_health_native_write_receipt_survives_completion_and_shutdown(tmp_path, monkeypatch, kind, route):
    from kestrel_sovereign.features.health import checks

    agent, db = await native_owner(tmp_path)
    await db.execute("CREATE TABLE health_log(id TEXT, agent_id TEXT, status TEXT, checks_json TEXT, overall_healthy INTEGER, created_at TEXT)")
    feature = HealthFeature(agent)
    feature._db, feature._agent_id = db, agent.did
    feature._background_task = None
    feature._health_persist_tasks = set()
    feature._in_memory_history = []
    feature._running = False
    original = terminal(kind)
    execute = db.execute
    async def healthy(*args):
        return [{"name": "database", "status": "pass"}]
    async def write_then_receipt(*args, **kwargs):
        result = await execute(*args, **kwargs)
        if "INSERT INTO health_log" in args[0]:
            raise original
        return result
    monkeypatch.setattr(checks, "run_standard_checks", healthy)
    monkeypatch.setattr(db, "execute", write_then_receipt)
    try:
        if route == "background":
            feature._interval_seconds = 0
            feature._start_background_loop()
            result = await await_owned_task(feature._background_task)
        else:
            result = await await_owned_task(asyncio.create_task(feature.run_once()))
        assert result.error is original
        assert await db.fetchval("SELECT COUNT(*) FROM health_log") == 1
        assert not feature._health_persist_tasks
        for _ in range(2):
            stopped = await await_owned_task(asyncio.create_task(feature.shutdown()))
            assert stopped.error is original
    finally:
        for task in list(feature._health_persist_tasks):
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "committed", "cancel_unknown", "group"])
@pytest.mark.parametrize("route", ["retry", "purge"])
async def test_delivery_actual_transaction_receipt_is_not_a_failed_tool_result(tmp_path, monkeypatch, kind, route):
    agent, db = await native_owner(tmp_path)
    feature = DeliveryFeature(agent)
    queue = feature._queue = DeliveryQueue(db, agent.did)
    await queue._ensure_tables()
    transaction = db.transaction
    original = terminal(kind)
    @asynccontextmanager
    async def transaction_then_receipt(*args, **kwargs):
        async with transaction(*args, **kwargs):
            yield
        raise original
    monkeypatch.setattr(db, "transaction", transaction_then_receipt)
    try:
        operation = feature.delivery_retry("missing") if route == "retry" else feature.delivery_purge()
        outcome = await await_owned_task(asyncio.create_task(operation))
        assert outcome.error is original
    finally:
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["inline_cancel", "checkpoint_failure"])
async def test_codex_native_finalization_never_downgrades_original_control(monkeypatch, route):
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter
    from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient

    monkeypatch.setattr("kestrel_sovereign.llm.codex_app_server.resolve_codex_binary", lambda: "offline-no-codex")
    app = CodexAppServerClient(binary="offline-no-codex")
    adapter = CodexAdapter()
    adapter._client = app
    entered, release = asyncio.Event(), asyncio.Event()
    readers, checkpoints = [], []
    original = receipt("cancel_unknown" if route == "checkpoint_failure" else "unknown")
    original_graph = tuple(_explicit_control_chain(original))
    checkpoint_error = OSError("ordinary checkpoint cleanup failure")
    async def nothing(*args, **kwargs):
        pass
    async def request(method, params=None, **kwargs):
        if method == "thread/start":
            return {"thread": {"id": "offline-original-thread"}}
        if method == "turn/start":
            readers.append(asyncio.create_task(app._handle_server_request(
                1, "item/tool/call", {"threadId": "offline-original-thread", "callId": "offline-effect", "tool": "effect", "arguments": {}},
            )))
            await entered.wait()
            return {"turn": {"id": "offline-original-turn"}}
        return {}
    async def execute(*args):
        entered.set()
        await release.wait()
        if route == "inline_cancel":
            raise asyncio.CancelledError("ordinary inline child cancellation")
        return {"success": True, "result": "original effect completed"}
    async def checkpoint(executed):
        checkpoints.extend(executed)
        raise checkpoint_error
    execute.persist_completed_effects = checkpoint
    async def events(*args, **kwargs):
        await entered.wait()
        release.set()
        raise original
        if False:
            yield {}
    monkeypatch.setattr(app, "ensure_started", nothing)
    monkeypatch.setattr(app, "request", request)
    monkeypatch.setattr(app, "_send", lambda _message: None)
    monkeypatch.setattr(app, "iter_turn_events", events)
    operation = asyncio.create_task(adapter.get_response(
        None, "auto", [{"role": "user", "content": "offline effect"}],
        tools=[{"type": "function", "function": {"name": "effect", "description": "offline", "parameters": {"type": "object"}}}],
        tool_executor=execute,
    ))
    try:
        outcome = await asyncio.wait_for(await_owned_task(operation), 2)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original in chain and execution_commit_outcome(outcome.error) == "unknown"
        assert tuple(_explicit_control_chain(original)) == original_graph
        assert all(task.done() for task in readers)
        assert not app._turn_sinks
        assert all(not lock.locked() for lock in adapter._thread_locks.values())
        if route == "checkpoint_failure":
            assert checkpoints, "cancellation skipped completed-effect checkpoint"
            assert checkpoint_error in chain, "checkpoint failure was discarded during original control retention"
        else:
            assert outcome.error is original
    finally:
        release.set()
        if not operation.done():
            operation.cancel()
        await await_owned_task(operation)
        for task in readers:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
