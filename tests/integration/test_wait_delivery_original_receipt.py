"""Native wait delivery receipts across actual dispatcher/completion consumers.

The cognition input performs native TaskManager I/O, not model execution or
Agent birth/READY. Agent background completion methods are production methods.
"""

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    owned_task_error,
)
from kestrel_sovereign.a2a.types import TaskState
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.wait import build_wait_complete_registration
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
from kestrel_sovereign.waits.reconciler import register_wait_watch
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_wait_consumers_original_control import (
    wait_native_runtime as wait_native_runtime,
)
from tests.utils.original_failure import OriginalFailure


@dataclass
class NativeDispatchInput:
    did: str
    native_read: object
    ordinary_failure: bool = False
    originals: list = field(default_factory=list)
    source_snapshots: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
    _background_tasks: set = field(default_factory=set)
    _durable_persistence_gate: object = field(default_factory=DurablePersistenceGate)

    def _get_durable_persistence_gate(self):
        return self._durable_persistence_gate

    _remember_shutdown_control = KestrelAgent._remember_shutdown_control
    _finish_background_task = KestrelAgent._finish_background_task

    def _track_background_task(self, operation, *, name):
        task = KestrelAgent._track_background_task(self, operation, name=name)
        self.tasks.append(task)
        return task

    async def process_input(self, prompt, **kwargs):
        try:
            row = await self.native_read()
        except BaseException as error:
            self.source_snapshots.append(OriginalFailure.capture(error))
            self.originals.append(error)
            raise
        if self.ordinary_failure:
            raise ValueError("ordinary cognition input failure after real native read")
        return "actual native task " + row.id


@pytest.fixture
async def real_dispatch(wait_native_runtime):
    backend, manager, task, runtime, feature, errors = wait_native_runtime
    log_store = SignalLogStore(runtime._raw_storage.db)
    await log_store.initialize()
    sources = SourceRegistry()
    sources.register(build_wait_complete_registration())
    owner = NativeDispatchInput(
        runtime.did, lambda: manager.get_task_for_recipient(task.id, runtime.did)
    )
    dispatcher = SignalDispatcher(
        agent=owner,
        registry=sources,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    await dispatcher.initialize_durable_delivery()
    runtime.dispatcher = dispatcher
    try:
        yield backend, manager, task, runtime, owner, dispatcher
    finally:
        for owned in owner.tasks:
            if not owned.done():
                owned.cancel("fixture retires its actual dispatch input")
            await await_owned_task(owned)
        await dispatcher.shutdown_durable_delivery()


@pytest.mark.asyncio
async def test_actual_agent_completion_receipt_survives_wait_harvest(
    real_dispatch, monkeypatch
):
    backend, manager, task, runtime, owner, dispatcher = real_dispatch
    await manager.update_status(
        task.id, TaskState.WORKING, recipient_agent_id=runtime.did
    )
    await manager.complete_task(
        task.id, "actual final row", recipient_agent_id=runtime.did
    )
    await register_wait_watch(runtime, "task:" + task.id)
    reconciler = runtime._wait_reconciler
    entered, release = asyncio.Event(), asyncio.Event()
    native_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    faulting = False

    async def held_release(checkout, typ, error, tb):
        result = await native_exit(checkout, typ, error, tb)
        if faulting:
            entered.set()
            await release.wait()
        return result

    async def actual_cognition_read():
        nonlocal faulting
        faulting = True
        return await manager.get_task_for_recipient(task.id, runtime.did)

    owner.native_read = actual_cognition_read
    backend._execution_custody = ExecutionCustody(GenerationFence())
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", held_release)
    try:
        await reconciler.reconcile()
        handle = reconciler._pending_signal_tasks[("task", task.id)]
        await asyncio.wait_for(entered.wait(), 5)
        handle.task.cancel("actual native dispatch observer withdrawal")
        await asyncio.sleep(0)
        assert not handle.task.done()
        release.set()
        done, pending = await asyncio.wait({handle.task}, timeout=5)
        assert done and not pending
        await asyncio.sleep(0)  # Let the REAL Agent completion callback retrieve first.
        assert owner.originals and is_execution_control_error(owner.originals[0])
        original = owner.originals[0]
        assert execution_commit_outcome(original) == "committed"
        assert getattr(handle.task, "_kestrel_owned_terminal_error", None) is original
        shared = vars(owner).get("_shutdown_control")
        owner.source_snapshots[0].assert_retained(shared)
        assert execution_commit_outcome(shared) == "committed"
        failure = None
        try:
            await reconciler.reconcile()
        except BaseException as error:
            failure = error
        assert failure is not None and original in tuple(
            _explicit_control_chain(failure)
        ), (
            "wait harvest reread a cause-free cancellation instead of the actual Agent receipt"
        )
        assert execution_commit_outcome(failure) == "committed"
        assert reconciler._pending_signal_tasks[("task", task.id)] is handle
    finally:
        faulting = False
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["enqueue", "surface", "retry"])
async def test_failed_wait_classification_cannot_become_restart_loss(
    real_dispatch, monkeypatch, route
):
    backend, manager, task, runtime, owner, dispatcher = real_dispatch
    await manager.update_status(
        task.id, TaskState.WORKING, recipient_agent_id=runtime.did
    )
    await manager.complete_task(
        task.id, "actual final row", recipient_agent_id=runtime.did
    )
    await register_wait_watch(runtime, "task:" + task.id)
    reconciler = runtime._wait_reconciler
    owner.ordinary_failure = route == "retry"
    native_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    original_ack = OSError("actual classification native release acknowledgement lost")
    faulting = False

    async def fail_actual_release(checkout, typ, error, tb):
        nonlocal faulting
        result = await native_exit(checkout, typ, error, tb)
        if faulting:
            faulting = False
            raise original_ack
        return result

    monkeypatch.setattr(
        asyncpg.pool.PoolAcquireContext, "__aexit__", fail_actual_release
    )
    backend._execution_custody = ExecutionCustody(GenerationFence())
    handle, wake = None, None
    if route == "enqueue":
        original_enqueue = dispatcher.enqueue_signal

        async def native_enqueue_input(signal):
            nonlocal faulting
            faulting = True
            await manager.get_task_for_recipient(task.id, runtime.did)
            return await original_enqueue(signal)

        monkeypatch.setattr(dispatcher, "enqueue_signal", native_enqueue_input)
    else:
        await reconciler.reconcile()
        handle = reconciler._pending_signal_tasks[("task", task.id)]
        wake = reconciler._pending_signal_wakes[("task", task.id)]
        await await_owned_task(handle.task)
        assert handle.task.result().status.value == (
            "failed" if route == "retry" else "ok"
        )
        faulting = True
        metadata = create_owned_task(
            manager.get_task_for_recipient(task.id, runtime.did)
        )
        runtime.owners.append(metadata)
        await await_owned_task(metadata)
        assert is_execution_control_error(owned_task_error(metadata))
        method = "surface_record" if route == "surface" else "cognition_retry_at"
        actual_metadata = getattr(dispatcher, method)

        def original_native_metadata(signal_id):
            return metadata.result()

        monkeypatch.setattr(dispatcher, method, original_native_metadata)

    first = None
    try:
        await reconciler.reconcile()
    except BaseException as error:
        first = error
    assert first is not None and is_execution_control_error(first)
    assert original_ack in tuple(_explicit_control_chain(first))
    assert execution_commit_outcome(first) == "committed"
    before = await reconciler._store.get("task", task.id)
    assert before.pending_signal_id
    if route != "enqueue":
        assert reconciler._pending_signal_tasks.get(("task", task.id)) is handle, (
            "original completed handle was discarded before metadata classification"
        )
        assert reconciler._pending_signal_wakes.get(("task", task.id)) is wake
        monkeypatch.setattr(dispatcher, method, actual_metadata)
    else:
        monkeypatch.setattr(dispatcher, "enqueue_signal", original_enqueue)
    # Underlying callback can now succeed. That does NOT settle the original
    # native control, and the SAME singleton must not pretend it restarted.
    second = None
    try:
        await reconciler.reconcile()
    except BaseException as error:
        second = error
    after = await reconciler._store.get("task", task.id)
    assert second is first, (
        f"same-process retry erased unresolved original wait control; pending={after.pending_signal_id!r}, delivery={after.last_delivery_status!r}"
    )
    assert after.pending_signal_id == before.pending_signal_id
    assert after.last_delivery_status == before.last_delivery_status
    assert after.last_delivery_attempts == before.last_delivery_attempts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "callback_kind"),
    [
        ("poll", "sync"),
        ("poll", "async"),
        ("register", "sync"),
        ("register", "async"),
        ("privacy", "sync"),
    ],
)
@pytest.mark.parametrize("answer_kind", ["task", "future"])
async def test_durable_resume_rejects_original_borrowed_native_callback(
    real_dispatch, monkeypatch, role, callback_kind, answer_kind
):
    from kestrel_sdk.tools import Outcome, WaitStatus

    from kestrel_sovereign.execution_custody import execution_cleanup_debts
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from kestrel_sovereign.waits.reconciler import register_wait_resume_consumer

    backend, manager, task, runtime, owner, dispatcher = real_dispatch
    consumer_id = "native-resume-" + uuid4().hex
    if role == "register":
        # Genuine durable registration is the trusted input behind this
        # registration adapter; it is NOT a fabricated success receipt.
        await dispatcher.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="wait.complete",
                agent_id=runtime.did,
                correlation_selector="payload.ref=task:" + task.id,
                max_attempts=0,
            )
        )
    entered, release = asyncio.Event(), asyncio.Event()
    native_exit = asyncpg.pool.PoolAcquireContext.__aexit__

    async def hold_original_release(checkout, typ, error, tb):
        result = await native_exit(checkout, typ, error, tb)
        if not entered.is_set() and typ is None:
            entered.set()
            await release.wait()
        return result

    async def actual_native_answer():
        row = await manager.get_task_for_recipient(task.id, runtime.did)
        assert row is not None
        return (
            WaitStatus(Outcome.PENDING, "actual stored task is pending")
            if role == "poll"
            else None
        )

    backend._execution_custody = ExecutionCustody(GenerationFence())
    monkeypatch.setattr(
        asyncpg.pool.PoolAcquireContext, "__aexit__", hold_original_release
    )
    native_owner = create_owned_task(actual_native_answer())
    runtime.owners.append(native_owner)
    projected = None
    if answer_kind == "future":
        projected = asyncio.get_running_loop().create_future()

        def completed(original):
            error = owned_task_error(original)
            if error is None:
                projected.set_result(original.result())
            else:
                projected.set_exception(error)

        native_owner.add_done_callback(completed)
    borrowed = native_owner if projected is None else projected

    def sync_callback(*args):
        return borrowed

    async def async_callback(*args):
        return borrowed

    callback = sync_callback if callback_kind == "sync" else async_callback
    provider = runtime.wait_registry.get("task")
    if role == "poll":
        monkeypatch.setattr(provider, "poll", callback)
    elif role == "register":
        monkeypatch.setattr(dispatcher, "register_durable_consumer", callback)
    else:
        monkeypatch.setattr(dispatcher, "durable_payload_elided_by", callback)
    caller = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        caller = create_owned_task(
            register_wait_resume_consumer(
                runtime, "task:" + task.id, consumer_id=consumer_id
            ),
            retain_originals=True,
        )
        done, pending = await asyncio.wait({caller}, timeout=0.08)
        failure = owned_task_error(caller) if done else None
        if pending:
            # Exercise the actual old direct-await cancellation handoff.
            caller.cancel("original durable-resume observer withdrawal")
            await asyncio.sleep(0)
        assert native_owner.cancelling() == 0 and not borrowed.cancelled(), (
            "resume caller adopted/cancelled a borrowed native operation"
        )
        assert failure is not None and is_execution_control_error(failure), (
            "durable resume accepted unresolved callback work or lost its native obligation"
        )
        assert getattr(failure, "original_awaitable", None) is borrowed
        assert any(
            resource is borrowed and closer is None
            for resource, closer in execution_cleanup_debts(failure)
        )
        assert not native_owner.done() and not borrowed.done()
        reconciler = getattr(runtime, "_wait_reconciler", None)
        state = (
            await reconciler._store.get("task", task.id)
            if reconciler is not None
            else None
        )
        active = await dispatcher.has_durable_consumer(consumer_id=consumer_id)
        if role == "privacy":
            assert state is None and not active
        else:
            assert state is not None and state.watching and active
        release.set()
        outcome = await await_owned_task(native_owner)
        assert outcome.error is None
        if role == "poll":
            assert outcome.result.outcome == Outcome.PENDING
        else:
            assert outcome.result is None
    finally:
        release.set()
        if caller is not None:
            if not caller.done():
                caller.cancel("fixture retires its original resume observer")
            await await_owned_task(caller)
        await await_owned_task(native_owner)
        await asyncio.sleep(0)
        if projected is not None and projected.done() and not projected.cancelled():
            projected.exception()


@pytest.mark.asyncio
@pytest.mark.parametrize("plain_callback", ["poll", "register"])
async def test_durable_resume_accepts_plain_results_from_actual_stores(
    real_dispatch, monkeypatch, plain_callback
):
    from kestrel_sdk.tools import Outcome, WaitStatus

    from kestrel_sovereign.signals import DurableConsumerRegistration
    from kestrel_sovereign.waits.reconciler import register_wait_resume_consumer

    backend, manager, task, runtime, owner, dispatcher = real_dispatch
    await manager.update_status(
        task.id, TaskState.WORKING, recipient_agent_id=runtime.did
    )
    await manager.complete_task(
        task.id, "actual terminal resume row", recipient_agent_id=runtime.did
    )
    backend._execution_custody = ExecutionCustody(GenerationFence())
    row = await manager.get_task_for_recipient(task.id, runtime.did)
    assert row.status.state == TaskState.COMPLETED
    consumer_id = "native-plain-resume-" + uuid4().hex
    if plain_callback == "poll":
        monkeypatch.setattr(
            runtime.wait_registry.get("task"),
            "poll",
            lambda handle: WaitStatus(Outcome.DONE, "actual stored task completed"),
        )
    else:
        await dispatcher.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="wait.complete",
                agent_id=runtime.did,
                correlation_selector="payload.ref=task:" + task.id,
                max_attempts=0,
            )
        )
        monkeypatch.setattr(
            dispatcher, "register_durable_consumer", lambda registration: None
        )
    result = await register_wait_resume_consumer(
        runtime, "task:" + task.id, consumer_id=consumer_id
    )
    assert (
        result.already_terminal is not None
        and result.already_terminal.outcome == Outcome.DONE
    )
    assert await dispatcher.has_durable_consumer(consumer_id=consumer_id)
