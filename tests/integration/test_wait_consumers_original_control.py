"""Actual native reads through public wait, reconcile and callback consumers.

Runtime/provider/dispatcher inputs below are structural native-I/O adapters, not
a live Agent/dispatcher, model execution, authenticated peer or READY proof.
"""

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import asyncpg
import pytest
from kestrel_sovereign._vendor.a2a.types import Message, Part, Role, SendMessageRequest, TaskState
from kestrel_sdk.signals.models import SignalHandle
from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    owned_task_error,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.features.wait.feature import WaitFeature
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import (
    WaitReconciler,
    _provider_origin_session,
    register_wait_watch,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.utils.native_a2a import manager_for


@dataclass
class NativeWaitRuntime:
    did: str
    wait_registry: WaitRegistry
    _raw_storage: AsyncStorage
    owners: list = field(default_factory=list)
    dispatcher: object = None

    def _track_background_task(self, operation, *, name):
        task = create_owned_task(operation, name=name)
        self.owners.append(task)
        return task


@pytest.fixture
async def wait_native_runtime(native_pg, tmp_path):
    backend, observer, schema = native_pg
    did = "did:web:native-wait-consumer:" + uuid4().hex
    manager = manager_for(backend, NativeRecipientScope(did), owns_backend=False)
    await manager.initialize()
    task = await manager.create_task(SendMessageRequest(message=Message(role=Role.ROLE_USER, parts=[Part(text='native consumer')], message_id=uuid4().hex, context_id=uuid4().hex)), creator_agent_id=did)
    local = SQLiteBackend(str(tmp_path / "actual-wait-consumer.db"))
    await local.connect()
    storage = AsyncStorage.from_backend(local, agent_id=did)
    await storage.initialize()
    runtime = NativeWaitRuntime(did, WaitRegistry(), storage)
    feature = TaskFeature(runtime)
    feature.set_task_manager(manager)
    await feature.initialize()
    await feature.post_all_features_loaded(runtime)
    errors = []
    wait_feature = WaitFeature(runtime)
    await wait_feature.initialize()
    try:
        yield backend, manager, task, runtime, wait_feature, errors
    finally:
        for owner in runtime.owners:
            if not owner.done():
                owner.cancel("fixture retires its own original wait callback")
            await await_owned_task(owner)
        backend._execution_custody = None
        try:
            await wait_feature.shutdown()
        except BaseException as error:
            assert errors and errors[0] in tuple(_explicit_control_chain(error))
        await feature.shutdown()
        await manager.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        "block",
        "signal-watch",
        "watch",
        "fallback-poll",
        "fallback-origin",
        "active-handles",
        "enqueue",
        "harvest",
        "retry-advice",
        "surface-ledger",
    ],
)
async def test_native_control_survives_actual_wait_consumers(
    wait_native_runtime, monkeypatch, route
):
    backend, manager, task, runtime, wait_feature, errors = wait_native_runtime
    did = runtime.did
    if route in {"fallback-origin", "enqueue", "harvest"}:
        await manager.update_status(task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=did)
        await manager.complete_task(
            task.id, "actual final task row", recipient_agent_id=did
        )

    class NativeProvider:
        kind = "native"
        signal = None

        async def owns_handle(self, handle):
            try:
                return await manager.get_task_for_recipient(handle, did) is not None
            except Exception as error:
                if route == "signal-watch":
                    raise ValueError(
                        "original native ownership callback failed"
                    ) from error
                raise

        async def poll(self, handle):
            row = await manager.get_task_for_recipient(handle, did)
            terminal = row is not None and row.status.state == TaskState.TASK_STATE_COMPLETED
            return WaitStatus(
                Outcome.DONE if terminal else Outcome.PENDING, "actual task state"
            )

        async def origin_session_id(self, handle):
            row = await manager.get_task_for_recipient(handle, did)
            return row.context_id if row is not None else None

    class NativeMonitor(NativeProvider):
        async def active_handles(self):
            row = await manager.get_task_for_recipient(task.id, did)
            return [row.id] if row is not None else []

    provider = NativeMonitor() if route == "active-handles" else NativeProvider()
    runtime.wait_registry.register(provider)
    kind = (
        "native"
        if route
        in {"signal-watch", "fallback-origin", "active-handles", "enqueue", "harvest"}
        else "task"
    )
    if route not in {
        "block",
        "signal-watch",
        "active-handles",
        "retry-advice",
        "surface-ledger",
    }:
        await register_wait_watch(runtime, kind + ":" + task.id)
    reconciler = getattr(runtime, "_wait_reconciler", None) or WaitReconciler(runtime)
    runtime._wait_reconciler = reconciler
    original_get = manager.get_task_for_recipient
    original_snapshot = manager.get_task_snapshot
    original_release = asyncpg.pool.PoolAcquireContext.__aexit__
    releases = []
    failed = asyncio.Event()
    original = OSError("actual wait consumer native release ACK lost")

    async def observe_get(*args, **kwargs):
        try:
            return await original_get(*args, **kwargs)
        except BaseException as error:
            errors.append(error)
            failed.set()
            raise

    async def observe_snapshot(*args, **kwargs):
        try:
            return await original_snapshot(*args, **kwargs)
        except BaseException as error:
            errors.append(error)
            failed.set()
            raise

    async def lose_ack(checkout, error_type, error, tb):
        result = await original_release(checkout, error_type, error, tb)
        releases.append(checkout)
        selected = (
            2
            if route == "fallback-origin"
            else 3
            if route in {"enqueue", "harvest"}
            else 1
        )
        if len(releases) == selected and error_type is None:
            raise original
        return result

    class NativeDispatcher:
        async def enqueue_signal(self, signal):
            if route == "harvest":
                owner = create_owned_task(manager.get_task_for_recipient(task.id, did))
                runtime.owners.append(owner)
                return SignalHandle(signal_id=signal.id, task=owner)
            await manager.get_task_for_recipient(task.id, did)
            raise AssertionError("native enqueue setup did not lose its actual ACK")

    if route in {"enqueue", "harvest"}:
        runtime.dispatcher = NativeDispatcher()
    backend._execution_custody = ExecutionCustody(GenerationFence())
    monkeypatch.setattr(manager, "get_task_for_recipient", observe_get)
    monkeypatch.setattr(manager, "get_task_snapshot", observe_snapshot)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    failure = None
    try:
        if route == "block":
            await runtime.wait_registry.wait(
                "task:" + task.id, timeout_seconds=0, poll_interval_seconds=1
            )
        elif route == "signal-watch":
            await wait_feature.wait(target="native:" + task.id, mode="signal")
        elif route.startswith("fallback-"):
            wait_feature._FALLBACK_RECONCILE_POLL_SECONDS = 0.005
            wait_feature._FALLBACK_RECONCILE_STALE_SECONDS = 0
            await wait_feature.post_all_features_loaded(runtime)
            owner = runtime.owners[-1]
            await asyncio.wait_for(failed.wait(), 5)
            done, pending = await asyncio.wait({owner}, timeout=0.08)
            if owner in done:
                failure = owned_task_error(owner)
            assert owner.done(), (
                "fallback driver consumed native control and stayed eligible to retry"
            )
            await asyncio.sleep(0)
            assert vars(wait_feature).get("_owned_background_control") is errors[0]
        elif route in {"retry-advice", "surface-ledger"}:
            owner = create_owned_task(manager.get_task_for_recipient(task.id, did))
            runtime.owners.append(owner)
            await asyncio.wait_for(await_owned_task(owner), 5)

            class NativeLedger:
                def cognition_retry_at(self, signal_id):
                    return owner.result()

                def surface_record(self, signal_id):
                    return owner.result()

            if route == "retry-advice":
                reconciler._cognition_retry_at(NativeLedger(), task.id)
            else:
                reconciler._resolve_visibility(
                    NativeLedger(), signal_id=task.id, bound=True
                )
        else:
            await reconciler.reconcile()
            if route == "harvest":
                await asyncio.wait_for(await_owned_task(runtime.owners[-1]), 5)
                await reconciler.reconcile()
    except BaseException as error:
        failure = error
    assert releases and errors and is_execution_control_error(errors[0])
    assert execution_commit_outcome(errors[0]) == "committed"
    assert original in tuple(_explicit_control_chain(errors[0]))
    assert failure is not None and errors[0] in tuple(
        _explicit_control_chain(failure)
    ), (
        "public wait/reconcile callback converted original native control into tool text, skip or retry"
    )
    assert execution_commit_outcome(failure) == "committed"
    if route in {"enqueue", "harvest"}:
        state = await reconciler._store.get(kind, task.id)
        assert (
            state.pending_signal_id
            and state.last_delivery_status != "dispatcher_raised"
        ), "native dispatch control was rewritten as ordinary failed delivery"


@pytest.mark.asyncio
@pytest.mark.parametrize("answer_kind", ["task", "future"])
@pytest.mark.parametrize(
    ("role", "callback_kind"),
    [
        ("ownership", "sync"),
        ("ownership", "async"),
        ("origin", "sync"),
        ("origin", "async"),
        ("retry-advice", "sync"),
        ("surface-ledger", "sync"),
    ],
)
@pytest.mark.parametrize("completion", ["answer", "native-control"])
async def test_borrowed_native_callback_is_not_an_ownership_or_session_answer(
    wait_native_runtime,
    monkeypatch,
    answer_kind,
    callback_kind,
    role,
    completion,
):
    backend, manager, task, runtime, wait_feature, errors = wait_native_runtime
    entered, release = asyncio.Event(), asyncio.Event()
    native_release = asyncpg.pool.PoolAcquireContext.__aexit__
    native_get = manager.get_task_for_recipient
    original = OSError("original borrowed lookup native ACK lost")

    async def hold_release(checkout, error_type, error, tb):
        result = await native_release(checkout, error_type, error, tb)
        if error_type is None:
            entered.set()
            await release.wait()
            if completion == "native-control":
                raise original
        return result

    async def observed_get(*args, **kwargs):
        try:
            return await native_get(*args, **kwargs)
        except BaseException as error:
            errors.append(error)
            raise

    async def native_answer():
        did = runtime.did if role == "origin" else runtime.did + ":foreign"
        row = await manager.get_task_for_recipient(task.id, did)
        if role in {"retry-advice", "surface-ledger"}:
            assert row is None
            return None
        return (
            row.context_id if role == "origin" and row is not None else row is not None
        )

    backend._execution_custody = ExecutionCustody(GenerationFence())
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", hold_release)
    monkeypatch.setattr(manager, "get_task_for_recipient", observed_get)
    native_owner = create_owned_task(native_answer())
    runtime.owners.append(native_owner)
    projected = None
    if answer_kind == "future":
        # A real projection of the original native Task's completion, not a
        # hand-completed ownership/session answer.
        projected = asyncio.get_running_loop().create_future()

        def completed(owner):
            error = owned_task_error(owner)
            if error is not None:
                projected.set_exception(error)
            else:
                projected.set_result(owner.result())

        native_owner.add_done_callback(completed)
    borrowed = projected if projected is not None else native_owner

    def synchronous_callback(handle):
        return borrowed

    async def asynchronous_callback(handle):
        return borrowed

    callback = (
        synchronous_callback if callback_kind == "sync" else asynchronous_callback
    )

    @dataclass
    class NativeLedgerAnswers:
        cognition_retry_at: object
        surface_record: object

    ledger = NativeLedgerAnswers(callback, callback)
    provider = runtime.wait_registry.get("task")
    monkeypatch.setattr(
        provider,
        "owns_handle" if role == "ownership" else "origin_session_id",
        callback,
        raising=False,
    )
    failure = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        try:
            if role == "ownership":
                await register_wait_watch(runtime, "task:" + task.id)
            elif role == "origin":
                await _provider_origin_session(provider, task.id)
            elif role == "retry-advice":
                WaitReconciler._cognition_retry_at(ledger, task.id)
            else:
                WaitReconciler._resolve_visibility(
                    ledger, signal_id=task.id, bound=True
                )
        except BaseException as error:
            failure = error
        assert not native_owner.done() and native_owner.cancelling() == 0
        assert not borrowed.done() and not borrowed.cancelled(), (
            "wait callback consumer adopted or cancelled a borrowed native owner"
        )
        watch_rows = await runtime._raw_storage.db.fetchone(
            "SELECT count(*) FROM wait_signal_state"
        )
        assert failure is not None and is_execution_control_error(failure), (
            f"pending native Task/Future was accepted as ownership or absent origin; watches={watch_rows}"
        )
        assert getattr(failure, "original_awaitable", None) is borrowed
        assert any(
            resource is borrowed and closer is None
            for resource, closer in execution_cleanup_debts(failure)
        )
        assert watch_rows == (0,)
        release.set()
        result = await asyncio.wait_for(await_owned_task(native_owner), 5)
        if completion == "native-control":
            assert result.error is not None and is_execution_control_error(result.error)
            assert execution_commit_outcome(result.error) == "committed"
            assert original in tuple(_explicit_control_chain(result.error))
        else:
            assert result.error is None
            expected = (
                task.context_id
                if role == "origin"
                else False
                if role == "ownership"
                else None
            )
            assert result.result == expected
    finally:
        release.set()
        await await_owned_task(native_owner)
        await asyncio.sleep(0)
        if projected is not None and projected.done() and not projected.cancelled():
            projected.exception()  # The fixture, not Core, owns this projection.


@pytest.mark.asyncio
async def test_native_task_row_is_not_a_boolean_ownership_answer(
    wait_native_runtime, monkeypatch
):
    backend, manager, task, runtime, wait_feature, errors = wait_native_runtime
    backend._execution_custody = ExecutionCustody(GenerationFence())
    provider = runtime.wait_registry.get("task")

    async def wrong_shape(handle):
        return await manager.get_task_for_recipient(handle, runtime.did)

    monkeypatch.setattr(provider, "owns_handle", wrong_shape)
    with pytest.raises(ValueError, match="bool or None"):
        await register_wait_watch(runtime, "task:" + task.id)
    assert await runtime._raw_storage.db.fetchone(
        "SELECT count(*) FROM wait_signal_state"
    ) == (0,)
