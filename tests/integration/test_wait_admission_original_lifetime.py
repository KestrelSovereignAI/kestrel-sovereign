"""Failed native wait lifetime cannot acknowledge new public obligations."""

import asyncio
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
)
from kestrel_sovereign.a2a.types import Message, TaskSendParams, TaskState, TextPart
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    is_execution_control_error,
)
from kestrel_sovereign.waits.reconciler import (
    register_wait_resume_consumer,
    register_wait_watch,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_wait_consumers_original_control import (
    wait_native_runtime as wait_native_runtime,
)
from tests.integration.test_wait_delivery_original_receipt import (
    real_dispatch as real_dispatch,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["failed", "in-flight-failure"])
@pytest.mark.parametrize("door", ["watch", "signal", "resume"])
async def test_original_failed_wait_lifetime_rejects_public_admission(
    real_dispatch, wait_native_runtime, monkeypatch, phase, door
):
    backend, manager, first_task, runtime, owner, dispatcher = real_dispatch
    wait_feature = wait_native_runtime[4]
    await manager.update_status(
        first_task.id, TaskState.WORKING, recipient_agent_id=runtime.did
    )
    await manager.complete_task(
        first_task.id, "actual first task completed", recipient_agent_id=runtime.did
    )
    second_task = await manager.create_task(
        TaskSendParams(
            id=uuid4().hex,
            sessionId=uuid4().hex,
            message=Message(
                role="user", parts=[TextPart(text="new native public wait")]
            ),
        ),
        agent_name=runtime.did,
    )
    await register_wait_watch(runtime, "task:" + first_task.id)
    reconciler = runtime._wait_reconciler
    consumer_id = "new-native-resume-" + uuid4().hex
    entered, release = asyncio.Event(), asyncio.Event()
    native_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    real_get = manager.get_task_for_recipient
    reads = []

    async def observed_get(*args, **kwargs):
        reads.append(args[0])
        return await real_get(*args, **kwargs)

    monkeypatch.setattr(manager, "get_task_for_recipient", observed_get)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    first_tick, metadata, admitter = None, None, None
    original = None
    selected = False

    async def hold_original_ack(checkout, typ, error, tb):
        result = await native_exit(checkout, typ, error, tb)
        if selected and not entered.is_set() and typ is None:
            entered.set()
            await release.wait()
            if phase == "in-flight-failure":
                raise OSError(
                    "actual enqueue original native release acknowledgement lost"
                )
        return result

    async def admit():
        ref = "task:" + second_task.id
        if door == "watch":
            return await register_wait_watch(runtime, ref)
        if door == "signal":
            return await wait_feature.wait(target=ref, mode="signal")
        return await register_wait_resume_consumer(
            runtime, ref, consumer_id=consumer_id
        )

    try:
        if phase == "failed":
            await reconciler.reconcile()
            handle = reconciler._pending_signal_tasks[("task", first_task.id)]
            await await_owned_task(handle.task)
            selected = True
            monkeypatch.setattr(
                asyncpg.pool.PoolAcquireContext, "__aexit__", hold_original_ack
            )

            async def metadata_read():
                await manager.get_task_for_recipient(first_task.id, runtime.did)

            metadata = create_owned_task(metadata_read())
            runtime.owners.append(metadata)
            await asyncio.wait_for(entered.wait(), 5)
            actual_surface = dispatcher.surface_record
            monkeypatch.setattr(
                dispatcher, "surface_record", lambda signal_id: metadata
            )
            try:
                await reconciler.reconcile()
            except BaseException as error:
                original = error
            assert original is not None and is_execution_control_error(original)
            assert reconciler._terminal_control is original
            release.set()
            assert (await await_owned_task(metadata)).error is None
            monkeypatch.setattr(dispatcher, "surface_record", actual_surface)
            # The original borrowed owner has physically settled and native DB
            # access is healthy. Neither is an ACK reconciling the failed consumer.
            assert await backend.fetch_val("SELECT 1") == 1
        else:
            actual_enqueue = dispatcher.enqueue_signal

            async def enqueue_input(signal):
                nonlocal selected
                selected = True
                await manager.get_task_for_recipient(first_task.id, runtime.did)
                return await actual_enqueue(signal)

            monkeypatch.setattr(dispatcher, "enqueue_signal", enqueue_input)
            monkeypatch.setattr(
                asyncpg.pool.PoolAcquireContext, "__aexit__", hold_original_ack
            )
            first_tick = create_owned_task(
                reconciler.reconcile(), retain_originals=True
            )
            await asyncio.wait_for(entered.wait(), 5)
            assert not first_tick.done()

        before_reads = len(reads)
        admitter = create_owned_task(admit(), retain_originals=True)
        if phase == "in-flight-failure":
            done, pending = await asyncio.wait({admitter}, timeout=0.08)
            assert not done and pending, (
                "new wait admission bypassed the existing in-flight reconciliation lock"
            )
            assert len(reads) == before_reads, (
                "admission ran native callbacks before owning its live lifetime"
            )
            release.set()
            original = (await await_owned_task(first_tick)).error
            assert original is not None and is_execution_control_error(original)
        outcome = await await_owned_task(admitter)
        state = await reconciler._store.get("task", second_task.id)
        assert outcome.error is original, (
            f"failed reconciler acknowledged a new {door} obligation: result={outcome.result!r}, watch={state!r}"
        )
        assert len(reads) == before_reads
        assert state is None
        assert not await dispatcher.has_durable_consumer(consumer_id)
    finally:
        release.set()
        if first_tick is not None:
            await await_owned_task(first_tick)
        if admitter is not None:
            if not admitter.done():
                admitter.cancel("fixture retires its original public admission")
            await await_owned_task(admitter)
        if metadata is not None:
            await await_owned_task(metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["watch", "signal", "resume"])
async def test_first_reference_read_joins_the_same_failed_registration_lifetime(
    real_dispatch, wait_native_runtime, monkeypatch, door
):
    backend, manager, first_task, runtime, owner, dispatcher = real_dispatch
    second_task = await manager.create_task(
        TaskSendParams(
            id=uuid4().hex,
            sessionId=uuid4().hex,
            message=Message(
                role="user", parts=[TextPart(text="concurrent first native wait")]
            ),
        ),
        agent_name=runtime.did,
    )
    assert getattr(runtime, "_wait_reconciler", None) is None
    backend._execution_custody = ExecutionCustody(GenerationFence())
    entered, release = asyncio.Event(), asyncio.Event()
    native_exit = asyncpg.pool.PoolAcquireContext.__aexit__

    async def hold_first_reference_ack(checkout, typ, error, tb):
        result = await native_exit(checkout, typ, error, tb)
        if not entered.is_set() and typ is None:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(
        asyncpg.pool.PoolAcquireContext, "__aexit__", hold_first_reference_ack
    )
    consumer_id = "first-native-resume-" + uuid4().hex

    async def admit():
        ref = "task:" + first_task.id
        if door == "watch":
            return await register_wait_watch(runtime, ref)
        if door == "signal":
            return await wait_native_runtime[4].wait(target=ref, mode="signal")
        return await register_wait_resume_consumer(
            runtime, ref, consumer_id=consumer_id
        )

    caller = create_owned_task(admit(), retain_originals=True)
    borrowed = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert getattr(runtime, "_wait_reconciler", None) is None
        # A genuinely concurrent valid registration creates the SINGLE shared
        # lifetime while the first reference is still in native observation.
        await register_wait_watch(runtime, "task:" + second_task.id)
        reconciler = runtime._wait_reconciler
        borrowed = create_owned_task(
            manager.get_task_for_recipient(second_task.id, runtime.did)
        )
        runtime.owners.append(borrowed)
        assert (await await_owned_task(borrowed)).error is None
        provider = runtime.wait_registry.get("task")
        actual_poll = provider.poll
        monkeypatch.setattr(provider, "poll", lambda handle: borrowed)
        failure = None
        try:
            await reconciler.reconcile()
        except BaseException as error:
            failure = error
        assert failure is not None and is_execution_control_error(failure)
        assert reconciler._terminal_control is failure
        monkeypatch.setattr(provider, "poll", actual_poll)
        release.set()
        result = await await_owned_task(caller)
        assert result.error is failure, (
            "first-reference continuation bypassed the concurrently failed original singleton"
        )
        assert runtime._wait_reconciler is reconciler
        assert await reconciler._store.get("task", first_task.id) is None
        assert not await dispatcher.has_durable_consumer(consumer_id)
    finally:
        release.set()
        if not caller.done():
            caller.cancel("fixture retires original first-reference caller")
        await await_owned_task(caller)
        if borrowed is not None:
            await await_owned_task(borrowed)
