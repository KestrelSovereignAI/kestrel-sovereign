"""Native subscriptions retain PostgreSQL receipts and retire actual sources.

The direct handler cases verify original in-process error identity. The loopback
case separately verifies official SDK event delivery and HTTP stream retirement.
"""

import asyncio

import asyncpg
import pytest
from kestrel_sdk.native_a2a.types import SubscribeToTaskRequest, Task, TaskState
from kestrel_sdk.native_a2a.utils.errors import UnsupportedOperationError

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.a2a.request_handler import NativeA2ARequestHandler
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    execution_commit_outcome,
    is_execution_control_error,
)
from tests.unit.test_a2a_native_client import (
    native_signing as native_signing,  # noqa: PLC0414 - pytest fixture export
)
from tests.unit.test_a2a_native_client import (
    network_peer as network_peer,  # noqa: PLC0414 - pytest fixture export
)
from tests.unit.test_a2a_native_client import (
    peer_session as peer_session,  # noqa: PLC0414 - pytest fixture export
)
from tests.unit.test_a2a_native_client import (
    recipient_identity as recipient_identity,  # noqa: PLC0414 - pytest fixture export
)
from tests.unit.test_a2a_native_client import (
    reserve,
)
from tests.unit.test_execution_custody_review43_native import carries
from tests.utils.native_a2a import (
    native_a2a_backend_factory as native_a2a_backend_factory,  # noqa: PLC0414 - pytest fixture export
)
from tests.utils.native_a2a_security import (
    admission as native_request_admission,  # noqa: F401 - pytest dependency fixture
)
from tests.utils.native_a2a_security import (
    signed_context,
)


async def opened_subscription(p, monkeypatch):
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    sources = []
    original = p.admission.manager.subscribe

    def observe(*args, **kwargs):
        source = original(*args, **kwargs)
        sources.append(source)
        return source

    monkeypatch.setattr(p.admission.manager, "subscribe", observe)
    handler = NativeA2ARequestHandler(p.admission.manager, p.admission.guard, p.session.agent_card)
    params = SubscribeToTaskRequest(id=accepted.response.id, tenant=p.admission.scope.tenant)
    context = signed_context(p.admission, params, operation="SubscribeToTask", proof_changes={
        "classical_kid": p.arguments["signer"].classical_kid,
        "pq_kid": p.arguments["signer"].pq_kid,
    })
    return accepted.response, handler.on_subscribe_to_task(params, context), sources


@pytest.mark.parametrize("route", ["terminal", "ordinary-cancel"])
async def test_native_handler_retires_original_subscription(peer_session, monkeypatch, route):
    p = peer_session
    task, stream, sources = await opened_subscription(p, monkeypatch)
    pending = None
    try:
        if route == "terminal":
            await p.admission.manager.update_status(task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=p.admission.scope.recipient_id)
            await p.admission.manager.complete_task(task.id, "done", recipient_agent_id=p.admission.scope.recipient_id)
            with pytest.raises(UnsupportedOperationError):
                await anext(stream)
        else:
            initial = await asyncio.wait_for(anext(stream), 5)
            assert isinstance(initial, Task) and initial.id == task.id
            pending = create_owned_task(anext(stream))
            async with asyncio.timeout(5):
                while not p.admission.manager._subscribers[task.id][0]._getters:
                    await asyncio.sleep(0.001)
            pending.cancel("Native consumer left after its actual snapshot")
            outcome = await asyncio.wait_for(await_owned_task(pending), 5)
            assert isinstance(outcome.error, asyncio.CancelledError)
            assert not is_execution_control_error(outcome.error)
        assert task.id not in p.admission.manager._subscribers
        assert len(sources) == 1 and all(source.ag_frame is None for source in sources)
    finally:
        if pending is not None:
            if not pending.done():
                pending.cancel()
            await await_owned_task(pending)
        await stream.aclose()


@pytest.mark.parametrize("route", ["release-error", "release-cancel", "caller-cancel"])
async def test_native_subscription_keeps_actual_postgres_release_receipt(peer_session, monkeypatch, route):
    p = peer_session
    backend = p.admission.backend
    if backend.backend_type != "postgres":
        pytest.skip("Actual transaction-bound release receipts require PostgreSQL")
    task, stream, sources = await opened_subscription(p, monkeypatch)
    read = p.admission.manager.task_store.get_for_creator
    native_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    reads, failures = [], []
    entered, release = asyncio.Event(), asyncio.Event()
    original = OSError("Actual native subscription release acknowledgement lost")
    armed = False

    async def observed_read(*args, **kwargs):
        nonlocal armed
        reads.append(asyncio.current_task())
        armed = len(reads) == 3
        try:
            return await read(*args, **kwargs)
        except BaseException as error:
            failures.append(error)
            raise
        finally:
            armed = False

    async def lost_ack(checkout, typ, error, tb):
        result = await native_exit(checkout, typ, error, tb)
        if armed and typ is None:
            if route == "caller-cancel":
                entered.set()
                await release.wait()
                return result
            if route == "release-cancel":
                raise asyncio.CancelledError("Actual native release withdrawal") from original
            raise original
        return result

    monkeypatch.setattr(p.admission.manager.task_store, "get_for_creator", observed_read)
    backend._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(p.admission.scope.recipient_id))
    pending = None
    try:
        with monkeypatch.context() as fault:
            fault.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lost_ack)
            pending = create_owned_task(anext(stream))
            if route == "caller-cancel":
                await asyncio.wait_for(entered.wait(), 5)
                pending.cancel("Consumer withdrew during its native release acknowledgement")
                await asyncio.sleep(0)
                release.set()
            outcome = await asyncio.wait_for(await_owned_task(pending), 5)
        assert len(reads) == 3 and failures
        assert is_execution_control_error(failures[0])
        assert execution_commit_outcome(failures[0]) == "committed"
        assert carries(outcome.error, failures[0]), "Native receipt became generic data or EOF"
        if route != "caller-cancel":
            assert carries(failures[0], original)
        assert task.id not in p.admission.manager._subscribers
        assert len(sources) == 1 and all(source.ag_frame is None for source in sources)
    finally:
        release.set()
        if pending is not None:
            if not pending.done():
                pending.cancel()
            await await_owned_task(pending)
        backend._execution_custody = None
        await stream.aclose()


async def test_sdk_loopback_subscription_delivers_native_terminal_and_retires(network_peer):
    p = network_peer
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    stream = p.session.subscribe(accepted.dispatch)
    try:
        first = await asyncio.wait_for(anext(stream), 5)
        assert first.task.id == accepted.response.id
        assert p.admission.manager._subscribers
        await p.admission.manager.update_status(accepted.response.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=p.admission.scope.recipient_id)
        await p.admission.manager.complete_task(accepted.response.id, "Actual native stream result", recipient_agent_id=p.admission.scope.recipient_id)
        async with asyncio.timeout(5):
            events = [event async for event in stream]
        assert events[-1].status_update.status.state == TaskState.TASK_STATE_COMPLETED
        assert events[-1].status_update.task_id == accepted.response.id
        assert not p.admission.manager._subscribers and not p.http.is_closed
    finally:
        await stream.aclose()
