"""Signed ASGI disconnect retires idle sources, not durable business tasks."""

import asyncio
import errno
import json
import os
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from google.protobuf.json_format import MessageToDict
from kestrel_sdk.native_a2a.client.interceptors import BeforeArgs
from kestrel_sdk.native_a2a.types import SubscribeToTaskRequest, TaskState
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from kestrel_sovereign.execution_custody import (
    execution_evidence_carriers,
    is_execution_control_error,
)
from tests.unit.test_a2a_native_http import wire as wire  # noqa: PLC0414
from tests.utils.native_a2a_security import PRINCIPAL, send


async def signed_native_request(wire, params, *, method, sdk_operation, rest_path, disconnected, respond):
    """Build one actual signed ASGI request for either SDK HTTP binding."""
    args = BeforeArgs(params, sdk_operation, wire.card)
    await wire.interceptor.before(args)
    headers = {**dict(wire.http.headers), **args.context.service_parameters}
    if wire.protocol == "JSONRPC":
        path = wire.binding.prefix + "/rpc"
        body = {"jsonrpc": "2.0", "id": "original-withdrawal", "method": method, "params": MessageToDict(params)}
    else:
        path = wire.binding.prefix + "/tenant-alpha/" + rest_path
        body = {} if sdk_operation == "subscribe" else MessageToDict(params)
    payload = json.dumps(body).encode()
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    app = Starlette(routes=[wire.binding.route])
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http", "path": path,
        "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "headers": [(str(k).lower().encode(), str(v).encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 1), "server": ("native-core.test", 80),
    }

    async def perform():
        await app(scope, receive, respond)

    return perform


@pytest.mark.parametrize("operation", ["send", "subscribe"])
@pytest.mark.parametrize("retirement", ["clean", "blocked", "failed"])
async def test_actual_disconnect_wakes_idle_subscription_and_joins_original_finally(
    wire, monkeypatch, tmp_path, operation, retirement
):
    first, disconnected = asyncio.Event(), asyncio.Event()
    closing, release = asyncio.Event(), asyncio.Event()
    sources, descriptors, physical, failures, statuses = [], [], [], [], []
    original_subscribe = wire.manager.subscribe

    async def witnessed_subscription(*args, **kwargs):
        source = original_subscribe(*args, **kwargs)
        sources.append(source)
        fd = os.open(tmp_path / "original-source", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        descriptors.append(fd)
        try:
            async for event in source:
                yield event
        finally:
            await source.aclose()
            closing.set()
            if retirement != "clean":
                await release.wait()
            os.close(fd)
            if retirement == "failed":
                try:
                    os.fsync(fd)
                except OSError as error:
                    physical.append(error)
                    raise

    monkeypatch.setattr(wire.manager, "subscribe", witnessed_subscription)
    if operation == "subscribe":
        task = await wire.manager.create_task(send(wire.admission), creator_agent_id=PRINCIPAL)
        params = SubscribeToTaskRequest(id=task.id, tenant=wire.admission.scope.tenant)
        method, sdk_operation = "SubscribeToTask", "subscribe"
        rest_path = f"tasks/{task.id}:subscribe"
    else:
        params = send(wire.admission)
        method, sdk_operation = "SendStreamingMessage", "send_message_streaming"
        rest_path = "message:stream"
    async def respond(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])
        if message["type"] == "http.response.body" and b"data:" in message.get("body", b""):
            first.set()

    perform = await signed_native_request(
        wire, params, method=method, sdk_operation=sdk_operation, rest_path=rest_path,
        disconnected=disconnected, respond=respond,
    )

    async def actual_request():
        try:
            await perform()
        except BaseException as error:
            failures.append(error)
            raise

    request_task = asyncio.create_task(actual_request())
    try:
        async with asyncio.timeout(5):
            await first.wait()
            while not any(q._getters for queues in wire.manager._subscribers.values() for q in queues):
                await asyncio.sleep(0)
        rows = await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks")
        assert len(rows) == 1
        task_id = rows[0][0]
        disconnected.set()
        async with asyncio.timeout(1):
            await closing.wait()
        if retirement != "clean":
            assert not request_task.done()
            os.fstat(descriptors[0])
        release.set()
        if retirement == "failed":
            with pytest.raises(BaseException) as caught:
                async with asyncio.timeout(5):
                    await asyncio.shield(request_task)
            assert is_execution_control_error(caught.value)
            assert len(physical) == 1 and physical[0].errno == errno.EBADF
            assert any(error is physical[0] for error in execution_evidence_carriers(caught.value))
        else:
            async with asyncio.timeout(5):
                await asyncio.shield(request_task)
            assert not failures
        assert statuses == [200]
        assert len(sources) == 1 and sources[0].ag_frame is None
        assert not wire.manager._subscribers and not wire.admission.lock.locked()
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
        snapshot = await wire.manager.get_task_snapshot(task_id, recipient_agent_id=wire.admission.scope.recipient_id)
        assert snapshot.task.status.state == TaskState.TASK_STATE_SUBMITTED
    finally:
        # Test-only recovery for the known broken implementation: a real task
        # event unblocks its SAME original queue, never Task.cancel or a fake
        # close receipt. This branch cannot make the assertions above pass.
        disconnected.set()
        release.set()
        if not request_task.done():
            rows = await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks")
            if rows:
                await wire.manager.complete_task(rows[0][0], "test-only original release", recipient_agent_id=wire.admission.scope.recipient_id)
        async with asyncio.timeout(5):
            await asyncio.gather(request_task, return_exceptions=True)


@pytest.mark.parametrize("operation", ["send", "subscribe", "blocking"])
@pytest.mark.parametrize("retirement", ["clean", "blocked", "failed"])
async def test_acquired_subscription_retires_before_failed_authority_exit_returns(
    wire, monkeypatch, tmp_path, operation, retirement
):
    original_subscribe = wire.manager.subscribe
    original_lease = wire.handler.guard.lifecycle_lease
    closing, release = asyncio.Event(), asyncio.Event()
    progressed = asyncio.Event()
    sources, wrappers, descriptors, owner_tasks, close_tasks, close_errors = [], [], [], [], [], []
    lease_error = HTTPException(403, detail="actual lease exit after subscription acquisition")
    armed = True

    async def witnessed_subscription(*args, **kwargs):
        source = original_subscribe(*args, **kwargs)
        sources.append(source)
        fd = os.open(tmp_path / "acquired-source", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        descriptors.append(fd)
        owner_tasks.append(asyncio.current_task())
        try:
            async for event in source:
                yield event
        finally:
            close_tasks.append(asyncio.current_task())
            await source.aclose()
            closing.set()
            progressed.set()
            if retirement != "clean":
                await release.wait()
            os.close(fd)
            if retirement == "failed":
                try:
                    os.fsync(fd)
                except OSError as error:
                    close_errors.append(error)
                    raise

    def acquire(*args, **kwargs):
        source = witnessed_subscription(*args, **kwargs)
        wrappers.append(source)
        return source

    @asynccontextmanager
    async def fail_original_lease_exit():
        nonlocal armed
        async with original_lease():
            yield
        if armed and wire.manager._subscribers:
            armed = False
            assert not wire.admission.lock.locked()
            raise lease_error

    monkeypatch.setattr(wire.manager, "subscribe", acquire)
    monkeypatch.setattr(wire.handler, "guard", replace(wire.handler.guard, lifecycle_lease=fail_original_lease_exit))
    if operation == "subscribe":
        task = await wire.manager.create_task(send(wire.admission), creator_agent_id=PRINCIPAL)
        params = SubscribeToTaskRequest(id=task.id, tenant=wire.admission.scope.tenant)
    else:
        params = send(wire.admission)
        params.configuration.return_immediately = False

    async def actual_request():
        client = wire.client(streaming=operation != "blocking", polling=False)
        stream = client.subscribe(params) if operation == "subscribe" else client.send_message(params)
        async for _ in stream:
            pytest.fail("Failed original acquisition lease delivered a response")

    request_task = asyncio.create_task(actual_request())
    request_task.add_done_callback(lambda _: progressed.set())
    try:
        async with asyncio.timeout(5):
            await progressed.wait()
        assert not armed, "The request never reached the injected acquisition lease exit"
        assert len(sources) == len(wrappers) == 1
        if request_task.done():
            outcome, = await asyncio.gather(request_task, return_exceptions=True)
            assert isinstance(outcome, BaseException)
            assert any(error is lease_error for error in execution_evidence_carriers(outcome))
        assert closing.is_set(), "The actual request exited without retiring its acquired source"
        if retirement != "clean":
            assert not request_task.done()
            os.fstat(descriptors[0])
        release.set()
        outcome, = await asyncio.gather(request_task, return_exceptions=True)
        assert isinstance(outcome, BaseException) and is_execution_control_error(outcome)
        assert any(error is lease_error for error in execution_evidence_carriers(outcome))
        if retirement == "failed":
            assert len(close_errors) == 1
            assert any(error is close_errors[0] for error in execution_evidence_carriers(outcome))
        assert owner_tasks == close_tasks and len(owner_tasks) == 1
        assert len(sources) == len(wrappers) == 1
        assert sources[0].ag_frame is wrappers[0].ag_frame is None
        assert not wire.manager._subscribers and not wire.admission.lock.locked()
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        release.set()
        # Test-only cleanup of leaked baseline originals, not an ACK or pass:
        # the actual failed request has already physically exited before these
        # captured abandoned generators can be retired for fixture isolation.
        async with asyncio.timeout(5):
            await asyncio.gather(request_task, return_exceptions=True)
        for source in wrappers:
            if source.ag_frame is not None:
                await source.aclose()


@pytest.mark.parametrize("trigger", ["idle", "retiring"])
@pytest.mark.parametrize("retirement", ["blocked", "failed"])
async def test_blocking_disconnect_never_cancels_original_finally(wire, monkeypatch, trigger, retirement):
    from kestrel_sovereign.execution_custody import execution_commit_outcome

    closing, release, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
    descriptors, originals, owners, retired, cancellations, failures, physical, statuses = [], [], [], [], [], [], [], []
    subscribe = wire.manager.subscribe

    async def original_subscription(*args, **kwargs):
        source = subscribe(*args, **kwargs)
        originals.append(source)
        owner = asyncio.current_task()
        owners.append(owner)
        descriptors.extend(os.pipe())
        try:
            async for event in source:
                yield event
        finally:
            await source.aclose()
            closing.set()
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                cancellations.append(error)
                raise
            assert asyncio.current_task() is owner and owner.cancelling() == 0
            for fd in descriptors:
                os.close(fd)
            retired.append(owner)
            if retirement == "failed":
                try:
                    os.fsync(descriptors[0])
                except OSError as error:
                    physical.append(error)
                    raise

    monkeypatch.setattr(wire.manager, "subscribe", original_subscription)
    params = send(wire.admission)
    params.configuration.return_immediately = False

    async def respond(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    perform = await signed_native_request(
        wire, params, method="SendMessage", sdk_operation="send_message", rest_path="message:send",
        disconnected=disconnected, respond=respond,
    )

    async def request():
        try:
            await perform()
        except BaseException as error:
            failures.append(error)
            raise

    original_request = asyncio.create_task(request())
    try:
        async with asyncio.timeout(5):
            await wire.submitted.wait()
            while not wire.manager._subscribers:
                await asyncio.sleep(0)
            if trigger == "retiring":
                await wire.manager.complete_task(
                    wire.submissions[0].task.id, "actual terminal event",
                    recipient_agent_id=wire.admission.scope.recipient_id,
                )
                await closing.wait()
        disconnected.set()
        async with asyncio.timeout(5):
            await closing.wait()
        await asyncio.wait((original_request,), timeout=0.05)
        assert not original_request.done(), "Disconnect acknowledged an unretired original finalizer"
        assert not cancellations and not retired and not statuses
        assert len(owners) == 1 and not owners[0].done() and owners[0].cancelling() == 0
        for fd in descriptors:
            os.fstat(fd)
        release.set()
        async with asyncio.timeout(5):
            outcome, = await asyncio.gather(original_request, return_exceptions=True)
        assert isinstance(outcome, BaseException) and is_execution_control_error(outcome)
        assert execution_commit_outcome(outcome) == "committed"
        # Starlette's normal Exception path emits500. Fatal BaseException
        # control aborts ASGI instead; never coerce that original graph merely
        # to manufacture an HTTP status after the client already disconnected.
        assert statuses == ([500] if isinstance(outcome, Exception) else [])
        assert retired == owners and not cancellations
        if retirement == "failed":
            assert len(physical) == 1 and physical[0].errno == errno.EBADF
            assert any(error is physical[0] for error in execution_evidence_carriers(outcome))
        assert all(source.ag_frame is None for source in originals)
        assert not wire.manager._subscribers and not wire.admission.lock.locked()
        snapshot = await wire.manager.get_task_snapshot(
            wire.submissions[0].task.id, recipient_agent_id=wire.admission.scope.recipient_id
        )
        assert snapshot.task.status.state == (
            TaskState.TASK_STATE_SUBMITTED if trigger == "idle" else TaskState.TASK_STATE_COMPLETED
        )
        for fd in descriptors:
            with pytest.raises(OSError) as closed:
                os.fstat(fd)
            assert closed.value.errno == errno.EBADF
    finally:
        # Test-only failed-baseline recovery, never a retirement ACK: release
        # the same original and explicitly reclaim ONLY its captured pipe FDs.
        disconnected.set()
        release.set()
        if not original_request.done() and not closing.is_set() and wire.submissions:
            await wire.manager.complete_task(
                wire.submissions[0].task.id, "test-only original release",
                recipient_agent_id=wire.admission.scope.recipient_id,
            )
        async with asyncio.timeout(5):
            await asyncio.gather(original_request, return_exceptions=True)
        for fd in descriptors:
            try:
                os.fstat(fd)
            except OSError as error:
                assert error.errno == errno.EBADF
            else:
                os.close(fd)
