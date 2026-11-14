"""Signed ASGI disconnect retires idle sources, not durable business tasks."""

import asyncio
import errno
import json
import os

import pytest
from google.protobuf.json_format import MessageToDict
from kestrel_sdk.native_a2a.client.interceptors import BeforeArgs
from kestrel_sdk.native_a2a.types import SubscribeToTaskRequest, TaskState
from starlette.applications import Starlette

from kestrel_sovereign.execution_custody import (
    execution_evidence_carriers,
    is_execution_control_error,
)
from tests.unit.test_a2a_native_http import wire as wire  # noqa: PLC0414
from tests.utils.native_a2a_security import PRINCIPAL, send


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
    args = BeforeArgs(params, sdk_operation, wire.card)
    await wire.interceptor.before(args)
    headers = {**dict(wire.http.headers), **args.context.service_parameters}
    if wire.protocol == "JSONRPC":
        path = wire.binding.prefix + "/rpc"
        body = {"jsonrpc": "2.0", "id": "idle-withdrawal", "method": method, "params": MessageToDict(params)}
    else:
        path = wire.binding.prefix + "/tenant-alpha/" + rest_path
        body = MessageToDict(params) if operation == "send" else {}
    payload = json.dumps(body).encode()
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def respond(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])
        if message["type"] == "http.response.body" and b"data:" in message.get("body", b""):
            first.set()

    app = Starlette(routes=[wire.binding.route])
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http", "path": path,
        "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "headers": [(str(k).lower().encode(), str(v).encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 1), "server": ("native-core.test", 80),
    }

    async def actual_request():
        try:
            await app(scope, receive, respond)
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
