"""Actual TaskManager subscription cleanup/control at the SSE helper boundary.

The routed-recipient view is an already-admitted input to this helper, NOT a
running Agent, signature-verification test or READY/genesis claim.
"""

import asyncio
from dataclasses import dataclass
from uuid import uuid4

import asyncpg
import pytest
from starlette.requests import Request

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.a2a.stores.unified import (
    ObservabilityStore,
    SessionService,
    TaskStore,
)
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.a2a.types import Message, TaskSendParams, TaskState, TextPart
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
    is_execution_control_error,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        "terminal",
        "terminal-no-new-task",
        "disconnect",
        "release-error",
        "release-cancel",
        "caller-cancel",
        "ordinary-cancel",
        "ordinary-asgi-error",
        "asgi-native-2.3",
        "asgi-native-2.4",
        "asgi-ordinary-cancel-2.3",
        "asgi-ordinary-cancel-2.4",
        "asgi-send-failure-2.3",
        "asgi-send-failure-2.4",
        "asgi-send-failure-no-new-task-2.4",
        "asgi-start-failure-2.3",
        "asgi-start-failure-2.4",
        "asgi-native-send-2.3",
        "asgi-native-send-2.4",
    ],
)
async def test_actual_task_sse_closes_original_subscription_and_retains_native_control(
    native_pg, monkeypatch, route
):
    from kestrel_sovereign.endpoints.agent import (
        _sse_connections,
        _task_subscription_response,
    )

    backend, observer, schema = native_pg
    manager = TaskManager(
        TaskStore(backend), SessionService(backend), ObservabilityStore(backend)
    )
    await manager.initialize()
    did = "did:web:original-sse-" + uuid4().hex
    task = await manager.create_task(
        TaskSendParams(
            id=uuid4().hex,
            sessionId=uuid4().hex,
            message=Message(role="user", parts=[TextPart(text="native SSE ownership")]),
        ),
        agent_name=did,
    )
    if route.startswith("terminal"):
        await manager.update_status(task.id, TaskState.WORKING, recipient_agent_id=did)
        await manager.complete_task(
            task.id, "actual terminal row", recipient_agent_id=did
        )

    @dataclass
    class RoutedRecipient:
        did: str
        task_manager: TaskManager

        @property
        def agent_id(self):
            return self.did

    agent = RoutedRecipient(did, manager)
    conn_key = ("127.0.0.1", did)
    initial_connections = _sse_connections.get(conn_key, 0)
    incoming = asyncio.Queue()
    if route == "disconnect":
        incoming.put_nowait({"type": "http.disconnect"})

    async def receive():
        if route == "ordinary-asgi-error":
            raise OSError("ordinary ASGI receive failed")
        return await incoming.get()

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/tasks/" + task.id + "/subscribe",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 49300),
            "server": ("127.0.0.1", 0),
            "scheme": "http",
        },
        receive=receive,
    )
    sources, failures, releases, frames, read_tasks = [], [], [], [], []
    get_failed = asyncio.Event()
    subscribe, native_get = manager.subscribe, manager.task_store.get_for_recipient

    def observed_source(*args, **kwargs):
        source = subscribe(*args, **kwargs)
        sources.append(source)  # Only retain/observe the ACTUAL original generator.
        return source

    async def observed_get(*args, **kwargs):
        read_tasks.append(asyncio.current_task())
        try:
            return await native_get(*args, **kwargs)
        except BaseException as error:
            failures.append(error)
            get_failed.set()
            raise

    monkeypatch.setattr(manager, "subscribe", observed_source)
    monkeypatch.setattr(manager.task_store, "get_for_recipient", observed_get)
    entered, release = asyncio.Event(), asyncio.Event()
    original = OSError("actual SSE native checkout release ACK lost")
    native_release = asyncpg.pool.PoolAcquireContext.__aexit__

    async def lose_release_ack(checkout, error_type, error, tb):
        result = await native_release(checkout, error_type, error, tb)
        releases.append(checkout)
        fault_read = 4 if route.startswith("asgi-native-send") else 3
        if len(releases) == fault_read and error_type is None:
            if route == "caller-cancel" or route.startswith("asgi-native"):
                entered.set()
                await release.wait()
                return result
            if route == "release-cancel":
                cancellation = asyncio.CancelledError(
                    "actual SSE native release withdrawal"
                )
                raise cancellation from original
            raise original
        return result

    response, failure, owner = None, None, None
    loop, factory = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refused_cleanup = []
    snapshot_seen = asyncio.Event()

    def forbid_cleanup_task(loop, operation, **kwargs):
        refused_cleanup.append(operation)
        raise RuntimeError("SSE cleanup must not need new Task admission")

    async def consume():
        async for frame in response.body_iterator:
            frames.append(frame)
            snapshot_seen.set()

    async def original_queue_is_waiting():
        while not manager._subscribers[task.id][0]._getters:
            await asyncio.sleep(0.001)

    async def send(message):
        if message["type"] == "http.response.start" and route.startswith(
            "asgi-start-failure"
        ):
            raise OSError("actual ASGI response start lost its client")
        if message["type"] == "http.response.body" and message.get("body"):
            frames.append(message["body"].decode())
            snapshot_seen.set()
            if route.startswith("asgi-native-send"):
                # Delivery itself can perform native work. Exercise control
                # outside the suspended generator, in the REAL ASGI producer.
                await manager.get_task_for_recipient(task.id, did)
            if route.startswith("asgi-send-failure"):
                if "no-new-task" in route:
                    loop.set_task_factory(forbid_cleanup_task)
                raise OSError("actual ASGI delivery lost its client")

    try:
        if (
            route.startswith("release-")
            or route == "caller-cancel"
            or route.startswith("asgi-native")
        ):
            backend._execution_custody = ExecutionCustody(GenerationFence())
            monkeypatch.setattr(
                asyncpg.pool.PoolAcquireContext, "__aexit__", lose_release_ack
            )
        response = await _task_subscription_response(request, agent, task.id)
        if route.startswith(("asgi-send-failure", "asgi-start-failure")):
            scope = dict(request.scope)
            scope["asgi"] = {
                "version": "3.0",
                "spec_version": route.rsplit("-", 1)[-1],
            }
            owner = create_owned_task(response(scope, receive, send))
            outcome = await asyncio.wait_for(await_owned_task(owner), 5)
            failure = outcome.error
        elif route.startswith("asgi-native"):
            scope = dict(request.scope)
            scope["asgi"] = {
                "version": "3.0",
                "spec_version": route.rsplit("-", 1)[-1],
            }
            owner = create_owned_task(response(scope, receive, send))
            await asyncio.wait_for(entered.wait(), 5)
            assert len(read_tasks) == (4 if route.startswith("asgi-native-send") else 3)
            read_tasks[-1].cancel("actual ASGI body stopped during its native ACK")
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(get_failed.wait(), 5)
            incoming.put_nowait({"type": "http.disconnect"})
            outcome = await asyncio.wait_for(await_owned_task(owner), 5)
            failure = outcome.error
        elif route.startswith("asgi-ordinary-cancel"):
            scope = dict(request.scope)
            scope["asgi"] = {
                "version": "3.0",
                "spec_version": route.removeprefix("asgi-ordinary-cancel-"),
            }
            owner = create_owned_task(response(scope, receive, send))
            await asyncio.wait_for(snapshot_seen.wait(), 5)
            await asyncio.wait_for(original_queue_is_waiting(), 5)
            assert len(read_tasks) == 3
            read_tasks[-1].cancel("ordinary ASGI body withdrawal after native read")
            incoming.put_nowait({"type": "http.disconnect"})
            outcome = await asyncio.wait_for(await_owned_task(owner), 5)
            failure = outcome.error
        elif route == "ordinary-cancel":
            owner = create_owned_task(consume())
            await asyncio.wait_for(snapshot_seen.wait(), 5)
            await asyncio.wait_for(original_queue_is_waiting(), 5)
            owner.cancel("ordinary SSE client left after its actual snapshot")
            outcome = await asyncio.wait_for(await_owned_task(owner), 5)
            failure = outcome.error
        elif route == "caller-cancel":
            owner = create_owned_task(consume())
            await asyncio.wait_for(entered.wait(), 5)
            owner.cancel("actual SSE client stopped during original native release ACK")
            await asyncio.sleep(0)
            release.set()
            outcome = await asyncio.wait_for(await_owned_task(owner), 5)
            failure = outcome.error
        else:
            if route == "terminal-no-new-task":
                frames.append(await anext(response.body_iterator))
                loop.set_task_factory(forbid_cleanup_task)
            try:
                await consume()
            except BaseException as error:
                failure = error
        assert len(sources) == (0 if route.startswith("asgi-start-failure") else 1)
        if (
            route.startswith("release-")
            or route == "caller-cancel"
            or route.startswith("asgi-native")
        ):
            assert (
                len(releases) >= 3
                and failures
                and is_execution_control_error(failures[0])
            ), "native setup did not produce actual canonical control"
            if route == "caller-cancel" or route.startswith("asgi-native"):
                assert isinstance(failures[0], asyncio.CancelledError)
                assert execution_commit_outcome(failures[0]) == "committed"
            else:
                assert carries(failures[0], original)
            assert (
                failure is not None
                and carries(failure, failures[0])
                and is_execution_control_error(failure)
            ), "SSE normalized original native control into generic data or EOF"
            assert execution_commit_outcome(failure) == execution_commit_outcome(
                failures[0]
            )
            assert not any("Internal server error" in frame for frame in frames)
        else:
            if route.startswith(("asgi-send-failure", "asgi-start-failure")):
                assert failure is not None and not is_execution_control_error(failure)
                assert bool(frames) == route.startswith("asgi-send-failure")
                assert failures == []
            else:
                assert failure is None
            if route.startswith("terminal"):
                assert any("end-of-stream" in frame for frame in frames)
            elif route == "ordinary-cancel" or route.startswith("asgi-ordinary-cancel"):
                assert frames and failures == []
                assert not any("Internal server error" in frame for frame in frames)
            elif route == "ordinary-asgi-error":
                assert failures == []
                assert any("Internal server error" in frame for frame in frames)
            elif not route.startswith(("asgi-send-failure", "asgi-start-failure")):
                assert frames == []
        assert not refused_cleanup, "SSE retirement required a new cleanup Task"
        assert task.id not in manager._subscribers, (
            "SSE stopped forwarding without closing its ORIGINAL registered subscription"
        )
        assert all(source.ag_frame is None for source in sources), (
            "original raw subscription generator remains suspended"
        )
        assert _sse_connections.get(conn_key, 0) == initial_connections, (
            "SSE connection quota survived original response retirement"
        )
    finally:
        # Clean the SAME native source only after the ownership assertions.
        loop.set_task_factory(factory)
        release.set()
        if owner is not None:
            if not owner.done():
                owner.cancel("fixture retires its own SSE observer")
            await await_owned_task(owner)
        backend._execution_custody = None
        monkeypatch.setattr(
            asyncpg.pool.PoolAcquireContext, "__aexit__", native_release
        )
        if response is not None:
            await response.body_iterator.aclose()
        for source in sources:
            await source.aclose()
        for operation in refused_cleanup:
            operation.close()
        await manager.close()
