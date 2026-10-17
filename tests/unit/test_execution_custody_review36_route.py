"""Private-route cancellation owns the real unpublished SDK client."""
import asyncio
from dataclasses import replace

import httpx
import openai
import pytest
from tests.utils.original_failure import OriginalFailure
from pydantic import SecretStr

from kestrel_sovereign._async_ownership import await_owned_task
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_private_inference_routing import _Host, _lease, _touch_current


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["none", "unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("boundary", ["publication_lock", "drain"])
async def test_cancelled_rotation_retires_unpublished_native_client(monkeypatch, kind, boundary):
    host = _Host()
    entered, release, built = asyncio.Event(), asyncio.Event(), asyncio.Event()
    clients, closes = [], []
    sdk_client = openai.AsyncOpenAI
    failure = None if kind == "none" else receipt(kind)
    source = OriginalFailure.capture(failure) if failure is not None else None

    async def request(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"id": "offline", "object": "chat.completion",
            "created": 1, "model": "local", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}]})

    def construct(**kwargs):
        http = httpx.AsyncClient(transport=httpx.MockTransport(request))
        client = sdk_client(http_client=http, **kwargs)
        index = len(clients)
        original_close = client.close

        async def close():
            closes.append(index)
            await original_close()
            if index == 1 and failure is not None:
                raise failure

        monkeypatch.setattr(client, "close", close)
        clients.append((client, http, original_close))
        if index == 1:
            built.set()
        return client

    monkeypatch.setattr("kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI", construct)
    lease = _lease()
    await host.activate_inference_lease(lease, capabilities=("chat",),
                                        touch_lease=_touch_current(host))

    async def invoke():
        async with host._remote_route_attempt(force_local_only=False, model_override=None) as route:
            await route.client.chat.completions.create(
                model="local", messages=[{"role": "user", "content": "offline"}])

    owner = asyncio.create_task(invoke())
    rotation = contender = None
    held, release_lock = asyncio.Event(), asyncio.Event()

    async def contend():
        async with host._remote_route_condition:
            held.set()
            await release_lock.wait()

    try:
        await asyncio.wait_for(entered.wait(), 5)
        rotated = replace(lease, route=replace(
            lease.route, endpoint=SecretStr("https://rotated.invalid/v1")))
        if boundary == "publication_lock":
            await host._remote_route_condition.acquire()
        rotation = asyncio.create_task(host.activate_inference_lease(
            rotated, capabilities=("chat",), touch_lease=_touch_current(host)))
        if boundary == "publication_lock":
            # Queue the real lock owners in FIFO order: activation's initial
            # check, then the contender, then activation's publication check.
            await asyncio.sleep(0)
            contender = asyncio.create_task(contend())
            await asyncio.sleep(0)
            host._remote_route_condition.release()
            await asyncio.wait_for(held.wait(), 5)
        await asyncio.wait_for(built.wait(), 5)
        assert host._remote_accepting is (boundary == "publication_lock")
        assert host._remote_inflight == 1
        rotation.cancel("original rotation caller cancellation")
        outcome = await await_owned_task(rotation)
        assert clients[1][1].is_closed, "cancelled activation leaked its unpublished native client"
        assert closes == [1]
        assert not clients[0][1].is_closed, "rotation retired the pinned old client early"
        assert host._remote_client is clients[0][0]
        assert host._remote_accepting is (boundary == "publication_lock")
        if failure is None:
            assert isinstance(outcome.error, asyncio.CancelledError)
        else:
            source.assert_retained(outcome.error, resource=clients[1][0])
            repeated = await await_owned_task(asyncio.create_task(
                host._close_remote_client(clients[1][0], lease_id=lease.lease_id)))
            source.assert_retained(repeated.error, resource=clients[1][0])
            assert closes == [1]
    finally:
        release_lock.set()
        if contender is not None:
            await await_owned_task(contender)
        release.set()
        await await_owned_task(owner)
        if rotation is not None:
            await await_owned_task(rotation)
        await await_owned_task(asyncio.create_task(host.deactivate_inference_lease(lease.lease_id)))
        for client, http, original_close in clients:
            await original_close()
            assert http.is_closed
