"""Callback validation releases topology ownership and rechecks before commit."""

import asyncio
import ipaddress
import threading

import httpx
import pytest
from kestrel_sovereign._vendor.a2a.client.errors import A2AClientError
from kestrel_sovereign._vendor.a2a.types import GetTaskRequest
from kestrel_sovereign._vendor.a2a.utils.errors import InternalError
from kestrel_sdk.security import outbound

from tests.unit.test_a2a_native_http import submit
from tests.unit.test_a2a_native_http import wire as wire  # noqa: PLC0414
from tests.unit.test_a2a_native_push_http import callback
from tests.unit.test_a2a_native_push_http import (
    native_push_runtime as native_push_runtime,  # noqa: PLC0414
)
from tests.utils.native_a2a_security import SENDER, send, signed_context
from tests.utils.native_push import intercept_push_commit_ack


async def _invoke_callback(wire, operation, task, config):
    if operation == "config":
        return await wire.client().create_task_push_notification_config(config)
    params = send(wire.admission)
    params.configuration.return_immediately = True
    if operation == "continue":
        params.message.task_id = task.id
        params.message.context_id = task.context_id
    params.configuration.task_push_notification_config.CopyFrom(config)
    if operation == "stream":
        # Exercise the real SSE handler through its first event, then close
        # the original generator without waiting for business-task completion.
        stream = wire.handler.on_message_send_stream(
            params,
            signed_context(wire.admission, params, operation="SendStreamingMessage"),
        )
        try:
            return await anext(stream)
        finally:
            await stream.aclose()
    return [event async for event in wire.client().send_message(params)]


@pytest.mark.parametrize("operation", ["send", "continue", "stream", "config"])
@pytest.mark.parametrize(
    "policy_change", ["none", "withdraw", "principal", "service-stop"]
)
async def test_callback_dns_releases_host_lease_and_rechecks_original_authority(
    wire, monkeypatch, operation, policy_change
):
    original_task = await submit(wire)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    original_resolver = outbound._resolve_addresses

    def blocked_dns(host, port):
        if host != "callback-dns.test":
            return original_resolver(host, port)
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(10):
            raise AssertionError("test did not release its original DNS operation")
        return [ipaddress.ip_address("8.8.8.8")]

    monkeypatch.setattr(outbound, "_resolve_addresses", blocked_dns)
    config = callback(wire, original_task.id if operation == "config" else "")
    config.url = "https://callback-dns.test/notify"
    sending = asyncio.create_task(
        _invoke_callback(wire, operation, original_task, config)
    )
    writer = None
    try:
        await asyncio.wait_for(entered.wait(), 5)

        async def topology_write():
            async with wire.admission.lock:
                return True

        writer = asyncio.create_task(topology_write())
        done, _ = await asyncio.wait((writer,), timeout=1)
        assert done and writer.result(), (
            "callback DNS retained the host-wide native lifecycle reader lease"
        )
        # A queued topology writer must not stall unrelated native task reads.
        fetched = await wire.client().get_task(
            GetTaskRequest(id=original_task.id, tenant=wire.admission.scope.tenant)
        )
        assert fetched.id == original_task.id
        async with wire.admission.lock:
            if policy_change == "withdraw":
                wire.admission.state.current = False
            elif policy_change == "principal":
                await wire.admission.backend.execute(
                    "UPDATE native_test_principals SET principal = ? WHERE signing_did = ?",
                    ("did:changed-principal", SENDER),
                )
            elif policy_change == "service-stop":
                await wire.manager.push_service.close()
        release.set()
        if policy_change == "none":
            assert await sending is not None
        else:
            if policy_change == "service-stop":
                with pytest.raises(InternalError):
                    await sending
            elif operation == "stream":
                from starlette.exceptions import HTTPException

                with pytest.raises(HTTPException) as refused:
                    await sending
                assert refused.value.status_code == 403
            else:
                with pytest.raises(A2AClientError) as refused:
                    await sending
                assert isinstance(refused.value.__cause__, httpx.HTTPStatusError)
                assert refused.value.__cause__.response.status_code == 403
            assert await wire.admission.backend.fetch_one(
                "SELECT COUNT(*) FROM a2a_push_configs"
            ) == (0,)
            assert len(wire.submissions) == 1, (
                "revoked callback admission committed or woke task input"
            )
            snapshot = await wire.manager.get_task_snapshot(
                original_task.id, recipient_agent_id=wire.admission.scope.recipient_id
            )
            assert snapshot.input_revision == 0
    finally:
        release.set()
        if writer is not None and not writer.done():
            writer.cancel()
        await asyncio.gather(
            *(task for task in (sending, writer) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.parametrize("operation", ["send", "continue", "config"])
async def test_callback_commit_retains_authority_until_original_sql_ack(
    wire, monkeypatch, operation
):
    task = await submit(wire)
    config = callback(wire, task.id if operation == "config" else "")
    committed, release = asyncio.Event(), asyncio.Event()
    outbox = wire.manager.push_service.worker.outbox

    async def original_commit_ack():
        committed.set()
        await release.wait()

    intercept_push_commit_ack(
        outbox,
        monkeypatch,
        original_commit_ack,
        owner=outbox if operation == "config" else wire.manager,
        method={
            "send": "create_task",
            "continue": "continue_task",
            "config": "register_prepared",
        }[operation],
    )
    sending = asyncio.create_task(_invoke_callback(wire, operation, task, config))
    writer = None
    try:
        await asyncio.wait_for(committed.wait(), 5)

        async def topology_write():
            async with wire.admission.lock:
                return True

        writer = asyncio.create_task(topology_write())
        done, _ = await asyncio.wait((writer,), timeout=0.1)
        assert not done, "callback commit lost its original topology admission"
        release.set()
        assert await sending is not None
        assert await writer
    finally:
        release.set()
        await asyncio.gather(
            *(owner for owner in (sending, writer) if owner is not None),
            return_exceptions=True,
        )


async def test_unauthorized_callback_does_not_resolve_peer_supplied_dns(
    wire, monkeypatch
):
    lookups = []

    def forbidden_lookup(host, port):
        lookups.append((host, port))
        raise AssertionError("unauthorized callback reached DNS")

    monkeypatch.setattr(outbound, "_resolve_addresses", forbidden_lookup)
    await wire.admission.backend.execute("DELETE FROM native_test_principals")
    params = send(wire.admission)
    params.configuration.task_push_notification_config.CopyFrom(callback(wire, ""))
    params.configuration.task_push_notification_config.url = "https://unauthorized.test/"
    with pytest.raises(A2AClientError) as refused:
        _ = [event async for event in wire.client().send_message(params)]
    assert isinstance(refused.value.__cause__, httpx.HTTPStatusError)
    assert refused.value.__cause__.response.status_code == 403
    assert not lookups and not wire.submissions
