"""Host-selected ACK correlation survives routing without leaking across requests."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.features import isolated_runtime as runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_ack_context_is_task_local_and_restored_after_every_exit(outcome):
    feature = object.__new__(runtime.ProxyFeature)
    entered, release = asyncio.Event(), asyncio.Event()
    observed = []

    async def route(payload):
        observed.append((payload, runtime._telegram_acknowledgement_session_id.get()))
        if payload == "bound":
            entered.set()
            await release.wait()
            if outcome == "error":
                raise RuntimeError("controlled route rejection")
        return payload

    feature._route_inbound = route

    async def bound():
        try:
            return await feature._route_validated_inbound(
                "bound", cursor_owned_protocol=True,
                telegram_terminal_disposition="care_acknowledgement",
                acknowledgement_session_id="run-a",
            )
        finally:
            assert runtime._telegram_acknowledgement_session_id.get() is None
            assert runtime._telegram_terminal_inbound_disposition.get() is None
            assert runtime._cursor_owned_inbound_protocol.get() is False

    task = asyncio.create_task(bound())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await feature._route_validated_inbound("ordinary", cursor_owned_protocol=False)
        assert runtime._telegram_acknowledgement_session_id.get() is None
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            if outcome == "error":
                with pytest.raises(RuntimeError, match="controlled route rejection"):
                    await task
            else:
                assert await task == "bound"
        assert observed == [("bound", "run-a"), ("ordinary", None)]
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_actual_proxy_route_passes_only_host_session_to_terminal_producer():
    feature = object.__new__(runtime.ProxyFeature)
    feature.agent = SimpleNamespace(did="did:test:ack-run")
    feature._require_event_ingress_work = lambda: None
    feature._authoritative_inbound_channel_type = lambda: "telegram"
    observed = []

    async def terminal(message, **kwargs):
        observed.append((message, kwargs))
        return "accepted"

    feature._channel_feature = lambda: SimpleNamespace(handle_terminal_inbound=terminal)
    payload = {
        "id": "telegram:v2:bot:42:update:12", "channel_type": "forged",
        "direction": "inbound",
        "sender": "123", "recipient": "123", "content": "OK",
        "session_id": "forged-run", "acknowledgement_session_id": "forged-run",
        "metadata": {"session_id": "forged-run"},
    }
    assert await feature._route_validated_inbound(
        payload, cursor_owned_protocol=True,
        telegram_terminal_disposition="care_acknowledgement",
        acknowledgement_session_id="original-run",
    ) == "accepted"
    assert observed[0][1] == {"disposition": "care_acknowledgement", "session_id": "original-run"}
    assert observed[0][0].agent_id == feature.agent.did
    assert observed[0][0].channel_type == "telegram"
    await feature._route_validated_inbound(
        payload, cursor_owned_protocol=True,
        telegram_terminal_disposition="care_acknowledgement",
    )
    assert observed[1][1] == {"disposition": "care_acknowledgement"}


@pytest.mark.asyncio
async def test_internal_route_rejects_session_on_non_ack_disposition():
    feature = object.__new__(runtime.ProxyFeature)
    with pytest.raises(runtime.ProtocolError, match="restricted"):
        await feature._route_validated_inbound(
            {}, cursor_owned_protocol=True,
            telegram_terminal_disposition="unsupported_update",
            acknowledgement_session_id="run-a",
        )
