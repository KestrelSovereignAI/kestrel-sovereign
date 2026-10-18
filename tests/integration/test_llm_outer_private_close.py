"""Original private SDK transport survives refusal of its outer cleanup wrapper.

The typed lease is a trusted API-input fixture only. This verifies native SDK/
HTTP retirement, NOT a provider canary, GPU allocation, Agent startup or READY.
No model or external-provider request is permitted. Rotation pins use only a
local callback returning that same trusted input snapshot, not a provider poll.
"""

import asyncio
import inspect
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from tests.utils.original_failure import OriginalFailure
from kestrel_sdk.llm import (
    InferenceLease,
    InferenceLeaseState,
    InferencePrivacy,
    InferenceRoute,
)
from pydantic import SecretStr

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.llm.lifecycle import retire_unpublished_llm
from kestrel_sovereign.llm.service import LLMService, LLMServiceError
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_cleanup_task_admission import _operation_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["direct", "unpublished", "rotation", "rotation-timeout"]
)
async def test_refused_outer_private_close_keeps_original_sdk_and_service(
    tmp_path, monkeypatch, route
):
    from kestrel_sovereign.llm import service as service_module

    monkeypatch.setattr(service_module, "load_section", lambda _name: {})
    peers, peer_closed, accepted = [], asyncio.Event(), asyncio.Event()

    async def peer(reader, writer):
        peers.append(asyncio.current_task())
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            assert header.split(b" ")[1] == b"/native-retirement-probe"
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            accepted.set()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            peer_closed.set()

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    endpoint = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    db = await AsyncDatabase.sqlite(str(tmp_path / "original-private-custody.db"))
    service = LLMService(usage_db=db, agent_data_dir=tmp_path)
    now = datetime.now(timezone.utc)
    lease = InferenceLease(
        lease_id="native-loopback-" + uuid4().hex,
        quote_id="typed-input",
        request_id="typed-input",
        owner_id="native-retirement-owner",
        provider_name="local-retirement-fixture",
        state=InferenceLeaseState.READY,
        model="no-model-invocation",
        runtime="openai",
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(minutes=5),
        hourly_cost_usd=Decimal("0"),
        estimated_total_cost_usd=Decimal("0"),
        route=InferenceRoute(
            endpoint=SecretStr(endpoint + "/v1"),
            model="no-model-invocation",
            api_key=SecretStr("inert-local-fixture-not-provider-credential"),
        ),
    )

    touch_calls = []

    async def trusted_snapshot_only(_lease_id):
        touch_calls.append(_lease_id)
        if route in ("rotation", "rotation-timeout"):
            # Trusted input snapshot only, not a provider status/canary result.
            assert _lease_id == lease.lease_id
            return service._remote_lease
        pytest.fail("native cleanup test must not touch a provider lease")

    loop, factory = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refusal, refused, owners = (
        RuntimeError("original outer remote route closer Task refused"),
        [],
        {},
    )
    client, rotated_client, rotation = None, None, None
    source = OriginalFailure.capture(refusal)

    def refuse_outer(loop, coroutine, **kwargs):
        if _operation_name(coroutine).endswith(".close_remote_route"):
            refused.append(coroutine)
            raise refusal
        return (
            factory(loop, coroutine, **kwargs)
            if factory
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        await service.activate_inference_lease(
            lease, capabilities=(), touch_lease=trusted_snapshot_only
        )
        client = service._remote_client
        assert client is not None
        response = await client._client.get(endpoint + "/native-retirement-probe")
        assert response.status_code == 200 and response.text == "ok"
        await asyncio.wait_for(accepted.wait(), 5)
        assert not client._client.is_closed and not peer_closed.is_set()
        async with (
            service._remote_route_attempt(force_local_only=False, model_override=None)
            if route in ("rotation", "rotation-timeout")
            else nullcontext()
        ) as snapshot:
            if route in ("rotation", "rotation-timeout"):
                if route == "rotation-timeout":
                    from kestrel_sovereign.llm import remote_backend

                    monkeypatch.setattr(remote_backend, "HTTP_TIMEOUT_MEDIUM", 0.1)
                assert snapshot.client is client and service._remote_inflight == 1
                waiting = asyncio.Event()
                original_wait = service._remote_route_condition.wait

                async def observed_original_wait():
                    waiting.set()
                    return await original_wait()

                monkeypatch.setattr(
                    service._remote_route_condition, "wait", observed_original_wait
                )
                rotated_lease = replace(
                    lease,
                    route=replace(
                        lease.route,
                        api_key=SecretStr("rotated-inert-local-fixture"),
                    ),
                )
                operation = service.activate_inference_lease(
                    rotated_lease,
                    capabilities=(),
                    touch_lease=trusted_snapshot_only,
                )
                rotation = create_owned_task(operation)
                await asyncio.wait_for(waiting.wait(), 5)
                rotated_client = operation.cr_frame.f_locals["client"]
                assert (
                    rotated_client is not client
                    and not rotated_client._client.is_closed
                )
            loop.set_task_factory(refuse_outer)
            caught = None
            try:
                if route == "unpublished":
                    await retire_unpublished_llm(
                        owners, "original-private", service, ()
                    )
                else:
                    await service.close()
            except BaseException as error:
                caught = error
            loop.set_task_factory(factory)
            if route == "rotation-timeout":
                result = await asyncio.wait_for(await_owned_task(rotation), 5)
                assert result.error is not None
                assert service._remote_accepting is False, (
                    "rotation timeout reopened original route after shutdown"
                )
        if rotation is not None:
            outcome = await asyncio.wait_for(await_owned_task(rotation), 5)
            assert outcome.error is not None, (
                "private rotation published after original shutdown"
            )
            assert (
                service._remote_client is client and rotated_client._client.is_closed
            ), "late private rotation replaced the original client after shutdown won"
        source.assert_retained(caught)
        assert refused and any(resource is service and task is None
                               for resource, task in execution_cleanup_debts(caught))
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused
        )
        if route == "unpublished":
            assert (
                "original-private" in owners
                and owners["original-private"].provider is service
            ), (
                "unpublished lifetime dropped the actual Service while its original SDK socket remained live"
            )
        assert service.has_pending_close_owners(), (
            "outer closer refusal hid the original live native SDK owner"
        )
        assert service._remote_route_closed is True, (
            "a refused cleanup wrapper left route activation open"
        )
        before = len(touch_calls)
        with pytest.raises(LLMServiceError):
            async with service._remote_route_attempt(
                force_local_only=False, model_override=None
            ):
                pytest.fail("original shutdown admitted a fresh private route attempt")
        assert len(touch_calls) == before, "shutdown reached the lease-touch callback"
        assert service._remote_client is client and not client._client.is_closed
        assert not peer_closed.is_set()
        assert is_execution_control_error(caught) and any(
            resource is service or resource is client
            for resource, _task in execution_cleanup_debts(caught)
        )
    finally:
        loop.set_task_factory(factory)
        if rotation is not None:
            await await_owned_task(rotation)
        for coroutine in refused:
            coroutine.close()
        if client is not None:
            await client.close()
            await asyncio.wait_for(peer_closed.wait(), 5)
        if rotated_client is not None:
            await rotated_client.close()
        try:
            await service.close()
        except BaseException as error:
            assert error is refusal or is_execution_control_error(error)
        await db.close()
        server.close()
        await server.wait_closed()
        for task in peers:
            await await_owned_task(task)
