"""Detached provider completion never buys replacement execution custody."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, ExecutionCustody, is_execution_control_error
from kestrel_sovereign.features.isolated_runtime import ProxyFeature, _HostIngressRequest
from tests.unit.test_execution_custody import Authority
from tests.unit.test_execution_custody_review15 import control_error


def proxy_fixture(call):
    scope = ExecutionCustody(Authority())
    client = SimpleNamespace(host_ingress_capabilities=SimpleNamespace(names=("ack",)), call_host_ingress=call)
    proxy = ProxyFeature.__new__(ProxyFeature)
    proxy.agent = SimpleNamespace(_execution_custody=scope)
    proxy.name = "original-feature"
    proxy._client = client
    proxy._terminal_lifecycle_latched = False
    proxy._event_ack_clients = []
    proxy._event_ack_tasks = set()
    proxy._fence_event_ingress_ack_source = lambda _client: None
    return proxy, client, scope


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_owner", [False, True])
async def test_ingress_retry_rechecks_original_scope_after_backoff(replace_owner):
    attempted = asyncio.Event()
    calls = []

    async def call(*args):
        calls.append(args)
        attempted.set()
        return {"status": "error", "http_status": 409}

    proxy, client, scope = proxy_fixture(call)
    proxy._schedule_event_ingress_acknowledgement(client, _HostIngressRequest("ack", {}))
    task = next(iter(proxy._event_ack_tasks))
    await attempted.wait()
    scope.revoke("original runtime revoked during ACK backoff")
    if replace_owner:
        proxy.agent._execution_custody = ExecutionCustody(Authority())
    with pytest.raises(ExecutionAuthorityError):
        await task
    assert len(calls) == 1
    assert proxy._client is client


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
async def test_ingress_rpc_control_is_not_normalized_to_retry(control, carrier):
    error = control_error(control, carrier)
    calls = []

    async def call(*args):
        calls.append(args)
        raise error

    proxy, client, _scope = proxy_fixture(call)
    with pytest.raises(BaseException) as caught:
        await proxy._await_event_ingress_ack_attempt(call, _HostIngressRequest("ack", {}), client)
    assert caught.value is error
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
async def test_queued_ingress_completion_cannot_ignore_predecessor_control(control, carrier):
    error = control_error(control, carrier)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def call(*args):
        calls.append(args)
        entered.set()
        await release.wait()
        raise error

    proxy, client, _scope = proxy_fixture(call)
    request = _HostIngressRequest("ack", {})
    proxy._schedule_event_ingress_acknowledgement(client, request)
    first = next(iter(proxy._event_ack_tasks))
    await entered.wait()
    proxy._schedule_event_ingress_acknowledgement(client, request)
    second = next(task for task in proxy._event_ack_tasks if task is not first)
    release.set()
    try:
        # The completion callback is the first receipt owner. A later raw
        # await of a cancelled Task discards its original cancellation cause;
        # every subsequent owner must use the same canonical receipt cache.
        outcome = await await_owned_task(second)
        assert outcome.error is error
        assert is_execution_control_error(outcome.error)
        assert (await await_owned_task(first)).error is error
        assert (await await_owned_task(second)).error is error
        assert proxy._event_ingress_control is error
        assert not proxy._event_ack_tasks
        assert not proxy._event_ack_clients
        assert len(calls) == 1
    finally:
        await await_owned_task(first)
        await await_owned_task(second)
