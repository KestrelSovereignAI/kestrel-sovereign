"""Real loopback proxy and PG writers keep their accepted original Tasks.

Phoenix is never launched and has no process/adopted PID. Only its actual
HTTPX proxy socket and isolated fixture PID file are retired. Dispatcher
fixture routing doubles are unused: these are exact writer-boundary tests.
"""
import asyncio

import httpx
import pytest
from fastapi import FastAPI
from kestrel_sdk.signals import Signal, SignalMode

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.phoenix_supervisor import PhoenixSupervisor
from kestrel_sovereign.server import _shutdown_phoenix
from kestrel_sovereign.signals import DurableConsumerRegistration
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery, join_owned
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.native_admission_fault import NativeAdmissionFault
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_phoenix_proxy_publication_keeps_original_http_client_and_stops_independently(
    native_pg, tmp_path, monkeypatch, mode, eager, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    peer_closed, peers = asyncio.Event(), set()

    async def peer(reader, writer):
        task = asyncio.current_task()
        peers.add(task)
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            peer_closed.set()
            peers.discard(task)

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    client = httpx.AsyncClient(trust_env=False)
    supervisor = PhoenixSupervisor(working_dir=tmp_path)
    supervisor._client = client
    supervisor.pid_file.write_text("declared fixture; no process or adopted PID")
    assert supervisor.process is None and supervisor._adopted_pid is None
    app = FastAPI()
    app.state.phoenix = supervisor
    fault = NativeAdmissionFault(lambda operation: getattr(operation, "cr_code", None) is PhoenixSupervisor.aclose.__code__, source.error, mode, eager=eager)
    owner = None
    try:
        assert (await client.get(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/native-proxy")).status_code == 200
        owner = create_owned_task(_shutdown_phoenix(app), retain_originals=True)
        with fault:
            result = await await_owned_task(owner)
        fault.assert_terminal_unstarted()
        source.assert_retained(result.error)
        assert app.state.phoenix_proxy_close_task is fault.task
        assert app.state.phoenix_proxy_close_resource is client
        assert any(resource is client and task is fault.task for resource, task in execution_cleanup_debts(result.error))
        assert supervisor._client is client and not client.is_closed and not peer_closed.is_set()
        assert not supervisor.pid_file.exists(), "independent actual stop did not run"
        retained = app.state.phoenix_proxy_close_error
        original_chain = tuple(_explicit_control_chain(retained))
        for _ in range(20):
            repeated = await await_owned_task(create_owned_task(_shutdown_phoenix(app), retain_originals=True))
            source.assert_retained(repeated.error)
            assert app.state.phoenix_proxy_close_task is fault.task
            assert app.state.phoenix_proxy_close_error is retained
            assert tuple(_explicit_control_chain(retained)) == original_chain
        assert supervisor._client is client and not client.is_closed
    finally:
        if owner is not None:
            await await_owned_task(owner)
        for task in fault.accepted:
            await await_owned_task(task)
        await client.aclose()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("writer", ["terminalizer", "reservation-repair"])
async def test_dispatcher_writer_publication_retains_original_before_storage_teardown(
    native_pg, monkeypatch, mode, eager, writer,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    agent, dispatcher, delivery, _signal = await setup_delivery(native_pg)
    store, observer = dispatcher._durable_store, native_pg[1]
    sibling = await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'")
    if writer == "terminalizer":
        target = type(dispatcher)._apply_cognition_control_terminal.__code__
        operation = dispatcher._terminalize_failed_cognition(delivery, source.error)
        resource = delivery
        identity = delivery.delivery_id
    else:
        await store.register_consumer(DurableConsumerRegistration(
            consumer_id="original-writer-publication", source="native.writer.publication", agent_id=agent.did,
        ))
        persisted = await store.persist_signal(
            Signal(source="native.writer.publication", kind="component", mode=SignalMode.COGNITION, target_agent=agent.did, payload={}),
            agent_id=agent.did, source_event_id="original-writer-publication", retention_days=1,
            transient_selector_payload={}, initial_lease_owner=dispatcher._durable_delivery_owner,
        )
        assert len(persisted.initial_reservations) == 1
        identity = persisted.initial_reservations[0].delivery_id
        target = type(dispatcher)._requeue_committed_initial_reservations.__code__
        operation = dispatcher._repair_post_commit_reservations(persisted)
        resource = dispatcher
    before = await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id=$1", identity)
    fault = NativeAdmissionFault(lambda operation: getattr(operation, "cr_code", None) is target, source.error, mode, eager=eager)
    owner = create_owned_task(operation, retain_originals=True)
    try:
        with fault:
            result = await await_owned_task(owner)
        fault.assert_terminal_unstarted()
        source.assert_retained(result.error)
        assert any(value is resource and task is fault.task for value, task in execution_cleanup_debts(result.error))
        assert await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id=$1", identity) == before
        assert await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'") == sibling
        if writer == "terminalizer" and fault.task is not None:
            assert fault.task in dispatcher._cognition_control_terminal_owners.values()
    finally:
        await await_owned_task(owner)
        for task in fault.accepted:
            await await_owned_task(task)
        await join_owned(agent, dispatcher)
