"""Original SDK transports, unpublished owners and usage writers retain custody."""

import asyncio
import inspect

import httpx2 as httpx
import openai
import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.llm.lifecycle import retire_unpublished_llm
from tests.unit.test_cleanup_task_admission import _partial_native_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["client", "unpublished"])
async def test_real_sdk_close_refusal_retains_inventory_and_closes_independent_peer(
    tmp_path, monkeypatch, refusal
):
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    service = agent.llm_service
    peers, peer_closed = set(), []

    async def peer(reader, writer):
        task = asyncio.current_task()
        peers.add(task)
        exited = asyncio.Event()
        peer_closed.append(exited)
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            peers.discard(task)
            exited.set()

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    endpoint = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/"
    transports = [httpx.AsyncClient(trust_env=False) for _ in range(2)]
    # Inert local constructor token only. Never reads a provider key, performs
    # an OpenAI request, or reaches any provider: actual TCP is loopback HTTPX.
    clients = [
        openai.AsyncOpenAI(
            api_key="inert-local-native-close-test",
            base_url=endpoint,
            http_client=transport,
        )
        for transport in transports
    ]
    for index, client in enumerate(clients):
        service.providers.append(
            {"name": f"actual-local-sdk-{index}", "client": client}
        )
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    failure = RuntimeError("actual original SDK close Task allocation refused")

    def factory(loop, coroutine, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        inner = frame.f_locals.get("operation") if frame is not None else None
        operation = inner if inspect.iscoroutine(inner) else coroutine
        frame, code = (
            getattr(operation, "cr_frame", None),
            getattr(operation, "cr_code", None),
        )
        selected = code is not None and (
            (
                refusal == "client"
                and code.co_qualname == "LLMService.close.<locals>.close_client"
                and frame.f_locals.get("client") is clients[0]
            )
            or (
                refusal == "unpublished"
                and code.co_qualname == "UnpublishedLLMOwner._close"
            )
        )
        if not refused and selected:
            refused.append((coroutine, operation))
            raise failure
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    caught, owners = None, {}
    try:
        for transport in transports:
            assert (await transport.get(endpoint)).text == "ok"
        assert len(peer_closed) == 2 and not any(
            event.is_set() for event in peer_closed
        )
        loop.set_task_factory(factory)
        try:
            if refusal == "client":
                await service.close()
            else:
                await retire_unpublished_llm(owners, "actual-sdk-owner", service, ())
        except BaseException as error:
            caught = error
        loop.set_task_factory(previous)
        assert refused and not transports[0].is_closed
        if refusal == "client":
            assert (
                service.has_pending_close_owners(),
                transports[1].is_closed,
                caught is not None and is_execution_control_error(caught),
            ) == (True, True, True), (
                "unstarted original SDK close lost custody or skipped an independent actual transport"
            )
            assert any(
                resource is clients[0]
                for resource, _ in execution_cleanup_debts(caught)
            )
            await asyncio.wait_for(peer_closed[1].wait(), 2)
            with pytest.raises(BaseException) as repeated:
                await service.close()
            assert repeated.value is caught and len(refused) == 1
        else:
            assert (
                "actual-sdk-owner" in owners
                and owners["actual-sdk-owner"].provider is service
                and caught is not None
                and is_execution_control_error(caught)
            ), (
                "unpublished Task refusal removed the original service with live native transports"
            )
            assert any(
                resource is service for resource, _ in execution_cleanup_debts(caught)
            )
            with pytest.raises(BaseException) as repeated:
                await retire_unpublished_llm(owners, "actual-sdk-owner", service, ())
            assert repeated.value is caught and len(refused) == 1
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        )
        assert not peer_closed[0].is_set()
        assert not agent._runtime_publication_ready.is_set()
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        for client in clients:
            await client.close()
        for event in peer_closed:
            await asyncio.wait_for(event.wait(), 2)
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)
        try:
            await agent.shutdown()
        except BaseException as error:
            # Assertions preceded manual physical fixture cleanup. Original
            # refusal debt can remain latched; never fake its lifecycle ACK.
            if not is_execution_control_error(error):
                raise
        await db.close()


@pytest.mark.asyncio
async def test_public_usage_close_retains_original_database_after_native_refusal(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.llm import service as service_module
    from tests.unit.test_host_context_native_cleanup import _actual_operation

    monkeypatch.setattr(service_module, "load_section", lambda _name: {})
    for name in ("KESTREL_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    service = service_module.LLMService(agent_data_dir=tmp_path)
    await service._ensure_db_initialized()
    db = service._usage_db
    assert service._usage_db_owned
    connection = db.backend._connection
    worker = getattr(connection, "_thread", connection)
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    failure = RuntimeError("actual owned usage database closer refused")
    source = OriginalFailure.capture(failure)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code, frame = (
            getattr(operation, "cr_code", None),
            getattr(operation, "cr_frame", None),
        )
        if (
            not refused
            and code is not None
            and code.co_qualname == "SQLiteBackend.close"
            and frame.f_locals.get("self") is db.backend
        ):
            refused.append((coroutine, operation))
            raise failure
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        loop.set_task_factory(factory)
        with pytest.raises(BaseException) as caught:
            await service.close_usage_db()
        loop.set_task_factory(previous)
        source.assert_retained(caught.value)
        assert is_execution_control_error(caught.value)
        assert service._usage_db is db and service._db_initialized, (
            "failed physical usage close discarded its original database"
        )
        assert worker.is_alive() and db.backend._connection is connection
        assert any(resource is db and task is None
                   for resource, task in execution_cleanup_debts(caught.value))
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        )
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await db.backend.close()
        try:
            await service.close()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        await db.close()


@pytest.mark.asyncio
async def test_refused_writer_wrapper_does_not_close_original_writers_database(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.llm import service as service_module
    from tests.unit.test_host_context_native_cleanup import _actual_operation

    monkeypatch.setattr(service_module, "load_section", lambda _name: {})
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.02)
    for name in ("KESTREL_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    service = service_module.LLMService(agent_data_dir=tmp_path)
    await service._ensure_db_initialized()
    db = service._usage_db
    assert service._usage_db_owned
    await db.execute("CREATE TABLE original_preference_writes (id INTEGER)")
    connection = db.backend._connection
    worker = getattr(connection, "_thread", connection)
    entered, release = asyncio.Event(), asyncio.Event()
    completed = []

    async def persist(model, vendor, route):
        # This original writer is held BETWEEN two acknowledged native writes,
        # not an invented commit receipt or a substituted database operation.
        await db.execute("INSERT INTO original_preference_writes VALUES (1)")
        entered.set()
        await release.wait()
        await db.execute("INSERT INTO original_preference_writes VALUES (2)")
        completed.append(True)

    service.set_preference_persistence_callback(persist)
    service.clear_model_preference()
    (writer,) = tuple(service._preference_persistence_tasks)
    await asyncio.wait_for(entered.wait(), 5)
    transport = httpx.AsyncClient(trust_env=False)
    client = openai.AsyncOpenAI(
        api_key="inert-local-native-close-test", http_client=transport
    )
    service.providers.append({"name": "actual-independent-sdk-owner", "client": client})
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    failure = RuntimeError("original preference drain wrapper refused")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if (
            not refused
            and code is not None
            and code.co_qualname == "LLMService.drain_preference_persistence"
        ):
            refused.append((coroutine, operation))
            raise failure
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        loop.set_task_factory(factory)
        with pytest.raises(RuntimeError) as caught:
            await service.close()
        loop.set_task_factory(previous)
        assert refused and any(error is failure for error in _explicit_control_chain(caught.value))
        debts = tuple(execution_cleanup_debts(caught.value))
        assert len(debts) == 1
        original_admission, accepted_task = debts[0]
        assert type(original_admission) is OwnedTaskAdmission
        assert accepted_task is None and original_admission.task is None
        assert original_admission._acceptance_unknown
        assert original_admission._reconciled_tasks == ()
        assert transport.is_closed, (
            "refused writer wrapper skipped independent original SDK cleanup"
        )
        assert (
            not writer.done()
            and worker.is_alive()
            and db.backend._connection is connection
        ), "missing wrapper was mistaken for a physically settled original writer"
        assert service._usage_db is db and service.has_pending_close_owners()
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        )
        release.set()
        await await_owned_task(writer)
        with pytest.raises(RuntimeError) as joined:
            await service.wait_for_close_completion()
        assert any(error is failure for error in _explicit_control_chain(joined.value))
        assert len(execution_cleanup_debts(joined.value)) == 1
        assert execution_cleanup_debts(joined.value)[0][0] is original_admission
        assert execution_cleanup_debts(joined.value)[0][1] is None
        # The genuine writer/DB are physically settled. The unproved wrapper
        # admission remains a separate original error, not a successful ACK.
        assert (
            completed == [True] and not worker.is_alive() and service._usage_db is None
        )
        assert not service.has_pending_close_owners()
    finally:
        loop.set_task_factory(previous)
        release.set()
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await await_owned_task(writer)
        await client.close()
        try:
            with pytest.raises(RuntimeError) as cleanup:
                await service.wait_for_close_completion()
            assert any(error is failure for error in _explicit_control_chain(cleanup.value))
            assert execution_cleanup_debts(cleanup.value)
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_actual_sdk_failed_close_ack_keeps_unpublished_service_fatal(
    tmp_path, monkeypatch
):
    # Reuse the actual loopback HTTPX lifetime fixture; its concrete supervisor
    # never starts/adopts a process. SDK.close operates on that ORIGINAL client.
    from tests.unit.test_server_original_shutdown_dependencies import (
        native_phoenix_proxy,
    )

    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    service, owners, calls = agent.llm_service, {}, []
    original = OSError("actual original SDK transport close ACK lost")
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    try:
        async with native_phoenix_proxy(tmp_path) as (_supervisor, transport, closed):
            client = openai.AsyncOpenAI(
                api_key="inert-local-native-close-test",
                base_url="http://127.0.0.1/unused-sdk-endpoint",
                http_client=transport,
            )
            service.providers.append(
                {"name": "original-loopback-sdk", "client": client}
            )
            native_close = transport.aclose

            async def lost_ack():
                await native_close()
                calls.append(True)
                raise original

            with monkeypatch.context() as fault:
                fault.setattr(transport, "aclose", lost_ack)
                with pytest.raises(BaseException) as caught:
                    await retire_unpublished_llm(
                        owners, "actual-sdk-owner", service, ()
                    )
                assert original in tuple(_explicit_control_chain(caught.value))
                assert is_execution_control_error(caught.value)
                assert tuple(_explicit_control_chain(original)) == graph
                assert tuple(execution_cleanup_debts(original)) == debts
                assert any(
                    resource is client and closer is not None and closer.done()
                    for resource, closer in execution_cleanup_debts(caught.value)
                )
                await asyncio.wait_for(closed.wait(), 2)
                assert transport.is_closed and service.has_pending_close_owners()
                assert owners["actual-sdk-owner"].provider is service
                with pytest.raises(BaseException) as repeated:
                    await retire_unpublished_llm(
                        owners, "actual-sdk-owner", service, (), join_native=True
                    )
                assert original in tuple(_explicit_control_chain(repeated.value)) and calls == [True]
                assert tuple(_explicit_control_chain(original)) == graph
                assert tuple(execution_cleanup_debts(original)) == debts
                assert not agent._runtime_publication_ready.is_set()
    finally:
        try:
            try:
                await agent.shutdown()
            except BaseException as error:
                if original not in tuple(_explicit_control_chain(error)):
                    raise
        finally:
            await db.close()
