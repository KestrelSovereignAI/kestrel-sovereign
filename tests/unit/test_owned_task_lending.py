"""Exact host-resource lending through the canonical task gate."""

import asyncio
import contextvars

import pytest

from kestrel_sovereign import async_ownership as ownership
from kestrel_sovereign._async_ownership import OwnedCallbackScope, without_owned_coordinators
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionStreamLifecycle,
    ProcessRuntimeExecutionFence,
    SourceWithdrawal,
    bind_execution_cleanup,
    bind_execution_custody,
    close_execution_stream,
    execution_stream,
    execution_evidence_carriers,
    owned_execution_stream,
    require_execution_work,
)
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock


class CapacityLoan:
    """A bounded host resource with one accounting entry per original receipt."""

    def __init__(self, capacity=8):
        self.parent = asyncio.current_task()
        self.capacity = capacity
        self.outstanding = {}
        self.retired = []
        self.retirement_changed = asyncio.Event()
        self.denial = None

    def reserve(self, *, creator, original_parent, admission, structural_role):
        assert original_parent is self.parent or any(
            task is original_parent for _, task, _ in self.outstanding.values()
        )
        assert admission.task is None
        if len(self.outstanding) >= self.capacity:
            raise RuntimeError("test host capacity exhausted")
        key = object()
        self.outstanding[key] = (creator, None, admission)
        return key

    def lend(self, reservation, *, task, admission):
        creator, _, original_admission = self.outstanding[reservation]
        assert creator is asyncio.current_task()
        assert admission is original_admission and admission.task is task
        self.outstanding[reservation] = (creator, task, admission)
        if self.denial is not None:
            raise self.denial

    def retire(self, reservation, *, retirement):
        _, task, admission = self.outstanding[reservation]
        assert retirement.admission is admission
        if retirement.state == "complete":
            assert retirement.task is admission.task and retirement.task.done()
            assert task is None or retirement.task is task
        else:
            assert retirement.state == "nonacceptance"
            assert task is None and retirement.task is None
        del self.outstanding[reservation]
        self.retired.append(retirement)
        self.retirement_changed.set()


@pytest.mark.parametrize("flavor", ["asyncio", "auto"])
@pytest.mark.parametrize("substitute", [False, True])
def test_actual_uvicorn_loop_selection_preserves_native_loan_ownership(flavor, substitute, monkeypatch):
    import importlib.util
    import uvicorn
    loop = uvicorn.Config("unused:app", loop=flavor, log_config=None).get_loop_factory()()
    if flavor == "auto" and importlib.util.find_spec("uvloop") is not None:
        import uvloop
        assert isinstance(loop, uvloop.Loop), "the installed production loop was not qualified"
    async def scenario():
        host = CapacityLoan()
        allocated = []
        original_create = loop.create_task
        def replacement(coroutine, **options):
            allocated.append(coroutine)
            return original_create(coroutine, **options)
        async def body():
            assert ownership.capture_owned_task_lender() is not None
            assert asyncio.get_running_loop() is loop
        try:
            if substitute:
                monkeypatch.setattr(loop, "create_task", replacement)
            else:
                def forbid_global_task_scan(*args, **options):
                    pytest.fail("qualified native constructor scanned unrelated live Tasks")
                monkeypatch.setattr(asyncio, "all_tasks", forbid_global_task_scan)
            with ownership.bind_owned_task_lender(host):
                if substitute:
                    with pytest.raises(RuntimeError, match="canonical native constructor"):
                        ownership.create_owned_task(body())
                else:
                    await ownership.create_owned_task(body())
            assert not allocated and not host.outstanding
            assert len(host.retired) == (0 if substitute else 1)
        finally:
            monkeypatch.undo()
    try:
        loop.run_until_complete(scenario())
    finally:
        loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["direct", "forwarder"])
@pytest.mark.parametrize("failed_close", [False, True])
async def test_known_source_close_failure_retains_loan_without_changing_original_error(tmp_path, profile, failed_close):
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator
    host = CapacityLoan()
    source_admission = OwnedSourceAdmission()
    original = OSError("original physical close" if failed_close else "original body refusal")
    handle = (tmp_path / "original-source-resource").open("wb")
    closes = []
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not failed_close:
                raise original
            raise StopAsyncIteration
        async def aclose(self):
            closes.append(asyncio.current_task())
            if failed_close:
                raise original
            handle.close()
    stream = None
    try:
        with ownership.bind_owned_task_lender(host):
            if profile == "direct":
                stream = await create_owned_iterator(lambda: Source(), operation="source-retirement-proof",
                    source_admission=source_admission)
            else:
                stream = execution_stream(None, Source())
                source_admission = stream._source_admission
            with pytest.raises(BaseException) as caught:
                await anext(stream)
            assert original in execution_evidence_carriers(caught.value)
            if profile == "direct":
                assert caught.value is original
            outcome = ownership.reconcile_owned_task_admission(source_admission.producer)
            assert outcome.state == ("unknown" if failed_close else "complete")
            if failed_close:
                assert original in source_admission._state.failures
                assert original in execution_evidence_carriers(outcome.error)
                assert "_kestrel_native_retirement_debts" not in vars(original)
            assert len(host.outstanding) == (1 if failed_close else 0)
            assert len(host.retired) == (0 if failed_close else 1)
            assert handle.closed is not failed_close
            assert len(closes) == 1
    finally:
        if stream is not None:
            try:
                await stream.aclose()
            except BaseException as closed:
                assert original in execution_evidence_carriers(closed)
        handle.close()


@pytest.mark.asyncio
async def test_source_lend_veto_retires_original_partial_without_decorating_veto():
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator
    host = CapacityLoan()
    original = OSError("original source loan veto")
    host.denial = original
    source_admission = OwnedSourceAdmission()
    entered = []
    async def body():
        entered.append(asyncio.current_task())
        yield
    with ownership.bind_owned_task_lender(host):
        with pytest.raises(BaseException) as caught:
            await create_owned_iterator(body, operation="source-lend-veto", source_admission=source_admission)
        assert original in execution_evidence_carriers(caught.value)
    assert "_kestrel_native_retirement_debts" not in vars(original)
    assert "_kestrel_failed_owned_iterator" not in vars(original)
    assert not entered and not host.outstanding and len(host.retired) == 1
    assert ownership.reconcile_owned_task_admission(source_admission.producer).state == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("fenced", [False, True])
async def test_distributed_stop_retires_real_heartbeat_when_work_settlement_is_refused(tmp_path, fenced):
    from contextlib import nullcontext
    from kestrel_sovereign.stop.invocation import DistributedInvocationRegistry, DistributedInvocationStore
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    database = await AsyncDatabase.sqlite(str(tmp_path / "original-heartbeat.db"))
    store = DistributedInvocationStore(database)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(store, poll_seconds=.05)
    registry.start()
    heartbeat = registry._heartbeat
    assert heartbeat is not None and heartbeat.alive
    host = CapacityLoan(capacity=0)
    scope = (bind_execution_custody(ProcessRuntimeExecutionFence("heartbeat-close"))
             if fenced else nullcontext())
    try:
        with scope, ownership.bind_owned_task_lender(host):
            with pytest.raises(BaseException) as caught:
                await registry.close()
        # SQL abandonment IS Work and its refusal is still observable. The
        # independent original heartbeat nevertheless has physically retired.
        assert any("test host capacity exhausted" in str(error)
                   for error in execution_evidence_carriers(caught.value))
        assert not heartbeat.alive and heartbeat.unretired() is None
        assert not host.outstanding and not host.retired
    finally:
        assert await heartbeat.stop(timeout=3)
        await registry.close()
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fenced", [False, True])
async def test_real_sdk_subprocess_stop_needs_no_work_loan(fenced):
    from contextlib import nullcontext
    import signal
    import sys
    from kestrel_sdk.isolated_feature.client import SubprocessIsolatedFeatureClient
    from kestrel_sovereign._async_process import start_async_process, _signal_process_tree, _process_tree_alive
    from kestrel_sovereign.features.isolated_runtime import _await_owned_facade_lifecycle_operation

    # A real SDK handle owns a real idle private child. This covers physical
    # stop admission, not feature initialization/handshake or agent readiness.
    command = [sys.executable, "-B", "-c", "import time; time.sleep(30)"]
    process = await start_async_process(command)
    client = SubprocessIsolatedFeatureClient(command, process=process)
    host, late = CapacityLoan(capacity=0), []
    scope = (bind_execution_custody(ProcessRuntimeExecutionFence("sdk-stop"))
             if fenced else nullcontext())
    try:
        with scope, ownership.bind_owned_task_lender(host):
            # Starting/resuming is still Work and must not take this path.
            with pytest.raises(BaseException):
                await _await_owned_facade_lifecycle_operation(
                    asyncio.sleep(0), name="original-work-refused", on_late_task=late.append)
            assert process.returncode is None and client.process is process
            await _await_owned_facade_lifecycle_operation(
                client.stop(), name="original-sdk-physical-stop", cleanup=True,
                on_late_task=late.append)
            require_execution_work()
        assert process.returncode is not None and not _process_tree_alive(process)
        assert client.process is None and not late and not host.outstanding
    finally:
        await client.stop()
        if _process_tree_alive(process):
            _signal_process_tree(process, signal.SIGKILL)
        await asyncio.wait_for(process.wait(), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("fenced", [False, True])
async def test_phoenix_proxy_original_http_client_closes_without_work_capacity(tmp_path, fenced):
    from contextlib import nullcontext
    from fastapi import FastAPI
    from kestrel_sovereign.phoenix_supervisor import PhoenixSupervisor
    from kestrel_sovereign.server import _shutdown_phoenix

    supervisor = PhoenixSupervisor(working_dir=tmp_path / "original-phoenix")
    client = supervisor._proxy_client()
    app = FastAPI()
    app.state.phoenix = supervisor
    host = CapacityLoan(capacity=0)
    scope = (bind_execution_custody(ProcessRuntimeExecutionFence("phoenix-close"))
             if fenced else nullcontext())
    try:
        with scope, ownership.bind_owned_task_lender(host):
            assert await _shutdown_phoenix(app) is False
            require_execution_work()
        assert client.is_closed and supervisor._client is None
        assert app.state.phoenix is None and not host.outstanding
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("fenced", [False, True])
async def test_tap_eof_closes_original_file_without_work_capacity(native_tool_effect_agent, tmp_path, fenced):
    from contextlib import nullcontext
    from kestrel_sovereign.streams.tap import _TapRequestFence, _TapResponseOwner

    agent = await native_tool_effect_agent(did="did:test:original-tap-retirement")
    request = "original-tap-retirement"
    generation = agent.register_active_request(request)
    handle = (tmp_path / "original-tap-body").open("w+b")
    handle.write(b"original body")
    handle.seek(0)
    closes = []

    class Body:
        def __aiter__(self):
            return self

        async def __anext__(self):
            chunk = handle.read()
            if chunk:
                return chunk
            raise StopAsyncIteration

        async def aclose(self):
            closes.append(ownership.capture_owned_task_lender())
            if fenced:
                with pytest.raises(ExecutionAuthorityError, match="cleanup"):
                    require_execution_work()
            else:
                require_execution_work()
            handle.close()

    stream = Body()
    response = _TapResponseOwner(stream, _TapRequestFence(agent, request, generation), ())
    host = CapacityLoan(capacity=0)
    scope = (bind_execution_custody(ProcessRuntimeExecutionFence("tap-retirement"))
             if fenced else nullcontext())
    try:
        with scope, ownership.bind_owned_task_lender(host):
            assert [chunk async for chunk in response] == [b"original body"]
            assert ownership.capture_owned_task_lender() is not None
            require_execution_work()
        assert handle.closed and closes == [None]
        assert response.retired and not host.outstanding and not host.retired
        # Tap retirement releases its retained delivery; the original request
        # producer still owns the first registration until it also completes.
        assert generation in agent._active_generations_for_request(request)
        agent._cleanup_cancelled_request(request, generation=generation)
        assert generation not in agent._active_generations_for_request(request)
    finally:
        handle.close()
        agent._cleanup_cancelled_request(request, generation=generation)


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["http-response", "http-inline", "database-facade", "database-backend", "database-owned-wrapper", "hold-preflight", "a2a-task-manager", "host-context"])
@pytest.mark.parametrize("fenced", [False, True])
async def test_public_physical_closers_do_not_reserve_new_work(tmp_path, resource, fenced):
    from contextlib import nullcontext
    host = CapacityLoan(capacity=0)
    scope = (bind_execution_custody(ProcessRuntimeExecutionFence("public-physical-close"))
             if fenced else nullcontext())
    if resource.startswith("http-"):
        from kestrel_sovereign.endpoints.closing_streaming_response import ClosingStreamingResponse
        handle = (tmp_path / "response-body").open("w+b")
        handle.write(b"original body")
        handle.seek(0)
        close_captures = []

        class Body:
            def __aiter__(self):
                return self

            async def __anext__(self):
                chunk = handle.read(64)
                if chunk:
                    return chunk
                raise StopAsyncIteration

            async def aclose(self):
                close_captures.append(ownership.capture_owned_task_lender())
                if fenced:
                    with pytest.raises(ExecutionAuthorityError, match="cleanup"):
                        require_execution_work()
                else:
                    require_execution_work()
                handle.close()

        body = Body()
        response = ClosingStreamingResponse(body, close_in_stream_task=resource == "http-inline")

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        try:
            with scope, ownership.bind_owned_task_lender(host):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
                assert ownership.capture_owned_task_lender() is not None
                require_execution_work()
            assert handle.closed and close_captures and all(capture is None for capture in close_captures)
        finally:
            handle.close()
    else:
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        database = await AsyncDatabase.sqlite(str(tmp_path / "original-database.sqlite"))
        connection = database._backend._connection
        worker = getattr(connection, "_thread", connection)
        if resource == "a2a-task-manager":
            from kestrel_sovereign.a2a.scope import NativeRecipientScope
            from kestrel_sovereign.a2a.task_manager import TaskManager
            manager = TaskManager.from_backend(database._backend,
                scope=NativeRecipientScope("did:test:original-close", "original-tenant"),
                owns_backend=True, include_memory=False, include_feedback=False)
        try:
            assert worker.is_alive()
            with scope, ownership.bind_owned_task_lender(host):
                if resource == "a2a-task-manager":
                    await manager.close()
                elif resource == "host-context":
                    from kestrel_sovereign.host_features.context import _close_partial_host_resources
                    await _close_partial_host_resources(None, None, None, database, owners={})
                elif resource == "hold-preflight":
                    from kestrel_sovereign.hold.state import _close_postgres_preflight_backends
                    assert await _close_postgres_preflight_backends(database._backend) == ()
                elif resource == "database-owned-wrapper":
                    from kestrel_sovereign.storage.async_database import _close_owned_database
                    await _close_owned_database(database)
                else:
                    await (database.close() if resource == "database-facade" else database._backend.close())
                assert ownership.capture_owned_task_lender() is not None
                require_execution_work()
            assert not worker.is_alive()
        finally:
            # Original resource disposal after a negative assertion is not an
            # acknowledgement to the host policy under test.
            await database.close()
    assert not host.outstanding and not host.retired


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["factory-close", "driver-retirement"])
async def test_sqla_original_workers_retire_without_work_capacity(entry):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from kestrel_sovereign.storage.sqla.session import SovereignSqlaSessionFactory
    factory = SovereignSqlaSessionFactory(create_async_engine("sqlite+aiosqlite:///:memory:"))
    host = CapacityLoan(capacity=0)
    try:
        async with factory.engine.connect() as connection:
            assert await connection.scalar(text("SELECT 1")) == 1
        workers = tuple(getattr(connection, "_thread", connection)
                        for connection in factory._sqlite_connections)
        assert workers and all(worker.is_alive() for worker in workers)
        if entry == "driver-retirement":
            # The synchronous pool hook requires the actual original engine
            # disposal owner, not a fabricated completed acknowledgement.
            factory._engine_dispose_task = ownership.create_owned_task(
                factory.engine.dispose(), lender_capture=None,
            )
        with ownership.bind_owned_task_lender(host):
            if entry == "driver-retirement":
                factory._begin_sqlite_connection_retirement()
                # Observe the SAME accepted native closers before final engine
                # disposal; no work-capacity refusal can stand in for retirement.
                for task in tuple(factory._driver_close_tasks):
                    outcome = await ownership.await_owned_task(task)
                    assert outcome.error is None
            await factory.close()
        assert all(not worker.is_alive() for worker in workers)
        assert not host.outstanding and not host.retired
    finally:
        await factory.close()


@pytest.mark.asyncio
async def test_native_http_exchange_retirement_needs_no_work_capacity(tmp_path):
    import httpx
    from kestrel_sovereign.a2a.native_inprocess_http import _NativeInProcessHTTPTransport
    handle = (tmp_path / "original-native-upload").open("w+b")
    retired = asyncio.Event()
    host = CapacityLoan(capacity=0)

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"original upload"

        async def aclose(self):
            assert ownership.capture_owned_task_lender() is None
            handle.close()

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"original response", "more_body": True})
            while (await receive())["type"] != "http.disconnect":
                pass
        finally:
            retired.set()

    transport = _NativeInProcessHTTPTransport(application=app, public_base_url="https://native.test",
        require_binding=lambda scope: None, buffer_bytes=65536)
    try:
        async with httpx.AsyncClient(transport=transport) as client:
            request = httpx.Request("POST", "https://native.test/native", stream=Body(), headers={"Host": "native.test"},
                extensions={"timeout": {"read": 5, "write": 5}})
            response = await client.send(request, stream=True)
            try:
                assert await anext(response.aiter_raw()) == b"original response"
                with ownership.bind_owned_task_lender(host):
                    # Public transport disposal also reaches the exchange
                    # without HTTPX's separate response-close cleanup scope.
                    await transport.aclose()
            finally:
                await response.aclose()
        assert retired.is_set() and handle.closed and not transport.exchanges
        assert not host.outstanding and not host.retired
    finally:
        handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["native-stream", "openai", "anthropic", "provider-client", "ollama-abandoned"])
@pytest.mark.parametrize("withdrawn", [False, True])
async def test_real_sdk_physical_close_uses_no_second_work_loan(tmp_path, profile, withdrawn):
    import json
    from types import SimpleNamespace
    import httpx2 as httpx
    from kestrel_sovereign.execution_custody import (
        close_native_execution_stream, execution_cleanup_debts,
    )
    host = CapacityLoan(capacity=1)
    fence = ProcessRuntimeExecutionFence("original-sdk-reader")
    handle = (tmp_path / "original-sdk-response").open("w+b")
    handle.write(b"original physical response")
    closes, requests = [], []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            chunk = {"id": "offline", "object": "chat.completion.chunk", "created": 1,
                "model": "offline", "choices": [{"index": 0,
                "delta": {"content": "original"}, "finish_reason": "stop"}]}
            yield b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n"

        async def aclose(self):
            closes.append(asyncio.current_task())
            assert ownership.capture_owned_task_lender() is None
            with pytest.raises(ExecutionAuthorityError):
                require_execution_work()
            handle.close()

    body = Body()
    def respond(request):
        assert request.url.host == "sdk.offline.invalid"
        requests.append(request)
        if withdrawn:
            fence.retire()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = None

    async def original():
        nonlocal client
        if profile == "native-stream":
            response = httpx.Response(200, stream=body)
            if withdrawn:
                fence.retire()
            await close_native_execution_stream(response)
        elif profile == "provider-client":
            from kestrel_sovereign.llm.lifecycle import start_llm_close_task, harvest_llm_close_task
            class PhysicalTransport(httpx.AsyncBaseTransport):
                async def handle_async_request(self, request):
                    pytest.fail("provider close must not dispatch a request")
                async def aclose(self):
                    await body.aclose()
            physical = httpx.AsyncClient(transport=PhysicalTransport())
            owner = SimpleNamespace()
            try:
                if withdrawn:
                    fence.retire()
                closer = start_llm_close_task(owner, "original-client", physical.aclose, resource=physical)
                closed = await ownership.await_owned_task(closer)
                assert closed.error is None
                assert harvest_llm_close_task(owner, "original-client", closer) is None
            finally:
                # Failure-path disposal is observed separately and cannot turn
                # the failed original retirement into a host acknowledgement.
                with bind_execution_cleanup(None):
                    await physical.aclose()
        elif profile == "ollama-abandoned":
            from kestrel_sovereign.llm.cancellation import await_or_cancelled
            response = httpx.Response(200, stream=body)
            async def opened():
                return response
            acquired = asyncio.create_task(opened())
            await acquired
            if withdrawn:
                fence.retire()
            await await_or_cancelled(acquired, lambda: True,
                abandoned_result_cleanup=close_native_execution_stream)
        elif profile == "openai":
            import openai
            from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
            client = openai.AsyncOpenAI(api_key="offline-transport-only", base_url="http://sdk.offline.invalid", http_client=http, max_retries=0)
            async for _ in OpenAIAdapter().get_streaming_response(client=client, model="offline",
                    messages=[{"role": "user", "content": "original"}]):
                pass
        else:
            import anthropic
            from kestrel_sovereign.llm.anthropic_adapter import _anthropic_stream_with_retry
            client = anthropic.AsyncAnthropic(api_key="offline-transport-only", base_url="http://sdk.offline.invalid", http_client=http, max_retries=0)
            async with _anthropic_stream_with_retry(client,
                    {"model": "offline", "messages": [{"role": "user", "content": "original"}], "max_tokens": 8},
                    route=None, retry_open=False):
                pass

    try:
        with bind_execution_custody(fence), ownership.bind_owned_task_lender(host):
            task = ownership.create_owned_task(original())
            result = await ownership.await_owned_task(task)
            if profile == "ollama-abandoned":
                terminal = result.error or result.cancellation
                assert any(isinstance(error, asyncio.CancelledError)
                           for error in execution_evidence_carriers(terminal))
                assert not execution_cleanup_debts(terminal)
            elif withdrawn and profile in {"openai", "anthropic"}:
                assert any(isinstance(error, ExecutionAuthorityError)
                           for error in execution_evidence_carriers(result.error))
                assert not execution_cleanup_debts(result.error)
            else:
                assert result.error is None and result.cancellation is None
            async with asyncio.timeout(3):
                while host.outstanding:
                    host.retirement_changed.clear()
                    await host.retirement_changed.wait()
        assert handle.closed and len(closes) == 1
        assert len(host.retired) == 1, "physical retirement acquired a second WORK loan"
        assert len(requests) == (1 if profile in {"openai", "anthropic"} else 0)
    finally:
        handle.close()
        with bind_execution_cleanup(None):
            await (client.close() if client is not None else http.aclose())


@pytest.mark.asyncio
async def test_sqlite_original_rollback_drain_needs_no_work_capacity(tmp_path):
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
    backend = SQLiteBackend(str(tmp_path / "rollback-drain.sqlite"))
    await backend.connect()
    connection = backend._connection
    worker = getattr(connection, "_thread", connection)
    await backend.execute("CREATE TABLE original_effect(value INTEGER)")
    await connection.execute("BEGIN")
    await connection.execute("INSERT INTO original_effect VALUES (1)")
    host = CapacityLoan(capacity=0)
    try:
        with ownership.bind_owned_task_lender(host):
            backend._handoff_cancelled_write(connection)
            drain = backend._cancelled_write_drain
            assert drain is not None
            outcome = await ownership.await_owned_task(drain)
            assert outcome.error is None and backend._cancelled_write_drain_error is None
        assert not connection.in_transaction
        row = await backend.fetch_one("SELECT COUNT(*) AS count FROM original_effect")
        assert row[0] == 0 and not host.outstanding and not host.retired
    finally:
        # Observe the original close even on the negative admission run. Any
        # retained admission error is evidence, not a reason to skip disposal.
        retained_error = backend._cancelled_write_drain_error
        closer = ownership.create_owned_task(backend.close(), lender_capture=None)
        closed = await ownership.await_owned_task(closer)
        assert not worker.is_alive()
        if closed.error is not None:
            assert retained_error in execution_evidence_carriers(closed.error)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sqlite_retained_physical_close_reaper_needs_no_work_capacity(tmp_path):
    import threading
    import time
    import aiosqlite
    from kestrel_sovereign.storage.db.sqlite import _close_aiosqlite_connection
    connection = await aiosqlite.connect(str(tmp_path / "retained-worker.sqlite"))
    worker = getattr(connection, "_thread", connection)
    entered, release = threading.Event(), threading.Event()
    retained = {}
    host = CapacityLoan(capacity=0)

    def hold():
        entered.set()
        if not release.wait(10):
            raise TimeoutError("original native worker was not released")
        return 1

    def original_query():
        cursor = connection._conn.execute("SELECT original_hold()")
        try:
            return cursor.fetchone()
        finally:
            cursor.close()

    await connection.create_function("original_hold", 0, hold)
    query = ownership.create_owned_task(connection._execute(original_query), lender_capture=None)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        with ownership.bind_owned_task_lender(host):
            with pytest.raises(BaseException):
                await _close_aiosqlite_connection(connection, retained_closes=retained,
                    deadline=time.monotonic() + 0.03, cancel_close_task_on_timeout=False)
            original = retained[connection]
            assert original.retirement_task is not None, "original retirement reaper requested WORK capacity"
        assert not host.outstanding and not host.retired
    finally:
        release.set()
        queried = await ownership.await_owned_task(query)
        if queried.error is not None:
            import sqlite3
            assert isinstance(queried.error, sqlite3.OperationalError)
        original = retained.get(connection)
        if original is not None and original.close_task is not None:
            await ownership.await_owned_task(original.close_task)
        else:
            await connection.close()
        if original is not None and original.retirement_task is not None:
            await ownership.await_owned_task(original.retirement_task)
        await asyncio.to_thread(worker.join, 3)
        assert not worker.is_alive()


@pytest.mark.asyncio
async def test_join_cannot_replace_failed_original_completion_capture(tmp_path):
    receipt = ownership.OwnedTaskAdmission()
    refusal = OSError("original completion capture registration failed")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    path = tmp_path / "unadmitted-payload"

    class UncapturedTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "close_unstarted":
                raise refusal
            return super().add_done_callback(callback, context=context)

    def factory(loop, coroutine, **options):
        return UncapturedTask(coroutine, loop=loop, **options)

    async def body():
        path.write_text("unadmitted body ran")

    payload = body()
    try:
        loop.set_task_factory(factory)
        with pytest.raises(BaseException) as caught:
            ownership.create_owned_task(payload, admission=receipt, lender_capture=None)
    finally:
        loop.set_task_factory(previous)
    assert refusal in execution_evidence_carriers(caught.value)
    assert receipt.task is not None
    outcome = await ownership.await_owned_task(receipt.task)
    assert refusal in execution_evidence_carriers(outcome.error)
    assert receipt.task.done() and payload.cr_frame is None and not path.exists()
    assert not receipt._completion_capture_registered
    assert ownership.reconcile_owned_task_admission(receipt).state == "unknown"


@pytest.mark.asyncio
async def test_running_original_source_closer_is_pending_not_final_unknown(tmp_path):
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator
    from kestrel_sovereign._async_ownership import _record_source_retirement

    handle = (tmp_path / "original-structural-source").open("w+b")
    source = OwnedSourceAdmission(retirement=ownership.OwnedTaskAdmission())
    host = CapacityLoan()
    entered, release = asyncio.Event(), asyncio.Event()
    source._state.resource_required = True
    async def retire_file():
        entered.set()
        await release.wait()
        handle.close()
        _record_source_retirement(source, closed=True)
    async def body():
        if False:
            yield
    closer = ownership.create_owned_task(retire_file(), admission=source.retirement,
        lender_capture=None, retirement_resource=handle)
    iterator = None
    try:
        with ownership.bind_owned_task_lender(host):
            iterator = await create_owned_iterator(body, operation="original-structural-closer",
                source_admission=source)
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)
            await entered.wait()
            await ownership.await_owned_task(source.producer.task)
            assert source.producer.task.done() and not closer.done() and not handle.closed
            retirement = ownership.reconcile_owned_task_admission(source.producer)
            assert retirement.state == "pending"
            assert closer in retirement.pending_tasks
            assert source.producer.task not in retirement.pending_tasks
            assert len(host.outstanding) == 1 and not host.retired
            release.set()
            await ownership.await_owned_task(closer)
            async with asyncio.timeout(3):
                while host.outstanding:
                    host.retirement_changed.clear()
                    await host.retirement_changed.wait()
            assert handle.closed and len(host.retired) == 1
    finally:
        release.set()
        await ownership.await_owned_task(closer)
        if iterator is not None:
            await iterator.aclose()
        handle.close()


@pytest.mark.asyncio
async def test_unknown_original_watcher_keeps_the_producer_loan_after_producer_join():
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator
    from kestrel_sovereign.execution_custody import execution_cleanup_debts
    host = CapacityLoan()
    source_admission = OwnedSourceAdmission()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    refusal = OSError("original watcher allocation is uncertain")
    refused = []
    entered = []
    def refuse_watcher(loop, coroutine, **options):
        refused.append(coroutine)
        raise refusal
    lend = host.lend
    def publish_then_change_factory(*args, **options):
        lend(*args, **options)
        # The producer already has its canonical native constructor. This
        # subsequent structural watcher is deliberately not a WORK loan.
        loop.set_task_factory(refuse_watcher)
    host.lend = publish_then_change_factory
    async def source():
        entered.append(asyncio.current_task())
        yield
    try:
        with ownership.bind_owned_task_lender(host):
            with pytest.raises(BaseException) as caught:
                await create_owned_iterator(source, operation="unknown-original-watcher",
                    source_admission=source_admission)
        loop.set_task_factory(previous)
        assert refusal in execution_evidence_carriers(caught.value)
        assert not entered and source_admission.producer.task.done()
        assert len(refused) == 1 and refused[0].cr_frame is None
        assert source_admission.watcher.task is None
        assert ownership.reconcile_owned_task_admission(source_admission.producer).state == "unknown"
        assert len(host.outstanding) == 1 and not host.retired
        debts = execution_cleanup_debts(caught.value)
        assert debts and all(task is not source_admission.producer.task for _, task in debts)
        assert "_kestrel_native_retirement_debts" not in vars(refusal)
    finally:
        loop.set_task_factory(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["configured_factory", "substituted_constructor", "changed_during_reserve"])
async def test_loan_requires_canonical_constructor_before_any_borrower_allocation(boundary, monkeypatch):
    host = CapacityLoan()
    receipt = ownership.OwnedTaskAdmission()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    native_create = asyncio.create_task
    allocated = []
    def factory(loop, coroutine, **options):
        task = asyncio.Task(coroutine, loop=loop, **options)
        allocated.append(task)
        return task
    def substitute(coroutine, **options):
        task = native_create(coroutine, **options)
        allocated.append(task)
        return task
    reserve = host.reserve
    def change_during_reserve(**options):
        reservation = reserve(**options)
        loop.set_task_factory(factory)
        return reservation
    async def body():
        pytest.fail("noncanonical constructor entered a borrowed payload")
    payload = body()
    try:
        if boundary == "configured_factory":
            loop.set_task_factory(factory)
        elif boundary == "substituted_constructor":
            monkeypatch.setattr(asyncio, "create_task", substitute)
        else:
            host.reserve = change_during_reserve
        with ownership.bind_owned_task_lender(host):
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(payload, admission=receipt)
    finally:
        loop.set_task_factory(previous)
        monkeypatch.undo()
        if allocated:
            await asyncio.gather(*allocated, return_exceptions=True)
    assert not allocated and receipt.task is None and payload.cr_frame is None
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"
    assert not host.outstanding
    assert len(host.retired) == (1 if boundary == "changed_during_reserve" else 0), (
        "an unsupported constructor must be rejected before reserving the host resource")


@pytest.mark.asyncio
async def test_lends_exact_task_before_payload_and_retains_physical_finalizer():
    host = CapacityLoan()
    release = asyncio.Event()
    entered = asyncio.Event()
    receipt = ownership.OwnedTaskAdmission()

    async def body():
        task = asyncio.current_task()
        assert list(host.outstanding.values()) == [(host.parent, task, receipt)]
        try:
            entered.set()
            return task
        finally:
            await release.wait()

    with ownership.bind_owned_task_lender(host):
        task = ownership.create_owned_task(body(), admission=receipt)
        try:
            await entered.wait()
            assert not host.retired
            assert ownership.reconcile_owned_task_admission(receipt).state == "pending"
        finally:
            release.set()
        assert await task is task
    assert not host.outstanding
    assert [item.admission for item in host.retired] == [receipt]


@pytest.mark.asyncio
async def test_explicit_fresh_context_lends_to_child_but_inherited_route_does_not():
    host = CapacityLoan()
    marker = contextvars.ContextVar("test_lender_private_marker", default=None)
    token = marker.set("original")

    async def fresh_body():
        assert marker.get() is None
        assert ownership.capture_owned_task_lender() is not None
        await ownership.create_owned_task(nested_body())

    async def nested_body():
        assert marker.get() is None
        assert ownership.capture_owned_task_lender() is not None

    async def raw_body():
        assert marker.get() == "original"
        assert ownership.capture_owned_task_lender() is None
        await ownership.create_owned_task(independent_body())

    async def independent_body():
        assert ownership.capture_owned_task_lender() is None
        assert not host.outstanding

    try:
        with ownership.bind_owned_task_lender(host):
            capture = ownership.capture_owned_task_lender()
            await asyncio.create_task(raw_body())
            await ownership.create_owned_task(independent_body(), context=contextvars.copy_context())
            await ownership.create_owned_task(
                fresh_body(), context=contextvars.Context(), lender_capture=capture,
            )
        with pytest.raises(RuntimeError, match="lender.*no longer live"):
            ownership.create_owned_task(independent_body(), lender_capture=capture)
    finally:
        marker.reset(token)
    assert not host.outstanding and len(host.retired) == 2


@pytest.mark.asyncio
async def test_already_lent_child_can_start_after_its_original_creator_finishes():
    host = CapacityLoan(capacity=2)
    children, captures, entered = [], [], []

    async def child():
        entered.append(asyncio.current_task())
        require_execution_work()

    async def creator():
        captures.append(ownership.capture_owned_task_lender())
        children.append(ownership.create_owned_task(child()))
        # No yield: the creator is done before the child's first native step.

    with ownership.bind_owned_task_lender(host):
        parent = ownership.create_owned_task(creator())
        await parent
        outcome = await ownership.await_owned_task(children[0])
        assert outcome.error is None and outcome.cancellation is None
        assert entered == children and parent.done()
        # An already-admitted child owns its loan; an old dead-parent capture
        # still cannot reserve NEW work after that parent's completion.
        with pytest.raises(RuntimeError, match="no longer live"):
            ownership.create_owned_task(child(), lender_capture=captures[0])
        async with asyncio.timeout(3):
            while host.outstanding:
                host.retirement_changed.clear()
                await host.retirement_changed.wait()
    assert not host.outstanding and len(host.retired) == 2


@pytest.mark.parametrize("flavor", ["asyncio", "auto"])
def test_foreign_loop_bound_native_constructor_is_refused_before_reserve(flavor, monkeypatch):
    import uvicorn
    loop = uvicorn.Config("unused:app", loop=flavor, log_config=None).get_loop_factory()()
    foreign = asyncio.new_event_loop()
    entered = []

    async def scenario():
        host = CapacityLoan()
        async def body():
            entered.append(asyncio.current_task())
        with ownership.bind_owned_task_lender(host):
            monkeypatch.setattr(loop, "create_task", foreign.create_task)
            try:
                with pytest.raises(RuntimeError, match="canonical native constructor"):
                    ownership.create_owned_task(body())
            finally:
                monkeypatch.undo()
        assert not entered and not host.outstanding and not host.retired
        assert not asyncio.all_tasks(foreign), "foreign native allocation preceded refusal"

    try:
        loop.run_until_complete(scenario())
    finally:
        monkeypatch.undo()
        # The negative pre-fix case still owns its exact isolated foreign loop
        # and all original wrappers allocated there; observe them before close.
        originals = tuple(asyncio.all_tasks(foreign))
        if originals:
            async def join_originals():
                await asyncio.gather(*originals, return_exceptions=True)
            foreign.run_until_complete(join_originals())
        foreign.close()
        loop.close()


@pytest.mark.asyncio
async def test_capacity_refusal_precedes_any_native_allocation():
    host = CapacityLoan(capacity=0)
    original_tasks = asyncio.all_tasks()
    receipt = ownership.OwnedTaskAdmission()

    async def body():
        pytest.fail("refused payload entered")

    payload = body()
    with ownership.bind_owned_task_lender(host):
        with pytest.raises(RuntimeError, match="test host capacity exhausted"):
            ownership.create_owned_task(payload, admission=receipt)
    assert asyncio.all_tasks() == original_tasks and receipt.task is None
    assert payload.cr_frame is None and not host.outstanding


@pytest.mark.asyncio
async def test_refused_lend_keeps_original_receipt_through_pre_entry_cancellation():
    host = CapacityLoan()
    host.denial = RuntimeError("test host vetoed accepted task")
    receipt = ownership.OwnedTaskAdmission()

    async def body():
        pytest.fail("failed start gate entered")

    with ownership.bind_owned_task_lender(host):
        with pytest.raises(RuntimeError, match="test host vetoed accepted task"):
            ownership.create_owned_task(body(), admission=receipt)
        task = receipt.task
        assert task is not None and not host.retired
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not host.outstanding and len(host.retired) == 1
    retired = host.retired[0]
    assert retired.task is task and retired.error is not None
    assert ownership.reconcile_owned_task_admission(receipt) == retired


@pytest.mark.asyncio
async def test_opaque_factory_is_refused_before_borrower_acceptance():
    host = CapacityLoan()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    receipt = ownership.OwnedTaskAdmission()
    invoked = False

    def factory(loop, coroutine, **kwargs):
        async def opaque():
            return await coroutine

        return asyncio.Task(opaque(), loop=loop, **kwargs)

    async def body():
        nonlocal invoked
        invoked = True

    with ownership.bind_owned_task_lender(host):
        loop.set_task_factory(factory)
        try:
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(body(), admission=receipt)
        finally:
            loop.set_task_factory(previous)
        assert receipt.task is None
    assert not invoked and not host.retired
    assert not host.outstanding
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["callback", "privacy_factory"])
async def test_fresh_callback_uses_original_capture_not_actual_creators_inheritance(boundary):
    host = CapacityLoan()
    marker = contextvars.ContextVar("test_lender_callback_marker", default=None)
    token = marker.set("private original context")
    actual_creator = None

    async def body():
        assert marker.get() is None
        assert ownership.capture_owned_task_lender() is not None
        assert [entry[:2] for entry in host.outstanding.values()] == [
            (actual_creator, asyncio.current_task()),
        ]

    async def callback(start):
        nonlocal actual_creator
        actual_creator = asyncio.current_task()
        assert actual_creator is not host.parent
        assert ownership.capture_owned_task_lender() is None
        await start(body, context=contextvars.Context())

    try:
        with ownership.bind_owned_task_lender(host):
            start = (OwnedCallbackScope().start if boundary == "callback" else
                     ReentrantTransitionLock().capture_native_work_factory(acquire_span=False))
            await asyncio.create_task(callback(start))
    finally:
        marker.reset(token)
    assert not host.outstanding and len(host.retired) == 1


@pytest.mark.asyncio
async def test_factory_captured_outside_a_root_cannot_enroll_in_active_root():
    host = CapacityLoan()
    lock = ReentrantTransitionLock()
    start = lock.capture_native_work_factory(acquire_span=False)

    async def body():
        assert ownership.capture_owned_task_lender() is None
        assert not host.outstanding

    with ownership.bind_owned_task_lender(host):
        await start(body)
        with without_owned_coordinators():
            await ownership.create_owned_task(body())
    assert not host.retired


@pytest.mark.asyncio
async def test_exact_cleanup_does_not_request_a_new_work_loan(tmp_path):
    host = CapacityLoan(capacity=1)
    closed = asyncio.Event()
    path = tmp_path / "original-resource"
    resource = path.open("wb")

    async def body():
        capture = ownership.capture_owned_task_lender()
        assert capture is not None and len(host.outstanding) == 1

        async def release():
            assert ownership.capture_owned_task_lender() is None
            with pytest.raises(ExecutionAuthorityError, match="cleanup"):
                require_execution_work()
            resource.close()
            closed.set()

        with bind_execution_cleanup(None):
            await ownership.create_owned_task(release(), lender_capture=capture)
        assert closed.is_set() and len(host.outstanding) == 1

    try:
        with bind_execution_custody(ProcessRuntimeExecutionFence("owned-task-lender-test")):
            with ownership.bind_owned_task_lender(host):
                await ownership.create_owned_task(body())
    finally:
        resource.close()
    assert len(host.retired) == 1 and not host.outstanding


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_reader", [False, True])
@pytest.mark.parametrize("lifecycle_factory", [False, True])
async def test_execution_stream_carries_original_work_loan_without_charging_its_closers(foreign_reader, lifecycle_factory):
    host = CapacityLoan(capacity=1)
    entered = []
    closed = []
    lifecycle = (ExecutionStreamLifecycle(ReentrantTransitionLock().capture_native_work_factory(), None, lambda: None)
                 if lifecycle_factory else None)

    async def source():
        producer = asyncio.current_task()
        entered.append(producer)
        assert ownership.capture_owned_task_lender() is not None
        assert [entry[1] for entry in host.outstanding.values()] == [producer]
        try:
            yield "original stream output"
        finally:
            closed.append(asyncio.current_task())

    async def consume(stream):
        assert await anext(stream) == "original stream output"
        assert len(host.outstanding) == 1

    with bind_execution_custody(ProcessRuntimeExecutionFence("owned-stream-lender-test")):
        with ownership.bind_owned_task_lender(host):
            stream = execution_stream(None, source(), lifecycle=lifecycle)
            try:
                if foreign_reader:
                    await asyncio.create_task(consume(stream))
                else:
                    await consume(stream)
            finally:
                await stream.aclose()
    assert entered == closed and len(entered) == 1
    assert not host.outstanding and host.retired[0].task is entered[0]


@pytest.mark.asyncio
async def test_ambiguous_exact_wrapper_never_receives_a_loan_or_dispatches():
    host = CapacityLoan()
    receipt = ownership.OwnedTaskAdmission()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    accepted, entered = [], []

    def duplicate(loop, coroutine, **options):
        accepted.extend(asyncio.Task(coroutine, loop=loop, **options) for _ in range(2))
        return accepted[0]

    async def body():
        entered.append(asyncio.current_task())

    with ownership.bind_owned_task_lender(host):
        loop.set_task_factory(duplicate)
        try:
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(body(), admission=receipt)
        finally:
            loop.set_task_factory(previous)
            await asyncio.gather(*accepted, return_exceptions=True)
    assert not entered and not accepted
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"
    assert not host.retired and not host.outstanding


@pytest.mark.asyncio
async def test_late_eager_wrapper_substitution_cannot_enter_the_payload():
    if not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager task factory unavailable")
    host = CapacityLoan()
    receipt = ownership.OwnedTaskAdmission()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    accepted, entered = [], []
    def factory(loop, coroutine, **options):
        def steal():
            accepted.append(asyncio.Task(coroutine, loop=loop, eager_start=True, **options))
        loop.call_soon(steal)
        return asyncio.Task(coroutine, loop=loop, **options)
    async def body():
        entered.append(asyncio.current_task())
    with ownership.bind_owned_task_lender(host):
        loop.set_task_factory(factory)
        try:
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(body(), admission=receipt)
        finally:
            loop.set_task_factory(previous)
        await asyncio.gather(*accepted, return_exceptions=True)
    assert not entered and not accepted
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"
    assert not host.retired and not host.outstanding


def test_foreign_loop_exact_wrapper_is_not_accepted_as_the_original_task():
    foreign = asyncio.new_event_loop()
    accepted, entered = [], []

    async def scenario():
        host = CapacityLoan()
        receipt = ownership.OwnedTaskAdmission()
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()

        def wrong_loop(_loop, coroutine, **options):
            task = asyncio.Task(coroutine, loop=foreign, **options)
            accepted.append(task)
            return task

        async def body():
            entered.append(asyncio.current_task())

        with ownership.bind_owned_task_lender(host):
            loop.set_task_factory(wrong_loop)
            try:
                with pytest.raises(RuntimeError, match="canonical native constructor"):
                    ownership.create_owned_task(body(), admission=receipt)
            finally:
                loop.set_task_factory(previous)
        assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"
        assert all(task is None for _, task, _ in host.outstanding.values())

    try:
        asyncio.run(scenario())
    finally:
        if accepted:
            async def join_originals():
                await asyncio.gather(*accepted, return_exceptions=True)
            foreign.run_until_complete(join_originals())
        foreign.close()
    assert not entered and not accepted


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["detached", "restored_cleanup", "withdrawn_stream"])
async def test_all_restrictions_suppress_even_explicit_old_lender_capture(boundary):
    from contextlib import ExitStack
    from kestrel_sovereign.execution_custody import (
        _StreamCleanupState, _bind_cleanup_flag, _bind_stream_cleanup,
    )

    host = CapacityLoan()
    fence = ProcessRuntimeExecutionFence("restricted-lender")
    async def independent():
        assert ownership.capture_owned_task_lender() is None
        assert not host.outstanding

    with ownership.bind_owned_task_lender(host):
        capture = ownership.capture_owned_task_lender()
        with ExitStack() as scope:
            if boundary == "detached":
                scope.enter_context(without_owned_coordinators())
            elif boundary == "restored_cleanup":
                scope.enter_context(_bind_cleanup_flag(True))
            else:
                scope.enter_context(_bind_stream_cleanup((_StreamCleanupState((fence,), True),)))
            await ownership.create_owned_task(independent(), lender_capture=capture)
            if boundary != "detached":
                with pytest.raises(RuntimeError, match="cleanup"):
                    with ownership.bind_owned_task_lender(CapacityLoan()):
                        pytest.fail("cleanup rebound WORK authority")
    assert not host.retired


@pytest.mark.asyncio
async def test_detached_body_context_cannot_restore_an_old_explicit_capture():
    host = CapacityLoan()
    with ownership.bind_owned_task_lender(host):
        capture = ownership.capture_owned_task_lender()
        with without_owned_coordinators():
            detached = contextvars.copy_context()
        async def independent():
            assert ownership.capture_owned_task_lender() is None
            assert not host.outstanding
        async def body():
            await ownership.create_owned_task(independent(), lender_capture=capture)
        await ownership.create_owned_task(body(), context=detached)
    assert not host.retired


@pytest.mark.asyncio
async def test_cleanup_suppression_covers_completion_publication_in_a_fresh_context():
    host = CapacityLoan()
    children = []
    async def body():
        assert ownership.capture_owned_task_lender() is None
    with ownership.bind_owned_task_lender(host):
        capture = ownership.capture_owned_task_lender()
        def completion(_task, _error):
            children.append(ownership.create_owned_task(body(), lender_capture=capture))
        with bind_execution_custody(ProcessRuntimeExecutionFence("cleanup-publication")), bind_execution_cleanup(None):
            await ownership.create_owned_task(body(), context=contextvars.Context(), on_completion=completion)
        await asyncio.gather(*children)
    assert not host.outstanding and not host.retired


@pytest.mark.asyncio
async def test_retained_callback_rechecks_its_producers_shared_withdrawal():
    host = CapacityLoan(capacity=2)
    finalizing, release = asyncio.Event(), asyncio.Event()
    scopes, entered = [], []
    attempt = None
    async def source():
        scopes.append(OwnedCallbackScope())
        try:
            yield "original"
        finally:
            finalizing.set()
            await release.wait()
    async def work():
        entered.append(asyncio.current_task())
    with bind_execution_custody(ProcessRuntimeExecutionFence("retained-source-callback")):
        with ownership.bind_owned_task_lender(host):
            stream = execution_stream(None, source())
            assert await anext(stream) == "original"
            close = asyncio.create_task(stream.aclose())
            try:
                await asyncio.wait_for(finalizing.wait(), 1)
                with pytest.raises(ExecutionAuthorityError, match="cleanup"):
                    attempt = scopes[0].start(work, context=contextvars.Context())
            finally:
                release.set()
                await close
                if attempt is not None:
                    await asyncio.gather(attempt, return_exceptions=True)
                scopes[0].close()
    assert not entered and not host.outstanding and len(host.retired) == 1


@pytest.mark.asyncio
async def test_independent_root_inside_detachment_cannot_revive_the_old_root():
    old, new = CapacityLoan(), CapacityLoan()
    async def independent():
        assert ownership.capture_owned_task_lender() is None
        assert not old.outstanding and not new.outstanding
    async def new_work():
        assert ownership.capture_owned_task_lender() is not None
        assert not old.outstanding and len(new.outstanding) == 1
    with ownership.bind_owned_task_lender(old):
        captured = ownership.capture_owned_task_lender()
        with without_owned_coordinators():
            with ownership.bind_owned_task_lender(new):
                await ownership.create_owned_task(independent(), lender_capture=captured)
                await ownership.create_owned_task(new_work())
    assert not old.retired and len(new.retired) == 1


@pytest.mark.asyncio
async def test_already_entered_body_context_remains_a_valid_non_eager_handoff():
    context = contextvars.copy_context()
    receipt = ownership.OwnedTaskAdmission()
    marker = contextvars.ContextVar("same-entered-context", default=None)
    context.run(marker.set, "original")
    async def body():
        assert marker.get() == "original"
        return asyncio.current_task()
    payload = body()
    try:
        task = context.run(ownership.create_owned_task, payload, context=context, admission=receipt)
        assert await task is receipt.task
        assert ownership.reconcile_owned_task_admission(receipt).state == "complete"
    finally:
        if receipt.task is None:
            payload.close()


@pytest.mark.asyncio
async def test_unfenced_physical_close_needs_no_spare_work_loan(tmp_path):
    host = CapacityLoan(capacity=0)
    resource = (tmp_path / "original-unfenced-close").open("wb")
    closed = []
    class Source:
        async def aclose(self):
            require_execution_work()  # Standalone authority semantics stay unchanged.
            assert ownership.capture_owned_task_lender() is None
            resource.close()
            closed.append(asyncio.current_task())
    try:
        with ownership.bind_owned_task_lender(host):
            await close_execution_stream(None, Source())
        assert resource.closed and len(closed) == 1 and not host.retired
    finally:
        resource.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["reserve", "publication", "lend", "before_step"])
async def test_retained_work_check_stays_live_through_admission_and_first_step(phase, monkeypatch):
    from kestrel_sovereign.execution_custody import _StreamCleanupState, _bind_stream_cleanup
    host = CapacityLoan()
    state = _StreamCleanupState((ProcessRuntimeExecutionFence("lender-admission-withdrawal"),))
    lends = []
    reserve, lend = host.reserve, host.lend
    def reserve_and_withdraw(**options):
        reservation = reserve(**options)
        if phase == "reserve":
            state.requested = True
        return reservation
    def lend_and_withdraw(reservation, **options):
        lends.append(options["task"])
        lend(reservation, **options)
        if phase == "lend":
            state.requested = True
    host.reserve, host.lend = reserve_and_withdraw, lend_and_withdraw
    receipt = ownership.OwnedTaskAdmission()
    entered = []
    from contextlib import ExitStack
    from kestrel_sovereign._async_ownership import bind_owned_coordinator
    coordinator = ReentrantTransitionLock()
    original_admit = coordinator.admit_owned_task
    def withdraw_after_publication(*args):
        admitted = original_admit(*args)
        state.requested = True
        return admitted
    async def body():
        entered.append(asyncio.current_task())
    with _bind_stream_cleanup((state,)), ownership.bind_owned_task_lender(host), ExitStack() as stack:
        try:
            if phase == "publication":
                monkeypatch.setattr(coordinator, "admit_owned_task", withdraw_after_publication)
                stack.enter_context(bind_owned_coordinator(coordinator))
            with pytest.raises(ExecutionAuthorityError, match="cleanup"):
                task = ownership.create_owned_task(body(), admission=receipt)
                if phase == "before_step":
                    state.requested = True
                    await task
        finally:
            if receipt.task is not None:
                await asyncio.gather(receipt.task, return_exceptions=True)
    assert not entered and not host.outstanding and len(host.retired) == 1
    if phase == "reserve":
        assert receipt.task is None and host.retired[0].state == "nonacceptance"
    assert len(lends) == (1 if phase in {"lend", "before_step"} else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", ["direct", "privacy", "callback"])
async def test_stream_paths_use_original_producer_receipt_and_captured_root(factory):
    from kestrel_sovereign._async_ownership import OwnedSourceAdmission, create_owned_iterator

    host = CapacityLoan(capacity=1)
    receipt = OwnedSourceAdmission()
    lock = ReentrantTransitionLock()
    scope = OwnedCallbackScope()
    def start_callback(body, context, *, admission, lender_capture):
        return scope.start(body, context=context, admission=admission, lender_capture=lender_capture)
    producer = []
    async def source():
        producer.append(asyncio.current_task())
        assert [entry[1:] for entry in host.outstanding.values()] == [
            (asyncio.current_task(), receipt.producer),
        ]
        assert ownership.capture_owned_task_lender() is not None
        yield "same original reader"

    with ownership.bind_owned_task_lender(host):
        iterator = await create_owned_iterator(
            source, operation="lender-direct-stream", source_admission=receipt,
            owner_factory=(lock.capture_native_work_factory() if factory == "privacy" else
                           start_callback if factory == "callback" else None),
        )
        try:
            assert iterator.producer_task is receipt.producer.task
            assert await anext(iterator) == "same original reader"
        finally:
            await iterator.aclose()
    assert host.retired[0].admission is receipt.producer and not host.outstanding


@pytest.mark.asyncio
async def test_eager_factory_is_refused_without_constructing_a_borrower():
    if not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager task factory unavailable")
    host = CapacityLoan()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    receipt = ownership.OwnedTaskAdmission()
    async def body():
        pytest.fail("withdrawn gate dispatched work")

    def factory(loop, coroutine, **options):
        task = asyncio.Task(coroutine, loop=loop, eager_start=True, **options)
        task.cancel("original factory cancellation")
        return task

    with ownership.bind_owned_task_lender(host):
        loop.set_task_factory(factory)
        try:
            with pytest.raises(RuntimeError, match="canonical native constructor"):
                ownership.create_owned_task(body(), admission=receipt)
            assert all(task is None for _, task, _ in host.outstanding.values())
        finally:
            loop.set_task_factory(previous)
            if receipt.task is not None:
                await asyncio.gather(receipt.task, return_exceptions=True)
    assert not host.outstanding and not host.retired and receipt.task is None
    assert ownership.reconcile_owned_task_admission(receipt).state == "nonacceptance"


@pytest.mark.asyncio
async def test_stream_capacity_is_checked_before_any_structural_allocation(monkeypatch):
    host = CapacityLoan(capacity=0)
    import kestrel_sovereign._async_ownership as canonical
    original_create = canonical.create_owned_task
    events = []
    original_reserve = host.reserve
    def reserve(**options):
        events.append("reserve")
        return original_reserve(**options)
    host.reserve = reserve
    def observe_structural_entry(operation, **options):
        if options.get("lender_capture") is None:
            events.append("structural-construction")
        return original_create(operation, **options)
    monkeypatch.setattr(canonical, "create_owned_task", observe_structural_entry)
    async def source():
        pytest.fail("capacity-refused producer dispatched")
        yield

    with ownership.bind_owned_task_lender(host):
        stream = execution_stream(None, source())
        try:
            with pytest.raises(RuntimeError, match="test host capacity exhausted"):
                await anext(stream)
        finally:
            await stream.aclose()
    assert events[0] == "reserve", events


@pytest.mark.asyncio
async def test_cooperative_stream_idle_waits_do_not_request_work_loans():
    host = CapacityLoan(capacity=1)
    async def source():
        assert ownership.capture_owned_task_lender() is not None
        assert len(host.outstanding) == 1
        yield "bounded cooperative stream"

    with ownership.bind_owned_task_lender(host):
        async with owned_execution_stream(None, source(), withdrawal=SourceWithdrawal()) as stream:
            await stream.reserve()
            assert await anext(stream) == "bounded cooperative stream"
    assert not host.outstanding and len(host.retired) == 1
