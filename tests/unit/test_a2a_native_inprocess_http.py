"""Physical incremental ASGI frames, deadlines and original cleanup custody."""
import asyncio

import httpx
import pytest

from kestrel_sovereign.a2a.native_inprocess_http import _NativeInProcessHTTPTransport
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    _explicit_control_chain,
)
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task


def transport_for(app, *, buffer_bytes=65536, check=None):
    def binding(scope):
        assert scope["app"] is app
        assert scope["path"] == "/native"
        if check is not None:
            check(scope)

    return _NativeInProcessHTTPTransport(
        application=app, public_base_url="https://native.test",
        require_binding=binding, buffer_bytes=buffer_bytes,
    )


async def disconnected(receive):
    while (await receive())["type"] != "http.disconnect":
        pass


async def test_incremental_body_is_observable_before_original_application_finishes():
    release, finished = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"initial", "more_body": True})
            await release.wait()
            await send({"type": "http.response.body", "body": b"terminal"})
        finally:
            finished.set()

    transport = transport_for(app)
    async with httpx.AsyncClient(transport=transport) as client:
        try:
            async with asyncio.timeout(1):
                async with client.stream("GET", "https://native.test/native") as response:
                    body = response.aiter_raw()
                    assert await anext(body) == b"initial"
                    assert not finished.is_set()
                    release.set()
                    assert [chunk async for chunk in body] == [b"terminal"]
        finally:
            release.set()
    assert finished.is_set()
    assert not transport.exchanges


@pytest.mark.parametrize("phase", ["headers", "body"])
async def test_read_deadline_retires_original_cooperative_application(phase):
    finished = asyncio.Event()

    async def app(scope, receive, send):
        try:
            if phase == "body":
                await send({"type": "http.response.start", "status": 200})
            await disconnected(receive)
        finally:
            finished.set()

    transport = transport_for(app)
    async with httpx.AsyncClient(transport=transport, timeout=0.05) as client:
        with pytest.raises(httpx.ReadTimeout, match="response deadline exceeded"):
            await client.get("https://native.test/native")
    assert finished.is_set()
    assert not transport.exchanges


async def test_write_deadline_retains_actual_request_stream_and_original_failure():
    from kestrel_sovereign.execution_custody import execution_cleanup_debts

    release, closed = asyncio.Event(), []

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await release.wait()
            yield b"late"

        async def aclose(self):
            closed.append(self)

    async def app(scope, receive, send):
        await receive()

    body = SlowBody()
    transport = transport_for(app)
    exchange = None
    try:
        with pytest.raises(BaseException) as failure:
            async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(0.05, write=0.005)) as client:
                # `stream=` retains HTTPX's original stream, not its opaque
                # AsyncIterable wrapper. A finite write is NOT permission to
                # cancel this producer or claim that it already closed.
                await client.send(httpx.Request(
                    "POST", "https://native.test/native", headers={"Host": "native.test"}, stream=body,
                ))
        assert any(isinstance(error, httpx.WriteTimeout) for error in _explicit_control_chain(failure.value))
        assert execution_cleanup_debts(failure.value)
        exchange, = transport.exchanges
        assert not closed and not exchange.request_source.producer_task.cancelling()
        assert not exchange.request_source.producer_task.done()
    finally:
        release.set()
        if exchange is not None:
            await exchange.source.join_source_owners(closing=True)
        with pytest.raises(BaseException) as retired:
            await transport.aclose()
        assert any(isinstance(error, httpx.WriteTimeout) for error in _explicit_control_chain(retired.value))
    assert closed == [body]
    assert exchange.source.producer_task.done() and exchange.request_source.producer_task.done()
    assert not transport.exchanges


async def test_streaming_upload_and_response_share_bounded_demand_not_one_physical_task():
    release, advanced, entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    originals, closed = [], []

    class OriginalBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            originals.append(asyncio.current_task())
            entered.set()
            yield b"first"
            await release.wait()
            yield b"second"
            advanced.set()

        async def aclose(self):
            closed.append(asyncio.current_task())

    async def app(scope, receive, send):
        assert (await receive())["body"] == b"first"
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"initial", "more_body": True})
        assert (await receive())["body"] == b"second"
        assert not advanced.is_set()  # The SAME input frame still awaits ACK.
        assert await receive() == {"type": "http.request", "body": b"", "more_body": False}
        await send({"type": "http.response.body", "body": b"terminal"})

    transport = transport_for(app)
    try:
        async with httpx.AsyncClient(transport=transport, timeout=1) as client:
            response = await client.send(httpx.Request(
                "POST", "https://native.test/native", headers={"Host": "native.test"}, stream=OriginalBody(),
            ), stream=True)
            try:
                body = response.aiter_raw()
                assert await anext(body) == b"initial"
                assert entered.is_set() and not advanced.is_set()
                exchange, = transport.exchanges
                assert exchange.request_source.producer_task is originals[0]
                assert originals[0] is not exchange.source.producer_task
                release.set()
                assert [chunk async for chunk in body] == [b"terminal"]
            finally:
                await response.aclose()
    finally:
        release.set()
    assert advanced.is_set() and len(originals) == 1
    assert closed == originals  # The original public stream closes on ITS task.
    assert not originals[0].cancelling() and originals[0].done()
    assert not transport.exchanges


async def test_write_failure_wakes_original_http_reader_even_if_application_catches_it():
    from kestrel_sovereign.execution_custody import execution_cleanup_debts, execution_evidence_carriers

    release, swallowed = asyncio.Event(), asyncio.Event()
    originals, closed = [], []

    class OriginalBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            originals.append(asyncio.current_task())
            await release.wait()
            yield b"late"

        async def aclose(self):
            closed.append(asyncio.current_task())

    async def app(scope, receive, send):
        try:
            await receive()
        except httpx.WriteTimeout:
            swallowed.set()
            await release.wait()

    transport = transport_for(app)

    async def consume():
        async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(0.5, write=0.02)) as client:
            await client.send(httpx.Request(
                "POST", "https://native.test/native", headers={"Host": "native.test"}, stream=OriginalBody(),
            ))

    reading, exchange = create_owned_task(consume()), None
    try:
        done, _ = await asyncio.wait((reading,), timeout=0.1)
        assert reading in done, "The HTTP reader waited for a recipient's error projection or cleanup"
        result = await await_owned_task(reading)
        exchange, = transport.exchanges
        write = await exchange.request_read.observe()
        assert isinstance(write.error, httpx.WriteTimeout)
        assert any(error is write.error for error in execution_evidence_carriers(result.error))
        assert execution_cleanup_debts(result.error)
        assert swallowed.is_set() and not closed
        assert not originals[0].done() and not originals[0].cancelling()
    finally:
        release.set()
        await await_owned_task(reading)
        if exchange is None:
            exchange = next(iter(transport.exchanges), None)
        if exchange is not None:
            await exchange.source.join_source_owners(closing=True)
            with pytest.raises(BaseException):
                await transport.aclose()
        else:
            await transport.aclose()
    assert closed == originals and originals[0].done()
    assert not transport.exchanges


@pytest.mark.parametrize("stop_at", [None, "native-inprocess-http", "native-inprocess-http-upload"])
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
async def test_original_upload_descriptor_closes_even_if_stop_precedes_source_entry(stop_at, eager):
    import os

    from tests.utils.native_admission_fault import before_native_source_entry
    from tests.utils.original_failure import OriginalUploadDescriptor

    body, exchanges = OriginalUploadDescriptor(), []
    reading, closing = None, None

    def withdraw():
        nonlocal closing
        if not exchanges:
            exchanges.extend(transport.exchanges)
        if stop_at == "native-inprocess-http":
            closing = create_owned_task(transport.aclose())
        else:
            reading.cancel("original caller withdrew before upload entry")

    async def app(scope, receive, send):
        if not exchanges:
            exchanges.extend(transport.exchanges)
        if stop_at is not None:
            while (await receive()).get("more_body", False):
                pass
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"done"})

    transport = transport_for(app)

    async def consume():
        async with httpx.AsyncClient(transport=transport, timeout=1) as client:
            return await client.send(httpx.Request(
                "POST", "https://native.test/native", headers={"Host": "native.test"}, stream=body,
            ))

    try:
        with before_native_source_entry(stop_at, withdraw, eager=eager) as originals:
            reading = create_owned_task(consume())
            result = await await_owned_task(reading)
            await transport.aclose()
        assert len(originals) == int(stop_at is not None)
        assert not body.entered
        assert all(source.producer_task.done() and source.owner_task.done() for source in originals)
        if stop_at == "native-inprocess-http-upload":
            assert isinstance(result.error, asyncio.CancelledError)
        elif stop_at == "native-inprocess-http":
            assert result.error is not None
            assert closing is not None and (await await_owned_task(closing)).error is None
        else:
            assert result.error is None and result.result.content == b"done"
        if not body.closed:
            os.fstat(body.descriptor)  # Verify the actual original resource still exists.
        assert body.closed, "The original upload descriptor was not retired before successful transport close"
        exchange, = exchanges
        assert body.closers == [exchange.close_task if stop_at == "native-inprocess-http"
                                else exchange.source.producer_task]
        assert not transport.exchanges
    finally:
        if reading is not None:
            await await_owned_task(reading)
        if closing is not None:
            await await_owned_task(closing)
        if not body.closed:
            await body.aclose()  # Failed test retains its exact original descriptor.


async def test_response_backpressure_is_one_bounded_canonical_frame():
    advanced, finished = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"123456789", "more_body": True})
            advanced.set()
            await send({"type": "http.response.body"})
        finally:
            finished.set()

    transport = transport_for(app, buffer_bytes=3)
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("GET", "https://native.test/native") as response:
            stream = response.aiter_raw()
            assert await anext(stream) == b"123"
            await asyncio.sleep(0)
            assert not advanced.is_set()
            assert await anext(stream) == b"456"
            assert not advanced.is_set()
            assert [chunk async for chunk in stream] == [b"789"]
    assert advanced.is_set() and finished.is_set()


async def test_close_after_headers_joins_same_original_source_without_payload_advancement():
    finished, advanced = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            if (await receive())["type"] == "http.disconnect":
                return
            advanced.set()
            await send({"type": "http.response.body", "body": b"not admitted"})
        finally:
            finished.set()

    transport = transport_for(app)
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("GET", "https://native.test/native"):
            assert not advanced.is_set()
        assert finished.is_set()
    assert not advanced.is_set()
    assert not transport.exchanges


async def test_original_control_identity_survives_actual_asgi_source_and_client_close():
    original = ExecutionCommitOutcomeError("unknown")

    async def app(scope, receive, send):
        raise original

    transport = transport_for(app)
    with pytest.raises(BaseException) as failure:
        async with httpx.AsyncClient(transport=transport) as client:
            await client.get("https://native.test/native")
    assert any(error is original for error in _explicit_control_chain(failure.value))
    assert all(exchange.source.producer_task.done() for exchange in transport.exchanges)
    assert not transport.exchanges


async def test_caller_cancellation_disconnects_and_joins_original_application():
    entered, finished = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            entered.set()
            await disconnected(receive)
        finally:
            finished.set()

    transport = transport_for(app)
    async with httpx.AsyncClient(transport=transport, timeout=0.1) as client:
        async def consume():
            async with client.stream("GET", "https://native.test/native") as response:
                await anext(response.aiter_raw())

        consuming = create_owned_task(consume())
        await entered.wait()
        consuming.cancel("original caller withdrew")
        outcome = await await_owned_task(consuming)
        assert any(isinstance(error, asyncio.CancelledError) and error.args == (
            "original caller withdrew",
        ) for error in _explicit_control_chain(outcome.error))
        assert finished.is_set()
        assert not transport.exchanges


async def test_overlapping_close_observers_join_one_original_failure_and_source():
    entered, release = asyncio.Event(), asyncio.Event()
    original = ExecutionCommitOutcomeError("unknown")
    finalizers = []

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
        finally:
            finalizers.append(asyncio.current_task())
            entered.set()
            await release.wait()
            raise original

    transport = transport_for(app)
    response = await transport.handle_async_request(httpx.Request(
        "GET", "https://native.test/native", extensions={"timeout": {"read": 1, "write": 1}},
    ))
    exchange, = transport.exchanges
    first = create_owned_task(response.aclose())
    await entered.wait()
    second = create_owned_task(transport.aclose())
    await asyncio.sleep(0)
    closer = exchange.close_task
    try:
        release.set()
        async with asyncio.timeout(1):
            outcomes = [await await_owned_task(first), await await_owned_task(second)]
        assert all(any(error is original for error in _explicit_control_chain(outcome.error))
                   for outcome in outcomes)
        assert finalizers == [exchange.source.producer_task]
        assert exchange.close_task is closer and closer.done()
        assert exchange.source.owner_task.done() and not transport.exchanges
    finally:
        release.set()
        await await_owned_task(first)
        await await_owned_task(second)


@pytest.mark.parametrize("phase", ["read", "early-close"])
async def test_error_middleware_does_not_turn_original_control_into_success(phase):
    from starlette.applications import Starlette
    from starlette.routing import Route

    original = ExecutionCommitOutcomeError("unknown")

    async def fail(request):
        raise original

    app = Starlette(routes=[Route("/native", fail)])
    transport = transport_for(app)
    with pytest.raises(BaseException) as failure:
        async with httpx.AsyncClient(transport=transport) as client:
            async with client.stream("GET", "https://native.test/native") as response:
                assert response.status_code == 500
                if phase == "read":
                    await response.aread()
    assert any(error is original for error in _explicit_control_chain(failure.value))
    assert not transport.exchanges


@pytest.mark.parametrize("after_body", [False, True])
async def test_base_middleware_early_close_is_not_a_private_source_error(after_body):
    from starlette.middleware.base import BaseHTTPMiddleware

    finished = asyncio.Event()

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"initial", "more_body": True})
            await disconnected(receive)
        finally:
            finished.set()

    async def dispatch(request, call_next):
        return await call_next(request)

    transport = transport_for(BaseHTTPMiddleware(app, dispatch=dispatch))
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("GET", "https://native.test/native") as response:
            if after_body:
                assert await anext(response.aiter_raw()) == b"initial"
    assert finished.is_set() and not transport.exchanges


@pytest.mark.parametrize("read_body", [False, True])
async def test_base_middleware_keeps_actual_application_failure_after_headers(read_body):
    from starlette.middleware.base import BaseHTTPMiddleware

    original = ExecutionCommitOutcomeError("unknown")
    raised, finished = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            raised.set()
            raise original
        finally:
            finished.set()

    async def dispatch(request, call_next):
        return await call_next(request)

    transport = transport_for(BaseHTTPMiddleware(app, dispatch=dispatch))
    with pytest.raises(BaseException) as failed:
        async with httpx.AsyncClient(transport=transport) as client:
            async with client.stream("GET", "https://native.test/native") as response:
                await raised.wait()
                if read_body:
                    await response.aread()
    assert any(error is original for error in _explicit_control_chain(failed.value))
    assert finished.is_set() and not transport.exchanges


async def test_ignoring_disconnect_cannot_starve_observer_or_fake_physical_retirement():
    release, finished = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            while not release.is_set():
                await send({"type": "http.response.body", "body": b"not admitted", "more_body": True})
        finally:
            finished.set()

    transport = transport_for(app)
    request = httpx.Request("GET", "https://native.test/native", extensions={
        "timeout": {"read": 0.02, "write": 0.02},
    })
    response = await transport.handle_async_request(request)
    try:
        async with asyncio.timeout(1):
            with pytest.raises(BaseException) as failed:
                await response.aclose()
        assert any(getattr(error, "_kestrel_native_retirement_debts", ())
                   for error in _explicit_control_chain(failed.value))
        exchange, = transport.exchanges
        assert not exchange.source.producer_task.done()
        assert not exchange.source.producer_task.cancelling() and not finished.is_set()
        with pytest.raises(httpx.StreamClosed):
            await anext(response.aiter_raw())
    finally:
        release.set()
        if transport.exchanges:
            exchange, = transport.exchanges
            async with asyncio.timeout(1):
                await await_owned_task(exchange.close_task)
        await transport.aclose()
    assert finished.is_set() and not transport.exchanges


@pytest.mark.parametrize("reason", ["read-timeout", "cancelled", "revoked"])
async def test_body_failure_retirement_timeout_retains_both_explicit_originals(reason):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError, execution_cleanup_debts

    reading, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    current = True
    denied = ExecutionAuthorityError("Original route was revoked")

    def check(scope):
        if not current:
            raise denied

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"initial", "more_body": True})
            await disconnected(receive)
        finally:
            await release.wait()
            finished.set()

    transport = transport_for(app, check=check)

    async def consume():
        nonlocal current
        async with httpx.AsyncClient(transport=transport, timeout=0.02) as client:
            async with client.stream("GET", "https://native.test/native") as response:
                body = response.aiter_raw()
                assert await anext(body) == b"initial"
                if reason == "revoked":
                    current = False
                reading.set()
                await anext(body)

    consuming = create_owned_task(consume())
    try:
        await reading.wait()
        if reason == "cancelled":
            consuming.cancel("original body reader withdrew")
        async with asyncio.timeout(1):
            outcome = await await_owned_task(consuming)
        assert outcome.error is not None
        errors = tuple(_explicit_control_chain(outcome.error))
        assert execution_cleanup_debts(outcome.error)
        if reason == "read-timeout":
            assert any(isinstance(error, httpx.ReadTimeout) for error in errors)
        elif reason == "cancelled":
            assert any(isinstance(error, asyncio.CancelledError) and error.args == (
                "original body reader withdrew",
            ) for error in errors)
        else:
            assert any(error is denied for error in errors)
        exchange, = transport.exchanges
        assert not exchange.source.producer_task.cancelling()
        assert not finished.is_set()
    finally:
        current = True
        release.set()
        await await_owned_task(consuming)
        if transport.exchanges:
            exchange, = transport.exchanges
            async with asyncio.timeout(1):
                await await_owned_task(exchange.close_task)
        await transport.aclose()
    assert finished.is_set() and not transport.exchanges


async def test_concurrent_close_cannot_strand_original_read_past_its_deadline():
    from kestrel_sovereign.execution_custody import execution_cleanup_debts
    from tests.utils.original_failure import fail_original_file_read
    from tests.utils.virtual_clock import await_within_hang_guard, pinned_loop_clock

    initial, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    actual = []

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"initial", "more_body": True})
            await disconnected(receive)
        finally:
            entered.set()
            await release.wait()
            fail_original_file_read(actual)

    transport = transport_for(app)

    async def consume():
        async with httpx.AsyncClient(transport=transport, timeout=0.02) as client:
            async with client.stream("GET", "https://native.test/native") as response:
                async for _ in response.aiter_raw():
                    initial.set()

    reading, closing, exchange = None, None, None
    try:
        with pinned_loop_clock() as clock:
            reading = create_owned_task(consume())
            reading.add_done_callback(lambda _: initial.set())
            await await_within_hang_guard(initial.wait(), step="original unsigned initial event")
            assert not reading.done(), "Original unsigned read ended before its initial event"
            await asyncio.sleep(0)
            exchange, = transport.exchanges
            closing = create_owned_task(transport.aclose())
            await await_within_hang_guard(entered.wait(), step="original unsigned finalizer entry")
            assert not reading.done() and not closing.done()
            clock.advance(0.02)
            await await_within_hang_guard(
                asyncio.wait((reading, closing)), step="original unsigned read and close deadlines",
            )
            closed = await await_owned_task(closing)
            assert execution_cleanup_debts(closed.error)
            observed = await await_owned_task(reading)
            assert any(isinstance(error, httpx.ReadTimeout) for error in _explicit_control_chain(observed.error))
            assert execution_cleanup_debts(observed.error)
            assert not exchange.source.producer_task.done() and not exchange.source.producer_task.cancelling()
    finally:
        release.set()
        result = (await await_within_hang_guard(await_owned_task(reading), step="original unsigned reader retirement")
                  if reading is not None else None)
        if closing is not None:
            await await_owned_task(closing)
        if exchange is None and transport.exchanges:
            exchange, = transport.exchanges
        if exchange is not None:
            async with asyncio.timeout(1):
                await exchange.source.join_source_owners(closing=True)
            retirement = None
            try:
                await transport.aclose()
            except BaseException as error:
                retirement = error
            assert len(actual) == 1
            assert any(error is actual[0] for failure in (result.error if result is not None else None, retirement)
                       for error in _explicit_control_chain(failure))
            assert exchange.source.producer_task.done() and exchange.source.owner_task.done()


@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
async def test_failed_close_admission_keeps_later_original_physical_source_error(mode, eager):
    from tests.utils.native_admission_fault import NativeAdmissionFault
    from tests.utils.original_failure import fail_original_file_read

    started, release = asyncio.Event(), asyncio.Event()
    actual = []

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            await disconnected(receive)
        finally:
            started.set()
            await release.wait()
            fail_original_file_read(actual)

    transport = transport_for(app)
    response = await transport.handle_async_request(httpx.Request(
        "GET", "https://native.test/native", extensions={"timeout": {"read": 0.02, "write": 0.02}},
    ))
    exchange, = transport.exchanges
    admission_error = OSError("original source-close admission failed")
    fault = NativeAdmissionFault(
        lambda operation: getattr(getattr(operation, "cr_code", None), "co_qualname", None)
        == "_Exchange.retire", admission_error, mode, eager=eager,
    )
    try:
        with fault, pytest.raises(BaseException) as first:
            await response.aclose()
        assert any(error is admission_error for error in _explicit_control_chain(first.value))
        await started.wait()
        closer, deadline = exchange.close_task, exchange.retirement_deadline
        release.set()
        async with asyncio.timeout(1):
            source = await exchange.source.join_source_owners(closing=True)
        assert len(actual) == 1
        assert any(error is actual[0] for error in _explicit_control_chain(source.error))
        with pytest.raises(BaseException) as later:
            await transport.aclose()
        errors = tuple(_explicit_control_chain(later.value))
        assert any(error is admission_error for error in errors)
        assert any(error is actual[0] for error in errors)
        assert exchange.close_task is closer and exchange.retirement_deadline == deadline
        assert exchange.source.producer_task.done() and exchange.source.owner_task.done()
        assert not exchange.source.producer_task.cancelling()
    finally:
        release.set()
        async with asyncio.timeout(1):
            await exchange.source.join_source_owners(closing=True)
            if exchange.close_task is not None:
                await await_owned_task(exchange.close_task)


async def test_slow_original_finally_is_bounded_private_debt_not_cancelled_or_replaced():
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = []

    async def app(scope, receive, send):
        original.append(asyncio.current_task())
        try:
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"initial", "more_body": True})
            await disconnected(receive)
        finally:
            started.set()
            await release.wait()
            finished.set()

    transport = transport_for(app)
    request = httpx.Request("GET", "https://native.test/native", extensions={
        "timeout": {"read": 0.02, "write": 0.02},
    })
    response = await transport.handle_async_request(request)
    try:
        assert await anext(response.aiter_raw()) == b"initial"
        async with asyncio.timeout(0.5):
            with pytest.raises(BaseException) as first:
                await response.aclose()
        assert started.is_set() and not finished.is_set()
        exchange, = transport.exchanges
        closer, deadline = exchange.close_task, exchange.retirement_deadline
        assert original == [exchange.source.producer_task]
        assert not original[0].cancelling()
        debts = [debt for error in _explicit_control_chain(first.value)
                 for debt in getattr(error, "_kestrel_native_retirement_debts", ())]
        assert (exchange, closer) in debts and (exchange, original[0]) in debts
        async with asyncio.timeout(0.5):
            with pytest.raises(BaseException):
                await transport.aclose()
        assert exchange.close_task is closer and exchange.retirement_deadline == deadline
        assert not original[0].cancelling() and not finished.is_set()
    finally:
        release.set()
        async with asyncio.timeout(1):
            if transport.exchanges:
                exchange, = transport.exchanges
                await await_owned_task(exchange.close_task)
            await transport.aclose()
    assert finished.is_set() and not transport.exchanges


@pytest.mark.parametrize("timeout", [None, True, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("phase", ["read", "write"])
async def test_invalid_deadlines_refuse_before_any_source_or_application(timeout, phase):
    called = []

    async def app(scope, receive, send):
        called.append(scope)

    values = {"read": 1, "write": 1, phase: timeout}
    transport = transport_for(app)
    with pytest.raises(ValueError, match=f"finite nonnegative {phase} timeout"):
        await transport.handle_async_request(httpx.Request(
            "GET", "https://native.test/native", extensions={"timeout": values},
        ))
    assert not called and not transport.exchanges


@pytest.mark.parametrize("messages", [
    [],
    [{"type": "http.response.body", "body": b"invalid"}],
    [{"type": "http.response.start", "status": True}],
    [{"type": "unsupported"}],
])
async def test_malformed_or_incomplete_application_never_returns_success(messages):
    finished = asyncio.Event()

    async def app(scope, receive, send):
        try:
            for message in messages:
                await send(message)
        finally:
            finished.set()

    transport = transport_for(app)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises((RuntimeError, ValueError)):
            await client.get("https://native.test/native")
    assert finished.is_set() and not transport.exchanges


@pytest.mark.parametrize("url,headers", [
    ("https://foreign.test/native", {}),
    ("http://native.test/native", {}),
    ("https://native.test:8443/native", {}),
    ("https://native.test/native", {"Host": "foreign.test"}),
])
async def test_different_origin_or_host_is_refused_before_application_entry(url, headers):
    called = []

    async def app(scope, receive, send):
        called.append(scope)

    transport = transport_for(app)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError):
            await client.get(url, headers=headers)
    assert called == [] and not transport.exchanges
