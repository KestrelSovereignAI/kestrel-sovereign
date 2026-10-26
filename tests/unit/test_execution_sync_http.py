"""Actual direct native TLS, original restrictions and joined thread custody."""

import asyncio
from contextvars import copy_context
from http.server import BaseHTTPRequestHandler
import ssl
import threading
from types import SimpleNamespace

import httpcore
from httpcore._backends.sync import SyncStream
import httpx
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ProcessRuntimeExecutionFence,
    bind_execution_custody,
    bind_native_execution_admission,
)
from kestrel_sovereign.execution_sync_http import (
    NativeSyncHTTPOutcomeUnknown,
    await_native_sync_http,
    bind_native_sync_http_execution_admission,
)
from tests.utils.tls import make_tls_contexts, tls_http_server


@pytest.fixture
def peer(tmp_path):
    server_context, client_context, _ = make_tls_contexts(tmp_path)
    state = SimpleNamespace(
        current=True, checks=0, requests=[], entered=threading.Event(), release=threading.Event()
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                pass  # Expected native peer retirement in denial/cancel tests.

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state.requests.append((self.path, body, self.client_address[1]))
            if self.path == "/hold":
                state.entered.set()
                if not state.release.wait(5):
                    return
            self.send_response(200)
            self.send_header("Content-Length", "16")
            self.end_headers()
            if self.path == "/held-body":
                state.entered.set()
                if not state.release.wait(5):
                    return
            self.wfile.write(b"private-response")

        def log_message(self, *args):
            pass

    def current():
        state.checks += 1
        if not state.current:
            raise ExecutionAuthorityError("original native restriction revoked")

    state.check = current
    with tls_http_server(server_context, Handler) as server:
        state.context, state.url, state.port = (
            client_context,
            f"https://127.0.0.1:{server.server_port}",
            server.server_port,
        )
        try:
            yield state
        finally:
            state.release.set()


def client(peer, **options):
    return httpx.Client(verify=peer.context, trust_env=False, timeout=2, **options)


@pytest.mark.parametrize("retirement", ["host-grant", "global-turn", "original-checker"])
async def test_write_predicates_cannot_retire_authority_then_send_native_body(
    peer, monkeypatch, retirement,
):
    writes = []
    state = {"checks": 0, "retired": False, "arm_original": False}
    native_send = ssl.SSLSocket.send

    def observe(self, data, flags=0):
        size = native_send(self, data, flags)
        if self.getpeername()[1] == peer.port:
            writes.append((state["retired"], size))
        return size

    monkeypatch.setattr(ssl.SSLSocket, "send", observe)
    with bind_execution_custody(ProcessRuntimeExecutionFence("owned-write-callback-stop")) as scope:
        def original_check():
            peer.check()
            if state["arm_original"]:
                scope.revoke("original checker retired the turn")
                state["retired"] = True

        def before_write():
            state["checks"] += 1
            if state["checks"] == 2:
                if retirement == "original-checker":
                    state["arm_original"] = True
                else:
                    state["retired"] = True
                    if retirement == "host-grant":
                        peer.current = False
                    else:
                        scope.revoke("extra predicate retired the turn")

        with client(peer) as native, bind_native_sync_http_execution_admission(
            native, original_check, check_before_request_write=before_write,
        ):
            with pytest.raises(NativeSyncHTTPOutcomeUnknown):
                await await_native_sync_http(
                    native, lambda c: c.post(peer.url, content=b"original").text,
                )
    assert state["retired"] and any(not retired for retired, _ in writes)
    assert not [size for retired, size in writes if retired], writes


async def test_real_native_https_keeps_original_checker_and_existing_pool(peer):
    with client(peer) as native:
        pool = native._transport._pool
        with bind_native_sync_http_execution_admission(native, peer.check):
            result = await await_native_sync_http(
                native, lambda c: c.post(peer.url, content=b"original").text
            )
        assert result == "private-response" and peer.checks > 5
        assert native._transport._pool is pool
    assert [item[1] for item in peer.requests] == [b"original"]


async def test_revoke_after_actual_verified_tls_before_first_http_byte(peer, monkeypatch):
    original = SyncStream.start_tls

    def after_tls(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        peer.current = False
        return stream

    monkeypatch.setattr(SyncStream, "start_tls", after_tls)
    with client(peer) as native, bind_native_sync_http_execution_admission(native, peer.check):
        with pytest.raises(ExecutionAuthorityError, match="revoked"):
            await await_native_sync_http(
                native, lambda c: c.post(peer.url, content=b"private-create")
            )
    assert peer.requests == []


async def test_every_partial_native_socket_send_rechecks_original_restriction(peer, monkeypatch):
    original = ssl.SSLSocket.send
    writes = []

    def partial(self, data, flags=0):
        if self.getpeername()[1] == peer.port:
            count = original(self, data[:17], flags)
            writes.append(count)
            peer.current = False
            return count
        return original(self, data, flags)

    monkeypatch.setattr(ssl.SSLSocket, "send", partial)
    with client(peer) as native, bind_native_sync_http_execution_admission(native, peer.check):
        with pytest.raises(NativeSyncHTTPOutcomeUnknown) as caught:
            await await_native_sync_http(
                native, lambda c: c.post(peer.url, content=b"private-create")
            )
        assert isinstance(caught.value.original_error, ExecutionAuthorityError)
    assert writes == [17] and peer.requests == []


async def test_pooled_reused_connection_uses_each_original_owner_not_first_owner(peer):
    with client(peer) as native:
        first = SimpleNamespace(active=True)

        def owner_one():
            if not first.active:
                raise ExecutionAuthorityError("first owner retired")

        with bind_native_sync_http_execution_admission(native, owner_one):
            await await_native_sync_http(
                native, lambda c: c.post(peer.url + "/one", content=b"one").text
            )
        first.active = False
        with bind_native_sync_http_execution_admission(native, peer.check):
            await await_native_sync_http(
                native, lambda c: c.post(peer.url + "/two", content=b"two").text
            )
    assert len(peer.requests) == 2
    assert peer.requests[0][2] == peer.requests[1][2], "Actual TLS connection was not reused"


async def test_original_body_revocation_refuses_private_bytes(peer, monkeypatch):
    original = SyncStream.read

    def read_then_revoke(self, *args, **kwargs):
        data = original(self, *args, **kwargs)
        if b"private-response" in data:
            peer.current = False
        return data

    monkeypatch.setattr(SyncStream, "read", read_then_revoke)
    with client(peer) as native, bind_native_sync_http_execution_admission(native, peer.check):
        with pytest.raises(NativeSyncHTTPOutcomeUnknown):
            await await_native_sync_http(
                native, lambda c: c.post(peer.url, content=b"original").text
            )
    assert len(peer.requests) == 1


async def test_actual_close_revocation_refuses_private_delivery(peer):
    def response_observer(response):
        original = response.stream.close

        def close_then_revoke():
            original()
            peer.current = False

        response.stream.close = close_then_revoke

    with client(peer, event_hooks={"response": [response_observer]}) as native:
        with bind_native_sync_http_execution_admission(native, peer.check):
            with pytest.raises(NativeSyncHTTPOutcomeUnknown):
                await await_native_sync_http(
                    native, lambda c: c.post(peer.url, content=b"original").text
                )
    assert len(peer.requests) == 1


def test_direct_synchronous_delivery_checks_after_native_close(peer):
    def observer(response):
        original_close = response.stream.close

        def close():
            original_close()
            peer.current = False

        response.stream.close = close

    with client(peer, event_hooks={"response": [observer]}) as native:
        with bind_native_sync_http_execution_admission(native, peer.check):
            with pytest.raises(NativeSyncHTTPOutcomeUnknown) as caught:
                native.post(peer.url, content=b"original")
            assert caught.value.native_result.content == b"private-response"
    assert len(peer.requests) == 1


async def test_cancel_fences_socket_and_physically_joins_original_native_thread(peer):
    finished = threading.Event()

    def effect(native):
        try:
            return native.post(peer.url + "/hold", content=b"original").text
        finally:
            finished.set()

    with client(peer) as native, bind_native_sync_http_execution_admission(native, peer.check):
        task = asyncio.create_task(await_native_sync_http(native, effect))
        try:
            assert await asyncio.to_thread(peer.entered.wait, 2)
            task.cancel()
            with pytest.raises(BaseException):
                await task
            assert finished.is_set(), "Caller escaped while native thread was still running"
        finally:
            peer.release.set()
            await asyncio.gather(task, return_exceptions=True)


def test_copied_original_context_cannot_borrow_a_later_callers_restriction(peer):
    with client(peer) as native:
        with bind_native_sync_http_execution_admission(native, peer.check):
            captured = copy_context()
        with bind_native_sync_http_execution_admission(native, lambda: None):
            with pytest.raises(ExecutionAuthorityError, match="retired"):
                captured.run(native.post, peer.url, content=b"forbidden")
        with pytest.raises(ExecutionAuthorityError, match="original restriction"):
            native.post(peer.url, content=b"unbound")
    assert peer.requests == []


def test_all_original_native_async_validators_are_refused_even_in_copied_context(peer):
    async def original_validator():
        pytest.fail("An async validator must not run through a sync bridge")

    with client(peer) as native:
        with bind_native_execution_admission(original_validator):
            with pytest.raises(ExecutionAuthorityError, match="async native admission"):
                with bind_native_sync_http_execution_admission(native, peer.check):
                    pytest.fail("Async gate was bypassed")
            copied = copy_context()
        with pytest.raises(ExecutionAuthorityError, match="async native admission"):
            copied.run(bind_native_sync_http_execution_admission(native, peer.check).__enter__)
    assert peer.requests == []


async def test_existing_canonical_execution_custody_remains_required(peer, monkeypatch):
    original = SyncStream.start_tls
    fence = ProcessRuntimeExecutionFence("test-original")

    def after_tls(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        fence.retire()
        return stream

    monkeypatch.setattr(SyncStream, "start_tls", after_tls)
    with client(peer) as native, bind_execution_custody(fence):
        with bind_native_sync_http_execution_admission(native, lambda: None):
            with pytest.raises(ExecutionAuthorityError, match="retired"):
                await await_native_sync_http(
                    native, lambda c: c.post(peer.url, content=b"forbidden")
                )
    assert peer.requests == []


@pytest.mark.parametrize(
    "mode", ["mock", "h2", "retries", "infinite", "already-used", "async-checker"]
)
def test_unsupported_native_configurations_refuse_before_work(peer, mode):
    options = {}
    if mode == "mock":
        options["transport"] = httpx.MockTransport(lambda request: httpx.Response(200))
    if mode == "retries":
        options["transport"] = httpx.HTTPTransport(verify=peer.context, retries=1)
    with client(peer, **options) as native:
        if mode == "h2":
            native._transport._pool._http2 = True
        if mode == "infinite":
            native.timeout = None
        if mode == "already-used":
            native.post(peer.url, content=b"ordinary-before-admission")

        async def asynchronous():
            pass

        with pytest.raises((TypeError, ExecutionAuthorityError)):
            with bind_native_sync_http_execution_admission(
                native, asynchronous if mode == "async-checker" else peer.check
            ):
                pytest.fail("Unsupported native boundary admitted")


async def test_cancelling_finished_request_owner_cannot_interrupt_new_pool_borrower(peer):
    first_done, release_first = threading.Event(), threading.Event()

    def first(native):
        value = native.post(peer.url + "/first", content=b"first").text
        first_done.set()
        release_first.wait(3)
        return value

    with client(peer) as native, bind_native_sync_http_execution_admission(native, peer.check):
        one = asyncio.create_task(await_native_sync_http(native, first))
        two = None
        try:
            assert await asyncio.to_thread(first_done.wait, 2)
            two = asyncio.create_task(
                await_native_sync_http(
                    native, lambda c: c.post(peer.url + "/hold", content=b"second").text
                )
            )
            assert await asyncio.to_thread(peer.entered.wait, 2)
            assert peer.requests[0][2] == peer.requests[1][2]
            one.cancel()
            release_first.set()
            with pytest.raises(asyncio.CancelledError):
                await one
            await asyncio.sleep(0.02)
            assert not two.done(), "Old owner interrupted the next owner's pooled TLS connection"
            peer.release.set()
            assert await two == "private-response"
        finally:
            release_first.set()
            peer.release.set()
            await asyncio.gather(
                one, *(tuple([two]) if two is not None else ()), return_exceptions=True
            )


async def test_original_thread_closes_and_refuses_unconsumed_stream_before_return(peer):
    with client(peer) as native, bind_native_sync_http_execution_admission(native, peer.check):
        with pytest.raises(NativeSyncHTTPOutcomeUnknown):
            await await_native_sync_http(
                native,
                lambda c: c.send(
                    c.build_request("POST", peer.url, content=b"original"),
                    stream=True,
                ),
            )
        assert not native._transport._pool.connections


async def test_mutable_request_metadata_cannot_replace_original_checker(peer):
    def observer(request):
        request.extensions["kestrel.native_admission"] = (lambda: None,)
        request.extensions["check_current"] = lambda: None
        peer.current = False

    with client(peer, event_hooks={"request": [observer]}) as native:
        with bind_native_sync_http_execution_admission(native, peer.check):
            with pytest.raises(ExecutionAuthorityError, match="revoked"):
                await await_native_sync_http(
                    native, lambda c: c.post(peer.url, content=b"forbidden")
                )
    assert peer.requests == []


async def test_request_observer_cannot_replace_the_native_backend_before_connect(peer):
    native = client(peer)

    def observer(request):
        native._transport._pool._network_backend = httpcore.SyncBackend()

    native.event_hooks["request"].append(observer)
    with native, bind_native_sync_http_execution_admission(native, peer.check):
        with pytest.raises(TypeError, match="boundary was replaced"):
            await await_native_sync_http(native, lambda c: c.post(peer.url, content=b"forbidden"))
    assert peer.requests == []


async def test_caller_cancellation_during_actual_native_close_still_joins_that_close(peer):
    closing, release, finished = threading.Event(), threading.Event(), threading.Event()

    def response_observer(response):
        original = response.stream.close

        def hold_close():
            closing.set()
            release.wait(3)
            try:
                original()
            finally:
                finished.set()

        response.stream.close = hold_close

    with client(peer, event_hooks={"response": [response_observer]}) as native:
        with bind_native_sync_http_execution_admission(native, peer.check):
            task = asyncio.create_task(
                await_native_sync_http(native, lambda c: c.post(peer.url, content=b"original").text)
            )
            try:
                assert await asyncio.to_thread(closing.wait, 2)
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done() and not finished.is_set()
                release.set()
                with pytest.raises(BaseException):
                    await task
                assert finished.is_set()
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "mode", ["redirects", "auth", "cookies", "request-timeout", "replaced-backend"]
)
async def test_unverified_request_replay_and_changed_native_boundaries_refuse(peer, mode):
    with client(peer) as native:
        if mode == "redirects":
            native.follow_redirects = True
        if mode == "auth":
            native.auth = httpx.BasicAuth("private-user", "private-password")
        if mode == "cookies":
            native.cookies.set("private-owner-cookie", "private-value")
        with pytest.raises((TypeError, ExecutionAuthorityError)):
            with bind_native_sync_http_execution_admission(native, peer.check):
                if mode == "replaced-backend":
                    native._transport._pool._network_backend = httpcore.SyncBackend()
                await await_native_sync_http(
                    native,
                    lambda c: c.post(
                        peer.url,
                        content=b"forbidden",
                        **({"timeout": None} if mode == "request-timeout" else {}),
                    ),
                )
    assert peer.requests == []


async def test_final_async_delivery_check_retains_exact_private_native_result(peer, monkeypatch):
    from kestrel_sovereign import execution_sync_http

    original = execution_sync_http.await_owned_operation

    async def revoke_after_thread_join(*args, **kwargs):
        outcome = await original(*args, **kwargs)
        peer.current = False
        return outcome

    monkeypatch.setattr(execution_sync_http, "await_owned_operation", revoke_after_thread_join)
    with client(peer) as native, bind_native_sync_http_execution_admission(native, peer.check):
        with pytest.raises(NativeSyncHTTPOutcomeUnknown) as caught:
            await await_native_sync_http(
                native, lambda c: c.post(peer.url, content=b"original").text
            )
        assert caught.value.native_result == "private-response"
        assert isinstance(caught.value.original_error, ExecutionAuthorityError)
        assert "private-response" not in str(caught.value) + repr(caught.value)
    assert len(peer.requests) == 1


async def test_async_operations_and_unexpected_awaitables_are_explicitly_refused(peer):
    async def asynchronous(_client):
        pytest.fail("Async bridge must never run")

    with client(peer) as native, bind_native_sync_http_execution_admission(native, peer.check):
        with pytest.raises(TypeError, match="synchronous operation required"):
            await await_native_sync_http(native, asynchronous)
        with pytest.raises(TypeError, match="returned an awaitable"):
            await await_native_sync_http(native, lambda c: asynchronous(c))
    assert peer.requests == []


async def test_queued_old_socket_cancel_cannot_close_a_new_borrow_on_same_connection(
    peer, monkeypatch
):
    from kestrel_sovereign import execution_sync_http

    close_waiting, cancel_queued, first_closed, second_waiting = (
        threading.Event() for _ in range(4)
    )
    original_cancel = execution_sync_http._SocketBorrow.cancel

    def delay_cancel(self):
        cancel_queued.set()
        # The event loop is deliberately blocked at this real cancellation
        # handoff. Existing native threads finish/borrow the same TLS socket.
        assert second_waiting.wait(2)
        original_cancel(self)

    monkeypatch.setattr(execution_sync_http._SocketBorrow, "cancel", delay_cancel)

    def observer(response):
        if response.request.url.path == "/first":
            original_close = response.stream.close

            def close():
                close_waiting.set()
                assert cancel_queued.wait(2)
                try:
                    original_close()
                finally:
                    first_closed.set()

            response.stream.close = close

    def second(native):
        assert first_closed.wait(2)
        with native.stream("POST", peer.url + "/held-body", content=b"second") as response:
            second_waiting.set()
            assert peer.release.wait(3)
            return response.read()

    with client(peer, event_hooks={"response": [observer]}) as native:
        with bind_native_sync_http_execution_admission(native, peer.check):
            one = asyncio.create_task(
                await_native_sync_http(
                    native, lambda c: c.post(peer.url + "/first", content=b"first").text
                )
            )
            two = None
            try:
                assert await asyncio.to_thread(close_waiting.wait, 2)
                two = asyncio.create_task(await_native_sync_http(native, second))
                await asyncio.sleep(0.01)
                one.cancel()
                with pytest.raises(BaseException):
                    await one
                assert second_waiting.is_set()
                assert peer.requests[0][2] == peer.requests[1][2]
                peer.release.set()
                assert await two == b"private-response"
            finally:
                cancel_queued.set()
                peer.release.set()
                await asyncio.gather(
                    one, *(tuple([two]) if two is not None else ()), return_exceptions=True
                )


async def test_request_write_restriction_expires_after_full_post_without_losing_response(
    peer,
):
    from kestrel_sovereign.execution_sync_http import (
        NATIVE_SYNC_HTTP_ADMISSION_API_VERSION,
    )

    assert NATIVE_SYNC_HTTP_ADMISSION_API_VERSION == 2
    write = SimpleNamespace(current=True, checks=0)

    def before_write():
        write.checks += 1
        if not write.current:
            raise ExecutionAuthorityError("original spending plan expired")

    with (
        client(peer) as native,
        bind_native_sync_http_execution_admission(
            native, peer.check, check_before_request_write=before_write
        ),
    ):
        task = asyncio.create_task(
            await_native_sync_http(
                native, lambda c: c.post(peer.url + "/hold", content=b"original").text
            )
        )
        try:
            assert await asyncio.to_thread(peer.entered.wait, 2)
            writes_at_acceptance = write.checks
            write.current = False
            peer.release.set()
            assert await task == "private-response"
            assert write.checks == writes_at_acceptance and writes_at_acceptance > 0
        finally:
            peer.release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert peer.current and len(peer.requests) == 1


async def test_request_write_restriction_refuses_after_tls_before_first_http_byte(
    peer, monkeypatch
):
    original = SyncStream.start_tls
    write = SimpleNamespace(current=True)

    def after_tls(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        write.current = False
        return stream

    def before_write():
        if not write.current:
            raise ExecutionAuthorityError("original spending plan expired")

    monkeypatch.setattr(SyncStream, "start_tls", after_tls)
    with (
        client(peer) as native,
        bind_native_sync_http_execution_admission(
            native, peer.check, check_before_request_write=before_write
        ),
    ):
        with pytest.raises(ExecutionAuthorityError, match="spending plan expired"):
            await await_native_sync_http(
                native, lambda c: c.post(peer.url, content=b"original")
            )
    assert peer.current and peer.requests == []


async def test_request_write_restriction_rechecks_each_actual_partial_send(
    peer, monkeypatch
):
    original = ssl.SSLSocket.send
    write = SimpleNamespace(current=True)
    writes = []

    def partial(self, data, flags=0):
        if self.getpeername()[1] == peer.port:
            count = original(self, data[:17], flags)
            writes.append(count)
            write.current = False
            return count
        return original(self, data, flags)

    def before_write():
        if not write.current:
            raise ExecutionAuthorityError("original spending plan expired")

    monkeypatch.setattr(ssl.SSLSocket, "send", partial)
    with (
        client(peer) as native,
        bind_native_sync_http_execution_admission(
            native, peer.check, check_before_request_write=before_write
        ),
    ):
        with pytest.raises(NativeSyncHTTPOutcomeUnknown) as caught:
            await await_native_sync_http(
                native, lambda c: c.post(peer.url, content=b"original")
            )
        assert isinstance(caught.value.original_error, ExecutionAuthorityError)
    assert peer.current and writes == [17] and peer.requests == []


@pytest.mark.parametrize("retirement", ["host-grant", "global-turn"])
async def test_request_write_restriction_cannot_replace_original_read_authority(peer, retirement):
    def before_write():
        return None  # Restriction permits bytes; this supplies no authority.

    with (
        bind_execution_custody(ProcessRuntimeExecutionFence("owned-write-terminal-authority")) as scope,
        client(peer) as native,
        bind_native_sync_http_execution_admission(
            native, peer.check, check_before_request_write=before_write
        ),
    ):
        task = asyncio.create_task(
            await_native_sync_http(
                native,
                lambda c: c.post(peer.url + "/held-body", content=b"original").text,
            )
        )
        try:
            assert await asyncio.to_thread(peer.entered.wait, 2)
            if retirement == "global-turn":
                scope.revoke("original global turn stopped")
            else:
                peer.current = False
            peer.release.set()
            with pytest.raises(NativeSyncHTTPOutcomeUnknown):
                await task
        finally:
            peer.release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert len(peer.requests) == 1


@pytest.mark.parametrize("mode", ["noncallable", "async-function"])
def test_request_write_restriction_rejects_invalid_binding_before_installation(
    peer, mode
):
    async def invalid():
        return None

    check = invalid if mode == "async-function" else False
    with client(peer) as native:
        with pytest.raises(TypeError, match="synchronous request-write restriction"):
            with bind_native_sync_http_execution_admission(
                native, peer.check, check_before_request_write=check
            ):
                pytest.fail("invalid extra restriction admitted")
        assert getattr(native, "_kestrel_sync_admission", None) is None
    assert peer.requests == []


@pytest.mark.parametrize("mode", ["false", "coroutine-result"])
async def test_request_write_restriction_requires_none_and_closes_coroutine_result(
    peer, mode
):
    results = []

    async def invalid():
        return None

    def check():
        result = invalid() if mode == "coroutine-result" else False
        results.append(result)
        return result

    with (
        client(peer) as native,
        bind_native_sync_http_execution_admission(
            native, peer.check, check_before_request_write=check
        ),
    ):
        with pytest.raises(TypeError, match="return None or raise"):
            await await_native_sync_http(
                native, lambda c: c.post(peer.url, content=b"original")
            )
    assert peer.requests == [] and len(results) == 1
    if mode == "coroutine-result":
        assert results[0].cr_frame is None


async def test_request_write_restriction_copied_binding_stays_retired(peer):
    with client(peer) as native:
        with bind_native_sync_http_execution_admission(
            native, peer.check, check_before_request_write=lambda: None
        ):
            copied = copy_context()
        with bind_native_sync_http_execution_admission(native, peer.check):
            with pytest.raises(ExecutionAuthorityError, match="restriction retired"):
                copied.run(native.post, peer.url, content=b"original")
    assert peer.requests == []
