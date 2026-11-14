"""Scripted server/writer faults with real original asyncio Task ownership.

No Unix/TLS/browser/native Process exists in these tests. The loader follows
the existing canonical worker test's finite sibling-module loading convention.
"""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_task, create_owned_task,
)
from kestrel_sovereign.execution_custody import _explicit_control_chain
from tests.shared.installed_worker_requirement import worker_module

ROOT = Path(__file__).parents[2]/"docker/visual-browser"
NAMES = ("controller", "wire", "sandbox_checks", "worker", "two_domain_worker")
previous = {name: sys.modules.get(name) for name in NAMES}
loaded = {}
try:
    for name in NAMES:
        spec = importlib.util.spec_from_file_location(name, ROOT/(name+".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    worker = loaded["two_domain_worker"]
finally:
    for name, original in previous.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class ScriptedServer:
    def __init__(self):
        self.closed = False
        self.wait_ack = 0
    def close(self):
        self.closed = True
    async def wait_closed(self):
        self.wait_ack += 1


class ScriptedWriter:
    def __init__(self):
        self.close_calls = self.wait_ack = 0
    def close(self):
        self.close_calls += 1
        raise OSError("scripted original writer close failure")
    async def wait_closed(self):
        self.wait_ack += 1
    def get_extra_info(self, name):
        return None  # Deliberately no peer: exercise original refusal cleanup.


class ScriptedController:
    _session = {"state": "ready"}
    def __init__(self):
        self.retired_ack = False
    def _live(self):
        raise worker.Refused("expired")
    async def retire(self):
        self.retired_ack = True


@pytest.mark.asyncio
async def test_original_browser_server_open_is_joined_before_withdrawal_settles(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    servers = []
    controller = ScriptedController()
    async def open_server(*args, **kwargs):
        server = ScriptedServer()
        servers.append(server)
        entered.set()
        await release.wait()
        return server
    monkeypatch.setattr(worker.asyncio, "start_unix_server", open_server)
    monkeypatch.setattr(worker, "os", SimpleNamespace(chmod=lambda *_: None))
    admission = OwnedTaskAdmission()
    task = create_owned_task(worker.serve(controller, "a"*64), admission=admission, name="scripted-browser-serve")
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        outcome = await await_owned_task(task)
        assert outcome.error is not None
        assert len(servers) == 1
        assert servers[0].closed and servers[0].wait_ack == 1
        assert controller.retired_ack
    finally:
        release.set()
        await await_owned_task(task)
        # Test-owned scripted objects only; no physical cleanup claim.
        for server in servers:
            if not server.closed:
                server.close()
                await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("arrival", ["active", "closing"])
async def test_original_browser_writer_close_error_still_joins_actual_writer(monkeypatch, arrival):
    controller, writer, servers = ScriptedController(), ScriptedWriter(), []
    async def open_server(callback, *args, **kwargs):
        server = ScriptedServer()
        servers.append(server)
        if len(servers) == 1:
            if arrival == "active":
                callback(None, writer)
            else:
                original_close = server.close
                def late_accept():
                    original_close()
                    callback(None, writer)
                server.close = late_accept
        return server
    monkeypatch.setattr(worker.asyncio, "start_unix_server", open_server)
    monkeypatch.setattr(worker, "os", SimpleNamespace(chmod=lambda *_: None))
    admission = OwnedTaskAdmission()
    task = create_owned_task(worker.serve(controller, "a"*64), admission=admission, name="scripted-browser-serve")
    outcome = await await_owned_task(task)
    assert outcome.error is not None
    assert all(server.closed and server.wait_ack == 1 for server in servers)
    assert writer.close_calls >= 1
    assert writer.wait_ack == 1
    assert controller.retired_ack


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["startup", "withdrawal", "stop"])
async def test_original_playwright_startup_failure_joins_original_pump_and_init(monkeypatch, failure):
    """Scripted canonical classes, real original Tasks; no Node/browser claim."""
    manager_module = worker_module("playwright.async_api._context_manager")
    import playwright._impl._connection as connection_module
    import playwright._impl._transport as transport_module

    entered, allow_failure, stopping, release = (asyncio.Event() for _ in range(4))
    originals, joined, nodes = [], [], []
    startup_error, stop_error = OSError("scripted original startup"), OSError("scripted original stop")
    class Node:
        def __init__(self):
            self.wait_ack = 0
            self.stdin = SimpleNamespace(close=lambda: None)
            nodes.append(self)
        async def wait(self):
            self.wait_ack += 1
            return 0
    class PipeTransport:
        def __init__(self, loop):
            self.on_error_future = loop.create_future()
            self._proc = Node()
        async def connect(self):
            return None
    class Connection:
        def __init__(self, _dispatcher, _factory, transport, loop):
            self._transport, self._loop = transport, loop
            self.playwright_future = loop.create_future()
            self._init_task = None
        async def run(self):
            originals.append(asyncio.current_task())
            await self._transport.connect()
            async def init():
                await release.wait()
                joined.append("init")
            self._init_task = asyncio.create_task(init())
            originals.append(self._init_task)
            entered.set()
            await allow_failure.wait()
            self.playwright_future.set_exception(startup_error)
            await stopping.wait()
            await release.wait()
            joined.append("pump")
        async def stop_async(self):
            stopping.set()
            if failure == "stop":
                raise stop_error
    # Replace only the explicitly scripted Playwright I/O classes. The
    # product's ownership helper and all returned asyncio Tasks are original.
    monkeypatch.setattr(manager_module, "Connection", Connection)
    monkeypatch.setattr(manager_module, "PipeTransport", PipeTransport)
    monkeypatch.setattr(connection_module, "Connection", Connection)
    monkeypatch.setattr(transport_module, "PipeTransport", PipeTransport)
    monkeypatch.setattr(worker.asyncio.subprocess, "Process", Node)
    async def consume():
        async with worker.original_playwright():
            raise AssertionError("a failed startup cannot publish Playwright")
    admission = OwnedTaskAdmission()
    owner = create_owned_task(consume(), admission=admission, name="scripted-original-playwright-owner")
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if failure == "withdrawal":
            owner.cancel()
        allow_failure.set()
        await asyncio.wait_for(stopping.wait(), 5)
        settled_early = False
        try:
            await asyncio.wait_for(asyncio.shield(owner), .05)
        except TimeoutError:
            pass  # The original cleanup must still own the held real Tasks.
        except BaseException:
            settled_early = True
        else:
            settled_early = True
        release.set()
        outcome = await await_owned_task(owner)
        controls = tuple(_explicit_control_chain(outcome.error)) if outcome.error is not None else ()
        assert not settled_early
        assert sorted(joined) == ["init", "pump"]
        assert all(task.done() for task in originals)
        assert len(nodes) == 1 and nodes[0].wait_ack == 1
        assert startup_error in controls
        if failure == "stop":
            assert stop_error in controls
    finally:
        allow_failure.set()
        stopping.set()
        release.set()
        await await_owned_task(owner)
        await asyncio.gather(*originals, return_exceptions=True)


# Review21 P2-2: genuine stdlib close Future, canonical Endpoint/serve and owned
# asyncio Tasks; server/peer/controller are explicit non-native I/O doubles.
class _Review21Transport(asyncio.Transport):
    def __init__(self, protocol):
        self.protocol, self.closing = protocol, False
        self.writes, self.closes = [], 0
        self.peer = object()
    def get_extra_info(self, name, default=None):
        return self.peer if name == "socket" else default
    def is_closing(self):
        return self.closing
    def write(self, value):
        if not self.closing:
            self.writes.append(bytes(value))
    def close_original(self, error):
        if not self.closing:
            self.closing = True
            self.protocol.connection_lost(error)
    def close(self):
        self.closes += 1
        self.close_original(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_close", ["clean", "reset"])
async def test_review21_original_endpoint_peer_close_does_not_poison_browser_retirement(monkeypatch, tmp_path, peer_close):
    """No real socket/browser/Host/ComputerSession or native ACK is fabricated."""
    import json
    from types import SimpleNamespace as Namespace
    from kestrel_sovereign.execution_custody import _explicit_control_chain

    entered, finish = asyncio.Event(), asyncio.Event()
    servers, peers, socket_paths = [], [], []
    reset = ConnectionResetError("scripted peer reset after original dispatch")
    accepted = []
    class Controller:
        def __init__(self):
            self._session, self.expired, self.retired = {"state": "ready"}, False, False
        async def act(self, *, session, request):
            accepted.append((session, request))
            entered.set()
            await finish.wait()
            return {"outcome": "completed", "action_id": request["action_id"]}
        def _live(self):
            if self.expired:
                raise worker.Refused("expired")
        async def retire(self):
            self.retired = True
            self._session["state"] = "closed"
    class PathDouble:
        def __init__(self, value):
            self.value = str(value)
        def __truediv__(self, name):
            return PathDouble(self.value+"/"+name)
        def __str__(self):
            return self.value
        def unlink(self, *, missing_ok):
            socket_paths.append((self.value, missing_ok))
    controller = Controller()
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
    transport = _Review21Transport(protocol)
    protocol.connection_made(transport)
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    original_close_waiter = protocol._get_close_waiter(writer)
    message = {"schema_version": 1, "secret": "a"*64, "id": "review21-original-request",
        "operation": "act", "arguments": {"session": {"scripted_controller_only": True},
            "request": {"action_id": "review21-original-action"}}}
    reader.feed_data(json.dumps(message).encode()+b"\n")
    async def open_server(callback, *args, **kwargs):
        server = ScriptedServer()
        servers.append(server)
        if len(servers) == 1:
            callback(reader, writer)
        return server
    def scripted_peer(actual, uid, gid):
        assert actual is transport.peer and (uid, gid) == (10001, 10003)
        peers.append(actual)
    monkeypatch.setattr(worker.asyncio, "start_unix_server", open_server)
    monkeypatch.setattr(worker, "os", Namespace(chmod=lambda *_: None))
    monkeypatch.setattr(worker, "Path", PathDouble)
    monkeypatch.setattr(worker, "require_peer", scripted_peer)
    admission = OwnedTaskAdmission()
    task = create_owned_task(worker.serve(controller, "a"*64), admission=admission,
        name="review21-original-browser-serve")
    primary, outcome = None, None
    try:
        async with asyncio.timeout(5):
            await entered.wait()
        # The original request has crossed the real Endpoint HMAC/parser and
        # entered one scripted controller operation before the peer disappears.
        transport.close_original(reset if peer_close == "reset" else None)
        controller.expired = True
        finish.set()
    except BaseException as error:
        primary = error
    finally:
        finish.set()
        controller.expired = True
        transport.close()
        outcome = await await_owned_task(task)  # SAME actual original Task.
    # Observe only after original serve/listeners/callback/controller joins.
    fact = {"scope": "SOURCE_SCRIPTED_IO_REAL_STDLIB_FUTURE_AND_TASK_ONLY",
        "peer_close": peer_close, "accepted_original_calls": len(accepted),
        "same_original_request_id": message["id"], "servers": len(servers),
        "all_scripted_servers_closed_and_waited": all(s.closed and s.wait_ack == 1 for s in servers),
        "controller_retired": controller.retired, "same_original_task_retained": admission.task is task,
        "actual_owned_Task_join_returned": outcome is not None,
        "error_type": None if outcome is None or outcome.error is None else type(outcome.error).__name__,
        "original_reset_in_error_graph": outcome is not None and outcome.error is not None and any(x is reset for x in _explicit_control_chain(outcome.error)),
        "native_socket_browser_container_or_wait_proof": False}
    (tmp_path/"review21-browser-close-facts.json").write_text(json.dumps(fact, indent=2))
    if primary is not None:
        if outcome.error is not None:
            raise BaseExceptionGroup("original test setup and original serving errors", [primary, outcome.error])
        raise primary
    assert admission.task is task and outcome is not None
    assert len(accepted) == 1 and accepted[0][1] is not None and accepted[0][1]["action_id"] == "review21-original-action"
    assert len(peers) == 1 and controller.retired
    assert protocol._get_close_waiter(writer) is original_close_waiter
    assert original_close_waiter.exception() is (reset if peer_close == "reset" else None)
    assert len(servers) == 2 and all(s.closed and s.wait_ack == 1 for s in servers)
    assert len(socket_paths) == 2 and all(x[1] for x in socket_paths)
    try:
        assert outcome.error is None, "completed original peer-close handling must not poison browser retirement"
    except AssertionError as oracle:
        raise oracle from outcome.error  # Preserve exact real error graph on baseline.


def _review21_original_stream(*, writer_type=asyncio.StreamWriter,
                              protocol_type=asyncio.StreamReaderProtocol):
    """Real stdlib stream/Future, explicitly scripted transport; no native I/O."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = protocol_type(reader, loop=loop)
    transport = _Review21Transport(protocol)
    protocol.connection_made(transport)
    writer = writer_type(transport, protocol, reader, loop)
    return writer, protocol, transport


async def _review21_join_original_stream(writer):
    return await worker.await_owned_operation(
        worker._join_original_writer_close(writer), retirement_resource=writer,
        retain_originals=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_error", [None, ConnectionResetError, BrokenPipeError])
async def test_review21_terminal_original_close_future_is_joined(peer_error):
    writer, protocol, transport = _review21_original_stream()
    waiter = protocol._get_close_waiter(writer)
    error = None if peer_error is None else peer_error("scripted terminal peer close")
    transport.close_original(error)
    outcome = await _review21_join_original_stream(writer)
    assert outcome.error is None and outcome.cancellation is None
    assert protocol._get_close_waiter(writer) is waiter and not waiter.cancelled()
    assert waiter.exception() is error and writer.is_closing()


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["error-identity", "future-identity", "closing"])
async def test_review21_peer_close_recognition_requires_original_identity(monkeypatch, guard):
    writer, protocol, transport = _review21_original_stream()
    waiter = protocol._get_close_waiter(writer)
    reset = ConnectionResetError("original stdlib close Future")
    transport.close_original(reset)
    expected, replacement = reset, None
    original_wait = writer.wait_closed
    if guard == "error-identity":
        expected = ConnectionResetError("different scripted operation failure")
        async def different_error():
            try:
                await original_wait()
            except ConnectionResetError as actual:
                assert actual is reset
                raise expected from actual
        monkeypatch.setattr(writer, "wait_closed", different_error)
    elif guard == "future-identity":
        replacement = asyncio.get_running_loop().create_future()
        replacement.set_exception(reset)
        async def different_future():
            try:
                await original_wait()
            finally:
                protocol._closed = replacement
        monkeypatch.setattr(writer, "wait_closed", different_future)
    else:
        monkeypatch.setattr(writer, "is_closing", lambda: False)
    try:
        outcome = await _review21_join_original_stream(writer)
        assert outcome.error is not None
        assert any(error is expected for error in _explicit_control_chain(outcome.error))
    finally:
        protocol._closed = waiter
        if replacement is not None:
            assert replacement.exception() is reset


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign", ["writer", "protocol"])
async def test_review21_peer_close_recognition_refuses_foreign_stream_types(foreign):
    class ForeignWriter(asyncio.StreamWriter):
        pass
    class ForeignProtocol(asyncio.StreamReaderProtocol):
        pass
    writer, protocol, transport = _review21_original_stream(
        writer_type=ForeignWriter if foreign == "writer" else asyncio.StreamWriter,
        protocol_type=ForeignProtocol if foreign == "protocol" else asyncio.StreamReaderProtocol,
    )
    reset = ConnectionResetError("foreign stream close remains an error")
    transport.close_original(reset)
    outcome = await _review21_join_original_stream(writer)
    assert outcome.error is not None
    assert any(error is reset for error in _explicit_control_chain(outcome.error))
    assert protocol._get_close_waiter(writer).exception() is reset


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [OSError, TimeoutError, RuntimeError])
async def test_review21_other_original_writer_close_failures_remain_errors(failure_type):
    writer, protocol, transport = _review21_original_stream()
    error = failure_type("scripted original closure failure")
    transport.close_original(error)
    outcome = await _review21_join_original_stream(writer)
    assert outcome.error is not None
    assert any(original is error for original in _explicit_control_chain(outcome.error))
    assert protocol._get_close_waiter(writer).exception() is error


@pytest.mark.asyncio
async def test_review21_peer_close_verification_error_retains_both_originals(monkeypatch):
    writer, _, transport = _review21_original_stream()
    reset = ConnectionResetError("original terminal peer close")
    verification = RuntimeError("scripted original is_closing observation failure")
    transport.close_original(reset)
    def failed_verification():
        raise verification
    monkeypatch.setattr(writer, "is_closing", failed_verification)
    outcome = await _review21_join_original_stream(writer)
    originals = tuple(_explicit_control_chain(outcome.error))
    assert any(error is reset for error in originals)
    assert any(error is verification for error in originals)


@pytest.mark.asyncio
async def test_review21_cancelled_close_future_remains_unknown():
    writer, protocol, transport = _review21_original_stream()
    waiter = protocol._get_close_waiter(writer)
    waiter.cancel()
    transport.close_original(None)  # Test-owned scripted transport only.
    outcome = await _review21_join_original_stream(writer)
    assert waiter.cancelled() and outcome.error is not None
    assert any(isinstance(error, asyncio.CancelledError)
               for error in _explicit_control_chain(outcome.error))


@pytest.mark.asyncio
async def test_review21_pending_close_still_joins_after_caller_withdrawal(monkeypatch):
    writer, protocol, transport = _review21_original_stream()
    waiter = protocol._get_close_waiter(writer)
    entered = asyncio.Event()
    original_wait = writer.wait_closed
    async def observed_original_wait():
        entered.set()
        await original_wait()
    monkeypatch.setattr(writer, "wait_closed", observed_original_wait)
    admission = OwnedTaskAdmission()
    owner = create_owned_task(_review21_join_original_stream(writer), admission=admission,
                              retain_originals=True, name="review21-original-close-owner")
    try:
        async with asyncio.timeout(5):
            await entered.wait()
        owner.cancel()
        await asyncio.sleep(0)
        settled_before_original = owner.done()
        assert not waiter.done()
        transport.close_original(None)
        outer = await await_owned_task(owner)
        assert not settled_before_original and outer.error is None
        assert admission.task is owner and outer.result.error is None
        assert isinstance(outer.result.cancellation, asyncio.CancelledError)
        assert protocol._get_close_waiter(writer) is waiter and waiter.result() is None
    finally:
        transport.close_original(None)
        await await_owned_task(owner)
