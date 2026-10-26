"""Real private IPC against an in-memory page; Docker proof runs separately."""
import asyncio
import importlib
import sys
import tempfile
from pathlib import Path

import pytest

from tests.unit.test_visual_browser_controller import action
from tests.unit.test_visual_browser_controller import controller as fixture_controller

ROOT = Path(__file__).parents[2] / "docker/visual-browser"
_names = ("controller", "wire", "sandbox_checks", "worker", "rpc_client")
_previous = {name: sys.modules.get(name) for name in _names}
_loaded = {}
try:
    for name in _names:
        spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _loaded[name] = module
    worker, wire, rpc, sandbox_checks = (_loaded[name] for name in ("worker", "wire", "rpc_client", "sandbox_checks"))
finally:
    for name, previous in _previous.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

SECRET = "a" * 64


@pytest.fixture
def ipc_dir():
    # pytest's nested temporary path can exceed macOS's Unix socket limit.
    with tempfile.TemporaryDirectory(prefix="kb-") as directory:
        yield Path(directory)


def controller(**options):
    _, page, session = fixture_controller()
    request = {key: session[key] for key in ("schema_version", "session_id", "scope", "lease_id", "surface", "expires_at")}
    request["required_capabilities"] = ["screenshot"]
    driver = worker.BrowserController(page, request, **options)
    return driver, page, driver.status(session["scope"], session["session_id"])


def message(operation, arguments, **overrides):
    return {"schema_version": 1, "secret": SECRET, "id": "rpc", "operation": operation,
            "arguments": arguments, **overrides}


async def exchange(path, request):
    reader, writer = await asyncio.open_unix_connection(str(path), limit=wire.MAX_RESPONSE)
    writer.write(request if isinstance(request, bytes) else wire.encode(request))
    await writer.drain()
    result = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return wire.decode(result) if result else None


@pytest.mark.asyncio
async def test_socket_auth_scope_limits_and_no_private_error_echo(ipc_dir):
    driver, page, session = controller()
    endpoint = worker.Endpoint(driver, SECRET)
    path = ipc_dir / "worker.sock"
    async with await asyncio.start_unix_server(endpoint.handle, str(path), limit=wire.MAX_REQUEST):
        args = {"scope": session["scope"], "session_id": session["session_id"]}
        assert (await exchange(path, message("status", args)))["result"] == session
        for secret in ("b" * 64, None, [], "a" * 65):
            result = await exchange(path, message("status", args, secret=secret))
            assert result["error"] == "unauthenticated"
            assert "result" not in result
        wrong = {**args, "scope": {**session["scope"], "principal_id": "another"}}
        assert (await exchange(path, message("status", wrong)))["error"] == "session_unavailable"
        assert (await exchange(path, message("shell", {"command": "PRIVATE"})))["error"] == "unsupported_request"
        for raw in (b'{"id":"one","id":"two"}\n', b'{"number":NaN}\n', b"[]\n", b"x" * (wire.MAX_REQUEST + 1) + b"\n"):
            result = await exchange(path, raw)
            assert result["error"] in {"invalid_request", "unauthenticated"}
            assert "PRIVATE" not in str(result)
    assert page.effects == []
    assert endpoint.connections == 0


@pytest.mark.asyncio
async def test_lost_transport_response_reconciles_without_repeating_effect(ipc_dir):
    driver, page, session = controller()
    endpoint = worker.Endpoint(driver, SECRET)
    path = ipc_dir / "worker.sock"
    request = await action(driver, session)
    page.gate = asyncio.Event()
    async with await asyncio.start_unix_server(endpoint.handle, str(path), limit=wire.MAX_REQUEST):
        _reader, writer = await asyncio.open_unix_connection(str(path))
        writer.write(wire.encode(message("act", {"session": session, "request": request})))
        await writer.drain()
        await page.entered.wait()
        writer.close()
        await writer.wait_closed()
        unknown = await exchange(path, message("action_status", {"session": session, "action_id": "input"}))
        assert unknown["result"]["outcome"] == "unknown"
        page.gate.set()
        result = await exchange(path, message("act", {"session": session, "request": request}))
        assert result["result"]["outcome"] == "completed"
        assert len(page.effects) == 1


@pytest.mark.asyncio
async def test_ipc_observation_and_control_share_the_execution_fence(ipc_dir):
    driver, _page, session = controller()
    endpoint = worker.Endpoint(driver, SECRET)
    path = ipc_dir / "worker.sock"
    async with await asyncio.start_unix_server(endpoint.handle, str(path), limit=wire.MAX_REQUEST):
        observed = await exchange(path, message("observe", {"session": session}))
        assert observed["result"]["frame"]["session_id"] == session["session_id"]
        request = {"schema_version": 1, "request_id": "pause", "session_id": session["session_id"],
                   "expected_control_epoch": 0, "operation": "pause"}
        paused = (await exchange(path, message("control", {"session": session, "request": request})))["result"]
        assert paused["state"] == "paused" and paused["control_epoch"] == 1
        assert (await exchange(path, message("observe", {"session": session})))["error"] == "stale_control"
        assert (await exchange(path, message("observe", {"session": paused})))["error"] == "input_fenced"


@pytest.mark.asyncio
async def test_excess_connections_do_not_queue_private_requests(ipc_dir):
    driver, _page, _session = controller()
    endpoint = worker.Endpoint(driver, SECRET)
    path = ipc_dir / "worker.sock"
    writers = []
    async with await asyncio.start_unix_server(endpoint.handle, str(path), limit=wire.MAX_REQUEST):
        try:
            for _ in range(4):
                _reader, writer = await asyncio.open_unix_connection(str(path))
                writers.append(writer)
            async with asyncio.timeout(2):
                while endpoint.connections != 4:
                    await asyncio.sleep(0)
                reader, writer = await asyncio.open_unix_connection(str(path))
                writers.append(writer)
                assert await reader.read() == b""
        finally:
            for writer in writers:
                writer.close()
                await writer.wait_closed()
            async with asyncio.timeout(2):
                while endpoint.connections:
                    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_expiry_retires_page_and_removes_socket(ipc_dir):
    elapsed = [0]
    driver, page, session = controller(monotonic=lambda: elapsed[0])
    task = asyncio.create_task(worker.serve(driver, SECRET, directory=ipc_dir))
    path = ipc_dir / "worker.sock"
    async with asyncio.timeout(2):
        while not path.exists():
            await asyncio.sleep(0.01)
        assert path.stat().st_mode & 0o777 == 0o600
        elapsed[0] = 301
        await task
    assert page.closed
    assert not path.exists()
    assert driver.status(session["scope"], session["session_id"])["state"] == "closed"


@pytest.mark.asyncio
async def test_closing_state_ends_worker_service_without_waiting_for_expiry(ipc_dir):
    driver, page, session = controller()
    task = asyncio.create_task(worker.serve(driver, SECRET, directory=ipc_dir))
    path = ipc_dir / "worker.sock"
    async with asyncio.timeout(2):
        while not path.exists():
            await asyncio.sleep(0.01)
        page.fail_close = True
        request = {"schema_version": 1, "request_id": "close", "session_id": session["session_id"],
                   "expected_control_epoch": 0, "operation": "close"}
        with pytest.raises(worker.Refused, match="control_outcome_unknown"):
            await driver.control(session, request)
        assert driver.status(session["scope"], session["session_id"])["state"] == "closing"
        await task
    assert not path.exists()  # main then closes the browser or exits the process.


def test_bootstrap_is_one_shot_and_secret_file_is_private(ipc_dir, monkeypatch):
    _driver, _page, session = controller()
    request = {key: session[key] for key in ("schema_version", "session_id", "scope", "lease_id", "surface", "expires_at")}
    request["required_capabilities"] = ["screenshot"]
    raw = wire.encode({"schema_version": 1, "secret": SECRET, "request": request, "document_html": "<input>", "incarnation": "c" * 64})
    monkeypatch.setattr(rpc, "DIRECTORY", ipc_dir)
    (ipc_dir / "incarnation").write_text("c" * 64)
    assert wire.decode(rpc.initialize(raw))["submitted"]
    path = ipc_dir / "bootstrap.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_bytes() == raw
    path.unlink()  # Worker consumes it; a second init still cannot replace custody.
    with pytest.raises(FileExistsError):
        rpc.initialize(raw)
    for invalid in (b"{}", wire.encode({"schema_version": True, "secret": SECRET, "request": request, "document_html": "", "incarnation": "c" * 64})):
        with pytest.raises(worker.Refused):
            worker.bootstrap(wire.decode(invalid))


def test_ipc_byte_budget_includes_worst_case_json_escaping():
    _driver, _page, session = controller()
    request = {key: session[key] for key in ("schema_version", "session_id", "scope", "lease_id", "surface", "expires_at")}
    request["required_capabilities"] = ["screenshot"]
    initial = {"schema_version": 1, "secret": SECRET, "request": request, "document_html": "\x00" * 65536, "incarnation": "c" * 64}
    assert len(wire.encode(worker.bootstrap(initial))) <= wire.MAX_REQUEST
    typed = message("act", {"session": session, "request": {"action": {"kind": "type", "text": "😀" * 16384}}})
    assert len(wire.encode(typed)) <= wire.MAX_REQUEST


@pytest.mark.parametrize("change", [{}, {"uid": 0}, {"CapEff": "1"}, {"CapBnd": "1"},
                                      {"CapPrm": "1"}, {"CapInh": "1"}, {"CapAmb": "1"},
                                      {"NoNewPrivs": "0"}, {"groups": [0]}, {"rtprio": (1, 1)}])
def test_shared_process_admission_checks_actual_limits(monkeypatch, change):
    checks = sandbox_checks
    values = {name: "0" for name in ("CapEff", "CapBnd", "CapPrm", "CapInh", "CapAmb")}
    values.update(NoNewPrivs="1")
    values.update({name: value for name, value in change.items() if name in values})
    from types import SimpleNamespace
    monkeypatch.setattr(checks, "Path", lambda _: SimpleNamespace(read_text=lambda: "\n".join(f"{name}:{value}" for name, value in values.items())))
    monkeypatch.setattr(checks.os, "getuid", lambda: change.get("uid", 1000))
    monkeypatch.setattr(checks.os, "getgroups", lambda: change.get("groups", [1000]))
    monkeypatch.setattr(checks.resource, "RLIMIT_RTPRIO", 14, raising=False)
    monkeypatch.setattr(checks.resource, "getrlimit", lambda _: change.get("rtprio", (0, 0)))
    if change:
        with pytest.raises(RuntimeError):
            checks.process_evidence()
    else:
        assert checks.process_evidence()["no_new_privileges"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("omit", [None, "PID namespaces", "Network namespaces", "Seccomp-BPF sandbox"])
async def test_shared_browser_admission_requires_all_sandboxes(omit):
    class Page:
        closed = False

        async def goto(self, url):
            assert url == "chrome://sandbox"

        def locator(self, selector):
            return self

        async def inner_text(self):
            return "\n".join(f"{name}\tYes" for name in ("PID namespaces", "Network namespaces", "Seccomp-BPF sandbox") if name != omit)

        async def close(self):
            self.closed = True

    page = Page()

    class Browser:
        version = "synthetic"

        async def new_page(self, **options):
            return page

    if omit:
        with pytest.raises(RuntimeError, match="required sandbox"):
            await worker.chromium_evidence(Browser())
    else:
        assert (await worker.chromium_evidence(Browser()))["browser_version"] == "synthetic"
    assert page.closed


@pytest.mark.asyncio
async def test_reserved_control_socket_fences_when_all_normal_slots_are_idle(ipc_dir):
    driver, page, session = controller()
    normal = worker.Endpoint(driver, SECRET)
    priority = worker.Endpoint(driver, SECRET, priority=True)
    path, control_path = ipc_dir / "worker.sock", ipc_dir / "control.sock"
    async with await asyncio.start_unix_server(normal.handle, str(path), limit=wire.MAX_REQUEST), \
               await asyncio.start_unix_server(priority.handle, str(control_path), limit=wire.MAX_REQUEST):
        idle = [await asyncio.open_unix_connection(str(path)) for _ in range(4)]
        await asyncio.sleep(.01)
        assert normal.connections == 4
        request = {"schema_version": 1, "request_id": "takeover", "session_id": session["session_id"],
                   "expected_control_epoch": 0, "operation": "takeover"}
        result = await asyncio.wait_for(exchange(control_path, message("control", {"session": session, "request": request})), 1)
        assert result["result"]["controller"] == "human"
        assert result["result"]["control_epoch"] == 1
        assert (await exchange(control_path, message("observe", {"session": session})))["error"] == "priority_operation_required"
        assert page.effects == []
        for _, writer in idle:
            writer.close()
            await writer.wait_closed()
        await asyncio.sleep(.01)


def test_bootstrap_cannot_cross_a_worker_process_replacement(ipc_dir, monkeypatch):
    _driver, _page, session = controller()
    request = {key: session[key] for key in ("schema_version", "session_id", "scope", "lease_id", "surface", "expires_at")}
    request["required_capabilities"] = ["screenshot"]
    raw = wire.encode({"schema_version": 1, "secret": SECRET, "request": request,
                       "document_html": "", "incarnation": "c" * 64})
    monkeypatch.setattr(rpc, "DIRECTORY", ipc_dir)
    (ipc_dir / "incarnation").write_text("d" * 64)
    with pytest.raises(ValueError, match="incarnation"):
        rpc.initialize(raw)
    assert not (ipc_dir / "initialized").exists()
    assert not (ipc_dir / "bootstrap.json").exists()
