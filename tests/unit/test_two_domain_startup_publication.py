"""Scripted listener I/O, real original Tasks; NOT native worker acceptance.

No socket, browser, container, altered role identity or registered profile is
used. #4078's original native-resource interleavings remain a separate gate.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from tests.unit.test_two_domain_browser_ownership import worker, ScriptedServer


@pytest.fixture
def startup(monkeypatch, tmp_path):
    events, servers, published = [], [], []
    state = SimpleNamespace(failure=None, blocked=False, entered=asyncio.Event(), release=asyncio.Event())
    class Controller:
        def __init__(self):
            self._session = {"state": "ready"}
            self.retired = False
        def _live(self):
            events.append("live")
        async def retire(self):
            self.retired = True
    controller = Controller()
    async def open_server(_callback, path, **_kwargs):
        name = Path(path).name
        events.append("open:"+name)
        if name == "control.sock":
            state.entered.set()
            if state.blocked:
                await state.release.wait()
            if state.failure == "open":
                raise original
            if state.failure == "retirement":
                controller._session["state"] = "closing"
        server = ScriptedServer()
        servers.append(server)
        return server
    def chmod(path, mode):
        name = Path(path).name
        events.append("chmod:"+name)
        assert mode == 0o660
        if name == "control.sock" and state.failure == "chmod":
            raise original
    def publish(path, content):
        assert Path(path).name == "admission.json"
        assert len(servers) == 2
        assert events == ["open:worker.sock", "chmod:worker.sock", "open:control.sock", "chmod:control.sock", "live"]
        if state.failure == "publication":
            raise original
        published.append(content)
        controller._session["state"] = "closing"
    original = OSError("original scripted startup failure")
    monkeypatch.setattr(worker.asyncio, "start_unix_server", open_server)
    monkeypatch.setattr(worker, "IPC", str(tmp_path))
    monkeypatch.setattr(worker, "os", SimpleNamespace(chmod=chmod))
    monkeypatch.setattr(worker, "publish_record", publish)
    async def serve(*, deadline_offset=20):
        return await worker.serve(controller, "a"*64, {"schema_version": 1},
                                  startup_deadline=asyncio.get_running_loop().time()+deadline_offset)
    return SimpleNamespace(state=state, controller=controller, servers=servers,
                           published=published, original=original, serve=serve)


@pytest.mark.asyncio
async def test_admission_follows_both_original_listener_returns_and_chmods(startup):
    await startup.serve()
    assert startup.published == [worker.encode({"schema_version": 1})]
    assert startup.controller.retired
    assert len(startup.servers) == 2
    assert all(server.closed and server.wait_ack == 1 for server in startup.servers)


@pytest.mark.asyncio
async def test_lifecycle_retirement_during_startup_cannot_publish_ready(startup):
    startup.state.failure = "retirement"
    await startup.serve()
    assert startup.published == [] and startup.controller.retired
    assert len(startup.servers) == 2
    assert all(server.closed and server.wait_ack == 1 for server in startup.servers)


@pytest.mark.asyncio
async def test_original_expired_startup_deadline_is_not_restarted_for_listeners(startup):
    with pytest.raises(BaseException):
        await startup.serve(deadline_offset=-1)
    assert startup.published == [] and startup.controller.retired
    assert all(server.closed and server.wait_ack == 1 for server in startup.servers)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["open", "chmod", "publication"])
async def test_startup_failure_publishes_nothing_and_joins_originals(startup, failure):
    startup.state.failure = failure
    with pytest.raises(BaseException) as caught:
        await startup.serve()
    assert startup.original in tuple(_explicit_control_chain(caught.value))
    assert startup.published == []
    assert len(startup.servers) == (1 if failure == "open" else 2)
    assert startup.controller.retired
    assert all(server.closed and server.wait_ack == 1 for server in startup.servers)


@pytest.mark.asyncio
async def test_withdrawal_joins_delayed_second_listener_without_publishing(startup):
    startup.state.blocked = True
    owner = asyncio.create_task(startup.serve())
    try:
        await asyncio.wait_for(startup.state.entered.wait(), 5)
        assert startup.published == []
        owner.cancel()
        await asyncio.sleep(0)
        assert not owner.done() and startup.published == []
    finally:
        startup.state.release.set()
        outcome = await await_owned_task(owner)
    assert outcome.error is not None
    assert startup.published == [] and startup.controller.retired
    assert len(startup.servers) == 2
    assert all(server.closed and server.wait_ack == 1 for server in startup.servers)
