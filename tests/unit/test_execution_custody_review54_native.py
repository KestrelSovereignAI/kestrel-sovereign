"""Original resources and receipts at the four review54 boundaries, offline."""
import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import Mock

import httpx
import pytest

from kestrel_sovereign._async_ownership import create_owned_task
from kestrel_sovereign.execution_custody import execution_terminal_error
from kestrel_sovereign.features.isolated_runtime import (
    IsolatedRuntimePreparationError,
    ProxyFeature,
)
from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerIdentity,
    PeerRequester,
)
from tests.unit.test_a2a_native_client import (
    native_signing as native_signing,  # noqa: PLC0414 - pytest fixture export
    peer_session as peer_session,  # noqa: PLC0414 - pytest fixture export
    recipient_identity as recipient_identity,  # noqa: PLC0414 - pytest fixture export
    reserve,
)
from kestrel_sovereign.a2a.native_client import NativePeerSession
from kestrel_sovereign.execution_custody import owned_native_http_client

from tests.unit.test_execution_custody_review43_native import carries
from tests.unit.test_execution_custody_review48_native import terminal
from tests.unit.test_execution_custody_review49_native import KINDS, join
from tests.unit.test_isolated_feature_runtime import (
    FakeIsolatedClient,
    _configure_idle_lifecycle,
    _idle_test_runtime,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("revalidation", ["valid", "not-executable", "missing-replacement"])
async def test_native_contained_bin_never_acquires_workspace_deletion_custody(monkeypatch, tmp_path, revalidation):
    agent = Mock(did="did:test:isolated-runtime", features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    workspace = feature._prepare_runtime_workspace()
    runnable = workspace / "operator-service"
    runnable.write_text("#!/bin/sh\nexit 0\n")
    runnable.chmod(0o700)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", str(runnable))
    feature.resolve_runtime_paths()
    feature._idle_retired = True
    feature._client = None
    if revalidation != "valid":
        if revalidation == "not-executable":
            runnable.chmod(0o600)
        else:
            monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", str(workspace / "missing"))
        with pytest.raises(Exception):
            feature.resolve_runtime_paths()
    assert not feature.runtime_telemetry_snapshot().cleanup_eligible
    with pytest.raises(IsolatedRuntimePreparationError, match="immutable BIN"):
        await feature.reclaim_idle_workspace()
    assert runnable.read_text() == "#!/bin/sh\nexit 0\n"
    assert workspace.is_dir()


async def accepted_task(p):
    dispatch, wire = await reserve(p)
    return (await p.session.send(dispatch, wire)).dispatch


def stream_response(request, p, body):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("boundary", ["authorization", "subscription"])
async def test_native_subscription_timeout_keeps_original_control(peer_session, monkeypatch, kind, boundary):
    p = peer_session
    dispatch = await accepted_task(p)
    original = terminal(kind)
    entered = asyncio.Event()
    effects = []

    async def native_pending():
        await p.admission.backend.execute("CREATE TABLE review54_effects(id INTEGER PRIMARY KEY)")
        await p.admission.backend.execute("INSERT INTO review54_effects VALUES (1)")
        effects.append(True)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise original

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            await native_pending()
            yield b""

    if boundary == "authorization":
        monkeypatch.setattr(p.session, "_authorize", native_pending)
    else:
        monkeypatch.setattr(p.http, "_transport", httpx.MockTransport(lambda request: stream_response(request, p, Body())))

    async def read():
        async with asyncio.timeout(0.05):
            return [event async for event in p.session.subscribe(dispatch)]

    result = await join(create_owned_task(read()))
    assert entered.is_set() and effects == [True]
    assert carries(result.error, original), "native deadline receipt became ordinary EOF"
    assert await p.admission.backend.fetch_val("SELECT COUNT(*) FROM review54_effects") == 1


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ["listing", "native-session"])
async def test_native_peer_temporary_client_close_physically_joins(peer_session, monkeypatch, kind, route):
    p = peer_session
    dispatch = await accepted_task(p)
    original = terminal(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    retired = []
    peer = PeerIdentity(p.admission.scope.recipient_id, "peer", "peer")
    requester = PeerRequester(p.store.sender_scope.recipient_id, object())
    base_transport = p.http._transport

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await p.admission.backend.execute("CREATE TABLE review54_effects(id INTEGER PRIMARY KEY)")
            await p.admission.backend.execute("INSERT INTO review54_effects VALUES (1)")
            if route == "listing":
                return httpx.Response(200, json=[])
            return await base_transport.handle_async_request(request)

        async def aclose(self):
            entered.set()
            await release.wait()
            retired.append(self)
            raise original

    def client(**kwargs):
        return httpx.AsyncClient(transport=Transport(), **kwargs)

    @asynccontextmanager
    async def native_session(selected, selected_peer):
        assert selected is requester and selected_peer is peer
        async with owned_native_http_client(client()) as http:
            session = NativePeerSession(**{**p.arguments, "http_client": http})
            try:
                yield session
            finally:
                await session.close()

    router = LocalHostPeerDirectory("https://peer.test", client_factory=client, native_session_factory=native_session)
    async def authorize(*args):
        return peer
    monkeypatch.setattr(router, "_authorize_peer", authorize)

    async def operation():
        if route == "listing":
            await router.list_peers(requester)
        else:
            async with router.open_a2a_session(requester, peer) as session:
                await session.get_task(dispatch)

    task = create_owned_task(operation())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "HTTPX closed flag hid unfinished native transport cleanup"
        release.set()
        outcome = await join(task)
        assert carries(execution_terminal_error(outcome.error, outcome.cancellation), original)
        assert len(retired) == 1
        assert await p.admission.backend.fetch_val("SELECT COUNT(*) FROM review54_effects") == 1
    finally:
        release.set()
        await join(task)


@pytest.mark.parametrize("kind", KINDS)
async def test_native_peer_subscription_response_close_physically_joins(peer_session, monkeypatch, kind):
    from google.protobuf.json_format import MessageToDict

    p = peer_session
    dispatch = await accepted_task(p)
    current = await p.session.get_task(dispatch)
    dispatch, response = current.dispatch, current.response
    original = terminal(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    closes = []

    class Body(httpx.AsyncByteStream):
        def __init__(self, request):
            native = {"task": MessageToDict(response)}
            if p.arguments["interface"].protocol_binding == "JSONRPC":
                native = {"jsonrpc": "2.0", "id": json.loads(request.content)["id"], "result": native}
            self.wire = ("data: " + json.dumps(native) + "\n\n").encode()

        async def __aiter__(self):
            yield self.wire

        async def aclose(self):
            entered.set()
            await release.wait()
            await p.admission.backend.execute("CREATE TABLE review54_effects(id INTEGER PRIMARY KEY)")
            await p.admission.backend.execute("INSERT INTO review54_effects VALUES (1)")
            closes.append(True)
            raise original

    monkeypatch.setattr(p.http, "_transport", httpx.MockTransport(lambda request: stream_response(request, p, Body(request))))
    stream = p.session.subscribe(dispatch)
    async def consume():
        return [event async for event in stream]
    task = create_owned_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "native response body cleanup escaped its original owner"
        release.set()
        outcome = await join(task)
        assert carries(execution_terminal_error(outcome.error, outcome.cancellation), original)
        assert closes == [True]
        assert await p.admission.backend.fetch_val("SELECT COUNT(*) FROM review54_effects") == 1
    finally:
        release.set()
        await join(task)
        await stream.aclose()
