"""Original resources and receipts at the four review54 boundaries, offline."""
import asyncio
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
from tests.unit.test_execution_custody_review29 import native_owner, retire
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


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("boundary", ["authorization", "subscription"])
async def test_native_subscription_timeout_keeps_original_control(tmp_path, monkeypatch, kind, boundary):
    agent, db = await native_owner(tmp_path)
    original = terminal(kind)
    entered = asyncio.Event()
    peer, requester = PeerIdentity("peer", "peer", "peer"), PeerRequester("caller", object())

    async def native_pending():
        await db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise original

    async def authorize(*args):
        if boundary == "authorization":
            await native_pending()
        return peer

    async def subscribe(*args):
        await native_pending()
        yield  # The real original async-generator frame must unwind.

    router = LocalHostPeerDirectory("http://offline.invalid", local_subscribe=subscribe)
    monkeypatch.setattr(router, "_authorize_peer", authorize)
    async def read():
        return [event async for event in router.subscribe_a2a_task(requester, peer, "task", timeout_seconds=0.05)]
    try:
        result = await join(create_owned_task(read()))
        assert entered.is_set()
        assert carries(result.error, original), "native deadline receipt became ordinary EOF"
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ["listing", "invoke", "send", "read", "cancel", "stop"])
async def test_native_peer_temporary_client_close_physically_joins(tmp_path, kind, route):
    agent, db = await native_owner(tmp_path)
    original = terminal(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    clients, retired = [], []
    peer, requester = PeerIdentity("peer", "peer", "peer"), PeerRequester("caller", object())

    class Transport(httpx.AsyncBaseTransport):
        target = False

        async def handle_async_request(self, request):
            self.target = route == "listing" or request.url.path != "/api/agents"
            if not self.target:
                return httpx.Response(200, json=[{"id": "peer", "name": "peer"}])
            await db.execute("INSERT INTO effects VALUES (1)")
            return httpx.Response(200, json={"id": "task"} if route != "listing" else [])

        async def aclose(self):
            if self.target:
                entered.set()
                await release.wait()
                retired.append(self)
                raise original

    def factory(**kwargs):
        client = httpx.AsyncClient(transport=Transport(), **kwargs)
        clients.append(client)
        return client

    router = LocalHostPeerDirectory("http://offline.invalid", client_factory=factory, principal_payload_factory=lambda *args: {})
    operations = {
        "listing": lambda: router.list_peers(requester),
        "invoke": lambda: router.invoke(requester, peer, "hello"),
        "send": lambda: router.send_a2a_task(requester, peer, {}),
        "read": lambda: router.get_a2a_task(requester, peer, "task"),
        "cancel": lambda: router.cancel_a2a_task(requester, peer, "task", {}),
        "stop": lambda: router.stop_peer(requester, peer, {}),
    }
    task = create_owned_task(operations[route]())
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
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        release.set()
        await join(task)
        for client in clients:
            await client.aclose()
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_native_peer_subscription_response_close_physically_joins(tmp_path, monkeypatch, kind):
    agent, db = await native_owner(tmp_path)
    original = terminal(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    peer, requester = PeerIdentity("peer", "peer", "peer"), PeerRequester("caller", object())
    closes = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"event: status\ndata: {}\n\n"

        async def aclose(self):
            entered.set()
            await release.wait()
            await db.execute("INSERT INTO effects VALUES (1)")
            closes.append(True)
            raise original

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Body())))
    router = LocalHostPeerDirectory("http://offline.invalid", client_factory=lambda **kwargs: client, principal_payload_factory=lambda *args: {})
    async def authorize(*args):
        return peer
    monkeypatch.setattr(router, "_authorize_peer", authorize)
    stream = router.subscribe_a2a_task(requester, peer, "task", timeout_seconds=2)
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
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        release.set()
        await join(task)
        await stream.aclose()
        await client.aclose()
        await retire(agent)
        await db.close()
