"""Scoped discovery and native-session admission, without the retired transport."""

from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerAccessDeniedError,
    PeerDirectoryConfigurationError,
    PeerIdentity,
    PeerNotFoundError,
    PeerProtocolError,
    PeerRequester,
)
from kestrel_sovereign.features.peers.feature import PeersFeature


class ScopedDirectory:
    def __init__(self):
        self.a, self.b = object(), object()
        self.entries = {
            self.a: [PeerIdentity("did:tenant-a:peer", "companion", "route-a", name="Companion")],
            self.b: [PeerIdentity("did:tenant-b:peer", "companion", "route-b", name="Companion")],
        }
        self.resolve_calls = []

    def scoped(self, requester):
        try:
            return self.entries[requester.authorization_scope]
        except KeyError:
            raise PeerAccessDeniedError("private provider diagnostic") from None

    async def list_peers(self, requester):
        return self.scoped(requester)

    async def resolve_peer(self, requester, name):
        self.resolve_calls.append((requester, name))
        return next((p for p in self.scoped(requester) if name.casefold() in {p.slug.casefold(),p.name.casefold()}), None)

    async def resolve_peer_by_agent_id(self, requester, identity):
        return next((p for p in self.scoped(requester) if p.agent_id == identity), None)


def test_local_directory_binds_one_native_session_owner_without_changing_roster():
    router = LocalHostPeerDirectory("https://parent.test", transport_key="roster-only")
    factory = lambda requester, peer: None
    assert not router.has_native_session_factory
    with pytest.raises(PeerDirectoryConfigurationError, match="callable"):
        router.bind_native_session_factory(None)
    router.bind_native_session_factory(factory)
    assert router.has_native_session_factory and router._native_session_factory is factory
    assert router._host_url == "https://parent.test"
    assert router._headers()["X-Kestrel-A2A-Key"] == "roster-only"
    with pytest.raises(PeerDirectoryConfigurationError, match="already bound"):
        router.bind_native_session_factory(lambda requester, peer: None)


def feature_for(directory, scope):
    agent = SimpleNamespace(did="did:caller", _agent_name="Caller")
    feature = PeersFeature(agent)
    feature._peer_router = directory
    feature._peer_requester = PeerRequester(agent.did, scope)
    feature._own_name = "Caller"
    return feature


@pytest.mark.parametrize("missing", ["router", "requester", "untrusted"])
def test_injected_directory_requires_the_authenticated_pair(missing):
    directory = ScopedDirectory()
    feature = feature_for(directory, directory.a)
    if missing == "router":
        feature._peer_router = None
    elif missing == "requester":
        feature._peer_requester = None
    else:
        feature._peer_requester = SimpleNamespace(identity="did:caller", authorization_scope=directory.a)
    with pytest.raises(PeerDirectoryConfigurationError):
        feature.hosted_peer_directory_context()


async def test_duplicate_names_resolve_within_each_opaque_scope():
    directory = ScopedDirectory()
    for scope, expected in ((directory.a, "did:tenant-a:peer"),(directory.b, "did:tenant-b:peer")):
        feature = feature_for(directory, scope)
        router, requester, peer = await feature._resolve_automatic_peer("Companion")
        assert router is directory and requester.authorization_scope is scope
        assert peer.agent_id == expected
        result = await feature.list_peers()
        assert result.status is ToolResultStatus.OK
        assert result.data["peers"][0]["slug"] == "companion"


@pytest.mark.parametrize("probe", ["did:tenant-b:peer", " DID:tenant-b:peer", None, 42])
async def test_did_and_non_name_probes_are_refused_before_directory_lookup(probe):
    directory = ScopedDirectory()
    feature = feature_for(directory, directory.a)
    with pytest.raises(PeerNotFoundError):
        await feature._resolve_automatic_peer(probe)
    assert directory.resolve_calls == []


async def test_retained_identity_ignores_name_reassignment_and_uses_current_route():
    directory = ScopedDirectory()
    feature = feature_for(directory, directory.a)
    original = directory.entries[directory.a][0]
    renamed = replace(original, slug="new-name", name="Renamed", routing_key="new-route")
    directory.entries[directory.a] = [renamed, PeerIdentity("did:replacement", "companion", "replacement")]
    assert (await feature._resolve_retained_automatic_peer(original.agent_id))[2] == renamed
    directory.entries[directory.a] = []
    with pytest.raises(PeerNotFoundError):
        await feature._resolve_retained_automatic_peer(original.agent_id)


@pytest.mark.parametrize("returned", ["not-a-peer", None, PeerIdentity("did:other", "other", "other")])
async def test_retained_resolution_refuses_invalid_or_substituted_identity(returned):
    directory = ScopedDirectory()
    directory.resolve_peer_by_agent_id = AsyncMock(return_value=returned)
    feature = feature_for(directory, directory.a)
    with pytest.raises((PeerProtocolError, PeerNotFoundError)):
        await feature._resolve_retained_automatic_peer("did:tenant-a:peer")


async def test_self_target_is_refused_by_stable_identity():
    directory = ScopedDirectory()
    directory.entries[directory.a] = [PeerIdentity("did:caller", "other-name", "route")]
    feature = feature_for(directory, directory.a)
    from kestrel_sovereign.features.peers.directory import PeerSelfTargetError
    with pytest.raises(PeerSelfTargetError):
        await feature._resolve_automatic_peer("other-name")


@pytest.mark.parametrize("rows", ["invalid", ["invalid"], [None]])
async def test_malformed_provider_listing_does_not_leak_private_diagnostics(rows):
    directory = ScopedDirectory()
    directory.list_peers = AsyncMock(return_value=rows)
    result = await feature_for(directory, directory.a).list_peers()
    assert result.status is ToolResultStatus.ERROR
    assert not result.data["peers"]
    assert "private" not in str(result)


async def test_unexpected_provider_error_is_sanitized():
    directory = ScopedDirectory()
    directory.resolve_peer = AsyncMock(side_effect=RuntimeError("private other-tenant connection"))
    with pytest.raises(PeerProtocolError, match="resolution failed") as caught:
        await feature_for(directory, directory.a)._resolve_automatic_peer("Companion")
    assert "other-tenant" not in str(caught.value)


@pytest.fixture
def local():
    state = SimpleNamespace(entries=[{"id":"did:peer", "name":"Display", "routing_name":"route", "status":"online"}], seen=[], opened=[])
    def handle(request):
        state.seen.append(request)
        return httpx.Response(200, json={"agents":state.entries})
    @asynccontextmanager
    async def factory(requester, peer):
        state.opened.append((requester, peer))
        yield state
    directory = LocalHostPeerDirectory(
        "https://host.test", transport_key="roster-only",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        native_session_factory=factory,
    )
    return SimpleNamespace(state=state, directory=directory, requester=PeerRequester("did:caller", object()))


async def test_local_discovery_uses_authenticated_roster_and_supplied_native_session(local):
    p = local
    peer = await p.directory.resolve_peer(p.requester, "Display")
    assert peer.agent_id == "did:peer" and peer.routing_key == "route"
    async with p.directory.open_a2a_session(p.requester, peer) as session:
        assert session is p.state
    assert len(p.state.seen) == 2
    from kestrel_sovereign.a2a.transport_auth import A2A_TRANSPORT_KEY_HEADER
    assert all(r.method == "GET" and r.url.path == "/api/agents" and r.headers[A2A_TRANSPORT_KEY_HEADER] == "roster-only" for r in p.state.seen)
    assert p.state.opened == [(p.requester, peer)]


@pytest.mark.parametrize("change", ["removed", "route", "duplicate"])
async def test_local_session_reauthorizes_retained_peer_before_native_factory(local, change):
    p = local
    peer = await p.directory.resolve_peer(p.requester, "Display")
    if change == "removed":
        p.state.entries = []
    elif change == "route":
        p.state.entries[0]["routing_name"] = "new-route"
    else:
        p.state.entries.append(dict(p.state.entries[0]))
    with pytest.raises(PeerNotFoundError):
        async with p.directory.open_a2a_session(p.requester, peer):
            pytest.fail("revoked route opened")
    assert p.state.opened == []


async def test_local_inbound_sender_authorization_requires_unique_live_identity(local):
    p = local
    assert await p.directory.authorize_inbound_sender(p.requester, "did:peer")
    assert not await p.directory.authorize_inbound_sender(p.requester, "Display")
    p.state.entries.append(dict(p.state.entries[0]))
    assert not await p.directory.authorize_inbound_sender(p.requester, "did:peer")


async def test_missing_native_factory_fails_before_any_peer_operation(local):
    p = local
    peer = await p.directory.resolve_peer(p.requester, "Display")
    p.directory._native_session_factory = None
    with pytest.raises(PeerDirectoryConfigurationError, match="native A2A session"):
        async with p.directory.open_a2a_session(p.requester, peer):
            pytest.fail("opened without the native session factory")
    assert p.state.opened == []


@pytest.mark.parametrize("row", [None, {}, {"name":"Display"}, {"id":"", "routing_name":"route"}])
async def test_local_roster_cannot_invent_durable_identity_from_display_fields(local, row):
    local.state.entries = [row]
    with pytest.raises(PeerProtocolError):
        await local.directory.list_peers(local.requester)
