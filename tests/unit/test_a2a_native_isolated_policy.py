"""Explicit isolation is a deny-all scope, never recovery from broken trust."""

import time
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from kestrel_sdk.native_a2a.client.errors import A2AClientError
from starlette.exceptions import HTTPException

from kestrel_sovereign.a2a.native_standalone import (
    _configure_single_agent_native_peer_policy,
    open_process_native_peer_session,
)
from kestrel_sovereign.features.peers.directory import (
    IsolatedPeerDirectory,
    LocalHostPeerDirectory,
    PeerAccessDeniedError,
    PeerDirectoryConfigurationError,
    PeerIdentity,
    PeerNotFoundError,
    PeerRequester,
)
from tests.unit.test_a2a_native_standalone import (
    signed_sdk_client,
)
from tests.unit.test_a2a_native_standalone import (
    single_candidate as single_candidate,  # noqa: PLC0414 - fixture re-export
)
from tests.unit.test_a2a_native_standalone import (
    single_identity as single_identity,  # noqa: PLC0414 - fixture dependency
)
from tests.utils.native_a2a_security import send, signed_context


def configure(recipient, environment, *, router=None, requester=None):
    return _configure_single_agent_native_peer_policy(
        recipient, environment, router=router, requester=requester,
    )


def test_missing_policy_preserves_original_even_when_malformed():
    recipient = SimpleNamespace(did="did:kestrel:isolated")
    malformed = object()
    assert configure(recipient, {}, router=malformed) == (malformed, None)
    assert vars(recipient) == {"did": "did:kestrel:isolated"}


@pytest.mark.parametrize("mode", ["", "disabled", "automatic", "ISOLATED", " isolated "])
def test_unknown_policy_refuses_without_mutation(mode):
    recipient = SimpleNamespace(did="did:kestrel:isolated")
    with pytest.raises(PeerDirectoryConfigurationError, match="must be 'isolated'"):
        configure(recipient, {"KESTREL_A2A_PEER_POLICY": mode})
    assert vars(recipient) == {"did": "did:kestrel:isolated"}


@pytest.mark.parametrize("variable", [
    "KESTREL_A2A_PEER_IDENTITY_ROOTS",
    "KESTREL_A2A_PEER_IDENTITY_DOCUMENTS",
    "KESTREL_A2A_PEER_IDENTITY_DOCUMENTS_FILE",
    "KESTREL_A2A_PEER_IDENTITY_DOCUMENTS_SHA256",
    "KESTREL_A2A_FEDERATED_DID",
    "KESTREL_HOST_URL",
])
@pytest.mark.parametrize("value", ["", "false", "configured"])
def test_isolation_refuses_peer_configuration_even_empty_or_disabled(variable, value):
    recipient = SimpleNamespace(did="did:kestrel:isolated")
    with pytest.raises(PeerDirectoryConfigurationError, match="conflicts"):
        configure(recipient, {"KESTREL_A2A_PEER_POLICY": "isolated", variable: value})
    assert vars(recipient) == {"did": "did:kestrel:isolated"}


@pytest.mark.parametrize("attribute", [
    "peer_directory_router", "peer_requester", "a2a_did_resolver",
    "_a2a_process_did_resolver", "a2a_inbound_sender_authorizer",
])
def test_isolation_cannot_replace_an_injected_or_revoked_dependency(attribute):
    original = object()
    recipient = SimpleNamespace(did="did:kestrel:isolated", **{attribute: original})
    with pytest.raises(PeerDirectoryConfigurationError, match="conflicts"):
        configure(recipient, {"KESTREL_A2A_PEER_POLICY": "isolated"})
    assert getattr(recipient, attribute) is original
    assert set(vars(recipient)) == {"did", attribute}


@pytest.mark.parametrize("live_context", ["router", "requester", "both"])
def test_isolation_cannot_replace_effective_feature_directory(live_context):
    recipient = SimpleNamespace(did="did:kestrel:isolated")
    router = LocalHostPeerDirectory("https://parent.test")
    requester = PeerRequester(recipient.did, object())
    with pytest.raises(PeerDirectoryConfigurationError, match="conflicts"):
        configure(
            recipient, {"KESTREL_A2A_PEER_POLICY": "isolated"},
            router=router if live_context != "requester" else None,
            requester=requester if live_context != "router" else None,
        )
    assert vars(recipient) == {"did": "did:kestrel:isolated"}


async def test_isolated_scope_denies_resolution_transport_and_all_signing_dids():
    recipient = SimpleNamespace(did="did:kestrel:isolated")
    directory, requester = configure(recipient, {"KESTREL_A2A_PEER_POLICY": "isolated"})
    assert isinstance(directory, IsolatedPeerDirectory)
    assert requester.identity == recipient.did
    assert requester.authorization_scope is directory
    assert await directory.list_peers(requester) == ()
    for sender in (recipient.did, "did:web:untrusted.test:agent", "", None):
        assert recipient.a2a_did_resolver(sender) is None
        assert recipient._a2a_process_did_resolver.directory_agent_id(sender) is None
        assert await directory.authorize_inbound_sender(requester, sender) is False
    with pytest.raises(PeerNotFoundError, match="no authorized peers"):
        await directory.resolve_peer(requester, "someone")
    with pytest.raises(PeerNotFoundError, match="no authorized peers"):
        await directory.resolve_peer_by_agent_id(requester, "did:kestrel:someone")
    with pytest.raises(PeerNotFoundError, match="no authorized peers"):
        directory.open_a2a_session(requester, PeerIdentity("did:kestrel:p", "p", "p"))


async def test_copied_requester_cannot_spend_isolated_scope():
    directory = IsolatedPeerDirectory("did:kestrel:isolated")
    forged = PeerRequester(directory.requester.identity, directory)
    with pytest.raises(PeerAccessDeniedError, match="does not own"):
        await directory.list_peers(forged)
    with pytest.raises(PeerAccessDeniedError, match="does not own"):
        await directory.authorize_inbound_sender(forged, "did:kestrel:p")
    with pytest.raises(PeerAccessDeniedError, match="does not own"):
        await directory.resolve_peer(forged, "p")
    with pytest.raises(PeerAccessDeniedError, match="does not own"):
        await directory.resolve_peer_by_agent_id(forged, "did:kestrel:p")
    with pytest.raises(PeerAccessDeniedError, match="does not own"):
        directory.open_a2a_session(forged, PeerIdentity("did:kestrel:p", "p", "p"))


@pytest_asyncio.fixture(params=["server", "direct"])
async def isolated_single(single_candidate, request):
    from kestrel_sovereign.a2a.native_standalone import (
        prepare_single_agent_native_publication,
    )
    from kestrel_sovereign.server import _onboard_single_agent_native

    p = single_candidate
    for attribute in (
        "peer_directory_router", "peer_requester", "a2a_did_resolver",
        "a2a_inbound_sender_authorizer",
    ):
        setattr(p.recipient, attribute, None)
    publish = (
        _onboard_single_agent_native if request.param == "server"
        else prepare_single_agent_native_publication
    )
    publication = await publish(p.app, p.recipient, {
        "KESTREL_BASE_URL": "http://single.native.test",
        "KESTREL_A2A_PUSH_ENABLED": "false",
        "KESTREL_A2A_PEER_POLICY": "isolated",
    })
    owner = p.recipient._native_a2a_runtime_authority
    try:
        yield SimpleNamespace(**vars(p), owner=owner, publication=publication)
    finally:
        await owner.close()


@pytest.mark.parametrize("protocol", ["JSONRPC", "HTTP+JSON"])
async def test_real_signed_sdk_sender_is_refused_without_creating_task(isolated_single, protocol):
    p = isolated_single
    async with httpx.AsyncClient(transport=httpx.ASGITransport(p.app)) as http:
        client = await signed_sdk_client(p, http, protocol, clock=time.time)
        with pytest.raises(A2AClientError) as refused:
            async for _event in client.send_message(send(p.admission)):
                pytest.fail("Isolated policy accepted a signed sender")
        assert refused.value.__cause__.response.status_code in {401, 403}
    assert await p.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []
    assert p.publication.is_published


@pytest.mark.parametrize("changed", ["peer_directory_router", "a2a_did_resolver"])
async def test_isolated_publication_still_refuses_runtime_replacement(isolated_single, changed):
    p = isolated_single
    setattr(p.recipient, changed, object())
    params = send(p.admission)
    selected = SimpleNamespace(**(vars(p.admission) | {"guard": p.publication._guard}))
    context = signed_context(selected, params)
    with pytest.raises(HTTPException) as refused:
        async with p.publication._guard.admit(params, "SendMessage", context):
            pytest.fail("Isolated policy accepted replacement runtime authority")
    assert refused.value.status_code == 403
    assert await p.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []


async def test_isolated_process_egress_refuses_before_any_transport(isolated_single):
    p = isolated_single
    with pytest.raises(PeerAccessDeniedError, match="no launcher-attested signing identity"):
        async with open_process_native_peer_session(
            p.owner,
            resolver=p.recipient._a2a_process_did_resolver,
            requester=p.recipient.peer_requester,
            peer=PeerIdentity("did:kestrel:unknown", "unknown", "unknown"),
            timeout=httpx.Timeout(1),
        ):
            pytest.fail("Isolated peer policy opened a transport")
    assert await p.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []


@pytest.mark.parametrize("entrypoint", ["server", "direct"])
@pytest.mark.parametrize("mode", [None, "", "automatic"])
async def test_server_does_not_infer_isolation_or_publish_on_bad_mode(single_candidate, mode, entrypoint):
    from kestrel_sovereign.a2a.native_standalone import (
        prepare_single_agent_native_publication,
    )
    from kestrel_sovereign.server import _onboard_single_agent_native

    p = single_candidate
    for attribute in (
        "peer_directory_router", "peer_requester", "a2a_did_resolver",
        "a2a_inbound_sender_authorizer",
    ):
        setattr(p.recipient, attribute, None)
    environment = {
        "KESTREL_BASE_URL": "http://single.native.test",
        "KESTREL_A2A_PUSH_ENABLED": "false",
    }
    if mode is not None:
        environment["KESTREL_A2A_PEER_POLICY"] = mode
    expected = "explicit scoped peer policy" if mode is None else "must be 'isolated'"
    publish = (
        _onboard_single_agent_native if entrypoint == "server"
        else prepare_single_agent_native_publication
    )
    with pytest.raises((ValueError, PeerDirectoryConfigurationError), match=expected):
        await publish(p.app, p.recipient, environment)
    assert not hasattr(p.recipient, "_native_a2a_runtime_authority")
    assert p.app.routes == []


@pytest.mark.parametrize("mode", ["isolated", "automatic"])
@pytest.mark.parametrize("entrypoint", ["registration", "publication"])
async def test_hosted_entrypoints_refuse_standalone_policy_before_recipient_access(
    monkeypatch, mode, entrypoint,
):
    from fastapi import FastAPI

    from kestrel_sovereign.a2a.native_publication import (
        prepare_hosted_native_publication,
    )
    from kestrel_sovereign.server import _onboard_host_registered_agent

    environment = {"KESTREL_A2A_PEER_POLICY": mode}
    monkeypatch.setenv("KESTREL_A2A_PEER_POLICY", mode)
    app = FastAPI()
    original_routes = tuple(app.routes)
    # These are deliberately unusable inputs: incompatible host configuration
    # must refuse BEFORE reading a recipient, installing trust or owning work.
    manager, recipient = object(), object()
    expected = "only supported by standalone" if mode == "isolated" else "must be 'isolated'"
    with pytest.raises(PeerDirectoryConfigurationError, match=expected):
        if entrypoint == "registration":
            await _onboard_host_registered_agent(app, manager, "test", recipient)
        else:
            await prepare_hosted_native_publication(app, manager, "test", recipient, environment)
    assert tuple(app.routes) == original_routes
    assert not hasattr(app.state, "native_a2a_http")


async def test_direct_standalone_publication_cannot_ignore_isolated_policy(single_candidate):
    from kestrel_sovereign.a2a.native_standalone import (
        prepare_single_agent_native_publication,
    )

    p = single_candidate
    try:
        # The fixture has a real initialized scoped TaskManager and an explicit
        # directory/resolver/authorizer that permits its signed test principal.
        # A library embedding must not publish that trust under an isolated
        # declaration, even without the server onboarding wrapper.
        with pytest.raises(PeerDirectoryConfigurationError, match="conflicts"):
            await prepare_single_agent_native_publication(p.app, p.recipient, {
                "KESTREL_BASE_URL": "http://single.native.test",
                "KESTREL_A2A_PUSH_ENABLED": "false",
                "KESTREL_A2A_PEER_POLICY": "isolated",
            })
    finally:
        owner = getattr(p.recipient, "_native_a2a_runtime_authority", None)
        if owner is not None:
            await owner.close()
