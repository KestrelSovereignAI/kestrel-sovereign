"""Launcher-attested process peers use real HTTP and the shared SDK session."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
from kestrel_sdk.native_a2a.types import Role, SendMessageRequest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from kestrel_sovereign.a2a.did_registry import (
    A2A_PEER_STABLE_AGENT_ID_FIELD,
    ProcessA2ADidResolver,
)
from kestrel_sovereign.a2a.native_publication import NativeHostingConfiguration
from kestrel_sovereign.a2a.native_routes import native_route_prefix
from kestrel_sovereign.a2a.native_standalone import (
    open_process_native_peer_session,
    prepare_single_agent_native_publication,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.transport_auth import A2A_TRANSPORT_KEY_HEADER
from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerAccessDeniedError,
    PeerNotFoundError,
    PeerRequester,
    PeerUnavailableError,
)
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from tests.unit.test_a2a_native_standalone import (
    single_candidate as single_candidate,  # noqa: PLC0414 - fixture re-export
)
from tests.unit.test_a2a_native_standalone import (
    single_identity as single_identity,  # noqa: PLC0414 - fixture dependency
)
from tests.utils.native_a2a import manager_for, serve_native_app
from tests.utils.native_a2a_security import PRINCIPAL, SENDER


@pytest_asyncio.fixture(params=["", "/proxy"])
async def process_peer(single_candidate, request):
    from kestrel_sovereign.server import _onboard_single_agent_native

    p = single_candidate
    recipient_id = p.recipient.did
    recipient_manager = manager_for(
        p.admission.backend, NativeRecipientScope(recipient_id), owns_backend=False
    )
    sender_manager = manager_for(
        p.admission.backend, NativeRecipientScope(PRINCIPAL), owns_backend=False
    )
    await recipient_manager.initialize()
    await sender_manager.initialize()
    for manager in (recipient_manager, sender_manager):
        manager.register_provider(
            "process-test", p.admission.manager.get_agent_skills(), object()
        )
    p.recipient.task_manager = recipient_manager
    state = SimpleNamespace(allowed=True, requests=[])

    async def roster(request):
        assert request.headers.get(A2A_TRANSPORT_KEY_HEADER) == "roster-only"
        return JSONResponse({"agents": [
            {"id": recipient_id, "name": "Target", "routing_name": "target"}
        ] if state.allowed else []})

    p.app.routes.append(Route("/api/agents", roster))
    async def record_requests(scope, receive, send):
        if scope["type"] == "http":
            state.requests.append((scope["path"], dict(scope["headers"])))
        await p.app(scope, receive, send)

    recipient_owner, sender_owner = None, None
    try:
        async with serve_native_app(record_requests, root_path=request.param) as base:
            await prepare_single_agent_native_publication(p.app, p.recipient, {
                "KESTREL_BASE_URL": base,
                "KESTREL_A2A_DISCOVERY_AGENT": "target",
                "KESTREL_A2A_PUSH_ENABLED": "false",
            })
            recipient_owner = p.recipient._native_a2a_runtime_authority
            resolver = ProcessA2ADidResolver((p.document | {
                A2A_PEER_STABLE_AGENT_ID_FIELD: recipient_id
            },))
            ready = asyncio.Event()
            ready.set()
            router = LocalHostPeerDirectory(base, transport_key="roster-only")
            requester = PeerRequester(PRINCIPAL, object())
            sender = SimpleNamespace(
                did=PRINCIPAL,
                agent_id=PRINCIPAL,
                task_manager=sender_manager,
                identity=AgentIdentity(
                    hybrid_keypair=p.admission.signing[0],
                    new_did=SENDER,
                    new_verification_methods=p.admission.signing[1]["verificationMethod"],
                ),
                _runtime_publication_ready=ready,
                _execution_custody=None,
                peer_directory_router=router,
                peer_requester=requester,
                a2a_did_resolver=resolver.resolve,
                _a2a_process_did_resolver=resolver,
                get_public_profile=p.recipient.get_public_profile,
                resolve_effective_name=p.recipient.resolve_effective_name,
            )
            # Actual server onboarding supplies the process session factory;
            # the test never injects it or calls a substitute transport.
            await _onboard_single_agent_native(Starlette(), sender, {
                "KESTREL_BASE_URL": base,
                "KESTREL_A2A_DISCOVERY_AGENT": "sender",
                "KESTREL_A2A_PUSH_ENABLED": "false",
            })
            sender_owner = sender._native_a2a_runtime_authority
            peer = await router.resolve_peer(requester, "Target")
            assert peer is not None
            yield SimpleNamespace(
                **vars(p), sender=sender, sender_owner=sender_owner,
                sender_manager=sender_manager, recipient_manager=recipient_manager,
                resolver=resolver, router=router, requester=requester, peer=peer,
                state=state,
            )
    finally:
        if sender_owner is not None:
            await sender_owner.close()
        if recipient_owner is not None:
            await recipient_owner.close()
        await sender_manager.close()
        await recipient_manager.close()


async def test_process_startup_factory_reaches_native_tasks_and_durable_creator(process_peer):
    p = process_peer
    assert p.router.has_native_session_factory, (
        "actual process onboarding did not bind its native session producer"
    )
    async with p.router.open_a2a_session(p.requester, p.peer) as session:
        dispatch, wire = await session.reserve(
            SendMessageRequest(
                message=new_text_message("Native process peer delivery", role=Role.ROLE_USER),
                configuration={"return_immediately": True},
            ),
            dispatch_tool="send_a2a_message", verb="message",
        )
        receipt = await session.send(dispatch, wire)
        observed = await session.get_task(receipt.dispatch)
    snapshot = await p.recipient_manager.get_task_snapshot(
        receipt.response.id, recipient_agent_id=p.recipient.did
    )
    assert observed.response.id == receipt.response.id
    assert snapshot.ownership.creator_agent_id == PRINCIPAL
    assert await p.sender_manager.outbound_store.get(dispatch.dispatch_id) == observed.dispatch
    native = [(path, headers) for path, headers in p.state.requests
              if native_route_prefix(p.peer.routing_key) in path]
    assert native and any(path.endswith("/rpc") for path, _ in native)
    assert all(A2A_TRANSPORT_KEY_HEADER.lower().encode() not in headers for _, headers in native)


async def test_process_session_rechecks_live_roster_after_discovery(process_peer):
    p = process_peer
    async with p.router.open_a2a_session(p.requester, p.peer) as session:
        p.state.allowed = False
        before = len(p.state.requests)
        with pytest.raises(PeerAccessDeniedError, match="current directory"):
            await session.reserve(
                SendMessageRequest(message=new_text_message("revoked", role=Role.ROLE_USER)),
                dispatch_tool="send_a2a_message", verb="message",
            )
        assert not any(path.endswith("/rpc") for path, _ in p.state.requests[before:])
    with pytest.raises(PeerNotFoundError):
        async with p.router.open_a2a_session(p.requester, p.peer):
            pytest.fail("removed peer admitted")


async def test_process_session_cannot_spend_a_foreign_requester_or_resolver(process_peer):
    p = process_peer
    for resolver, requester in [
        (ProcessA2ADidResolver(()), p.requester),
        (p.resolver, PeerRequester(PRINCIPAL, object())),
    ]:
        before = len(p.state.requests)
        with pytest.raises(PeerAccessDeniedError, match="trusted peer binding"):
            async with open_process_native_peer_session(
                p.sender_owner, resolver=resolver, requester=requester,
                peer=p.peer, timeout=httpx.Timeout(5),
            ):
                pytest.fail("foreign authority admitted")
        assert len(p.state.requests) == before


async def test_process_session_requires_a_launcher_attested_peer_identity(process_peer):
    from dataclasses import replace

    p = process_peer
    before = len(p.state.requests)
    with pytest.raises(PeerAccessDeniedError, match="launcher-attested signing identity"):
        async with open_process_native_peer_session(
            p.sender_owner, resolver=p.resolver, requester=p.requester,
            peer=replace(p.peer, agent_id="did:unattested"), timeout=httpx.Timeout(5),
        ):
            pytest.fail("unattested peer admitted")
    assert len(p.state.requests) == before


async def test_signed_peer_card_cannot_redirect_the_trusted_process_route(process_peer):
    p = process_peer
    publication = p.recipient._native_a2a_publication
    publication.configuration = NativeHostingConfiguration("https://unexpected.test")
    with pytest.raises(PeerUnavailableError, match="trusted host route"):
        async with p.router.open_a2a_session(p.requester, p.peer):
            pytest.fail("signed card redirected egress")
    assert not any(path.endswith("/rpc") for path, _ in p.state.requests)
