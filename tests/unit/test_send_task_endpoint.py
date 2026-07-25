"""``POST /api/agent/tasks/send`` — send-side artifact ingress (#1525).

A sender may attach durable handoff payload (planning docs, evidence
bundles, saved-memory/recall references, logs, diffs) to an outgoing
A2A task at creation time. The receiving endpoint validates the
``artifacts`` list and persists them on the task at SUBMITTED so the
recipient can retrieve them from the task store before producing any
response. This is the send-side mirror of the responder-side
``attach_artifact_to_a2a_task`` flow.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

import kestrel_sovereign.endpoints.agent as agent_endpoint
from kestrel_sovereign.a2a.types import (
    Task,
    TaskState,
    TaskStatus,
)


@pytest.fixture
def app_with_send(monkeypatch):
    limiter = Limiter(key_func=get_remote_address)
    monkeypatch.setattr(agent_endpoint, "limiter", limiter)

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(agent_endpoint.router)
    return app


def _stub_agent():
    """Agent whose ``create_task`` echoes a real Task built from the
    params + artifacts it was handed, so the test can assert both the
    call arguments and the serialized response envelope."""
    agent = MagicMock()
    agent.did = "did:test:recipient"
    agent.agent_id = agent.did
    agent._agent_name = "recipient"
    agent.a2a_did_resolver = None
    agent.a2a_inbound_sender_authorizer = None
    agent.peer_directory_router = None
    agent.peer_requester = None
    agent._a2a_inbound_scoped_policy_required = False
    agent._a2a_host_manager = None
    agent.task_manager = MagicMock()

    async def fake_create_task(*, params, agent_name, artifacts=None):
        return Task(
            id=params.id,
            sessionId=params.sessionId,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[params.message],
            metadata=params.metadata or {},
            artifacts=list(artifacts) if artifacts else None,
        )

    agent.task_manager.create_task = AsyncMock(side_effect=fake_create_task)
    return agent


def _attach(app: FastAPI, agent) -> None:
    @app.middleware("http")
    async def _attach_agent(request, call_next):
        request.state.agent = agent
        return await call_next(request)


def _body(**overrides):
    body = {
        "id": "task-1",
        "sessionId": "sess-1",
        "message": {"role": "user", "parts": [{"type": "text", "text": "do it"}]},
        "metadata": {"sender": "emma"},
    }
    body.update(overrides)
    return body


def test_send_task_persists_sender_artifacts(app_with_send):
    agent = _stub_agent()
    _attach(app_with_send, agent)

    body = _body(
        artifacts=[
            {
                "name": "plan",
                "parts": [{"type": "text", "text": "step one"}],
                "index": 0,
                "lastChunk": True,
                "metadata": {"origin": "saved_item"},
            },
            {
                "name": "references",
                "parts": [{"type": "data", "data": {"ref_type": "memory", "id": "m1"}}],
                "index": 0,
                "metadata": {"kind": "reference"},
            },
        ]
    )

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 200
    # create_task received parsed Artifact objects in order.
    call = agent.task_manager.create_task.await_args
    artifacts = call.kwargs["artifacts"]
    assert artifacts is not None and len(artifacts) == 2
    assert artifacts[0].name == "plan"
    assert artifacts[0].metadata == {"origin": "saved_item"}
    assert artifacts[0].parts[0].text == "step one"
    assert artifacts[1].name == "references"
    assert artifacts[1].parts[0].data == {"ref_type": "memory", "id": "m1"}

    # Recipient retrieval: the serialized task envelope carries them in
    # order with structured metadata intact.
    payload = resp.json()
    assert [a["name"] for a in payload["artifacts"]] == ["plan", "references"]
    assert payload["artifacts"][0]["lastChunk"] is True
    assert payload["artifacts"][1]["parts"][0]["data"] == {
        "ref_type": "memory",
        "id": "m1",
    }


def test_send_task_without_artifacts_passes_none(app_with_send):
    agent = _stub_agent()
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=_body())

    assert resp.status_code == 200
    assert agent.task_manager.create_task.await_args.kwargs["artifacts"] is None


def test_send_task_rejects_non_list_artifacts(app_with_send):
    agent = _stub_agent()
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=_body(artifacts="nope"))

    assert resp.status_code == 400
    agent.task_manager.create_task.assert_not_awaited()


def test_send_task_rejects_malformed_artifact(app_with_send):
    agent = _stub_agent()
    _attach(app_with_send, agent)

    # parts is required on an Artifact — a missing body must 400, not 500.
    with TestClient(app_with_send) as client:
        resp = client.post(
            "/api/agent/tasks/send", json=_body(artifacts=[{"name": "bad"}])
        )

    assert resp.status_code == 400
    agent.task_manager.create_task.assert_not_awaited()


# --------------------------------------------------------------------------
# Cryptographic sender verification (#1673)
# --------------------------------------------------------------------------


_SENDER_DID = "did:web:example.com:agent:emma"


class _InboundAuthorizer:
    def __init__(
        self,
        *,
        allowed: bool,
        requires_verified_sender: bool = True,
        scope_validator=None,
    ):
        self.allowed = allowed
        self.requires_verified_sender = requires_verified_sender
        self._scope_validator = scope_validator or (lambda: True)
        self.calls: list[str] = []

    def has_valid_current_scope(self) -> bool:
        return self._scope_validator() is True

    async def authorize(self, sender_did: str) -> bool:
        self.calls.append(sender_did)
        return self.allowed


class _MutatingInboundAuthorizer(_InboundAuthorizer):
    def __init__(self, mutation, *, scope_validator=None):
        super().__init__(allowed=True, scope_validator=scope_validator)
        self._mutation = mutation

    async def authorize(self, sender_did: str) -> bool:
        self.calls.append(sender_did)
        await asyncio.sleep(0)
        self._mutation()
        return True


def _signer_and_doc():
    from datetime import datetime, timezone
    from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
    from kestrel_sovereign.identity.did_web import build_verification_methods
    from kestrel_sovereign.a2a.envelope_signing import (
        bound_envelope_fields, sign_envelope, canonical_message,
    )

    kp = generate_hybrid_keypair()
    doc = {"id": _SENDER_DID, "verificationMethod": build_verification_methods(_SENDER_DID, kp.public_keys())}

    def sign(part_texts, *, task_id="task-1", session_id="sess-1", metadata=None, artifacts=None):
        # Mirror exactly what the endpoint signs/verifies: the canonical
        # (structure-preserving) message form + authoritative sessionId + the
        # behaviour-steering bound fields derived from the same metadata (#1721).
        return sign_envelope(
            kp, sender=_SENDER_DID, task_id=task_id,
            message=canonical_message(part_texts), session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            bound=bound_envelope_fields(metadata, artifacts=artifacts),
        )

    return sign, doc


def _install_hosted_a2a_manager(agent, document, authorize_inbound_sender):
    from kestrel_sovereign.a2a.did_registry import install_a2a_did_resolver
    from kestrel_sovereign.a2a.inbound_authorization import (
        install_a2a_inbound_sender_authorizer,
        mark_a2a_inbound_scoped_policy,
    )
    from kestrel_sovereign.features.peers.directory import PeerRequester
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    sender = SimpleNamespace(
        agent_id="did:pkh:hosted:sender",
        did="did:pkh:hosted:sender",
        identity=SimpleNamespace(
            is_hybrid=True,
            signing_did=_SENDER_DID,
            new_verification_methods=list(document["verificationMethod"]),
        ),
    )
    manager = AgentManager()
    manager._register_agent("sender", sender)
    manager._register_agent("recipient", agent)
    agent.peer_directory_router = SimpleNamespace(
        authorize_inbound_sender=authorize_inbound_sender,
    )
    agent.peer_requester = PeerRequester(agent.agent_id, object())
    mark_a2a_inbound_scoped_policy(agent, required=True)
    install_a2a_did_resolver(manager, recipient=agent)
    install_a2a_inbound_sender_authorizer(manager, recipient=agent)
    agent._a2a_host_manager = manager
    manager.install_a2a_hosted_policy(
        agent,
        resolver=agent.a2a_did_resolver,
        authorizer=agent.a2a_inbound_sender_authorizer,
        router=agent.peer_directory_router,
        requester=agent.peer_requester,
    )
    return manager, sender


def _install_hosted_legacy_unsigned_manager(
    agent,
    authorize_inbound_sender,
    *,
    sender_name="legacy",
    sender_display_name=None,
    sender_identity=None,
):
    """Build the production hosted policy shape for unsigned compatibility."""
    from kestrel_sovereign.a2a.did_registry import install_a2a_did_resolver
    from kestrel_sovereign.a2a.inbound_authorization import (
        install_a2a_inbound_sender_authorizer,
        mark_a2a_inbound_scoped_policy,
    )
    from kestrel_sovereign.features.peers.directory import PeerRequester
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    sender = SimpleNamespace(
        agent_id=f"did:pkh:hosted:{sender_name}",
        did=f"did:pkh:hosted:{sender_name}",
        _agent_name=sender_display_name or sender_name,
        identity=sender_identity,
    )
    manager = AgentManager()
    manager._register_agent(sender_name, sender)
    manager._register_agent("recipient", agent)
    agent.peer_directory_router = SimpleNamespace(
        authorize_inbound_sender=authorize_inbound_sender,
    )
    agent.peer_requester = PeerRequester(agent.agent_id, object())
    mark_a2a_inbound_scoped_policy(agent, required=True)
    install_a2a_did_resolver(manager, recipient=agent)
    install_a2a_inbound_sender_authorizer(manager, recipient=agent)
    agent._a2a_host_manager = manager
    manager.install_a2a_hosted_policy(
        agent,
        resolver=agent.a2a_did_resolver,
        authorizer=agent.a2a_inbound_sender_authorizer,
        router=agent.peer_directory_router,
        requester=agent.peer_requester,
    )
    return manager, sender


def test_unsigned_envelope_accepted_and_marked_unverified(app_with_send):
    agent = _stub_agent()
    agent.a2a_did_resolver = None  # explicit: no resolver wired
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=_body())

    assert resp.status_code == 200
    # Verdict recorded for downstream governance.
    assert agent.task_manager.create_task.await_args.kwargs["params"].metadata["sender_verified"] is False


def test_installed_standalone_authorizer_keeps_unsigned_compatibility(
    app_with_send,
):
    authorizer = _InboundAuthorizer(
        allowed=True,
        requires_verified_sender=False,
    )
    agent = _stub_agent()
    agent.a2a_inbound_sender_authorizer = authorizer
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=_body())

    assert resp.status_code == 200
    assert authorizer.calls == []
    assert (
        agent.task_manager.create_task.await_args.kwargs["params"]
        .metadata["sender_verified"]
        is False
    )


def test_valid_signed_envelope_verifies(app_with_send):
    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc if did == _SENDER_DID else None
    _attach(app_with_send, agent)

    body = _body(metadata={"sender": _SENDER_DID, "signature": sign(["do it"])})
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 200
    assert agent.task_manager.create_task.await_args.kwargs["params"].metadata["sender_verified"] is True


def test_scoped_valid_signed_sender_is_authorized_after_verification(
    app_with_send,
):
    sign, doc = _signer_and_doc()
    authorizer = _InboundAuthorizer(allowed=True)
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc if did == _SENDER_DID else None
    agent.a2a_inbound_sender_authorizer = authorizer
    _attach(app_with_send, agent)

    body = _body(metadata={"sender": _SENDER_DID, "signature": sign(["do it"])})
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 200
    assert authorizer.calls == [_SENDER_DID]
    assert (
        agent.task_manager.create_task.await_args.kwargs["params"]
        .metadata["sender_verified"]
        is True
    )


def test_scoped_other_user_valid_signature_is_rejected(app_with_send):
    sign, doc = _signer_and_doc()
    authorizer = _InboundAuthorizer(allowed=False)
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc if did == _SENDER_DID else None
    agent.a2a_inbound_sender_authorizer = authorizer
    _attach(app_with_send, agent)

    body = _body(metadata={"sender": _SENDER_DID, "signature": sign(["do it"])})
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 403
    assert resp.json()["detail"] == "A2A sender authorization failed"
    assert authorizer.calls == [_SENDER_DID]
    agent.task_manager.create_task.assert_not_awaited()


def test_scoped_unsigned_envelope_rejected_even_without_global_flag(
    app_with_send,
    monkeypatch,
):
    monkeypatch.delenv("KESTREL_A2A_REQUIRE_SIGNED", raising=False)
    authorizer = _InboundAuthorizer(allowed=True)
    agent = _stub_agent()
    agent.a2a_inbound_sender_authorizer = authorizer
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=_body())

    assert resp.status_code == 403
    assert authorizer.calls == []
    agent.task_manager.create_task.assert_not_awaited()


def test_hosted_exact_non_hybrid_local_sender_keeps_unsigned_compatibility(
    app_with_send,
    monkeypatch,
):
    """Hosted policy permits only its current local pre-ceremony peer."""
    monkeypatch.delenv("KESTREL_A2A_REQUIRE_SIGNED", raising=False)
    agent = _stub_agent()
    directory = AsyncMock(return_value=True)
    _manager, sender = _install_hosted_legacy_unsigned_manager(agent, directory)
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        response = client.post(
            "/api/agent/tasks/send",
            json=_body(metadata={"sender": "legacy"}),
        )

    assert response.status_code == 200
    assert (
        agent.task_manager.create_task.await_args.kwargs["params"]
        .metadata["sender_verified"]
        is False
    )
    directory.assert_awaited_once_with(agent.peer_requester, sender.agent_id)


def test_hosted_unsigned_sender_uses_published_display_name_not_routing_key(
    app_with_send,
    monkeypatch,
):
    """Peers publish the live display name even when routing differs."""

    monkeypatch.delenv("KESTREL_A2A_REQUIRE_SIGNED", raising=False)
    agent = _stub_agent()
    directory = AsyncMock(return_value=True)
    _manager, sender = _install_hosted_legacy_unsigned_manager(
        agent,
        directory,
        sender_name="routing-alice",
        sender_display_name="Alice",
    )
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        published = client.post(
            "/api/agent/tasks/send", json=_body(metadata={"sender": "Alice"})
        )
        routing_key = client.post(
            "/api/agent/tasks/send",
            json=_body(id="routing-key", metadata={"sender": "routing-alice"}),
        )

    assert published.status_code == 200
    assert routing_key.status_code == 403
    directory.assert_awaited_once_with(agent.peer_requester, sender.agent_id)


def test_hosted_unsigned_ambiguous_display_name_is_rejected(
    app_with_send,
    monkeypatch,
):
    """Two live agents with one published name fail closed before directory IO."""

    monkeypatch.delenv("KESTREL_A2A_REQUIRE_SIGNED", raising=False)
    agent = _stub_agent()
    directory = AsyncMock(return_value=True)
    manager, _sender = _install_hosted_legacy_unsigned_manager(
        agent,
        directory,
        sender_name="routing-alice",
        sender_display_name="Alice",
    )
    manager._register_agent(
        "routing-alice-duplicate",
        SimpleNamespace(
            agent_id="did:pkh:hosted:routing-alice-duplicate",
            did="did:pkh:hosted:routing-alice-duplicate",
            _agent_name="Alice",
            identity=None,
        ),
    )
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        response = client.post(
            "/api/agent/tasks/send", json=_body(metadata={"sender": "Alice"})
        )

    assert response.status_code == 403
    directory.assert_not_awaited()
    agent.task_manager.create_task.assert_not_awaited()


def test_hosted_unsigned_unknown_or_cross_scope_sender_is_rejected(
    app_with_send,
    monkeypatch,
):
    """No external/global unsigned fallback survives user-scoped hosting."""
    monkeypatch.delenv("KESTREL_A2A_REQUIRE_SIGNED", raising=False)
    agent = _stub_agent()
    directory = AsyncMock(return_value=False)
    _manager, sender = _install_hosted_legacy_unsigned_manager(agent, directory)
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        unknown = client.post(
            "/api/agent/tasks/send", json=_body(metadata={"sender": "external"})
        )
        denied = client.post(
            "/api/agent/tasks/send", json=_body(id="cross-scope", metadata={"sender": "legacy"})
        )

    assert unknown.status_code == 403
    assert denied.status_code == 403
    directory.assert_awaited_once_with(agent.peer_requester, sender.agent_id)
    agent.task_manager.create_task.assert_not_awaited()


def test_hosted_hybrid_sender_cannot_downgrade_to_unsigned(
    app_with_send,
    monkeypatch,
):
    """A loaded signing-capable peer must supply a verified hybrid envelope."""
    monkeypatch.delenv("KESTREL_A2A_REQUIRE_SIGNED", raising=False)
    agent = _stub_agent()
    directory = AsyncMock(return_value=True)
    _install_hosted_legacy_unsigned_manager(
        agent,
        directory,
        sender_identity=SimpleNamespace(
            is_hybrid=True,
            hybrid_keypair=object(),
            new_verification_methods=[object()],
        ),
    )
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        response = client.post(
            "/api/agent/tasks/send", json=_body(metadata={"sender": "legacy"})
        )

    assert response.status_code == 403
    directory.assert_not_awaited()
    agent.task_manager.create_task.assert_not_awaited()


def test_scoped_sender_missing_explicit_authorizer_fails_closed(app_with_send):
    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc if did == _SENDER_DID else None
    agent.peer_directory_router = object()
    agent.peer_requester = object()
    _attach(app_with_send, agent)

    body = _body(metadata={"sender": _SENDER_DID, "signature": sign(["do it"])})
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 403
    assert resp.json()["detail"] == "A2A sender authorization unavailable"
    agent.task_manager.create_task.assert_not_awaited()


def test_monotonic_hosted_marker_survives_removal_of_every_live_scope_seam(
    app_with_send,
):
    from kestrel_sovereign.a2a.inbound_authorization import (
        mark_a2a_inbound_scoped_policy,
    )

    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    agent.peer_directory_router = object()
    agent.peer_requester = object()
    agent.a2a_inbound_sender_authorizer = _InboundAuthorizer(allowed=True)
    mark_a2a_inbound_scoped_policy(agent, required=True)

    # Revoke every removable live seam. The registration-owned marker remains.
    agent.peer_directory_router = None
    agent.peer_requester = None
    agent.a2a_inbound_sender_authorizer = None
    agent.a2a_did_resolver = lambda did: doc if did == _SENDER_DID else None
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        unsigned = client.post("/api/agent/tasks/send", json=_body())
        signed = client.post(
            "/api/agent/tasks/send",
            json=_body(
                id="task-signed-after-revocation",
                metadata={
                    "sender": _SENDER_DID,
                    "signature": sign(
                        ["do it"],
                        task_id="task-signed-after-revocation",
                    ),
                },
            ),
        )

    assert unsigned.status_code == 403
    assert "unsigned envelope rejected" in unsigned.json()["detail"]
    assert signed.status_code == 403
    assert signed.json()["detail"] == "A2A sender authorization unavailable"
    agent.task_manager.create_task.assert_not_awaited()


@pytest.mark.parametrize(
    "mutation_kind",
    ["remove_authorizer", "replace_router", "replace_resolver"],
)
def test_scope_change_during_async_did_resolution_fails_closed(
    app_with_send,
    mutation_kind,
):
    from kestrel_sovereign.a2a.inbound_authorization import (
        mark_a2a_inbound_scoped_policy,
    )

    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    authorizer = _InboundAuthorizer(allowed=True)
    agent.peer_directory_router = object()
    agent.peer_requester = object()
    agent.a2a_inbound_sender_authorizer = authorizer
    mark_a2a_inbound_scoped_policy(agent, required=True)

    async def mutating_resolver(did):
        await asyncio.sleep(0)
        if mutation_kind == "remove_authorizer":
            agent.a2a_inbound_sender_authorizer = None
        elif mutation_kind == "replace_router":
            agent.peer_directory_router = object()
        else:
            agent.a2a_did_resolver = lambda _did: None
        return doc if did == _SENDER_DID else None

    agent.a2a_did_resolver = mutating_resolver
    _attach(app_with_send, agent)
    body = _body(metadata={"sender": _SENDER_DID, "signature": sign(["do it"])})

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 403
    assert (
        resp.json()["detail"]
        == "A2A sender authorization context changed during verification"
    )
    assert authorizer.calls == []
    agent.task_manager.create_task.assert_not_awaited()


@pytest.mark.parametrize(
    "mutation_kind",
    ["remove_requester", "replace_authorizer", "remove_router_method"],
)
def test_scope_change_during_awaited_authorization_fails_closed(
    app_with_send,
    mutation_kind,
):
    from kestrel_sovereign.a2a.inbound_authorization import (
        mark_a2a_inbound_scoped_policy,
    )

    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    agent.peer_directory_router = MagicMock()
    agent.peer_directory_router.authorize_inbound_sender = AsyncMock(
        return_value=True,
    )
    agent.peer_requester = object()

    def mutate_scope():
        if mutation_kind == "remove_requester":
            agent.peer_requester = None
        elif mutation_kind == "replace_authorizer":
            agent.a2a_inbound_sender_authorizer = _InboundAuthorizer(
                allowed=True,
            )
        else:
            agent.peer_directory_router.authorize_inbound_sender = None

    authorizer = _MutatingInboundAuthorizer(
        mutate_scope,
        scope_validator=lambda: (
            agent.peer_requester is not None
            and callable(
                getattr(
                    agent.peer_directory_router,
                    "authorize_inbound_sender",
                    None,
                )
            )
        ),
    )
    agent.a2a_inbound_sender_authorizer = authorizer
    agent.a2a_did_resolver = lambda did: doc if did == _SENDER_DID else None
    mark_a2a_inbound_scoped_policy(agent, required=True)
    _attach(app_with_send, agent)
    body = _body(metadata={"sender": _SENDER_DID, "signature": sign(["do it"])})

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 403
    if mutation_kind == "remove_router_method":
        assert (
            resp.json()["detail"]
            == "A2A sender authorization context is invalid"
        )
    else:
        assert (
            resp.json()["detail"]
            == "A2A sender authorization context changed during authorization"
        )
    assert authorizer.calls == [_SENDER_DID]
    agent.task_manager.create_task.assert_not_awaited()


@pytest.mark.parametrize("mutation_kind", ["replace_sender", "rotate_sender_key"])
def test_live_sender_identity_change_during_authorization_fails_closed(
    app_with_send,
    mutation_kind,
):
    from kestrel_sovereign.identity.did_web import build_verification_methods
    from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair

    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    manager_box = {}
    sender_box = {}

    async def mutating_authorization(_requester, _sender_id):
        await asyncio.sleep(0)
        sender = sender_box["sender"]
        if mutation_kind == "replace_sender":
            replacement = SimpleNamespace(
                agent_id=sender.agent_id,
                did=sender.did,
                identity=SimpleNamespace(
                    is_hybrid=True,
                    signing_did=_SENDER_DID,
                    new_verification_methods=list(
                        sender.identity.new_verification_methods
                    ),
                ),
            )
            manager_box["manager"]._agents["sender"] = replacement
        else:
            rotated = generate_hybrid_keypair()
            sender.identity.new_verification_methods = (
                build_verification_methods(
                    _SENDER_DID,
                    rotated.public_keys(),
                )
            )
        return True

    manager, sender = _install_hosted_a2a_manager(
        agent,
        doc,
        mutating_authorization,
    )
    manager_box["manager"] = manager
    sender_box["sender"] = sender
    _attach(app_with_send, agent)
    body = _body(metadata={"sender": _SENDER_DID, "signature": sign(["do it"])})

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 403
    assert (
        resp.json()["detail"]
        == "A2A sender identity changed during authorization"
    )
    agent.task_manager.create_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_sender_removal_waits_for_atomic_a2a_task_commit():
    from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart

    sign, doc = _signer_and_doc()
    authorization_started = asyncio.Event()
    allow_authorization = asyncio.Event()

    async def blocked_authorization(_requester, _sender_id):
        authorization_started.set()
        await allow_authorization.wait()
        return True

    agent = _stub_agent()
    manager, sender = _install_hosted_a2a_manager(
        agent,
        doc,
        blocked_authorization,
    )
    sender.shutdown = AsyncMock()
    metadata = {"sender": _SENDER_DID}
    metadata["signature"] = sign(["do it"])
    params = TaskSendParams(
        id="task-1",
        sessionId="sess-1",
        message=Message(role="user", parts=[TextPart(text="do it")]),
        metadata=metadata,
    )

    request_task = asyncio.create_task(
        agent_endpoint._create_verified_a2a_task(
            agent,
            params,
            params.message.parts,
            [],
            [],
        )
    )
    await asyncio.wait_for(authorization_started.wait(), timeout=1)
    removal = asyncio.create_task(manager.remove_agent("sender"))
    await asyncio.sleep(0)

    assert removal.done() is False
    assert manager.get_agent("sender") is sender

    allow_authorization.set()
    created = await asyncio.wait_for(request_task, timeout=1)
    assert created.id == "task-1"
    assert await asyncio.wait_for(removal, timeout=1) is True
    assert manager.get_agent("sender") is None


@pytest.mark.asyncio
async def test_recipient_removal_waits_for_atomic_a2a_task_commit():
    from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart

    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    manager, _sender = _install_hosted_a2a_manager(
        agent,
        doc,
        AsyncMock(return_value=True),
    )
    agent.shutdown = AsyncMock()
    create_started = asyncio.Event()
    allow_create = asyncio.Event()
    original_create = agent.task_manager.create_task.side_effect

    async def blocked_create(**kwargs):
        create_started.set()
        await allow_create.wait()
        return await original_create(**kwargs)

    agent.task_manager.create_task.side_effect = blocked_create
    metadata = {"sender": _SENDER_DID}
    metadata["signature"] = sign(["do it"])
    params = TaskSendParams(
        id="task-1",
        sessionId="sess-1",
        message=Message(role="user", parts=[TextPart(text="do it")]),
        metadata=metadata,
    )

    request_task = asyncio.create_task(
        agent_endpoint._create_verified_a2a_task(
            agent,
            params,
            params.message.parts,
            [],
            [],
        )
    )
    await asyncio.wait_for(create_started.wait(), timeout=1)
    removal = asyncio.create_task(manager.remove_agent("recipient"))
    await asyncio.sleep(0)

    assert removal.done() is False
    assert manager.get_agent("recipient") is agent

    allow_create.set()
    created = await asyncio.wait_for(request_task, timeout=1)
    assert created.id == "task-1"
    assert await asyncio.wait_for(removal, timeout=1) is True
    assert manager.get_agent("recipient") is None


@pytest.mark.asyncio
async def test_removal_winning_lease_rejects_queued_stale_recipient():
    from fastapi import HTTPException
    from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart

    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    manager, _sender = _install_hosted_a2a_manager(
        agent,
        doc,
        AsyncMock(return_value=True),
    )
    agent.shutdown = AsyncMock()
    metadata = {"sender": _SENDER_DID}
    metadata["signature"] = sign(["do it"])
    params = TaskSendParams(
        id="task-1",
        sessionId="sess-1",
        message=Message(role="user", parts=[TextPart(text="do it")]),
        metadata=metadata,
    )
    lease = manager.a2a_lifecycle_lease()
    await lease.acquire()
    removal = asyncio.create_task(manager.remove_agent("recipient"))
    while not getattr(lease, "_waiters", None):
        await asyncio.sleep(0)
    request_task = asyncio.create_task(
        agent_endpoint._create_verified_a2a_task(
            agent,
            params,
            params.message.parts,
            [],
            [],
        )
    )
    while len(getattr(lease, "_waiters", ())) < 2:
        await asyncio.sleep(0)
    lease.release()

    assert await asyncio.wait_for(removal, timeout=1) is True
    with pytest.raises(HTTPException) as rejected:
        await asyncio.wait_for(request_task, timeout=1)
    assert rejected.value.status_code == 403
    assert "no longer published" in str(rejected.value.detail)
    agent.task_manager.create_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_manager_policy_is_atomic_during_task_persistence():
    from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart

    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    manager, _sender = _install_hosted_a2a_manager(
        agent,
        doc,
        AsyncMock(return_value=True),
    )
    create_started = asyncio.Event()
    allow_create = asyncio.Event()
    original_create = agent.task_manager.create_task.side_effect

    async def blocked_create(**kwargs):
        create_started.set()
        await allow_create.wait()
        return await original_create(**kwargs)

    agent.task_manager.create_task.side_effect = blocked_create
    original_policy = manager.a2a_hosted_policy_for(agent)
    metadata = {"sender": _SENDER_DID}
    metadata["signature"] = sign(["do it"])
    params = TaskSendParams(
        id="task-1",
        sessionId="sess-1",
        message=Message(role="user", parts=[TextPart(text="do it")]),
        metadata=metadata,
    )
    request_task = asyncio.create_task(
        agent_endpoint._create_verified_a2a_task(
            agent,
            params,
            params.message.parts,
            [],
            [],
        )
    )
    await asyncio.wait_for(create_started.wait(), timeout=1)

    # Raw compatibility attributes are no longer policy mutation authority.
    agent.peer_directory_router = None
    agent.peer_requester = None
    agent.a2a_inbound_sender_authorizer = None
    replacement = asyncio.create_task(
        manager.replace_a2a_hosted_policy(
            agent,
            resolver=original_policy.resolver,
            authorizer=original_policy.authorizer,
            router=original_policy.router,
            requester=original_policy.requester,
        )
    )
    await asyncio.sleep(0)
    assert replacement.done() is False

    allow_create.set()
    created = await asyncio.wait_for(request_task, timeout=1)
    replacement_policy = await asyncio.wait_for(replacement, timeout=1)

    assert created.id == "task-1"
    assert replacement_policy.generation > original_policy.generation
    assert manager.a2a_hosted_policy_for(agent) is replacement_policy


def test_valid_signed_envelope_with_artifacts_verifies(app_with_send):
    """#1721 regression: a signed envelope that carries top-level artifacts must
    verify — the receiver binds the RAW wire artifacts (not a non-existent
    ``params.artifacts``), matching what the signer bound."""
    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc if did == _SENDER_DID else None
    _attach(app_with_send, agent)

    artifacts = [{"name": "plan", "parts": [{"type": "text", "text": "step one"}], "index": 0}]
    body = _body(
        artifacts=artifacts,
        metadata={"sender": _SENDER_DID, "signature": sign(["do it"], artifacts=artifacts)},
    )
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 200
    assert agent.task_manager.create_task.await_args.kwargs["params"].metadata["sender_verified"] is True


def test_signed_envelope_with_tampered_artifacts_rejected_403(app_with_send):
    """Altering an artifact after signing fails verification (#1721)."""
    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc
    _attach(app_with_send, agent)

    signed_artifacts = [{"name": "plan", "parts": [{"type": "text", "text": "step one"}], "index": 0}]
    tampered = [{"name": "plan", "parts": [{"type": "text", "text": "INJECTED"}], "index": 0}]
    body = _body(
        artifacts=tampered,
        metadata={"sender": _SENDER_DID, "signature": sign(["do it"], artifacts=signed_artifacts)},
    )
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 403


def test_tampered_signed_envelope_rejected_403(app_with_send):
    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc
    _attach(app_with_send, agent)

    # Signature is over "something else" but the message body says "do it".
    body = _body(metadata={"sender": _SENDER_DID, "signature": sign(["something else"])})
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 403
    agent.task_manager.create_task.assert_not_awaited()


def test_failed_crypto_never_calls_scoped_authorizer(app_with_send):
    sign, doc = _signer_and_doc()
    authorizer = _InboundAuthorizer(allowed=True)
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc
    agent.a2a_inbound_sender_authorizer = authorizer
    _attach(app_with_send, agent)

    body = _body(
        metadata={
            "sender": _SENDER_DID,
            "signature": sign(["different signed content"]),
        }
    )
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 403
    assert authorizer.calls == []
    agent.task_manager.create_task.assert_not_awaited()


def test_session_swap_rejected_403(app_with_send):
    """A signature bound to one session can't be replayed under another (#1673
    codex fix): sign for sessionId 's2' but submit the task as 'sess-1'."""
    sign, doc = _signer_and_doc()
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc
    _attach(app_with_send, agent)

    body = _body(metadata={"sender": _SENDER_DID, "signature": sign(["do it"], session_id="s2")})
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)  # body sessionId is 'sess-1'

    assert resp.status_code == 403
    agent.task_manager.create_task.assert_not_awaited()


def test_empty_signature_block_rejected_not_downgraded(app_with_send):
    """A present-but-empty signature block ({}) must be rejected, not treated
    as 'unsigned' (#1673 codex fix)."""
    _, doc = _signer_and_doc()
    agent = _stub_agent()
    agent.a2a_did_resolver = lambda did: doc
    _attach(app_with_send, agent)

    body = _body(metadata={"sender": _SENDER_DID, "signature": {}})
    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=body)

    assert resp.status_code == 403
    agent.task_manager.create_task.assert_not_awaited()


def test_require_signed_rejects_unsigned_403(app_with_send, monkeypatch):
    monkeypatch.setenv("KESTREL_A2A_REQUIRE_SIGNED", "1")
    agent = _stub_agent()
    agent.a2a_did_resolver = None
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=_body())

    assert resp.status_code == 403
    agent.task_manager.create_task.assert_not_awaited()
