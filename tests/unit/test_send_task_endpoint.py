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
    agent._agent_name = "recipient"
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


def test_unsigned_envelope_accepted_and_marked_unverified(app_with_send):
    agent = _stub_agent()
    agent.a2a_did_resolver = None  # explicit: no resolver wired
    _attach(app_with_send, agent)

    with TestClient(app_with_send) as client:
        resp = client.post("/api/agent/tasks/send", json=_body())

    assert resp.status_code == 200
    # Verdict recorded for downstream governance.
    assert agent.task_manager.create_task.await_args.kwargs["params"].metadata["sender_verified"] is False


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
