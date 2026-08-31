"""A2A verification-document lookup and signed-envelope verification."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from kestrel_sovereign.a2a.did_registry import (
    A2A_PEER_IDENTITY_ROOTS_ENV,
    HostA2ADidResolver,
    ProcessA2ADidResolver,
    install_a2a_did_resolver,
)
from kestrel_sovereign.a2a.envelope_signing import (
    bound_envelope_fields,
    sign_envelope,
    verify_inbound_envelope,
)
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair


def _hybrid_agent(signing_did: str, *, agent_id: str | None = None):
    """Return a stand-in agent and the keypair used by its hybrid identity."""
    keypair = generate_hybrid_keypair()
    identity = SimpleNamespace(
        is_hybrid=True,
        signing_did=signing_did,
        new_verification_methods=build_verification_methods(
            signing_did,
            keypair.public_keys(),
        ),
    )
    stable_id = agent_id or signing_did
    agent = SimpleNamespace(
        agent_id=stable_id,
        did=stable_id,
        identity=identity,
        a2a_did_resolver=None,
    )
    return agent, keypair


def _legacy_agent():
    """A pre-ceremony identity has no hybrid verification document."""
    return SimpleNamespace(
        agent_id="did:pkh:legacy",
        did="did:pkh:legacy",
        identity=SimpleNamespace(
            is_hybrid=False,
            signing_did="did:pkh:legacy",
            new_verification_methods=None,
        ),
        a2a_did_resolver=None,
    )


class _Manager:
    def __init__(self, agents: dict):
        self._agents = agents

    def list_agents(self):
        return dict(self._agents)


DID_A = "did:web:example.com:agent:emma"
DID_B = "did:web:example.com:agent:claw"


def test_resolves_known_hybrid_agent_to_doc_with_vms():
    agent_a, _ = _hybrid_agent(DID_A)
    resolver = HostA2ADidResolver(_Manager({"emma": agent_a}))

    document = resolver.resolve(DID_A)

    assert document is not None
    assert document["id"] == DID_A
    assert document["verificationMethod"]


def test_unknown_did_resolves_none():
    agent_a, _ = _hybrid_agent(DID_A)
    resolver = HostA2ADidResolver(_Manager({"emma": agent_a}))

    assert resolver.resolve("did:web:example.com:agent:nobody") is None
    assert resolver.resolve("") is None


def test_legacy_non_hybrid_agent_not_resolvable():
    resolver = HostA2ADidResolver(_Manager({"old": _legacy_agent()}))

    assert resolver.resolve("did:pkh:legacy") is None


def test_install_injects_distinct_verification_resolver_per_agent():
    agent_a, _ = _hybrid_agent(DID_A)
    agent_b, _ = _hybrid_agent(DID_B)
    manager = _Manager({"emma": agent_a, "claw": agent_b})

    install_a2a_did_resolver(manager)

    assert callable(agent_a.a2a_did_resolver)
    assert callable(agent_b.a2a_did_resolver)
    assert agent_a.a2a_did_resolver.__self__ is not agent_b.a2a_did_resolver.__self__
    assert agent_a.a2a_did_resolver(DID_B)["id"] == DID_B
    assert agent_b.a2a_did_resolver(DID_A)["id"] == DID_A


def test_federated_fallback_off_by_default_returns_none():
    resolver = HostA2ADidResolver(_Manager({}))

    assert resolver.resolve("did:web:other-host.org:agent:x") is None


def test_resolver_is_document_lookup_not_peer_authorization():
    """A local document remains resolvable regardless of peer-directory state."""
    sender, _ = _hybrid_agent(DID_A, agent_id="did:pkh:tenant-b:emma")
    recipient, _ = _hybrid_agent(DID_B, agent_id="did:pkh:tenant-a:claw")
    # Deliberately attach an incomplete/revoked-looking hosted context. DID
    # lookup must not invoke or infer peer authorization.
    recipient.peer_directory_router = object()
    recipient.peer_requester = None
    manager = _Manager({"emma": sender, "claw": recipient})
    install_a2a_did_resolver(manager, recipient=recipient)

    document = recipient.a2a_did_resolver(DID_A)

    assert document is not None
    assert document["id"] == DID_A


def test_resolver_composes_with_envelope_verifier_end_to_end():
    sender, sender_keypair = _hybrid_agent(DID_A)
    recipient, _ = _hybrid_agent(DID_B)
    manager = _Manager({"emma": sender, "claw": recipient})
    install_a2a_did_resolver(manager)
    metadata = {"sender": DID_A}
    metadata["signature"] = sign_envelope(
        sender_keypair,
        sender=DID_A,
        task_id="t1",
        message="hello",
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id="s1",
        bound=bound_envelope_fields(metadata),
    )

    verdict = asyncio.run(
        verify_inbound_envelope(
            metadata,
            task_id="t1",
            message="hello",
            session_id="s1",
            resolver=recipient.a2a_did_resolver,
        )
    )

    assert verdict.ok is True
    assert verdict.verified is True


def test_tampered_envelope_rejected_through_resolver():
    sender, sender_keypair = _hybrid_agent(DID_A)
    recipient, _ = _hybrid_agent(DID_B)
    manager = _Manager({"emma": sender, "claw": recipient})
    install_a2a_did_resolver(manager)
    signature = sign_envelope(
        sender_keypair,
        sender=DID_A,
        task_id="t1",
        message="hello",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    verdict = asyncio.run(
        verify_inbound_envelope(
            {"sender": DID_A, "signature": signature},
            task_id="t1",
            message="TAMPERED",
            resolver=recipient.a2a_did_resolver,
        )
    )

    assert verdict.ok is False


def test_cold_registration_installs_only_new_recipient_resolver():
    """Cold onboarding leaves an existing recipient's resolver untouched."""
    warm, _ = _hybrid_agent(DID_A)
    manager = _Manager({"warm": warm})
    install_a2a_did_resolver(manager, recipient=warm)
    warm_resolver = warm.a2a_did_resolver.__self__

    cold, _ = _hybrid_agent(DID_B)
    manager._agents["cold"] = cold
    install_a2a_did_resolver(manager, recipient=cold)

    assert warm.a2a_did_resolver.__self__ is warm_resolver
    assert cold.a2a_did_resolver.__self__ is not warm_resolver
    assert cold.a2a_did_resolver(DID_A)["id"] == DID_A


def test_process_registry_installs_local_peer_verification(
    tmp_path,
    monkeypatch,
):
    from kestrel_sovereign.a2a.transport_auth import A2A_TRANSPORT_KEY_ENV
    from kestrel_sovereign.features.peers.feature import PeersFeature

    sender, sender_keypair = _hybrid_agent(DID_A)
    recipient, _ = _hybrid_agent(DID_B)
    sender_root = tmp_path / "sender"
    sender_root.mkdir()
    (sender_root / "sender_did.json").write_text(
        json.dumps(
            {
                "id": DID_A,
                "verificationMethod": (
                    sender.identity.new_verification_methods
                ),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(
        A2A_PEER_IDENTITY_ROOTS_ENV,
        json.dumps([str(sender_root)]),
    )
    monkeypatch.setenv(A2A_TRANSPORT_KEY_ENV, "process-transport-key")
    feature = PeersFeature(recipient)
    asyncio.run(feature.initialize())
    metadata = {"sender": DID_A, "a2a_verb": "read_task"}
    metadata["signature"] = sign_envelope(
        sender_keypair,
        sender=DID_A,
        task_id="process-read",
        message="read_task:process-read",
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id="a2a-read_task:process-read",
        bound=bound_envelope_fields(metadata),
    )

    verdict = asyncio.run(
        verify_inbound_envelope(
            metadata,
            task_id="process-read",
            message="read_task:process-read",
            session_id="a2a-read_task:process-read",
            resolver=recipient.a2a_did_resolver,
        )
    )

    assert callable(recipient.a2a_did_resolver)
    assert verdict.ok is True
    assert verdict.sender == DID_A


def test_process_registry_resolves_rotated_successor_material(tmp_path):
    sender, _ = _hybrid_agent(DID_A)
    sender_root = tmp_path / "rotated-sender"
    successions = sender_root / "successions"
    successions.mkdir(parents=True)
    (successions / "rotation.json").write_text(
        json.dumps(
            {
                "successor_did": DID_A,
                "successor_verification_methods": (
                    sender.identity.new_verification_methods
                ),
            }
        ),
        encoding="utf-8",
    )

    document = ProcessA2ADidResolver((sender_root,)).resolve(DID_A)

    assert document is not None
    assert document["id"] == DID_A
    assert document["verificationMethod"] == (
        sender.identity.new_verification_methods
    )


def test_process_registry_snapshot_cannot_be_rewritten_by_sibling(tmp_path):
    """A sibling cannot replace a peer's verification key after startup."""

    sender, _ = _hybrid_agent(DID_A)
    attacker, _ = _hybrid_agent(DID_A)
    sender_root = tmp_path / "sender"
    sender_root.mkdir()
    identity_path = sender_root / "sender_did.json"
    identity_path.write_text(
        json.dumps(
            {
                "id": DID_A,
                "verificationMethod": sender.identity.new_verification_methods,
            }
        ),
        encoding="utf-8",
    )
    resolver = ProcessA2ADidResolver((sender_root,))

    original = resolver.resolve(DID_A)
    identity_path.write_text(
        json.dumps(
            {
                "id": DID_A,
                "verificationMethod": attacker.identity.new_verification_methods,
            }
        ),
        encoding="utf-8",
    )

    assert resolver.resolve(DID_A) == original
