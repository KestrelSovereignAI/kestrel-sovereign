"""Host-level A2A DID resolver (#1705).

The host builds a same-host DID registry from loaded agents' in-memory hybrid
identities and injects a resolver into each agent (consumed by the #1673
/tasks/send verification). These tests use real keypairs so the resolver +
#1673 verifier compose end-to-end.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.a2a.did_registry import HostA2ADidResolver, install_a2a_did_resolver
from kestrel_sovereign.a2a.envelope_signing import sign_envelope, verify_inbound_envelope


def _hybrid_agent(did: str):
    """A stand-in agent whose `.identity` exposes the hybrid fields the
    resolver reads, plus the real keypair for end-to-end signing."""
    kp = generate_hybrid_keypair()
    vms = build_verification_methods(did, kp.public_keys())
    identity = SimpleNamespace(is_hybrid=True, signing_did=did, new_verification_methods=vms)
    agent = SimpleNamespace(identity=identity, a2a_did_resolver=None)
    return agent, kp


def _legacy_agent():
    """A pre-ceremony agent: identity present but not hybrid -> not resolvable."""
    identity = SimpleNamespace(is_hybrid=False, signing_did="did:pkh:legacy", new_verification_methods=None)
    return SimpleNamespace(identity=identity, a2a_did_resolver=None)


class _Manager:
    def __init__(self, agents: dict):
        self._agents = agents

    def list_agents(self):
        return dict(self._agents)


DID_A = "did:web:example.com:agent:emma"
DID_B = "did:web:example.com:agent:claw"


def test_resolves_known_hybrid_agent_to_doc_with_vms():
    agent_a, _ = _hybrid_agent(DID_A)
    mgr = _Manager({"emma": agent_a})
    resolver = HostA2ADidResolver(mgr)

    doc = resolver.resolve(DID_A)
    assert doc is not None
    assert doc["id"] == DID_A
    assert isinstance(doc["verificationMethod"], list) and doc["verificationMethod"]


def test_unknown_did_resolves_none():
    agent_a, _ = _hybrid_agent(DID_A)
    resolver = HostA2ADidResolver(_Manager({"emma": agent_a}))
    assert resolver.resolve("did:web:example.com:agent:nobody") is None
    assert resolver.resolve("") is None


def test_legacy_non_hybrid_agent_not_resolvable():
    resolver = HostA2ADidResolver(_Manager({"old": _legacy_agent()}))
    assert resolver.resolve("did:pkh:legacy") is None


def test_install_injects_resolver_into_every_agent():
    agent_a, _ = _hybrid_agent(DID_A)
    agent_b, _ = _hybrid_agent(DID_B)
    mgr = _Manager({"emma": agent_a, "claw": agent_b})

    install_a2a_did_resolver(mgr)

    assert callable(agent_a.a2a_did_resolver)
    assert callable(agent_b.a2a_did_resolver)
    # Each agent's injected resolver can resolve the OTHER agent (cross-peer).
    assert agent_a.a2a_did_resolver(DID_B)["id"] == DID_B
    assert agent_b.a2a_did_resolver(DID_A)["id"] == DID_A


def test_federated_fallback_off_by_default_returns_none():
    resolver = HostA2ADidResolver(_Manager({}))
    # External did:web, no local match, fallback off -> None (not a network call).
    assert resolver.resolve("did:web:other-host.org:agent:x") is None


def test_resolver_composes_with_1673_verifier_end_to_end():
    """The whole point: a signed envelope from a same-host peer verifies via
    the injected resolver."""
    agent_a, kp_a = _hybrid_agent(DID_A)
    agent_b, _ = _hybrid_agent(DID_B)
    mgr = _Manager({"emma": agent_a, "claw": agent_b})
    install_a2a_did_resolver(mgr)

    # emma signs an envelope; claw verifies it using its injected resolver.
    ts = datetime.now(timezone.utc).isoformat()
    block = sign_envelope(kp_a, sender=DID_A, task_id="t1", message="hello", timestamp=ts, session_id="s1")
    metadata = {"sender": DID_A, "signature": block}

    verdict = asyncio.run(verify_inbound_envelope(
        metadata, task_id="t1", message="hello", session_id="s1",
        resolver=agent_b.a2a_did_resolver,
    ))
    assert verdict.ok is True and verdict.verified is True


def test_tampered_envelope_rejected_through_resolver():
    agent_a, kp_a = _hybrid_agent(DID_A)
    agent_b, _ = _hybrid_agent(DID_B)
    mgr = _Manager({"emma": agent_a, "claw": agent_b})
    install_a2a_did_resolver(mgr)

    ts = datetime.now(timezone.utc).isoformat()
    block = sign_envelope(kp_a, sender=DID_A, task_id="t1", message="hello", timestamp=ts)
    metadata = {"sender": DID_A, "signature": block}

    verdict = asyncio.run(verify_inbound_envelope(
        metadata, task_id="t1", message="TAMPERED", resolver=agent_b.a2a_did_resolver,
    ))
    assert verdict.ok is False
