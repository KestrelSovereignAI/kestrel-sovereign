"""A2A verification-document lookup and signed-envelope verification."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.a2a.did_registry import (
    A2A_PEER_IDENTITY_DOCUMENTS_ENV,
    A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV,
    A2A_PEER_IDENTITY_DOCUMENTS_SHA256_ENV,
    A2A_PEER_IDENTITY_ROOTS_ENV,
    A2A_PEER_STABLE_AGENT_ID_FIELD,
    HostA2ADidResolver,
    ProcessA2ADidResolver,
    ProcessA2ADidResolverConfigurationError,
    install_a2a_did_resolver,
    install_process_a2a_did_resolver,
    launcher_attested_a2a_verification_document,
    stage_process_a2a_did_registry,
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
    monkeypatch.setenv(
        A2A_PEER_IDENTITY_DOCUMENTS_ENV,
        json.dumps(
            [
                {
                    "id": DID_A,
                    "verificationMethod": sender.identity.new_verification_methods,
                }
            ]
        ),
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


def test_process_registry_resolves_launcher_attested_successor_material():
    sender, _ = _hybrid_agent(DID_A)
    attested = {
        "id": DID_A,
        "verificationMethod": sender.identity.new_verification_methods,
    }

    document = ProcessA2ADidResolver((attested,)).resolve(DID_A)

    assert document is not None
    assert document["id"] == DID_A
    assert document["verificationMethod"] == (
        sender.identity.new_verification_methods
    )


def test_process_registry_maps_signing_did_to_attested_stable_agent_id():
    sender, _ = _hybrid_agent(DID_A)
    stable_agent_id = "did:pkh:eip155:1:0xstable"
    attested = {
        "id": DID_A,
        "verificationMethod": sender.identity.new_verification_methods,
        A2A_PEER_STABLE_AGENT_ID_FIELD: stable_agent_id,
    }

    resolver = ProcessA2ADidResolver((attested,))

    assert resolver.directory_agent_id(DID_A) == stable_agent_id
    assert resolver.directory_agent_id("did:web:example.test:unknown") is None
    # The internal launcher binding is not part of the DID document passed to
    # the cryptographic envelope verifier.
    assert A2A_PEER_STABLE_AGENT_ID_FIELD not in resolver.resolve(DID_A)


def test_process_registry_rejects_two_signing_dids_for_one_stable_agent_id():
    """A stale rotation fork cannot inherit the live peer's directory authority."""

    sender_a, _ = _hybrid_agent(DID_A)
    sender_b, _ = _hybrid_agent(DID_B)
    stable_agent_id = "did:pkh:eip155:1:0xstable"

    with pytest.raises(
        ProcessA2ADidResolverConfigurationError,
        match="duplicate stable agent id",
    ):
        ProcessA2ADidResolver(
            (
                {
                    "id": DID_A,
                    "verificationMethod": sender_a.identity.new_verification_methods,
                    A2A_PEER_STABLE_AGENT_ID_FIELD: stable_agent_id,
                },
                {
                    "id": DID_B,
                    "verificationMethod": sender_b.identity.new_verification_methods,
                    A2A_PEER_STABLE_AGENT_ID_FIELD: stable_agent_id,
                },
            )
        )


def test_process_registry_file_handoff_is_authenticated_and_one_shot(
    tmp_path,
    monkeypatch,
):
    """The Windows-safe handoff preserves the launcher's immutable snapshot."""

    sender, _ = _hybrid_agent(DID_A)
    documents = [
        {
            "id": DID_A,
            "verificationMethod": sender.identity.new_verification_methods,
        }
    ]
    monkeypatch.setattr(
        "kestrel_sovereign.a2a.did_registry."
        "_platform_requires_process_registry_file",
        lambda _encoded: True,
    )
    environment, registry_path = stage_process_a2a_did_registry(
        documents,
        launch_root=tmp_path,
    )
    assert registry_path is not None
    if os.name == "posix":
        assert registry_path.stat().st_mode & 0o077 == 0

    recipient = SimpleNamespace(a2a_did_resolver=None)
    resolver = install_process_a2a_did_resolver(
        recipient,
        environment=environment,
    )

    assert resolver is not None
    assert resolver.resolve(DID_A) == documents[0]
    assert not registry_path.exists()


def test_windows_registry_transport_switches_at_environment_value_limit(
    tmp_path,
    monkeypatch,
):
    from kestrel_sovereign.a2a import did_registry

    monkeypatch.setattr(did_registry.sys, "platform", "win32")
    small_environment, small_path = stage_process_a2a_did_registry(
        [{"payload": "small"}],
        launch_root=tmp_path,
    )
    large_environment, large_path = stage_process_a2a_did_registry(
        [{"payload": "K" * 33_000}],
        launch_root=tmp_path,
    )

    assert A2A_PEER_IDENTITY_DOCUMENTS_ENV in small_environment
    assert small_path is None
    assert A2A_PEER_IDENTITY_DOCUMENTS_ENV not in large_environment
    assert A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV in large_environment
    assert large_path is not None
    large_path.unlink()


def test_linux_registry_transport_accounts_for_environment_entry_overhead(
    tmp_path,
    monkeypatch,
):
    """Near-limit JSON must not exceed Linux's per-entry execve boundary."""
    from kestrel_sovereign.a2a import did_registry

    monkeypatch.setattr(did_registry.sys, "platform", "linux")
    empty_document = json.dumps(
        [{"payload": ""}],
        sort_keys=True,
        separators=(",", ":"),
    )
    target_bytes = did_registry._MAX_PROCESS_REGISTRY_BYTES - 1
    documents = [
        {"payload": "K" * (target_bytes - len(empty_document.encode("utf-8")))}
    ]

    environment, registry_path = stage_process_a2a_did_registry(
        documents,
        launch_root=tmp_path,
    )

    assert A2A_PEER_IDENTITY_DOCUMENTS_ENV not in environment
    assert A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV in environment
    assert registry_path is not None
    registry_path.unlink()


def test_supported_hybrid_fleet_fits_file_backed_registry_cap(
    tmp_path,
    monkeypatch,
):
    """The default 64-agent fleet must fit with normal hybrid DID documents."""
    from kestrel_sovereign.a2a import did_registry

    monkeypatch.setattr(
        did_registry,
        "_platform_requires_process_registry_file",
        lambda _encoded: True,
    )
    documents = [
        {
            "id": f"did:web:example.test:peer-{index}",
            "verificationMethod": [
                {
                    "id": f"did:web:example.test:peer-{index}#key-1",
                    "controller": f"did:web:example.test:peer-{index}",
                    "publicKeyMultibase": "z" + ("K" * 3_200),
                }
            ],
        }
        for index in range(64)
    ]

    environment, registry_path = stage_process_a2a_did_registry(
        documents,
        launch_root=tmp_path,
    )

    assert A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV in environment
    assert registry_path is not None
    registry_path.unlink()


def test_process_registry_file_handoff_keeps_feature_initialize_idempotent(
    tmp_path,
    monkeypatch,
):
    sender, _ = _hybrid_agent(DID_A)
    documents = [
        {
            "id": DID_A,
            "verificationMethod": sender.identity.new_verification_methods,
        }
    ]
    monkeypatch.setattr(
        "kestrel_sovereign.a2a.did_registry."
        "_platform_requires_process_registry_file",
        lambda _encoded: True,
    )
    environment, registry_path = stage_process_a2a_did_registry(
        documents,
        launch_root=tmp_path,
    )
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    recipient = SimpleNamespace(a2a_did_resolver=None)

    first = install_process_a2a_did_resolver(recipient)
    installed_resolver = recipient.a2a_did_resolver
    second = install_process_a2a_did_resolver(recipient)

    assert first is not None
    assert second is None
    assert recipient.a2a_did_resolver is installed_resolver
    assert registry_path is not None
    assert not registry_path.exists()
    assert A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV not in os.environ
    assert A2A_PEER_IDENTITY_DOCUMENTS_SHA256_ENV not in os.environ


def test_process_registry_file_rejects_substituted_keys_and_cleans_up(
    tmp_path,
    monkeypatch,
):
    sender, _ = _hybrid_agent(DID_A)
    attacker, _ = _hybrid_agent(DID_A)
    original = [
        {
            "id": DID_A,
            "verificationMethod": sender.identity.new_verification_methods,
        }
    ]
    monkeypatch.setattr(
        "kestrel_sovereign.a2a.did_registry."
        "_platform_requires_process_registry_file",
        lambda _encoded: True,
    )
    environment, registry_path = stage_process_a2a_did_registry(
        original,
        launch_root=tmp_path,
    )
    assert registry_path is not None
    registry_path.write_text(
        json.dumps(
            [
                {
                    "id": DID_A,
                    "verificationMethod": (
                        attacker.identity.new_verification_methods
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ProcessA2ADidResolverConfigurationError,
        match="failed authentication",
    ):
        install_process_a2a_did_resolver(
            SimpleNamespace(a2a_did_resolver=None),
            environment=environment,
        )

    assert not registry_path.exists()


def test_process_registry_rejects_ambiguous_inline_and_file_sources(tmp_path):
    launch_dir = tmp_path / ".kestrel-launch"
    launch_dir.mkdir(mode=0o700)
    registry_path = launch_dir / "a2a-peer-identities-conflict.json"
    registry_path.write_text("[]", encoding="utf-8")
    registry_path.chmod(0o600)

    with pytest.raises(
        ProcessA2ADidResolverConfigurationError,
        match="ambiguous launch sources",
    ):
        install_process_a2a_did_resolver(
            SimpleNamespace(a2a_did_resolver=None),
            environment={
                A2A_PEER_IDENTITY_DOCUMENTS_ENV: "[]",
                A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV: str(registry_path),
                A2A_PEER_IDENTITY_DOCUMENTS_SHA256_ENV: hashlib.sha256(
                    b"[]"
                ).hexdigest(),
            },
        )


def test_process_registry_snapshot_cannot_be_rewritten_by_caller():
    """A caller cannot replace launcher-attested keys after installation."""

    sender, _ = _hybrid_agent(DID_A)
    attacker, _ = _hybrid_agent(DID_A)
    attested = {
        "id": DID_A,
        "verificationMethod": sender.identity.new_verification_methods,
    }
    resolver = ProcessA2ADidResolver((attested,))

    original = resolver.resolve(DID_A)
    attested["verificationMethod"] = attacker.identity.new_verification_methods

    assert resolver.resolve(DID_A) == original


def test_process_registry_rejects_peer_authored_unbound_identity_root(tmp_path):
    """A managed child must not turn a sibling-writable root into DID authority."""

    attacker, _ = _hybrid_agent(DID_A)
    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    (attacker_root / "forged_did.json").write_text(
        json.dumps(
            {
                "id": DID_A,
                "verificationMethod": attacker.identity.new_verification_methods,
            }
        ),
        encoding="utf-8",
    )
    recipient = SimpleNamespace(a2a_did_resolver=None)

    with pytest.raises(
        ProcessA2ADidResolverConfigurationError,
        match="identity roots are not an attested registry",
    ):
        install_process_a2a_did_resolver(
            recipient,
            environment={
                A2A_PEER_IDENTITY_ROOTS_ENV: json.dumps([str(attacker_root)])
            },
        )


def test_launcher_attestation_rejects_did_document_without_matching_custody(
    tmp_path,
    monkeypatch,
):
    """A forged public document cannot borrow a peer's durable DID."""

    from kestrel_sovereign.identity.inception_did_web import create_did_web_identity
    from kestrel_sovereign.inception_service import save_born_hybrid_identity
    from kestrel_sovereign.security.crypto_suite import (
        ALG_SLH_DSA_SHA2_128S,
        get_suite,
    )

    master_key = "launcher-attestation-test-key-32-bytes"
    monkeypatch.setenv("KESTREL_DATA_KEY", master_key)
    identity = create_did_web_identity("example.com", "bound-peer")
    archival = get_suite(ALG_SLH_DSA_SHA2_128S).generate_keypair()
    save_born_hybrid_identity(
        identity.did_document,
        identity,
        archival,
        "bound-peer",
        tmp_path,
    )
    monkeypatch.setenv("KESTREL_DATA_KEY", "wrong-child-key")
    original = launcher_attested_a2a_verification_document(
        tmp_path,
        expected_agent_did=identity.did,
        master_key=master_key,
    )
    assert original is not None
    assert original["id"] == identity.did

    attacker, _ = _hybrid_agent(identity.did)
    forged = dict(identity.did_document)
    forged["verificationMethod"] = attacker.identity.new_verification_methods
    (tmp_path / "bound-peer_did.json").write_text(
        json.dumps(forged),
        encoding="utf-8",
    )

    with pytest.raises(
        ProcessA2ADidResolverConfigurationError,
        match="failed cryptographic validation",
    ):
        launcher_attested_a2a_verification_document(
            tmp_path,
            expected_agent_did=identity.did,
            master_key=master_key,
        )
