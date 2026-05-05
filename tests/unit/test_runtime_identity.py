"""
Tests for :mod:`kestrel_sovereign.identity.runtime_identity`.

The runtime loader has to handle three states:

1. Pre-ceremony: legacy ECDSA key only on disk
2. Post-ceremony: legacy + hybrid + archival keys + succession statement
3. Inconsistent: any partial state (statement without keys, etc.) — must fail

These tests build a real agent dir on tmp_path using the rotation
ceremony itself (so we exercise the full cycle: ceremony writes,
loader reads). No mocks — the on-disk shape produced by
``run_rotation_ceremony`` is the spec the loader must read.
"""

from __future__ import annotations

import json
import pytest

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.identity.runtime_identity import (
    AgentIdentity,
    RuntimeIdentityError,
    load_agent_identity,
)
from kestrel_sovereign.identity.succession import SuccessionStatement
from kestrel_sovereign.inception_service import (
    public_key_to_ethereum_address,
    create_did_document,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    ALG_ED25519,
    ALG_ML_DSA_65,
    ALG_SLH_DSA_SHA2_128S,
    Keypair,
    Secp256k1Suite,
    SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kestrel_data_key(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "x" * 32)


@pytest.fixture
def legacy_agent_on_disk(tmp_path, kestrel_data_key):
    """Mint a legacy did:pkh agent and persist to disk in the same shape
    the production inception path produces. Returns (storage_dir,
    legacy_key_id, legacy_did, legacy_keypair)."""
    storage = SecureKeyStorage(storage_dir=tmp_path)
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    address = public_key_to_ethereum_address(kp.public_key)
    legacy_did = f"did:pkh:eip155:1:{address}"
    key_id = f"kestrel_{address}"

    storage.save_private_key(kp.private_key, key_id)

    # Mirror inception's DID document shape
    did_doc = {
        "@context": "https://w3id.org/did/v1",
        "id": legacy_did,
        "publicKey": [
            {
                "id": f"{legacy_did}#keys-1",
                "type": "EcdsaSecp256k1VerificationKey2019",
                "controller": legacy_did,
                "publicKeyHex": kp.public_key.public_bytes(
                    encoding=__import__(
                        "cryptography.hazmat.primitives.serialization",
                        fromlist=["Encoding"],
                    ).Encoding.X962,
                    format=__import__(
                        "cryptography.hazmat.primitives.serialization",
                        fromlist=["PublicFormat"],
                    ).PublicFormat.UncompressedPoint,
                ).hex(),
            }
        ],
    }
    (tmp_path / f"{key_id}.json").write_text(json.dumps(did_doc, indent=2))

    return tmp_path, key_id, legacy_did, kp


@pytest.fixture
def post_ceremony_agent_on_disk(legacy_agent_on_disk, kestrel_data_key):
    """Run the rotation ceremony against the legacy agent and persist
    everything in the canonical layout. Returns the same tuple plus
    the ceremony result."""
    storage_dir, legacy_key_id, legacy_did, legacy_kp = legacy_agent_on_disk
    storage = SecureKeyStorage(storage_dir=storage_dir)

    secp = Secp256k1Suite()
    legacy_vms = build_verification_methods(legacy_did, [(secp, legacy_kp.public_key)])
    legacy_kid = legacy_vms[0]["id"].rsplit("#", 1)[-1]

    archival_kp = SLHDSASHA2128sSuite().generate_keypair()
    result = run_rotation_ceremony(
        predecessor_did=legacy_did,
        predecessor_keypair=legacy_kp,
        predecessor_kid=legacy_kid,
        predecessor_verification_methods=legacy_vms,
        new_did_domain="agents.test.example",
        new_did_slug="testbot",
        reason="runtime loader test",
        archival_keypair=archival_kp,
    )

    # Persist exactly like quantum_kestrel_1_ceremony.py does
    new_kp = result.new_identity.keypair
    storage.save_private_key(new_kp.classical.private_key, "testbot_ed25519")
    storage.save_secret_bytes(new_kp.pq.private_key, "testbot_mldsa65")
    storage.save_secret_bytes(archival_kp.private_key, "testbot_archival_slhdsa")
    storage.save_secret_bytes(archival_kp.public_key, "testbot_archival_slhdsa_pub")

    successions_dir = storage_dir / "successions"
    successions_dir.mkdir()
    (successions_dir / "testbot.json").write_text(
        json.dumps(result.succession_statement.to_dict(), indent=2)
    )

    return storage_dir, legacy_key_id, legacy_did, legacy_kp, result


# ---------------------------------------------------------------------------
# Pre-ceremony: legacy-only agent
# ---------------------------------------------------------------------------

def test_load_legacy_only_agent(legacy_agent_on_disk):
    storage_dir, key_id, legacy_did, legacy_kp = legacy_agent_on_disk
    identity = load_agent_identity(key_id, storage_dir)
    assert isinstance(identity, AgentIdentity)
    assert identity.is_hybrid is False
    assert identity.signing_did == legacy_did
    assert identity.legacy_did == legacy_did
    assert identity.legacy_keypair.suite_id == ALG_ECDSA_SECP256K1_SHA256
    # Roundtrip the legacy public key
    pre_pub = legacy_kp.public_key.public_numbers()
    post_pub = identity.legacy_keypair.public_key.public_numbers()
    assert pre_pub.x == post_pub.x and pre_pub.y == post_pub.y
    # Hybrid fields are None
    assert identity.hybrid_keypair is None
    assert identity.new_did is None
    assert identity.succession_chain is None
    assert identity.archival_keypair is None


def test_legacy_agent_missing_did_doc_raises(tmp_path, kestrel_data_key):
    """Legacy key on disk but no DID document — load must fail."""
    from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
    storage = SecureKeyStorage(storage_dir=tmp_path)
    kp = Secp256k1Suite().generate_keypair()
    storage.save_private_key(kp.private_key, "kestrel_0xdead")
    # No JSON file
    with pytest.raises(FileNotFoundError, match="DID document not found"):
        load_agent_identity("kestrel_0xdead", tmp_path)


# ---------------------------------------------------------------------------
# Post-ceremony: hybrid agent
# ---------------------------------------------------------------------------

def test_load_hybrid_agent(post_ceremony_agent_on_disk):
    storage_dir, key_id, legacy_did, legacy_kp, result = post_ceremony_agent_on_disk
    identity = load_agent_identity(key_id, storage_dir)
    assert identity.is_hybrid is True
    assert identity.signing_did == result.new_identity.did
    assert identity.new_did == result.new_identity.did
    assert identity.legacy_did == legacy_did
    # Hybrid keypair carries both halves
    assert identity.hybrid_keypair.classical.suite_id == ALG_ED25519
    assert identity.hybrid_keypair.pq.suite_id == ALG_ML_DSA_65
    # Archival keypair recovered
    assert identity.archival_keypair.suite_id == ALG_SLH_DSA_SHA2_128S
    # Chain built from the single statement
    assert len(identity.succession_chain) == 1
    # Succession statement readable
    assert identity.succession_statement.predecessor_did == legacy_did
    assert identity.succession_statement.successor_did == result.new_identity.did
    # New DID document carries both verification methods
    new_doc = identity.new_did_document
    assert new_doc["id"] == result.new_identity.did
    assert len(new_doc["verificationMethod"]) == 2


def test_hybrid_agent_can_sign_and_self_verify(post_ceremony_agent_on_disk):
    """End-to-end: load identity, sign with the hybrid keypair, verify."""
    storage_dir, key_id, legacy_did, _, result = post_ceremony_agent_on_disk
    identity = load_agent_identity(key_id, storage_dir)

    from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
    classical_kid = identity.new_did_document["verificationMethod"][0]["id"]\
        .rsplit("#", 1)[-1]
    pq_kid = identity.new_did_document["verificationMethod"][1]["id"]\
        .rsplit("#", 1)[-1]
    sigs = sign_hybrid(
        b"runtime-loaded hybrid signing test",
        identity.hybrid_keypair,
        classical_kid=classical_kid,
        pq_kid=pq_kid,
    )
    assert len(sigs) == 2
    assert {s["alg"] for s in sigs} == {ALG_ED25519, ALG_ML_DSA_65}


# ---------------------------------------------------------------------------
# Inconsistent on-disk state
# ---------------------------------------------------------------------------

def test_succession_present_but_hybrid_keys_missing_raises(
    post_ceremony_agent_on_disk,
):
    """Delete the hybrid private keys after the ceremony. The loader
    must raise rather than silently fall back to the legacy key (that
    would mask a partial-ceremony failure)."""
    storage_dir, key_id, *_ = post_ceremony_agent_on_disk
    (storage_dir / "testbot_ed25519.key.enc").unlink()
    with pytest.raises(RuntimeIdentityError, match="classical hybrid key.*missing"):
        load_agent_identity(key_id, storage_dir)


def test_succession_predecessor_mismatch_raises(post_ceremony_agent_on_disk):
    """Tamper with the on-disk DID document so the legacy DID no
    longer matches the succession statement's predecessor — load
    must raise."""
    storage_dir, key_id, legacy_did, _, _ = post_ceremony_agent_on_disk
    did_path = storage_dir / f"{key_id}.json"
    doc = json.loads(did_path.read_text())
    doc["id"] = "did:pkh:eip155:1:0x0000000000000000000000000000000000000000"
    did_path.write_text(json.dumps(doc))
    with pytest.raises(RuntimeIdentityError, match="predecessor.*loaded legacy DID"):
        load_agent_identity(key_id, storage_dir)


def test_multiple_succession_statements_raises(post_ceremony_agent_on_disk):
    """Drop a second succession statement next to the first. Until
    multi-succession runtime support ships, this must error rather
    than picking arbitrarily."""
    storage_dir, key_id, *_ = post_ceremony_agent_on_disk
    (storage_dir / "successions" / "second.json").write_text("{}")
    with pytest.raises(RuntimeIdentityError, match="multiple succession statements"):
        load_agent_identity(key_id, storage_dir)


def test_archival_keypair_missing_raises(post_ceremony_agent_on_disk):
    """Delete the archival SLH-DSA private file. Must raise (the
    succession statement carries an archival signature, so the key
    was minted; missing it means inconsistent state)."""
    storage_dir, key_id, *_ = post_ceremony_agent_on_disk
    (storage_dir / "testbot_archival_slhdsa.bytes.enc").unlink()
    with pytest.raises(RuntimeIdentityError, match="archival SLH-DSA private key"):
        load_agent_identity(key_id, storage_dir)


# ---------------------------------------------------------------------------
# Empty successions/ dir behaves like no dir
# ---------------------------------------------------------------------------

def test_empty_successions_dir_treated_as_legacy(legacy_agent_on_disk):
    """An empty successions/ subdir doesn't make an agent hybrid."""
    storage_dir, key_id, legacy_did, _ = legacy_agent_on_disk
    (storage_dir / "successions").mkdir()
    identity = load_agent_identity(key_id, storage_dir)
    assert identity.is_hybrid is False
    assert identity.signing_did == legacy_did
