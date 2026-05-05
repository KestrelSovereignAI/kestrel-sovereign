"""
sign_mandate / verify_mandate hybrid-format tests.

Covers:
- Pre-ceremony parent: bare-hex ECDSA wire format unchanged
- Post-ceremony parent: ``hybrid:`` prefix, base64-wrapped JSON
  signatures array
- Round-trip verification on both paths
- Tamper detection on the hybrid path
- HYBRID_REQUIRED rule: stripping the PQ half is rejected
- Missing parent_verification_methods on hybrid verify is rejected
"""

from __future__ import annotations

import base64
import json
import pytest

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.inception_service import (
    public_key_to_ethereum_address,
)
from kestrel_sovereign.security.crypto_suite import (
    Secp256k1Suite, SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage
from kestrel_sovereign.spawn.mandate import (
    SpawnMandate, sign_mandate, verify_mandate,
)


@pytest.fixture
def kestrel_data_key(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "x" * 32)


@pytest.fixture
def hybrid_parent(tmp_path, kestrel_data_key):
    """Mint a hybrid parent agent: legacy ECDSA + post-ceremony hybrid
    keypair on disk + succession statement. Returns the AgentIdentity
    bundle ready to pass to sign_mandate."""
    storage = SecureKeyStorage(storage_dir=tmp_path)
    secp = Secp256k1Suite()
    legacy_kp = secp.generate_keypair()
    address = public_key_to_ethereum_address(legacy_kp.public_key)
    legacy_did = f"did:pkh:eip155:1:{address}"
    key_id = f"kestrel_{address}"
    storage.save_private_key(legacy_kp.private_key, key_id)

    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat,
    )
    pub_hex = legacy_kp.public_key.public_bytes(
        encoding=Encoding.X962, format=PublicFormat.UncompressedPoint,
    ).hex()
    (tmp_path / f"{key_id}.json").write_text(json.dumps({
        "@context": "https://w3id.org/did/v1",
        "id": legacy_did,
        "publicKey": [{
            "id": f"{legacy_did}#keys-1",
            "type": "EcdsaSecp256k1VerificationKey2019",
            "controller": legacy_did,
            "publicKeyHex": pub_hex,
        }],
    }))

    legacy_vms = build_verification_methods(legacy_did, [(secp, legacy_kp.public_key)])
    archival_kp = SLHDSASHA2128sSuite().generate_keypair()
    result = run_rotation_ceremony(
        predecessor_did=legacy_did,
        predecessor_keypair=legacy_kp,
        predecessor_kid=legacy_vms[0]["id"].rsplit("#", 1)[-1],
        predecessor_verification_methods=legacy_vms,
        new_did_domain="agents.test.example",
        new_did_slug="parent",
        reason="mandate hybrid test",
        archival_keypair=archival_kp,
    )
    new_kp = result.new_identity.keypair
    storage.save_private_key(new_kp.classical.private_key, "parent_ed25519")
    storage.save_secret_bytes(new_kp.pq.private_key, "parent_mldsa65")
    storage.save_secret_bytes(archival_kp.private_key, "parent_archival_slhdsa")
    storage.save_secret_bytes(archival_kp.public_key, "parent_archival_slhdsa_pub")
    successions_dir = tmp_path / "successions"
    successions_dir.mkdir()
    (successions_dir / "parent.json").write_text(
        json.dumps(result.succession_statement.to_dict(), indent=2)
    )

    identity = load_agent_identity(key_id, storage_dir=tmp_path)
    return identity, legacy_kp, result


def _make_mandate() -> SpawnMandate:
    return SpawnMandate(
        parent_did="did:pkh:eip155:1:0xPARENTPLACEHOLDER",
        purpose="test mandate",
        ttl_seconds=3600,
    )


# ---------------------------------------------------------------------------
# Legacy path (no parent_identity)
# ---------------------------------------------------------------------------

def test_legacy_mandate_signs_bare_hex():
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    mandate = _make_mandate()
    signed = sign_mandate(mandate, kp.private_key)
    # Bare hex, no prefix
    assert not signed.parent_signature.startswith("hybrid:")
    bytes.fromhex(signed.parent_signature)  # parses as hex
    assert verify_mandate(signed, kp.public_key) is True


def test_legacy_tamper_detected():
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    mandate = _make_mandate()
    signed = sign_mandate(mandate, kp.private_key)
    signed.purpose = "MALICIOUS"
    assert verify_mandate(signed, kp.public_key) is False


# ---------------------------------------------------------------------------
# Hybrid path
# ---------------------------------------------------------------------------

def test_hybrid_mandate_uses_prefix(hybrid_parent):
    identity, legacy_kp, _ = hybrid_parent
    mandate = _make_mandate()
    signed = sign_mandate(
        mandate, legacy_kp.private_key, parent_identity=identity,
    )
    assert signed.parent_signature.startswith("hybrid:")
    payload = json.loads(
        base64.b64decode(signed.parent_signature[len("hybrid:"):]).decode()
    )
    algs = {entry["alg"] for entry in payload}
    assert algs == {"ed25519", "ml-dsa-65"}


def test_hybrid_mandate_round_trip_verifies(hybrid_parent):
    identity, legacy_kp, _ = hybrid_parent
    mandate = _make_mandate()
    signed = sign_mandate(
        mandate, legacy_kp.private_key, parent_identity=identity,
    )
    ok = verify_mandate(
        signed,
        legacy_kp.public_key,  # legacy pub still passed; ignored for hybrid path
        parent_verification_methods=identity.new_verification_methods,
    )
    assert ok is True


def test_hybrid_mandate_tamper_detected(hybrid_parent):
    identity, legacy_kp, _ = hybrid_parent
    mandate = _make_mandate()
    signed = sign_mandate(
        mandate, legacy_kp.private_key, parent_identity=identity,
    )
    signed.purpose = "MALICIOUS"
    ok = verify_mandate(
        signed,
        legacy_kp.public_key,
        parent_verification_methods=identity.new_verification_methods,
    )
    assert ok is False


def test_hybrid_mandate_strip_pq_half_rejected(hybrid_parent):
    """Modify the encoded payload to remove the ml-dsa-65 entry. The
    classical signature is still valid, but HYBRID_REQUIRED rejects."""
    identity, legacy_kp, _ = hybrid_parent
    mandate = _make_mandate()
    signed = sign_mandate(
        mandate, legacy_kp.private_key, parent_identity=identity,
    )
    payload_bytes = base64.b64decode(signed.parent_signature[len("hybrid:"):])
    payload = json.loads(payload_bytes.decode())
    classical_only = [s for s in payload if s["alg"] == "ed25519"]
    new_b64 = base64.b64encode(json.dumps(classical_only).encode()).decode()
    signed.parent_signature = "hybrid:" + new_b64
    ok = verify_mandate(
        signed,
        legacy_kp.public_key,
        parent_verification_methods=identity.new_verification_methods,
    )
    assert ok is False


def test_hybrid_mandate_without_vms_rejected(hybrid_parent):
    """Caller forgot to pass parent_verification_methods — must reject
    rather than fall through to a wrong code path."""
    identity, legacy_kp, _ = hybrid_parent
    mandate = _make_mandate()
    signed = sign_mandate(
        mandate, legacy_kp.private_key, parent_identity=identity,
    )
    ok = verify_mandate(signed, legacy_kp.public_key)
    assert ok is False


def test_garbage_signature_format_rejected(hybrid_parent):
    """Not a known prefix and not parseable as hex → rejected."""
    identity, legacy_kp, _ = hybrid_parent
    mandate = _make_mandate()
    mandate.parent_signature = "rsapss:gibberish"
    ok = verify_mandate(mandate, legacy_kp.public_key)
    assert ok is False
