"""
sign_package / verify_package_signature hybrid-format tests.

Covers:
- Pre-ceremony agent: sign_package writes ``signature`` (v1 hex), no
  ``signatures`` array. verify_package_signature accepts (legacy path).
- Post-ceremony agent: sign_package writes BOTH ``signatures`` (v2
  hybrid array) and ``signature`` (legacy ECDSA fallback).
  verify_package_signature picks the v2 array first.
- Tamper detection on both paths
- Stripping the PQ half from a hybrid signatures array is rejected
  (HYBRID_REQUIRED on identity-package signing)
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from kestrel_sovereign.identity.identity_package import AgentIdentityPackage
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.identity.signing import (
    sign_package, verify_package_signature,
)
from kestrel_sovereign.inception_service import (
    public_key_to_ethereum_address,
)
from kestrel_sovereign.security.crypto_suite import (
    Secp256k1Suite, SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage


@pytest.fixture
def kestrel_data_key(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "x" * 32)


@pytest.fixture
def legacy_agent_dir(tmp_path, kestrel_data_key):
    storage = SecureKeyStorage(storage_dir=tmp_path)
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    address = public_key_to_ethereum_address(kp.public_key)
    legacy_did = f"did:pkh:eip155:1:{address}"
    key_id = f"kestrel_{address}"
    storage.save_private_key(kp.private_key, key_id)

    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat,
    )
    pub_hex = kp.public_key.public_bytes(
        encoding=Encoding.X962, format=PublicFormat.UncompressedPoint,
    ).hex()
    did_doc = {
        "@context": "https://w3id.org/did/v1",
        "id": legacy_did,
        "publicKey": [{
            "id": f"{legacy_did}#keys-1",
            "type": "EcdsaSecp256k1VerificationKey2019",
            "controller": legacy_did,
            "publicKeyHex": pub_hex,
        }],
    }
    (tmp_path / f"{key_id}.json").write_text(json.dumps(did_doc, indent=2))
    return tmp_path, key_id, legacy_did, kp


@pytest.fixture
def post_ceremony_agent_dir(legacy_agent_dir, kestrel_data_key):
    storage_dir, legacy_key_id, legacy_did, legacy_kp = legacy_agent_dir
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
        reason="sign_package hybrid test",
        archival_keypair=archival_kp,
    )
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


def _make_package(did: str) -> AgentIdentityPackage:
    return AgentIdentityPackage(
        did=did,
        agent_name="testbot",
        created_at="2026-01-01T00:00:00+00:00",
        constitution_hash="0" * 64,
        constitution_text="ignored for signing test",
        export_timestamp="2026-05-05T12:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Pre-ceremony: legacy path unchanged
# ---------------------------------------------------------------------------

def test_legacy_agent_signs_v1_only(legacy_agent_dir):
    storage_dir, _, legacy_did, _ = legacy_agent_dir
    pkg = _make_package(legacy_did)
    signed = sign_package(pkg, storage_dir=storage_dir)
    assert signed.signature, "v1 signature must be populated"
    assert not signed.signatures, "legacy agent must not write v2 signatures"
    ok, msg = verify_package_signature(signed, storage_dir=storage_dir)
    assert ok, msg


def test_legacy_tamper_detected(legacy_agent_dir):
    storage_dir, _, legacy_did, _ = legacy_agent_dir
    pkg = _make_package(legacy_did)
    signed = sign_package(pkg, storage_dir=storage_dir)
    signed.agent_name = "MALICIOUSLY MODIFIED"
    ok, msg = verify_package_signature(signed, storage_dir=storage_dir)
    assert not ok, msg


# ---------------------------------------------------------------------------
# Post-ceremony: hybrid path
# ---------------------------------------------------------------------------

def test_hybrid_agent_signs_both_v1_and_v2(post_ceremony_agent_dir):
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    pkg = _make_package(legacy_did)
    signed = sign_package(pkg, storage_dir=storage_dir)
    # v1 fallback present (so old importers still work)
    assert signed.signature, "v1 fallback signature must be populated"
    # v2 hybrid array present with both algs
    assert len(signed.signatures) == 2
    algs = {entry["alg"] for entry in signed.signatures}
    assert algs == {"ed25519", "ml-dsa-65"}
    # Verification methods mirrored
    assert len(signed.verification_methods) == 2


def test_hybrid_round_trip_verifies(post_ceremony_agent_dir):
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    pkg = _make_package(legacy_did)
    signed = sign_package(pkg, storage_dir=storage_dir)
    ok, msg = verify_package_signature(signed, storage_dir=storage_dir)
    assert ok, msg
    assert "hybrid" in msg.lower()


def test_hybrid_tamper_detected(post_ceremony_agent_dir):
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    pkg = _make_package(legacy_did)
    signed = sign_package(pkg, storage_dir=storage_dir)
    signed.agent_name = "MALICIOUSLY MODIFIED"
    ok, _ = verify_package_signature(signed, storage_dir=storage_dir)
    assert not ok


def test_stripping_pq_half_rejected(post_ceremony_agent_dir):
    """Remove the ml-dsa-65 entry from the v2 signatures array.
    The classical signature still verifies on its own, but the
    HYBRID_REQUIRED rule must reject the stripped payload."""
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    pkg = _make_package(legacy_did)
    signed = sign_package(pkg, storage_dir=storage_dir)
    # Strip the PQ signature
    signed.signatures = [
        s for s in signed.signatures if s["alg"] != "ml-dsa-65"
    ]
    ok, msg = verify_package_signature(signed, storage_dir=storage_dir)
    assert not ok
    assert "ml-dsa-65" in msg or "missing" in msg.lower()


def test_corrupting_one_v2_signature_rejected(post_ceremony_agent_dir):
    """Flip a hex char in one of the two signatures. The other still
    verifies, but verify must reject because every listed sig must
    crypto-verify."""
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    pkg = _make_package(legacy_did)
    signed = sign_package(pkg, storage_dir=storage_dir)
    sig0 = signed.signatures[0]
    sig0["sig"] = (
        ("0" if sig0["sig"][0] != "0" else "1") + sig0["sig"][1:]
    )
    ok, _ = verify_package_signature(signed, storage_dir=storage_dir)
    assert not ok


def test_v2_signature_with_wrong_alg_for_kid_rejected(post_ceremony_agent_dir):
    """Signature claims alg=ed25519 but kid resolves to ml-dsa-65 VM."""
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    pkg = _make_package(legacy_did)
    signed = sign_package(pkg, storage_dir=storage_dir)
    # Swap the algs but keep kids
    signed.signatures[0]["alg"], signed.signatures[1]["alg"] = (
        signed.signatures[1]["alg"], signed.signatures[0]["alg"],
    )
    ok, _ = verify_package_signature(signed, storage_dir=storage_dir)
    assert not ok
