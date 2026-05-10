"""
Quantum Hardening epic — cross-wave end-to-end integration (#921).

The unit suite covers each wave's primitives in isolation. This file
exercises the seams BETWEEN them — the path a real Kestrel agent
walks when migrating, signing, sealing, and shipping under the new
hybrid stack:

    Wave 2 hybrid identity
        -> Wave 3 succession ceremony + chain walker
        -> Wave 3 SLH-DSA archival countersignature
        -> Wave 4 KEM-wrapped sealed capsule for the rotated identity
        -> Wave 5 SLH-DSA-signed release manifest
        -> Wave 0C AEAD persistence (key storage + tokens)

If any cross-wave call signature drifts, this test breaks loudly.
That's the value: a regression in succession.SuccessionStatement that
unit tests still pass would surface here when the chain walker
consumes its output.

These tests use real :class:`SecureKeyStorage`, real file IO, and
real ``kestrel release sign`` / ``verify`` CLI dispatch — no mocks.
They are deliberately slower than unit tests (SLH-DSA keygen alone
costs ~50ms) and live under ``tests/integration/`` so the fast
``./run_tests.py --unit`` loop stays fast.

The seam test also acts as the deployment dry-run for the four
agent migrations (Kestrel #1, Meridian, Emma, Frinz tenants) — it
runs the exact procedure documented in
``docs/architecture/security/SUCCESSION_RUNBOOK.md`` end-to-end
against throwaway keys, so a regression to ANY primitive the
runbook depends on fails CI before a real ceremony fails in
production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from kestrel_sovereign.cli_release import cmd_release_sign, cmd_release_verify
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.identity.succession_chain import (
    verify_artifact_against_chain,
)
from kestrel_sovereign.inception_service import public_key_to_ethereum_address
from kestrel_sovereign.security.crypto_suite import (
    Secp256k1Suite,
    SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage
from kestrel_sovereign.security.kem_suite import (
    ALG_ML_KEM_768,
    ALG_X25519,
    get_kem_suite,
)
from kestrel_sovereign.security.multikey import public_key_to_multibase
from kestrel_sovereign.security.sealed_capsule import (
    SealedCapsuleError,
    open_capsule,
    seal_capsule,
)
from kestrel_sovereign.security.verify_policy import VerifyPolicy
from kestrel_sdk.security.aead import AEADCipher, DecryptionError


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kestrel_data_key(monkeypatch):
    """Set KESTREL_DATA_KEY for SecureKeyStorage round-trips.

    Every integration test in this file persists key material via
    SecureKeyStorage, which requires the master key be configured at
    process scope.
    """
    monkeypatch.setenv("KESTREL_DATA_KEY", "x" * 32)


@pytest.fixture
def legacy_kestrel_identity():
    """Legacy ECDSA-only Kestrel-#1-style identity.

    Modeled exactly on the rotation_ceremony unit fixture: the DID is
    derived from the keypair so verify_did_binding() against did:pkh
    accepts it. This is the SAME shape Kestrel #1, Emma, Meridian, and
    Frinz tenants carry on disk today.
    """
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    address = public_key_to_ethereum_address(kp.public_key)
    did = f"did:pkh:eip155:1:{address}"
    vms = build_verification_methods(did, [(secp, kp.public_key)])
    return {
        "did": did,
        "kp": kp,
        "kid": vms[0]["id"].rsplit("#", 1)[-1],
        "vms": vms,
    }


# ---------------------------------------------------------------------------
# Wave 2 + 3: identity rotation + chain walker
# ---------------------------------------------------------------------------

def test_wave_2_3_seam_legacy_to_hybrid_rotation_and_artifact_verification(
    legacy_kestrel_identity, kestrel_data_key, tmp_path,
):
    """The migration story end-to-end:

    1. Legacy ECDSA agent runs the rotation ceremony with an SLH-DSA
       archival countersignature.
    2. Both halves of the new hybrid identity get persisted via
       SecureKeyStorage (the operational shape from the runbook).
    3. A post-cutoff artifact is hybrid-signed by the new identity.
    4. The chain walker verifies it against the original root_did
       under HYBRID_REQUIRED policy.

    A break in any of {ceremony output schema, chain build_chain
    invariants, verify_artifact_against_chain anchor check, hybrid
    keypair sign_hybrid signature shape, kid format} fails this.
    """
    slh = SLHDSASHA2128sSuite()
    archival_kp = slh.generate_keypair()

    result = run_rotation_ceremony(
        predecessor_did=legacy_kestrel_identity["did"],
        predecessor_keypair=legacy_kestrel_identity["kp"],
        predecessor_kid=legacy_kestrel_identity["kid"],
        predecessor_verification_methods=legacy_kestrel_identity["vms"],
        new_did_domain="agents.kestrel-sovereign.test",
        new_did_slug="kestrel-1",
        reason="quantum-hardening migration",
        effective_from="2026-05-04T18:00:00+00:00",
        archival_keypair=archival_kp,
    )

    storage = SecureKeyStorage(storage_dir=tmp_path / "rotated-keys")
    classical_kp = result.new_identity.keypair.classical
    pq_kp = result.new_identity.keypair.pq
    storage.save_secret_bytes(
        classical_kp.private_key.private_bytes_raw(), "kestrel1_classical",
    )
    storage.save_secret_bytes(pq_kp.private_key, "kestrel1_pq")
    storage.save_secret_bytes(archival_kp.private_key, "kestrel1_archival")
    assert storage.has_secret_bytes("kestrel1_classical")
    assert storage.has_secret_bytes("kestrel1_pq")
    assert storage.has_secret_bytes("kestrel1_archival")
    assert storage.load_secret_bytes("kestrel1_pq") == pq_kp.private_key

    classical_kid = result.new_identity.did_document["verificationMethod"][0]["id"]\
        .rsplit("#", 1)[-1]
    pq_kid = result.new_identity.did_document["verificationMethod"][1]["id"]\
        .rsplit("#", 1)[-1]
    payload = b"kestrel1 attesting under the new hybrid identity, post-cutoff"
    artifact_signatures = sign_hybrid(
        payload, result.new_identity.keypair,
        classical_kid=classical_kid, pq_kid=pq_kid,
    )

    new_did_doc_vms = list(
        result.new_identity.did_document["verificationMethod"]
    )
    def _resolver(did: str) -> dict:
        if did == result.new_identity.did:
            return {"id": did, "verificationMethod": new_did_doc_vms}
        raise ValueError(f"unknown did: {did!r}")

    verdict = verify_artifact_against_chain(
        root_did=legacy_kestrel_identity["did"],
        root_verification_methods=legacy_kestrel_identity["vms"],
        chain=result.chain,
        artifact_timestamp="2026-06-01T00:00:00+00:00",
        artifact_payload=payload,
        artifact_signatures=artifact_signatures,
        policy=VerifyPolicy.HYBRID_REQUIRED,
        did_web_resolver=_resolver,
    )
    assert verdict.ok, verdict.reason
    assert verdict.active_identity.did == result.new_identity.did
    assert verdict.active_identity.post_cutoff


def test_chain_anchor_mismatch_rejects_unrelated_chain(
    legacy_kestrel_identity, kestrel_data_key,
):
    """Defense-in-depth at the seam: a chain that doesn't anchor to the
    supplied root_did MUST be rejected even if every signature is
    crypto-valid. This is the codex P1 round 11 attacker-takeover guard.
    """
    secp = Secp256k1Suite()
    other_kp = secp.generate_keypair()
    other_addr = public_key_to_ethereum_address(other_kp.public_key)
    other_did = f"did:pkh:eip155:1:{other_addr}"
    other_vms = build_verification_methods(other_did, [(secp, other_kp.public_key)])

    other_result = run_rotation_ceremony(
        predecessor_did=other_did,
        predecessor_keypair=other_kp,
        predecessor_kid=other_vms[0]["id"].rsplit("#", 1)[-1],
        predecessor_verification_methods=other_vms,
        new_did_domain="agents.kestrel-sovereign.test",
        new_did_slug="other-agent",
        reason="unrelated chain",
    )

    payload = b"impersonation attempt"
    classical_kid = other_result.new_identity.did_document\
        ["verificationMethod"][0]["id"].rsplit("#", 1)[-1]
    pq_kid = other_result.new_identity.did_document\
        ["verificationMethod"][1]["id"].rsplit("#", 1)[-1]
    sigs = sign_hybrid(
        payload, other_result.new_identity.keypair,
        classical_kid=classical_kid, pq_kid=pq_kid,
    )

    new_vms = list(other_result.new_identity.did_document["verificationMethod"])
    def _resolver(did: str) -> dict:
        if did == other_result.new_identity.did:
            return {"id": did, "verificationMethod": new_vms}
        raise ValueError(did)

    verdict = verify_artifact_against_chain(
        root_did=legacy_kestrel_identity["did"],  # pretending to be us
        root_verification_methods=legacy_kestrel_identity["vms"],
        chain=other_result.chain,
        artifact_timestamp="2026-06-01T00:00:00+00:00",
        artifact_payload=payload,
        artifact_signatures=sigs,
        policy=VerifyPolicy.HYBRID_REQUIRED,
        did_web_resolver=_resolver,
    )
    assert not verdict.ok
    assert "anchor" in verdict.reason


# ---------------------------------------------------------------------------
# Wave 0C: AEAD migration round-trip with real storage
# ---------------------------------------------------------------------------

def test_aead_v2_round_trip_through_secure_key_storage(
    kestrel_data_key, tmp_path,
):
    """SecureKeyStorage internally uses AEADCipher. Save raw PQ bytes,
    load them back, byte-equal. This is the path used by every wave's
    persisted secret — Wave 3 archival keys, Wave 5 release keys.
    """
    storage = SecureKeyStorage(storage_dir=tmp_path / "keys")
    slh = SLHDSASHA2128sSuite()
    kp = slh.generate_keypair()

    storage.save_secret_bytes(kp.private_key, "slh_secret")
    storage.save_secret_bytes(kp.public_key, "slh_secret_pub")

    assert storage.load_secret_bytes("slh_secret") == kp.private_key
    assert storage.load_secret_bytes("slh_secret_pub") == kp.public_key


def test_aead_cross_version_legacy_fernet_then_v2_write(kestrel_data_key):
    """The Wave 0C migration story: legacy data still decrypts, new
    writes are KSAv2. This is the same path scripts/rotate_agent_key.py
    already exercises in production migrations.
    """
    raw_key = b"k" * 32
    aead = AEADCipher(raw_key)

    plaintext = b"a conversation row from before Wave 0C"
    v2_token = aead.encrypt(plaintext)
    assert v2_token.startswith(b"KSAv2:")
    assert aead.decrypt(v2_token) == plaintext

    from cryptography.fernet import Fernet
    import base64
    fernet = Fernet(base64.urlsafe_b64encode(raw_key))
    fernet_token = fernet.encrypt(plaintext)
    assert aead.decrypt(fernet_token) == plaintext


def test_aead_aad_binding_rejects_swapped_context(kestrel_data_key):
    """AAD binding works end-to-end: encrypt with AAD=A, attempt
    decrypt with AAD=B fails. This is the property the sealed-capsule
    format/version binding relies on.
    """
    aead = AEADCipher(b"k" * 32)
    token = aead.encrypt(b"payload", aad=b"context-A")
    with pytest.raises(DecryptionError):
        aead.decrypt(token, aad=b"context-B")


# ---------------------------------------------------------------------------
# Wave 4: sealed capsule with SecureKeyStorage-persisted PQ secrets
# ---------------------------------------------------------------------------

def test_wave_4_sealed_capsule_with_persisted_recipient_keys(
    kestrel_data_key, tmp_path,
):
    """Operational seam: the recipient's hybrid KEM keypair is at rest
    via SecureKeyStorage. Sender seals a capsule against the public
    halves; recipient loads private halves from disk and opens it.

    A leak in the load_secret_bytes serialization (e.g. silent
    truncation, encoding drift) would make the decapsulation derive
    a different AES key and the AEAD auth tag would fail.
    """
    storage = SecureKeyStorage(storage_dir=tmp_path / "kem-keys")

    classical_suite = get_kem_suite(ALG_X25519)
    pq_suite = get_kem_suite(ALG_ML_KEM_768)
    classical_kp = classical_suite.generate_keypair()
    pq_kp = pq_suite.generate_keypair()

    storage.save_secret_bytes(
        classical_kp.private_key.private_bytes_raw(), "recipient_x25519",
    )
    storage.save_secret_bytes(pq_kp.private_key, "recipient_mlkem")

    payload = b"sovereign companion conversation export, recipient-pinned"
    capsule = seal_capsule(
        payload,
        recipient_classical_public_key=classical_kp.public_key,
        recipient_pq_public_key=pq_kp.public_key,
    )

    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
    )
    classical_priv_raw = storage.load_secret_bytes("recipient_x25519")
    classical_priv_obj = X25519PrivateKey.from_private_bytes(classical_priv_raw)
    classical_kp_loaded = type(classical_kp)(
        suite_id=classical_kp.suite_id,
        private_key=classical_priv_obj,
        public_key=classical_priv_obj.public_key(),
    )
    pq_priv_raw = storage.load_secret_bytes("recipient_mlkem")
    pq_kp_loaded = type(pq_kp)(
        suite_id=pq_kp.suite_id,
        private_key=pq_priv_raw,
        public_key=pq_kp.public_key,
    )

    recovered = open_capsule(capsule, classical_kp_loaded, pq_kp_loaded)
    assert recovered == payload


def test_wave_4_sealed_capsule_rejects_wrong_recipient(kestrel_data_key):
    """Negative seam: a capsule sealed for recipient A must NOT open
    with recipient B's keypair. Different decapsulation result →
    different derived AES key → AEAD auth fails. This is the property
    we rely on to scope a capsule to one recipient."""
    classical_suite = get_kem_suite(ALG_X25519)
    pq_suite = get_kem_suite(ALG_ML_KEM_768)
    a_classical = classical_suite.generate_keypair()
    a_pq = pq_suite.generate_keypair()
    b_classical = classical_suite.generate_keypair()
    b_pq = pq_suite.generate_keypair()

    capsule = seal_capsule(
        b"only for A",
        recipient_classical_public_key=a_classical.public_key,
        recipient_pq_public_key=a_pq.public_key,
    )
    with pytest.raises(SealedCapsuleError):
        open_capsule(capsule, b_classical, b_pq)


# ---------------------------------------------------------------------------
# Wave 5: release sign + verify CLI round-trip
# ---------------------------------------------------------------------------

def test_wave_5_cli_release_sign_then_verify_round_trip(
    kestrel_data_key, tmp_path,
):
    """Full ``kestrel release sign`` then ``kestrel release verify``
    cycle through the CLI module's command handlers. Exercises:
    - SecureKeyStorage with the ``_pub`` sidecar convention
    - SLH-DSA signature production
    - Manifest JSON format + path-traversal invariants
    - Pinned multibase verification

    A regression in ``_load_slh_keypair`` (e.g. forgetting the
    ``_pub`` sidecar) or in manifest schema validation would fail
    here even if both halves still pass their unit tests.
    """
    storage_dir = tmp_path / "release-keys"
    storage_dir.mkdir()
    storage = SecureKeyStorage(storage_dir=storage_dir)
    slh = SLHDSASHA2128sSuite()
    kp = slh.generate_keypair()
    storage.save_secret_bytes(kp.private_key, "release-key")
    storage.save_secret_bytes(kp.public_key, "release-key_pub")

    artifacts = tmp_path / "dist"
    artifacts.mkdir()
    (artifacts / "kestrel-1.2.3-py3-none-any.whl").write_bytes(b"\x00" * 4096)
    (artifacts / "kestrel-1.2.3.tar.gz").write_bytes(b"\xff" * 8192)

    manifest_path = artifacts / "release-manifest.json"
    sign_args = argparse.Namespace(
        artifacts_dir=str(artifacts),
        release_tag="v1.2.3",
        key_id="release-key",
        signer_did="did:web:agents.kestrel-sovereign.test:kestrel-1",
        kid="release-key-1",
        output=str(manifest_path),
        storage_dir=str(storage_dir),
    )
    rc = cmd_release_sign(sign_args)
    assert rc == 0
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["release_tag"] == "v1.2.3"
    assert {a["path"] for a in manifest["artifacts"]} == {
        "kestrel-1.2.3-py3-none-any.whl",
        "kestrel-1.2.3.tar.gz",
    }

    pub_mb = public_key_to_multibase(slh, kp.public_key)
    verify_args = argparse.Namespace(
        manifest=str(manifest_path),
        artifacts_dir=str(artifacts),
        trusted_signer_multibase=pub_mb,
    )
    assert cmd_release_verify(verify_args) == 0


def test_wave_5_verify_rejects_tampered_artifact(kestrel_data_key, tmp_path):
    """Sign a manifest, then modify a single byte in one artifact.
    Verify must reject. This is the post-publication tamper-detection
    property the manifest exists for.
    """
    storage_dir = tmp_path / "rk"
    storage_dir.mkdir()
    storage = SecureKeyStorage(storage_dir=storage_dir)
    slh = SLHDSASHA2128sSuite()
    kp = slh.generate_keypair()
    storage.save_secret_bytes(kp.private_key, "k1")
    storage.save_secret_bytes(kp.public_key, "k1_pub")

    artifacts = tmp_path / "art"
    artifacts.mkdir()
    artifact_file = artifacts / "wheel.whl"
    artifact_file.write_bytes(b"original content " * 100)

    manifest_path = artifacts / "release-manifest.json"
    rc = cmd_release_sign(argparse.Namespace(
        artifacts_dir=str(artifacts),
        release_tag="v0.0.1",
        key_id="k1",
        signer_did="",
        kid="k1",
        output=str(manifest_path),
        storage_dir=str(storage_dir),
    ))
    assert rc == 0

    artifact_file.write_bytes(b"tampered content " * 100)

    pub_mb = public_key_to_multibase(slh, kp.public_key)
    rc = cmd_release_verify(argparse.Namespace(
        manifest=str(manifest_path),
        artifacts_dir=str(artifacts),
        trusted_signer_multibase=pub_mb,
    ))
    assert rc != 0


# ---------------------------------------------------------------------------
# Full-stack seam: rotated identity signs a release for itself
# ---------------------------------------------------------------------------

def test_full_stack_rotated_identity_signs_release_then_seals_for_recipient(
    legacy_kestrel_identity, kestrel_data_key, tmp_path,
):
    """The most complete seam in the file. Walks the entire epic:

    1. Legacy ECDSA agent rotates to hybrid did:web (Wave 2 + 3)
    2. The new agent's archival SLH-DSA key signs a release manifest
       (Wave 5) — release-signing keys are SLH-DSA per the manifest spec.
    3. A sealed capsule wraps the signed manifest for a peer recipient
       (Wave 4).
    4. The recipient opens the capsule and verifies the release manifest
       under the same trusted-signer multibase. End-to-end auth
       preservation through the encrypted-shipment path.

    This is the closest CI test we have to the real Kestrel #1 →
    Meridian capsule shipment that the runbook describes.
    """
    slh = SLHDSASHA2128sSuite()
    archival_kp = slh.generate_keypair()

    rotation = run_rotation_ceremony(
        predecessor_did=legacy_kestrel_identity["did"],
        predecessor_keypair=legacy_kestrel_identity["kp"],
        predecessor_kid=legacy_kestrel_identity["kid"],
        predecessor_verification_methods=legacy_kestrel_identity["vms"],
        new_did_domain="agents.kestrel-sovereign.test",
        new_did_slug="kestrel-1",
        reason="full-stack seam",
        archival_keypair=archival_kp,
    )

    storage_dir = tmp_path / "release-keys"
    storage_dir.mkdir()
    storage = SecureKeyStorage(storage_dir=storage_dir)
    storage.save_secret_bytes(archival_kp.private_key, "release-key")
    storage.save_secret_bytes(archival_kp.public_key, "release-key_pub")

    artifacts = tmp_path / "dist"
    artifacts.mkdir()
    (artifacts / "kestrel.whl").write_bytes(b"release artifact bytes")

    # Write the manifest OUTSIDE the artifacts tree so it isn't seen as
    # an extra unmanifested file by the verifier. The CLI's sign path
    # has a self-referential skip when output lands in artifacts_dir,
    # but the seam we care about here is "manifest travels through a
    # capsule and lands elsewhere," so keep them physically separate.
    manifest_path = tmp_path / "release-manifest.json"
    rc = cmd_release_sign(argparse.Namespace(
        artifacts_dir=str(artifacts),
        release_tag="v1.0.0-pq",
        key_id="release-key",
        signer_did=rotation.new_identity.did,
        kid="release-key-1",
        output=str(manifest_path),
        storage_dir=str(storage_dir),
    ))
    assert rc == 0
    manifest_bytes = manifest_path.read_bytes()

    classical_suite = get_kem_suite(ALG_X25519)
    pq_suite = get_kem_suite(ALG_ML_KEM_768)
    recipient_classical = classical_suite.generate_keypair()
    recipient_pq = pq_suite.generate_keypair()

    capsule = seal_capsule(
        manifest_bytes,
        recipient_classical_public_key=recipient_classical.public_key,
        recipient_pq_public_key=recipient_pq.public_key,
    )

    recovered = open_capsule(capsule, recipient_classical, recipient_pq)
    assert recovered == manifest_bytes

    recovered_manifest = tmp_path / "recovered-manifest.json"
    recovered_manifest.write_bytes(recovered)

    pub_mb = public_key_to_multibase(slh, archival_kp.public_key)
    rc = cmd_release_verify(argparse.Namespace(
        manifest=str(recovered_manifest),
        artifacts_dir=str(artifacts),
        trusted_signer_multibase=pub_mb,
    ))
    assert rc == 0
