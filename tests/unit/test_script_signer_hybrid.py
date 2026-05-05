"""
ScriptSigner hybrid-format tests — Quantum Hardening epic, signing-path
follow-up to PR #999.

Covers:
- Pre-ceremony agent: signs ``ecdsa:`` (unchanged), verify still works
- Post-ceremony agent: signs ``hybrid:`` and verify accepts the new format
- Cross-format: a script signed pre-ceremony (legacy ``ecdsa:``) still
  verifies after the agent rotates (legacy keypair is still loaded)
- Tamper detection: any tampered byte in the hybrid payload makes
  verify return False
- Bad-shape rejection: ``hmac:``, unknown prefix, empty payload all
  reject as before
"""

from __future__ import annotations

import json
import os
import shutil
import pytest
from pathlib import Path

from kestrel_sovereign.features.compute.script_signer import ScriptSigner
from kestrel_sovereign.features.compute.models import ComputeScript
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.inception_service import (
    public_key_to_ethereum_address,
)
from kestrel_sovereign.security.crypto_suite import (
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
def legacy_agent_dir(tmp_path, kestrel_data_key):
    """Mint a legacy did:pkh agent + persist the key bundle + DID JSON."""
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
    """Run the rotation ceremony so the agent dir contains both legacy
    keys and the hybrid bundle + succession statement."""
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
        reason="ScriptSigner hybrid test",
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


def _make_script(content: str = "print('hello')") -> ComputeScript:
    return ComputeScript(
        id="00000000-0000-0000-0000-000000000001",
        name="t",
        language="python",
        content=content,
        purpose="hybrid round-trip",
    )


# ---------------------------------------------------------------------------
# Pre-ceremony: ecdsa: format unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_legacy_agent_signs_ecdsa(legacy_agent_dir):
    storage_dir, _, legacy_did, _ = legacy_agent_dir
    db_path = str(storage_dir / "x.db")  # path is only used to derive dir
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    script = _make_script()
    sig = await signer.sign(script)
    assert sig.startswith("ecdsa:"), f"expected ecdsa: prefix, got {sig[:20]}"
    script.signature = sig
    assert await signer.verify(script) is True


# ---------------------------------------------------------------------------
# Post-ceremony: hybrid: format
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_agent_signs_hybrid_format(post_ceremony_agent_dir):
    storage_dir, _, legacy_did, _, result = post_ceremony_agent_dir
    db_path = str(storage_dir / "x.db")
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    script = _make_script("print('hybrid world')")
    sig = await signer.sign(script)
    assert sig.startswith("hybrid:"), f"expected hybrid: prefix, got {sig[:20]}"

    # Decode the wrapped JSON and confirm both algs are present
    import base64
    payload = json.loads(base64.b64decode(sig[len("hybrid:"):]).decode())
    algs = {entry["alg"] for entry in payload}
    assert "ed25519" in algs
    assert "ml-dsa-65" in algs


@pytest.mark.asyncio
async def test_hybrid_round_trip_verifies(post_ceremony_agent_dir):
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    db_path = str(storage_dir / "x.db")
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    script = _make_script("round trip test")
    script.signature = await signer.sign(script)
    assert await signer.verify(script) is True


@pytest.mark.asyncio
async def test_hybrid_tamper_detected(post_ceremony_agent_dir):
    """Modify the script content after signing — verify must fail."""
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    db_path = str(storage_dir / "x.db")
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    script = _make_script("original content")
    script.signature = await signer.sign(script)
    # Tamper
    script.content = "MALICIOUSLY MODIFIED"
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_hybrid_payload_tamper_detected(post_ceremony_agent_dir):
    """Flip a byte inside the base64 hybrid payload — verify must fail."""
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    db_path = str(storage_dir / "x.db")
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    script = _make_script("payload tamper test")
    script.signature = await signer.sign(script)
    prefix, payload = script.signature.split(":", 1)
    # Replace one character mid-payload
    flipped = payload[:50] + ("A" if payload[50] != "A" else "B") + payload[51:]
    script.signature = f"{prefix}:{flipped}"
    assert await signer.verify(script) is False


# ---------------------------------------------------------------------------
# Cross-format: legacy artifact still verifies after rotation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pre_ceremony_signature_still_verifies_post_ceremony(
    post_ceremony_agent_dir,
):
    """A script signed BEFORE the rotation (with ecdsa: format) must
    still verify AFTER the rotation, because the agent's runtime
    keeps the legacy keypair loaded for backward compat."""
    storage_dir, _, legacy_did, legacy_kp, _ = post_ceremony_agent_dir
    db_path = str(storage_dir / "x.db")

    # Sign with the legacy keypair directly (simulating an old artifact)
    import base64, hashlib
    from kestrel_sovereign.security.crypto_suite import (
        ALG_ECDSA_SECP256K1_SHA256, get_suite,
    )
    secp = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    script = _make_script("legacy-signed before rotation")
    canonical = f"{script.name}|{script.language}|{script.content}|{script.purpose}"
    content_hash = hashlib.sha256(canonical.encode()).hexdigest()
    content_hash_bytes = hashlib.sha256(content_hash.encode()).digest()
    legacy_sig = secp.sign(content_hash_bytes, legacy_kp.private_key)
    script.signature = "ecdsa:" + base64.b64encode(legacy_sig).decode()

    # Now use the post-ceremony signer to verify
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    assert await signer.verify(script) is True


# ---------------------------------------------------------------------------
# Bad-shape rejection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hmac_prefix_rejected(post_ceremony_agent_dir):
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    db_path = str(storage_dir / "x.db")
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    script = _make_script()
    script.signature = "hmac:anything"
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_unknown_prefix_rejected(post_ceremony_agent_dir):
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    db_path = str(storage_dir / "x.db")
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    script = _make_script()
    script.signature = "rsapss:base64data"
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_empty_signature_rejected(post_ceremony_agent_dir):
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    db_path = str(storage_dir / "x.db")
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    script = _make_script()
    script.signature = ""
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_post_destruction_hybrid_signing_still_works(post_ceremony_agent_dir):
    """After scripts/quantum_destroy_legacy_key.py removes the legacy
    .key.enc, the agent should still be able to sign new scripts via
    the hybrid keypair. Codex P1 catch on PR #1004: the script_signer
    gate previously raised because self._private_key was None even
    though hybrid signing was viable."""
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    address = legacy_did.split(":")[-1]
    # Simulate destruction
    (storage_dir / f"kestrel_{address}.key.enc").unlink()

    db_path = str(storage_dir / "x.db")
    signer = ScriptSigner(agent_did=f"did:ethr:{address}", db_path=db_path)
    script = _make_script("post-destruction signing")
    sig = await signer.sign(script)
    assert sig.startswith("hybrid:"), f"expected hybrid, got {sig[:20]}"
    script.signature = sig
    assert await signer.verify(script) is True


@pytest.mark.asyncio
async def test_hybrid_payload_not_a_list_rejected(post_ceremony_agent_dir):
    storage_dir, _, legacy_did, _, _ = post_ceremony_agent_dir
    db_path = str(storage_dir / "x.db")
    signer = ScriptSigner(agent_did=f"did:ethr:{legacy_did.split(':')[-1]}", db_path=db_path)
    script = _make_script()
    import base64
    script.signature = "hybrid:" + base64.b64encode(b'{"not": "a list"}').decode()
    assert await signer.verify(script) is False
