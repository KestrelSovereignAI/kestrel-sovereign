"""
Sealed capsule tests — Wave 4 sub-PR 3 (#919).

End-to-end tests for the seal+open flow that wraps payloads with
hybrid KEM (X25519 + ML-KEM-768) and AES-256-GCM.

Covers:
- Round-trip: seal then open returns the original payload
- Two calling conventions: HybridKEMKeypair vs (classical, pq) split
- Empty payload, large payload (1 MB)
- Distinct seals to the same recipient produce distinct envelopes
- Wrong recipient: AEAD authentication fails (every tamper mode
  collapses into "AEAD authentication failed")
- Tampered classical KEM ciphertext → fails
- Tampered PQ KEM ciphertext → fails
- Tampered AEAD ciphertext → fails
- Tampered embedded classical pubkey → fails (precise error, not
  generic AEAD failure)
- Tampered embedded PQ pubkey → fails (precise error)
- Algorithm-pair mismatch (capsule says ml-kem-768, keypair is some
  other PQ alg) → precise error
- Format/version mismatch in envelope → precise error
- Malformed JSON / non-string capsule → typed error
- Format-version AAD binding: a v2-claimed envelope can't be opened
  by v1 (cross-version replay protection)
"""

from __future__ import annotations

import json

import pytest

from kestrel_sovereign.security.hybrid_kem import (
    HybridKEMKeypair,
    generate_hybrid_kem_keypair,
)
from kestrel_sovereign.security.kem_suite import (
    ALG_ML_KEM_768,
    ALG_X25519,
    MLKEM768Suite,
    X25519Suite,
)
from kestrel_sovereign.security.sealed_capsule import (
    CAPSULE_FORMAT_ID,
    CAPSULE_FORMAT_VERSION,
    SealedCapsuleError,
    open_capsule,
    seal_capsule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hybrid_kp() -> HybridKEMKeypair:
    return generate_hybrid_kem_keypair()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_round_trip_basic(hybrid_kp):
    payload = b"top secret identity package"
    capsule = seal_capsule(
        payload,
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    recovered = open_capsule(capsule, hybrid_kp)
    assert recovered == payload


def test_open_capsule_two_calling_conventions(hybrid_kp):
    payload = b"x"
    capsule = seal_capsule(
        payload,
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    # Convention 1: HybridKEMKeypair
    a = open_capsule(capsule, hybrid_kp)
    # Convention 2: split halves
    b = open_capsule(capsule, hybrid_kp.classical, hybrid_kp.pq)
    assert a == b == payload


def test_round_trip_empty_payload(hybrid_kp):
    capsule = seal_capsule(
        b"",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    assert open_capsule(capsule, hybrid_kp) == b""


def test_round_trip_large_payload(hybrid_kp):
    """1 MB exercises the AEAD layer beyond the trivial-size case."""
    import os
    payload = os.urandom(1_048_576)
    capsule = seal_capsule(
        payload,
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    assert open_capsule(capsule, hybrid_kp) == payload


def test_distinct_seals_produce_distinct_envelopes(hybrid_kp):
    """Each seal samples fresh randomness for both halves of the hybrid
    KEM AND for the AEAD nonce — two seals of the same payload to the
    same recipient produce different envelopes."""
    capsule_a = seal_capsule(
        b"same",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    capsule_b = seal_capsule(
        b"same",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    assert capsule_a != capsule_b
    # Both still open to the same plaintext
    assert open_capsule(capsule_a, hybrid_kp) == b"same"
    assert open_capsule(capsule_b, hybrid_kp) == b"same"


# ---------------------------------------------------------------------------
# Wire format checks
# ---------------------------------------------------------------------------

def test_envelope_has_expected_structure(hybrid_kp):
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    env = json.loads(capsule)
    assert env["format"] == CAPSULE_FORMAT_ID
    assert env["version"] == CAPSULE_FORMAT_VERSION
    assert env["kem"]["classical_alg"] == ALG_X25519
    assert env["kem"]["pq_alg"] == ALG_ML_KEM_768
    assert env["kem"]["classical_pub_multibase"].startswith("z")
    assert env["kem"]["pq_pub_multibase"].startswith("z")
    assert env["ciphertext"].startswith("KSAv2:")


# ---------------------------------------------------------------------------
# Wrong recipient
# ---------------------------------------------------------------------------

def test_wrong_recipient_aead_authentication_fails(hybrid_kp):
    """An attacker holding a different keypair cannot decapsulate the
    capsule. The hybrid KEM produces a different derived secret →
    AEAD tag verification fails."""
    capsule = seal_capsule(
        b"sealed",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    other = generate_hybrid_kem_keypair()
    # The "embedded pubkey check" fires first because the capsule
    # carries the original recipient's pubkeys, not the attacker's.
    with pytest.raises(SealedCapsuleError, match="does not match the supplied"):
        open_capsule(capsule, other)


# ---------------------------------------------------------------------------
# Tampering
# ---------------------------------------------------------------------------

def _tamper(capsule: str, *, field: str, position: int = 5) -> str:
    """Helper: flip one bit of a base64 byte field in the capsule."""
    env = json.loads(capsule)
    if field in ("classical_ct", "pq_ct"):
        section = env["kem"]
        body = section[field]
    elif field == "ciphertext":
        body = env[field]
    else:
        raise AssertionError(f"unknown tamper field {field}")
    chars = list(body)
    chars[position] = "A" if chars[position] != "A" else "B"
    new_body = "".join(chars)
    if field == "ciphertext":
        env[field] = new_body
    else:
        env["kem"][field] = new_body
    return json.dumps(env, separators=(",", ":"))


def test_tampered_classical_ct_fails(hybrid_kp):
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    bad = _tamper(capsule, field="classical_ct")
    with pytest.raises(SealedCapsuleError):
        open_capsule(bad, hybrid_kp)


def test_tampered_pq_ct_fails(hybrid_kp):
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    bad = _tamper(capsule, field="pq_ct")
    with pytest.raises(SealedCapsuleError):
        open_capsule(bad, hybrid_kp)


def test_tampered_aead_ct_fails(hybrid_kp):
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    bad = _tamper(capsule, field="ciphertext", position=10)
    with pytest.raises(SealedCapsuleError, match="AEAD authentication"):
        open_capsule(bad, hybrid_kp)


def test_tampered_embedded_classical_pubkey_fails(hybrid_kp):
    """Swap the embedded classical pubkey for an attacker's. The
    embedded-pubkey check fires before AEAD decrypt, giving a precise
    error rather than a generic AEAD failure."""
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    env = json.loads(capsule)
    other = X25519Suite().generate_keypair()
    from kestrel_sovereign.security.multikey import public_key_to_multibase
    env["kem"]["classical_pub_multibase"] = public_key_to_multibase(
        X25519Suite(), other.public_key,
    )
    bad = json.dumps(env, separators=(",", ":"))
    with pytest.raises(SealedCapsuleError, match="classical_pub_multibase does not match"):
        open_capsule(bad, hybrid_kp)


def test_tampered_embedded_pq_pubkey_fails(hybrid_kp):
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    env = json.loads(capsule)
    other = MLKEM768Suite().generate_keypair()
    from kestrel_sovereign.security.multikey import public_key_to_multibase
    env["kem"]["pq_pub_multibase"] = public_key_to_multibase(
        MLKEM768Suite(), other.public_key,
    )
    bad = json.dumps(env, separators=(",", ":"))
    with pytest.raises(SealedCapsuleError, match="pq_pub_multibase does not match"):
        open_capsule(bad, hybrid_kp)


# ---------------------------------------------------------------------------
# Format / version
# ---------------------------------------------------------------------------

def test_unknown_format_rejected(hybrid_kp):
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    env = json.loads(capsule)
    env["format"] = "kestrel-sealed-capsule-v999"
    bad = json.dumps(env)
    with pytest.raises(SealedCapsuleError, match="unknown capsule format"):
        open_capsule(bad, hybrid_kp)


def test_unknown_version_rejected(hybrid_kp):
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    env = json.loads(capsule)
    env["version"] = 99
    bad = json.dumps(env)
    with pytest.raises(SealedCapsuleError, match="unknown capsule version"):
        open_capsule(bad, hybrid_kp)


def test_format_version_aad_binds_aead(hybrid_kp):
    """Cross-version replay defense: the AEAD's AAD includes the
    capsule's format+version. If we patched the envelope to claim a
    different version while preserving the ciphertext, AEAD
    authentication would fail. (We test the version-rejection path
    above; this confirms the AAD binding is real by constructing a
    "v1 ct sealed under v2 AAD" mismatch.)
    """
    from kestrel_sdk.security.aead import AEADCipher
    from kestrel_sovereign.security.hybrid_kem import (
        DEFAULT_DERIVED_SECRET_BYTES, encapsulate_hybrid,
    )
    from kestrel_sovereign.security.sealed_capsule import _capsule_aad
    # Seal manually with a version=2 AAD even though the format claims v1
    hybrid_ct, derived_key = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
        out_len=DEFAULT_DERIVED_SECRET_BYTES,
    )
    aead = AEADCipher(derived_key)
    # Encrypt with WRONG AAD (v2)
    bogus_token = aead.encrypt(b"x", aad=_capsule_aad("kestrel-sealed-capsule-v1", 2))

    from kestrel_sovereign.security.multikey import public_key_to_multibase
    import base64

    def b64(x): return base64.urlsafe_b64encode(x).decode("ascii").rstrip("=")
    env = {
        "format": "kestrel-sealed-capsule-v1",
        "version": 1,  # but AAD'd as v2 above
        "kem": {
            "classical_alg": "x25519",
            "pq_alg": "ml-kem-768",
            "classical_ct": b64(hybrid_ct.classical_ct),
            "pq_ct": b64(hybrid_ct.pq_ct),
            "classical_pub_multibase": public_key_to_multibase(
                X25519Suite(), hybrid_kp.classical.public_key,
            ),
            "pq_pub_multibase": public_key_to_multibase(
                MLKEM768Suite(), hybrid_kp.pq.public_key,
            ),
        },
        "ciphertext": bogus_token.decode("ascii"),
    }
    bad = json.dumps(env, separators=(",", ":"))
    with pytest.raises(SealedCapsuleError, match="AEAD authentication"):
        open_capsule(bad, hybrid_kp)


# ---------------------------------------------------------------------------
# Algorithm-pair mismatch
# ---------------------------------------------------------------------------

def test_classical_alg_mismatch_with_keypair_fails(hybrid_kp):
    """Capsule claims ``classical_alg = something-else`` but caller
    passes an x25519 keypair → precise error, no silent continuation."""
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    env = json.loads(capsule)
    env["kem"]["classical_alg"] = "fake-classical-alg"
    bad = json.dumps(env)
    with pytest.raises(SealedCapsuleError, match="capsule classical_alg"):
        open_capsule(bad, hybrid_kp)


def test_pq_alg_mismatch_with_keypair_fails(hybrid_kp):
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    env = json.loads(capsule)
    env["kem"]["pq_alg"] = "fake-pq-alg"
    bad = json.dumps(env)
    with pytest.raises(SealedCapsuleError, match="capsule pq_alg"):
        open_capsule(bad, hybrid_kp)


# ---------------------------------------------------------------------------
# Malformed envelope
# ---------------------------------------------------------------------------

def test_non_string_capsule_rejected(hybrid_kp):
    with pytest.raises(SealedCapsuleError, match="must be a JSON string"):
        open_capsule(b"not-a-string", hybrid_kp)  # type: ignore[arg-type]


def test_invalid_json_rejected(hybrid_kp):
    with pytest.raises(SealedCapsuleError, match="JSON parse failed"):
        open_capsule("{not-json", hybrid_kp)


def test_non_object_json_rejected(hybrid_kp):
    with pytest.raises(SealedCapsuleError, match="must be a JSON object"):
        open_capsule('["array", "not", "object"]', hybrid_kp)


def test_missing_kem_section_rejected(hybrid_kp):
    bad = json.dumps({
        "format": CAPSULE_FORMAT_ID,
        "version": CAPSULE_FORMAT_VERSION,
        "ciphertext": "x",
    })
    with pytest.raises(SealedCapsuleError, match="missing 'kem'"):
        open_capsule(bad, hybrid_kp)


def test_missing_kem_field_rejected(hybrid_kp):
    bad = json.dumps({
        "format": CAPSULE_FORMAT_ID,
        "version": CAPSULE_FORMAT_VERSION,
        "kem": {"classical_alg": "x25519"},  # missing everything else
        "ciphertext": "x",
    })
    with pytest.raises(SealedCapsuleError, match="kem missing fields"):
        open_capsule(bad, hybrid_kp)


def test_truncated_kem_ciphertext_surfaces_as_sealed_capsule_error(hybrid_kp):
    """Codex P2: a malformed KEM ciphertext (e.g. truncated to wrong
    length) used to leak KEMSuiteError past the capsule API boundary.
    Now wrapped into SealedCapsuleError so callers have a single
    error-type contract for any attacker-controlled capsule input."""
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    env = json.loads(capsule)
    # Truncate the classical_ct to half its length — wrong length will
    # be rejected at the KEM layer
    import base64
    raw = base64.urlsafe_b64decode(env["kem"]["classical_ct"] + "==")
    env["kem"]["classical_ct"] = base64.urlsafe_b64encode(raw[:16]).decode().rstrip("=")
    bad = json.dumps(env, separators=(",", ":"))
    with pytest.raises(SealedCapsuleError, match="KEM decapsulation failed"):
        open_capsule(bad, hybrid_kp)


def test_open_rejects_both_calling_conventions_at_once(hybrid_kp):
    capsule = seal_capsule(
        b"x",
        recipient_classical_public_key=hybrid_kp.classical.public_key,
        recipient_pq_public_key=hybrid_kp.pq.public_key,
    )
    with pytest.raises(SealedCapsuleError, match="either a HybridKEMKeypair OR"):
        open_capsule(capsule, hybrid_kp, hybrid_kp.pq)
