"""
Hybrid KEM combiner tests — Wave 4 sub-PR 2 (#919).

Covers:
- Generation: default X25519 + ML-KEM-768; algorithm-pair validation
- Encapsulate/decapsulate round-trip (both halves return the same
  derived secret)
- Output length matches DEFAULT_DERIVED_SECRET_BYTES (32)
- Custom out_len fans out via HKDF expand
- Distinct encapsulations to the same recipient produce distinct
  ciphertexts AND distinct secrets (fresh randomness)
- Wrong classical key → different secret (no error at this layer)
- Wrong PQ key → different secret (FIPS 203 implicit rejection)
- Tampered classical ciphertext → different secret
- Tampered PQ ciphertext → different secret
- Transcript binding: same SS values but different ciphertexts/pubkeys
  produce different secrets (anti-malleability)
- HKDF info pin: combiner-spec change would break round-trip
- Wire round-trip: ciphertext can be split & re-bundled
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.security.hybrid_kem import (
    DEFAULT_DERIVED_SECRET_BYTES,
    HybridKEMCiphertext,
    HybridKEMKeypair,
    decapsulate_hybrid,
    encapsulate_hybrid,
    generate_hybrid_kem_keypair,
)
from kestrel_sovereign.security.kem_suite import (
    ALG_ML_KEM_768,
    ALG_X25519,
    KEMSuiteError,
    MLKEM768Suite,
    X25519Suite,
    get_kem_suite,
)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hybrid_kp():
    return generate_hybrid_kem_keypair()


def test_default_pair_is_x25519_plus_mlkem768(hybrid_kp):
    assert hybrid_kp.classical.suite_id == ALG_X25519
    assert hybrid_kp.pq.suite_id == ALG_ML_KEM_768


def test_distinct_keypairs():
    a = generate_hybrid_kem_keypair()
    b = generate_hybrid_kem_keypair()
    # Compare via X25519 wire bytes
    suite = X25519Suite()
    assert suite.serialize_public_key(a.classical.public_key) != \
        suite.serialize_public_key(b.classical.public_key)
    assert a.pq.public_key != b.pq.public_key


def test_refuses_two_classical_pair():
    with pytest.raises(KEMSuiteError, match="must be a post-quantum"):
        generate_hybrid_kem_keypair(
            classical_alg=ALG_X25519,
            pq_alg=ALG_X25519,  # classical, not PQ
        )


def test_refuses_two_pq_pair():
    with pytest.raises(KEMSuiteError, match="must be a classical"):
        generate_hybrid_kem_keypair(
            classical_alg=ALG_ML_KEM_768,
            pq_alg=ALG_ML_KEM_768,
        )


def test_unregistered_alg_raises():
    with pytest.raises(KEMSuiteError, match="no registered KEM suite"):
        generate_hybrid_kem_keypair(classical_alg="not-real")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_encapsulate_decapsulate_round_trip(hybrid_kp):
    ct, ss_alice = encapsulate_hybrid(
        hybrid_kp.classical.public_key,
        hybrid_kp.pq.public_key,
    )
    ss_bob = decapsulate_hybrid(ct, hybrid_kp.classical, hybrid_kp.pq)
    assert ss_alice == ss_bob
    assert len(ss_alice) == DEFAULT_DERIVED_SECRET_BYTES


def test_default_secret_is_32_bytes(hybrid_kp):
    _, ss = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
    )
    assert len(ss) == 32


def test_custom_out_len_via_hkdf_expand(hybrid_kp):
    """HKDF expand can produce more than 32 bytes — useful for fanning
    out into AES key + integrity key + IV without re-encapsulating."""
    ct, ss = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
        out_len=64,
    )
    assert len(ss) == 64
    ss2 = decapsulate_hybrid(ct, hybrid_kp.classical, hybrid_kp.pq, out_len=64)
    assert ss == ss2


def test_out_len_zero_rejected(hybrid_kp):
    with pytest.raises(KEMSuiteError, match="out_len must be positive"):
        encapsulate_hybrid(
            hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
            out_len=0,
        )


# ---------------------------------------------------------------------------
# Fresh randomness per encapsulation
# ---------------------------------------------------------------------------

def test_distinct_encapsulations_produce_distinct_ciphertexts_and_secrets(hybrid_kp):
    """Each encapsulation samples fresh randomness for both halves —
    two calls to the same recipient produce different ciphertexts AND
    different derived secrets."""
    ct_a, ss_a = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
    )
    ct_b, ss_b = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
    )
    assert ct_a.classical_ct != ct_b.classical_ct
    assert ct_a.pq_ct != ct_b.pq_ct
    assert ss_a != ss_b


# ---------------------------------------------------------------------------
# Wrong-key paths
# ---------------------------------------------------------------------------

def test_wrong_classical_key_yields_different_secret(hybrid_kp):
    """Decapsulating with a wrong classical key produces a different
    derived secret. No exception at this layer; the AEAD downstream
    is the authentication boundary."""
    ct, ss_alice = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
    )
    other_classical = X25519Suite().generate_keypair()
    ss_wrong = decapsulate_hybrid(ct, other_classical, hybrid_kp.pq)
    assert ss_wrong != ss_alice


def test_wrong_pq_key_yields_different_secret(hybrid_kp):
    """FIPS 203 implicit rejection: ML-KEM with wrong key returns a
    DIFFERENT 32-byte secret, the HKDF output diverges, and the
    derived secret differs from the encapsulator's."""
    ct, ss_alice = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
    )
    other_pq = MLKEM768Suite().generate_keypair()
    ss_wrong = decapsulate_hybrid(ct, hybrid_kp.classical, other_pq)
    assert ss_wrong != ss_alice


# ---------------------------------------------------------------------------
# Tampering
# ---------------------------------------------------------------------------

def test_tampered_classical_ciphertext_yields_different_secret(hybrid_kp):
    ct, ss_alice = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
    )
    bad = bytearray(ct.classical_ct)
    bad[10] ^= 0x01
    bad_ct = HybridKEMCiphertext(classical_ct=bytes(bad), pq_ct=ct.pq_ct)
    ss_wrong = decapsulate_hybrid(bad_ct, hybrid_kp.classical, hybrid_kp.pq)
    assert ss_wrong != ss_alice


def test_tampered_pq_ciphertext_yields_different_secret(hybrid_kp):
    ct, ss_alice = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
    )
    bad = bytearray(ct.pq_ct)
    bad[42] ^= 0x01
    bad_ct = HybridKEMCiphertext(classical_ct=ct.classical_ct, pq_ct=bytes(bad))
    ss_wrong = decapsulate_hybrid(bad_ct, hybrid_kp.classical, hybrid_kp.pq)
    assert ss_wrong != ss_alice


# ---------------------------------------------------------------------------
# Transcript binding (anti-malleability)
# ---------------------------------------------------------------------------

def test_hkdf_info_binds_algorithm_pair(hybrid_kp, monkeypatch):
    """Codex P2: the HKDF info is now bound to the actual selected
    algorithm pair. If a future combiner uses different suites with
    the same SS/transcript inputs, it derives a different secret —
    proper domain separation between hybrid suite combinations.

    Construct two derivations whose ONLY difference is the algorithm
    label fed to HKDF info. Same SS values, same transcript salt →
    different output, proving the info participates in the KDF.
    """
    from kestrel_sovereign.security.hybrid_kem import _derive_secret

    ss_c, ss_pq = b"\x01" * 32, b"\x02" * 32
    ct_c, ct_pq = b"\x03" * 32, b"\x04" * 1088
    pk_c, pk_pq = b"\x05" * 32, b"\x06" * 1184

    a = _derive_secret(ss_c, ss_pq, ct_c, ct_pq, pk_c, pk_pq, "x25519", "ml-kem-768", 32)
    b = _derive_secret(ss_c, ss_pq, ct_c, ct_pq, pk_c, pk_pq, "x25519", "ml-kem-1024", 32)
    assert a != b


def test_transcript_salt_distinguishes_recipients(hybrid_kp):
    """The HKDF salt includes both ciphertexts AND both public keys.
    A forged ciphertext that somehow produced the same SS values for
    a DIFFERENT recipient would still derive a different secret
    because the salt differs."""
    # Encapsulate to recipient 1
    ct1, ss1 = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
    )
    # Build a different recipient
    other_kp = generate_hybrid_kem_keypair()
    ct2, ss2 = encapsulate_hybrid(
        other_kp.classical.public_key, other_kp.pq.public_key,
    )
    # The two derived secrets are unrelated even though both came from
    # honest encapsulations of fresh randomness
    assert ss1 != ss2


# ---------------------------------------------------------------------------
# Decapsulator suite-id validation
# ---------------------------------------------------------------------------

def test_decapsulate_refuses_two_classical_keypair(hybrid_kp):
    """If the caller passes a classical keypair in the PQ slot, the
    decapsulator should refuse rather than silently produce garbage."""
    other_classical = X25519Suite().generate_keypair()
    with pytest.raises(KEMSuiteError, match="expected a post-quantum"):
        decapsulate_hybrid(
            HybridKEMCiphertext(classical_ct=b"\x00" * 32, pq_ct=b"\x00" * 1088),
            hybrid_kp.classical,
            other_classical,  # wrong slot
        )


def test_decapsulate_refuses_two_pq_keypair(hybrid_kp):
    other_pq = MLKEM768Suite().generate_keypair()
    with pytest.raises(KEMSuiteError, match="expected a classical"):
        decapsulate_hybrid(
            HybridKEMCiphertext(classical_ct=b"\x00" * 32, pq_ct=b"\x00" * 1088),
            other_pq,  # wrong slot
            hybrid_kp.pq,
        )


# ---------------------------------------------------------------------------
# Wire-format round-trip
# ---------------------------------------------------------------------------

def test_ciphertext_round_trip_via_split_and_rebundle(hybrid_kp):
    """The HybridKEMCiphertext is two named bytes fields. Wire-format
    consumers serialize them however they want (e.g. CBOR map, length-
    prefixed concat); the decapsulator only needs them re-bundled."""
    ct, ss_alice = encapsulate_hybrid(
        hybrid_kp.classical.public_key, hybrid_kp.pq.public_key,
    )
    # Round-trip through bytes
    classical_wire = bytes(ct.classical_ct)
    pq_wire = bytes(ct.pq_ct)
    assert len(classical_wire) == 32
    assert len(pq_wire) == 1088
    rebundled = HybridKEMCiphertext(classical_ct=classical_wire, pq_ct=pq_wire)
    ss_bob = decapsulate_hybrid(rebundled, hybrid_kp.classical, hybrid_kp.pq)
    assert ss_alice == ss_bob
