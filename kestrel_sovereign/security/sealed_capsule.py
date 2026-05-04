"""
Sealed capsule — Wave 4 sub-PR 3 of Quantum Hardening (#921, #919).

A **sealed capsule** wraps arbitrary payload bytes (e.g. an
identity-package JSON, a CAR archive, a backup tarball) for a specific
recipient using:

1. Hybrid KEM (X25519 + ML-KEM-768) to derive a 32-byte AES key (#974)
2. AES-256-GCM (Wave 0C's :class:`AEADCipher`) to encrypt the payload
3. AAD binding to the capsule version + format identifier so a future
   v2 capsule format cannot replay-confuse a v1 verifier

Wire format (JSON, base64-url for byte fields)
----------------------------------------------

::

    {
      "format": "kestrel-sealed-capsule-v1",
      "version": 1,
      "kem": {
        "classical_alg": "x25519",
        "pq_alg": "ml-kem-768",
        "classical_ct": "<base64>",
        "pq_ct": "<base64>",
        "classical_pub_multibase": "zXyz...",
        "pq_pub_multibase": "z..."
      },
      "ciphertext": "KSAv2:<base64>"
    }

The recipient's public keys are embedded in W3C Multikey form so a
consumer can verify the capsule was sealed for their identity without
an out-of-band lookup. The hybrid KEM's HKDF salt already includes
both ciphertexts AND both pubkeys, so a tampered embedded pubkey
results in a different derived secret → AEAD authentication failure.

Threat model
------------

- HNDL adversary captures the capsule today, decrypts later: blocked
  by the hybrid KEM. They must break BOTH X25519 (Shor) AND ML-KEM-768
  (Module-LWE break) to derive the AES key.
- Tampering with the AEAD ciphertext: blocked by AES-256-GCM tag.
- Tampering with the KEM ciphertexts: changes the HKDF transcript
  salt, derives a different AES key, AEAD authentication fails.
- Tampering with the embedded pubkeys: same — different transcript,
  different key, AEAD fails.
- Replay across capsule versions: ``format`` + ``version`` are AAD'd
  into the AEAD so a v2 capsule cannot be misinterpreted as v1.

What this module DOES NOT do
----------------------------

- Sender authentication. A capsule is a one-shot encrypted shipment;
  it doesn't prove who sealed it. Pair with a signed identity-package
  envelope (Wave 1+ identity-package v2 ``signatures`` array) when
  sender authentication is needed.
- Multi-recipient. Each capsule binds to one recipient's hybrid
  keypair. Multi-recipient is a separate epic.
- Streaming. The whole payload is encrypted in one shot. Caller
  chunks if needed.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Optional

from kestrel_sdk.security.aead import AEADCipher, DecryptionError
from kestrel_sovereign.security.hybrid_kem import (
    DEFAULT_DERIVED_SECRET_BYTES,
    HybridKEMCiphertext,
    HybridKEMKeypair,
    decapsulate_hybrid,
    encapsulate_hybrid,
)
from kestrel_sovereign.security.kem_suite import (
    ALG_ML_KEM_768,
    ALG_X25519,
    KEMSuite,
    KEMSuiteError,
    get_kem_suite,
)
from kestrel_sovereign.security.multikey import (
    multibase_to_kem_public_key,
    public_key_to_multibase,
)


CAPSULE_FORMAT_ID = "kestrel-sealed-capsule-v1"
CAPSULE_FORMAT_VERSION = 1


class SealedCapsuleError(Exception):
    """Raised on capsule format errors, malformed envelopes, or
    decapsulation/decrypt failures."""


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(s: str) -> bytes:
    """Tolerant decode: re-pads strings stripped of `=` padding."""
    if not isinstance(s, str):
        raise SealedCapsuleError(f"expected base64 string; got {type(s).__name__}")
    pad = "=" * ((4 - len(s) % 4) % 4)
    try:
        return base64.urlsafe_b64decode(s + pad)
    except Exception as e:
        raise SealedCapsuleError(f"capsule base64 decode failed: {e}") from e


def _capsule_aad(format_id: str, version: int) -> bytes:
    """AAD bound into the AEAD layer.

    Codex-anticipated guard: prevents a future v2 capsule body from
    being replayed under a v1 envelope (or vice versa). Format-id and
    version go in cleartext in the JSON header; binding them here means
    the verifier can't be tricked by header-swap.
    """
    return f"{format_id}:{version}".encode("utf-8")


# ---------------------------------------------------------------------------
# Seal
# ---------------------------------------------------------------------------

def seal_capsule(
    payload: bytes,
    *,
    recipient_classical_public_key: Any,
    recipient_pq_public_key: Any,
    classical_alg: str = ALG_X25519,
    pq_alg: str = ALG_ML_KEM_768,
) -> str:
    """Seal ``payload`` for a recipient identified by their hybrid public keys.

    Args:
        payload: arbitrary bytes to encrypt.
        recipient_classical_public_key: e.g. an ``X25519PublicKey``.
        recipient_pq_public_key: raw 1184-byte ML-KEM-768 public key.
        classical_alg / pq_alg: override the default hybrid suite pair.

    Returns:
        JSON-serialized capsule string (UTF-8). Caller can transport
        it through any text-safe channel; bytes are base64-url
        unpadded inside.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise SealedCapsuleError(
            f"payload must be bytes; got {type(payload).__name__}"
        )

    classical_suite: KEMSuite = get_kem_suite(classical_alg)
    pq_suite: KEMSuite = get_kem_suite(pq_alg)

    # 1) Hybrid KEM → 32-byte derived secret (the AES-256 key)
    hybrid_ct, derived_key = encapsulate_hybrid(
        recipient_classical_public_key,
        recipient_pq_public_key,
        classical_alg=classical_alg,
        pq_alg=pq_alg,
        out_len=DEFAULT_DERIVED_SECRET_BYTES,
    )

    # 2) AES-256-GCM the payload, AAD-binding capsule format+version
    aead = AEADCipher(derived_key)
    aead_token = aead.encrypt(
        bytes(payload),
        aad=_capsule_aad(CAPSULE_FORMAT_ID, CAPSULE_FORMAT_VERSION),
    )

    # 3) Embed recipient pubkeys as Multikey for self-describing wire form
    classical_pub_mb = public_key_to_multibase(
        classical_suite, recipient_classical_public_key,
    )
    pq_pub_mb = public_key_to_multibase(
        pq_suite, recipient_pq_public_key,
    )

    envelope = {
        "format": CAPSULE_FORMAT_ID,
        "version": CAPSULE_FORMAT_VERSION,
        "kem": {
            "classical_alg": classical_alg,
            "pq_alg": pq_alg,
            "classical_ct": _b64_encode(hybrid_ct.classical_ct),
            "pq_ct": _b64_encode(hybrid_ct.pq_ct),
            "classical_pub_multibase": classical_pub_mb,
            "pq_pub_multibase": pq_pub_mb,
        },
        "ciphertext": aead_token.decode("ascii"),
    }
    return json.dumps(envelope, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Open
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ParsedCapsule:
    classical_alg: str
    pq_alg: str
    classical_ct: bytes
    pq_ct: bytes
    classical_pub_multibase: str
    pq_pub_multibase: str
    aead_token: bytes


def _parse_capsule(envelope_str: str) -> _ParsedCapsule:
    if not isinstance(envelope_str, str):
        raise SealedCapsuleError(
            f"capsule must be a JSON string; got {type(envelope_str).__name__}"
        )
    try:
        env = json.loads(envelope_str)
    except json.JSONDecodeError as e:
        raise SealedCapsuleError(f"capsule JSON parse failed: {e}") from e
    if not isinstance(env, dict):
        raise SealedCapsuleError(
            f"capsule must be a JSON object; got {type(env).__name__}"
        )

    fmt = env.get("format")
    ver = env.get("version")
    if fmt != CAPSULE_FORMAT_ID:
        raise SealedCapsuleError(
            f"unknown capsule format {fmt!r}; this build handles only "
            f"{CAPSULE_FORMAT_ID!r}"
        )
    if ver != CAPSULE_FORMAT_VERSION:
        raise SealedCapsuleError(
            f"unknown capsule version {ver!r}; this build handles only "
            f"v{CAPSULE_FORMAT_VERSION}"
        )

    kem = env.get("kem")
    if not isinstance(kem, dict):
        raise SealedCapsuleError("capsule missing 'kem' object")

    required = {
        "classical_alg", "pq_alg",
        "classical_ct", "pq_ct",
        "classical_pub_multibase", "pq_pub_multibase",
    }
    missing = required - set(kem.keys())
    if missing:
        raise SealedCapsuleError(f"capsule kem missing fields: {sorted(missing)}")

    aead_token_str = env.get("ciphertext")
    if not isinstance(aead_token_str, str):
        raise SealedCapsuleError("capsule missing 'ciphertext' string")

    return _ParsedCapsule(
        classical_alg=kem["classical_alg"],
        pq_alg=kem["pq_alg"],
        classical_ct=_b64_decode(kem["classical_ct"]),
        pq_ct=_b64_decode(kem["pq_ct"]),
        classical_pub_multibase=kem["classical_pub_multibase"],
        pq_pub_multibase=kem["pq_pub_multibase"],
        aead_token=aead_token_str.encode("ascii"),
    )


def open_capsule(
    capsule: str,
    classical_keypair: HybridKEMKeypair | Any,
    pq_keypair: Optional[Any] = None,
) -> bytes:
    """Open a sealed capsule.

    Two calling conventions, in order of recommendation:

    1. ``open_capsule(capsule, hybrid_keypair)`` — pass a
       :class:`HybridKEMKeypair`; the function pulls both halves out.
    2. ``open_capsule(capsule, classical_keypair, pq_keypair)`` — pass
       the two ``KEMKeypair`` halves separately.

    Returns the original payload bytes.

    Raises :class:`SealedCapsuleError` on malformed envelopes, version
    mismatch, KEM mismatch, or AEAD authentication failure (which
    surfaces every tampering mode under one error type).
    """
    if isinstance(classical_keypair, HybridKEMKeypair):
        if pq_keypair is not None:
            raise SealedCapsuleError(
                "pass either a HybridKEMKeypair OR (classical_kp, pq_kp), not both"
            )
        hybrid = classical_keypair
        classical_kp = hybrid.classical
        pq_kp = hybrid.pq
    else:
        if pq_keypair is None:
            raise SealedCapsuleError(
                "pq_keypair is required when classical_keypair is not a HybridKEMKeypair"
            )
        classical_kp = classical_keypair
        pq_kp = pq_keypair

    parsed = _parse_capsule(capsule)

    # Algorithm-pair sanity: the capsule's claimed algs must match the
    # caller's keypair suite ids. A mismatch means either a wrong
    # keypair or a tampered envelope; refuse rather than silently
    # produce a different KDF output.
    if parsed.classical_alg != classical_kp.suite_id:
        raise SealedCapsuleError(
            f"capsule classical_alg={parsed.classical_alg!r} but "
            f"classical_keypair.suite_id={classical_kp.suite_id!r}"
        )
    if parsed.pq_alg != pq_kp.suite_id:
        raise SealedCapsuleError(
            f"capsule pq_alg={parsed.pq_alg!r} but "
            f"pq_keypair.suite_id={pq_kp.suite_id!r}"
        )

    # Embedded-pubkey check: the caller's keypair public keys must
    # match the multibase strings in the envelope. A mismatch could
    # only succeed if the AEAD auth tag also passed (the salt is bound)
    # but failing fast here gives a precise error message instead of
    # a generic "decryption failed".
    classical_suite = get_kem_suite(parsed.classical_alg)
    pq_suite = get_kem_suite(parsed.pq_alg)
    expected_classical_mb = public_key_to_multibase(
        classical_suite, classical_kp.public_key,
    )
    expected_pq_mb = public_key_to_multibase(
        pq_suite, pq_kp.public_key,
    )
    if parsed.classical_pub_multibase != expected_classical_mb:
        raise SealedCapsuleError(
            "capsule classical_pub_multibase does not match the supplied "
            "classical keypair's public key"
        )
    if parsed.pq_pub_multibase != expected_pq_mb:
        raise SealedCapsuleError(
            "capsule pq_pub_multibase does not match the supplied "
            "pq keypair's public key"
        )

    # Hybrid decapsulate → 32-byte derived secret. Wrap KEM-layer
    # exceptions in SealedCapsuleError so the public API has a single
    # error type for callers (codex P2 review): a truncated
    # ``classical_ct``, a malformed X25519 ephemeral pubkey, or a
    # length-mismatched ML-KEM ciphertext used to leak KEMSuiteError
    # past the capsule API boundary.
    hybrid_ct = HybridKEMCiphertext(
        classical_ct=parsed.classical_ct,
        pq_ct=parsed.pq_ct,
    )
    try:
        derived_key = decapsulate_hybrid(
            hybrid_ct, classical_kp, pq_kp,
            out_len=DEFAULT_DERIVED_SECRET_BYTES,
        )
    except KEMSuiteError as e:
        raise SealedCapsuleError(
            f"capsule KEM decapsulation failed: {e}"
        ) from e

    # AEAD-decrypt with the same AAD binding. AES-GCM authentication
    # failure surfaces all tamper modes (KEM ct, PQ ct, AEAD ct).
    aead = AEADCipher(derived_key)
    try:
        return aead.decrypt(
            parsed.aead_token,
            aad=_capsule_aad(CAPSULE_FORMAT_ID, CAPSULE_FORMAT_VERSION),
        )
    except DecryptionError as e:
        raise SealedCapsuleError(f"capsule AEAD authentication failed: {e}") from e


__all__ = [
    "CAPSULE_FORMAT_ID",
    "CAPSULE_FORMAT_VERSION",
    "SealedCapsuleError",
    "open_capsule",
    "seal_capsule",
]
