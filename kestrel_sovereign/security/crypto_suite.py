"""
CryptoSuite — pluggable signature suites for Kestrel.

Wave 1 of the Quantum Hardening epic (#921, issue #916). Replaces the
ad-hoc ``cryptography.hazmat.primitives.asymmetric.ec`` calls scattered
across identity, spawn-mandate, script-signing, and constitution-anchoring
sites with a single abstraction so that:

- Wave 2 can plug in ``Ed25519Suite`` and ``MLDSA65Suite`` for hybrid
  identity (classical + post-quantum in parallel) without touching the
  call sites.
- Wave 3's succession ceremonies can countersign with ``SLHDSASHA2128sSuite``.
- Wave 4 can introduce KEM suites (X25519 + ML-KEM-768) for export
  wrapping under the same registry pattern.
- The choice of post-quantum library (``pqcrypto``, ``oqs-python``, etc.)
  is reversible — multiple suites for the same algorithm can coexist
  behind the registry while we run KAT vectors.

Wave 1 ships only the abstraction and one concrete suite —
``Secp256k1Suite`` — which exactly replicates today's behavior. No
production caller is wired through it yet; that's Wave 1 sub-PR 2+.

Threat-model framing
--------------------

The abstraction itself does not strengthen any threat surface. Its value
is making the *substitution* of stronger primitives in Waves 2-4 a small,
focused diff against well-tested call sites instead of a sprawling search-
and-replace through the codebase. This is what enables ``hybrid-required``
verification policies in Wave 1's later PRs without touching every
``private_key.sign(...)`` site again.

Algorithm identifier registry
-----------------------------

Every suite has a stable ``alg_id`` string that ships in identity-package
v2 ``signatures: [{alg, kid, sig}]`` arrays and in ``verificationMethod``
type fields. Reserved identifiers:

- ``ecdsa-secp256k1-sha256`` — classical, current default (``Secp256k1Suite``)
- ``ed25519`` — Wave 2 (classical half of hybrid)
- ``ml-dsa-65`` — Wave 2 (PQ half of hybrid; NIST FIPS 204 Cat-3)
- ``slh-dsa-sha2-128s`` — Wave 3 (succession + checkpoint countersigning)
- ``ml-kem-768`` — Wave 4 (KEM, distinct interface from signing)

New IDs are added by registering a new ``CryptoSuite`` subclass; the
registry keys on the suite's ``alg_id`` class attribute.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Type


# ---------------------------------------------------------------------------
# Reserved algorithm identifiers
# ---------------------------------------------------------------------------

ALG_ECDSA_SECP256K1_SHA256 = "ecdsa-secp256k1-sha256"
ALG_ED25519 = "ed25519"
ALG_ML_DSA_65 = "ml-dsa-65"
ALG_SLH_DSA_SHA2_128S = "slh-dsa-sha2-128s"
ALG_ML_KEM_768 = "ml-kem-768"


# ---------------------------------------------------------------------------
# Keypair container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Keypair:
    """A keypair owned by a specific suite.

    ``private_key`` and ``public_key`` are opaque to callers — they hand
    them back to the suite's ``sign`` / ``verify`` methods without
    inspecting them. Concrete types vary per suite (e.g.
    ``EllipticCurvePrivateKey`` for secp256k1, raw bytes for ML-DSA).
    """
    suite_id: str
    private_key: Any
    public_key: Any


# ---------------------------------------------------------------------------
# CryptoSuite abstract base
# ---------------------------------------------------------------------------

class CryptoSuite(abc.ABC):
    """Pluggable signature suite.

    Concrete subclasses MUST set ``alg_id`` to a unique stable identifier
    (see the reserved list at module scope) and implement the abstract
    methods. KEM suites (Wave 4) extend a separate ``KEMSuite``
    interface — kept distinct from signing because the operation
    semantics differ.

    Optional class attributes:

    - ``public_key_multicodec`` — the multicodec varint bytes that
      identify this suite's public-key shape in W3C Multikey
      (``publicKeyMultibase``) strings. Required for any suite whose
      keys appear in identity-package v2 ``verificationMethods`` or
      ``did:web`` documents. If unset, the suite cannot produce a
      ``z...``-prefixed multibase string and ``multikey.public_key_to_multibase``
      raises.
    - ``is_post_quantum`` — True for lattice/hash-based PQ suites
      (ML-DSA, ML-KEM, SLH-DSA), False for classical suites
      (secp256k1, Ed25519). The Wave 1 sub-PR 4 verify-policy module
      uses this to enforce ``HYBRID_REQUIRED`` (≥1 classical + ≥1 PQ
      signature) and ``PQ_REQUIRED`` (≥1 PQ signature) without
      hardcoding alg_id lists.
    """

    alg_id: ClassVar[str]
    public_key_multicodec: ClassVar[bytes] = b""
    is_post_quantum: ClassVar[bool] = False

    @abc.abstractmethod
    def generate_keypair(self) -> Keypair:
        """Produce a fresh keypair owned by this suite."""

    @abc.abstractmethod
    def sign(self, data: bytes, private_key: Any) -> bytes:
        """Sign ``data`` with ``private_key``. Returns the raw signature.

        Raises ``CryptoSuiteError`` on failure; never returns a fallback or
        partial signature.
        """

    @abc.abstractmethod
    def verify(self, data: bytes, signature: bytes, public_key: Any) -> bool:
        """Return True iff ``signature`` is valid over ``data`` under
        ``public_key``. Returns False (not raises) on any verification
        failure — bad signature, malformed input, wrong key.
        """

    @abc.abstractmethod
    def serialize_public_key(self, public_key: Any) -> bytes:
        """Legacy on-the-wire serialization for backwards-compatible
        artifact fields (e.g. ``publicKeyHex`` in v1 DID documents).

        For ECC suites this is the uncompressed X9.62 point — same shape
        ``inception_service.public_key_to_hex`` already emits — so v1
        readers don't break.

        For W3C Multikey / ``publicKeyMultibase`` callers, use
        ``serialize_public_key_for_multikey`` instead — that emits the
        format the multicodec table specifies (e.g. compressed 33-byte
        secp256k1, raw 32-byte ed25519). Mixing the two will produce
        valid-looking but cross-implementation-incompatible identifiers.
        """

    @abc.abstractmethod
    def deserialize_public_key(self, raw: bytes) -> Any:
        """Inverse of ``serialize_public_key`` (legacy uncompressed form)."""

    def serialize_public_key_for_multikey(self, public_key: Any) -> bytes:
        """Serialize a public key in the W3C Multikey-compatible shape
        for this suite's ``public_key_multicodec``.

        The default implementation raises — suites that participate in
        Multikey / ``did:key`` / ``did:web`` verification methods MUST
        override to emit the spec-mandated format. Failing loud here
        prevents a future suite from accidentally shipping its legacy
        uncompressed bytes under multicodec 0xe7 (which the spec defines
        as 33-byte compressed) and producing identifiers that other
        implementations reject or rederive to a different key shape.
        """
        raise CryptoSuiteError(
            f"{type(self).__name__} (alg_id={self.alg_id!r}) does not "
            f"implement serialize_public_key_for_multikey. Override on "
            f"the suite class with the format mandated by its "
            f"public_key_multicodec entry in the multicodec table."
        )

    def deserialize_public_key_from_multikey(self, raw: bytes) -> Any:
        """Inverse of ``serialize_public_key_for_multikey``."""
        raise CryptoSuiteError(
            f"{type(self).__name__} (alg_id={self.alg_id!r}) does not "
            f"implement deserialize_public_key_from_multikey."
        )


class CryptoSuiteError(Exception):
    """Suite-level signing/verification failure."""


# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, CryptoSuite] = {}


def register_suite(suite: CryptoSuite) -> None:
    """Register an instance of a suite under its ``alg_id``.

    Re-registration with a different instance for the same ``alg_id``
    raises — guards against silent overwrites during library bake-off
    when multiple modules import competing implementations.
    """
    existing = _REGISTRY.get(suite.alg_id)
    if existing is not None and existing is not suite:
        raise CryptoSuiteError(
            f"Suite already registered for alg_id={suite.alg_id!r} "
            f"(existing={type(existing).__name__}, new={type(suite).__name__})."
        )
    _REGISTRY[suite.alg_id] = suite


def get_suite(alg_id: str) -> CryptoSuite:
    """Look up the suite for ``alg_id``. Raises if unregistered."""
    try:
        return _REGISTRY[alg_id]
    except KeyError as e:
        raise CryptoSuiteError(
            f"No suite registered for alg_id={alg_id!r}. "
            f"Registered: {sorted(_REGISTRY)}."
        ) from e


def list_registered() -> list[str]:
    """Return the alg_ids of all currently registered suites."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Secp256k1Suite — classical, current default
# ---------------------------------------------------------------------------

class Secp256k1Suite(CryptoSuite):
    """ECDSA over secp256k1 with SHA-256.

    Behavior-preserving wrapper around the existing
    ``cryptography.hazmat.primitives.asymmetric.ec`` calls in
    ``inception_service``, ``identity/signing``, ``spawn/mandate``, and
    ``features/compute/script_signer``. Wave 1 sub-PRs migrate those
    sites to call through this suite.

    Public-key serialization uses uncompressed X9.62 format (the same
    65-byte ``04 || X(32) || Y(32)`` shape ``inception_service`` already
    emits as ``publicKeyHex``), so it round-trips cleanly with both
    legacy and v2 identity packages.

    Threat note: secp256k1 is Shor-vulnerable. Wave 2 introduces hybrid
    Ed25519 + ML-DSA-65 for new identities; this suite continues to
    decrypt/verify legacy artifacts under the temporal-validity rules
    in ``docs/architecture/security/SERIALIZATION_COMPATIBILITY.md``.
    """

    alg_id: ClassVar[str] = ALG_ECDSA_SECP256K1_SHA256
    # Multicodec 0xe7 (secp256k1-pub), varint-encoded.
    public_key_multicodec: ClassVar[bytes] = b"\xe7\x01"

    def generate_keypair(self) -> Keypair:
        from cryptography.hazmat.primitives.asymmetric import ec
        priv = ec.generate_private_key(ec.SECP256K1())
        return Keypair(
            suite_id=self.alg_id,
            private_key=priv,
            public_key=priv.public_key(),
        )

    def sign(self, data: bytes, private_key: Any) -> bytes:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes
        try:
            return private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        except Exception as e:
            raise CryptoSuiteError(f"secp256k1 sign failed: {e}") from e

    def verify(self, data: bytes, signature: bytes, public_key: Any) -> bool:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes
        try:
            public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False

    def serialize_public_key(self, public_key: Any) -> bytes:
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat,
        )
        return public_key.public_bytes(
            encoding=Encoding.X962,
            format=PublicFormat.UncompressedPoint,
        )

    def deserialize_public_key(self, raw: bytes) -> Any:
        from cryptography.hazmat.primitives.asymmetric import ec
        try:
            return ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(), raw,
            )
        except Exception as e:
            raise CryptoSuiteError(
                f"secp256k1 public-key deserialization failed: {e}"
            ) from e

    def serialize_public_key_for_multikey(self, public_key: Any) -> bytes:
        """W3C Multikey form for secp256k1: 33-byte compressed X9.62.

        The multicodec table (https://github.com/multiformats/multicodec)
        defines 0xe7 ``secp256k1-pub`` as a 33-byte compressed point with
        a leading 0x02 or 0x03 byte indicating Y parity. did:key and the
        W3C Multikey / did-controller specs both rely on this shape;
        emitting the legacy 65-byte uncompressed point under the same
        codec produces strings other implementations reject or rederive
        to a different key.
        """
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat,
        )
        return public_key.public_bytes(
            encoding=Encoding.X962,
            format=PublicFormat.CompressedPoint,
        )

    def deserialize_public_key_from_multikey(self, raw: bytes) -> Any:
        """Inverse of ``serialize_public_key_for_multikey``.

        Accepts a 33-byte compressed X9.62 point. The underlying
        ``cryptography`` library handles compressed→uncompressed
        decompression internally.
        """
        from cryptography.hazmat.primitives.asymmetric import ec
        try:
            return ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(), raw,
            )
        except Exception as e:
            raise CryptoSuiteError(
                f"secp256k1 multikey public-key deserialization failed: {e}"
            ) from e


# Register the default suite at import time. Future suites in Waves 2-4
# register themselves the same way from their own modules.
register_suite(Secp256k1Suite())
