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


# ---------------------------------------------------------------------------
# Ed25519Suite — classical, hybrid-identity classical half (Wave 2)
# ---------------------------------------------------------------------------

class Ed25519Suite(CryptoSuite):
    """Ed25519 (RFC 8032) signing.

    Classical half of the Wave 2 hybrid-identity composite (Ed25519 +
    ML-DSA-65). Already in use by ``storage/providers/storacha_ucan.py``
    for UCAN v1 invocations; this suite registers it under the same
    abstraction so the hybrid signer in Wave 2 sub-PR 4+ can pick it up.

    Multikey shape
    --------------

    - ``alg_id``: ``"ed25519"``
    - ``public_key_multicodec``: ``b"\\xed\\x01"`` (multicodec 0xed
      = ``ed25519-pub``, varint-encoded; matches W3C did:key spec).
    - Public-key bytes: 32-byte raw — Ed25519 has no compressed/uncompressed
      distinction, so the legacy and multikey forms are identical.

    Threat note
    -----------

    Ed25519 is Shor-vulnerable like all classical signatures. Wave 2
    pairs it with ML-DSA-65 in a hybrid identity; the classical half
    provides decades-aged cryptanalysis confidence while the PQ half
    handles the future-quantum-adversary case.
    """

    alg_id: ClassVar[str] = ALG_ED25519
    # Multicodec 0xed (ed25519-pub), varint-encoded.
    public_key_multicodec: ClassVar[bytes] = b"\xed\x01"
    is_post_quantum: ClassVar[bool] = False

    def generate_keypair(self) -> Keypair:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        priv = ed25519.Ed25519PrivateKey.generate()
        return Keypair(
            suite_id=self.alg_id,
            private_key=priv,
            public_key=priv.public_key(),
        )

    def sign(self, data: bytes, private_key: Any) -> bytes:
        try:
            # Ed25519 has built-in hashing; no separate hash algorithm needed.
            return private_key.sign(data)
        except Exception as e:
            raise CryptoSuiteError(f"ed25519 sign failed: {e}") from e

    def verify(self, data: bytes, signature: bytes, public_key: Any) -> bool:
        try:
            public_key.verify(signature, data)
            return True
        except Exception:
            return False

    def serialize_public_key(self, public_key: Any) -> bytes:
        """Raw 32-byte Ed25519 public key.

        Same shape used by ``storage/providers/storacha_ucan.py`` and by
        the W3C Multikey codec — Ed25519 has only one canonical form.
        """
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat,
        )
        return public_key.public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        )

    def deserialize_public_key(self, raw: bytes) -> Any:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        try:
            return ed25519.Ed25519PublicKey.from_public_bytes(raw)
        except Exception as e:
            raise CryptoSuiteError(
                f"ed25519 public-key deserialization failed: {e}"
            ) from e

    # Multikey serialization is identical to the legacy form for Ed25519
    # (no compressed/uncompressed distinction); reuse the same methods so
    # there's a single canonical wire format.
    def serialize_public_key_for_multikey(self, public_key: Any) -> bytes:
        return self.serialize_public_key(public_key)

    def deserialize_public_key_from_multikey(self, raw: bytes) -> Any:
        return self.deserialize_public_key(raw)


# ---------------------------------------------------------------------------
# MLDSA65Suite — post-quantum, hybrid-identity PQ half (Wave 2)
# ---------------------------------------------------------------------------

class MLDSA65Suite(CryptoSuite):
    """ML-DSA-65 (NIST FIPS 204) signing.

    Post-quantum half of the Wave 2 hybrid-identity composite. NIST Cat-3
    parameter set: balanced security level (≈ AES-192 equivalent under
    classical attacks), public key 1952 bytes, signature 3309 bytes.

    Library
    -------

    Backed by ``pqcrypto.sign.ml_dsa_65`` — CFFI bindings to PQClean-
    derived C implementations, prebuilt wheels available on PyPI for
    our target platforms (macOS arm64, Linux x86_64, Windows). Selected
    per the PRD-v2 §9 bake-off criterion: "prebuilt wheels /
    no compile-on-deploy" beats FIPS-validation-status as the gating
    factor (CNSA 2.0 supports planning now without a vetted impl).

    The choice is reversible — multiple suites for the same algorithm
    can coexist behind the registry while we run KAT vectors against
    alternative implementations (oqs-python, upstream ``cryptography``
    when it ships PQ).

    Multikey shape
    --------------

    - ``alg_id``: ``"ml-dsa-65"``
    - ``public_key_multicodec``: ``b"\\x87\\x24"`` (multicodec 0x1207
      ``ml-dsa-65-pub``, varint-encoded). The W3C/IETF multicodec
      table entry is currently *proposed*; treat as experimental until
      finalized. Wire format will stay byte-stable as the spec firms up.
    - Public key bytes: 1952 raw bytes (pqcrypto returns these directly).

    Sign/verify failure semantics
    -----------------------------

    pqcrypto's ``verify`` returns ``False`` cleanly for all failure
    modes — wrong key, tampered data, malformed signature — no
    exceptions to catch. ``sign`` raises only on truly broken inputs
    (wrong-length secret key, etc.); this suite wraps those into
    ``CryptoSuiteError`` for uniform error handling.
    """

    alg_id: ClassVar[str] = ALG_ML_DSA_65
    # Multicodec 0x1207 (ml-dsa-65-pub, proposed), varint-encoded.
    # See https://github.com/multiformats/multicodec for the registry.
    public_key_multicodec: ClassVar[bytes] = b"\x87\x24"
    is_post_quantum: ClassVar[bool] = True

    # NIST FIPS 204 Cat-3 sizes — pinned as class attributes so callers
    # can size buffers without importing pqcrypto themselves.
    PUBLIC_KEY_SIZE: ClassVar[int] = 1952
    SECRET_KEY_SIZE: ClassVar[int] = 4032
    SIGNATURE_SIZE: ClassVar[int] = 3309

    def generate_keypair(self) -> Keypair:
        from pqcrypto.sign import ml_dsa_65
        # pqcrypto returns (public, secret) — note the order
        public_bytes, secret_bytes = ml_dsa_65.generate_keypair()
        return Keypair(
            suite_id=self.alg_id,
            private_key=secret_bytes,
            public_key=public_bytes,
        )

    def sign(self, data: bytes, private_key: Any) -> bytes:
        from pqcrypto.sign import ml_dsa_65
        if not isinstance(private_key, (bytes, bytearray)):
            raise CryptoSuiteError(
                f"ml-dsa-65 private_key must be bytes ({self.SECRET_KEY_SIZE} "
                f"bytes); got {type(private_key).__name__}"
            )
        try:
            return ml_dsa_65.sign(bytes(private_key), data)
        except Exception as e:
            raise CryptoSuiteError(f"ml-dsa-65 sign failed: {e}") from e

    def verify(self, data: bytes, signature: bytes, public_key: Any) -> bool:
        from pqcrypto.sign import ml_dsa_65
        if not isinstance(public_key, (bytes, bytearray)):
            return False
        if not isinstance(signature, (bytes, bytearray)):
            return False
        try:
            return bool(ml_dsa_65.verify(bytes(public_key), data, bytes(signature)))
        except Exception:
            return False

    def serialize_public_key(self, public_key: Any) -> bytes:
        """Raw 1952-byte ML-DSA-65 public key.

        pqcrypto already returns the public key as raw bytes, so this is
        an identity cast (with type validation).
        """
        if not isinstance(public_key, (bytes, bytearray)):
            raise CryptoSuiteError(
                f"ml-dsa-65 public_key must be bytes; got "
                f"{type(public_key).__name__}"
            )
        return bytes(public_key)

    def deserialize_public_key(self, raw: bytes) -> Any:
        if not isinstance(raw, (bytes, bytearray)):
            raise CryptoSuiteError(
                f"ml-dsa-65 raw public key must be bytes; got "
                f"{type(raw).__name__}"
            )
        if len(raw) != self.PUBLIC_KEY_SIZE:
            raise CryptoSuiteError(
                f"ml-dsa-65 public key must be {self.PUBLIC_KEY_SIZE} bytes; "
                f"got {len(raw)}"
            )
        return bytes(raw)

    # Multikey serialization is identical to the legacy form — ML-DSA-65
    # has only one canonical wire representation. Single canonical form
    # = no chance of cross-implementation incompatibility under the
    # multicodec.
    def serialize_public_key_for_multikey(self, public_key: Any) -> bytes:
        return self.serialize_public_key(public_key)

    def deserialize_public_key_from_multikey(self, raw: bytes) -> Any:
        return self.deserialize_public_key(raw)


# ---------------------------------------------------------------------------
# SLHDSASHA2128sSuite — hash-based PQ, succession + checkpoint signing (Wave 3)
# ---------------------------------------------------------------------------

class SLHDSASHA2128sSuite(CryptoSuite):
    """SLH-DSA-SHA2-128s (NIST FIPS 205) signing.

    Conservative-tier post-quantum signature: security relies only on
    the cryptographic hash function (SHA-2 here), so the only quantum
    speedup is Grover's algorithm — which halves preimage security
    (256-bit → ~128-bit) but does NOT break it the way Shor breaks
    ECC and RSA. This is the most defensible long-horizon choice for
    irrevocable, hand-signed events: succession statements, checkpoint
    rotations, release signatures.

    Trade-off: signatures are **7856 bytes** — ~2× ML-DSA-65 and ~120×
    Ed25519. Public keys are tiny (32 bytes) but signature size is the
    cost we pay for the hash-only security argument. SLH-DSA is
    therefore reserved for **infrequent, long-lived** artifacts; the
    high-throughput signing path remains hybrid (Ed25519 + ML-DSA-65).

    Library
    -------

    Backed by ``pqcrypto.sign.sphincs_sha2_128s_simple`` — the FIPS 205
    "SLH-DSA-SHA2-128s" parameter set is the same as the
    ``sphincs+-sha2-128s-simple`` algorithm in PQClean's lineage.
    pqcrypto exposes only the ``_simple`` variants (the FIPS-205-aligned
    set), not the original SPHINCS+ "robust" variants. Same library
    family as ML-DSA-65 (#950), no compile-on-deploy.

    Multikey shape
    --------------

    - ``alg_id``: ``"slh-dsa-sha2-128s"``
    - ``public_key_multicodec``: ``b"\\x88\\x24"`` (multicodec 0x1208,
      varint-encoded). The W3C/IETF multicodec table entry is currently
      *proposed*; treat as experimental until finalized.
    - Public-key bytes: 32 raw bytes.

    Sign/verify failure semantics
    -----------------------------

    Mirrors :class:`MLDSA65Suite`: pqcrypto's verify returns False
    cleanly for any failure mode; sign raises only on broken inputs
    (wrong-length secret key, etc.) and we wrap those into
    :class:`CryptoSuiteError`.
    """

    alg_id: ClassVar[str] = ALG_SLH_DSA_SHA2_128S
    # Multicodec 0x1208 (slh-dsa-sha2-128s-pub, proposed), varint-encoded.
    public_key_multicodec: ClassVar[bytes] = b"\x88\x24"
    is_post_quantum: ClassVar[bool] = True

    # NIST FIPS 205 SLH-DSA-SHA2-128s sizes — pinned as class attributes
    # so callers can size buffers without importing pqcrypto themselves.
    PUBLIC_KEY_SIZE: ClassVar[int] = 32
    SECRET_KEY_SIZE: ClassVar[int] = 64
    SIGNATURE_SIZE: ClassVar[int] = 7856

    def generate_keypair(self) -> Keypair:
        from pqcrypto.sign import sphincs_sha2_128s_simple as slh
        public_bytes, secret_bytes = slh.generate_keypair()
        return Keypair(
            suite_id=self.alg_id,
            private_key=secret_bytes,
            public_key=public_bytes,
        )

    def sign(self, data: bytes, private_key: Any) -> bytes:
        from pqcrypto.sign import sphincs_sha2_128s_simple as slh
        if not isinstance(private_key, (bytes, bytearray)):
            raise CryptoSuiteError(
                f"slh-dsa-sha2-128s private_key must be bytes "
                f"({self.SECRET_KEY_SIZE} bytes); got "
                f"{type(private_key).__name__}"
            )
        try:
            return slh.sign(bytes(private_key), data)
        except Exception as e:
            raise CryptoSuiteError(f"slh-dsa-sha2-128s sign failed: {e}") from e

    def verify(self, data: bytes, signature: bytes, public_key: Any) -> bool:
        from pqcrypto.sign import sphincs_sha2_128s_simple as slh
        if not isinstance(public_key, (bytes, bytearray)):
            return False
        if not isinstance(signature, (bytes, bytearray)):
            return False
        try:
            return bool(slh.verify(bytes(public_key), data, bytes(signature)))
        except Exception:
            return False

    def serialize_public_key(self, public_key: Any) -> bytes:
        """Raw 32-byte SLH-DSA-SHA2-128s public key.

        pqcrypto returns the public key as raw bytes, so this is an
        identity cast with type validation.
        """
        if not isinstance(public_key, (bytes, bytearray)):
            raise CryptoSuiteError(
                f"slh-dsa-sha2-128s public_key must be bytes; got "
                f"{type(public_key).__name__}"
            )
        return bytes(public_key)

    def deserialize_public_key(self, raw: bytes) -> Any:
        if not isinstance(raw, (bytes, bytearray)):
            raise CryptoSuiteError(
                f"slh-dsa-sha2-128s raw public key must be bytes; got "
                f"{type(raw).__name__}"
            )
        if len(raw) != self.PUBLIC_KEY_SIZE:
            raise CryptoSuiteError(
                f"slh-dsa-sha2-128s public key must be "
                f"{self.PUBLIC_KEY_SIZE} bytes; got {len(raw)}"
            )
        return bytes(raw)

    # Single canonical wire form — matches legacy serialization exactly.
    def serialize_public_key_for_multikey(self, public_key: Any) -> bytes:
        return self.serialize_public_key(public_key)

    def deserialize_public_key_from_multikey(self, raw: bytes) -> Any:
        return self.deserialize_public_key(raw)


# Register suites at import time. Wave 4 (ML-KEM) registers from its
# own module since KEM has a distinct interface from signing.
register_suite(Secp256k1Suite())
register_suite(Ed25519Suite())
register_suite(MLDSA65Suite())
register_suite(SLHDSASHA2128sSuite())
