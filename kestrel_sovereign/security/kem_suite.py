"""
KEMSuite — pluggable Key Encapsulation Mechanism suites for Kestrel.

Wave 4 of the Quantum Hardening epic (#921, #919). Where Wave 1's
:class:`CryptoSuite` covers signing, this module covers KEMs — the
primitive used to wrap a symmetric key for export confidentiality
(CAR exports, capsule sharing, future encrypted backups).

Why a separate abstraction from CryptoSuite
-------------------------------------------

KEM and signing have fundamentally different shapes:

- **Signing**: ``sign(data, priv) -> sig`` and ``verify(data, sig, pub)``
- **KEM**: ``encapsulate(pub) -> (ciphertext, shared_secret)`` and
  ``decapsulate(ciphertext, priv) -> shared_secret``

A KEM never signs anything; a signature suite never produces a shared
secret. Putting them under one ABC would force every implementer to
stub the wrong half. They share the *registry* concept (alg_id +
multicodec + multikey serialization), but the algorithmic interfaces
diverge cleanly.

Threat model framing
--------------------

KEM ciphertexts captured today and stored by an HNDL adversary will
be decrypted whenever a quantum computer with enough qubits exists.
Wave 4 mitigates by wrapping every export ciphertext with a HYBRID
KEM: X25519 (classical) ⊕ ML-KEM-768 (NIST FIPS 203 Cat-3, lattice-
based PQ). The shared secrets are concatenated and run through
HKDF-SHA256 to produce the symmetric key used by AES-256-GCM.

Forward safety: if either half is broken, the other still bounds an
attacker's success. To recover the wrapped key the adversary must
break BOTH X25519 (Shor on ECC) AND ML-KEM-768 (cryptanalytic break
of Module-LWE) — independently, not jointly. CNSA 2.0 + IETF
draft-ietf-pquip-hybrid-kems are explicit about this.

Algorithm identifier registry
-----------------------------

Reserved KEM identifiers:

- ``x25519`` — classical (RFC 7748 Curve25519 ECDH; the KEM construction
  follows draft-ietf-cose-hpke-encrypt with HKDF for shared-secret
  derivation)
- ``ml-kem-768`` — Wave 4 (NIST FIPS 203 Cat-3, lattice-based)

Wave 5 may add ML-KEM-1024 for higher security tiers on
release-signing artifacts; the abstraction supports it without
further changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Tuple, Type


# ---------------------------------------------------------------------------
# Reserved KEM algorithm identifiers
# ---------------------------------------------------------------------------

ALG_X25519 = "x25519"
ALG_ML_KEM_768 = "ml-kem-768"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class KEMSuiteError(Exception):
    """Raised on KEM operation failures (invalid keys, decapsulation
    failures, malformed ciphertext, etc.)."""


# ---------------------------------------------------------------------------
# KEMKeypair container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KEMKeypair:
    """A KEM keypair owned by a specific suite.

    ``private_key`` and ``public_key`` are opaque to callers. Concrete
    types vary per suite (e.g. ``X25519PrivateKey`` for x25519, raw
    bytes for ML-KEM-768).
    """

    suite_id: str
    private_key: Any
    public_key: Any


# ---------------------------------------------------------------------------
# KEMSuite ABC
# ---------------------------------------------------------------------------

class KEMSuite(abc.ABC):
    """Abstract pluggable Key Encapsulation Mechanism suite.

    Concrete suites set ``alg_id`` and ``public_key_multicodec`` at
    class scope and implement five methods: ``generate_keypair``,
    ``encapsulate``, ``decapsulate``, ``serialize_public_key``,
    ``deserialize_public_key`` (plus matching ``_for_multikey`` /
    ``_from_multikey`` variants where the wire forms differ).
    """

    alg_id: ClassVar[str] = ""
    public_key_multicodec: ClassVar[bytes] = b""
    is_post_quantum: ClassVar[bool] = False

    @abc.abstractmethod
    def generate_keypair(self) -> KEMKeypair:
        ...

    @abc.abstractmethod
    def encapsulate(self, public_key: Any) -> Tuple[bytes, bytes]:
        """Encapsulate a fresh shared secret to ``public_key``.

        Returns ``(ciphertext, shared_secret)``. ``shared_secret`` is
        the bytes the recipient will recover via :meth:`decapsulate`.
        Caller passes ``shared_secret`` to a KDF (typically
        HKDF-SHA256) before using as a symmetric key — never use the
        raw shared secret as an AEAD key.
        """
        ...

    @abc.abstractmethod
    def decapsulate(self, ciphertext: bytes, private_key: Any) -> bytes:
        """Recover the shared secret encapsulated by :meth:`encapsulate`.

        Raises :class:`KEMSuiteError` on malformed ciphertext or wrong
        key. Note that for ML-KEM the FIPS-203 implicit-rejection rule
        means decapsulation against a wrong key returns a *different*
        but still 32-byte secret; the AEAD that follows will fail
        authentication and the wrong key is detected at that layer.
        """
        ...

    @abc.abstractmethod
    def serialize_public_key(self, public_key: Any) -> bytes:
        """Wire serialization of a public key (legacy form)."""
        ...

    @abc.abstractmethod
    def deserialize_public_key(self, raw: bytes) -> Any:
        ...

    def serialize_public_key_for_multikey(self, public_key: Any) -> bytes:
        """Default: same as legacy. Override per-suite if the W3C
        Multikey codec mandates a different on-the-wire shape."""
        return self.serialize_public_key(public_key)

    def deserialize_public_key_from_multikey(self, raw: bytes) -> Any:
        return self.deserialize_public_key(raw)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_KEM_REGISTRY: Dict[str, KEMSuite] = {}


def register_kem_suite(suite: KEMSuite) -> None:
    """Add a KEM suite to the registry under its ``alg_id``.

    Called at module import time by each concrete suite. Re-registering
    the same alg_id silently overwrites — supports the alternative-
    implementation pattern (e.g. swapping pqcrypto for oqs-python
    behind the same id) without forcing a registry reset.
    """
    if not suite.alg_id:
        raise KEMSuiteError(
            f"{type(suite).__name__} missing alg_id; cannot register"
        )
    _KEM_REGISTRY[suite.alg_id] = suite


def get_kem_suite(alg_id: str) -> KEMSuite:
    suite = _KEM_REGISTRY.get(alg_id)
    if suite is None:
        raise KEMSuiteError(
            f"no registered KEM suite for alg_id={alg_id!r}; "
            f"registered: {sorted(_KEM_REGISTRY)}"
        )
    return suite


def list_registered_kems() -> list[str]:
    return list(_KEM_REGISTRY)


# ---------------------------------------------------------------------------
# X25519Suite — classical KEM half (Wave 4)
# ---------------------------------------------------------------------------

class X25519Suite(KEMSuite):
    """X25519 (RFC 7748) Diffie-Hellman as a KEM.

    Construction follows the standard ECIES-style pattern:

    - ``encapsulate(pub_R)``: generate ephemeral keypair (sk_E, pk_E);
      compute shared = ECDH(sk_E, pk_R); return
      ``(ciphertext = pk_E_bytes, shared_secret = shared)``.
    - ``decapsulate(ciphertext = pk_E_bytes, sk_R)``: parse pk_E from
      the ciphertext bytes; compute shared = ECDH(sk_R, pk_E); return
      ``shared``.

    The shared secret is the raw 32-byte X25519 output. Callers MUST
    feed it through a KDF (HKDF-SHA256 in Wave 4's hybrid combiner)
    before using as a symmetric key — RFC 7748 §6.1 explicitly warns
    against using the raw output directly.

    Threat note
    -----------

    X25519 is Shor-vulnerable like all classical ECDH. Wave 4 pairs
    it with ML-KEM-768 in a hybrid KEM; the classical half provides
    decades-aged cryptanalytic confidence while the PQ half handles
    the future-quantum-adversary case.
    """

    alg_id: ClassVar[str] = ALG_X25519
    # Multicodec 0xec (x25519-pub), varint-encoded
    public_key_multicodec: ClassVar[bytes] = b"\xec\x01"
    is_post_quantum: ClassVar[bool] = False

    PUBLIC_KEY_SIZE: ClassVar[int] = 32
    PRIVATE_KEY_SIZE: ClassVar[int] = 32
    CIPHERTEXT_SIZE: ClassVar[int] = 32  # ciphertext is the ephemeral pubkey
    SHARED_SECRET_SIZE: ClassVar[int] = 32

    def generate_keypair(self) -> KEMKeypair:
        from cryptography.hazmat.primitives.asymmetric import x25519
        priv = x25519.X25519PrivateKey.generate()
        return KEMKeypair(
            suite_id=self.alg_id,
            private_key=priv,
            public_key=priv.public_key(),
        )

    def encapsulate(self, public_key: Any) -> Tuple[bytes, bytes]:
        from cryptography.hazmat.primitives.asymmetric import x25519
        try:
            ephemeral = x25519.X25519PrivateKey.generate()
            shared = ephemeral.exchange(public_key)
            ct = self.serialize_public_key(ephemeral.public_key())
            return ct, shared
        except Exception as e:
            raise KEMSuiteError(f"x25519 encapsulate failed: {e}") from e

    def decapsulate(self, ciphertext: bytes, private_key: Any) -> bytes:
        from cryptography.hazmat.primitives.asymmetric import x25519
        if not isinstance(ciphertext, (bytes, bytearray)):
            raise KEMSuiteError(
                f"x25519 ciphertext must be bytes; got "
                f"{type(ciphertext).__name__}"
            )
        if len(ciphertext) != self.CIPHERTEXT_SIZE:
            raise KEMSuiteError(
                f"x25519 ciphertext must be {self.CIPHERTEXT_SIZE} bytes; "
                f"got {len(ciphertext)}"
            )
        try:
            ephemeral_pub = x25519.X25519PublicKey.from_public_bytes(bytes(ciphertext))
            return private_key.exchange(ephemeral_pub)
        except Exception as e:
            raise KEMSuiteError(f"x25519 decapsulate failed: {e}") from e

    def serialize_public_key(self, public_key: Any) -> bytes:
        """Raw 32-byte X25519 public key (RFC 7748 wire format)."""
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat,
        )
        return public_key.public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        )

    def deserialize_public_key(self, raw: bytes) -> Any:
        from cryptography.hazmat.primitives.asymmetric import x25519
        if not isinstance(raw, (bytes, bytearray)):
            raise KEMSuiteError(
                f"x25519 raw public key must be bytes; got {type(raw).__name__}"
            )
        if len(raw) != self.PUBLIC_KEY_SIZE:
            raise KEMSuiteError(
                f"x25519 public key must be {self.PUBLIC_KEY_SIZE} bytes; "
                f"got {len(raw)}"
            )
        try:
            return x25519.X25519PublicKey.from_public_bytes(bytes(raw))
        except Exception as e:
            raise KEMSuiteError(
                f"x25519 public-key deserialization failed: {e}"
            ) from e


# ---------------------------------------------------------------------------
# MLKEM768Suite — post-quantum KEM half (Wave 4)
# ---------------------------------------------------------------------------

class MLKEM768Suite(KEMSuite):
    """ML-KEM-768 (NIST FIPS 203, Cat-3) Module-LWE-based KEM.

    Library
    -------

    Backed by ``pqcrypto.kem.ml_kem_768`` — same library family as
    Wave 2's ML-DSA-65 (#950) and Wave 3's SLH-DSA-128s (#960). No
    compile-on-deploy; prebuilt wheels for macOS arm64, Linux x86_64,
    Windows.

    pqcrypto's ML-KEM API:

    - ``generate_keypair()`` → ``(public_bytes, secret_bytes)``
    - ``encrypt(public_bytes)`` → ``(ciphertext, shared_secret)``
    - ``decrypt(secret_bytes, ciphertext)`` → ``shared_secret``

    The implementation follows FIPS 203's implicit-rejection rule:
    decapsulating with a wrong key produces a 32-byte secret that
    differs from the encapsulator's, but no error is raised. The AEAD
    layer downstream catches the mismatch via authentication failure.

    Sizes (Cat-3, FIPS 203 Table 1)
    -------------------------------

    - Public key: 1184 bytes
    - Secret key: 2400 bytes
    - Ciphertext: 1088 bytes
    - Shared secret: 32 bytes (the same size as the AES-256 key it
      will eventually derive)

    Multikey shape
    --------------

    - ``alg_id``: ``"ml-kem-768"``
    - ``public_key_multicodec``: ``b"\\x89\\x24"`` (multicodec 0x1209,
      proposed; treat as experimental).
    """

    alg_id: ClassVar[str] = ALG_ML_KEM_768
    # Multicodec 0x1209 (ml-kem-768-pub, proposed).
    public_key_multicodec: ClassVar[bytes] = b"\x89\x24"
    is_post_quantum: ClassVar[bool] = True

    PUBLIC_KEY_SIZE: ClassVar[int] = 1184
    SECRET_KEY_SIZE: ClassVar[int] = 2400
    CIPHERTEXT_SIZE: ClassVar[int] = 1088
    SHARED_SECRET_SIZE: ClassVar[int] = 32

    def generate_keypair(self) -> KEMKeypair:
        from pqcrypto.kem import ml_kem_768
        public_bytes, secret_bytes = ml_kem_768.generate_keypair()
        return KEMKeypair(
            suite_id=self.alg_id,
            private_key=secret_bytes,
            public_key=public_bytes,
        )

    def encapsulate(self, public_key: Any) -> Tuple[bytes, bytes]:
        from pqcrypto.kem import ml_kem_768
        if not isinstance(public_key, (bytes, bytearray)):
            raise KEMSuiteError(
                f"ml-kem-768 public_key must be bytes ({self.PUBLIC_KEY_SIZE} "
                f"bytes); got {type(public_key).__name__}"
            )
        if len(public_key) != self.PUBLIC_KEY_SIZE:
            raise KEMSuiteError(
                f"ml-kem-768 public_key must be {self.PUBLIC_KEY_SIZE} bytes; "
                f"got {len(public_key)}"
            )
        try:
            ct, ss = ml_kem_768.encrypt(bytes(public_key))
            return ct, ss
        except Exception as e:
            raise KEMSuiteError(f"ml-kem-768 encapsulate failed: {e}") from e

    def decapsulate(self, ciphertext: bytes, private_key: Any) -> bytes:
        from pqcrypto.kem import ml_kem_768
        if not isinstance(private_key, (bytes, bytearray)):
            raise KEMSuiteError(
                f"ml-kem-768 private_key must be bytes ({self.SECRET_KEY_SIZE} "
                f"bytes); got {type(private_key).__name__}"
            )
        if not isinstance(ciphertext, (bytes, bytearray)):
            raise KEMSuiteError(
                f"ml-kem-768 ciphertext must be bytes; got "
                f"{type(ciphertext).__name__}"
            )
        if len(ciphertext) != self.CIPHERTEXT_SIZE:
            raise KEMSuiteError(
                f"ml-kem-768 ciphertext must be {self.CIPHERTEXT_SIZE} bytes; "
                f"got {len(ciphertext)}"
            )
        try:
            return ml_kem_768.decrypt(bytes(private_key), bytes(ciphertext))
        except Exception as e:
            raise KEMSuiteError(f"ml-kem-768 decapsulate failed: {e}") from e

    def serialize_public_key(self, public_key: Any) -> bytes:
        """Raw 1184-byte ML-KEM-768 public key (FIPS 203 wire format).

        pqcrypto already returns the public key as raw bytes, so this
        is an identity cast with type validation.
        """
        if not isinstance(public_key, (bytes, bytearray)):
            raise KEMSuiteError(
                f"ml-kem-768 public_key must be bytes; got "
                f"{type(public_key).__name__}"
            )
        return bytes(public_key)

    def deserialize_public_key(self, raw: bytes) -> Any:
        if not isinstance(raw, (bytes, bytearray)):
            raise KEMSuiteError(
                f"ml-kem-768 raw public key must be bytes; got "
                f"{type(raw).__name__}"
            )
        if len(raw) != self.PUBLIC_KEY_SIZE:
            raise KEMSuiteError(
                f"ml-kem-768 public key must be {self.PUBLIC_KEY_SIZE} bytes; "
                f"got {len(raw)}"
            )
        return bytes(raw)


# Register at import time. The hybrid combiner in Wave 4 sub-PR 2 looks
# these up via ``get_kem_suite`` rather than importing the classes
# directly, mirroring the CryptoSuite registry pattern.
register_kem_suite(X25519Suite())
register_kem_suite(MLKEM768Suite())


__all__ = [
    "ALG_ML_KEM_768",
    "ALG_X25519",
    "KEMKeypair",
    "KEMSuite",
    "KEMSuiteError",
    "MLKEM768Suite",
    "X25519Suite",
    "get_kem_suite",
    "list_registered_kems",
    "register_kem_suite",
]
