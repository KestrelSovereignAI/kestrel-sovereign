"""
AEAD container — versioned AES-256-GCM with Fernet-compatible read path.

Wave 0C of the Quantum Hardening epic (#921, issue #915). Replaces Fernet
(AES-128-CBC + HMAC-SHA256) with AES-256-GCM as Kestrel's symmetric AEAD,
without orphaning data already encrypted under Fernet.

Format
------

A v2 token is the byte string::

    "KSAv2:" || strict_urlsafe_b64encode(alg_id(1) || nonce(12) || ct_with_tag)

where:

- ``alg_id`` is a single byte identifying the AEAD suite. Currently::

      ALG_AES_256_GCM = 0x01

  The byte sits in the framing (so a future ``alg_id=0x02`` can be added
  without bumping ``v``) **and** is bound into the GCM authentication
  tag via Associated Data, so flipping it to swap suites silently fails
  authentication.

- ``nonce`` is 12 random bytes (96 bits, AES-GCM standard).
- ``ct_with_tag`` is AES-256-GCM ciphertext + 16-byte authentication tag.

The base64 encoding is **strict urlsafe** (RFC 4648 §5 with validation):
non-alphabet characters and trailing junk are rejected by ``decrypt``,
preventing multiple textual encodings of the same ciphertext from passing
verification.

The 6-byte magic prefix ``KSAv2:`` is chosen so that:

- It cannot collide with a valid Fernet token, which is URL-safe base64
  starting with ``g`` (the encoding of Fernet's version byte ``0x80``).
- It's plain ASCII, easy to detect by humans reading rows in a database.
- The remainder is URL-safe base64, JSON-, URL-, header-safe.

Optional caller-supplied Associated Data (AAD) is supported on top of the
internal ``alg_id`` AAD: if the caller passes ``aad``, the AEAD authenticates
``alg_id || aad``. Mismatched or missing caller AAD on decrypt fails; AAD
is not stored in the token. The recommended pattern is to derive AAD from
out-of-band context (e.g., ``agent_id || row_id || "conversation"``).

Backwards compatibility
-----------------------

``AEADCipher.decrypt`` recognises both v2 tokens and legacy Fernet tokens.
Detection is purely by the ``KSAv2:`` prefix; absence of the prefix routes
to the Fernet decode path. Existing data therefore continues to work
without any migration step. New writes always emit v2.

This contract is the foundation for the rest of the wave plan: every later
artifact format follows the same ``KSAv*:``-prefix-and-strict-base64 shape
and includes an ``alg_id`` byte for crypto-agility.

Threat-model framing
--------------------

Per ``docs/architecture/security/PQ_THREAT_MODEL.md``, local AEAD with
locally-derived keys is *not* HNDL-vulnerable. Wave 0C is hygiene: replace
AES-128 (Grover-degraded to ~64-bit effective) with AES-256 (still 128-bit
effective post-Grover). It is not the surface PQ KEM wrapping addresses
(that is Wave 4, for export and capsule sharing).
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import Optional, Union

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .exceptions import DecryptionError


# Magic prefix marking a Wave-0C v2 token. Length 6 bytes; chosen so the
# remainder of the token is URL-safe base64 and the whole token cannot
# collide with a Fernet token (which always starts with the URL-safe-base64
# encoding of 0x80, i.e. "g").
KSA_V2_PREFIX = b"KSAv2:"

NONCE_SIZE = 12  # AES-GCM standard 96-bit nonce
KEY_SIZE = 32    # AES-256
ALG_ID_SIZE = 1  # one-byte algorithm identifier
GCM_TAG_SIZE = 16

# Algorithm identifier registry. Values are stable bytes that ship in every
# v2 token; new suites get new IDs without bumping the version prefix.
# Reserve 0x00 ("none") so a zero byte from a corrupted/all-zero payload
# does not silently map to a real suite.
ALG_NONE = 0x00
ALG_AES_256_GCM = 0x01


def _strict_urlsafe_b64decode(data: bytes) -> bytes:
    """Strict URL-safe base64 decode.

    ``base64.urlsafe_b64decode`` is permissive: it silently ignores
    characters outside the alphabet and accepts trailing junk. That lets
    a tampered-with token like ``cipher.encrypt(pt) + b"!!!"`` round-trip
    successfully — multiple encodings of the same ciphertext, despite
    the AEAD authentication tag.

    Use ``base64.b64decode`` with ``altchars=b"-_"`` and ``validate=True``
    to enforce a single canonical encoding. Any non-base64 character or
    non-multiple-of-4 length raises ``binascii.Error``.
    """
    return base64.b64decode(data, altchars=b"-_", validate=True)


class AEADCipher:
    """
    Drop-in replacement for ``cryptography.fernet.Fernet`` using AES-256-GCM.

    Encrypts always to v2 (``KSAv2:`` prefix + strict-base64 of
    ``alg_id || nonce || ct+tag``). Decrypts both v2 and legacy Fernet
    tokens, so existing data keeps working without a migration step.

    The constructor accepts either:

    - a 32-byte raw key (preferred), or
    - a 44-byte URL-safe-base64 Fernet key (legacy compatibility — the
      key is base64-decoded back to its 32 raw bytes).

    The same key value works for AES-256-GCM (raw 32 bytes) and for the
    legacy Fernet decode path (URL-safe-base64 of the same 32 bytes).
    This means a single key rotates from Fernet to v2 cleanly: old data
    still decrypts, new data is written as v2.

    AAD is optional and out-of-band; pass it explicitly to ``encrypt`` and
    ``decrypt`` if you want context-binding. An ``alg_id`` byte is always
    bound into the AEAD tag automatically.
    """

    # The suite this instance writes. Reading dispatches on the token's
    # alg_id byte; writing always uses ALG_AES_256_GCM.
    ALG_ID_WRITE = ALG_AES_256_GCM

    __slots__ = ("_key", "_aes", "_legacy_fernet_b64")

    def __init__(self, key: Union[bytes, str]):
        raw = self._coerce_to_raw_key(key)
        self._key: bytes = raw
        self._aes = AESGCM(raw)
        # Pre-compute the URL-safe-base64 form for the legacy Fernet decode
        # path, but defer constructing the Fernet object until it's needed
        # (decrypt of a non-v2 token).
        self._legacy_fernet_b64: bytes = base64.urlsafe_b64encode(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: Union[bytes, str], aad: Optional[bytes] = None) -> bytes:
        """Encrypt to a v2 token. Always writes v2; never emits Fernet."""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")
        nonce = os.urandom(NONCE_SIZE)
        full_aad = self._compose_aad(self.ALG_ID_WRITE, aad)
        ct = self._aes.encrypt(nonce, plaintext, full_aad)
        framed = bytes([self.ALG_ID_WRITE]) + nonce + ct
        return KSA_V2_PREFIX + base64.urlsafe_b64encode(framed)

    def decrypt(self, token: Union[bytes, str], aad: Optional[bytes] = None) -> bytes:
        """Decrypt either a v2 token or a legacy Fernet token.

        AAD only applies to v2 tokens; passing AAD on a legacy Fernet
        decode raises ``DecryptionError`` because Fernet has no AAD support
        and a silent ignore would mask a binding mismatch.
        """
        if isinstance(token, str):
            token = token.encode("ascii")

        if token.startswith(KSA_V2_PREFIX):
            body = token[len(KSA_V2_PREFIX):]
            try:
                framed = _strict_urlsafe_b64decode(body)
            except (binascii.Error, ValueError) as e:
                raise DecryptionError(
                    f"v2 token base64 decode failed (non-canonical encoding): {e}"
                ) from e
            if len(framed) < ALG_ID_SIZE + NONCE_SIZE + GCM_TAG_SIZE:
                raise DecryptionError("v2 token too short to contain alg_id+nonce+tag")
            alg_id = framed[0]
            nonce = framed[ALG_ID_SIZE:ALG_ID_SIZE + NONCE_SIZE]
            ct = framed[ALG_ID_SIZE + NONCE_SIZE:]

            if alg_id != ALG_AES_256_GCM:
                raise DecryptionError(
                    f"v2 token uses unknown alg_id 0x{alg_id:02x}; this AEADCipher "
                    f"build only handles AES-256-GCM (0x{ALG_AES_256_GCM:02x})."
                )

            full_aad = self._compose_aad(alg_id, aad)
            try:
                return self._aes.decrypt(nonce, ct, full_aad)
            except Exception as e:
                raise DecryptionError(
                    f"v2 AES-GCM decryption failed (wrong key, AAD, alg_id, or tampering): {e}"
                ) from e

        # Legacy Fernet path
        if aad is not None:
            raise DecryptionError(
                "AAD passed but token is legacy Fernet (no AAD support). "
                "Re-encrypt the data as v2 before binding AAD."
            )
        try:
            fernet = Fernet(self._legacy_fernet_b64)
            return fernet.decrypt(token)
        except InvalidToken as e:
            raise DecryptionError(
                f"Legacy Fernet decryption failed (wrong key or corrupted): {e}"
            ) from e
        except Exception as e:
            raise DecryptionError(f"Legacy Fernet decryption failed: {e}") from e

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_v2(token: Union[bytes, str]) -> bool:
        """Return True iff the token is a v2 (Wave-0C) AEAD token."""
        if isinstance(token, str):
            token = token.encode("ascii", errors="ignore")
        return token.startswith(KSA_V2_PREFIX)

    @staticmethod
    def generate_key() -> bytes:
        """Generate a fresh 32-byte key, encoded as URL-safe base64.

        Drop-in replacement for ``Fernet.generate_key()``. The 44-byte
        ASCII output works directly as an ``AEADCipher`` constructor
        argument and as a legacy ``Fernet`` key (for migration). Raw
        random source is ``os.urandom``.
        """
        return base64.urlsafe_b64encode(os.urandom(KEY_SIZE))

    @staticmethod
    def _compose_aad(alg_id: int, user_aad: Optional[bytes]) -> bytes:
        """Always-bound prefix: ``alg_id || (user_aad or empty)``.

        Binding ``alg_id`` into AAD means a token whose framing alg_id byte
        has been flipped (to coerce a different suite) fails authentication
        even though the byte itself sits in clear in the framing.
        """
        return bytes([alg_id]) + (user_aad or b"")

    @staticmethod
    def _coerce_to_raw_key(key: Union[bytes, str]) -> bytes:
        if isinstance(key, str):
            key = key.encode("ascii")
        if len(key) == KEY_SIZE:
            return key
        # Try URL-safe base64 decode (legacy Fernet key shape: 44 bytes ASCII
        # encoding 32 raw bytes).
        try:
            decoded = base64.urlsafe_b64decode(key)
        except Exception as e:
            raise ValueError(
                f"AEADCipher key must be 32 raw bytes or a URL-safe-base64-encoded "
                f"32-byte key; got {len(key)} bytes that are not valid base64: {e}"
            ) from e
        if len(decoded) != KEY_SIZE:
            raise ValueError(
                f"AEADCipher key after base64 decode must be {KEY_SIZE} bytes; got {len(decoded)}"
            )
        return decoded
