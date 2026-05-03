"""
AEAD container — versioned AES-256-GCM with Fernet-compatible read path.

Wave 0C of the Quantum Hardening epic (#921, issue #915). Replaces Fernet
(AES-128-CBC + HMAC-SHA256) with AES-256-GCM as Kestrel's symmetric AEAD,
without orphaning data already encrypted under Fernet.

Format
------

A v2 token is the byte string::

    "KSAv2:" || urlsafe_b64encode(nonce || ciphertext_with_tag)

where ``nonce`` is 12 random bytes (96 bits, AES-GCM standard) and
``ciphertext_with_tag`` is the AES-256-GCM output (plaintext encryption +
16-byte authentication tag).

The 6-byte magic prefix ``KSAv2:`` is chosen so that:

- It cannot collide with a valid Fernet token, which is URL-safe base64
  starting with ``g`` (the encoding of Fernet's version byte ``0x80``).
- It's plain ASCII, easy to detect by humans reading rows in a database.
- The remainder is URL-safe base64, so the whole token is safe to put
  in JSON, URLs, headers, or anywhere a string is expected.

Optional Associated Data (AAD) is supported: if the caller passes ``aad``,
it is bound into the GCM tag. Decryption with mismatched AAD fails; AAD is
not stored in the token. The recommended pattern is to derive AAD from
out-of-band context (e.g. ``agent_id || row_id || "conversation"``).

Backwards compatibility
-----------------------

``AEADCipher.decrypt`` recognises both v2 tokens and legacy Fernet tokens.
Detection is purely by the ``KSAv2:`` prefix; absence of the prefix routes
to the Fernet decode path. Existing data therefore continues to work
without any migration step. New writes always emit v2.

This contract is the foundation for the rest of the wave plan: every later
artifact format will follow the same ``v2:`` prefix-and-base64 shape.

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


class AEADCipher:
    """
    Drop-in replacement for ``cryptography.fernet.Fernet`` using AES-256-GCM.

    Encrypts always to v2 (``KSAv2:`` prefix). Decrypts both v2 and legacy
    Fernet tokens, so existing data keeps working without a migration step.

    The constructor accepts either:

    - a 32-byte raw key (preferred), or
    - a 44-byte URL-safe-base64 Fernet key (legacy compatibility — the
      key is base64-decoded back to its 32 raw bytes).

    The same key is used for AES-256-GCM (raw 32 bytes) and for the
    legacy Fernet decode path (URL-safe-base64 of the same 32 bytes).
    This means a single key value rotates from Fernet to v2 cleanly:
    old data still decrypts, new data is written as v2.

    AAD is optional and out-of-band; pass it explicitly to ``encrypt`` and
    ``decrypt`` if you want context-binding.
    """

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
        ct = self._aes.encrypt(nonce, plaintext, aad)
        return KSA_V2_PREFIX + base64.urlsafe_b64encode(nonce + ct)

    def decrypt(self, token: Union[bytes, str], aad: Optional[bytes] = None) -> bytes:
        """Decrypt either a v2 token or a legacy Fernet token.

        AAD only applies to v2 tokens; passing AAD on a legacy Fernet
        decode raises ``DecryptionError`` because Fernet has no AAD support
        and a silent ignore would mask a binding mismatch.
        """
        if isinstance(token, str):
            token = token.encode("ascii")

        if token.startswith(KSA_V2_PREFIX):
            try:
                payload = base64.urlsafe_b64decode(token[len(KSA_V2_PREFIX):])
            except Exception as e:
                raise DecryptionError(f"v2 token base64 decode failed: {e}") from e
            if len(payload) < NONCE_SIZE + 16:
                raise DecryptionError("v2 token too short to contain nonce+tag")
            nonce, ct = payload[:NONCE_SIZE], payload[NONCE_SIZE:]
            try:
                return self._aes.decrypt(nonce, ct, aad)
            except Exception as e:
                raise DecryptionError(
                    f"v2 AES-GCM decryption failed (wrong key, AAD, or tampering): {e}"
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
