"""Backwards-compat decrypt wrapper that handles pre-v2 raw AES-GCM
ciphertext (#1458).

The encryption stack has been through three on-disk formats for
purpose-keyed ciphertext (``service-keys``, ``conversations``,
``wallet``, ``backup``):

  1. **Pre-Wave-0C raw AES-GCM** — ``nonce(12) || ct+tag``. No
     version prefix, no AAD. Written by the earliest ``encrypt()``
     implementation. Examples in the wild: agent service-keys rows
     created on agents booted in early 2026 before the quantum
     hardening epic landed.
  2. **Legacy Fernet** — URL-safe-base64 of ``0x80 || timestamp ||
     IV || ct || HMAC``. Drop-in replacement for raw AES-GCM during
     the SDK transition.
  3. **v2 / Wave 0C AEAD** (``KSAv2:`` prefix + strict-base64 of
     ``alg_id || nonce || ct+tag``). The current write format.

``AEADCipher.decrypt`` from ``kestrel_sdk.security.aead`` handles
formats (2) and (3). Format (1) was unhandled and raised
``DecryptionError`` at boot for agents whose ``agent_service_keys``
or ``host_service_keys`` rows were written in the pre-Wave-0C era —
the most visible symptom is "Could not activate agent OpenRouter
key: Legacy Fernet decryption failed" in ``logs/host.log`` followed
by downstream provider failures when memory_feature / LLM service
tries to embed or call.

This module bridges format (1) by attempting raw AES-GCM with the
same purpose-derived key whenever the SDK's ``decrypt()`` rejects a
candidate. A successful legacy decrypt is logged at WARNING with the
agent_did + purpose so operators know which rows still need a write-
side rotation to v2; the rotation is left to a separate
backfill/cron task because it requires per-row writes that can race
with the runtime read path this module sits on.

Scope: every caller that reads ``encrypted_key`` blobs written by
historic ``encrypt(agent_did, purpose, ...)`` calls should route
through ``decrypt_with_legacy_fallback`` instead of the SDK's
``decrypt`` directly. As of this change, ``service_key_storage`` and
``host_key_storage`` are the two real-world callers; both go through
this helper.
"""

from __future__ import annotations

import logging
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kestrel_sdk.security.encryption import decrypt as _sdk_decrypt
from kestrel_sdk.security.encryption import get_agent_key
from kestrel_sdk.security.exceptions import DecryptionError

logger = logging.getLogger(__name__)


# Matches ``kestrel_sdk.security.aead.NONCE_SIZE`` AND
# ``kestrel_sovereign.security.agent_encryption.NONCE_SIZE`` (both 12
# bytes, the AES-GCM standard 96-bit nonce). Duplicated here rather
# than imported so this module's import graph stays tight — see the
# encryption_backfill module's same pattern.
_PRE_V2_NONCE_SIZE = 12
# A GCM authentication tag is 16 bytes; ciphertext shorter than
# nonce + tag cannot possibly be a valid pre-v2 row.
_GCM_TAG_SIZE = 16


def decrypt_with_legacy_fallback(
    agent_did: str,
    purpose: str,
    ciphertext: bytes,
) -> bytes:
    """Decrypt ``ciphertext`` written under ``(agent_did, purpose)``.

    Resolution order:

    1. SDK ``decrypt`` (v2 ``KSAv2:`` or legacy Fernet).
    2. Pre-v2 raw AES-GCM with the purpose-derived key — ciphertext
       layout ``nonce(12) || ct+tag``. No AAD, no version prefix.

    Returns the plaintext bytes. Raises ``DecryptionError`` only when
    BOTH formats reject the input; the error from the SDK path is
    surfaced as the cause so post-mortem chains stay readable.

    A successful pre-v2 fallback logs once at WARNING with the agent
    DID prefix and purpose so an operator can plan a rotation pass.
    """
    try:
        return _sdk_decrypt(agent_did, purpose, ciphertext)
    except DecryptionError as sdk_err:
        # Only attempt the pre-v2 raw-AES-GCM path if the ciphertext
        # is at least long enough to contain a nonce and a GCM tag.
        # A shorter blob can't be a valid raw-AES-GCM row, so we
        # surface the SDK's error without a misleading second attempt.
        if len(ciphertext) < _PRE_V2_NONCE_SIZE + _GCM_TAG_SIZE:
            raise

        try:
            key = get_agent_key(agent_did, purpose)
        except Exception:
            # Re-raise the SDK error — key derivation failures are not
            # specific to the legacy path and the SDK path failed on
            # the same key. The DecryptionError carries the surface
            # diagnostic, this exception just means we can't even try
            # the fallback so there's nothing to add.
            raise sdk_err

        nonce = ciphertext[:_PRE_V2_NONCE_SIZE]
        ct_with_tag = ciphertext[_PRE_V2_NONCE_SIZE:]
        try:
            plaintext = AESGCM(key).decrypt(nonce, ct_with_tag, None)
        except InvalidTag:
            # The ciphertext didn't authenticate under the pre-v2
            # layout either. Surface the SDK error (which represented
            # the actual user-facing format attempt) as the diagnostic;
            # the InvalidTag from this branch is internal evidence that
            # raw AES-GCM is also not the right format.
            raise sdk_err
        except Exception as e:
            # Defensive: a non-MAC failure here would be unexpected
            # (e.g. a key-size mismatch). Bubble it up wrapped so the
            # caller still gets a DecryptionError contract.
            raise DecryptionError(
                f"pre-v2 raw AES-GCM fallback failed: {type(e).__name__}: {e}"
            ) from e

        logger.warning(
            "Decrypted pre-v2 ciphertext via raw AES-GCM fallback "
            "(agent_did=%s..., purpose=%s, ciphertext_len=%d). "
            "Row should be rotated to v2 (KSAv2:) on next write. "
            "See #1458.",
            agent_did[:30] if agent_did else "<none>",
            purpose,
            len(ciphertext),
        )
        return plaintext


__all__ = ["decrypt_with_legacy_fallback"]
