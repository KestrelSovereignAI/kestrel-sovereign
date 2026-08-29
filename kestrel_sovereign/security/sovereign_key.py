"""Process-local provenance for the sovereign API key.

The HTTP server may mint a temporary bootstrap key so a local development
session remains usable.  That key authenticates requests for the lifetime of
the process, but it is not durable authority for a whole-host mutation whose
evidence must survive a restart.  Keep only a fingerprint here: callers can
distinguish that bootstrap credential without retaining another copy of the
secret.
"""

from __future__ import annotations

import hashlib
import hmac


_ephemeral_key_fingerprint: bytes | None = None


def _fingerprint(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


def mark_ephemeral_sovereign_key(key: str) -> None:
    """Record that *key* was generated for this process, not configured."""

    global _ephemeral_key_fingerprint
    _ephemeral_key_fingerprint = _fingerprint(key)


def is_ephemeral_sovereign_key(key: str) -> bool:
    """Return whether *key* is the process-generated bootstrap credential."""

    fingerprint = _ephemeral_key_fingerprint
    return fingerprint is not None and hmac.compare_digest(
        _fingerprint(key), fingerprint
    )
