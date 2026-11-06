"""Stable random roots wrapped by the host's rotatable data-encryption key.

Callers own the schema lock and transaction. They must retain the wrapped root
and never initialize a replacement when records depend on a missing root.
"""

from __future__ import annotations

import secrets

from kestrel_sdk.resources import ResourceError
from kestrel_sdk.security.aead import AEADCipher, KEY_SIZE
from kestrel_sdk.security.exceptions import DecryptionError


def resolve_envelope_key(
    cipher: AEADCipher,
    wrapped_key: bytes | None,
    *,
    has_dependents: bool,
    integrity_error: type[ResourceError],
    missing_message: str,
    unavailable_message: str,
    invalid_message: str,
) -> tuple[bytes, bytes | None]:
    """Return the stable root and a new envelope only for an empty namespace.

    The caller persists a returned envelope in its existing transaction and
    publishes the root in memory only after the transaction is acknowledged.
    Decryption failures never cause replacement or expose stored key material.
    """
    if wrapped_key is None:
        if has_dependents:
            raise integrity_error(missing_message)
        key = secrets.token_bytes(KEY_SIZE)
        return key, cipher.encrypt(key)

    try:
        key = cipher.decrypt(wrapped_key)
    except DecryptionError:
        raise integrity_error(unavailable_message) from None
    if len(key) != KEY_SIZE:
        raise integrity_error(invalid_message)
    return key, None
