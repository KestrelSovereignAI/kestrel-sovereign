"""Shared peer-transport credential construction.

Peer authentication is deliberately distinct from sovereign authority.  A
multi-process host must nevertheless hand every child the *same* peer key so
the local fleet can authenticate over the signal rails.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from collections.abc import MutableMapping

from kestrel_sovereign.security.sovereign_key import (
    is_ephemeral_sovereign_key,
    normalize_sovereign_api_key,
)


_PEER_KEY_DERIVATION_CONTEXT = b"kestrel-local-peer-transport-v1"
_PEER_KEY_PROVENANCE_ENV = "KESTREL_INTERNAL_PEER_API_KEY_PROVENANCE"
_PEER_KEY_PROVENANCE_PREFIX = "auto-v1:"


def _automatic_peer_key_provenance(peer_key: str) -> str:
    """Return a non-secret marker that identifies one host-generated key."""

    digest = hashlib.sha256(peer_key.encode("utf-8")).hexdigest()
    return f"{_PEER_KEY_PROVENANCE_PREFIX}{digest}"


def derive_peer_api_key(sovereign_key: str) -> str:
    """Derive a stable, one-way, domain-separated host peer credential."""

    normalized = normalize_sovereign_api_key(sovereign_key)
    digest = hmac.new(
        normalized.encode("utf-8"),
        _PEER_KEY_DERIVATION_CONTEXT,
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def ensure_peer_api_key(
    environment: MutableMapping[str, str] | None = None,
    *,
    sovereign_key: str | None = None,
) -> str:
    """Return and publish one normalized, non-sovereign peer credential.

    ``environment`` may be a child launch environment whose project ``.env``
    has already won precedence.  Publishing the selected value back to
    ``os.environ`` makes a generated key stable across later child launches
    and the in-process host server.
    """

    target = os.environ if environment is None else environment
    raw_sovereign_key = sovereign_key or target.get("KESTREL_API_KEY")
    raw_peer_key = target.get("KESTREL_PEER_API_KEY")
    had_peer_key = bool(raw_peer_key)
    provenance = target.get(_PEER_KEY_PROVENANCE_ENV, "")
    peer_was_automatic = False
    if raw_peer_key:
        normalized_existing = normalize_sovereign_api_key(raw_peer_key)
        peer_was_automatic = secrets.compare_digest(
            provenance,
            _automatic_peer_key_provenance(normalized_existing),
        )

    sovereign_is_durable = bool(
        raw_sovereign_key
        and not is_ephemeral_sovereign_key(
            normalize_sovereign_api_key(raw_sovereign_key)
        )
    )
    if not raw_peer_key or (peer_was_automatic and sovereign_is_durable):
        raw_peer_key = (
            derive_peer_api_key(raw_sovereign_key)
            if raw_sovereign_key
            else secrets.token_urlsafe(32)
        )
    peer_key = normalize_sovereign_api_key(raw_peer_key)

    if raw_sovereign_key:
        normalized_sovereign_key = normalize_sovereign_api_key(
            raw_sovereign_key
        )
        if secrets.compare_digest(peer_key, normalized_sovereign_key):
            raise RuntimeError(
                "KESTREL_PEER_API_KEY must be distinct from KESTREL_API_KEY"
            )

    target["KESTREL_PEER_API_KEY"] = peer_key
    os.environ["KESTREL_PEER_API_KEY"] = peer_key
    if not had_peer_key or peer_was_automatic:
        marker = _automatic_peer_key_provenance(peer_key)
        target[_PEER_KEY_PROVENANCE_ENV] = marker
        os.environ[_PEER_KEY_PROVENANCE_ENV] = marker
    else:
        # A project ``.env`` may replace an inherited automatic peer key. Its
        # differing value is explicit operator configuration, so do not let the
        # inherited marker make a later sovereign rotation overwrite it.
        target.pop(_PEER_KEY_PROVENANCE_ENV, None)
        os.environ.pop(_PEER_KEY_PROVENANCE_ENV, None)
    return peer_key
