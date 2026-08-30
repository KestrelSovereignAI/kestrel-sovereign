"""Shared peer-transport credential construction.

Peer authentication is deliberately distinct from sovereign authority.  A
multi-process host must nevertheless hand every child the *same* peer key so
the local fleet can authenticate over the signal rails.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import MutableMapping

from kestrel_sovereign.security.sovereign_key import (
    normalize_sovereign_api_key,
)


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
    raw_peer_key = target.get("KESTREL_PEER_API_KEY")
    if not raw_peer_key:
        raw_peer_key = secrets.token_urlsafe(32)
    peer_key = normalize_sovereign_api_key(raw_peer_key)

    raw_sovereign_key = sovereign_key or target.get("KESTREL_API_KEY")
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
    return peer_key
