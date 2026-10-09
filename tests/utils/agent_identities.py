"""Real ``AgentIdentity`` values in each of the three on-disk shapes.

Built from freshly generated keys rather than loaded from an agent directory:
the tests that use them care which DIDs an identity carries, not how they
were loaded.
"""

from __future__ import annotations

from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)


def legacy_identity(did: str) -> AgentIdentity:
    """An agent that has only its classical ``did:pkh``."""
    return AgentIdentity(
        legacy_did=did,
        legacy_keypair=get_suite(ALG_ECDSA_SECP256K1_SHA256).generate_keypair(),
        legacy_did_document={},
    )


def rotated_identity(legacy_did: str, successor_did: str) -> AgentIdentity:
    """An agent rotated from ``legacy_did`` onto ``successor_did``."""
    return AgentIdentity(
        legacy_did=legacy_did,
        legacy_keypair=get_suite(ALG_ECDSA_SECP256K1_SHA256).generate_keypair(),
        legacy_did_document={},
        hybrid_keypair=generate_hybrid_keypair(),
        new_did=successor_did,
    )


def born_hybrid_identity(did: str) -> AgentIdentity:
    """An agent that never had a classical identity (#2397)."""
    return AgentIdentity(hybrid_keypair=generate_hybrid_keypair(), new_did=did)
