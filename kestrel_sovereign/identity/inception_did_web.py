"""
did:web hybrid-identity inception — Wave 2 sub-PR 4 (#917).

Bridges the Wave 2 building blocks into one entry point:

- :func:`kestrel_sovereign.identity.hybrid_keypair.generate_hybrid_keypair`
- :func:`kestrel_sovereign.identity.did_web.build_did`
- :func:`kestrel_sovereign.identity.did_web.build_did_document`

A caller invokes :func:`create_did_web_identity` with a domain and
agent slug; the function returns the bundle a caller needs to
publish a hybrid agent: the DID URI, the DID document JSON-ready
dict to drop at ``https://<domain>/<slug>/did.json``, and the
HybridKeypair to keep private (caller stores via the existing
SecureKeyStorage / KMS path).

Scope discipline
----------------

This module is intentionally **separate** from
``kestrel_sovereign.inception_service`` so:

1. Existing ``did:pkh:eip155:1:<eth-address>`` agents (Kestrel #1,
   Emma, Meridian, Frinz tenants) are byte-stable through this
   release. Their inception path (``inception_service.create_kestrel_
   identity_async``) is untouched. The Wave 0C sub-PR 4 readiness
   check applies before any flip.
2. The new path is opt-in. Wave 3 (succession statements) carries
   the migration story for existing agents — their controllers sign
   a succession statement with SLH-DSA from a HYBRID identity that
   replaces the legacy did:pkh, and the chain walker honors the
   temporal cutoff via :mod:`verify_policy`'s
   ``post_cutoff_classical_allowed`` hook.
3. Reviewers can read this PR's diff without paging in the 600-line
   ``inception_service`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from kestrel_sovereign.identity.did_web import (
    build_did,
    build_did_document,
)
from kestrel_sovereign.identity.hybrid_keypair import (
    DEFAULT_CLASSICAL_ALG,
    DEFAULT_PQ_ALG,
    HybridKeypair,
    generate_hybrid_keypair,
)


@dataclass(frozen=True)
class HybridDidWebIdentity:
    """Result of a hybrid did:web inception.

    - ``did``: ``did:web:<domain>:<slug>`` URI
    - ``did_document``: W3C DID document with both Multikey verification
      methods (Ed25519 + ML-DSA-65 by default). Caller publishes this
      verbatim at ``did_to_url(did)``.
    - ``keypair``: composite ``HybridKeypair``. Caller is responsible
      for secure storage of both private keys (the existing
      ``SecureKeyStorage`` path or a future KMS-backed alternative).
    """

    did: str
    did_document: dict
    keypair: HybridKeypair


def create_did_web_identity(
    domain: str,
    slug: str,
    *,
    extra_path_segments: Sequence[str] = (),
    classical_alg: str = DEFAULT_CLASSICAL_ALG,
    pq_alg: str = DEFAULT_PQ_ALG,
    also_known_as: Optional[Sequence[str]] = None,
    services: Optional[Sequence[dict]] = None,
) -> HybridDidWebIdentity:
    """Mint a fresh hybrid did:web identity.

    Args:
        domain: bare hostname (no scheme/port). E.g. ``"example.com"``.
        slug: agent slug, used as the first path segment of the DID.
            Resolves to ``https://<domain>/<slug>/did.json``.
        extra_path_segments: optional additional path segments after
            ``slug`` (e.g. ``("v1",)`` → ``did:web:example.com:slug:v1``).
        classical_alg: classical suite id. Default ``ed25519``.
        pq_alg: post-quantum suite id. Default ``ml-dsa-65``.
        also_known_as: optional ``alsoKnownAs`` entries (e.g. a legacy
            ``did:pkh`` carried forward during succession migration).
        services: optional service endpoints (Wave 4 CAR / capsule
            sharing will populate this).

    Returns:
        :class:`HybridDidWebIdentity` with the DID URI, DID document
        ready to publish, and HybridKeypair to safeguard.
    """
    if not slug:
        from kestrel_sovereign.identity.did_web import DidWebError
        raise DidWebError("slug must be non-empty")

    keypair = generate_hybrid_keypair(
        classical_alg=classical_alg,
        pq_alg=pq_alg,
    )
    path_segments = [slug, *extra_path_segments]
    did = build_did(domain, path_segments)
    did_document = build_did_document(
        did,
        keypair.public_keys(),
        also_known_as=also_known_as,
        services=services,
    )

    return HybridDidWebIdentity(
        did=did,
        did_document=did_document,
        keypair=keypair,
    )


__all__ = [
    "HybridDidWebIdentity",
    "create_did_web_identity",
]
