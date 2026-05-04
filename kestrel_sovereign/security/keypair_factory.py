"""
KeypairFactory — uniform keypair lifecycle across CryptoSuites.

Wave 1 sub-PR 2 of the Quantum Hardening epic (#921, #916). Thin
orchestration layer over the suite registry: callers say
``KeypairFactory.generate("ed25519")`` once Wave 2 lands and get back
the right ``Keypair`` without importing the suite's module directly.

The factory is intentionally minimal in this PR. Wave 2+ extends it
with hybrid-keypair generation (multiple suites in one logical
identity), keystore round-trips, and Multikey-DID-document helpers.
For now it owns:

- ``generate(suite_id)`` — delegates to the registered suite.
- ``generate_default()`` — returns a keypair for the project's default
  suite. The default is currently ``Secp256k1Suite``; once Wave 2's
  hybrid identity ships, the default flips to a composite multi-suite
  shape.
- Multibase helpers re-exported for convenience.

By keeping the factory thin and registry-driven, the migration in Wave
1 sub-PR 5 only needs to touch four call-sites once; future suites
plug in via ``register_suite`` without ever editing this module.
"""

from __future__ import annotations

from typing import Optional

from .crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    CryptoSuite,
    CryptoSuiteError,
    Keypair,
    get_suite,
    list_registered,
)
from .multikey import (
    multibase_to_public_key,
    public_key_to_multibase,
)


# The default suite id used when the caller doesn't specify one. This
# stays at secp256k1 through Wave 1; Wave 2's hybrid-identity work
# bumps the default to a composite (Ed25519 + ML-DSA-65) shape.
DEFAULT_SUITE_ID = ALG_ECDSA_SECP256K1_SHA256


class KeypairFactory:
    """Suite-aware keypair generation entry point.

    Stateless: every method is effectively a static dispatch through
    the registry. Implemented as a class with classmethods rather than
    free functions to keep the API namespace organized as Waves 2-4
    add hybrid generation, keystore I/O, and Multikey-document helpers.
    """

    DEFAULT_SUITE_ID = DEFAULT_SUITE_ID

    @classmethod
    def generate(cls, suite_id: Optional[str] = None) -> Keypair:
        """Generate a fresh keypair from the named suite.

        Raises ``CryptoSuiteError`` if the suite is not registered.
        """
        sid = suite_id or cls.DEFAULT_SUITE_ID
        suite = get_suite(sid)
        return suite.generate_keypair()

    @classmethod
    def generate_default(cls) -> Keypair:
        """Generate a keypair from the current default suite.

        The default is project-wide policy. Wave 1 keeps it at
        ``ecdsa-secp256k1-sha256`` so existing callers see no behavior
        change; Wave 2 flips it to a composite hybrid shape.
        """
        return cls.generate(cls.DEFAULT_SUITE_ID)

    @classmethod
    def public_key_to_multibase(cls, keypair: Keypair) -> str:
        """Encode a keypair's public key as a W3C Multikey string.

        Convenience wrapper around ``multikey.public_key_to_multibase``
        that pulls the suite from the keypair's ``suite_id``.
        """
        suite = get_suite(keypair.suite_id)
        return public_key_to_multibase(suite, keypair.public_key)

    @classmethod
    def multibase_to_public_key(cls, multibase_str: str):
        """Decode a Multikey string back to ``(suite_id, public_key)``."""
        suite, pub = multibase_to_public_key(multibase_str)
        return suite.alg_id, pub

    @classmethod
    def registered_suite_ids(cls) -> list[str]:
        """All currently-registered suite ids. Useful for diagnostics."""
        return list_registered()


__all__ = [
    "DEFAULT_SUITE_ID",
    "KeypairFactory",
]
