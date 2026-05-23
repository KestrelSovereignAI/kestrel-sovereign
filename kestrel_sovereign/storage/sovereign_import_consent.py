"""
Sovereignty-CAR import-consent wiring (#1379, follow-up to #1273).

The #1273 primitive :func:`verify_import_consent` is shaped for
:class:`AgentIdentityPackage`: it calls
:func:`identity.signing.verify_package_signature` to attest "this
package was signed by ``package.did``." A **sovereignty CAR** doesn't
carry a signature field; its source-attestation is structural —
:class:`SovereignImportVerifier` checks that every CAR block hashes to
its CID, that the root manifest is well-formed, and that the keyring
decrypts under the source's user secret (which only the source could
have produced).

That structural proof IS the CAR-side equivalent of the package
signature. By the time this module is reached, the caller has already
run :func:`SovereignImportVerifier.verify` and obtained a verified
manifest. The remaining gate is *owner consent* — the grant-only
checks: owner signature, source-binding, host-binding, expiry,
revocation. Those are exactly what :func:`access_grant.verify_grant`
provides.

This module:

  * exports :func:`verify_car_import_consent`, the CAR-side wrapper
    that calls :func:`verify_grant` with ``source_did = manifest.agent_did``
    and ``package_signed_by_source = True`` (the caller's CAR-integrity
    contract).
  * does NOT re-implement the CAR-integrity check — that stays in
    :class:`SovereignImportVerifier`. Calling this function without
    first running ``SovereignImportVerifier.verify(car_bytes)`` would
    skip CAR integrity entirely, so the contract is "verify first,
    consent second."
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Optional

from kestrel_sovereign.identity.access_grant import (
    ConsentVerification,
    DataAccessGrant,
    verify_grant,
)


async def verify_car_import_consent(
    manifest: Any,
    grant: DataAccessGrant,
    *,
    host_did: str,
    revoked_grant_ids: Optional[Iterable[str]] = None,
    did_web_resolver: Optional[Any] = None,
    now: Optional[datetime] = None,
) -> ConsentVerification:
    """Verify a :class:`DataAccessGrant` authorizes restoring a
    sovereignty CAR into the agent identified by *host_did*.

    Caller contract: ``manifest`` MUST come from a successful
    :func:`SovereignImportVerifier.verify` call. The CAR's structural
    integrity (block-hash, manifest well-formedness, keyring
    decryptability) is the CAR-side equivalent of the AgentIdentity
    package's signature attestation — that's what proves the source
    actually produced this CAR. By the time the manifest is in hand
    the source-attestation is established; this function only adds
    the owner-consent gate.

    Args:
        manifest: The verified :class:`RootManifest` returned by
            :func:`SovereignImportVerifier.verify`. Only
            ``manifest.agent_did`` is read.
        grant: The owner-signed :class:`DataAccessGrant`.
        host_did: The receiving agent's own DID. The grant's
            ``host_did`` field MUST equal this; otherwise an attacker
            could replay a grant minted for one agent against another
            on the same deployment.
        revoked_grant_ids: See :func:`verify_grant`.
        did_web_resolver: See :func:`verify_grant`.
        now: See :func:`verify_grant`.
    """
    source_did = getattr(manifest, "agent_did", "") or ""
    return await verify_grant(
        grant,
        source_did=source_did,
        host_did=host_did,
        package_signed_by_source=True,
        package_signed_reason="CAR integrity established by SovereignImportVerifier",
        revoked_grant_ids=revoked_grant_ids,
        did_web_resolver=did_web_resolver,
        now=now,
    )


__all__ = ["verify_car_import_consent"]
