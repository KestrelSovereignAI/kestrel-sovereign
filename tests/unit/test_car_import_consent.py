"""
Unit tests for #1379 — CAR-side consent verification.

Covers:
- ``verify_grant`` (the new public primitive extracted from
  ``verify_import_consent``): all five named checks fire independently
  when the caller supplies ``package_signed_by_source=True/False``
  along with ``source_did``.
- ``verify_car_import_consent`` (the CAR wrapper): composes
  ``verify_grant`` with the CAR's manifest, taking
  ``manifest.agent_did`` as the source DID and treating CAR integrity
  as the source-attestation (caller's contract).

The semantic invariant: the only difference between
``verify_import_consent`` (AgentIdentityPackage) and
``verify_car_import_consent`` (sovereignty CAR) is HOW the
"package signed by source" check is satisfied — the grant-side checks
are identical. These tests pin that invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.identity.access_grant import (
    REJECT_GRANT_EXPIRED_OR_REVOKED,
    REJECT_GRANT_NAMES_DIFFERENT_SOURCE,
    REJECT_GRANT_SIGNATURE,
    REJECT_GRANT_TARGETS_DIFFERENT_HOST,
    REJECT_PACKAGE_SIGNATURE,
    DataAccessGrant,
    compute_grant_id,
    finalize,
    sign_owner,
    verify_grant,
)
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
from kestrel_sovereign.storage.sovereign_import_consent import (
    verify_car_import_consent,
)


# ---------------------------------------------------------------------------
# Fixtures — owner / source / host (identical shape to #1273 test_access_grant)
# ---------------------------------------------------------------------------

def _pkh_agent(label: str):
    from kestrel_sovereign.inception_service import (
        public_key_to_ethereum_address,
    )
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    address = public_key_to_ethereum_address(kp.public_key)
    did = f"did:pkh:eip155:1:{address}"
    vms = build_verification_methods(did, [(secp, kp.public_key)])
    kid = vms[0]["id"].rsplit("#", 1)[-1]
    return {"did": did, "kp": kp, "kid": kid, "vms": vms, "label": label}


@pytest.fixture(scope="module")
def owner():
    return _pkh_agent("owner")


@pytest.fixture(scope="module")
def source():
    return _pkh_agent("source")


@pytest.fixture(scope="module")
def host():
    return _pkh_agent("host")


@pytest.fixture
def signed_grant(owner, source, host):
    g = DataAccessGrant(
        owner_did=owner["did"],
        source_did=source["did"],
        host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        purpose="car import test",
        owner_verification_methods=owner["vms"],
    )
    return finalize(sign_owner(g, [(owner["kp"], owner["kid"])]))


@dataclass
class _StubManifest:
    """Minimal manifest shape used by verify_car_import_consent — only
    ``agent_did`` is read."""
    agent_did: str


# ---------------------------------------------------------------------------
# verify_grant — package_signed_by_source=True path (CAR-style caller)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_grant_happy_path(signed_grant, source, host):
    """Caller asserts package_signed_by_source=True (CAR integrity
    already verified). All five checks pass."""
    result = await verify_grant(
        signed_grant,
        source_did=source["did"],
        host_did=host["did"],
        package_signed_by_source=True,
    )
    assert result.ok is True
    assert result.package_signed_by_source is True
    assert result.grant_signed_by_owner is True
    assert result.grant_names_source is True
    assert result.grant_targets_host is True
    assert result.grant_not_expired_or_revoked is True


@pytest.mark.asyncio
async def test_verify_grant_caller_signals_package_failed(
    signed_grant, source, host,
):
    """Caller can also report a package-side failure; the result then
    surfaces it through package_signed_by_source=False with the
    REJECT_PACKAGE_SIGNATURE reason code."""
    result = await verify_grant(
        signed_grant,
        source_did=source["did"],
        host_did=host["did"],
        package_signed_by_source=False,
        package_signed_reason="CAR block-hash mismatch",
    )
    assert result.ok is False
    assert result.package_signed_by_source is False
    assert REJECT_PACKAGE_SIGNATURE in result.reason
    assert "CAR block-hash mismatch" in result.reason


@pytest.mark.asyncio
async def test_verify_grant_source_mismatch(signed_grant, host):
    """Grant names source A, but caller's source_did is B → rejected."""
    other_source = "did:pkh:eip155:1:0x" + "aa" * 20
    result = await verify_grant(
        signed_grant,
        source_did=other_source,
        host_did=host["did"],
        package_signed_by_source=True,
    )
    assert result.ok is False
    assert result.grant_names_source is False
    assert REJECT_GRANT_NAMES_DIFFERENT_SOURCE in result.reason


@pytest.mark.asyncio
async def test_verify_grant_host_mismatch(signed_grant, source):
    other_host = "did:pkh:eip155:1:0x" + "bb" * 20
    result = await verify_grant(
        signed_grant,
        source_did=source["did"],
        host_did=other_host,
        package_signed_by_source=True,
    )
    assert result.ok is False
    assert result.grant_targets_host is False
    assert REJECT_GRANT_TARGETS_DIFFERENT_HOST in result.reason


@pytest.mark.asyncio
async def test_verify_grant_revocation(signed_grant, source, host):
    canonical = compute_grant_id(signed_grant)
    result = await verify_grant(
        signed_grant,
        source_did=source["did"],
        host_did=host["did"],
        package_signed_by_source=True,
        revoked_grant_ids={canonical},
    )
    assert result.ok is False
    assert result.grant_signed_by_owner is True
    assert result.grant_not_expired_or_revoked is False
    assert REJECT_GRANT_EXPIRED_OR_REVOKED in result.reason


@pytest.mark.asyncio
async def test_verify_grant_unsigned_grant(owner, source, host):
    unsigned = DataAccessGrant(
        owner_did=owner["did"], source_did=source["did"], host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        owner_verification_methods=owner["vms"],
    )
    unsigned = finalize(unsigned)
    result = await verify_grant(
        unsigned,
        source_did=source["did"],
        host_did=host["did"],
        package_signed_by_source=True,
    )
    assert result.ok is False
    assert result.grant_signed_by_owner is False
    assert REJECT_GRANT_SIGNATURE in result.reason


@pytest.mark.asyncio
async def test_verify_grant_expiry_now_override(owner, source, host):
    """The ``now`` parameter lets tests pin a clock for expiry checks."""
    expiring_at = "2027-01-01T00:00:00+00:00"
    grant = DataAccessGrant(
        owner_did=owner["did"], source_did=source["did"], host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        expires_at=expiring_at,
        owner_verification_methods=owner["vms"],
    )
    grant = finalize(sign_owner(grant, [(owner["kp"], owner["kid"])]))

    # Before expiry — ok.
    before = await verify_grant(
        grant,
        source_did=source["did"],
        host_did=host["did"],
        package_signed_by_source=True,
        now=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert before.ok is True

    # After expiry — rejected.
    after = await verify_grant(
        grant,
        source_did=source["did"],
        host_did=host["did"],
        package_signed_by_source=True,
        now=datetime(2028, 1, 1, tzinfo=timezone.utc),
    )
    assert after.ok is False
    assert after.grant_not_expired_or_revoked is False


# ---------------------------------------------------------------------------
# verify_car_import_consent — CAR-side wrapper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_car_consent_happy_path(signed_grant, source, host):
    """Manifest names the source agent; grant matches; consent passes."""
    manifest = _StubManifest(agent_did=source["did"])
    result = await verify_car_import_consent(
        manifest, signed_grant, host_did=host["did"],
    )
    assert result.ok is True
    assert result.package_signed_by_source is True  # caller's contract
    assert result.grant_signed_by_owner is True
    assert result.grant_names_source is True
    assert result.grant_targets_host is True
    assert result.grant_not_expired_or_revoked is True
    assert result.canonical_grant_id == compute_grant_id(signed_grant)


@pytest.mark.asyncio
async def test_car_consent_source_mismatch_rejected(signed_grant, host):
    """Manifest's agent_did doesn't match grant.source_did — rejected
    with the same named reason the AgentIdentityPackage path uses."""
    manifest = _StubManifest(
        agent_did="did:pkh:eip155:1:0x" + "cc" * 20
    )
    result = await verify_car_import_consent(
        manifest, signed_grant, host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_names_source is False
    assert REJECT_GRANT_NAMES_DIFFERENT_SOURCE in result.reason


@pytest.mark.asyncio
async def test_car_consent_host_mismatch_rejected(signed_grant, source):
    manifest = _StubManifest(agent_did=source["did"])
    other_host = "did:pkh:eip155:1:0x" + "dd" * 20
    result = await verify_car_import_consent(
        manifest, signed_grant, host_did=other_host,
    )
    assert result.ok is False
    assert result.grant_targets_host is False
    assert REJECT_GRANT_TARGETS_DIFFERENT_HOST in result.reason


@pytest.mark.asyncio
async def test_car_consent_revocation_rejected(signed_grant, source, host):
    canonical = compute_grant_id(signed_grant)
    manifest = _StubManifest(agent_did=source["did"])
    result = await verify_car_import_consent(
        manifest, signed_grant, host_did=host["did"],
        revoked_grant_ids={canonical},
    )
    assert result.ok is False
    assert result.grant_not_expired_or_revoked is False
    assert REJECT_GRANT_EXPIRED_OR_REVOKED in result.reason


@pytest.mark.asyncio
async def test_car_consent_tampered_grant_signature_fails(
    signed_grant, source, host,
):
    """Tampering a signed field (purpose) after signing invalidates
    the owner signature even when the manifest matches."""
    tampered = replace(signed_grant, purpose="tampered: " + signed_grant.purpose)
    manifest = _StubManifest(agent_did=source["did"])
    result = await verify_car_import_consent(
        manifest, tampered, host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_signed_by_owner is False
    assert REJECT_GRANT_SIGNATURE in result.reason


@pytest.mark.asyncio
async def test_car_consent_empty_manifest_agent_did(signed_grant, host):
    """A manifest whose agent_did is empty fails the source-binding
    check (can't bind a grant to a missing source DID)."""
    manifest = _StubManifest(agent_did="")
    result = await verify_car_import_consent(
        manifest, signed_grant, host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_names_source is False


@pytest.mark.asyncio
async def test_car_consent_spoofed_grant_id_does_not_change_canonical(
    signed_grant, source, host,
):
    """Same invariant as #1273: the verifier always recomputes the
    canonical grant id from the signable payload. A caller who spoofs
    ``grant.grant_id`` to an allowlisted value cannot change what
    audit/policy sees."""
    real_canonical = compute_grant_id(signed_grant)
    spoofed = replace(signed_grant, grant_id="allowlisted-imposter")
    manifest = _StubManifest(agent_did=source["did"])
    result = await verify_car_import_consent(
        manifest, spoofed, host_did=host["did"],
    )
    assert result.ok is True
    assert result.canonical_grant_id == real_canonical
    assert result.canonical_grant_id != "allowlisted-imposter"
