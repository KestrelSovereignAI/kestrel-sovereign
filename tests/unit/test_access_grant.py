"""
Data-access grant schema + verification tests (#1273).

Covers:
- Canonical signable payload: byte-stable, sorted-key, deterministic
- compute_grant_id stable across to_dict/from_dict round-trips
- sign_owner produces v2-array signatures keyed by alg
- finalize stamps grant_id + created_at without invalidating signatures
- verify_import_consent happy path: all five named checks pass
- verify_import_consent reject paths exercise each named check
  independently and surface distinct reason codes:
    package_signature_invalid
    grant_signature_invalid
    grant_names_different_source
    grant_targets_different_host
    grant_expired_or_revoked
- Expiry semantics: missing expires_at => no expiry; malformed =>
  treated as expired (fail-closed); future => not expired
- Revocation: revoked=True rejects without invalidating signatures
- Owner DID binding: an attacker embedding their own VMs alongside
  the victim's DID is rejected via verify_did_binding inside
  _verify_owner_signatures
- Tampered grant after signing: signature no longer verifies
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from kestrel_sovereign.identity.access_grant import (
    ConsentVerification,
    DataAccessGrant,
    REJECT_GRANT_EXPIRED_OR_REVOKED,
    REJECT_GRANT_NAMES_DIFFERENT_SOURCE,
    REJECT_GRANT_SIGNATURE,
    REJECT_GRANT_TARGETS_DIFFERENT_HOST,
    REJECT_PACKAGE_SIGNATURE,
    compute_grant_id,
    finalize,
    signable_payload,
    sign_owner,
    verify_import_consent,
)
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite


# ---------------------------------------------------------------------------
# Fixtures — owner / source / host
# ---------------------------------------------------------------------------

def _pkh_agent(label: str):
    """One legacy did:pkh:eip155 agent with a single secp256k1 key.

    Returns a dict ready to drop into a DataAccessGrant's
    *_verification_methods or to use as the source/host DID.
    """
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
    """A fully signed + finalized DataAccessGrant naming owner/source/host."""
    grant = DataAccessGrant(
        owner_did=owner["did"],
        source_did=source["did"],
        host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        purpose="unit test",
        owner_verification_methods=owner["vms"],
    )
    grant = sign_owner(grant, [(owner["kp"], owner["kid"])])
    return finalize(grant)


class _StubPackage:
    """Minimal package shape for primitive-level tests. The package-
    signature check is monkey-patched out so we exercise the
    primitive's composition logic rather than ``verify_package_signature``
    (which has its own dedicated test suite)."""

    def __init__(self, did: str):
        self.did = did
        self.signature = ""
        self.signatures: list = []
        self.verification_methods: list = []
        self.content_hash = ""


def _stub_package_verifier(result_ok: bool, reason: str = ""):
    async def fake_verify(package: Any):
        return result_ok, reason
    return fake_verify


@pytest.fixture
def package_signed_ok(monkeypatch):
    """Treat the package-signature check as passing so the test can
    isolate the GRANT-side logic. Returns a function the test calls
    with a package."""
    from kestrel_sovereign.identity import access_grant

    monkeypatch.setattr(
        access_grant,
        "_verify_package_signed_by_source",
        _stub_package_verifier(True),
    )


@pytest.fixture
def package_signed_invalid(monkeypatch):
    from kestrel_sovereign.identity import access_grant

    monkeypatch.setattr(
        access_grant,
        "_verify_package_signed_by_source",
        _stub_package_verifier(False, "bad sig"),
    )


# ---------------------------------------------------------------------------
# Canonical payload + grant_id
# ---------------------------------------------------------------------------

def test_signable_payload_is_deterministic(signed_grant):
    a = signable_payload(signed_grant)
    b = signable_payload(signed_grant)
    assert a == b
    # sorted-key compact JSON: keys ordered alphabetically, no
    # separator whitespace (string *values* may contain spaces, so
    # we only assert against the separator forms).
    assert b'"owner_did":' in a
    assert b", " not in a
    assert b'": "' not in a


def test_compute_grant_id_excludes_id_and_created_at(owner, source, host):
    g1 = DataAccessGrant(
        owner_did=owner["did"], source_did=source["did"], host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        owner_verification_methods=owner["vms"],
    )
    id_before_finalize = compute_grant_id(g1)
    g2 = finalize(g1)
    # finalize stamps grant_id and created_at; neither is in the signed
    # payload, so the computed id is unchanged.
    assert g2.grant_id == id_before_finalize
    assert compute_grant_id(g2) == id_before_finalize


def test_to_from_dict_roundtrip_preserves_signatures(signed_grant):
    restored = DataAccessGrant.from_dict(signed_grant.to_dict())
    assert restored == signed_grant
    assert restored.owner_signatures == signed_grant.owner_signatures


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_all_five_checks_pass(
    package_signed_ok, signed_grant, source, host,
):
    package = _StubPackage(did=source["did"])
    result = await verify_import_consent(
        package, signed_grant, host_did=host["did"],
    )
    assert isinstance(result, ConsentVerification)
    assert result.ok is True
    assert result.package_signed_by_source is True
    assert result.grant_signed_by_owner is True
    assert result.grant_names_source is True
    assert result.grant_targets_host is True
    assert result.grant_not_expired_or_revoked is True
    assert result.reason == "consent verified"


# ---------------------------------------------------------------------------
# Reject paths — each named check exercised independently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_package_signature_invalid_rejected(
    package_signed_invalid, signed_grant, source, host,
):
    package = _StubPackage(did=source["did"])
    result = await verify_import_consent(
        package, signed_grant, host_did=host["did"],
    )
    assert result.ok is False
    assert result.package_signed_by_source is False
    assert REJECT_PACKAGE_SIGNATURE in result.reason
    # The other checks still ran — composition isn't short-circuit.
    assert result.grant_signed_by_owner is True
    assert result.grant_names_source is True


@pytest.mark.asyncio
async def test_unsigned_grant_rejected(
    package_signed_ok, owner, source, host,
):
    unsigned = DataAccessGrant(
        owner_did=owner["did"], source_did=source["did"], host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        owner_verification_methods=owner["vms"],
    )
    unsigned = finalize(unsigned)  # signed=False; finalize still works
    package = _StubPackage(did=source["did"])
    result = await verify_import_consent(
        package, unsigned, host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_signed_by_owner is False
    assert REJECT_GRANT_SIGNATURE in result.reason


@pytest.mark.asyncio
async def test_tampered_grant_signature_no_longer_verifies(
    package_signed_ok, signed_grant, source, host,
):
    # Build a tampered copy with a different purpose AFTER signing.
    from dataclasses import replace
    tampered = replace(signed_grant, purpose="tampered " + signed_grant.purpose)
    package = _StubPackage(did=source["did"])
    result = await verify_import_consent(
        package, tampered, host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_signed_by_owner is False


@pytest.mark.asyncio
async def test_grant_names_different_source_rejected(
    package_signed_ok, signed_grant, host,
):
    # Package's DID doesn't match the grant's source_did.
    package = _StubPackage(did="did:pkh:eip155:1:0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    result = await verify_import_consent(
        package, signed_grant, host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_names_source is False
    assert REJECT_GRANT_NAMES_DIFFERENT_SOURCE in result.reason


@pytest.mark.asyncio
async def test_grant_targets_different_host_rejected(
    package_signed_ok, signed_grant, source,
):
    result = await verify_import_consent(
        _StubPackage(did=source["did"]),
        signed_grant,
        host_did="did:pkh:eip155:1:0x0000000000000000000000000000000000000001",
    )
    assert result.ok is False
    assert result.grant_targets_host is False
    assert REJECT_GRANT_TARGETS_DIFFERENT_HOST in result.reason


@pytest.mark.asyncio
async def test_revoked_grant_rejected(
    package_signed_ok, signed_grant, source, host,
):
    from dataclasses import replace
    revoked = replace(signed_grant, revoked=True)
    # revoked is NOT a signed field, so the signature still verifies
    # but the consent gate fails.
    result = await verify_import_consent(
        _StubPackage(did=source["did"]),
        revoked,
        host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_signed_by_owner is True  # signature unaffected
    assert result.grant_not_expired_or_revoked is False
    assert REJECT_GRANT_EXPIRED_OR_REVOKED in result.reason


@pytest.mark.asyncio
async def test_expired_grant_rejected(
    package_signed_ok, owner, source, host,
):
    past = "2020-01-01T00:00:00+00:00"
    expired = DataAccessGrant(
        owner_did=owner["did"], source_did=source["did"], host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        expires_at=past,
        purpose="expired",
        owner_verification_methods=owner["vms"],
    )
    expired = sign_owner(expired, [(owner["kp"], owner["kid"])])
    expired = finalize(expired)
    result = await verify_import_consent(
        _StubPackage(did=source["did"]),
        expired,
        host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_not_expired_or_revoked is False
    assert REJECT_GRANT_EXPIRED_OR_REVOKED in result.reason


@pytest.mark.asyncio
async def test_future_expiry_accepted(
    package_signed_ok, owner, source, host,
):
    future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
    grant = DataAccessGrant(
        owner_did=owner["did"], source_did=source["did"], host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        expires_at=future, purpose="future",
        owner_verification_methods=owner["vms"],
    )
    grant = finalize(sign_owner(grant, [(owner["kp"], owner["kid"])]))
    result = await verify_import_consent(
        _StubPackage(did=source["did"]),
        grant,
        host_did=host["did"],
    )
    assert result.ok is True


@pytest.mark.asyncio
async def test_malformed_expires_at_treated_as_expired(
    package_signed_ok, owner, source, host,
):
    grant = DataAccessGrant(
        owner_did=owner["did"], source_did=source["did"], host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        expires_at="not-a-real-timestamp",
        purpose="malformed expiry",
        owner_verification_methods=owner["vms"],
    )
    grant = finalize(sign_owner(grant, [(owner["kp"], owner["kid"])]))
    result = await verify_import_consent(
        _StubPackage(did=source["did"]),
        grant,
        host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_not_expired_or_revoked is False


@pytest.mark.asyncio
async def test_owner_did_not_bound_to_vms_rejected(
    package_signed_ok, source, host,
):
    """Attacker scenario: claim a victim owner's DID but embed an
    attacker-controlled VM and sign with the attacker's key. Without
    verify_did_binding inside _verify_owner_signatures, the signature
    crypto-verifies (against the attacker's own key) and the grant
    falsely succeeds. With binding enforced, the DID-mismatch is
    caught and grant_signed_by_owner=False.
    """
    attacker = _pkh_agent("attacker")
    victim_owner_did = "did:pkh:eip155:1:0x" + "11" * 20

    # Grant claims the victim's DID but embeds the attacker's VMs.
    grant = DataAccessGrant(
        owner_did=victim_owner_did,
        source_did=source["did"], host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        purpose="impersonation attempt",
        owner_verification_methods=attacker["vms"],
    )
    grant = finalize(sign_owner(grant, [(attacker["kp"], attacker["kid"])]))

    result = await verify_import_consent(
        _StubPackage(did=source["did"]),
        grant,
        host_did=host["did"],
    )
    assert result.ok is False
    assert result.grant_signed_by_owner is False
