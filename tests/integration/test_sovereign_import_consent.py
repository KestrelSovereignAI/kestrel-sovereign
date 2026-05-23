"""
Integration tests for #1379 — sovereignty-CAR consent gate wired into
``SovereignStorageAdapter.import_agent(grant=...)``.

The unit tests in ``tests/unit/test_car_import_consent.py`` cover the
primitive (``verify_car_import_consent``). These tests cover the WIRING
through the storage adapter:

- ``grant=None`` preserves pre-#1379 behavior bit-for-bit.
- valid grant + matching host_did + matching source → import succeeds.
- consent rejection leaves the host DB UNTOUCHED (same invariant as
  the existing continuity-gate rejection path).
- audit log records the canonical grant_id on both the success and
  rejection paths.
- host_policy False rejects an otherwise-valid grant.
- ``grant`` provided without ``host_did`` fails closed.
- ``host_did`` without a grant is ignored.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from kestrel_sovereign.filecoin_adapter import StorageTier
from kestrel_sovereign.identity.access_grant import (
    DataAccessGrant,
    compute_grant_id,
    finalize,
    sign_owner,
)
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def _pkh_agent():
    from kestrel_sovereign.inception_service import (
        public_key_to_ethereum_address,
    )
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    address = public_key_to_ethereum_address(kp.public_key)
    did = f"did:pkh:eip155:1:{address}"
    vms = build_verification_methods(did, [(secp, kp.public_key)])
    kid = vms[0]["id"].rsplit("#", 1)[-1]
    return {"did": did, "kp": kp, "kid": kid, "vms": vms}


@pytest.fixture(scope="module")
def owner():
    return _pkh_agent()


def _grant(owner, source_did: str, host_did: str, *, purpose: str = "1379 test"):
    g = DataAccessGrant(
        owner_did=owner["did"],
        source_did=source_did,
        host_did=host_did,
        issued_at="2026-05-23T00:00:00+00:00",
        purpose=purpose,
        owner_verification_methods=owner["vms"],
    )
    return finalize(sign_owner(g, [(owner["kp"], owner["kid"])]))


async def _seed_and_export(storage, *, source_did: str, secret: str = "1379-secret"):
    """Add a sentinel message and produce a sovereignty CAR CID."""
    await storage.add_conversation(
        "user", "seed message",
        metadata={"timestamp": "2026-05-23T10:00:00Z"},
    )
    adapter = SovereignStorageAdapter(storage.db, user_secret=secret)
    cid = await adapter.export_agent(source_did, storage_tier=StorageTier.LOCAL_ONLY)
    return adapter, cid


# ---------------------------------------------------------------------------
# Wiring tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grant_none_preserves_existing_behavior(temp_db):
    """No grant → identical pre-#1379 import path runs."""
    source_did = "did:pkh:eip155:1:0xa1" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        # Wipe conversations so the import has something to do.
        await storage.db.execute_commit("DELETE FROM conversation_history")
        result = await adapter.import_agent(cid)
        assert result.success is True
        assert result.status == "imported"


@pytest.mark.asyncio
async def test_valid_grant_passes_and_audit_records_grant_id(
    temp_db, owner,
):
    source_did = "did:pkh:eip155:1:0xb2" + "00" * 19
    host_did = "did:pkh:eip155:1:0xb3" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        grant = _grant(owner, source_did=source_did, host_did=host_did)
        await storage.db.execute_commit("DELETE FROM conversation_history")

        result = await adapter.import_agent(
            cid, grant=grant, host_did=host_did,
        )
        assert result.success is True
        assert result.status == "imported"

        # Audit row records the canonical grant id (verifier-recomputed,
        # not the spoofable grant.grant_id field).
        log = await adapter.get_import_log(limit=5)
        latest = log[0]
        assert latest["status"] == "imported"
        assert latest["grant_id"] == compute_grant_id(grant)


@pytest.mark.asyncio
async def test_grant_host_mismatch_rejected_db_untouched(temp_db, owner):
    """A grant naming the wrong host DID is refused; the host DB is left
    untouched (same invariant as the continuity-rejection path)."""
    source_did = "did:pkh:eip155:1:0xc1" + "00" * 19
    real_host = "did:pkh:eip155:1:0xc2" + "00" * 19
    wrong_host = "did:pkh:eip155:1:0xc3" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        # Grant targets wrong_host but we ask to import as real_host.
        grant = _grant(owner, source_did=source_did, host_did=wrong_host)

        before = await storage.get_conversation_history()
        assert len(before) == 1

        result = await adapter.import_agent(
            cid, grant=grant, host_did=real_host,
        )
        assert result.success is False
        assert result.status == "rejected"
        assert "grant_targets_different_host" in (result.reject_reason or "")

        # Host DB untouched.
        after = await storage.get_conversation_history()
        assert len(after) == 1
        assert after[0]["content"] == "seed message"


@pytest.mark.asyncio
async def test_grant_source_mismatch_rejected(temp_db, owner):
    source_did = "did:pkh:eip155:1:0xd1" + "00" * 19
    host_did = "did:pkh:eip155:1:0xd2" + "00" * 19
    other_source = "did:pkh:eip155:1:0xd3" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        # Grant names a DIFFERENT source than the manifest carries.
        grant = _grant(owner, source_did=other_source, host_did=host_did)

        result = await adapter.import_agent(
            cid, grant=grant, host_did=host_did,
        )
        assert result.success is False
        assert "grant_names_different_source" in (result.reject_reason or "")


@pytest.mark.asyncio
async def test_revoked_grant_rejected(temp_db, owner):
    source_did = "did:pkh:eip155:1:0xe1" + "00" * 19
    host_did = "did:pkh:eip155:1:0xe2" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        grant = _grant(owner, source_did=source_did, host_did=host_did)
        revoked_id = compute_grant_id(grant)

        result = await adapter.import_agent(
            cid, grant=grant, host_did=host_did,
            revoked_grant_ids={revoked_id},
        )
        assert result.success is False
        assert "grant_expired_or_revoked" in (result.reject_reason or "")

        # Audit row records the same canonical id that was revoked.
        log = await adapter.get_import_log(limit=5)
        assert log[0]["grant_id"] == revoked_id
        assert log[0]["status"] == "rejected"


@pytest.mark.asyncio
async def test_grant_without_host_did_fails_closed(temp_db, owner):
    """A grant without a paired host_did has no receiver DID to bind
    against — fail closed rather than fall back to trusting the grant."""
    source_did = "did:pkh:eip155:1:0xf1" + "00" * 19
    host_did = "did:pkh:eip155:1:0xf2" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        grant = _grant(owner, source_did=source_did, host_did=host_did)

        result = await adapter.import_agent(cid, grant=grant)  # no host_did
        assert result.success is False
        assert result.status == "rejected"
        assert "host_did" in (result.reject_reason or "")


@pytest.mark.asyncio
async def test_host_did_without_grant_is_ignored(temp_db):
    """``host_did`` without a grant is informational only — the
    pre-#1379 path runs unchanged."""
    source_did = "did:pkh:eip155:1:0x10" + "00" * 19
    host_did = "did:pkh:eip155:1:0x11" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        await storage.db.execute_commit("DELETE FROM conversation_history")
        result = await adapter.import_agent(cid, host_did=host_did)
        assert result.success is True
        assert result.status == "imported"


@pytest.mark.asyncio
async def test_host_policy_rejects_valid_grant(temp_db, owner):
    """host_policy runs AFTER consent verifies; returning False rejects
    with a distinct host_policy_rejected reason. The policy receives
    the verifier-recomputed canonical id, not grant.grant_id."""
    source_did = "did:pkh:eip155:1:0x21" + "00" * 19
    host_did = "did:pkh:eip155:1:0x22" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        grant = _grant(owner, source_did=source_did, host_did=host_did)

        seen: dict = {}
        def policy(g: DataAccessGrant, canonical_id: str) -> bool:
            seen["canonical_id"] = canonical_id
            seen["spoofed_id"] = g.grant_id
            return False

        result = await adapter.import_agent(
            cid, grant=grant, host_did=host_did, host_policy=policy,
        )
        assert result.success is False
        assert "host_policy_rejected" in (result.reject_reason or "")
        # Policy received the canonical id, not whatever grant.grant_id holds.
        assert seen["canonical_id"] == compute_grant_id(grant)


@pytest.mark.asyncio
async def test_host_policy_not_consulted_when_consent_fails(temp_db, owner):
    """If consent fails, host_policy MUST NOT be called — the rejection
    reason must be the consent failure, not the policy."""
    source_did = "did:pkh:eip155:1:0x31" + "00" * 19
    host_did = "did:pkh:eip155:1:0x32" + "00" * 19
    wrong_host = "did:pkh:eip155:1:0x33" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        # Mint grant for wrong_host so consent fails before policy runs.
        grant = _grant(owner, source_did=source_did, host_did=wrong_host)

        calls = {"n": 0}
        def policy(g, canonical_id):
            calls["n"] += 1
            return True

        result = await adapter.import_agent(
            cid, grant=grant, host_did=host_did, host_policy=policy,
        )
        assert result.success is False
        assert calls["n"] == 0
        assert "grant_targets_different_host" in (result.reject_reason or "")


@pytest.mark.asyncio
async def test_audit_log_grant_id_on_rejection(temp_db, owner):
    """Consent-rejection audit rows still record the canonical grant id
    so auditors can trace which grant was offered, even when refused."""
    source_did = "did:pkh:eip155:1:0x41" + "00" * 19
    host_did = "did:pkh:eip155:1:0x42" + "00" * 19
    wrong_host = "did:pkh:eip155:1:0x43" + "00" * 19
    async with Storage(db_path=temp_db) as storage:
        adapter, cid = await _seed_and_export(storage, source_did=source_did)
        grant = _grant(owner, source_did=source_did, host_did=wrong_host)

        await adapter.import_agent(
            cid, grant=grant, host_did=host_did,
        )
        log = await adapter.get_import_log(limit=5)
        assert log[0]["status"] == "rejected"
        # canonical id is what audit sees, regardless of what
        # grant.grant_id holds.
        assert log[0]["grant_id"] == compute_grant_id(grant)
