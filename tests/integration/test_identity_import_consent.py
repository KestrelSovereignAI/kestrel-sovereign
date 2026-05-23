"""
Integration tests for #1273 — `verify_import_consent` wired into
`IdentityImporter.import_package`.

Goal: confirm the consent gate gates the import flow correctly when a
``grant=`` is provided, without disturbing the pre-#1273 behavior when
no grant is given. The primitive itself has full unit-test coverage in
``tests/unit/test_access_grant.py``; these tests cover the WIRING:

- ``grant=None`` preserves existing behavior bit-for-bit
- valid grant + valid package + matching host_did → import proceeds
- grant.host_did ≠ target_agent_id → rejects with the named reason
- grant.source_did ≠ package.did → rejects
- revoked grant rejects
- target_agent_id missing → fails-closed with a clear error
- host_policy callable evaluated AFTER consent ok=True; False rejects

The package-side signature check is isolated to a stub (monkey-patched)
so this file doesn't have to set up the full hybrid-signing chain —
that path is covered by the project's existing signing tests.
"""

from __future__ import annotations

import hashlib
import pytest
import pytest_asyncio
import tempfile
from pathlib import Path

from kestrel_sovereign.identity import (
    AgentIdentityPackage,
    IdentityImporter,
    SubstrateType,
)
from kestrel_sovereign.identity.access_grant import (
    DataAccessGrant,
    finalize,
    sign_owner,
)
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
from kestrel_sovereign.storage.async_database import AsyncDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_db():
    """Minimal SQLite test database with the tables the importer touches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_consent.db"
        db = await AsyncDatabase.sqlite(str(db_path))
        await db.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT,
                label TEXT,
                properties TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT,
                PRIMARY KEY (source_id, target_id, label)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memory_episodes (
                id TEXT PRIMARY KEY, agent_id TEXT, title TEXT,
                summary TEXT, timespan_start TEXT, timespan_end TEXT,
                key_message_ids TEXT, emotional_arc TEXT, created_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS saved_items (
                id TEXT PRIMARY KEY, agent_id TEXT, item_type TEXT,
                name TEXT, summary TEXT, content TEXT, content_hash TEXT,
                ipfs_cid TEXT, source_type TEXT, source_ref TEXT,
                schema_id TEXT, tags TEXT, metadata TEXT,
                created_at TEXT, updated_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS temporal_patterns (
                id TEXT PRIMARY KEY, agent_id TEXT, pattern_type TEXT,
                description TEXT, trigger_conditions TEXT,
                confidence REAL, observations INTEGER,
                created_at TEXT, updated_at TEXT
            )
        """)
        await db.commit()
        yield db
        await db.close()


def _pkh_agent():
    """One legacy did:pkh:eip155 agent with a single secp256k1 key."""
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


@pytest.fixture(scope="module")
def source():
    return _pkh_agent()


@pytest.fixture(scope="module")
def host():
    return _pkh_agent()


@pytest.fixture
def signed_package(source):
    """A minimal AgentIdentityPackage with the source's DID. The
    package-signature check is isolated via monkeypatch so we don't
    need to set up the full hybrid-signing chain here."""
    pkg = AgentIdentityPackage(
        did=source["did"],
        agent_name="Consent Test Agent",
        created_at="2026-05-23T00:00:00Z",
        constitution_hash="",
        constitution_text="",
        source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
    )
    pkg.content_hash = pkg.compute_content_hash()
    # Mark as "signed" so the importer's existing has_signature branch
    # runs; the actual cryptographic verification is monkey-patched
    # below to focus these tests on the consent WIRING.
    pkg.signature = "deadbeef"
    return pkg


@pytest.fixture
def signed_grant(owner, source, host):
    g = DataAccessGrant(
        owner_did=owner["did"],
        source_did=source["did"],
        host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        purpose="integration test",
        owner_verification_methods=owner["vms"],
    )
    g = sign_owner(g, [(owner["kp"], owner["kid"])])
    return finalize(g)


@pytest.fixture
def stub_package_sig_ok(monkeypatch):
    """Isolate the package-side signature check: pretend the package
    signs correctly. The grant-side checks (the gate this PR adds) still
    run for real.
    """
    from kestrel_sovereign.identity import access_grant

    async def _ok(package):
        return True, "stubbed ok"
    monkeypatch.setattr(
        access_grant, "_verify_package_signed_by_source", _ok,
    )

    # Also stub the importer's OWN signature check so verify_signature=True
    # doesn't try to load a real private key from disk.
    async def _verify_sig(self, package):
        return True
    monkeypatch.setattr(
        IdentityImporter, "_verify_signature", _verify_sig,
    )


# ---------------------------------------------------------------------------
# Wiring tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grant_none_preserves_existing_behavior(test_db, signed_package):
    """When no grant is provided, behavior is identical to pre-#1273."""
    importer = IdentityImporter(test_db, target_agent_id=signed_package.did)
    result = await importer.import_package(signed_package, verify_signature=False)
    assert result.success is True


@pytest.mark.asyncio
async def test_valid_grant_passes_import(
    test_db, stub_package_sig_ok, signed_package, signed_grant, host,
):
    importer = IdentityImporter(test_db, target_agent_id=host["did"])
    result = await importer.import_package(signed_package, grant=signed_grant)
    assert result.success is True, getattr(result, "errors", None)


@pytest.mark.asyncio
async def test_grant_targets_different_host_rejected(
    test_db, stub_package_sig_ok, signed_package, signed_grant,
):
    other_host = "did:pkh:eip155:1:0x" + "11" * 20
    importer = IdentityImporter(test_db, target_agent_id=other_host)
    result = await importer.import_package(signed_package, grant=signed_grant)
    assert result.success is False
    assert any(
        "grant_targets_different_host" in e for e in result.errors
    ), result.errors


@pytest.mark.asyncio
async def test_grant_names_different_source_rejected(
    test_db, stub_package_sig_ok, owner, host, signed_package,
):
    # Grant names a different source than the package's DID.
    other_source = "did:pkh:eip155:1:0x" + "22" * 20
    bad_grant = DataAccessGrant(
        owner_did=owner["did"],
        source_did=other_source,
        host_did=host["did"],
        issued_at="2026-05-23T00:00:00+00:00",
        owner_verification_methods=owner["vms"],
    )
    bad_grant = finalize(sign_owner(bad_grant, [(owner["kp"], owner["kid"])]))
    importer = IdentityImporter(test_db, target_agent_id=host["did"])
    result = await importer.import_package(signed_package, grant=bad_grant)
    assert result.success is False
    assert any("grant_names_different_source" in e for e in result.errors)


@pytest.mark.asyncio
async def test_revoked_grant_ids_rejects(
    test_db, stub_package_sig_ok, signed_package, signed_grant, host,
):
    from kestrel_sovereign.identity.access_grant import compute_grant_id

    canonical_id = compute_grant_id(signed_grant)
    importer = IdentityImporter(test_db, target_agent_id=host["did"])
    result = await importer.import_package(
        signed_package, grant=signed_grant,
        revoked_grant_ids={canonical_id},
    )
    assert result.success is False
    assert any("grant_expired_or_revoked" in e for e in result.errors)


@pytest.mark.asyncio
async def test_grant_requires_target_agent_id(
    test_db, stub_package_sig_ok, signed_package, signed_grant,
):
    """A grant without target_agent_id has no host_did to check against."""
    importer = IdentityImporter(test_db, target_agent_id=None)
    result = await importer.import_package(signed_package, grant=signed_grant)
    assert result.success is False
    assert any(
        "target_agent_id" in e and "consent grant" in e
        for e in result.errors
    ), result.errors


@pytest.mark.asyncio
async def test_host_policy_rejects_otherwise_valid_grant(
    test_db, stub_package_sig_ok, signed_package, signed_grant, host,
):
    """host_policy is an additional filter ON TOP of a valid grant —
    never a substitute. If consent verifies but host_policy returns
    False, the import is rejected with a distinct host_policy_rejected
    reason.
    """
    def deny_all(grant):
        return False
    importer = IdentityImporter(test_db, target_agent_id=host["did"])
    result = await importer.import_package(
        signed_package, grant=signed_grant, host_policy=deny_all,
    )
    assert result.success is False
    assert any("host_policy_rejected" in e for e in result.errors)


@pytest.mark.asyncio
async def test_host_policy_not_consulted_when_consent_fails(
    test_db, stub_package_sig_ok, signed_package, signed_grant,
):
    """If the grant fails consent verification, host_policy is NOT
    asked — the rejection reason must be the consent failure, not the
    policy. Enforces the ordering invariant from #1273.
    """
    other_host = "did:pkh:eip155:1:0x" + "33" * 20
    called = {"n": 0}
    def policy(grant):
        called["n"] += 1
        return True
    importer = IdentityImporter(test_db, target_agent_id=other_host)
    result = await importer.import_package(
        signed_package, grant=signed_grant, host_policy=policy,
    )
    assert result.success is False
    assert called["n"] == 0
    assert any("grant_targets_different_host" in e for e in result.errors)
