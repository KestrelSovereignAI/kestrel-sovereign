"""
Born-hybrid inception (#2397): new agents mint a hybrid did:web identity
(Ed25519 + ML-DSA-65) by default — no classical secp256k1 key ever exists.

Covers the full birth-to-boot loop:

- inception writes hybrid keys + <slug>_did.json, no legacy material
- runtime loader loads the agent WITHOUT a legacy key id
- the loaded identity signs and verifies hybrid artifacts
- fail-loud contracts: missing domain, missing KESTREL_DATA_KEY
- the classical did:pkh path remains available as an explicit opt-out
"""

import json
import os
from pathlib import Path

import pytest

from kestrel_sovereign.inception_service import (
    DID_WEB_DOMAIN_ENV,
    IDENTITY_METHOD_ENV,
    create_kestrel_identity_async,
    resolve_identity_method,
    slugify_agent_name,
)
from kestrel_sovereign.identity.runtime_identity import (
    RuntimeIdentityError,
    load_agent_identity,
)

# These tests exercise identity minting, not constitution sourcing. Pass None
# so inception anchors the packaged governing source (config.CONSTITUTION_PATH)
# — the exact bytes the periodic integrity audit recomputes. Passing the docs
# copy is now correctly refused as a non-authoritative override (#2463).
CONSTITUTION = None
TEST_DOMAIN = "agents.kestrel-sovereign.test"
TEST_DATA_KEY = "test-master-key-for-encryption-32chars!"


@pytest.fixture
def hybrid_env(monkeypatch):
    """Environment a born-hybrid mint requires: encrypted key storage
    and a did:web domain."""
    monkeypatch.setenv("KESTREL_DATA_KEY", TEST_DATA_KEY)
    monkeypatch.setenv(DID_WEB_DOMAIN_ENV, TEST_DOMAIN)
    monkeypatch.delenv(IDENTITY_METHOD_ENV, raising=False)


# ---------------------------------------------------------------------------
# Method + slug resolution
# ---------------------------------------------------------------------------

def test_default_method_is_did_web(monkeypatch):
    monkeypatch.delenv(IDENTITY_METHOD_ENV, raising=False)
    assert resolve_identity_method() == "did:web"
    assert resolve_identity_method("did:pkh") == "did:pkh"
    monkeypatch.setenv(IDENTITY_METHOD_ENV, "did:pkh")
    assert resolve_identity_method() == "did:pkh"
    # Param wins over env
    assert resolve_identity_method("did:web") == "did:web"


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown identity method"):
        resolve_identity_method("did:ethr")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Emma", "emma"),
        ("Eldercare Companion", "eldercare-companion"),
        ("Kestrel-Test-001", "kestrel-test-001"),
        ("  spaced   out  ", "spaced-out"),
    ],
)
def test_slugify_agent_name(name, expected):
    assert slugify_agent_name(name) == expected


def test_slugify_rejects_empty():
    with pytest.raises(ValueError, match="empty did:web slug"):
        slugify_agent_name("!!!")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_slug", ["Upper", "a/b", "a.b", "a%b", "", "a_b"])
async def test_explicit_slug_validated_before_any_disk_writes(tmp_path, hybrid_env, bad_slug):
    """An explicit did_web_slug is a DID segment, a filename prefix, AND a
    key-storage id — reject anything outside [a-z0-9-] before touching disk."""
    with pytest.raises(ValueError, match="did_web_slug"):
        await create_kestrel_identity_async(
            str(tmp_path), CONSTITUTION, agent_name="Slugtest",
            did_web_slug=bad_slug,
        )
    assert not (tmp_path / "kestrel_prime.db").exists()
    assert not list(tmp_path.glob("*.enc"))


@pytest.mark.asyncio
async def test_invalid_identity_method_leaves_no_db(tmp_path, hybrid_env):
    with pytest.raises(ValueError, match="Unknown identity method"):
        await create_kestrel_identity_async(
            str(tmp_path), CONSTITUTION, agent_name="Badmethod",
            identity_method="did:ethr",
        )
    assert not (tmp_path / "kestrel_prime.db").exists()


# ---------------------------------------------------------------------------
# Birth: inception mints born-hybrid by default
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inception_defaults_to_born_hybrid(tmp_path, hybrid_env):
    creds = await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Testbird",
    )

    # Derived slugs carry an entropy suffix (names aren't unique across
    # a domain); the DID prefix is deterministic.
    assert creds.agent_did.startswith(f"did:web:{TEST_DOMAIN}:testbird-")
    slug = creds.agent_did.rsplit(":", 1)[-1]

    # Hybrid material on disk, encrypted
    assert (tmp_path / f"{slug}_ed25519.key.enc").exists()
    assert (tmp_path / f"{slug}_mldsa65.bytes.enc").exists()
    assert (tmp_path / f"{slug}_archival_slhdsa.bytes.enc").exists()
    assert (tmp_path / f"{slug}_archival_slhdsa_pub.bytes.enc").exists()

    # DID document stored locally, publishable verbatim
    doc = json.loads((tmp_path / f"{slug}_did.json").read_text())
    assert doc["id"] == creds.agent_did
    vm_types = {vm["type"] for vm in doc["verificationMethod"]}
    assert vm_types == {"Multikey"}
    assert len(doc["verificationMethod"]) == 2  # Ed25519 + ML-DSA-65

    # NO classical material — the whole point
    assert not list(tmp_path.glob("kestrel_0x*"))
    assert not list(tmp_path.glob("*.pem"))
    assert not (tmp_path / "successions").exists()


@pytest.mark.asyncio
async def test_derived_slugs_are_unique_across_inceptions(tmp_path, hybrid_env):
    """Same agent name twice under one domain must NOT mint the same DID
    (a did:web URI is a public trust anchor — codex round 4 P1)."""
    a = await create_kestrel_identity_async(
        str(tmp_path / "a"), CONSTITUTION, agent_name="Kestrel Agent",
    )
    b = await create_kestrel_identity_async(
        str(tmp_path / "b"), CONSTITUTION, agent_name="Kestrel Agent",
    )
    assert a.agent_did != b.agent_did


@pytest.mark.asyncio
async def test_explicit_slug_used_verbatim(tmp_path, hybrid_env):
    creds = await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Emma", did_web_slug="emma",
    )
    assert creds.agent_did == f"did:web:{TEST_DOMAIN}:emma"
    assert (tmp_path / "emma_did.json").exists()


@pytest.mark.asyncio
async def test_born_hybrid_agent_loads_and_signs(tmp_path, hybrid_env):
    creds = await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Signer Bird",
    )

    ident = load_agent_identity(None, storage_dir=tmp_path)
    assert ident.is_hybrid
    assert ident.is_born_hybrid
    assert ident.legacy_did is None
    assert ident.legacy_keypair is None
    assert ident.signing_did == creds.agent_did
    assert ident.succession_statement is None
    assert ident.archival_keypair is not None

    from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid, verify_hybrid
    probe = b"born-hybrid-first-artifact"
    sigs = sign_hybrid(probe, ident.hybrid_keypair)
    result = verify_hybrid(probe, sigs, ident.new_verification_methods)
    assert result.ok, result.reason


@pytest.mark.asyncio
async def test_child_did_document_carries_controller(tmp_path, hybrid_env):
    await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Chick",
        parent_did=f"did:web:{TEST_DOMAIN}:parent",
    )
    docs = list(tmp_path.glob("*_did.json"))
    assert len(docs) == 1
    doc = json.loads(docs[0].read_text())
    assert doc["controller"] == f"did:web:{TEST_DOMAIN}:parent"


# ---------------------------------------------------------------------------
# Fail-loud contracts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_domain_fails_loud_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", TEST_DATA_KEY)
    monkeypatch.delenv(DID_WEB_DOMAIN_ENV, raising=False)
    monkeypatch.delenv(IDENTITY_METHOD_ENV, raising=False)

    with pytest.raises(ValueError, match=DID_WEB_DOMAIN_ENV):
        await create_kestrel_identity_async(
            str(tmp_path), CONSTITUTION, agent_name="Nodomain",
        )
    # No half-born agent left behind
    assert not (tmp_path / "kestrel_prime.db").exists()
    assert not list(tmp_path.glob("nodomain*"))


@pytest.mark.asyncio
async def test_missing_data_key_fails_loud_no_plaintext_pq(tmp_path, monkeypatch):
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    monkeypatch.setenv(DID_WEB_DOMAIN_ENV, TEST_DOMAIN)
    monkeypatch.delenv(IDENTITY_METHOD_ENV, raising=False)

    from kestrel_sovereign.security.key_storage import MasterKeyNotConfiguredError
    with pytest.raises(MasterKeyNotConfiguredError, match="KESTREL_DATA_KEY"):
        await create_kestrel_identity_async(
            str(tmp_path), CONSTITUTION, agent_name="Nokey",
        )
    # Nothing secret written in plaintext, no partial key material
    assert not list(tmp_path.glob("nokey*"))
    assert not (tmp_path / "kestrel_prime.db").exists()


# ---------------------------------------------------------------------------
# Classical opt-out unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_did_pkh_opt_out_still_mints_classical(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", TEST_DATA_KEY)
    monkeypatch.delenv(DID_WEB_DOMAIN_ENV, raising=False)

    creds = await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Classic",
        identity_method="did:pkh",
    )
    assert creds.agent_did.startswith("did:pkh:eip155:1:0x")
    assert list(tmp_path.glob("kestrel_0x*.key.enc"))
    assert list(tmp_path.glob("kestrel_0x*.json"))
    assert not list(tmp_path.glob("*_did.json"))

    key_id = sorted(tmp_path.glob("kestrel_0x*.json"))[0].stem
    ident = load_agent_identity(key_id, storage_dir=tmp_path)
    assert not ident.is_hybrid
    assert ident.legacy_did == creds.agent_did


# ---------------------------------------------------------------------------
# Loader edge states
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loader_refuses_mixed_born_hybrid_and_succession(tmp_path, hybrid_env):
    await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Mixed",
    )
    (tmp_path / "successions").mkdir()
    (tmp_path / "successions" / "mixed.json").write_text("{}")

    with pytest.raises(RuntimeIdentityError, match="BOTH"):
        load_agent_identity(None, storage_dir=tmp_path)


def test_loader_no_material_raises_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", TEST_DATA_KEY)
    with pytest.raises(FileNotFoundError, match="no identity material"):
        load_agent_identity(None, storage_dir=tmp_path)


@pytest.mark.asyncio
async def test_loader_refuses_incomplete_born_hybrid(tmp_path, hybrid_env):
    await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Partial",
    )
    next(tmp_path.glob("*_mldsa65.bytes.enc")).unlink()
    with pytest.raises(RuntimeIdentityError, match="mldsa65"):
        load_agent_identity(None, storage_dir=tmp_path)


# ---------------------------------------------------------------------------
# Re-inception must not clobber an existing identity (codex P2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reinception_refuses_to_overwrite_keys_without_force(tmp_path, hybrid_env):
    await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Keeper",
    )
    with pytest.raises(FileExistsError):
        await create_kestrel_identity_async(
            str(tmp_path), CONSTITUTION, agent_name="Keeper",
        )


@pytest.mark.asyncio
async def test_reinception_with_force_backs_up_old_keys(tmp_path, hybrid_env):
    old = await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Reborn",
    )
    old_slug = old.agent_did.rsplit(":", 1)[-1]
    old_key = (tmp_path / f"{old_slug}_ed25519.key.enc").read_bytes()

    new = await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Reborn", force=True,
    )
    new_slug = new.agent_did.rsplit(":", 1)[-1]
    # Fresh identity in place, prior keys recoverable from backups, and
    # exactly ONE live DID document (the loader refuses ambiguity)
    assert (tmp_path / f"{new_slug}_ed25519.key.enc").exists()
    assert [d.name for d in tmp_path.glob("*_did.json")] == [f"{new_slug}_did.json"]
    backups = list(tmp_path.glob(f"{old_slug}_ed25519.key.enc.backup-*"))
    assert backups, "expected the prior classical key to be backed up"
    assert backups[0].read_bytes() == old_key
    assert list(tmp_path.glob(f"{old_slug}_mldsa65.bytes.enc.backup-*"))
    # And the new agent actually loads
    ident = load_agent_identity(None, storage_dir=tmp_path)
    assert ident.new_did == new.agent_did


# ---------------------------------------------------------------------------
# Spawn mandates from a born-hybrid parent (codex P1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sign_mandate_with_born_hybrid_parent_no_legacy_key(tmp_path, hybrid_env):
    from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate, verify_mandate

    creds = await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Parentbird",
    )
    parent = load_agent_identity(None, storage_dir=tmp_path)
    assert parent.legacy_keypair is None

    mandate = SpawnMandate(
        parent_did=creds.agent_did, purpose="hatch a helper",
        ttl_seconds=3600, max_child_depth=1,
    )
    sign_mandate(mandate, None, parent_identity=parent)
    assert mandate.parent_signature and mandate.parent_signature.startswith("hybrid:")
    assert verify_mandate(mandate, None, parent_identity=parent)


@pytest.mark.asyncio
async def test_malformed_domain_cleans_up_db(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", TEST_DATA_KEY)
    monkeypatch.setenv(DID_WEB_DOMAIN_ENV, "https://agents.example.com")  # scheme = malformed
    monkeypatch.delenv(IDENTITY_METHOD_ENV, raising=False)

    with pytest.raises(Exception):
        await create_kestrel_identity_async(
            str(tmp_path), CONSTITUTION, agent_name="Baddomain",
        )
    # No half-created database left behind — a retry must not need force=True
    assert not (tmp_path / "kestrel_prime.db").exists()


@pytest.mark.asyncio
async def test_identity_package_sign_verify_roundtrip_born_hybrid(tmp_path, hybrid_env):
    """A born-hybrid agent can produce and verify a signed identity
    package (codex round-2 P2: signing used to anchor via the legacy
    key id, which cannot parse did:web)."""
    from kestrel_sovereign.identity.identity_package import AgentIdentityPackage
    from kestrel_sovereign.identity.signing import (
        sign_package, verify_package_signature,
    )

    creds = await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Exporter",
    )
    package = AgentIdentityPackage(
        did=creds.agent_did, agent_name="Exporter",
        created_at="2026-07-11T00:00:00Z",
        constitution_hash="abc123", constitution_text="# Test",
    )
    signed = sign_package(package, storage_dir=tmp_path)
    assert signed.signatures, "expected v2 hybrid signatures"
    algs = {s["alg"] for s in signed.signatures}
    assert algs == {"ed25519", "ml-dsa-65"}

    ok, msg = verify_package_signature(signed, storage_dir=tmp_path)
    assert ok, msg


@pytest.mark.asyncio
async def test_identity_package_verify_rejects_wrong_local_agent(tmp_path, hybrid_env):
    """The local trust anchor must BE the package's DID — a dir holding a
    different born-hybrid agent cannot anchor someone else's package."""
    from kestrel_sovereign.identity.identity_package import AgentIdentityPackage
    from kestrel_sovereign.identity.signing import (
        sign_package, verify_package_signature,
    )

    creds = await create_kestrel_identity_async(
        str(tmp_path), CONSTITUTION, agent_name="Realbird",
    )
    other_dir = tmp_path / "other-agent"
    other_dir.mkdir()
    await create_kestrel_identity_async(
        str(other_dir), CONSTITUTION, agent_name="Otherbird",
    )

    package = AgentIdentityPackage(
        did=creds.agent_did, agent_name="Realbird",
        created_at="2026-07-11T00:00:00Z",
        constitution_hash="abc123", constitution_text="# Test",
    )
    signed = sign_package(package, storage_dir=tmp_path)
    ok, msg = verify_package_signature(signed, storage_dir=other_dir)
    assert not ok
    assert "custody" in msg or "not" in msg


def test_sign_mandate_refuses_keyless_parent():
    from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate

    mandate = SpawnMandate(
        parent_did="did:web:example.com:ghost", purpose="none",
        ttl_seconds=60, max_child_depth=1,
    )
    with pytest.raises(ValueError, match="neither a hybrid identity nor a legacy"):
        sign_mandate(mandate, None, parent_identity=None)


# ---------------------------------------------------------------------------
# Quickstart zero-config contract (CI clean-install regression)
# ---------------------------------------------------------------------------

def test_quickstart_defaults_domain_to_localhost(tmp_path, monkeypatch):
    """kestrel setup --quickstart on a clean box must not block on the
    did:web domain: it defaults to 'localhost' (recorded, persisted)
    rather than falling back to a classical identity."""
    from kestrel_sovereign.setup.steps.agent import _ensure_did_web_domain
    from kestrel_sovereign.setup.context import Flow, SetupContext

    monkeypatch.delenv(DID_WEB_DOMAIN_ENV, raising=False)
    monkeypatch.delenv(IDENTITY_METHOD_ENV, raising=False)

    ctx = SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=Flow.QUICKSTART,
        prompter=None,  # quickstart branch never prompts
    )
    assert _ensure_did_web_domain(ctx) is True
    assert os.environ.get(DID_WEB_DOMAIN_ENV) == "localhost"
    assert not ctx.blockers
    assert any("localhost" in r for r in ctx.changes)
    assert f"{DID_WEB_DOMAIN_ENV}=localhost" in (tmp_path / ".env").read_text()
