"""The demo seed carries a pinned signer and only its own initial test state."""

import lzma
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from kestrel_sovereign.identity.custody_bundle import (
    IdentityCustodyError,
    create_identity_bundle,
)
from kestrel_sovereign.identity.demo_seed import main, restore_demo_seed
from kestrel_sovereign.identity.identity_package import AgentIdentityPackage
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.identity.signing import sign_package, verify_package_signature
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.security.crypto_suite import MLDSA65Suite


@pytest.mark.asyncio
async def test_demo_seed_reuses_pinned_keys_and_rejects_unrelated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    source = tmp_path / "source"
    identity = await create_kestrel_identity_async(
        str(source), agent_name="Public Demo", is_demo=True, is_test_instance=True,
        did_web_domain="agents.custody.test", did_web_slug="public-demo",
    )
    did = identity.agent_did
    snapshot = source / "kestrel_prime.db"
    bundle = create_identity_bundle(source, expected_did=did).encode()
    database_seed = tmp_path / "initial.db.xz"
    database_seed.write_bytes(lzma.compress(snapshot.read_bytes()))

    first = tmp_path / "first"
    first.mkdir()
    (first / "SOUL.md").write_text("Container demo profile")
    monkeypatch.chdir(tmp_path)
    second = Path("second")
    for agent_dir in (first, second):
        restore_demo_seed(agent_dir, database_seed, bundle, expected_did=did)
        restored = load_agent_identity(None, storage_dir=agent_dir)
        assert restored.signing_did == did
        assert restored.new_verification_methods == load_agent_identity(
            None, storage_dir=source
        ).new_verification_methods
        assert (agent_dir / snapshot.name).read_bytes() == snapshot.read_bytes()
    assert (first / "SOUL.md").read_text() == "Container demo profile"

    with pytest.raises(IdentityCustodyError, match="existing agent state"):
        restore_demo_seed(first, database_seed, bundle, expected_did=did)
    with closing(sqlite3.connect(snapshot)) as db:
        with db:
            db.execute("UPDATE graph_nodes SET properties = '{}' WHERE node_id = ?", (did,))
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    database_seed.write_bytes(lzma.compress(snapshot.read_bytes()))
    unrelated = tmp_path / "unrelated"
    with pytest.raises(IdentityCustodyError, match="does not bind to the pinned test identity"):
        restore_demo_seed(unrelated, database_seed, bundle, expected_did=did)
    assert not unrelated.exists()


def test_entrypoint_refuses_different_configured_discovery_storage(tmp_path, monkeypatch):
    registry = tmp_path / "multi_agent.toml"
    registry.write_text(
        '[agents.demo]\n'
        f'data_dir = "{tmp_path / "other"}"\n'
        'port = 8801\nautostart = true\n'
    )
    monkeypatch.setattr(sys, "argv", [
        "demo_seed", "--config", str(registry),
        "--agent-data-dir", str(tmp_path / "data"),
        "--host-control-dir", str(tmp_path / "host"),
        "--registration", "demo",
        "--database-seed", str(tmp_path / "db.xz"),
        "--bundle-seed", str(tmp_path / "bundle"),
        "--expected-did", "did:web:agents.custody.test:demo",
    ])
    with pytest.raises(IdentityCustodyError, match="configured discovery registration"):
        main()


@pytest.mark.asyncio
async def test_mldsa_codec_change_preserves_signed_identity_packages(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    with monkeypatch.context() as legacy:
        legacy.setattr(MLDSA65Suite, "public_key_multicodec", b"\x87\x24")
        identity = await create_kestrel_identity_async(
            str(tmp_path), agent_name="Legacy Demo", is_demo=True, is_test_instance=True,
            did_web_domain="agents.custody.test", did_web_slug="legacy",
        )
        package = sign_package(AgentIdentityPackage(
            did=identity.agent_did, agent_name="Legacy Demo", created_at="2026-01-01",
            constitution_hash="demo-hash", constitution_text="Demo constitution",
        ), storage_dir=tmp_path)
    assert verify_package_signature(package, storage_dir=tmp_path)
