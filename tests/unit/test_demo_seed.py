"""The demo seed carries a pinned signer and only its own initial test state."""

import lzma
import sqlite3
from contextlib import closing

import pytest

from kestrel_sovereign.identity.custody_bundle import (
    IDENTITY_BUNDLE_ENV,
    IdentityCustodyError,
    create_identity_bundle,
)
from kestrel_sovereign.identity.demo_seed import restore_demo_seed
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.inception_service import create_kestrel_identity_async


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
    monkeypatch.setenv(IDENTITY_BUNDLE_ENV, create_identity_bundle(source, expected_did=did))
    database_seed = tmp_path / "initial.db.xz"
    database_seed.write_bytes(lzma.compress(snapshot.read_bytes()))

    first = tmp_path / "first"
    first.mkdir()
    (first / "SOUL.md").write_text("Container demo profile")
    second = tmp_path / "second"
    for agent_dir in (first, second):
        restore_demo_seed(agent_dir, database_seed, expected_did=did)
        restored = load_agent_identity(None, storage_dir=agent_dir)
        assert restored.signing_did == did
        assert restored.new_verification_methods == load_agent_identity(
            None, storage_dir=source
        ).new_verification_methods
        assert (agent_dir / snapshot.name).read_bytes() == snapshot.read_bytes()
    assert (first / "SOUL.md").read_text() == "Container demo profile"

    with pytest.raises(IdentityCustodyError, match="existing agent state"):
        restore_demo_seed(first, database_seed, expected_did=did)
    with closing(sqlite3.connect(snapshot)) as db:
        with db:
            db.execute("UPDATE graph_nodes SET properties = '{}' WHERE node_id = ?", (did,))
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    database_seed.write_bytes(lzma.compress(snapshot.read_bytes()))
    unrelated = tmp_path / "unrelated"
    with pytest.raises(IdentityCustodyError, match="does not bind to the pinned test identity"):
        restore_demo_seed(unrelated, database_seed, expected_did=did)
    assert not unrelated.exists()
