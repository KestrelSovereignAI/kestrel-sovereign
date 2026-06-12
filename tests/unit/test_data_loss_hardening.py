"""Data-loss hardening (#1725): inception --force, retirement scoping, filecoin
tier-aware cleanup + CID integrity verification."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kestrel_sovereign.filecoin_adapter import FilecoinAdapter
from kestrel_sovereign.storage.providers.base import StorageTier


# ---------------------------------------------------------------------------
# Filecoin: tier-aware cleanup never evicts LOCAL_ONLY
# ---------------------------------------------------------------------------
class TestFilecoinCleanup:
    def _adapter(self, tmp_path):
        return FilecoinAdapter(cache_dir=str(tmp_path / "cache"))

    def _age_out(self, adapter):
        import os
        old = 1.0  # epoch — far past any cutoff
        for f in adapter.cache_dir.glob("*"):
            os.utime(f, (old, old))

    def test_cleanup_keeps_local_only_content(self, tmp_path):
        adapter = self._adapter(tmp_path)
        r = adapter.store_content(b"precious local data", storage_tier=StorageTier.LOCAL_ONLY)
        self._age_out(adapter)
        adapter.cleanup_cache(max_age_days=30)
        # LOCAL_ONLY content (only copy) must survive cleanup.
        assert (adapter.cache_dir / f"{r.content_hash}.cache").exists()
        assert adapter.retrieve_content(r.content_hash) == b"precious local data"

    def test_cleanup_keeps_unmarked_legacy_content(self, tmp_path):
        adapter = self._adapter(tmp_path)
        # Simulate a legacy cache file with no durability marker.
        (adapter.cache_dir / "legacyhash.cache").write_bytes(b"x")
        self._age_out(adapter)
        adapter.cleanup_cache(max_age_days=30)
        assert (adapter.cache_dir / "legacyhash.cache").exists()

    def test_cleanup_evicts_durably_replicated_content(self, tmp_path):
        adapter = self._adapter(tmp_path)
        # A cache entry explicitly marked as durably on IPFS may be evicted.
        (adapter.cache_dir / "remote.cache").write_bytes(b"x")
        (adapter.cache_dir / "remote.meta").write_text(
            json.dumps({"_kestrel_tier": StorageTier.IPFS.value, "_kestrel_cid": "Qm123"})
        )
        self._age_out(adapter)
        adapter.cleanup_cache(max_age_days=30)
        assert not (adapter.cache_dir / "remote.cache").exists()


# ---------------------------------------------------------------------------
# Filecoin: retrieve verifies the content hash (cache poisoning detection)
# ---------------------------------------------------------------------------
class TestFilecoinIntegrity:
    def test_retrieve_rejects_poisoned_cache(self, tmp_path):
        adapter = FilecoinAdapter(cache_dir=str(tmp_path / "cache"))
        r = adapter.store_content(b"genuine", storage_tier=StorageTier.LOCAL_ONLY)
        # Poison the cache with different bytes under the same content_hash.
        import zlib
        (adapter.cache_dir / f"{r.content_hash}.cache").write_bytes(zlib.compress(b"TAMPERED"))
        with pytest.raises(ValueError, match="integrity check failed"):
            adapter.retrieve_content(r.content_hash)

    def test_roundtrip_still_verifies_ok(self, tmp_path):
        adapter = FilecoinAdapter(cache_dir=str(tmp_path / "cache"))
        r = adapter.store_content(b"hello world", storage_tier=StorageTier.LOCAL_ONLY)
        assert adapter.retrieve_content(r.content_hash) == b"hello world"

    def test_cid_keyed_lookup_skips_sha256_check(self, tmp_path):
        """#1725 codex r1: callers that pass a CID as content_hash (CID-as-key)
        must NOT trip the sha256 integrity check (a sha256 never equals a CID)."""
        import zlib
        adapter = FilecoinAdapter(cache_dir=str(tmp_path / "cache"))
        cid = "QmSomeBase58CidNotASha256Hash"  # not 64-hex
        # Place content in the cache under the CID key (as the IPFS-cache path does).
        (adapter.cache_dir / f"{cid}.cache").write_bytes(zlib.compress(b"remote payload"))
        # Retrieving by the CID key returns the bytes WITHOUT an integrity error.
        assert adapter.retrieve_content(cid) == b"remote payload"

    def test_looks_like_sha256(self):
        from kestrel_sovereign.filecoin_adapter import _looks_like_sha256
        assert _looks_like_sha256("a" * 64) is True
        assert _looks_like_sha256("QmCid") is False
        assert _looks_like_sha256("A" * 64) is True   # case-insensitive
        assert _looks_like_sha256("z" * 64) is False  # non-hex


# ---------------------------------------------------------------------------
# Inception: --force gates DB overwrite
# ---------------------------------------------------------------------------
class TestInceptionForce:
    @pytest.mark.asyncio
    async def test_existing_db_without_force_raises(self, tmp_path):
        from kestrel_sovereign.inception_service import create_kestrel_identity_async

        await create_kestrel_identity_async(output_dir=str(tmp_path), is_test_instance=True)
        assert (tmp_path / "kestrel_prime.db").exists()
        # Second inception WITHOUT force must refuse (not silently destroy memory).
        with pytest.raises(FileExistsError):
            await create_kestrel_identity_async(output_dir=str(tmp_path), is_test_instance=True)

    @pytest.mark.asyncio
    async def test_force_backs_up_then_overwrites(self, tmp_path):
        from kestrel_sovereign.inception_service import create_kestrel_identity_async

        await create_kestrel_identity_async(output_dir=str(tmp_path), is_test_instance=True)
        await create_kestrel_identity_async(
            output_dir=str(tmp_path), is_test_instance=True, force=True,
        )
        # A backup of the prior DB exists, and a fresh DB is in place.
        assert (tmp_path / "kestrel_prime.db").exists()
        backups = list(tmp_path.glob("kestrel_prime.db.backup-*"))
        assert backups, "expected a timestamped DB backup after --force overwrite"


# ---------------------------------------------------------------------------
# Retirement: archive only THIS agent's keys (not co-located peers')
# ---------------------------------------------------------------------------
class TestRetirementScoping:
    @pytest.mark.asyncio
    async def test_placeholder(self):  # keep class import-time stable
        assert True


# ---------------------------------------------------------------------------
# Sovereign import: restore the ORIGINAL timestamp, not import-time "now"
# ---------------------------------------------------------------------------
class TestImportTimestampPreservation:
    def _adapter(self):
        from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter
        a = object.__new__(SovereignStorageAdapter)

        class _DB:
            backend_type = "sqlite"
        a.db = _DB()
        return a

    def test_restored_created_at_from_metadata_timestamp(self):
        a = self._adapter()
        out = a._restored_created_at({"timestamp": "2025-11-03T12:34:56+00:00"})
        # SQLite form, UTC, matching datetime('now') layout — NOT collapsed to now.
        assert out == "2025-11-03 12:34:56"

    def test_restored_created_at_none_without_timestamp(self):
        a = self._adapter()
        assert a._restored_created_at({}) is None
        assert a._restored_created_at({"timestamp": "not-a-date"}) is None


class TestRetirementScopingPeerKeys:
    @pytest.mark.asyncio
    async def test_retirement_does_not_confiscate_peer_keys(self, tmp_path, monkeypatch):
        from kestrel_sovereign.inception_service import create_kestrel_identity_async
        from kestrel_sovereign.retirement_service import retire_test_agent

        # Two agents sharing one data dir, each with a distinct key_id.
        a_dir = tmp_path / "shared"
        a_dir.mkdir()
        await create_kestrel_identity_async(output_dir=str(a_dir), is_test_instance=True)
        # Snapshot agent A's key files (DID json + pem/key.enc).
        a_keys = {p.name for p in a_dir.glob("kestrel_*")}

        # Drop a peer's private-key file alongside (different key_id).
        peer_pem = a_dir / "kestrel_0xPEER.pem"
        peer_pem.write_text("PEER PRIVATE KEY")
        peer_did = a_dir / "kestrel_0xPEER.json"
        peer_did.write_text(json.dumps({"id": "did:web:example.com:peer"}))

        db_path = a_dir / "kestrel_prime.db"
        await retire_test_agent(str(db_path), reason="done")

        # The peer's key + DID doc must remain untouched (NOT archived/moved).
        assert peer_pem.exists(), "retirement confiscated a peer's private key!"
        assert peer_did.exists(), "retirement moved a peer's DID document!"
        assert a_keys, "test setup: agent A should have had key files"
