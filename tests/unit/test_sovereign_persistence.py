"""
Tests for sovereign-first persistence architecture.

Covers: TrustTier, SovereignIPFSTarget, simplified SyncService.
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.storage.sync.targets import (
    TrustTier,
    SyncTarget,
    SyncResult,
    SovereignIPFSTarget,
    GCSTarget,
    LighthouseTarget,
    S3Target,
)
from kestrel_sovereign.storage.sync.service import SyncService


# ---------------------------------------------------------------------------
# TrustTier
# ---------------------------------------------------------------------------

class TestTrustTier:
    def test_ordering(self):
        assert TrustTier.SOVEREIGN.value < TrustTier.FEDERATED.value
        assert TrustTier.FEDERATED.value < TrustTier.DELEGATED.value
        assert TrustTier.DELEGATED.value < TrustTier.EXPEDIENT.value

    def test_all_four_tiers_exist(self):
        assert len(TrustTier) == 4

    def test_names(self):
        assert TrustTier.SOVEREIGN.name == "SOVEREIGN"
        assert TrustTier.FEDERATED.name == "FEDERATED"
        assert TrustTier.DELEGATED.name == "DELEGATED"
        assert TrustTier.EXPEDIENT.name == "EXPEDIENT"


# ---------------------------------------------------------------------------
# Trust tier assignments
# ---------------------------------------------------------------------------

class TestTargetTrustTiers:
    def test_s3_is_expedient(self):
        target = S3Target(bucket="test")
        assert target.trust_tier == TrustTier.EXPEDIENT

    def test_gcs_is_expedient(self):
        target = GCSTarget(bucket="test")
        assert target.trust_tier == TrustTier.EXPEDIENT

    def test_lighthouse_is_delegated(self):
        target = LighthouseTarget(api_key="test", agent_id="test")
        assert target.trust_tier == TrustTier.DELEGATED

    def test_sovereign_ipfs_is_sovereign(self):
        target = SovereignIPFSTarget(api_url="http://localhost:5001", agent_id="test")
        assert target.trust_tier == TrustTier.SOVEREIGN

    def test_default_trust_tier_is_expedient(self):
        """SyncTarget ABC default is EXPEDIENT for safety."""
        class MinimalTarget(SyncTarget):
            @property
            def name(self): return "test"
            async def sync_snapshot(self, db_path): pass
            async def sync_wal(self, wal_path, position): pass
            async def get_latest_position(self): return None

        assert MinimalTarget().trust_tier == TrustTier.EXPEDIENT


# ---------------------------------------------------------------------------
# SovereignIPFSTarget
# ---------------------------------------------------------------------------

class TestSovereignIPFSTarget:
    def test_name(self):
        target = SovereignIPFSTarget(api_url="http://10.0.0.1:5001", agent_id="Emma")
        assert target.name == "ipfs://Emma"

    def test_api_url_strips_trailing_slash(self):
        target = SovereignIPFSTarget(api_url="http://10.0.0.1:5001/", agent_id="test")
        assert target.api_url == "http://10.0.0.1:5001"

    @pytest.mark.asyncio
    async def test_sync_wal_is_noop(self):
        target = SovereignIPFSTarget(api_url="http://localhost:5001", agent_id="test")
        result = await target.sync_wal(Path("/fake"), 0)
        assert result.success is True
        assert result.bytes_synced == 0

    @pytest.mark.asyncio
    async def test_get_latest_position_suppresses_wal(self):
        target = SovereignIPFSTarget(api_url="http://localhost:5001", agent_id="test")
        pos = await target.get_latest_position()
        assert pos == 2**63

    def test_manifest_path(self):
        with tempfile.TemporaryDirectory() as d:
            target = SovereignIPFSTarget(
                api_url="http://localhost:5001",
                agent_id="Emma",
                state_dir=Path(d),
            )
            assert target._manifest_path == Path(d) / ".sovereign_ipfs_manifest_Emma.json"

    def test_manifest_path_none_without_state_dir(self):
        target = SovereignIPFSTarget(api_url="http://localhost:5001", agent_id="test")
        assert target._manifest_path is None

    @pytest.mark.asyncio
    async def test_sync_snapshot_dedup(self):
        """Skip upload if content hash unchanged."""
        with tempfile.TemporaryDirectory() as d:
            state_dir = Path(d)
            db_path = state_dir / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE t(x)")
            conn.commit()
            conn.close()

            target = SovereignIPFSTarget(
                api_url="http://localhost:5001",
                agent_id="test",
                state_dir=state_dir,
            )

            # Write a manifest with the current DB's hash
            import hashlib
            with open(db_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()

            manifest = {"content_hash": content_hash, "cid": "QmTest"}
            target._save_local_manifest(manifest)

            result = await target.sync_snapshot(db_path)
            assert result.success is True
            assert result.bytes_synced == 0
            assert result.metadata["skipped"] is True

    @pytest.mark.asyncio
    async def test_sync_snapshot_uploads(self):
        """Upload when content changed."""
        with tempfile.TemporaryDirectory() as d:
            state_dir = Path(d)
            db_path = state_dir / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE t(x)")
            conn.commit()
            conn.close()

            target = SovereignIPFSTarget(
                api_url="http://localhost:5001",
                agent_id="test",
                state_dir=state_dir,
            )

            mock_response = MagicMock()
            mock_response.json.return_value = {"Hash": "QmNewCID123", "Size": "4096"}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            target._client = mock_client

            result = await target.sync_snapshot(db_path)
            assert result.success is True
            assert result.bytes_synced > 0
            assert result.metadata["cid"] == "QmNewCID123"

            # Verify manifest was saved
            manifest = target._load_local_manifest()
            assert manifest["cid"] == "QmNewCID123"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        target = SovereignIPFSTarget(api_url="http://localhost:5001", agent_id="test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ID": "12D3KooW...", "AgentVersion": "kubo/0.34.0"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        target._client = mock_client

        assert await target.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        target = SovereignIPFSTarget(api_url="http://unreachable:5001", agent_id="test")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        target._client = mock_client

        assert await target.health_check() is False

    @pytest.mark.asyncio
    async def test_restore_no_cid(self):
        """Restore returns None when no CID is available."""
        target = SovereignIPFSTarget(api_url="http://localhost:5001", agent_id="test")
        result = await target.restore_snapshot(Path("/tmp/restore.db"))
        assert result is None

    @pytest.mark.asyncio
    async def test_restore_from_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            state_dir = Path(d)
            dest = state_dir / "restored.db"
            target = SovereignIPFSTarget(
                api_url="http://localhost:5001",
                agent_id="test",
                state_dir=state_dir,
            )
            target._save_local_manifest({"cid": "QmRestoreCID"})

            mock_response = MagicMock()
            mock_response.content = b"fake-sqlite-data"
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            target._client = mock_client

            result = await target.restore_snapshot(dest)
            assert result is not None
            assert result.success is True
            assert result.bytes_synced == len(b"fake-sqlite-data")
            assert dest.read_bytes() == b"fake-sqlite-data"


# ---------------------------------------------------------------------------
# Simplified SyncService
# ---------------------------------------------------------------------------

class TestSyncService:
    @pytest.mark.asyncio
    async def test_targets_by_trust_order(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            db_path.touch()

            svc = SyncService(db_path=str(db_path))

            sovereign = SovereignIPFSTarget(api_url="http://localhost:5001", agent_id="t")
            gcs = GCSTarget(bucket="test")
            lighthouse = LighthouseTarget(api_key="k", agent_id="t")

            # Add in wrong order
            svc.add_target(gcs)
            svc.add_target(lighthouse)
            svc.add_target(sovereign)

            ordered = svc.targets_by_trust
            assert ordered[0].trust_tier == TrustTier.SOVEREIGN
            assert ordered[1].trust_tier == TrustTier.DELEGATED
            assert ordered[2].trust_tier == TrustTier.EXPEDIENT

    @pytest.mark.asyncio
    async def test_force_snapshot_fans_out(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE t(x)")
            conn.commit()
            conn.close()

            svc = SyncService(db_path=str(db_path))

            mock_target_1 = MagicMock(spec=SyncTarget)
            mock_target_1.name = "target1"
            mock_target_1.trust_tier = TrustTier.SOVEREIGN
            mock_target_1.sync_snapshot = AsyncMock(return_value=SyncResult(
                success=True, target_name="target1", bytes_synced=100,
                frames_synced=0, timestamp=datetime.now(timezone.utc),
            ))

            mock_target_2 = MagicMock(spec=SyncTarget)
            mock_target_2.name = "target2"
            mock_target_2.trust_tier = TrustTier.EXPEDIENT
            mock_target_2.sync_snapshot = AsyncMock(return_value=SyncResult(
                success=True, target_name="target2", bytes_synced=100,
                frames_synced=0, timestamp=datetime.now(timezone.utc),
            ))

            svc.add_target(mock_target_1)
            svc.add_target(mock_target_2)

            results = await svc.force_snapshot()
            assert len(results) == 2
            assert results["target1"].success is True
            assert results["target2"].success is True
            mock_target_1.sync_snapshot.assert_called_once()
            mock_target_2.sync_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_by_trust_order(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            db_path.touch()
            dest = Path(d) / "restored.db"

            svc = SyncService(db_path=str(db_path))

            # Sovereign target fails
            failing = MagicMock(spec=SyncTarget)
            failing.name = "sovereign"
            failing.trust_tier = TrustTier.SOVEREIGN
            failing.health_check = AsyncMock(return_value=True)
            failing.restore_snapshot = AsyncMock(return_value=None)

            # Expedient target succeeds
            succeeding = MagicMock(spec=SyncTarget)
            succeeding.name = "gcs"
            succeeding.trust_tier = TrustTier.EXPEDIENT
            succeeding.health_check = AsyncMock(return_value=True)
            succeeding.restore_snapshot = AsyncMock(return_value=SyncResult(
                success=True, target_name="gcs", bytes_synced=500,
                frames_synced=0, timestamp=datetime.now(timezone.utc),
            ))

            svc.add_target(succeeding)  # Added first but lower trust
            svc.add_target(failing)      # Added second but higher trust

            result = await svc.restore_by_trust(dest)
            assert result is not None
            assert result.target_name == "gcs"
            # Sovereign was tried first (by trust order), then fell through
            failing.restore_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_running_property(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            db_path.touch()

            svc = SyncService(db_path=str(db_path))
            assert svc.is_running is False

            mock = MagicMock(spec=SyncTarget)
            mock.name = "t"
            mock.trust_tier = TrustTier.EXPEDIENT
            svc.add_target(mock)
            assert svc.is_running is True

    @pytest.mark.asyncio
    async def test_no_background_tasks(self):
        """SyncService should not spawn background tasks on start()."""
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            db_path.touch()

            svc = SyncService(db_path=str(db_path))
            await svc.start()
            # No _sync_task or _snapshot_task attributes
            assert not hasattr(svc, "_sync_task")
            assert not hasattr(svc, "_snapshot_task")
            await svc.stop()
