"""
Unit tests for Lighthouse sync target — state persistence for ephemeral environments.

Tests restore_snapshot, CID tracking, manifest upload/download, and cold-start flow.
"""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from kestrel_sovereign.storage.sync.targets import LighthouseTarget, SyncResult


class TestLighthouseTargetInit:
    """Test LighthouseTarget initialization and properties."""

    def test_init_defaults(self):
        target = LighthouseTarget(api_key="test-key", agent_id="agent-123")
        assert target.api_key == "test-key"
        assert target.agent_id == "agent-123"
        assert target._state_dir is None
        assert target._latest_cid is None

    def test_init_with_state_dir(self, tmp_path):
        target = LighthouseTarget(
            api_key="test-key", agent_id="agent-123", state_dir=tmp_path
        )
        assert target._state_dir == tmp_path

    def test_name_property(self):
        target = LighthouseTarget(api_key="k", agent_id="did:key:abc")
        assert target.name == "lighthouse://did:key:abc"

    def test_manifest_path_with_state_dir(self, tmp_path):
        target = LighthouseTarget(
            api_key="k", agent_id="agent-1", state_dir=tmp_path
        )
        assert target._manifest_path == tmp_path / ".lighthouse_manifest_agent-1.json"

    def test_manifest_path_without_state_dir(self):
        target = LighthouseTarget(api_key="k", agent_id="agent-1")
        assert target._manifest_path is None


class TestLocalManifest:
    """Test local manifest save/load for CID tracking."""

    def test_save_and_load_manifest(self, tmp_path):
        target = LighthouseTarget(
            api_key="k", agent_id="test", state_dir=tmp_path
        )
        manifest = {
            "agent_id": "test",
            "snapshot_cid": "QmTest123",
            "snapshot_size": 1024,
        }
        target._save_local_manifest(manifest)

        loaded = target._load_local_manifest()
        assert loaded is not None
        assert loaded["snapshot_cid"] == "QmTest123"
        assert loaded["snapshot_size"] == 1024

    def test_load_manifest_missing(self, tmp_path):
        target = LighthouseTarget(
            api_key="k", agent_id="test", state_dir=tmp_path
        )
        assert target._load_local_manifest() is None

    def test_load_manifest_corrupt(self, tmp_path):
        target = LighthouseTarget(
            api_key="k", agent_id="test", state_dir=tmp_path
        )
        # Write corrupt data
        manifest_path = tmp_path / ".lighthouse_manifest_test.json"
        manifest_path.write_text("not json{{{")

        assert target._load_local_manifest() is None

    def test_save_manifest_no_state_dir(self):
        target = LighthouseTarget(api_key="k", agent_id="test")
        # Should not raise — just silently skip
        target._save_local_manifest({"cid": "test"})

    def test_load_manifest_no_state_dir(self):
        target = LighthouseTarget(api_key="k", agent_id="test")
        assert target._load_local_manifest() is None


def _mock_rest_client(upload_responses=None, download_content=None, uploads_list=None,
                       balance_data=None, upload_error=None):
    """Create a mock LighthouseRestClient for testing."""
    mock_client = AsyncMock()

    if upload_error:
        mock_client.upload = AsyncMock(side_effect=upload_error)
    elif upload_responses:
        mock_client.upload = AsyncMock(side_effect=upload_responses)
    else:
        mock_client.upload = AsyncMock(return_value={"Hash": "QmDefault", "Size": "0"})

    mock_client.download = AsyncMock(return_value=download_content or b"")
    mock_client.get_uploads = AsyncMock(return_value={"fileList": uploads_list or [], "totalFiles": len(uploads_list or [])})
    mock_client.get_balance = AsyncMock(return_value=balance_data or {"data": {"dataUsed": "0"}})
    mock_client.close = AsyncMock()

    return mock_client


class TestSyncSnapshot:
    """Test snapshot upload with manifest tracking."""

    @pytest.fixture
    def target(self, tmp_path):
        return LighthouseTarget(
            api_key="test-key", agent_id="agent-1", state_dir=tmp_path
        )

    @pytest.mark.asyncio
    async def test_sync_snapshot_success(self, target, tmp_path):
        """Snapshot upload should return CID and save manifest."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"SQLite database content")

        mock_client = _mock_rest_client(upload_responses=[
            {"Hash": "QmSnapshot123", "Size": "23"},
            {"Hash": "QmManifest456", "Size": "100"},
        ])

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.sync_snapshot(db_path)

        assert result.success is True
        assert result.bytes_synced == 23
        assert result.metadata["cid"] == "QmSnapshot123"
        assert target._latest_cid == "QmSnapshot123"

        # Verify manifest was saved locally
        manifest = target._load_local_manifest()
        assert manifest is not None
        assert manifest["snapshot_cid"] == "QmSnapshot123"

    @pytest.mark.asyncio
    async def test_sync_snapshot_upload_failure(self, target, tmp_path):
        """Upload failure should return unsuccessful SyncResult."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"data")

        mock_client = _mock_rest_client(upload_error=ConnectionError("network down"))

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.sync_snapshot(db_path)

        assert result.success is False
        assert "network down" in result.error

    @pytest.mark.asyncio
    async def test_sync_snapshot_no_cid_in_response(self, target, tmp_path):
        """Should fail cleanly if upload returns no CID."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"data")

        mock_client = _mock_rest_client(upload_responses=[{}])

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.sync_snapshot(db_path)

        assert result.success is False
        assert "No CID" in result.error

    @pytest.mark.asyncio
    async def test_sync_snapshot_manifest_failure_nonfatal(self, target, tmp_path):
        """Manifest upload failure should not fail the snapshot."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"data")

        mock_client = _mock_rest_client()
        # Snapshot succeeds, manifest upload fails
        mock_client.upload = AsyncMock(side_effect=[
            {"Hash": "QmSnapshot123", "Size": "4"},
            ConnectionError("manifest upload failed"),
        ])

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.sync_snapshot(db_path)

        # Snapshot should still succeed
        assert result.success is True
        assert result.metadata["cid"] == "QmSnapshot123"


class TestRestoreSnapshot:
    """Test cold-start restore from Lighthouse."""

    @pytest.fixture
    def target(self, tmp_path):
        return LighthouseTarget(
            api_key="test-key", agent_id="agent-1", state_dir=tmp_path
        )

    @pytest.mark.asyncio
    async def test_restore_from_env_var(self, target, tmp_path):
        """Should restore using LIGHTHOUSE_STATE_CID env var."""
        dest = tmp_path / "restored.db"
        db_content = b"restored database content"

        mock_client = _mock_rest_client(download_content=db_content)

        with patch.dict("os.environ", {"LIGHTHOUSE_STATE_CID": "QmExplicit123"}):
            with patch(
                "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
                return_value=mock_client,
            ):
                result = await target.restore_snapshot(dest)

        assert result is not None
        assert result.success is True
        assert result.bytes_synced == len(db_content)
        assert dest.read_bytes() == db_content

    @pytest.mark.asyncio
    async def test_restore_from_local_manifest(self, target, tmp_path):
        """Should restore using locally saved manifest."""
        dest = tmp_path / "restored.db"
        db_content = b"from manifest"

        target._save_local_manifest({"snapshot_cid": "QmFromManifest"})

        mock_client = _mock_rest_client(download_content=db_content)

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.restore_snapshot(dest)

        assert result is not None
        assert result.success is True
        assert result.metadata["cid"] == "QmFromManifest"

    @pytest.mark.asyncio
    async def test_restore_no_snapshot_available(self, target, tmp_path):
        """Should return None when no snapshot exists."""
        dest = tmp_path / "restored.db"

        with patch.object(target, "_query_uploads_api", return_value=None):
            result = await target.restore_snapshot(dest)

        assert result is None
        assert not dest.exists()

    @pytest.mark.asyncio
    async def test_restore_download_failure(self, target, tmp_path):
        """Should return failed SyncResult on download error."""
        dest = tmp_path / "restored.db"
        target._save_local_manifest({"snapshot_cid": "QmBadCID"})

        mock_client = AsyncMock()
        mock_client.download = AsyncMock(side_effect=ConnectionError("gateway unreachable"))
        mock_client.close = AsyncMock()

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.restore_snapshot(dest)

        assert result is not None
        assert result.success is False
        assert "gateway unreachable" in result.error

    @pytest.mark.asyncio
    async def test_restore_creates_parent_dirs(self, target, tmp_path):
        """Should create parent directories if they don't exist."""
        dest = tmp_path / "nested" / "dir" / "restored.db"
        target._save_local_manifest({"snapshot_cid": "QmNested"})

        mock_client = _mock_rest_client(download_content=b"data")

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.restore_snapshot(dest)

        assert result.success is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_restore_empty_content(self, target, tmp_path):
        """Should return None for empty downloads."""
        dest = tmp_path / "restored.db"
        target._save_local_manifest({"snapshot_cid": "QmEmpty"})

        mock_client = _mock_rest_client(download_content=b"")

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.restore_snapshot(dest)

        assert result is None


class TestResolveCID:
    """Test CID resolution priority chain."""

    @pytest.fixture
    def target(self, tmp_path):
        return LighthouseTarget(
            api_key="test-key", agent_id="agent-1", state_dir=tmp_path
        )

    @pytest.mark.asyncio
    async def test_env_var_takes_priority(self, target):
        """Env var should override all other sources."""
        target._save_local_manifest({"snapshot_cid": "QmManifest"})
        target._latest_cid = "QmMemory"

        with patch.dict("os.environ", {"LIGHTHOUSE_STATE_CID": "QmEnvVar"}):
            cid = await target._resolve_latest_cid()

        assert cid == "QmEnvVar"

    @pytest.mark.asyncio
    async def test_local_manifest_second_priority(self, target):
        """Local manifest should be used when no env var."""
        target._save_local_manifest({"snapshot_cid": "QmManifest"})
        target._latest_cid = "QmMemory"

        cid = await target._resolve_latest_cid()
        assert cid == "QmManifest"

    @pytest.mark.asyncio
    async def test_memory_cache_third_priority(self, target):
        """In-memory CID should be used when no manifest."""
        target._latest_cid = "QmMemory"

        cid = await target._resolve_latest_cid()
        assert cid == "QmMemory"

    @pytest.mark.asyncio
    async def test_uploads_api_last_resort(self, target):
        """Should query uploads API as last resort."""
        with patch.object(
            target, "_query_uploads_api", return_value="QmFromAPI"
        ) as mock_api:
            cid = await target._resolve_latest_cid()

        assert cid == "QmFromAPI"
        mock_api.assert_called_once()


class TestQueryUploadsAPI:
    """Test Lighthouse uploads API querying."""

    @pytest.fixture
    def target(self):
        return LighthouseTarget(api_key="test-key", agent_id="agent-1")

    @pytest.mark.asyncio
    async def test_finds_snapshot_by_tag(self, target):
        """Should find latest snapshot matching agent tag."""
        uploads = [
            {"tag": "kestrel-state-agent-1", "cid": "QmOlder", "createdAt": "2026-01-01T00:00:00Z"},
            {"tag": "kestrel-state-agent-1", "cid": "QmNewer", "createdAt": "2026-02-01T00:00:00Z"},
            {"tag": "kestrel-state-other-agent", "cid": "QmOther", "createdAt": "2026-03-01T00:00:00Z"},
        ]

        mock_client = _mock_rest_client(uploads_list=uploads)

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            cid = await target._query_uploads_api()

        assert cid == "QmNewer"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_snapshots(self, target):
        """Should return None if no matching uploads."""
        mock_client = _mock_rest_client(uploads_list=[])

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            cid = await target._query_uploads_api()

        assert cid is None

    @pytest.mark.asyncio
    async def test_handles_api_error(self, target):
        """Should return None on API error."""
        mock_client = AsyncMock()
        mock_client.get_uploads = AsyncMock(side_effect=ConnectionError("timeout"))
        mock_client.close = AsyncMock()

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            cid = await target._query_uploads_api()

        assert cid is None


class TestSyncWAL:
    """Test WAL sync (no-op for Lighthouse)."""

    @pytest.mark.asyncio
    async def test_sync_wal_is_noop(self, tmp_path):
        target = LighthouseTarget(
            api_key="k", agent_id="test", state_dir=tmp_path
        )
        wal_path = tmp_path / "test.db-wal"
        wal_path.write_bytes(b"wal data")

        result = await target.sync_wal(wal_path, position=0)

        assert result.success is True
        assert result.bytes_synced == 0


class TestGetLatestPosition:
    """Test WAL position tracking (returns max to prevent WAL sync)."""

    @pytest.mark.asyncio
    async def test_returns_max_to_prevent_sync(self):
        target = LighthouseTarget(api_key="k", agent_id="test")
        pos = await target.get_latest_position()
        assert pos == 2**63


class TestHealthCheck:
    """Test Lighthouse connectivity check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        target = LighthouseTarget(api_key="test-key", agent_id="test")

        mock_client = _mock_rest_client(balance_data={"data": {"dataUsed": "0"}})

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        target = LighthouseTarget(api_key="bad-key", agent_id="test")

        mock_client = AsyncMock()
        mock_client.get_balance = AsyncMock(side_effect=Exception("auth failed"))
        mock_client.close = AsyncMock()

        with patch(
            "kestrel_storage_lighthouse.lighthouse_rest.LighthouseRestClient",
            return_value=mock_client,
        ):
            result = await target.health_check()

        assert result is False
