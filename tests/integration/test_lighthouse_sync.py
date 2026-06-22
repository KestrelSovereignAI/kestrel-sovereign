"""
Unit tests for Lighthouse sync target — state persistence for ephemeral environments.

Tests restore_snapshot, CID tracking, manifest upload/download, and cold-start flow.
"""

import json
import re
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from kestrel_sovereign.storage.car_builder import CARBuilder
from kestrel_sovereign.storage.sync.retention import RetentionPolicy
from kestrel_sovereign.storage.sync.targets import LighthouseTarget, SyncResult


LIGHTHOUSE_REST_CLIENT = "kestrel_sovereign.storage.providers.lighthouse_rest.LighthouseRestClient"


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
        mock_client.upload_car = AsyncMock(side_effect=upload_error)
    elif upload_responses:
        mock_client.upload_car = AsyncMock(side_effect=upload_responses[:1])
        mock_client.upload = AsyncMock(side_effect=upload_responses[1:])
    else:
        mock_client.upload_car = AsyncMock(
            return_value={"Hash": "QmDefault", "Size": "0"}
        )
        mock_client.upload = AsyncMock(return_value={"Hash": "QmDefault", "Size": "0"})

    mock_client.download = AsyncMock(return_value=download_content or b"")
    mock_client.get_uploads = AsyncMock(
        return_value={
            "fileList": uploads_list or [],
            "totalFiles": len(uploads_list or []),
        }
    )
    mock_client.delete_file = AsyncMock(return_value={"deleted": True})
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
            LIGHTHOUSE_REST_CLIENT,
            return_value=mock_client,
        ):
            result = await target.sync_snapshot(db_path)

        assert result.success is True
        assert result.bytes_synced == 23
        assert result.metadata["cid"] == "QmSnapshot123"
        assert result.metadata["format"] == "car-v1/raw-sqlite"
        assert target._latest_cid == "QmSnapshot123"
        mock_client.upload_car.assert_awaited_once()
        upload_car_kwargs = mock_client.upload_car.await_args.kwargs
        assert re.fullmatch(
            r"kestrel_state__agent-1__\d{8}_\d{6}\.car",
            upload_car_kwargs["filename"],
        )

        # Verify manifest was saved locally
        manifest = target._load_local_manifest()
        assert manifest is not None
        assert manifest["snapshot_cid"] == "QmSnapshot123"
        assert manifest["snapshot_format"] == "car-v1/raw-sqlite"
        assert manifest["raw_snapshot_size"] == len(b"SQLite database content")
        assert manifest["snapshot_payload_cid"].startswith("b")
        assert re.fullmatch(
            r"kestrel_manifest__agent-1__\d{8}_\d{6}\.json",
            manifest["manifest_upload_name"],
        )

    @pytest.mark.asyncio
    async def test_sync_snapshot_upload_failure(self, target, tmp_path):
        """Upload failure should return unsuccessful SyncResult."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"data")

        mock_client = _mock_rest_client(upload_error=ConnectionError("network down"))

        with patch(
            LIGHTHOUSE_REST_CLIENT,
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
            LIGHTHOUSE_REST_CLIENT,
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
        mock_client.upload_car = AsyncMock(
            return_value={"Hash": "QmSnapshot123", "Size": "4"}
        )
        mock_client.upload = AsyncMock(
            side_effect=ConnectionError("manifest upload failed")
        )

        with patch(
            LIGHTHOUSE_REST_CLIENT,
            return_value=mock_client,
        ):
            result = await target.sync_snapshot(db_path)

        # Snapshot should still succeed
        assert result.success is True
        assert result.metadata["cid"] == "QmSnapshot123"

    def test_build_snapshot_car_round_trips_raw_content(self, target):
        content = b"SQLite database content"

        car_bytes, payload_cid = target._build_snapshot_car(content)

        assert payload_cid.startswith("b")
        assert target._extract_snapshot_content(car_bytes) == content


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
                LIGHTHOUSE_REST_CLIENT,
                return_value=mock_client,
            ):
                result = await target.restore_snapshot(dest)

        assert result is not None
        assert result.success is True
        assert result.bytes_synced == len(db_content)
        assert dest.read_bytes() == db_content

    @pytest.mark.asyncio
    async def test_restore_from_car_snapshot(self, target, tmp_path):
        """Should extract raw SQLite bytes from CAR snapshots."""
        dest = tmp_path / "restored.db"
        db_content = b"restored database content"
        builder = CARBuilder()
        payload_cid = builder.add_raw_block(db_content)
        builder.set_root(payload_cid)

        mock_client = _mock_rest_client(download_content=builder.build())

        with patch.dict("os.environ", {"LIGHTHOUSE_STATE_CID": "QmExplicit123"}):
            with patch(
                LIGHTHOUSE_REST_CLIENT,
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
            LIGHTHOUSE_REST_CLIENT,
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
            LIGHTHOUSE_REST_CLIENT,
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
            LIGHTHOUSE_REST_CLIENT,
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
            LIGHTHOUSE_REST_CLIENT,
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
    async def test_finds_snapshot_by_structured_agent_filename(self, target):
        """Should find latest snapshot matching agent-scoped structured name."""
        uploads = [
            {
                "fileName": "kestrel_state__agent-1__20260101_000000.car",
                "cid": "QmOlder",
            },
            {
                "fileName": "kestrel_state__agent-1__20260201_000000.car",
                "cid": "QmNewer",
            },
            {
                "fileName": "kestrel_state__agent-2__20260301_000000.car",
                "cid": "QmOther",
            },
        ]

        mock_client = _mock_rest_client(uploads_list=uploads)

        with patch(
            LIGHTHOUSE_REST_CLIENT,
            return_value=mock_client,
        ):
            cid = await target._query_uploads_api()

        assert cid == "QmNewer"

    @pytest.mark.asyncio
    async def test_does_not_use_cross_agent_legacy_db_heuristic(self, target):
        """Regression: unscoped kestrel*.db uploads must never restore cross-agent."""
        uploads = [
            {
                "fileName": "kestrel_prime.db",
                "cid": "QmAgentBFlatDb",
                "createdAt": "2026-03-01T00:00:00Z",
            },
            {
                "fileName": "kestrel_state__agent-2__20260301_000000.car",
                "cid": "QmAgentBStructured",
            },
        ]

        mock_client = _mock_rest_client(uploads_list=uploads)

        with patch(
            LIGHTHOUSE_REST_CLIENT,
            return_value=mock_client,
        ):
            cid = await target._query_uploads_api()

        assert cid is None

    @pytest.mark.asyncio
    async def test_legacy_manifest_restore_still_works(self, target):
        """Legacy flat snapshots remain restorable only through agent manifest."""
        uploads = [
            {
                "fileName": "manifest_agent-1.json",
                "cid": "QmManifest",
                "createdAt": "2026-02-01T00:00:00Z",
            },
            {
                "fileName": "manifest_agent-2.json",
                "cid": "QmOtherManifest",
                "createdAt": "2026-03-01T00:00:00Z",
            },
        ]

        listing_client = _mock_rest_client(uploads_list=uploads)
        manifest_client = _mock_rest_client(
            download_content=b'{"agent_id":"agent-1","snapshot_cid":"QmLegacySnapshot"}'
        )

        with patch(
            LIGHTHOUSE_REST_CLIENT,
            side_effect=[listing_client, manifest_client],
        ):
            cid = await target._query_uploads_api()

        assert cid == "QmLegacySnapshot"

    @pytest.mark.asyncio
    async def test_rejects_manifest_for_different_agent(self, target):
        manifest_client = _mock_rest_client(
            download_content=b'{"agent_id":"agent-2","snapshot_cid":"QmWrongAgent"}'
        )

        with patch(
            LIGHTHOUSE_REST_CLIENT,
            return_value=manifest_client,
        ):
            cid = await target._read_manifest_cid("QmManifest")

        assert cid is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_snapshots(self, target):
        """Should return None if no matching uploads."""
        mock_client = _mock_rest_client(uploads_list=[])

        with patch(
            LIGHTHOUSE_REST_CLIENT,
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
            LIGHTHOUSE_REST_CLIENT,
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


class TestPrune:
    @pytest.mark.asyncio
    async def test_prune_scopes_to_agent_and_leaves_unattributed_legacy_wal(self):
        target = LighthouseTarget(api_key="test-key", agent_id="agent-1")
        uploads = [
            {
                "fileName": "kestrel_state__agent-1__20260101_000000.car",
                "cid": "QmOldSnapshot",
                "id": "file-old-snapshot",
            },
            {
                "fileName": "kestrel_state__agent-1__20260201_000000.car",
                "cid": "QmNewSnapshot",
                "id": "file-new-snapshot",
            },
            {
                "fileName": "kestrel_manifest__agent-1__20260101_000000.json",
                "cid": "QmOldManifest",
                "id": "file-old-manifest",
            },
            {
                "fileName": "kestrel_manifest__agent-1__20260201_000000.json",
                "cid": "QmNewManifest",
                "id": "file-new-manifest",
            },
            {
                "fileName": "kestrel_prime.db-wal",
                "cid": "QmLegacyWal",
                "createdAt": "2026-01-15T00:00:00Z",
            },
            {
                "fileName": "kestrel_state__agent-2__20240101_000000.car",
                "cid": "QmOtherAgent",
            },
        ]
        mock_client = _mock_rest_client(uploads_list=uploads)

        with patch(
            LIGHTHOUSE_REST_CLIENT,
            return_value=mock_client,
        ):
            result = await target.prune(
                RetentionPolicy.from_config(
                    {
                        "backup": {
                            "retention": {
                                "working_memory": {
                                    "keep_all_days": 0,
                                    "weekly_forever": False,
                                    "monthly_forever": False,
                                },
                                "identity": {
                                    "keep_all_days": 0,
                                    "weekly_until_months": 0,
                                    "monthly_forever": False,
                                },
                            }
                        }
                    }
                )
            )

        # Only this agent's structured snapshots + manifests are scanned (4).
        # The unattributed legacy `*-wal` is NOT pruned here (shared-key
        # data-loss risk); another agent's structured upload is never touched.
        assert result["scanned"] == 4
        deleted = {call.args[0] for call in mock_client.delete_file.await_args_list}
        assert "file-old-snapshot" in deleted
        assert "file-old-manifest" in deleted
        assert "QmLegacyWal" not in deleted
        assert "QmOtherAgent" not in deleted


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
            LIGHTHOUSE_REST_CLIENT,
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
            LIGHTHOUSE_REST_CLIENT,
            return_value=mock_client,
        ):
            result = await target.health_check()

        assert result is False


@pytest.mark.asyncio
async def test_prune_paginates_with_id_cursor_across_pages():
    """prune() must consume all pages via the last-item id cursor (the real API
    returns no nextLastKey/lastKey), not stop after the first page."""
    from unittest.mock import AsyncMock
    from kestrel_sovereign.storage.sync.lighthouse_target import LighthouseTarget
    from kestrel_sovereign.storage.sync.retention import RetentionPolicy

    pages = {
        None: {"fileList": [
            {"cid": "QmA", "id": "a", "fileName": "kestrel_state__agent-1__20240101_000000.car",
             "createdAt": "2024-01-01T00:00:00Z"},
            {"cid": "QmB", "id": "b", "fileName": "kestrel_state__agent-1__20240108_000000.car",
             "createdAt": "2024-01-08T00:00:00Z"}], "totalFiles": 3},
        "b": {"fileList": [
            {"cid": "QmC", "id": "c", "fileName": "kestrel_state__agent-1__20240115_000000.car",
             "createdAt": "2024-01-15T00:00:00Z"}], "totalFiles": 3},
        "c": {"fileList": [], "totalFiles": 3},
    }
    calls = []
    client = AsyncMock()
    async def _get_uploads(last_key=None):
        calls.append(last_key); return pages[last_key]
    client.get_uploads = _get_uploads
    client.delete_file = AsyncMock(return_value={"deleted": True})
    client.close = AsyncMock()

    target = LighthouseTarget(api_key="k", agent_id="agent-1")
    with patch(LIGHTHOUSE_REST_CLIENT, return_value=client):
        result = await target.prune(RetentionPolicy.from_config(
            {"backup": {"retention": {"working_memory": {
                "keep_all_days": 0, "weekly_forever": False, "monthly_forever": False}}}}))

    # All three pages consumed via id cursor; older snapshots pruned, newest kept.
    assert calls == [None, "b", "c"]
    assert result["scanned"] == 3
