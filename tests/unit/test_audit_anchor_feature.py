"""
Unit tests for the Audit Trail Anchoring feature.

Tests the AuditHasher for deterministic hashing and the AuditAnchorFeature
for anchoring, verifying, and reporting on audit trail integrity.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.audit_anchor.hasher import AuditHasher


# =========================================================================
# AuditHasher Tests
# =========================================================================


class TestAuditHasher:
    """Tests for deterministic hashing of audit entries."""

    def test_hasher_deterministic(self):
        """Same entries produce the same hash every time."""
        entries = [
            {"id": 1, "action": "tool_execution", "decision": "allowed", "created_at": "2026-01-01T00:00:00"},
            {"id": 2, "action": "tool_execution", "decision": "denied", "created_at": "2026-01-01T00:01:00"},
        ]

        hash1 = AuditHasher.hash_entries(entries)
        hash2 = AuditHasher.hash_entries(entries)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_hasher_different_order_same_hash(self):
        """Entries in different order still produce the same hash (sorted by created_at)."""
        entries_a = [
            {"id": 1, "action": "exec", "created_at": "2026-01-01T00:00:00"},
            {"id": 2, "action": "exec", "created_at": "2026-01-01T00:01:00"},
        ]
        entries_b = [
            {"id": 2, "action": "exec", "created_at": "2026-01-01T00:01:00"},
            {"id": 1, "action": "exec", "created_at": "2026-01-01T00:00:00"},
        ]

        assert AuditHasher.hash_entries(entries_a) == AuditHasher.hash_entries(entries_b)

    def test_hasher_detects_changes(self):
        """Modified entry produces a different hash."""
        original = [
            {"id": 1, "action": "tool_execution", "decision": "allowed", "created_at": "2026-01-01T00:00:00"},
        ]
        modified = [
            {"id": 1, "action": "tool_execution", "decision": "denied", "created_at": "2026-01-01T00:00:00"},
        ]

        assert AuditHasher.hash_entries(original) != AuditHasher.hash_entries(modified)

    def test_hasher_empty_entries(self):
        """Empty entry list still produces a valid hash."""
        h = AuditHasher.hash_entries([])
        assert isinstance(h, str)
        assert len(h) == 64

    def test_serialize_entries_is_bytes(self):
        """serialize_entries returns bytes."""
        entries = [{"id": 1, "created_at": "2026-01-01"}]
        result = AuditHasher.serialize_entries(entries)
        assert isinstance(result, bytes)

    def test_serialize_entries_sorted_keys(self):
        """Serialized JSON has sorted keys for determinism."""
        entries = [{"z_field": "last", "a_field": "first", "created_at": "2026-01-01"}]
        result = AuditHasher.serialize_entries(entries)
        decoded = result.decode("utf-8")
        # "a_field" should come before "z_field" in the JSON
        assert decoded.index("a_field") < decoded.index("z_field")


# =========================================================================
# AuditAnchorFeature Tests
# =========================================================================


async def _setup_audit_db(db_path: str, audit_entries: list):
    """Create a real SQLite DB with security_audit_log entries."""
    import aiosqlite

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS security_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT,
                tool_name TEXT,
                action TEXT,
                decision TEXT,
                user_choice TEXT,
                args_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for entry in audit_entries:
            await conn.execute(
                """INSERT INTO security_audit_log
                   (feature_name, tool_name, action, decision, user_choice, args_summary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.get("feature_name", "TestFeature"),
                    entry.get("tool_name", "test_tool"),
                    entry.get("action", "tool_execution"),
                    entry.get("decision", "allowed"),
                    entry.get("user_choice"),
                    entry.get("args_summary"),
                    entry.get("created_at", datetime.now(timezone.utc).isoformat()),
                ),
            )
        await conn.commit()


async def _make_mock_agent(tmp_path, audit_entries=None):
    """Create a mock agent with storage and optionally a SecurityFeature."""
    agent = MagicMock()
    agent.agent_id = "test-agent-001"

    # Mock AsyncStorage
    storage = MagicMock()
    storage.store_file = AsyncMock(return_value="fakehash123")
    storage.add_node = AsyncMock()

    # Mock AsyncDatabase
    db = MagicMock()
    db.execute = AsyncMock(return_value=0)
    db.fetchone = AsyncMock(return_value=None)
    db.fetchall = AsyncMock(return_value=[])
    db.fetchval = AsyncMock(return_value=0)
    db.table_exists = AsyncMock(return_value=True)
    storage.db = db

    agent.storage = storage

    # Mock SecurityFeature with PermissionStore
    if audit_entries is not None:
        db_path = str(tmp_path / "test_permissions.db")
        await _setup_audit_db(db_path, audit_entries)

        permission_store = MagicMock()
        permission_store.db_path = db_path

        security_feature = MagicMock()
        security_feature.permission_store = permission_store
        type(security_feature).__name__ = "SecurityFeature"

        agent.features = {"SecurityFeature": security_feature}
    else:
        agent.features = {}

    return agent


@pytest_asyncio.fixture
async def feature_no_entries(tmp_path):
    """AuditAnchorFeature with no audit entries."""
    from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature

    agent = await _make_mock_agent(tmp_path, audit_entries=[])
    feature = AuditAnchorFeature(agent)
    await feature.initialize()
    return feature


@pytest_asyncio.fixture
async def feature_with_entries(tmp_path):
    """AuditAnchorFeature with some audit entries."""
    from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature

    entries = [
        {"feature_name": "WalletAgent", "tool_name": "get_balance", "action": "tool_execution",
         "decision": "allowed", "created_at": "2026-01-15T10:00:00"},
        {"feature_name": "WalletAgent", "tool_name": "send_tokens", "action": "tool_execution",
         "decision": "denied", "created_at": "2026-01-15T10:01:00"},
        {"feature_name": "MCPAgent", "tool_name": "execute_server", "action": "tool_execution",
         "decision": "user_approved", "user_choice": "session", "created_at": "2026-01-15T10:02:00"},
    ]
    agent = await _make_mock_agent(tmp_path, audit_entries=entries)
    feature = AuditAnchorFeature(agent)
    await feature.initialize()
    return feature


class TestAuditAnchorFeature:
    """Tests for the AuditAnchorFeature."""

    @pytest.mark.asyncio
    async def test_anchor_audit_no_entries(self, feature_no_entries):
        """Returns 'nothing to anchor' when no audit entries exist."""
        result = await feature_no_entries.anchor_audit()
        assert result.data["status"] == "nothing_to_anchor"

    @pytest.mark.asyncio
    async def test_anchor_audit_creates_record(self, feature_with_entries):
        """Anchoring with entries stores a record in the database."""
        result = await feature_with_entries.anchor_audit()

        assert result.data["status"] == "anchored"
        assert result.data["entries_count"] == 3
        assert len(result.data["anchor_hash"]) == 64  # SHA-256
        assert result.data["storage_ref"] == "fakehash123"
        assert result.data["anchor_id"] is not None

        # Verify store_file was called with serialized entries
        feature_with_entries.agent.storage.store_file.assert_called_once()

        # Verify db.execute was called to insert the anchor record
        feature_with_entries.agent.storage.db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_anchor_audit_stores_graph_node(self, feature_with_entries):
        """Anchoring stores a graph node for the anchor."""
        await feature_with_entries.anchor_audit()
        feature_with_entries.agent.storage.add_node.assert_called_once()

        # Verify the graph node has expected properties
        call_args = feature_with_entries.agent.storage.add_node.call_args
        node = call_args[0][0]
        assert node.node_type == "audit_anchor"
        assert "anchor_hash" in node.properties

    @pytest.mark.asyncio
    async def test_anchor_audit_fallback_no_file_storage(self, feature_with_entries):
        """Anchor still succeeds when store_file raises an exception."""
        feature_with_entries.agent.storage.store_file = AsyncMock(
            side_effect=Exception("Storage unavailable")
        )
        result = await feature_with_entries.anchor_audit()

        assert result.data["status"] == "anchored"
        assert result.data["storage_ref"] is None
        assert result.data["entries_count"] == 3

    @pytest.mark.asyncio
    async def test_anchor_audit_includes_destructive_audit_log(self, tmp_path):
        """Existing anchor flow also captures isolated destructive audit rows."""
        from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature
        from kestrel_sovereign.storage.destructive_audit import (
            DestructiveAuditEvent,
            DestructiveAuditLog,
            hash_rows,
        )

        agent = await _make_mock_agent(tmp_path, audit_entries=[])
        audit = DestructiveAuditLog(tmp_path / "kestrel_audit.db")
        await audit.append(
            DestructiveAuditEvent(
                agent_id="did:test:audit",
                operation_type="purge_all",
                row_count=1,
                pre_operation_hash=hash_rows([{"id": 1, "content": "gone"}]),
                snapshot_reference="inline:sha256",
                scope={"table": "conversation_history"},
                reason="test",
            )
        )
        agent.storage.destructive_audit = audit

        feature = AuditAnchorFeature(agent)
        await feature.initialize()

        result = await feature.anchor_audit()

        assert result.data["status"] == "anchored"
        assert result.data["entries_count"] == 1

    @pytest.mark.asyncio
    async def test_verify_audit_no_anchors(self, feature_no_entries):
        """Verify returns 'no_anchors' when none exist."""
        result = await feature_no_entries.verify_audit()
        assert result.data["status"] == "no_anchors"

    @pytest.mark.asyncio
    async def test_verify_audit_passes(self, tmp_path):
        """Verification passes when entries have not been tampered with."""
        from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature

        entries = [
            {"feature_name": "F1", "tool_name": "t1", "action": "exec",
             "decision": "allowed", "created_at": "2026-03-01T12:00:00"},
            {"feature_name": "F2", "tool_name": "t2", "action": "exec",
             "decision": "denied", "created_at": "2026-03-01T12:01:00"},
        ]
        agent = await _make_mock_agent(tmp_path, audit_entries=entries)

        # Pre-compute the expected hash for these entries as they'll appear from DB
        import aiosqlite
        perm_store = agent.features["SecurityFeature"].permission_store
        async with aiosqlite.connect(perm_store.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, feature_name, tool_name, action, decision, "
                "user_choice, args_summary, created_at FROM security_audit_log ORDER BY created_at ASC"
            )
            rows = await cursor.fetchall()
            db_entries = [dict(row) for row in rows]

        expected_hash = AuditHasher.hash_entries(db_entries)

        # Mock the anchor record in the database
        anchor_row = (
            "anchor-001", "test-agent", expected_hash, "ref123",
            2, "2026-03-01T12:00:00", "2026-03-01T12:01:00", "2026-03-01T12:02:00",
        )
        agent.storage.db.fetchall = AsyncMock(return_value=[anchor_row])
        agent.storage.db.table_exists = AsyncMock(return_value=True)

        feature = AuditAnchorFeature(agent)
        await feature.initialize()

        result = await feature.verify_audit()

        assert result.data["status"] == "verified"
        assert result.data["total_anchors"] == 1
        assert result.data["passed"] == 1
        assert result.data["failed"] == 0

    @pytest.mark.asyncio
    async def test_verify_audit_detects_tampering(self, tmp_path):
        """Verification fails when entries have been modified after anchoring."""
        from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature

        entries = [
            {"feature_name": "F1", "tool_name": "t1", "action": "exec",
             "decision": "allowed", "created_at": "2026-03-01T12:00:00"},
        ]
        agent = await _make_mock_agent(tmp_path, audit_entries=entries)

        # Use a WRONG hash to simulate tampering
        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        anchor_row = (
            "anchor-001", "test-agent", wrong_hash, "ref123",
            1, "2026-03-01T12:00:00", "2026-03-01T12:00:00", "2026-03-01T12:02:00",
        )
        agent.storage.db.fetchall = AsyncMock(return_value=[anchor_row])
        agent.storage.db.table_exists = AsyncMock(return_value=True)

        feature = AuditAnchorFeature(agent)
        await feature.initialize()

        result = await feature.verify_audit()

        assert result.data["status"] == "integrity_failure"
        assert result.data["failed"] == 1
        assert result.data["details"][0]["status"] == "FAIL"
        assert result.data["details"][0]["match"] is False

    @pytest.mark.asyncio
    async def test_anchor_status(self, feature_with_entries):
        """anchor_status returns correct counts."""
        # No anchors yet
        feature_with_entries.agent.storage.db.fetchone = AsyncMock(return_value=None)
        feature_with_entries.agent.storage.db.fetchval = AsyncMock(return_value=0)

        result = await feature_with_entries.anchor_status()

        assert result.data["last_anchor_at"] is None
        assert result.data["total_anchors"] == 0
        assert result.data["entries_since_last"] == 3  # 3 entries exist, none anchored
        assert result.data["auto_anchor_threshold"] == 50

    @pytest.mark.asyncio
    async def test_anchor_status_with_existing_anchor(self, tmp_path):
        """anchor_status reflects existing anchors correctly."""
        from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature

        entries = [
            {"feature_name": "F1", "tool_name": "t1", "action": "exec",
             "decision": "allowed", "created_at": "2026-03-01T12:00:00"},
            {"feature_name": "F2", "tool_name": "t2", "action": "exec",
             "decision": "denied", "created_at": "2026-03-01T12:05:00"},
        ]
        agent = await _make_mock_agent(tmp_path, audit_entries=entries)

        # Simulate one existing anchor covering the first entry
        agent.storage.db.fetchone = AsyncMock(return_value=("2026-03-01T12:00:00",))
        agent.storage.db.fetchval = AsyncMock(return_value=1)
        agent.storage.db.table_exists = AsyncMock(return_value=True)

        feature = AuditAnchorFeature(agent)
        await feature.initialize()

        result = await feature.anchor_status()

        assert result.data["last_anchor_at"] == "2026-03-01T12:00:00"
        assert result.data["total_anchors"] == 1
        # Only the second entry (12:05) is after the anchor at 12:00
        assert result.data["entries_since_last"] == 1

    @pytest.mark.asyncio
    async def test_on_audit_complete_auto_anchors(self, tmp_path):
        """on_audit_complete triggers auto-anchor when threshold is reached."""
        from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature

        # Create 55 entries to exceed the AUTO_ANCHOR_THRESHOLD of 50
        entries = [
            {"feature_name": f"Feature{i}", "tool_name": "tool", "action": "exec",
             "decision": "allowed", "created_at": f"2026-03-01T12:{i:02d}:00"}
            for i in range(55)
        ]
        agent = await _make_mock_agent(tmp_path, audit_entries=entries)
        agent.storage.db.table_exists = AsyncMock(return_value=True)
        # No previous anchors
        agent.storage.db.fetchone = AsyncMock(return_value=None)

        feature = AuditAnchorFeature(agent)
        await feature.initialize()

        await feature.on_audit_complete({"is_valid": True, "message": "OK"})

        # Verify store_file was called (anchor was triggered)
        agent.storage.store_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_audit_complete_no_auto_anchor_below_threshold(self, tmp_path):
        """on_audit_complete does NOT auto-anchor when below threshold."""
        from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature

        # Only 5 entries (well below threshold of 50)
        entries = [
            {"feature_name": f"F{i}", "tool_name": "tool", "action": "exec",
             "decision": "allowed", "created_at": f"2026-03-01T12:0{i}:00"}
            for i in range(5)
        ]
        agent = await _make_mock_agent(tmp_path, audit_entries=entries)
        agent.storage.db.table_exists = AsyncMock(return_value=True)
        agent.storage.db.fetchone = AsyncMock(return_value=None)

        feature = AuditAnchorFeature(agent)
        await feature.initialize()

        await feature.on_audit_complete({"is_valid": True, "message": "OK"})

        # store_file should NOT have been called
        agent.storage.store_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_without_storage(self):
        """Initialize gracefully handles missing storage."""
        from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature

        agent = MagicMock()
        agent.storage = None

        feature = AuditAnchorFeature(agent)
        # Should not raise
        await feature.initialize()

    @pytest.mark.asyncio
    async def test_anchor_no_security_feature(self, tmp_path):
        """Anchor returns nothing when SecurityFeature is not available."""
        from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature

        agent = await _make_mock_agent(tmp_path, audit_entries=None)  # No security feature
        agent.storage.db.table_exists = AsyncMock(return_value=True)
        agent.storage.db.fetchone = AsyncMock(return_value=None)

        feature = AuditAnchorFeature(agent)
        await feature.initialize()

        result = await feature.anchor_audit()
        assert result.data["status"] == "nothing_to_anchor"

    @pytest.mark.asyncio
    async def test_feature_has_correct_tools(self, feature_no_entries):
        """Feature exposes the expected tools."""
        tools = feature_no_entries.get_tools()
        tool_names = {t.name for t in tools}
        assert "audit_anchor" in tool_names
        assert "audit_verify" in tool_names
        assert "audit_anchor_status" in tool_names

    @pytest.mark.asyncio
    async def test_feature_tool_description(self, feature_no_entries):
        """Feature has a meaningful tool description."""
        desc = feature_no_entries.tool_description
        assert "audit" in desc.lower()
        assert "Article II" in desc


# =========================================================================
# Run tests
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
