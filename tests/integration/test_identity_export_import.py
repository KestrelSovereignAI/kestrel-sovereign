#!/usr/bin/env pytest
"""
Integration tests for Identity Export/Import functionality.

Tests the full export and import cycle with a real database.
"""
import json
import pytest
import pytest_asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from kestrel_sovereign.identity import (
    AgentIdentityPackage,
    IdentityExporter,
    IdentityImporter,
    export_identity,
    import_identity,
    sign_package,
    verify_package_signature,
    SubstrateType,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest_asyncio.fixture
async def test_db():
    """Create a temporary test database with schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_identity.db"
        db = await AsyncDatabase.sqlite(str(db_path))

        # Create minimal schema for testing
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
            CREATE TABLE IF NOT EXISTS files (
                content_hash TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                content BLOB,
                metadata TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                role TEXT,
                content TEXT,
                metadata TEXT,
                created_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memory_episodes (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                title TEXT,
                summary TEXT,
                timespan_start TEXT,
                timespan_end TEXT,
                key_message_ids TEXT,
                emotional_arc TEXT,
                created_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS saved_items (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                item_type TEXT,
                name TEXT,
                summary TEXT,
                content TEXT,
                content_hash TEXT,
                ipfs_cid TEXT,
                embedding BLOB,
                source_type TEXT,
                source_ref TEXT,
                schema_id TEXT,
                tags TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS temporal_patterns (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                pattern_type TEXT,
                description TEXT,
                trigger_conditions TEXT,
                confidence REAL,
                observations INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reflection_insights (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                session_id TEXT,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                evidence TEXT,
                confidence REAL DEFAULT 0.5,
                actionable INTEGER DEFAULT 0,
                suggested_action TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS wallet_state (
                agent_id TEXT PRIMARY KEY,
                main_balance TEXT NOT NULL,
                audit_balance TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS wallet_transactions (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                amount TEXT,
                memo TEXT,
                created_at TEXT
            )
        """)
        await db.commit()

        yield db

        await db.close()


@pytest.fixture
async def populated_db(test_db):
    """Create a database with sample data for export testing."""
    agent_id = "did:pkh:eip155:1:0xTestAgent123"

    # Insert agent node
    await test_db.execute(
        """INSERT INTO graph_nodes (node_id, node_type, label, properties)
           VALUES (?, 'agent', 'Test Agent', ?)""",
        (agent_id, json.dumps({
            "created_at": "2025-01-01T00:00:00Z",
            "agent_name": "Test Agent",
            "constitution_hash": "test_constitution_hash",
        }))
    )

    # Insert conversation history
    for i in range(5):
        await test_db.execute(
            """INSERT INTO conversation_history (agent_id, role, content, metadata, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (agent_id, "user" if i % 2 == 0 else "assistant",
             f"Test message {i}", "{}", f"2025-01-{i+1:02d}T12:00:00Z")
        )

    # Insert memory episodes
    await test_db.execute(
        """INSERT INTO memory_episodes (id, agent_id, title, summary, timespan_start, timespan_end,
           key_message_ids, emotional_arc, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("ep_001", agent_id, "First Episode", "A test episode",
         "2025-01-01T00:00:00Z", "2025-01-01T23:59:59Z",
         '["msg1", "msg2"]', "curiosity → understanding", "2025-01-02T00:00:00Z")
    )

    # Insert saved items
    await test_db.execute(
        """INSERT INTO saved_items (id, agent_id, item_type, name, summary, content, content_hash,
           source_type, schema_id, tags, metadata, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("item_001", agent_id, "note", "Test Note", "A test note",
         '{"content": "test"}', "hash123", "manual", None,
         '["test", "sample"]', '{}', "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
    )

    # Insert temporal patterns
    await test_db.execute(
        """INSERT INTO temporal_patterns (id, agent_id, pattern_type, description,
           trigger_conditions, confidence, observations, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("pat_001", agent_id, "weekly_rhythm", "User active on weekends",
         '{"day": "saturday"}', 0.8, 10, "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
    )

    # Insert wallet state
    await test_db.execute(
        """INSERT INTO wallet_state (agent_id, main_balance, audit_balance, updated_at)
           VALUES (?, ?, ?, ?)""",
        (agent_id, "500.50", "0.0", "2025-01-01T00:00:00Z")
    )

    await test_db.commit()

    return test_db, agent_id


class TestIdentityExporter:
    """Tests for IdentityExporter class."""

    @pytest.mark.asyncio
    async def test_export_basic(self, populated_db):
        """Test basic export functionality."""
        db, agent_id = populated_db

        exporter = IdentityExporter(db, agent_id)
        package = await exporter.export()

        assert package.did == agent_id
        assert package.agent_name == "Test Agent"
        assert package.package_version is not None
        assert package.export_timestamp is not None

    @pytest.mark.asyncio
    async def test_export_includes_episodes(self, populated_db):
        """Test that export includes memory episodes."""
        db, agent_id = populated_db

        exporter = IdentityExporter(db, agent_id)
        package = await exporter.export()

        assert len(package.episodes) == 1
        assert package.episodes[0]["title"] == "First Episode"
        assert package.episodes[0]["emotional_arc"] == "curiosity → understanding"

    @pytest.mark.asyncio
    async def test_export_includes_saved_items(self, populated_db):
        """Test that export includes saved items."""
        db, agent_id = populated_db

        exporter = IdentityExporter(db, agent_id)
        package = await exporter.export()

        assert len(package.saved_items) == 1
        assert package.saved_items[0]["name"] == "Test Note"

    @pytest.mark.asyncio
    async def test_export_includes_temporal_patterns(self, populated_db):
        """Test that export includes temporal patterns."""
        db, agent_id = populated_db

        exporter = IdentityExporter(db, agent_id)
        package = await exporter.export()

        assert len(package.temporal_patterns) == 1
        assert package.temporal_patterns[0]["pattern_type"] == "weekly_rhythm"

    @pytest.mark.asyncio
    async def test_export_includes_wallet(self, populated_db):
        """Test that export includes wallet state."""
        db, agent_id = populated_db

        exporter = IdentityExporter(db, agent_id)
        package = await exporter.export(include_wallet_history=True)

        assert package.wallet_balance == "500.50"

    @pytest.mark.asyncio
    async def test_export_personality_extraction(self, populated_db):
        """Test personality fingerprint extraction."""
        db, agent_id = populated_db

        exporter = IdentityExporter(db, agent_id)
        package = await exporter.export()

        # Should have extracted personality from conversation history
        assert package.personality is not None
        assert package.personality.communication_style is not None

    @pytest.mark.asyncio
    async def test_export_convenience_function(self, populated_db):
        """Test the export_identity convenience function."""
        db, agent_id = populated_db

        package = await export_identity(db, agent_id)

        assert package.did == agent_id
        assert package.content_hash is not None


class TestIdentityImporter:
    """Tests for IdentityImporter class."""

    @pytest.fixture
    def sample_package(self):
        """Create a sample package for import testing."""
        import hashlib
        constitution_text = "# Test Constitution"
        constitution_hash = hashlib.sha256(constitution_text.encode()).hexdigest()
        return AgentIdentityPackage(
            did="did:pkh:eip155:1:0xImportTest",
            agent_name="Import Test Agent",
            created_at="2025-01-01T00:00:00Z",
            constitution_hash=constitution_hash,
            constitution_text=constitution_text,
            episodes=[
                {
                    "id": "ep_import_001",
                    "title": "Imported Episode",
                    "summary": "An imported episode",
                    "timespan_start": "2025-01-01T00:00:00Z",
                    "timespan_end": "2025-01-01T23:59:59Z",
                    "key_message_ids": [],
                    "emotional_arc": "test",
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ],
            saved_items=[
                {
                    "id": "item_import_001",
                    "item_type": "note",
                    "name": "Imported Note",
                    "summary": "Test",
                    "content": "{}",
                    "content_hash": "abc",
                    "ipfs_cid": None,
                    "source_type": "manual",
                    "source_ref": None,
                    "schema_id": None,
                    "tags": [],
                    "metadata": {},
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                }
            ],
            temporal_patterns=[
                {
                    "id": "pat_import_001",
                    "pattern_type": "test_pattern",
                    "description": "Test pattern",
                    "trigger_conditions": {},
                    "confidence": 0.5,
                    "observations": 1,
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                }
            ],
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
        )

    @pytest.mark.asyncio
    async def test_import_basic(self, test_db, sample_package):
        """Test basic import functionality."""
        importer = IdentityImporter(test_db)
        result = await importer.import_package(sample_package, verify_signature=False)

        assert result.success is True
        assert result.agent_id == sample_package.did
        assert result.migration_id.startswith("mig_")

    @pytest.mark.asyncio
    async def test_import_episodes(self, test_db, sample_package):
        """Test that episodes are imported."""
        importer = IdentityImporter(test_db)
        result = await importer.import_package(sample_package, verify_signature=False)

        assert result.stats.get("episodes_imported") == 1

        # Verify in database - ID is prefixed with agent_id[:20]
        expected_id = f"{sample_package.did[:20]}_ep_import_001"
        row = await test_db.fetchone(
            "SELECT title FROM memory_episodes WHERE id = ?",
            (expected_id,)
        )
        assert row is not None
        assert row[0] == "Imported Episode"

    @pytest.mark.asyncio
    async def test_import_saved_items(self, test_db, sample_package):
        """Test that saved items are imported."""
        importer = IdentityImporter(test_db)
        result = await importer.import_package(sample_package, verify_signature=False)

        assert result.stats.get("saved_items_imported") == 1

        # Verify in database - ID is prefixed with agent_id[:20]
        expected_id = f"{sample_package.did[:20]}_item_import_001"
        row = await test_db.fetchone(
            "SELECT name FROM saved_items WHERE id = ?",
            (expected_id,)
        )
        assert row is not None
        assert row[0] == "Imported Note"

    @pytest.mark.asyncio
    async def test_import_temporal_patterns(self, test_db, sample_package):
        """Test that temporal patterns are imported."""
        importer = IdentityImporter(test_db)
        result = await importer.import_package(sample_package, verify_signature=False)

        assert result.stats.get("temporal_patterns_imported") == 1

    @pytest.mark.asyncio
    async def test_import_records_migration(self, test_db, sample_package):
        """Test that migration is recorded in graph."""
        importer = IdentityImporter(test_db, target_substrate="openai:gpt")
        result = await importer.import_package(sample_package, verify_signature=False)

        # Check migration record was created
        row = await test_db.fetchone(
            "SELECT properties FROM graph_nodes WHERE node_id = ?",
            (result.migration_id,)
        )
        assert row is not None
        props = json.loads(row[0])
        assert props["source_substrate"] == SubstrateType.ANTHROPIC_CLAUDE.value
        assert props["target_substrate"] == "openai:gpt"

    @pytest.mark.asyncio
    async def test_import_replace_mode(self, test_db, sample_package):
        """Test import with replace mode clears existing data."""
        # First import
        importer = IdentityImporter(test_db)
        await importer.import_package(sample_package, verify_signature=False)

        # Second import with replace mode
        sample_package.episodes[0]["title"] = "Updated Episode"
        importer2 = IdentityImporter(test_db)
        result = await importer2.import_package(
            sample_package,
            verify_signature=False,
            merge_mode="replace"
        )

        # Should have only one episode with updated title
        rows = await test_db.fetchall(
            "SELECT title FROM memory_episodes WHERE agent_id = ?",
            (sample_package.did,)
        )
        assert len(rows) == 1
        assert rows[0][0] == "Updated Episode"

    @pytest.mark.asyncio
    async def test_import_convenience_function(self, test_db, sample_package):
        """Test the import_identity convenience function."""
        result = await import_identity(test_db, sample_package, verify_signature=False)

        assert result.success is True
        assert result.agent_id == sample_package.did


class TestExportImportRoundTrip:
    """Test full export -> import cycle."""

    @pytest.mark.asyncio
    async def test_full_roundtrip(self, populated_db):
        """Test exporting from one DB and importing to another."""
        source_db, source_agent_id = populated_db
        target_agent_id = "did:pkh:eip155:1:0xTargetAgent"

        # Export from source
        package = await export_identity(source_db, source_agent_id)

        # Modify DID for target (simulating migration)
        # Note: This invalidates the content hash, so recompute it
        package.did = target_agent_id
        package.content_hash = package.compute_content_hash()

        # Import to same database with different agent_id
        # This simulates migrating to a new agent identity
        result = await import_identity(
            source_db,
            package,
            target_agent_id=target_agent_id,
            verify_signature=False,
        )

        assert result.success is True
        assert result.stats.get("episodes_imported") == 1
        assert result.stats.get("saved_items_imported") == 1
        assert result.stats.get("temporal_patterns_imported") == 1

    @pytest.mark.asyncio
    async def test_json_roundtrip(self, populated_db):
        """Test export -> JSON -> import cycle."""
        source_db, source_agent_id = populated_db
        target_agent_id = "did:pkh:eip155:1:0xRestored"

        # Export and serialize to JSON
        package = await export_identity(source_db, source_agent_id)
        json_str = package.to_json()

        # Deserialize and import
        restored_package = AgentIdentityPackage.from_json(json_str)
        restored_package.did = target_agent_id
        # Recompute content hash after DID modification
        restored_package.content_hash = restored_package.compute_content_hash()

        result = await import_identity(
            source_db,
            restored_package,
            target_agent_id=target_agent_id,
            verify_signature=False,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_content_hash_preserved(self, populated_db):
        """Test that content hash is consistent through serialization."""
        source_db, source_agent_id = populated_db

        package = await export_identity(source_db, source_agent_id)
        original_hash = package.content_hash

        # Serialize and deserialize
        json_str = package.to_json()
        restored = AgentIdentityPackage.from_json(json_str)

        # Hash should match
        assert restored.content_hash == original_hash
        assert restored.verify_content_hash() is True
