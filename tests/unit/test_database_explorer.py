"""Security regressions for the read-only database explorer."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from kestrel_sovereign.endpoints.database import (
    ALLOWED_TABLES,
    list_database_tables,
    query_database_table,
)
from kestrel_sovereign.endpoints.memories import (
    delete_memory,
    get_memory_detail,
)
from kestrel_sovereign.endpoints.files import check_file, serve_file
from kestrel_sovereign.endpoints.sovereignty import get_storage_stats
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.db import TransactionError
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.privacy import PrivacyMode


def _request(storage, agent_id=None):
    agent = SimpleNamespace(storage=storage)
    if agent_id is not None:
        agent.agent_id = agent_id
    return SimpleNamespace(state=SimpleNamespace(agent=agent))


@pytest.mark.asyncio
async def test_shared_database_scopes_authorized_rows_and_blocks_documents(tmp_path):
    """Two agent consoles cannot cross either scoped or unowned boundaries."""
    storage = await AsyncStorage.create_sqlite(str(tmp_path / "shared.db"))
    agent_a = "did:test:agent-a"
    agent_b = "did:test:agent-b"
    hostile = '<img src=x onerror="globalThis.databaseXssExecuted = true">'
    unicode_long = "漢字🦅 café — Δοκιμή " * 40

    try:
        await storage.db.execute_many(
            """
            INSERT INTO conversation_history (agent_id, role, content, metadata)
            VALUES (?, ?, ?, ?)
            """,
            [
                (agent_a, "user", hostile, "{}"),
                (agent_a, "assistant", unicode_long, "{}"),
                (agent_b, "user", "agent-b-private-message", "{}"),
            ],
        )
        await storage.db.execute_many(
            """
            INSERT INTO graph_nodes (node_id, node_type, label, properties)
            VALUES (?, ?, ?, ?)
            """,
            [
                (agent_a, "agent", "Agent A", "{}"),
                ("node-a", "concept", "agent-a-node", f'{{"agent_id":"{agent_a}"}}'),
                (agent_b, "agent", "Agent B", "{}"),
                ("node-b", "concept", "agent-b-node", f'{{"agent_id":"{agent_b}"}}'),
            ],
        )
        await storage.db.execute_many(
            """
            INSERT INTO graph_edges (source_id, target_id, label, properties)
            VALUES (?, ?, ?, ?)
            """,
            [
                (agent_a, "node-a", "has_avatar", "{}"),
                (
                    agent_a,
                    "node-b",
                    "cross-tenant",
                    '{"secret":"agent-b-edge-payload"}',
                ),
            ],
        )
        await storage.db.execute_many(
            "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
            [
                (agent_a, agent_a),
                ("node-a", agent_a),
                (agent_b, agent_b),
                ("node-b", agent_b),
            ],
        )
        await storage.db.execute(
            """
            INSERT INTO graph_edge_owners
                (source_id, target_id, label, agent_id)
            VALUES (?, ?, ?, ?)
            """,
            (agent_a, "node-a", "has_avatar", agent_a),
        )
        await storage.db.execute(
            "INSERT INTO document_chunks (file_hash, content) VALUES (?, ?)",
            ("shared-file-hash", "unowned imported document text"),
        )

        result_a = await query_database_table(
            _request(storage, agent_a),
            "conversation_history",
            limit=50,
            offset=0,
            search=None,
        )
        assert result_a["total_rows"] == 2
        assert {row["agent_id"] for row in result_a["rows"]} == {agent_a}
        assert hostile in {row["content"] for row in result_a["rows"]}
        rendered_long = next(
            row["content"] for row in result_a["rows"] if row["role"] == "assistant"
        )
        assert rendered_long == unicode_long[:500] + "..."

        result_b = await query_database_table(
            _request(storage, agent_b),
            "conversation_history",
            limit=50,
            offset=0,
            search=None,
        )
        assert [row["content"] for row in result_b["rows"]] == [
            "agent-b-private-message"
        ]

        graph_a = await query_database_table(
            _request(storage, agent_a),
            "graph_nodes",
            limit=50,
            offset=0,
            search=None,
        )
        graph_b = await query_database_table(
            _request(storage, agent_b),
            "graph_nodes",
            limit=50,
            offset=0,
            search=None,
        )
        assert {row["label"] for row in graph_a["rows"]} == {
            "Agent A",
            "agent-a-node",
        }
        assert {row["label"] for row in graph_b["rows"]} == {
            "Agent B",
            "agent-b-node",
        }

        edges_a = await query_database_table(
            _request(storage, agent_a),
            "graph_edges",
            limit=50,
            offset=0,
            search=None,
        )
        assert [row["label"] for row in edges_a["rows"]] == ["has_avatar"]
        assert "node-b" not in repr(edges_a)
        assert "agent-b-edge-payload" not in repr(edges_a)

        listing = await list_database_tables(_request(storage, agent_a))
        chunks = next(
            table for table in listing["tables"] if table["name"] == "document_chunks"
        )
        edge_table = next(
            table for table in listing["tables"] if table["name"] == "graph_edges"
        )
        assert edge_table["row_count"] == 1
        assert chunks["queryable"] is False
        assert chunks["row_count"] == -1
        assert listing["db_size"] == -1
        assert listing["db_path"] is None
        assert str(tmp_path / "shared.db") not in repr(listing)

        for document_table in ("documents", "document_chunks", "fts_documents"):
            assert document_table not in ALLOWED_TABLES
            for agent_id in (agent_a, agent_b):
                with pytest.raises(HTTPException) as exc_info:
                    await query_database_table(
                        _request(storage, agent_id),
                        document_table,
                        limit=50,
                        offset=0,
                        search=None,
                    )
                assert exc_info.value.status_code == 403
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_legacy_graph_ownership_backfill_is_precise_and_idempotent(tmp_path):
    """Legacy message/governance edges survive without admitting cross-links."""
    db_path = tmp_path / "legacy-graph.db"
    agent_a = "did:test:legacy-a"
    agent_b = "did:test:legacy-b"
    constitution_hash = "constitution-hash-a"
    message_a = f"message:{agent_a}:m-1"
    concept_a = f"concept:{agent_a}:mom"
    concept_b = f"concept:{agent_b}:secret"
    parent = "did:test:legacy-parent"

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT
            );
            CREATE TABLE graph_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT,
                PRIMARY KEY (source_id, target_id, label)
            );
            """
        )
        conn.executemany(
            "INSERT INTO graph_nodes VALUES (?, ?, ?, ?)",
            [
                (
                    agent_a,
                    "agent",
                    "legacy-a-root",
                    json.dumps({"constitution_hash": constitution_hash}),
                ),
                (agent_b, "agent", "legacy-b-root", "{}"),
                (
                    constitution_hash,
                    "document",
                    "KESTREL_CONSTITUTION",
                    "{}",
                ),
                (message_a, "message", "legacy-message", "{}"),
                (
                    concept_a,
                    "concept",
                    "mom",
                    json.dumps({"agent_id": agent_a}),
                ),
                (
                    concept_b,
                    "concept",
                    "foreign-secret",
                    json.dumps({"agent_id": agent_b}),
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO graph_edges VALUES (?, ?, ?, ?)",
            [
                (message_a, concept_a, "mentions", "{}"),
                (
                    message_a,
                    concept_b,
                    "cross_tenant",
                    json.dumps({"secret": "must-not-leak"}),
                ),
                (agent_a, constitution_hash, "governed_by", "{}"),
                (agent_a, parent, "spawned_by", '{"purpose":"legacy-child"}'),
            ],
        )

    storage = await AsyncStorage.create_sqlite(str(db_path))
    try:
        request = _request(storage, agent_a)
        nodes = await query_database_table(
            request, "graph_nodes", limit=50, offset=0, search=None
        )
        assert {row["label"] for row in nodes["rows"]} == {
            "KESTREL_CONSTITUTION",
            "legacy-a-root",
            "legacy-message",
            "mom",
        }

        edges = await query_database_table(
            request, "graph_edges", limit=50, offset=0, search=None
        )
        assert {row["label"] for row in edges["rows"]} == {
            "mentions",
            "governed_by",
            "spawned_by",
        }
        assert concept_b not in repr(edges)
        assert "must-not-leak" not in repr(edges)

        episode_id = "legacy-episode-a"
        await storage.db.execute(
            """
            INSERT INTO memory_episodes (id, agent_id, title)
            VALUES (?, ?, ?)
            """,
            (episode_id, agent_a, "legacy episode"),
        )
        await storage.db.execute(
            "INSERT INTO graph_nodes VALUES (?, ?, ?, ?)",
            (episode_id, "episode", "legacy-episode-node", "{}"),
        )
        await storage.db.execute(
            "INSERT INTO graph_edges VALUES (?, ?, ?, ?)",
            (agent_a, episode_id, "remembers", "{}"),
        )
        await storage.db._backfill_graph_ownership()
        backfilled_edges = await query_database_table(
            request, "graph_edges", limit=50, offset=0, search=None
        )
        assert {row["label"] for row in backfilled_edges["rows"]} == {
            "governed_by",
            "mentions",
            "remembers",
            "spawned_by",
        }
        node_owner_count = await storage.db.fetchone(
            "SELECT COUNT(*) FROM graph_node_owners WHERE agent_id = ?",
            (agent_a,),
        )
        edge_owner_count = await storage.db.fetchone(
            "SELECT COUNT(*) FROM graph_edge_owners WHERE agent_id = ?",
            (agent_a,),
        )
        assert node_owner_count[0] == 5
        assert edge_owner_count[0] == 4
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_shared_constitution_node_and_each_governed_edge_are_owned(tmp_path):
    """Content-addressed nodes can be shared without sharing private edges."""
    storage = await AsyncStorage.create_sqlite(str(tmp_path / "shared-graph.db"))
    agent_a = "did:test:constitution-a"
    agent_b = "did:test:constitution-b"
    constitution_hash = "same-constitution-hash"
    graph_a = AsyncGraphStore(storage.db, agent_id=agent_a)
    graph_b = AsyncGraphStore(storage.db, agent_id=agent_b)

    try:
        document = GraphNode(
            constitution_hash,
            "document",
            "KESTREL_CONSTITUTION",
            {"hash": constitution_hash, "created_at": "2026-01-01T00:00:00+00:00"},
        )
        await graph_a.add_node(document)
        await graph_a.add_node(
            GraphNode(
                agent_a,
                "agent",
                "agent-a",
                {"constitution_hash": constitution_hash},
            )
        )
        await graph_a.add_edge(agent_a, constitution_hash, "governed_by")

        await graph_b.add_node(
            GraphNode(
                constitution_hash,
                "document",
                "KESTREL_CONSTITUTION",
                {
                    "hash": constitution_hash,
                    "created_at": "2026-02-01T00:00:00+00:00",
                },
            )
        )
        await graph_b.add_node(
            GraphNode(
                agent_b,
                "agent",
                "agent-b",
                {"constitution_hash": constitution_hash},
            )
        )
        await graph_b.add_edge(agent_b, constitution_hash, "governed_by")

        stored_document = await graph_a.get_node(constitution_hash)
        assert stored_document.properties["created_at"] == (
            "2026-01-01T00:00:00+00:00"
        )
        with pytest.raises(TransactionError, match="owned by another agent"):
            await graph_b.add_node(
                GraphNode(
                    constitution_hash,
                    "document",
                    "tampered-label",
                    {"hash": constitution_hash},
                )
            )

        for agent_id, own_source, foreign_source in (
            (agent_a, agent_a, agent_b),
            (agent_b, agent_b, agent_a),
        ):
            nodes = await query_database_table(
                _request(storage, agent_id),
                "graph_nodes",
                limit=50,
                offset=0,
                search=None,
            )
            assert constitution_hash in {row["node_id"] for row in nodes["rows"]}

            edges = await query_database_table(
                _request(storage, agent_id),
                "graph_edges",
                limit=50,
                offset=0,
                search=None,
            )
            assert [(row["source_id"], row["label"]) for row in edges["rows"]] == [
                (own_source, "governed_by")
            ]
            assert foreign_source not in repr(edges)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_bound_graph_api_hides_foreign_documents_and_prevents_delete(tmp_path):
    """The live memories endpoint cannot bypass graph ownership via its facade."""
    db_path = str(tmp_path / "shared-live-memories.db")
    agent_a = "did:test:live-agent-a"
    agent_b = "did:test:live-agent-b"
    storage_a = AsyncStorage(db_path, agent_id=agent_a)
    storage_b = AsyncStorage(db_path, agent_id=agent_b)
    await storage_a.initialize()
    await storage_b.initialize()

    try:
        await storage_a.graph.add_node(GraphNode(agent_a, "agent", "Agent A", {}))
        await storage_b.graph.add_node(GraphNode(agent_b, "agent", "Agent B", {}))

        private_bytes = b"agent-b private document body"
        private_hash = await storage_b.store_file(private_bytes, "private-b.txt")
        await storage_b.graph.add_node(
            GraphNode(
                private_hash,
                "document",
                "Agent B private document",
                {"hash": private_hash},
            )
        )
        private_memory = "memory:agent-b-private"
        await storage_b.graph.add_node(
            GraphNode(private_memory, "memory", "Agent B private memory", {})
        )

        with pytest.raises(HTTPException) as document_error:
            await get_memory_detail(_request(storage_a, agent_a), private_hash)
        assert document_error.value.status_code == 404
        assert private_bytes.decode() not in str(document_error.value.detail)

        with pytest.raises(HTTPException) as delete_error:
            await delete_memory.__wrapped__(
                _request(storage_a, agent_a), private_memory
            )
        assert delete_error.value.status_code == 404
        assert await storage_b.graph.get_node(private_memory) is not None

        assert await storage_a.graph.get_node(private_hash) is None
        assert await storage_a.graph.get_nodes_by_type("document") == []
        assert await storage_a.graph.query_nodes_by_type_and_property(
            "document", order_by_created=False
        ) == []

        # The content-addressed blob has its own tenant capability. Knowing a
        # foreign hash cannot bypass either the storage facade or GET/HEAD.
        assert await storage_a.retrieve_file(private_hash) is None
        assert await storage_a.get_file_metadata(private_hash) is None
        assert not await storage_a.files.file_exists(private_hash)
        for endpoint in (serve_file, check_file):
            with pytest.raises(HTTPException) as file_error:
                await endpoint(private_hash, _request(storage_a, agent_a))
            assert file_error.value.status_code == 404

        assert await storage_b.retrieve_file(private_hash) == private_bytes
    finally:
        await storage_a.close()
        await storage_b.close()


@pytest.mark.asyncio
async def test_unowned_legacy_file_fails_closed_for_bound_storage(tmp_path):
    """A legacy generic blob cannot be claimed or read from a tenant store."""
    import hashlib

    storage = AsyncStorage(
        str(tmp_path / "legacy-unowned-file.db"),
        agent_id="did:test:file-owner",
    )
    await storage.initialize()
    content = b"legacy private bytes"
    content_hash = hashlib.sha256(content).hexdigest()
    try:
        await storage.db.execute(
            "INSERT INTO files (content_hash, original_name, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            (
                content_hash,
                "legacy.txt",
                content,
                '{"agent_id":"did:test:file-owner"}',
            ),
        )

        await storage.db._backfill_file_ownership()
        assert await storage.retrieve_file(content_hash) is None
        assert not await storage.files.file_exists(content_hash)
        with pytest.raises(TransactionError, match="unowned legacy file"):
            await storage.store_file(content, "claim.txt")
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_upgrade_backfills_only_proven_legacy_file_references(tmp_path):
    """Owned document graph references survive; metadata-only blobs do not."""
    import hashlib

    db_path = tmp_path / "legacy-file-reference.db"
    agent_id = "did:test:legacy-file-owner"
    constitution = b"legacy constitution bytes"
    generic = b"legacy generic bytes"
    constitution_hash = hashlib.sha256(constitution).hexdigest()
    generic_hash = hashlib.sha256(generic).hexdigest()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE files (
                content_hash TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                content BLOB,
                metadata TEXT
            );
            CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT
            );
            CREATE TABLE graph_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT,
                PRIMARY KEY (source_id, target_id, label)
            );
            """
        )
        conn.executemany(
            "INSERT INTO files VALUES (?, ?, ?, ?)",
            [
                (constitution_hash, "constitution.md", constitution, "{}"),
                (
                    generic_hash,
                    "generic.txt",
                    generic,
                    json.dumps({"agent_id": agent_id}),
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO graph_nodes VALUES (?, ?, ?, ?)",
            [
                (
                    agent_id,
                    "agent",
                    "Legacy Agent",
                    json.dumps({"constitution_hash": constitution_hash}),
                ),
                (
                    constitution_hash,
                    "document",
                    "KESTREL_CONSTITUTION",
                    "{}",
                ),
            ],
        )
        conn.execute(
            "INSERT INTO graph_edges VALUES (?, ?, 'governed_by', '{}')",
            (agent_id, constitution_hash),
        )

    storage = AsyncStorage(str(db_path), agent_id=agent_id)
    await storage.initialize()
    try:
        assert await storage.retrieve_file(constitution_hash) == constitution
        assert await storage.retrieve_file(generic_hash) is None
        owners = await storage.db.fetchall(
            "SELECT content_hash, agent_id FROM file_owners ORDER BY content_hash"
        )
        assert owners == [(constitution_hash, agent_id)]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_shared_storage_stats_are_tenant_scoped_and_hide_host_layout(tmp_path):
    db_path = str(tmp_path / "shared-stats.db")
    agent_a = "did:test:stats-a"
    agent_b = "did:test:stats-b"
    storage_a = AsyncStorage(db_path, agent_id=agent_a)
    storage_b = AsyncStorage(db_path, agent_id=agent_b)
    await storage_a.initialize()
    await storage_b.initialize()

    try:
        await storage_a.graph.add_node(GraphNode(agent_a, "agent", "A", {}))
        await storage_a.graph.add_node(
            GraphNode("memory:a", "memory", "A memory", {})
        )
        await storage_b.graph.add_node(GraphNode(agent_b, "agent", "B", {}))
        await storage_b.graph.add_node(
            GraphNode("memory:b", "memory", "B memory", {})
        )
        await storage_b.graph.add_node(
            GraphNode("memory:b2", "memory", "B memory 2", {})
        )
        await storage_a.add_conversation("user", "a-only")
        await storage_b.add_conversation("user", "b-one")
        await storage_b.add_conversation("assistant", "b-two")
        bytes_a = b"agent-a-file"
        bytes_b = b"agent-b-file-that-is-longer"
        await storage_a.store_file(bytes_a, "a.txt")
        await storage_b.store_file(bytes_b, "b.txt")

        stats_a = await get_storage_stats(_request(storage_a, agent_a))
        stats_b = await get_storage_stats(_request(storage_b, agent_b))

        assert stats_a["database"] == {"path": None, "size_bytes": -1}
        assert db_path not in repr(stats_a)
        assert stats_a["conversations"] == {"count": 1}
        assert stats_b["conversations"] == {"count": 2}
        assert stats_a["graph_nodes"] == {"agent": 1, "memory": 1}
        assert stats_b["graph_nodes"] == {"agent": 1, "memory": 2}
        # At-rest encryption can change physical byte length; the isolation
        # contract is one owned row per tenant, never the two-row aggregate.
        assert stats_a["files"]["count"] == 1
        assert stats_b["files"]["count"] == 1
        assert stats_a["files"]["size_bytes"] > 0
        assert stats_b["files"]["size_bytes"] > 0

        isolated_a = PrivacyEnforcingStorage(storage_a, PrivacyMode.ISOLATED)
        hidden = await get_storage_stats(_request(isolated_a, agent_a))
        assert hidden["conversations"] == {"count": 0}
        assert hidden["graph_nodes"] == {}
        assert hidden["files"] == {"count": 0, "size_bytes": 0}
    finally:
        await storage_a.close()
        await storage_b.close()


@pytest.mark.asyncio
async def test_bound_delete_releases_only_callers_shared_witnesses(tmp_path):
    """Deleting a shared node retains the peer's physical node and edge."""
    storage = await AsyncStorage.create_sqlite(str(tmp_path / "shared-delete.db"))
    agent_a = "did:test:delete-a"
    agent_b = "did:test:delete-b"
    graph_a = AsyncGraphStore(storage.db, agent_id=agent_a)
    graph_b = AsyncGraphStore(storage.db, agent_id=agent_b)
    first = GraphNode("shared:first", "concept", "first", {"stable": True})
    second = GraphNode("shared:second", "concept", "second", {"stable": True})

    try:
        for graph in (graph_a, graph_b):
            await graph.add_node(first)
            await graph.add_node(second)
            await graph.add_edge(
                first.node_id,
                second.node_id,
                "related",
                {"stable": True},
            )

        await graph_a.delete_node(first.node_id)

        assert await graph_a.get_node(first.node_id) is None
        assert await graph_a.get_edges(second.node_id, direction="in") == []
        assert await graph_b.get_node(first.node_id) == first
        assert [edge.label for edge in await graph_b.get_edges(
            second.node_id, direction="in"
        )] == ["related"]

        owners = await storage.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
            (first.node_id,),
        )
        assert owners == [(agent_b,)]
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_id", [None, ""])
async def test_queryable_tables_fail_closed_without_proven_agent_identity(
    tmp_path, agent_id
):
    storage = await AsyncStorage.create_sqlite(str(tmp_path / "missing-owner.db"))
    try:
        await storage.db.execute(
            """
            INSERT INTO conversation_history (agent_id, role, content, metadata)
            VALUES (?, ?, ?, ?)
            """,
            ("did:test:owner", "user", "must-not-leak", "{}"),
        )

        result = await query_database_table(
            _request(storage, agent_id),
            "conversation_history",
            limit=50,
            offset=0,
            search=None,
        )
        assert result["rows"] == []
        assert result["total_rows"] == 0

        listing = await list_database_tables(_request(storage, agent_id))
        conversation = next(
            table
            for table in listing["tables"]
            if table["name"] == "conversation_history"
        )
        assert conversation["row_count"] == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hostile_table",
    [
        "conversation_history; DROP TABLE conversation_history",
        "graph_nodes/<img src=x onerror=alert(1)>",
        'graph_edges" onclick="alert(1)',
        "表",
    ],
)
async def test_table_identifier_allowlist_rejects_hostile_names(hostile_table):
    with pytest.raises(HTTPException) as exc_info:
        await query_database_table(
            SimpleNamespace(),
            hostile_table,
            limit=50,
            offset=0,
            search=None,
        )
    assert exc_info.value.status_code == 403
