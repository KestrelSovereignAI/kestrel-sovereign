"""Security regressions for the read-only database explorer."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from kestrel_sovereign.endpoints.database import (
    ALLOWED_TABLES,
    list_database_tables,
    query_database_table,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage


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
                ("node-a", "concept", "agent-a-node", f'{{"agent_id":"{agent_a}"}}'),
                ("node-b", "concept", "agent-b-node", f'{{"agent_id":"{agent_b}"}}'),
            ],
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
        assert [row["label"] for row in graph_a["rows"]] == ["agent-a-node"]
        assert [row["label"] for row in graph_b["rows"]] == ["agent-b-node"]

        listing = await list_database_tables(_request(storage, agent_a))
        chunks = next(
            table for table in listing["tables"] if table["name"] == "document_chunks"
        )
        assert chunks["queryable"] is False

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
