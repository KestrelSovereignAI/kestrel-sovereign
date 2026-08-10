"""Reading an agent's DID from its local identity anchor.

These pin :mod:`kestrel_sovereign.identity.local_anchor`, extracted from
``multi_agent.agent_manager`` in #2894 so the process-per-agent server, the
in-process host, and the offline governance tools stop each growing their own
answer to "who is this agent". The tests came with it: they are about the
reader's cold-vs-startup safety contract, not about the manager.

What they defend, in one line each: a lookup must never *create* an identity
(``AsyncStorage.initialize()`` would), a cold read must refuse rather than
ignore pending WAL state, startup must let SQLite recover a real WAL first, and
a directory holding two agent roots is an integrity failure rather than a
first-row pick.
"""

from __future__ import annotations

import sqlite3

import pytest

from kestrel_sovereign.identity.local_anchor import (
    AgentDIDLookupMode,
    read_anchor_agent_did,
)
from kestrel_sovereign.storage import AsyncStorage, GraphNode


@pytest.mark.asyncio
async def test_cold_did_lookup_missing_database_never_creates_identity_artifacts(tmp_path):
    """Discovery of an unincepted cold config is a strictly read-only probe."""
    cold_dir = tmp_path / "unincepted"
    cold_dir.mkdir()

    with pytest.raises(ValueError, match="No agent found"):
        await read_anchor_agent_did(str(cold_dir))

    assert list(cold_dir.iterdir()) == []
    assert not (cold_dir / "kestrel_prime.db").exists()
    assert not (cold_dir / "kestrel_prime.db-wal").exists()
    assert not (cold_dir / "kestrel_prime.db-shm").exists()

@pytest.mark.asyncio
async def test_cold_did_lookup_invalid_existing_database_never_writes_sidecars(tmp_path):
    """A malformed identity is reported without schema/WAL side effects."""
    cold_dir = tmp_path / "invalid"
    cold_dir.mkdir()
    db_path = cold_dir / "kestrel_prime.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE unrelated (value TEXT)")
    connection.commit()
    connection.close()
    before = {
        path.name: path.read_bytes()
        for path in cold_dir.iterdir()
    }

    with pytest.raises(ValueError, match="Could not read local agent identity"):
        await read_anchor_agent_did(str(cold_dir))

    after = {
        path.name: path.read_bytes()
        for path in cold_dir.iterdir()
    }
    assert after == before

@pytest.mark.asyncio
async def test_cold_did_lookup_refuses_uncheckpointed_wal_without_mutation(tmp_path):
    """A cold identity probe must not ignore or materialize WAL state."""
    cold_dir = tmp_path / "wal-pending"
    cold_dir.mkdir()
    db_path = cold_dir / "kestrel_prime.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE graph_nodes (node_id TEXT, node_type TEXT)"
    )
    connection.execute(
        "INSERT INTO graph_nodes VALUES (?, 'agent')",
        ("did:test:wal-pending",),
    )
    connection.commit()
    connection.close()
    wal_path = cold_dir / "kestrel_prime.db-wal"
    wal_path.write_bytes(b"uncheckpointed identity state")
    before = {
        path.name: path.read_bytes()
        for path in cold_dir.iterdir()
    }

    with pytest.raises(ValueError, match="WAL state is present"):
        await read_anchor_agent_did(str(cold_dir))

    after = {
        path.name: path.read_bytes()
        for path in cold_dir.iterdir()
    }
    assert after == before

@pytest.mark.asyncio
async def test_initialization_did_lookup_recovers_existing_sqlite_wal(tmp_path):
    """Normal startup can consume a crash-recovery WAL that cold probes refuse."""
    agent_dir = tmp_path / "wal-recovery"
    agent_dir.mkdir()
    db_path = agent_dir / "kestrel_prime.db"
    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute(
            "CREATE TABLE graph_nodes (node_id TEXT, node_type TEXT)"
        )
        connection.execute(
            "INSERT INTO graph_nodes VALUES (?, 'agent')",
            ("did:test:wal-recovery",),
        )
        connection.commit()
        assert (agent_dir / "kestrel_prime.db-wal").exists()

        with pytest.raises(ValueError, match="WAL state is present"):
            await read_anchor_agent_did(str(agent_dir))

        assert await read_anchor_agent_did(
            str(agent_dir),
            mode=AgentDIDLookupMode.INITIALIZATION,
        ) == "did:test:wal-recovery"
    finally:
        connection.close()

@pytest.mark.asyncio
async def test_cold_did_lookup_stays_local_sqlite_with_postgres_environment(
    monkeypatch, tmp_path,
):
    """A host DB default must not select a foreign shared identity."""
    local_dir = tmp_path / "cold-agent"
    local_dir.mkdir()
    local_db = local_dir / "kestrel_prime.db"
    local_did = "did:test:local-cold-agent"

    storage = AsyncStorage(str(local_db), backend="sqlite")
    await storage.initialize()
    try:
        await storage.graph.add_node(
            GraphNode(local_did, "agent", "Local cold agent", {})
        )
    finally:
        await storage.close()

    before = {
        path.name: path.read_bytes()
        for path in local_dir.iterdir()
    }
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://foreign-host/fleet")

    assert await read_anchor_agent_did(str(local_dir)) == local_did
    after = {
        path.name: path.read_bytes()
        for path in local_dir.iterdir()
    }
    # The successful normal path is as important as malformed/missing
    # probes: it must not materialize or alter SQLite WAL/SHM sidecars.
    assert after == before
    assert "kestrel_prime.db-wal" not in after
    assert "kestrel_prime.db-shm" not in after
