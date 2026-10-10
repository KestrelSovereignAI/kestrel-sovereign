"""A verified exit must retain native graph/content custody through commit."""

from uuid import uuid4
import asyncio
from contextlib import suppress
from dataclasses import replace

import pytest

from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_refusal_races import _agent
from kestrel_sovereign.agent.constitution import ConstitutionMixin


async def _assert_integrity_refusal(agent, before):
    """A rejected exit may record restriction, never authority or success."""
    current = await agent._constitution_state_store.load(agent.agent_id)
    assert current.safe_mode is True
    assert current.safe_mode_cause == "integrity"
    assert current.revision == before.revision + 1
    assert current.safe_mode_reason == agent._constitution_audit_commit_error
    assert replace(
        current, safe_mode_reason=before.safe_mode_reason,
        safe_mode_cause=before.safe_mode_cause, revision=before.revision,
        updated_at=before.updated_at,
    ) == before
    events = await agent._constitution_state_store.list_events(agent.agent_id)
    assert events[-1]["event_type"] == "safe_mode_entered"
    assert events[-1]["reason"] == current.safe_mode_reason


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_exit_does_not_lock_unreserved_stale_governance_rows(db_backend):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL independent stale-target cleanup custody")
    import asyncpg

    identity = "did:test:exit-stale-custody:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    peer = await asyncpg.connect(db_backend._dsn)
    try:
        agent = await _agent(storage)
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "constitution.md"
        )
        await storage.add_node(
            GraphNode(
                node_id=identity,
                node_type="agent",
                label="stale custody",
                properties={"constitution_hash": digest},
            )
        )
        await agent._anchor_constitution_governance(digest)
        stale = uuid4().hex
        await storage.add_node(
            GraphNode(node_id=stale, node_type="document", label="stale", properties={})
        )
        await storage.add_edge(identity, stale, "governed_by")
        async with peer.transaction():
            await peer.fetch(
                "SELECT target_id FROM graph_edge_owners WHERE source_id=$1 AND target_id=$2 FOR UPDATE",
                identity,
                stale,
            )

            # Exit reserves identity/current digest only. It must not reach
            # into a cleanup's independently held stale-target ownership.
            async def exit_custody():
                async with storage.transaction():
                    await ConstitutionMixin._lock_verified_constitution_exit(agent)

            await asyncio.wait_for(exit_custody(), timeout=2)
    finally:
        await peer.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("evidence", ["identity-owner", "edge", "edge-owner"])
async def test_exit_requires_actual_governance_witness_custody(
    db_backend, monkeypatch, evidence
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL absent-governance witness replacement")
    import asyncpg

    identity = "did:test:governance-lock-row:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    external = await asyncpg.connect(db_backend._dsn)
    try:
        agent = await _agent(storage)
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md"
        )
        await storage.add_node(
            GraphNode(
                node_id=identity,
                node_type="agent",
                label="row witness",
                properties={"constitution_hash": digest},
            )
        )
        await agent._anchor_constitution_governance(digest)
        await agent.enter_safe_mode("governance custody race")
        before = await agent._constitution_state_store.load(identity)
        table, predicate, params = {
            "identity-owner": (
                "graph_node_owners",
                "node_id=$1 AND agent_id=$1",
                [identity],
            ),
            "edge": (
                "graph_edges",
                "source_id=$1 AND target_id=$2 AND label='governed_by'",
                [identity, digest],
            ),
            "edge-owner": (
                "graph_edge_owners",
                "source_id=$1 AND target_id=$2 AND label='governed_by' AND agent_id=$1",
                [identity, digest],
            ),
        }[evidence]
        snapshot = await external.fetchrow(
            f"SELECT * FROM {table} WHERE {predicate}", *params
        )
        native_fetch = db_backend.fetch_all
        reached = []

        async def replaced(query, arguments=()):
            # Match the physical witness read, not the ownership EXISTS clause
            # embedded in the preceding complete graph-node reservation.
            columns_prefix = "node_id" if evidence == "identity-owner" else "target_id"
            is_witness_read = (
                query.startswith(f"SELECT node_id FROM {table} ")
                if evidence == "identity-owner"
                else query.startswith(f"SELECT {columns_prefix}")
                and f"FROM {table} " in query
            )
            if not reached and is_witness_read and "FOR UPDATE" in query:
                await external.execute(
                    f"DELETE FROM {table} WHERE {predicate}", *params
                )
                result = await native_fetch(query, arguments)
                assert result == []
                columns = list(snapshot.keys())
                await external.execute(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('$' + str(i + 1) for i in range(len(columns)))})",
                    *snapshot.values(),
                )
                reached.append(True)
                return result
            return await native_fetch(query, arguments)

        monkeypatch.setattr(db_backend, "fetch_all", replaced)
        result = await agent.exit_safe_mode(authorization="fixture sovereign")
        assert reached == [True], result
        assert result.startswith("Safe Mode remains active:"), result
        await _assert_integrity_refusal(agent, before)
    finally:
        await external.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("evidence", ["file", "file-owner"])
async def test_exit_refuses_absent_lock_row_even_when_recreated_before_verification(
    db_backend, monkeypatch, evidence
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL absent-row replacement custody")
    import asyncpg

    identity = "did:test:exit-absent-lock:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    external = await asyncpg.connect(db_backend._dsn)
    try:
        agent = await _agent(storage)
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md"
        )
        await storage.add_node(
            GraphNode(
                node_id=identity,
                node_type="agent",
                label="absent",
                properties={"constitution_hash": digest},
            )
        )
        await agent._anchor_constitution_governance(digest)
        await agent.enter_safe_mode("absent-lock race")
        before = await agent._constitution_state_store.load(identity)
        table = "files" if evidence == "file" else "file_owners"
        snapshot = await external.fetchrow(
            f"SELECT * FROM {table} WHERE content_hash=$1"
            + (" AND agent_id=$2" if evidence == "file-owner" else ""),
            *([digest, identity] if evidence == "file-owner" else [digest]),
        )
        native_fetch = db_backend.fetch_one
        reached = []

        async def absent_then_recreated(query, params=()):
            if not reached and query == (
                "SELECT content_hash FROM "
                + table
                + " WHERE content_hash = ?"
                + (" AND agent_id = ?" if evidence == "file-owner" else "")
                + " FOR UPDATE"
            ):
                await external.execute(
                    f"DELETE FROM {table} WHERE content_hash=$1"
                    + (" AND agent_id=$2" if evidence == "file-owner" else ""),
                    *params,
                )
                result = await native_fetch(query, params)
                assert result is None
                columns = list(snapshot.keys())
                await external.execute(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('$' + str(i + 1) for i in range(len(columns)))})",
                    *snapshot.values(),
                )
                reached.append(True)
                return result
            return await native_fetch(query, params)

        monkeypatch.setattr(db_backend, "fetch_one", absent_then_recreated)
        result = await agent.exit_safe_mode(authorization="explicit fixture owner")
        assert reached == [True]
        assert result.startswith("Safe Mode remains active:"), result
        assert agent._safe_mode is True
        await _assert_integrity_refusal(agent, before)
    finally:
        await external.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_exit_takes_edge_owners_before_edge_rows(db_backend, monkeypatch):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL independent physical owner/edge lock order")
    import asyncpg

    identity = "did:test:exit-lock-order:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    external = probe = transaction = task = None
    try:
        agent = await _agent(storage)
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md"
        )
        await storage.add_node(
            GraphNode(
                node_id=identity,
                node_type="agent",
                label="exit order",
                properties={"constitution_hash": digest},
            )
        )
        await agent._anchor_constitution_governance(digest)
        external = await asyncpg.connect(db_backend._dsn)
        probe = await asyncpg.connect(db_backend._dsn)
        transaction = external.transaction()
        await transaction.start()
        await external.execute("SET LOCAL lock_timeout = '100ms'")
        await external.fetchrow(
            "SELECT agent_id FROM graph_edge_owners WHERE source_id=$1 AND target_id=$2 AND label='governed_by' AND agent_id=$1 FOR UPDATE",
            identity,
            digest,
        )
        reached = asyncio.Event()
        pid = None
        native_fetch = type(db_backend).fetch_all

        async def observed_fetch(backend, query, params=()):
            nonlocal pid
            if (
                asyncio.current_task() is task
                and "FROM graph_edge_owners" in query
                and "FOR UPDATE" in query
            ):
                pid = await backend.fetch_val("SELECT pg_backend_pid()")
                reached.set()
            return await native_fetch(backend, query, params)

        monkeypatch.setattr(type(db_backend), "fetch_all", observed_fetch)

        async def verify_exit():
            async with storage.transaction():
                await ConstitutionMixin._lock_verified_constitution_exit(agent)

        task = asyncio.create_task(verify_exit())
        await asyncio.wait_for(reached.wait(), 5)
        async with asyncio.timeout(5):
            while (
                await probe.fetchval(
                    "SELECT wait_event_type FROM pg_stat_activity WHERE pid=$1", pid
                )
                != "Lock"
            ):
                await asyncio.sleep(0.02)
        # Exit waiting on the owner must not already own the edge. Otherwise
        # the actual owner-first native deletion order creates a lock cycle.
        assert (
            await external.execute(
                "UPDATE graph_edges SET properties=properties WHERE source_id=$1 AND target_id=$2 AND label='governed_by'",
                identity,
                digest,
            )
            == "UPDATE 1"
        )
        await transaction.rollback()
        transaction = None
        await asyncio.wait_for(task, 5)
    finally:
        if transaction is not None:
            await transaction.rollback()
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task
        if external is not None:
            await external.close()
        if probe is not None:
            await probe.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("damage", [None, "edge", "ownership", "failed_genesis", "malformed_genesis", "encrypted_blob"])
async def test_native_exit_refuses_governing_drift_after_verification(
    db_backend, damage
):
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:exit-witness:" + uuid4().hex
    )
    await storage.initialize()
    original_blob = None
    try:
        agent = await _agent(storage)
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md"
        )
        await storage.add_node(
            GraphNode(
                node_id=agent.agent_id,
                node_type="agent",
                label="exit",
                properties={"constitution_hash": digest},
            )
        )
        await agent._anchor_constitution_governance(digest)
        await agent.enter_safe_mode("verified exit race fixture")
        original_blob = await storage.db.fetchone("SELECT content, metadata FROM files WHERE content_hash=?", (digest,))
        before = await agent._constitution_state_store.load(agent.agent_id)
        events = await agent._constitution_state_store.list_events(agent.agent_id)
        native_persist = agent._persist_constitution_runtime_state
        failures = []
        native_mark = agent._mark_constitution_state_unavailable

        def record_native_failure(exc, **kwargs):
            failures.append(exc)
            return native_mark(exc, **kwargs)

        agent._mark_constitution_state_unavailable = record_native_failure

        async def drift_then_persist(**kwargs):
            # The real native verifier has returned. Mutate actual durable SQL
            # before the exit's owning transaction, without changing its CAS.
            if damage is not None and kwargs.get("event_type") == "safe_mode_exited":
                if damage == "edge":
                    await storage.delete_edge(agent.agent_id, digest, "governed_by")
                elif damage == "ownership":
                    await storage.db.execute_commit(
                        "DELETE FROM file_owners WHERE content_hash = ? AND agent_id = ?",
                        (digest, agent.agent_id),
                    )
                elif damage == "encrypted_blob":
                    await storage.db.execute_commit(
                        "UPDATE files SET content=?, metadata=? WHERE content_hash=?",
                        (b"corrupt encrypted governing blob", '{"enc":true}', digest),
                    )
                else:
                    node = await storage.get_node(agent.agent_id)
                    node.properties["genesis_audit"] = {
                        "status": "failed",
                        "constitution_hash": digest,
                        "risk_level": 3,
                        "completed_at": "2026-10-09T21:00:00Z",
                        "audited": True,
                        "reasoning": "Concurrent rejection",
                        "provenance": "fixture",
                    }
                    if damage == "malformed_genesis":
                        node.properties["genesis_audit"]["completed_at"] = "not an instant"
                    await storage.add_node(node)
            return await native_persist(**kwargs)

        agent._persist_constitution_runtime_state = drift_then_persist
        result = await agent.exit_safe_mode(authorization="explicit test owner")
        if damage is None:
            assert "deactivated" in result, result
            assert failures == []
            assert agent._safe_mode is False
            assert (
                await agent._constitution_state_store.load(agent.agent_id)
            ).safe_mode is False
            assert (await agent._constitution_state_store.list_events(agent.agent_id))[
                -1
            ]["event_type"] == "safe_mode_exited"
            return
        assert result.startswith("Safe Mode remains active:"), result
        # Native integrity refusal is not a database availability failure.
        assert failures == []
        expected_reason = {
            "edge": "Missing or mis-targeted governed_by edge",
            "ownership": "Anchored constitution blob is missing",
            "failed_genesis": "requires a passed genesis receipt",
            "malformed_genesis": "invalid completion time",
            "encrypted_blob": "decrypt",
        }[damage]
        detail = agent._constitution_audit_commit_error
        assert expected_reason in (detail.lower() if damage == "encrypted_blob" else detail)
        assert "integrity verification refused" in result
        assert "could not be persisted" not in result
        assert agent._safe_mode is True
        await _assert_integrity_refusal(agent, before)
        final_events = await agent._constitution_state_store.list_events(agent.agent_id)
        assert final_events[:-1] == events
    finally:
        if damage == "encrypted_blob" and original_blob is not None:
            await storage.db.execute_commit(
                "UPDATE files SET content=?, metadata=? WHERE content_hash=?", (*original_blob, digest),
            )
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "row", ["edge", "edge-owner", "file", "file-owner", "identity-owner", "identity"]
)
async def test_native_exit_retains_postgres_evidence_rows_through_commit(
    db_backend, row
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL physical row-lock proof")
    import asyncpg

    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:held-exit:" + uuid4().hex
    )
    await storage.initialize()
    external = None
    try:
        agent = await _agent(storage)
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md"
        )
        await storage.add_node(
            GraphNode(
                node_id=agent.agent_id,
                node_type="agent",
                label="exit",
                properties={"constitution_hash": digest},
            )
        )
        await agent._anchor_constitution_governance(digest)
        await agent.enter_safe_mode("physical lock proof")
        external = await asyncpg.connect(db_backend._dsn)
        await external.execute("SET lock_timeout = '100ms'")
        statements = {
            "edge": (
                "DELETE FROM graph_edges WHERE source_id=$1 AND target_id=$2 AND label='governed_by'",
                (agent.agent_id, digest),
            ),
            "edge-owner": (
                "DELETE FROM graph_edge_owners WHERE source_id=$1 AND target_id=$2 AND agent_id=$1 AND label='governed_by'",
                (agent.agent_id, digest),
            ),
            "file": ("DELETE FROM files WHERE content_hash=$1", (digest,)),
            "file-owner": (
                "DELETE FROM file_owners WHERE content_hash=$1 AND agent_id=$2",
                (digest, agent.agent_id),
            ),
            "identity-owner": (
                "DELETE FROM graph_node_owners WHERE node_id=$1 AND agent_id=$1",
                (agent.agent_id,),
            ),
            "identity": (
                "UPDATE graph_nodes SET properties='{}' WHERE node_id=$1",
                (agent.agent_id,),
            ),
        }
        query, params = statements[row]
        native_verify = agent._verify_constitution_integrity
        held = False

        async def verify_and_try_independent_writer():
            nonlocal held
            verified = await native_verify()
            if storage.owns_open_transaction:
                assert verified[0] is True
                # The server, not a mocked lock or a timing assertion, proves
                # an independent native writer cannot alter this evidence.
                with pytest.raises(asyncpg.LockNotAvailableError):
                    await external.execute(query, *params)
                held = True
            return verified

        agent._verify_constitution_integrity = verify_and_try_independent_writer
        result = await agent.exit_safe_mode(authorization="explicit test owner")
        assert "deactivated" in result, result
        assert held is True
        assert (
            await agent._constitution_state_store.load(agent.agent_id)
        ).safe_mode is False
        # The same writer succeeds after the actual owner commits; no leaked
        # transaction or permanent test lock supplied the apparent protection.
        assert (await external.execute(query, *params)).endswith("1")
    finally:
        if external is not None:
            await external.close()
        await storage.close()
