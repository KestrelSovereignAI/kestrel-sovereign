"""Actual PostgreSQL attestation rollback must preserve its explicit cause."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import asyncpg

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    bind_execution_custody_snapshot,
    is_execution_control_error,
)
from kestrel_sovereign.features.memory.reflection_hook import (
    RetrievalMarks,
    RetrievedMemoryCandidate,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.memory_attestation_state import (
    ATTESTATION_MARKS_TABLE,
    ensure_attestation_tables,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.unit.test_execution_custody_review43_native import carries

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_actual_pg_attestation_rollback_retains_query_authority_cause(native_pg, monkeypatch):
    backend, _, _ = native_pg
    db = await AsyncDatabase.from_connected_backend(backend)
    await ensure_attestation_tables(db)
    scope = ExecutionCustody(GenerationFence())
    receipts = []
    writes = []
    native_execute = asyncpg.Connection.execute

    async def execute_then_revoke(connection, query, *args, **kwargs):
        result = await native_execute(connection, query, *args, **kwargs)
        if "INSERT INTO effects" in query:
            writes.append(result)
            scope.revoke("original attestation admission lost after native query")
        return result

    monkeypatch.setattr(asyncpg.Connection, "execute", execute_then_revoke)

    @asynccontextmanager
    async def durable():
        yield True

    async def mark_applied(*args, **kwargs):
        try:
            await backend.execute("INSERT INTO effects VALUES (201, 'must not write')")
        except BaseException as error:
            receipts.append((error, error.__cause__))
            raise

    marks = RetrievalMarks(db, "did:test:attestation-original")
    memory = SimpleNamespace(durable_writes=durable, mark_applied=mark_applied)
    item = RetrievedMemoryCandidate(201, "offline", datetime.now(timezone.utc).isoformat(), None, "user")
    with bind_execution_custody_snapshot((scope,)):
        with pytest.raises(BaseException) as caught:
            await marks.record(memory, item, reason="offline")
    assert len(receipts) == 1 and receipts[0][1] is not None
    assert writes == ["INSERT 0 1"], "control was not emitted by an actual native post-query check"
    original, cause = receipts[0]
    assert carries(caught.value, original)
    assert original.__cause__ is cause, "attestation unwrap destroyed original native cause"
    assert is_execution_control_error(caught.value)
    assert await db.fetchval(f"SELECT COUNT(*) FROM {ATTESTATION_MARKS_TABLE}") == 0
    assert await db.fetchval("SELECT COUNT(*) FROM effects") == 0
