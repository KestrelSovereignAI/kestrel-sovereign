"""Real authorized constitutional exit commits retain native ACK outcomes."""
import asyncio
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    bind_execution_custody_snapshot,
    execution_commit_outcome,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_turn_admission import _ready_turn
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("boundary", ["commit", "release"])
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_actual_constitution_exit_preserves_committed_or_unknown_receipt(native_pg, monkeypatch, boundary, carrier):
    backend, observer, schema = native_pg
    storage = AsyncStorage(backend=backend, agent_id="did:test:review54:" + uuid4().hex)
    await storage.initialize()
    agent, _ = await _ready_turn(storage)
    assert await agent.enter_safe_mode("original restriction")
    original = OSError("native acknowledgement lost") if carrier == "error" else asyncio.CancelledError("native acknowledgement cancelled")
    native_fetch = asyncpg.Connection.fetchrow
    native_commit = asyncpg.transaction.Transaction.__aexit__
    native_release = asyncpg.pool.PoolAcquireContext.__aexit__
    updated, acknowledgements = [], []

    async def observed_update(connection, query, *args, **kwargs):
        result = await native_fetch(connection, query, *args, **kwargs)
        if "UPDATE constitution_runtime_state SET" in query:
            updated.append(result)
        return result

    async def lost_commit(transaction, error_type, error, traceback):
        result = await native_commit(transaction, error_type, error, traceback)
        if boundary == "commit" and updated and not acknowledgements and error is None and not transaction._nested:
            acknowledgements.append(True)
            raise original
        return result

    async def lost_release(checkout, error_type, error, traceback):
        result = await native_release(checkout, error_type, error, traceback)
        if boundary == "release" and updated and not acknowledgements and error is None:
            acknowledgements.append(True)
            raise original
        return result

    monkeypatch.setattr(asyncpg.Connection, "fetchrow", observed_update)
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", lost_commit)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lost_release)
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    try:
        with bind_execution_custody_snapshot((scope,)):
            with pytest.raises(BaseException) as caught:
                await agent.exit_safe_mode(authorization="explicit offline sovereign")
        assert updated and acknowledgements == [True]
        assert execution_commit_outcome(caught.value) == ("unknown" if boundary == "commit" else "committed")
        assert caught.value.__cause__ is original
        # A real committed row/event, not a helper exception masquerading as SQL.
        assert await observer.fetchval(f'SELECT safe_mode FROM "{schema}".constitution_runtime_state WHERE agent_id=$1', agent.agent_id) is False
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".constitution_runtime_events WHERE agent_id=$1 AND event_type=\'safe_mode_exited\'', agent.agent_id) == 1
        assert agent._safe_mode is True, "uncertain publication opened cognition before reconciliation"
        assert agent._safe_mode_cause != "state_not_persisted", "committed exit fabricated a never-persisted fact"
    finally:
        monkeypatch.undo()
        backend._execution_custody = None
        await storage.close()
