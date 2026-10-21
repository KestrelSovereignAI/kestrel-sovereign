"""Actual observability DELETE uncertainty survives the independent sweep."""

from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_table", ["a2a_tool_dispatches", "a2a_observability"])
async def test_public_observability_purge_retains_original_native_delete(
    native_pg, monkeypatch, failed_table
):
    backend, observer, schema = native_pg
    store = ObservabilityStore(backend)
    await store.initialize()
    await backend.execute(
        "INSERT INTO a2a_observability(id,agent_name,event_type) VALUES ('owned', 'original', 'native')"
    )
    await backend.execute(
        "INSERT INTO a2a_tool_dispatches(agent_did,turn_id,tool_name,adapter,args_redacted,result_status,latency_ms) VALUES ('did:test:original', 'original', 'native', 'offline', '{}', 'success', 0)"
    )
    backend._execution_custody = ExecutionCustody(GenerationFence())
    execute = backend.execute
    original_errors, native_acks, calls = [], [], []

    async def actual_delete(statement, *args, **kwargs):
        if statement.startswith("DELETE FROM "):
            calls.append(statement)
        if statement.startswith("DELETE FROM " + failed_table + " "):
            native_acks.append(
                lose_native_ack(
                    monkeypatch,
                    boundary="commit",
                    original=OSError("original actual DELETE ACK lost"),
                )
            )
        try:
            return await execute(statement, *args, **kwargs)
        except BaseException as error:
            original_errors.append(error)
            raise

    monkeypatch.setattr(backend, "execute", actual_delete)
    try:
        with pytest.raises(BaseException) as propagated:
            await store.purge_observability_since(
                (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
                agent_did="did:test:original",
                agent_name="original",
            )
        assert len(calls) == 2 and native_acks == [[True]]
        assert len(original_errors) == (
            2 if failed_table == "a2a_tool_dispatches" else 1
        )
        primary = original_errors[0]
        assert execution_commit_outcome(primary) == "unknown"
        assert carries(propagated.value, primary), (
            "observability purge stringified its actual native control owner"
        )
        assert execution_commit_outcome(propagated.value) == "unknown"
        for table in ("a2a_tool_dispatches", "a2a_observability"):
            # The second independent sweep is attempted, but a quarantined
            # original backend must not admit new SQL after the first UNKNOWN.
            expected = int(
                failed_table == "a2a_tool_dispatches" and table == "a2a_observability"
            )
            assert (
                await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".{table}')
                == expected
            )
    finally:
        backend._execution_custody = None
