"""Cooperative Stop drain keeps an actual native cleanup tail."""

import asyncio

import asyncpg
import pytest
from fastapi import FastAPI

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.stop.authority import StopCleanupRegistry
from kestrel_sovereign.stop.types import StopDisposition, StopOutcome, StopScope
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
async def test_bounded_original_cooperative_tail_remains_owned(native_pg, monkeypatch):
    from kestrel_sovereign import server

    backend, observer, schema = native_pg
    backend._execution_custody = ExecutionCustody(GenerationFence())
    registry = StopCleanupRegistry()
    native_exit, committed, release = (
        asyncpg.transaction.Transaction.__aexit__,
        asyncio.Event(),
        asyncio.Event(),
    )
    commits = []
    original = OSError("actual cooperative Stop tail COMMIT ACK lost")

    async def held_ack(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        if error_type is None and not transaction._nested and not commits:
            commits.append(True)
            committed.set()
            await release.wait()
            raise original
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_ack)

    async def original_tail():
        await backend.execute(
            "INSERT INTO effects VALUES (?, ?)",
            (142, "actual original cooperative tail"),
        )
        return StopOutcome(
            scope=StopScope.AGENT,
            requested_target="did:test:original",
            resolved_target="did:test:original",
            agent_id="did:test:original",
            disposition=StopDisposition.STOPPED,
            correlation_id="native-stop-tail",
        )

    tail = create_owned_task(original_tail())
    registry.retain(tail)
    original_drain = registry.drain

    async def bounded_original_drain():
        return await original_drain(timeout_seconds=0.02)

    monkeypatch.setattr(registry, "drain", bounded_original_drain)
    app = FastAPI()
    app.state.stop_cleanup_registry = registry
    try:
        await asyncio.wait_for(committed.wait(), 5)
        caught = None
        try:
            await server._shutdown_stop_cleanup(app)
        except BaseException as error:
            caught = error
        assert not tail.done() and tail.cancelling() == 0
        assert app.state.stop_cleanup_registry is registry, (
            "bounded Stop drain erased its live original native tail"
        )
        assert caught is not None and is_execution_control_error(caught)
        assert any(
            resource is registry and owner is tail
            for resource, owner in execution_cleanup_debts(caught)
        )
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=142'
            )
            == 1
        )
        release.set()
        result = await await_owned_task(tail)
        assert execution_commit_outcome(result.error) == "unknown"
        assert execution_commit_outcome(registry._terminal_error) == "unknown"
    finally:
        release.set()
        await await_owned_task(tail)
        try:
            await original_drain(timeout_seconds=1)
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        backend._execution_custody = None
