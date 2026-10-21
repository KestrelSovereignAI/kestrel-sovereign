"""Exact original turn-scope denials versus foreign/mixed native control.

Scopes are explicit trusted conformance inputs under one unchanged real native
runtime fence, not new Agent genesis/READY or independent policy grants.
"""

import pytest

from kestrel_sovereign.execution_custody import (
    AdvisoryExecutionFence,
    ExecutionCustody,
    _explicit_control_chain,
    bind_execution_custody_snapshot,
    execution_commit_outcome,
    execution_denial_scopes,
    is_execution_control_error,
)
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["original-turn", "foreign", "mixed"])
async def test_original_memory_turn_denial_cannot_poison_a_different_admitted_turn(
    native_memory_owners, monkeypatch, kind
):
    n = native_memory_owners
    row = await n.pg_storage.db.fetchval(
        "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
        (n.did, "assistant", "original scoped rehearsal", "{}"),
    )
    runtime = ExecutionCustody(GenerationFence())
    n.backend._execution_custody = n.service._execution_custody = runtime
    foreign = ExecutionCustody(AdvisoryExecutionFence(n.lease))
    turn = ExecutionCustody(GenerationFence())
    retriever = n.pg_memory.retriever
    actual = retriever.update_access
    sources = []
    if kind != "original-turn":

        async def actual_then_foreign_control(*args, **kwargs):
            if kind == "mixed":
                turn.revoke("declared original turn withdrawal before native work")
            try:
                await actual(*args, **kwargs)
            except BaseException as local:
                if kind != "mixed":
                    raise
                sources.append(local)
            try:
                foreign.require_work()
            except BaseException as error:
                sources.append(error)
                if kind == "mixed":
                    raise BaseExceptionGroup("declared mixed callback control", sources)
                raise

        monkeypatch.setattr(retriever, "update_access", actual_then_foreign_control)

    first = None
    with bind_execution_custody_snapshot((turn,)):
        if kind == "original-turn":
            # Capture the genuinely live original first, then withdraw it;
            # the canonical binding correctly refuses an already-dead scope.
            turn.revoke("declared original turn withdrawal before native work")
        try:
            await retriever.record_accesses([row], n.did)
        except BaseException as error:
            first = error
    assert first is not None and is_execution_control_error(first)
    assert execution_commit_outcome(first) is None
    denied = execution_denial_scopes(first)
    assert denied and all(isinstance(scope, ExecutionCustody) for scope in denied)
    if kind == "original-turn":
        assert denied == (turn,)
    else:
        assert foreign in denied
        assert all(
            source in tuple(_explicit_control_chain(first)) for source in sources
        )

    next_turn = ExecutionCustody(GenerationFence())
    failure = None
    with bind_execution_custody_snapshot((next_turn,)):
        try:
            if kind == "original-turn":
                await retriever.record_accesses([row], n.did)
            else:
                await retriever.retrieve(
                    "original scoped rehearsal", n.did, read_only=True
                )
        except BaseException as error:
            failure = error
    if kind == "original-turn":
        assert failure is None, (
            f"old turn denial poisoned independent admitted turn: {failure!r}"
        )
        assert retriever._access_update_control is None
    else:
        assert failure is not None and is_execution_control_error(failure)
        assert all(
            source in tuple(_explicit_control_chain(failure)) for source in sources
        )
    assert n.backend._execution_custody is runtime
    runtime.require_work()
    assert await n.backend.fetch_val("SELECT generation FROM authority WHERE id=1") == 1
    count = await n.observer.fetchval(
        f"SELECT COALESCE((metadata::jsonb->>'access_count')::int,0) FROM \"{n.schema}\".conversation_history WHERE id=$1",
        row,
    )
    assert count == (0 if kind == "mixed" else 1)
    if kind == "original-turn":
        with pytest.raises(BaseException):
            turn.require_work()


@pytest.mark.asyncio
async def test_original_scope_denial_with_actual_committed_effect_is_not_local_only(
    native_memory_owners,
):
    from contextlib import asynccontextmanager

    from kestrel_sovereign.execution_custody import execution_cleanup_debts
    from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter

    n = native_memory_owners
    row = await n.pg_storage.db.fetchval(
        "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
        (n.did, "assistant", "actual post-COMMIT turn withdrawal", "{}"),
    )
    runtime, turn = (
        ExecutionCustody(GenerationFence()),
        ExecutionCustody(GenerationFence()),
    )
    n.backend._execution_custody = n.service._execution_custody = runtime
    original_pool, fired = n.backend._pool, []

    @asynccontextmanager
    async def original_pool_wrapper(native):
        async with native as connection:
            yield connection
        if not fired:
            fired.append(True)
            turn.revoke("original turn withdrew AFTER native COMMIT and RESET")
            turn.require_work()

    n.backend._pool = PostgresPoolAdapter(original_pool, original_pool_wrapper)
    first = None
    try:
        with bind_execution_custody_snapshot((turn,)):
            try:
                await n.pg_memory.retriever.record_accesses([row], n.did)
            except BaseException as error:
                first = error
        assert fired and first is not None
        assert execution_commit_outcome(first) == "committed"
        assert not execution_cleanup_debts(first)
        assert execution_denial_scopes(first) == (turn,)
        runtime.require_work()
        assert n.pg_memory.retriever._access_update_control is first

        next_turn, later = ExecutionCustody(GenerationFence()), None
        with bind_execution_custody_snapshot((next_turn,)):
            try:
                await n.pg_memory.retriever.record_accesses([row], n.did)
            except BaseException as error:
                later = error
        assert later is first, "confirmed effect was mistaken for a pure local denial"
        assert (
            await n.observer.fetchval(
                f"SELECT (metadata::jsonb->>'access_count')::int FROM \"{n.schema}\".conversation_history WHERE id=$1",
                row,
            )
            == 1
        )
    finally:
        n.backend._pool = original_pool
