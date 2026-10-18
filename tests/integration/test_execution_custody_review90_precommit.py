"""A cancellable committability probe is not acceptance of native COMMIT."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot,
    execution_commit_outcome,
    execution_denial_scopes,
)
from kestrel_sovereign.storage.db import postgres
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("entry", ["query", "transaction"])
@pytest.mark.parametrize("retired", [False, True])
async def test_retirement_after_native_proof_precedes_commit_acceptance(
    native_pg, monkeypatch, entry, retired,
):
    backend, observer, schema = native_pg
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original precommit owner"))
    entered, release = asyncio.Event(), asyncio.Event()
    original_proof = postgres._prove_committable

    async def held_proof(connection):
        await original_proof(connection)
        entered.set()
        await release.wait()

    monkeypatch.setattr(postgres, "_prove_committable", held_proof)

    async def write():
        with bind_execution_custody_snapshot((scope,)):
            if entry == "transaction":
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES (900, 'original body')")
            else:
                await backend.execute("INSERT INTO effects VALUES (900, 'original body')")

    owner = create_owned_task(write())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if retired:
            scope.revoke("original owner retired before COMMIT acceptance")
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(owner), 3)
        count = await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects')
        assert count == (0 if retired else 1), "retired precommit work reached native COMMIT"
        assert execution_commit_outcome(outcome.error) is None
        if retired:
            assert execution_denial_scopes(outcome.error) == (scope,)
        else:
            assert outcome.error is None
            scope.require_work()
    finally:
        release.set()
        await await_owned_task(owner)
