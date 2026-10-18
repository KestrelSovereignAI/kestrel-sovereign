"""Failed native preflight cleanup, not successful dual-cluster Hold boot."""

import asyncio
import inspect
import os
import sys

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome,
)
from kestrel_sovereign.hold import state
from kestrel_sovereign.storage.db import postgres
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review30_pg import lose_ack
from tests.utils.postgres_schema import with_search_path


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refused", "close-ack", "withdrawal"])
async def test_failed_native_preflight_retains_body_close_and_observer(
    native_pg, monkeypatch, mode,
):
    backend, observer, schema = native_pg
    committed, release_commit = asyncio.Event(), asyncio.Event()

    async def effect():
        with bind_execution_custody(GenerationFence()):
            await backend.execute("INSERT INTO effects VALUES (196, 'native Hold close evidence')")

    with monkeypatch.context() as ack:
        lose_ack(ack, cancelled=False, entered=committed, release=release_commit)
        source = create_owned_task(effect(), retain_originals=True)
        try:
            await asyncio.wait_for(committed.wait(), 5)
            release_commit.set()
            original = (await await_owned_task(source)).error
        finally:
            release_commit.set()
            await await_owned_task(source)
    assert original is not None and execution_commit_outcome(original) == "unknown"
    original_graph = tuple(_explicit_control_chain(original))
    original_debts = tuple(execution_cleanup_debts(original))
    backend_type, actual_close = postgres.PostgresBackend, postgres.PostgresBackend.close
    actual_cleanup, actual_join = state._close_postgres_preflight_backends, ownership.await_owned_task
    created, body_errors, withdrawals, refused = [], [], [], []
    close_entered, close_release = asyncio.Event(), asyncio.Event()
    allocation = RuntimeError("actual Hold native closer Task admission refused")
    loop, previous_factory = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    owner = None

    def construct(**kwargs):
        resource = backend_type(**kwargs)
        created.append(resource)
        return resource

    async def close(resource):
        if resource not in created:
            return await actual_close(resource)
        close_entered.set()
        await close_release.wait()
        await actual_close(resource)
        if resource is created[-1]:
            # The exact receipt is produced by a real prior PostgreSQL COMMIT
            # acknowledgement failure, not a fabricated classification flag.
            raise original

    async def cleanup(*resources, **kwargs):
        body_errors.append(sys.exception())
        return await actual_cleanup(*resources, **kwargs)

    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome

    def factory(loop, coroutine, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        operation = frame.f_locals.get("operation", coroutine) if frame else coroutine
        code = getattr(operation, "cr_code", None)
        # Observe the actual old aggregate admission or the canonical native
        # close admission. Names are only fixture scheduling boundaries.
        if (mode == "refused" and not refused and code is not None and
                (code.co_qualname.endswith("_close_postgres_preflight_backends.<locals>.close_all")
                 or code is close.__code__)):
            refused.append((coroutine, operation))
            raise allocation
        return (previous_factory(loop, coroutine, **kwargs) if previous_factory else
                asyncio.Task(coroutine, loop=loop, **kwargs))

    try:
        monkeypatch.setattr(postgres, "PostgresBackend", construct)
        monkeypatch.setattr(backend_type, "close", close)
        monkeypatch.setattr(state, "_close_postgres_preflight_backends", cleanup)
        monkeypatch.setattr(ownership, "await_owned_task", join)
        loop.set_task_factory(factory)
        dsn = with_search_path(os.environ["TEST_POSTGRES_URL"], schema)
        owner = create_owned_task(state.preflight_postgres_hold_custody(dsn, dsn), retain_originals=True)
        if mode != "refused":
            await asyncio.wait_for(close_entered.wait(), 5)
            if mode == "withdrawal":
                owner.cancel("first actual native Hold preflight withdrawal")
                await asyncio.sleep(0)
                owner.cancel("repeated actual native Hold preflight withdrawal")
        close_release.set()
        outcome = await actual_join(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert len(created) == 2 and len(body_errors) == 1 and body_errors[0] is not None
        assert body_errors[0] in chain, "Hold preflight replaced its actual opening failure"
        if mode == "refused":
            assert allocation in chain and refused
            assert any(resource is created[0] and task is None
                       for resource, task in execution_cleanup_debts(outcome.error)), "Hold preflight discarded exact unstarted native closer debt"
            assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                       for pair in refused for coro in pair)
            assert not created[1].is_connected, "one admission refusal skipped independent native retirement"
        else:
            assert original in chain and execution_commit_outcome(outcome.error) == "unknown", "Hold preflight dropped original native close control"
            assert not any(resource.is_connected for resource in created)
        if mode == "withdrawal":
            assert withdrawals and all(error in chain for error in withdrawals), "Hold preflight reconstructed observer cancellation"
        assert tuple(_explicit_control_chain(original)) == original_graph
        assert tuple(execution_cleanup_debts(original)) == original_debts
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=196') == 1
    finally:
        loop.set_task_factory(previous_factory)
        close_release.set()
        if owner is not None:
            await actual_join(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        for resource in created:
            await actual_close(resource)
