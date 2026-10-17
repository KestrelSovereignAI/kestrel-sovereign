"""Original preflight probe admission and peer control on actual local PG."""
import asyncio
import inspect
import os

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome,
)
from kestrel_sovereign.hold import state
from kestrel_sovereign.storage.db import postgres
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review30_pg import lose_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.postgres_schema import with_search_path


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["admission", "late-peer"])
async def test_public_preflight_joins_original_probes_before_retirement(native_pg, monkeypatch, mode):
    backend, observer, schema = native_pg
    backend_type = postgres.PostgresBackend
    actual_connect, actual_close = backend_type.connect, backend_type.close
    actual_read = state._read_raw_postgres_cluster_identity
    created, accepted, refused, pending_at_close = [], [], [], []
    originals, original_graphs, original_debts, withdrawals = [], [], [], []
    allocation = RuntimeError("original second native probe admission refused")
    release, peer_ready = asyncio.Event(), asyncio.Event()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    owner = None

    def construct(**kwargs):
        resource = backend_type(**kwargs)
        created.append(resource)
        return resource

    async def connect(resource):
        if mode == "admission" and resource in created:
            await release.wait()
        await actual_connect(resource)

    async def close(resource):
        pending_at_close.extend(task for task in accepted if not task.done())
        await actual_close(resource)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if mode == "admission" and getattr(operation, "cr_code", None) is connect.__code__:
            if accepted:
                refused.append((coroutine, operation))
                raise allocation
            task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def read(resource, *, label):
        if label == "primary":
            await peer_ready.wait()
            return await resource.fetch_all("SELECT * FROM absent_original_probe_relation")
        identity = await actual_read(resource, label=label)
        peer_ready.set()
        try:
            await release.wait()
        except asyncio.CancelledError as cancellation:
            withdrawals.append(cancellation)
            with monkeypatch.context() as ack:
                entered, commit_release = asyncio.Event(), asyncio.Event()
                commit_release.set()
                lose_ack(ack, cancelled=False, entered=entered, release=commit_release)
                try:
                    with bind_execution_custody(GenerationFence()):
                        await backend.execute("INSERT INTO effects VALUES (221, 'original peer probe cleanup')")
                except BaseException as error:
                    originals.append(error)
                    original_graphs.append(tuple(_explicit_control_chain(error)))
                    original_debts.append(tuple(execution_cleanup_debts(error)))
                    raise cancellation from error
        return identity

    try:
        monkeypatch.setattr(postgres, "PostgresBackend", construct)
        monkeypatch.setattr(backend_type, "connect", connect)
        monkeypatch.setattr(backend_type, "close", close)
        if mode == "late-peer":
            monkeypatch.setattr(state, "_read_raw_postgres_cluster_identity", read)
        loop.set_task_factory(factory)
        dsn = with_search_path(os.environ["TEST_POSTGRES_URL"], schema)
        owner = create_owned_task(state.preflight_postgres_hold_custody(dsn, dsn), retain_originals=True)
        outcome = await await_owned_task(owner)
        assert not pending_at_close, "accepted native probe escaped into backend retirement"
        if mode == "admission":
            assert allocation in tuple(_explicit_control_chain(outcome.error))
            assert accepted and all(task.done() for task in accepted)
            assert refused and all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        else:
            chain = tuple(_explicit_control_chain(outcome.error))
            assert len(originals) == 1 and originals[0] in chain, "native peer UNKNOWN disappeared during probe cleanup"
            assert len(withdrawals) == 1 and withdrawals[0] in chain
            assert execution_commit_outcome(outcome.error) == "unknown"
            assert tuple(_explicit_control_chain(originals[0])) == original_graphs[0]
            assert tuple(execution_cleanup_debts(originals[0])) == original_debts[0]
            assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=221') == 1
        assert not any(resource.is_connected for resource in created)
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        for resource in created:
            await actual_close(resource)
