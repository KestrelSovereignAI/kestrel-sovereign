"""Actual PostgreSQL commit/deadline through manual and sleep consumers."""

import asyncio

import asyncpg
import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage import memory_system as memory_module
from kestrel_sovereign.storage.memory_system import (
    MemoryConsolidationTimeoutError,
    MemorySystem,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_storage_backend_parity import _incepted_assertion_identity
from tests.unit.test_cleanup_task_admission import _partial_native_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["memory-system", "manual-tool", "sleep"])
@pytest.mark.parametrize("boundary", ["ack-failure", "deadline"])
async def test_actual_memory_consumer_keeps_native_commit_and_timeout_control(
    native_pg, tmp_path, monkeypatch, door, boundary
):
    backend, observer, schema = native_pg
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    did, identity = await _incepted_assertion_identity(
        tmp_path, "native-maintenance-consumer"
    )
    storage = AsyncStorage(
        backend=backend,
        agent_id=did,
        _assertion_tenant_capability=_resolve_authenticated_agent_assertion_capability(
            did, identity
        ),
    )
    await storage.initialize()
    memory = MemorySystem(storage, did)
    await memory.initialize()
    agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
    agent.did = did
    agent.storage = agent._raw_storage = storage
    agent.memory_system = memory
    feature = MemoryFeature(agent)
    await feature.initialize()
    budget = 6.54321
    monkeypatch.setattr(memory_module, "_consolidation_timeout_seconds", lambda: budget)
    real_timeout = asyncio.timeout
    deadlines, source_errors, releases = [], [], []
    released, allow_ack = asyncio.Event(), asyncio.Event()
    original = OSError("actual native maintenance consumer release ACK lost")
    task = None

    def observe_actual_deadline(seconds):
        deadline = real_timeout(seconds)
        if seconds == budget and asyncio.current_task() is task:
            deadlines.append(deadline)
        return deadline

    monkeypatch.setattr(asyncio, "timeout", observe_actual_deadline)
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__

    async def original_native_release(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if deadlines and not releases and typ is None:
            releases.append(asyncio.current_task())
            released.set()
            if boundary == "ack-failure":
                raise original
            await allow_ack.wait()
        return result

    monkeypatch.setattr(
        asyncpg.pool.PoolAcquireContext, "__aexit__", original_native_release
    )

    def forward(actual):
        async def operation(*args, **kwargs):
            try:
                return await actual(*args, **kwargs)
            except BaseException as error:
                source_errors.append(error)
                raise

        return operation

    for name in ("fetch_one", "fetch_all", "fetch_val", "execute"):
        monkeypatch.setattr(backend, name, forward(getattr(backend, name)))
    backend._execution_custody = ExecutionCustody(GenerationFence())

    async def invoke():
        if door == "memory-system":
            return await memory.consolidate()
        if door == "manual-tool":
            return await feature.memory_consolidate()
        return await agent.sleep(skip_reflection=True, skip_export=True)

    task = asyncio.create_task(invoke())
    result = failure = None
    try:
        await asyncio.wait_for(released.wait(), 5)
        if boundary == "deadline":
            assert len(deadlines) == 1 and not deadlines[0].expired()
            deadlines[0].reschedule(asyncio.get_running_loop().time() + 0.01)
            await asyncio.sleep(0.03)
            assert deadlines[0].expired(), "actual consolidation clock never fired"
            assert not task.done(), "deadline abandoned original native release owner"
            allow_ack.set()
        try:
            result = await asyncio.wait_for(task, 5)
        except BaseException as error:
            failure = error
        assert releases and all(owner.done() for owner in releases)
        assert source_errors, (
            "actual native read did not report its original terminal receipt"
        )
        chain = tuple(_explicit_control_chain(failure))
        assert failure is not None and all(error in chain for error in source_errors), (
            f"native maintenance receipt became report/tool result {result!r}"
        )
        assert (
            is_execution_control_error(failure)
            and execution_commit_outcome(failure) == "committed"
        )
        if boundary == "deadline":
            assert isinstance(failure, MemoryConsolidationTimeoutError), (
                "did not reach actual typed maintenance deadline"
            )
        else:
            assert original in chain
    finally:
        allow_ack.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        backend._execution_custody = None
        await storage.close()
        await agent.llm_service.close()
        await usage_db.close()
