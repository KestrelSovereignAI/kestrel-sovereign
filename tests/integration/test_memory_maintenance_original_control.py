"""Actual native memory maintenance; not model or Agent READY acceptance."""

from datetime import datetime, timedelta, timezone

import asyncpg
import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.memory_system import MemorySystem
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_storage_backend_parity import _incepted_assertion_identity
from tests.unit.test_cleanup_task_admission import _partial_native_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["consolidator", "memory-system", "agent-sleep"])
async def test_native_consolidation_retains_actual_control(
    native_pg, tmp_path, monkeypatch, door
):
    backend, control, schema = native_pg
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    did, identity = await _incepted_assertion_identity(tmp_path, "native-consolidation")
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
    agent = usage_db = None
    if door == "agent-sleep":
        agent, usage_db = await _partial_native_agent(tmp_path, monkeypatch)
        agent.memory_system = memory
    original = OSError("actual native consolidation release acknowledgement lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    faulted, source_errors = [], []

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

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
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    result = failure = None
    try:
        try:
            if door == "consolidator":
                result = await memory.consolidator.run_consolidation()
            elif door == "memory-system":
                result = await memory.consolidate()
            else:
                result = await agent._consolidate_memories()
        except BaseException as error:
            failure = error
        assert faulted and source_errors, (
            "did not reach actual native PostgreSQL release"
        )
        assert failure is not None and original in tuple(
            _explicit_control_chain(failure)
        ), f"native memory control became report {result!r}"
        assert (
            is_execution_control_error(failure)
            and execution_commit_outcome(failure) == "committed"
        )
    finally:
        backend._execution_custody = None
        await storage.close()
        if agent is not None:
            await agent.llm_service.close()
            await usage_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["storage", "memory-system"])
@pytest.mark.parametrize("fault", ["none", "scan", "graph"])
async def test_native_retention_keeps_actual_control(
    native_pg, tmp_path, monkeypatch, door, fault
):
    backend, control, schema = native_pg
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    did, identity = await _incepted_assertion_identity(tmp_path, "native-retention")
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
    episode_id = "native-retention-episode"
    old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2000)
    await backend.execute(
        "INSERT INTO memory_episodes(id,agent_id,title,created_at,importance) VALUES (?,?,?,?,?)",
        (episode_id, did, "aged fixture episode", old, 0.01),
    )
    await storage.graph.add_node(
        GraphNode(episode_id, "episode", "aged fixture episode", {"agent_id": did})
    )
    from kestrel_sovereign.storage import retention

    monkeypatch.setattr(
        retention,
        "load_forgetting_config",
        lambda: {"enabled": True, "delete_threshold": 0.2, "grace_days": 1},
    )
    actual_execute = backend.execute
    graph_deleted = []

    async def forward_execute(sql, *args, **kwargs):
        result = await actual_execute(sql, *args, **kwargs)
        if "DELETE FROM graph_nodes" in sql:
            graph_deleted.append(sql)
        return result

    monkeypatch.setattr(backend, "execute", forward_execute)
    original = OSError("actual native forgetting release ACK lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    faulted = []

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if (
            not faulted
            and typ is None
            and (fault == "scan" or (fault == "graph" and graph_deleted))
        ):
            faulted.append(checkout)
            raise original
        return result

    backend._execution_custody = ExecutionCustody(GenerationFence())
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    result = failure = None
    try:
        try:
            result = (
                await storage.purge_decayed_episodes(delete_threshold=0.2, grace_days=1)
                if door == "storage"
                else await memory._forget_decayed_episodes()
            )
        except BaseException as error:
            failure = error
        if fault == "none":
            assert failure is None and result == 1, (
                f"native timestamp columns rejected retention cutoff: {failure!r}; result={result!r}"
            )
        else:
            assert faulted, (
                f"did not reach native {fault} release; failure={failure!r}; result={result!r}"
            )
            assert failure is not None and original in tuple(
                _explicit_control_chain(failure)
            ), f"native forgetting control became zero/skipped result {result!r}"
            assert is_execution_control_error(failure)
            if fault == "graph":
                assert (
                    await control.fetchval(
                        f'SELECT count(*) FROM "{schema}".graph_nodes WHERE node_id=$1',
                        episode_id,
                    )
                    == 0
                )
                assert (
                    await control.fetchval(
                        f'SELECT count(*) FROM "{schema}".memory_episodes WHERE id=$1',
                        episode_id,
                    )
                    == 1
                )
    finally:
        backend._execution_custody = None
        await storage.close()
