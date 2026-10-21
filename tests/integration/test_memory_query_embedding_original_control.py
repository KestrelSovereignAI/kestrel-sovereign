"""Actual localhost embedding SDK plus native PostgreSQL accounting control."""

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    is_execution_control_error,
)
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door", ["retriever", "saved-items", "consolidator", "saved-write"]
)
async def test_actual_query_embedding_native_accounting_control(
    native_memory_owners, monkeypatch, door
):
    owners = native_memory_owners
    backend, observer, schema = owners.backend, owners.observer, owners.schema
    memory, service, requests = owners.pg_memory, owners.service, owners.requests
    saved, consolidator = owners.pg_saved, owners.sqlite_consolidator
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = service._execution_custody = scope
    original = OSError("actual embedding accounting release ACK lost")
    wrote, faulted = [], []
    actual_execute = asyncpg.Connection.execute

    async def observe_write(connection, sql, *args, **kwargs):
        result = await actual_execute(connection, sql, *args, **kwargs)
        if "INSERT INTO model_usage" in sql:
            wrote.append(sql)
        return result

    monkeypatch.setattr(asyncpg.Connection, "execute", observe_write)
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if wrote and not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    result = failure = None
    try:
        if door == "retriever":
            result = await memory.retriever._embed_query("local fixture")
        elif door == "saved-items":
            result = await saved.search("local fixture")
        elif door == "saved-write":
            result = await saved.save_item(
                item_type="excerpt",
                name="local fixture",
                content="local fixture",
                pin_to_ipfs=False,
            )
        else:
            result = await consolidator._knn_episode_ids("local fixture", 5)
    except BaseException as error:
        failure = error
    assert requests and wrote and faulted, (
        "did not reach actual SDK embedding + native usage COMMIT"
    )
    assert (
        await observer.fetchval(
            f'SELECT total_tokens FROM "{schema}".model_usage WHERE model_id=$1',
            "local-test-embedding",
        )
        == 3
    )
    assert failure is not None and original in tuple(
        _explicit_control_chain(failure)
    ), (
        f"actual embedding accounting control became fallback {result!r}, failure={failure!r}"
    )
    assert is_execution_control_error(failure)
