"""Actual retrieval query/counter ACK faults and declared native callback inputs."""

import struct
from contextvars import ContextVar

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.storage.conversation_created_at import canonical_created_at
from tests.integration.native_memory_original_control import actual_native_loss
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.native_memory_original_control import (
    native_vector_consumers as native_vector_consumers,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_saved_items_original_control import (
    native_recall as native_recall,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["salient", "lexical", "access"])
async def test_actual_retrieval_native_query_and_counter_control(
    native_recall, native_pg, monkeypatch, door
):
    backend, storage, memory, agent, feature, store = native_recall
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    row = await storage.db.fetchval(
        "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
        (agent.did, "user", "local fixture", "{}"),
    )
    conversations = memory.retriever.conversations
    method = {
        "salient": "get_salient_memory_candidates",
        "lexical": "get_lexical_memory_candidates",
        "access": "atomic_increment_metadata_counter",
    }[door]
    actual = getattr(conversations, method)
    active = ContextVar("actual_retrieval_boundary", default=False)

    async def original_method(*args, **kwargs):
        token = active.set(True)
        try:
            return await actual(*args, **kwargs)
        finally:
            active.reset(token)

    monkeypatch.setattr(conversations, method, original_method)
    observed, faulted = [], []
    target = (
        "UPDATE conversation_history"
        if door == "access"
        else "FROM conversation_history"
    )
    for name in ("fetch", "fetchrow", "fetchval", "execute"):
        execute = getattr(asyncpg.Connection, name)

        def forward(execute):
            async def original_sql(connection, sql, *args, **kwargs):
                result = await execute(connection, sql, *args, **kwargs)
                if active.get() and target in sql:
                    observed.append(sql)
                return result

            return original_sql

        monkeypatch.setattr(asyncpg.Connection, name, forward(execute))
    original = OSError("actual native retrieval query/counter COMMIT release ACK lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if active.get() and observed and not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    result = failure = None
    try:
        result = (
            await memory.retriever.update_access(row, agent.did)
            if door == "access"
            else await memory.retriever.retrieve(
                "local fixture", agent.did, read_only=True
            )
        )
    except BaseException as error:
        failure = error
    assert observed and faulted, "did not reach actual targeted native operation"
    if door == "access":
        assert (
            await native_pg[1].fetchval(
                f"SELECT (metadata::jsonb->>'access_count')::int FROM \"{native_pg[2]}\".conversation_history WHERE id=$1",
                row,
            )
            == 1
        )
    assert failure is not None and original in tuple(
        _explicit_control_chain(failure)
    ), f"native {door} control became fallback {result!r}, failure={failure!r}"
    assert is_execution_control_error(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["row", "legacy-row", "control-typeerror", "hydrate"])
async def test_actual_retrieval_callback_control_is_not_compatibility_fallback(
    native_vector_consumers, monkeypatch, door
):
    memory, saved, factory, service, requests, lease, did = native_vector_consumers
    db, conversations = saved.db, memory.retriever.conversations
    assert conversations.agent_id == did
    count = 1001 if door == "hydrate" else 1
    await db.execute(
        "WITH RECURSIVE seq(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM seq WHERE n<?) "
        "INSERT INTO conversation_history (id,agent_id,role,content,metadata,created_at) "
        "SELECT n,?,'user',CASE WHEN n=1 THEN 'opaque old message' ELSE 'recent noise' END,'{}', "
        "CASE WHEN n=1 THEN ? ELSE ? END FROM seq",
        (
            count,
            did,
            canonical_created_at("2020-01-01T00:00:00+00:00"),
            canonical_created_at("2026-10-16T00:00:00+00:00"),
        ),
    )
    if door == "hydrate":
        profile = service.get_embedding_service().current_profile_id()
        await db.execute(
            "UPDATE conversation_history SET embedding_vec=?,embedding_profile_id=? WHERE id=1",
            (struct.pack("<3f", 1.0, 0.0, 0.0), profile),
        )
        recent = await conversations.get_conversation_history(limit=1000)
        assert len(recent) == 1000
        assert all(str(message["id"]) != "1" for message in recent)
        assert await conversations.get_messages_by_ids([1])
    method = "get_messages_by_ids" if door == "hydrate" else "get_message_embeddings"
    actual = getattr(conversations, method)
    results, originals = [], []

    async def original_method(*args, **kwargs):
        result = await actual(*args, **kwargs)
        if door == "hydrate" and not any(
            str(message["id"]) == "1" for message in result
        ):
            # Candidate builders share this same public store method. Target
            # only the actual older vector-hit hydration, not their empty read.
            return result
        results.append(result)
        try:
            actual_native_loss(lease, originals)
        except BaseException as error:
            if door == "control-typeerror":
                # Declared compatibility-shim TypeError with a REAL original
                # native authority cause, not Python's argument-binding error.
                raise TypeError("declared metadata compatibility wrapper") from error
            raise
        return result

    if door == "legacy-row":

        async def original_legacy(ids):
            return await original_method(ids)

        monkeypatch.setattr(conversations, method, original_legacy)
    else:
        monkeypatch.setattr(conversations, method, original_method)
    result = failure = None
    try:
        result = await memory.retriever.retrieve(
            "local fixture", did, read_only=True, min_score=0, min_relevance=0
        )
    except BaseException as error:
        failure = error
    assert results and originals and requests, (
        "actual retrieval consumer/callback was not reached"
    )
    if door == "hydrate":
        assert any(str(message["id"]) == "1" for message in results[0]), (
            f"actual hydration rows were {results[0]!r}"
        )
    assert failure is not None and all(
        error in tuple(_explicit_control_chain(failure)) for error in originals
    ), f"native {door} callback control became fallback {result!r}"
    assert is_execution_control_error(failure)
