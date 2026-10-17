"""Healthy scoped memory must select original native reads before forbidden SQLA."""

import pytest

from kestrel_sovereign.execution_custody import ExecutionCustody
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["retriever", "saved-items"])
async def test_healthy_scoped_native_memory_read(
    native_memory_owners, monkeypatch, door
):
    from kestrel_sovereign.storage import sqla

    n = native_memory_owners
    scope = ExecutionCustody(GenerationFence())
    n.backend._execution_custody = n.service._execution_custody = scope
    await n.pg_storage.db.execute(
        "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?)",
        (n.did, "assistant", "Trusted retained fixture memory", "{}"),
    )
    actual, attempted = sqla.make_session_factory, []

    def observed_factory(*args, **kwargs):
        attempted.append(True)
        return actual(*args, **kwargs)

    monkeypatch.setattr(sqla, "make_session_factory", observed_factory)
    failure = result = None
    try:
        result = (
            await n.pg_memory.retriever.retrieve(
                "fixture", n.did, read_only=True, min_score=0, min_relevance=0
            )
            if door == "retriever"
            else await n.pg_saved.search("local fixture")
        )
    except BaseException as error:
        failure = error
    assert n.requests, "actual localhost embedding provider was not invoked"
    assert await n.backend.fetch_val("SELECT generation FROM authority WHERE id=1") == 1
    assert failure is None and isinstance(result, list), (
        f"healthy original native reads failed: {failure!r}"
    )
    if door == "retriever":
        assert result
    else:
        assert result == []
