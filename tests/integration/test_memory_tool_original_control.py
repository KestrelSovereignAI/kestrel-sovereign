"""Actual public emotional recall consumers preserve native read receipts."""

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.features.memory.feature import MemoryFeature
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_saved_items_original_control import (
    native_recall as native_recall,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", [None, TypeError])
async def test_actual_public_emotional_recall_keeps_native_control(
    native_recall, monkeypatch, wrapper
):
    backend, storage, memory, agent, save_feature, store = native_recall
    agent.memory_system = memory
    feature = MemoryFeature(agent)
    await feature.initialize()
    actual = memory.retriever.retrieve
    if wrapper is not None:

        async def compatibility_carrier(*args, **kwargs):
            try:
                return await actual(*args, **kwargs)
            except BaseException as error:
                if is_execution_control_error(error):
                    raise wrapper("declared retrieval compatibility carrier") from error
                raise

        monkeypatch.setattr(memory.retriever, "retrieve", compatibility_carrier)
    original = OSError("actual public emotional recall read release ACK lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    observed = []

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if not observed and typ is None:
            observed.append(checkout)
            raise original
        return result

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    failure = result = None
    try:
        result = await feature.recall_emotional(query="actual retained memory")
    except BaseException as error:
        failure = error
    assert observed
    assert failure is not None and original in tuple(
        _explicit_control_chain(failure)
    ), f"actual public memory consumer made native control a ToolResult: {result!r}"
    assert is_execution_control_error(failure)
