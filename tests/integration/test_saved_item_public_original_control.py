"""Actual native public stash/excerpt/recall consumers, no model/IPFS calls."""

from datetime import datetime, timezone

import asyncpg
import pytest

from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_saved_items_original_control import (
    native_recall as native_recall,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door",
    [
        "stash",
        "excerpt",
        "facts",
        "list",
        "get",
        "delete-read",
        "delete-commit",
        "legacy-read",
    ],
)
async def test_public_saved_item_consumers_keep_original_native_control(
    native_recall, native_pg, monkeypatch, door
):
    backend, storage, memory, agent, feature, store = native_recall
    feature.context_manager = ContextManager(
        storage=storage, agent_id=agent.did, llm_service=agent.llm_service
    )
    if door == "delete-commit":
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        await storage.db.execute(
            "INSERT INTO saved_items (id,agent_id,item_type,name,content,summary,content_hash,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "local-delete-item",
                agent.did,
                "excerpt",
                "local fixture",
                "local content",
                "local summary",
                "local-hash",
                now,
                now,
            ),
        )
    target = (
        "FROM conversation_history"
        if door in {"stash", "excerpt"}
        else "FROM graph_nodes"
        if door == "facts"
        else "DELETE FROM saved_items"
        if door == "delete-commit"
        else "FROM saved_items"
    )
    observed, faulted = [], []
    for name in ("fetch", "fetchrow", "fetchval", "execute"):
        actual = getattr(asyncpg.Connection, name)

        def forward(actual):
            async def execute(connection, sql, *args, **kwargs):
                result = await actual(connection, sql, *args, **kwargs)
                if target in sql:
                    observed.append(sql)
                return result

            return execute

        monkeypatch.setattr(asyncpg.Connection, name, forward(actual))
    original = OSError("actual native public saved-item release ACK lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if observed and not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    result = failure = None
    try:
        if door == "stash":
            result = await feature.save_stash(
                stash_id="local-stash", name="local fixture"
            )
        elif door == "excerpt":
            result = await feature.save_excerpt("last_1", "local fixture")
        elif door == "facts":
            result = await feature.recall("local fixture")
        elif door == "list":
            result = await feature.recall_list()
        elif door == "get":
            result = await feature.recall_get("local-absent")
        elif door == "legacy-read":
            result = await store._legacy_in_python_search(
                "local fixture", [1.0, 0.0, 0.0], None, 5
            )
        else:
            result = await feature.recall_delete("local-delete-item")
    except BaseException as error:
        failure = error
    assert observed and faulted, f"did not reach actual native {door} query/release"
    if door == "delete-commit":
        # Independent original fixture observer; no renewed backend admission.
        assert (
            await native_pg[1].fetchval(
                f'SELECT COUNT(*) FROM "{native_pg[2]}".saved_items WHERE id=$1',
                "local-delete-item",
            )
            == 0
        )
    assert failure is not None and original in tuple(
        _explicit_control_chain(failure)
    ), f"native {door} control became {result!r}, failure={failure!r}"
    assert is_execution_control_error(failure)
