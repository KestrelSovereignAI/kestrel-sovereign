"""A real settings echo may schedule a second native profile writer."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import Request

from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin
from kestrel_sovereign.endpoints import models
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["native-control", "native-cancel", "ordinary", "healthy"]
)
async def test_actual_echo_scheduled_writer_finishes_before_settings_success(
    native_memory_owners, monkeypatch, kind
):
    n = native_memory_owners
    monkeypatch.setenv("KESTREL_EMBEDDING_DIM", "3")

    class Owner(ModelPreferenceMixin):
        pass

    agent = Owner()
    agent.agent_id, agent._raw_storage = n.did, n.pg_storage
    agent.storage, agent.llm_service = n.pg_storage, n.service
    monkeypatch.setattr(models, "get_agent", lambda request: agent)
    # Explicit configured capability state, not a discovered provider receipt.
    # The actual resolver re-advertises this known pin during the echo.
    n.service.providers[0]["capabilities"]["supports_embeddings"] = False
    n.service.set_embedding_route_persistence_callback(agent._persist_embedding_route)
    echo_entered, writer_done = asyncio.Event(), asyncio.Event()

    async def refresh():
        try:
            await agent.refresh_active_embedding_profile()
        finally:
            writer_done.set()

    n.service.set_embedding_profile_change_callback(refresh)
    actual_echo = n.service.aget_embedding_settings

    async def echo():
        echo_entered.set()
        return await actual_echo()

    monkeypatch.setattr(n.service, "aget_embedding_settings", echo)
    actual_counts = models._apply_stale_embedding_counts

    async def counts(settings, owner, service=None):
        # Coordinator input: serialize only this observer behind the ACTUAL
        # writer, so a concurrent read cannot receive the injected write ACK.
        await asyncio.wait_for(writer_done.wait(), timeout=10)
        return await actual_counts(settings, owner, service=service)

    monkeypatch.setattr(models, "_apply_stale_embedding_counts", counts)

    db, pool = n.pg_storage.db, n.backend._pool
    actual_execute = db.execute
    original = OSError("actual post-echo writer release acknowledgement lost")
    armed, reached, sources, fired = False, [], [], []

    @asynccontextmanager
    async def lose_ack(native):
        async with native as connection:
            yield connection
        if armed and not fired:
            fired.append(True)
            if kind == "native-cancel":
                raise asyncio.CancelledError(
                    "post-echo native release withdrawal"
                ) from original
            raise original

    async def execute(sql, *args, **kwargs):
        nonlocal armed
        selected = (
            echo_entered.is_set()
            and "INSERT OR REPLACE INTO agent_metadata" in sql
            and "active_embedding_profile" in args[0]
            and not reached
        )
        if selected:
            reached.append(sql)
            armed = kind.startswith("native")
        try:
            result = await actual_execute(sql, *args, **kwargs)
        except BaseException as error:
            if selected:
                sources.append(error)
            raise
        if selected and kind == "ordinary":
            raise OSError("ordinary post-echo callback unavailable after ACK")
        return result

    monkeypatch.setattr(db, "execute", execute)
    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    if kind.startswith("native"):
        n.backend._pool = PostgresPoolAdapter(pool, lose_ack)

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"embedding_route": null}',
            "more_body": False,
        }

    request = Request({"type": "http", "method": "POST", "path": "/component"}, receive)
    failure = result = None
    try:
        try:
            result = await models.set_embedding_settings(request)
        except BaseException as error:
            failure = error
        # Always join actual tasks independently, including on the unfixed
        # response path, without converting its discarded receipt to success.
        retained = None
        try:
            await n.service.drain_preference_persistence()
        except BaseException as error:
            retained = error
        assert echo_entered.is_set() and writer_done.is_set() and reached, (
            f"actual post-echo profile writer not reached: {failure!r} {result!r}"
        )
        raw = await n.observer.fetchval(
            f'SELECT value FROM "{n.schema}".agent_metadata WHERE agent_id=$1 AND key=$2',
            n.did,
            "active_embedding_profile",
        )
        assert (
            json.loads(raw)["profile_id"]
            == n.service.get_embedding_service().current_profile_id()
        )
        if kind.startswith("native"):
            assert (
                fired
                and sources
                and execution_commit_outcome(sources[0]) == "committed"
            )
            assert original in tuple(_explicit_control_chain(sources[0]))
            assert retained is not None and sources[0] in tuple(
                _explicit_control_chain(retained)
            )
            assert failure is not None and is_execution_control_error(failure), (
                f"settings success escaped its actual post-echo writer: {failure!r} {result!r}"
            )
            assert sources[0] in tuple(_explicit_control_chain(failure))
        else:
            assert failure is None and result["success"] is True and retained is None
        assert not n.requests and n.catalog_requests
    finally:
        n.backend._pool = pool
        n.backend._execution_custody = None
