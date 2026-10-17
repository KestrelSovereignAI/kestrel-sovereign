"""Actual embedding preference writers and settings consumers retain native control.

Request/Agent owner projections are component inputs, not auth or READY proof.
Native SQL/COMMIT/RESET/release execute before the deliberately lost local ACK.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

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
from tests.utils.postgres_schema import pgvector_schema, quoted_search_path


@pytest.mark.asyncio
async def test_local_sdk_catalog_fixture_is_acknowledged(native_memory_owners):
    n = native_memory_owners
    catalog = await n.service.providers[0]["client"].models.list()
    assert {item.id for item in catalog.data} == {"local-test-embedding", "gpt-4o"}
    assert n.catalog_requests == ["/v1/models"]
    assert not n.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["drain", "route-model-http", "route-http"])
@pytest.mark.parametrize("kind", ["native-control", "ordinary", "healthy"])
@pytest.mark.parametrize("carrier", ["native", "value"])
async def test_actual_preference_writer_and_settings_keep_original_control(
    native_memory_owners, monkeypatch, door, kind, carrier
):
    n = native_memory_owners
    db = n.pg_storage.db
    route = n.service.providers[0]["name"]
    n.service.set_route_embedding_model(route, "local-test-embedding", 3, persist=False)

    class Owner(ModelPreferenceMixin):
        pass

    agent = Owner()
    agent.agent_id = n.did
    agent._raw_storage = agent.storage = n.pg_storage
    agent.llm_service = n.service
    n.service.set_route_embedding_model_persistence_callback(
        agent._persist_route_embedding_models
    )
    n.service.set_embedding_route_persistence_callback(agent._persist_embedding_route)
    monkeypatch.setattr(models, "get_agent", lambda request: agent)
    key = (
        agent.EMBEDDING_ROUTE_KEY
        if door == "route-http"
        else agent.EMBEDDING_MODEL_OVERRIDES_KEY
    )
    original = OSError("actual preference native release acknowledgement lost")
    actual_execute = db.execute
    pool, armed = n.backend._pool, False
    reached, sources, fired = [], [], []

    @asynccontextmanager
    async def lose_ack(native):
        async with native as connection:
            yield connection
        if armed and not fired:
            fired.append(True)
            raise original

    async def execute(sql, *args, **kwargs):
        nonlocal armed
        selected = (
            "INSERT OR REPLACE INTO agent_metadata" in sql
            and key in args[0]
            and not reached
        )
        if selected:
            reached.append(True)
            armed = kind == "native-control"
        try:
            result = await actual_execute(sql, *args, **kwargs)
        except BaseException as error:
            if selected:
                sources.append(error)
                if carrier == "value":
                    raise ValueError(
                        "declared preference compatibility carrier"
                    ) from error
            raise
        if selected and kind == "ordinary":
            raise (
                ValueError("ordinary persistence callback unavailable")
                if carrier == "value"
                else OSError("ordinary persistence callback unavailable")
            )
        return result

    monkeypatch.setattr(db, "execute", execute)
    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    if kind == "native-control":
        n.backend._pool = PostgresPoolAdapter(pool, lose_ack)

    payload = (
        {"embedding_route": None}
        if door == "route-http"
        else {"route": route, "embedding_model": None}
    )
    body = json.dumps(payload).encode()
    delivered = False

    async def receive():
        nonlocal delivered
        assert not delivered
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/component"}, receive)
    failure = result = None
    try:
        try:
            if door == "drain":
                await n.service.aset_route_embedding_model(route, None)
                result = await models._drain_preference_persistence(agent)
            elif door == "route-model-http":
                result = await models.set_route_embedding_model(request)
            else:
                result = await models.set_embedding_settings(request)
        except BaseException as error:
            failure = error
        # Independently join the ACTUAL scheduled writers even on the unfixed
        # route endpoint, which returned before they ran. No fake Task/ACK.
        writes = tuple(n.service._preference_persistence_tasks)
        if writes:
            await asyncio.gather(*writes, return_exceptions=True)
        retained = None
        try:
            await n.service.drain_preference_persistence()
        except BaseException as error:
            retained = error
        assert reached, (
            f"actual preference metadata writer not reached: {failure!r} {result!r}"
        )
        stored = await n.observer.fetchval(
            f'SELECT value FROM "{n.schema}".agent_metadata WHERE agent_id=$1 AND key=$2',
            n.did,
            key,
        )
        assert json.loads(stored) == (None if door == "route-http" else {})
        if kind == "native-control":
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
                f"actual preference writer control became settings success/HTTP replacement: "
                f"door={door} carrier={carrier} failure={failure!r} result={result!r}"
            )
            assert sources[0] in tuple(_explicit_control_chain(failure))
        else:
            assert failure is None and retained is None
            assert result is None if door == "drain" else result["success"] is True
        assert not n.requests, "clearing local preferences must not request a model"
        assert not n.service._preference_persistence_tasks
    finally:
        n.backend._pool = pool
        n.backend._execution_custody = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door",
    [
        "cli-route",
        "cli-pins",
        "reindex-route",
        "reindex-pins",
        "cli-audit-registry",
        "cli-audit-table",
        "cli-active-report",
        "cli-active-reindex",
        "settings-classify",
    ],
)
@pytest.mark.parametrize("kind", ["native-control", "ordinary", "healthy"])
async def test_actual_embedding_settings_and_cli_reads_preserve_control(
    native_memory_owners, monkeypatch, door, kind
):
    from kestrel_sovereign import cli_embeddings
    from kestrel_sovereign.storage.active_embedding_profile import (
        record_active_embedding_profile,
    )

    n = native_memory_owners
    db = n.pg_storage.db
    service = n.service.get_embedding_service()
    route = n.service.providers[0]["name"]
    monkeypatch.setenv("KESTREL_EMBEDDING_DIM", "3")
    # Actual component DDL, not whole startup migration acceptance. Install the
    # extension when absent and resolve its actual namespace; a fresh CI DB has
    # no public.vector merely because the image includes pgvector binaries.
    vector_namespace = quoted_search_path(await pgvector_schema(db))
    for table in ("saved_items", "conversation_history", "document_chunks"):
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT"
        )
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_vec {vector_namespace}.vector(3)"
        )
    await db.execute(
        "CREATE TABLE IF NOT EXISTS embedding_profiles (id TEXT PRIMARY KEY, provider TEXT, model TEXT, dim INTEGER, space_id TEXT, normalized BOOLEAN)"
    )
    await db.execute(
        "INSERT INTO saved_items (id,agent_id,item_type,name,content,content_hash) VALUES (?,?,?,?,?,?)",
        (
            "settings-read",
            n.did,
            "excerpt",
            "native",
            "Stored local context.",
            "settings-read",
        ),
    )
    await db.execute(
        "INSERT OR REPLACE INTO agent_metadata (agent_id,key,value) VALUES (?,?,?)",
        (n.did, "embedding_route", json.dumps(route)),
    )
    await db.execute(
        "INSERT OR REPLACE INTO agent_metadata (agent_id,key,value) VALUES (?,?,?)",
        (
            n.did,
            "embedding_model_overrides",
            json.dumps({route: {"model": "local-test-embedding", "dim": 3}}),
        ),
    )
    await record_active_embedding_profile(db, n.did, service)
    agent = SimpleNamespace(agent_id=n.did, storage=n.pg_storage, llm_service=n.service)
    original = OSError("actual settings read native release acknowledgement lost")
    pool, armed = n.backend._pool, False
    sources, fired, reached = [], [], []
    actual_fetch = db.fetchall

    @asynccontextmanager
    async def lose_ack(native):
        async with native as connection:
            yield connection
        if armed and not fired:
            fired.append(True)
            raise original

    async def fetch(sql, *args, **kwargs):
        nonlocal armed
        params = args[0] if args else ()
        selected = not reached and (
            door in {"cli-route", "reindex-route"}
            and "embedding_route" in params
            or door in {"cli-pins", "reindex-pins"}
            and "embedding_model_overrides" in params
            or door == "cli-audit-registry"
            and "SELECT id, provider, model, dim FROM embedding_profiles" in sql
            or door == "cli-audit-table"
            and "GROUP BY embedding_profile_id" in sql
            or door in {"cli-active-report", "cli-active-reindex"}
            and "active_embedding_profile" in params
            or door == "settings-classify"
            and "SELECT id, summary, content FROM saved_items" in sql
        )
        if selected:
            reached.append(sql)
            armed = kind == "native-control"
        try:
            result = await actual_fetch(sql, *args, **kwargs)
        except BaseException as error:
            if selected:
                sources.append(error)
            raise
        if selected and kind == "ordinary":
            raise OSError("ordinary settings read callback unavailable after ACK")
        return result

    monkeypatch.setattr(db, "fetchall", fetch)
    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    if kind == "native-control":
        n.backend._pool = PostgresPoolAdapter(pool, lose_ack)
    failure = result = None
    try:
        try:
            if door == "cli-route":
                result = await cli_embeddings._load_persisted_embedding_route(db, n.did)
            elif door == "cli-pins":
                result = await cli_embeddings._load_persisted_route_embedding_models(
                    db, n.did
                )
            elif door.startswith("reindex-"):
                result = await models._resolve_reindex_target(
                    agent, db=db, agent_id=n.did
                )
            elif door.startswith("cli-audit-"):
                result = await cli_embeddings._audit(
                    db, table="saved_items", agent_id=n.did
                )
            elif door == "cli-active-report":
                result = await cli_embeddings._active_profile_report(
                    db, ("saved_items",)
                )
            elif door == "cli-active-reindex":
                result = await cli_embeddings._reindex(
                    db,
                    table="saved_items",
                    agent_id=n.did,
                    batch=2,
                    rate_limit=0,
                    dry_run=True,
                    apply=False,
                    embedding_service=service,
                    target_profile_id=service.current_profile_id(),
                    target_dim=3,
                )
            else:
                result = await models._classify_stale_embedding_rows(agent)
        except BaseException as error:
            failure = error
        assert reached, (
            f"actual settings/CLI read was not reached: {door} {failure!r} {result!r}"
        )
        if kind == "native-control":
            assert (
                fired
                and sources
                and execution_commit_outcome(sources[0]) == "committed"
            )
            assert original in tuple(_explicit_control_chain(sources[0]))
            assert failure is not None and is_execution_control_error(failure), (
                f"actual settings/CLI original became ordinary unavailable/refusal: "
                f"door={door} failure={failure!r} result={result!r}"
            )
            assert sources[0] in tuple(_explicit_control_chain(failure))
        else:
            assert failure is None, (
                f"ordinary/healthy behavior changed: {door} {failure!r}"
            )
            if door == "cli-route":
                assert result == (
                    (False, None) if kind == "ordinary" else (True, route)
                )
            elif door == "cli-pins":
                assert result == (
                    {}
                    if kind == "ordinary"
                    else {route: {"model": "local-test-embedding", "dim": 3}}
                )
            elif door.startswith("reindex-"):
                assert result[0].current_profile_id() == service.current_profile_id()
            elif door.startswith("cli-audit-"):
                assert result == (
                    1 if door == "cli-audit-table" and kind == "ordinary" else 0
                )
            elif door == "cli-active-report":
                assert result["status"] == (
                    "unreadable" if kind == "ordinary" else "recorded"
                )
            elif door == "cli-active-reindex":
                assert result == (2 if kind == "ordinary" else 0)
            else:
                assert result == {
                    "stale": 0 if kind == "ordinary" else 1,
                    "actionable": 0 if kind == "ordinary" else 1,
                    "unembeddable": 0,
                }
        assert not n.requests, "read-only settings/CLI paths must not request a model"
    finally:
        n.backend._pool = pool
        n.backend._execution_custody = None
