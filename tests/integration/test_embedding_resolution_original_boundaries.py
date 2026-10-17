"""Original native denial across embedding availability/validation boundaries.

Real PG corpus/settings SQL and the actual local-SDK LLM service execute first.
The retired physical PG lease supplies an original denial at explicitly named
consumer callbacks. This is boundary propagation proof, not an assertion that
each synchronous callback performs PG I/O, nor full Agent/HTTP/SDK ownership.
"""

import builtins
import inspect
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException, Request

from kestrel_sovereign import cli_embeddings
from kestrel_sovereign.endpoints import models
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.llm import service as service_module
from kestrel_sovereign.llm.service import EmbeddingSpaceConflictError
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from kestrel_sovereign.storage.sqla import conversation_message
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_embedding_job_original_control import (
    _job_agent,
    _seed_job_corpus,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg

DOORS = (
    "http-models-discover",
    "http-models-shared",
    "http-models-universal",
    "http-parity-json",
    "http-reindex-json",
    "http-get-reconcile",
    "http-get-classify",
    "http-route-reconcile",
    "http-model-reconcile",
    "http-route-value",
    "http-model-value",
    "http-model-conflict",
    "http-route-service",
    "http-parity",
    "reindex-runtime-read",
    "reindex-runtime-clear",
    "reindex-runtime-pin",
    "reindex-reconcile",
    "reindex-route",
    "reindex-resolve-provider",
    "reindex-resolve-model",
    "cli-config-pin",
    "cli-config-reconcile",
    "cli-config-route",
    "cli-target-route",
    "cli-target-service",
    "cli-target-profile",
    "cli-agent-import",
    "cli-agent-construct",
)


def test_sync_target_lookup_cannot_import_or_construct_an_unowned_service(monkeypatch):
    """The old target import/constructor doors no longer exist; allocation
    remains covered on the canonical cli-agent-import/construct doors above.
    """
    actual_import = builtins.__import__
    def imported(name, *args, **kwargs):
        if name == "kestrel_sovereign.llm.service":
            pytest.fail("synchronous target lookup imported an unowned service factory")
        return actual_import(name, *args, **kwargs)
    def construct():
        pytest.fail("synchronous target lookup allocated an unowned service")
    monkeypatch.setattr(service_module, "LLMService", construct)
    monkeypatch.setattr(builtins, "__import__", imported)
    error, service, profile = cli_embeddings._resolve_target()
    assert error and "explicitly owned" in error and service is None and profile is None


@pytest.mark.asyncio
@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize("kind", ["native-control", "ordinary", "healthy"])
async def test_actual_embedding_resolution_keeps_original_denial(
    native_memory_owners, tmp_path, monkeypatch, door, kind
):
    n = native_memory_owners
    monkeypatch.setenv("KESTREL_EMBEDDING_DIM", "3")
    # Declared deployment-schema input matches the ACTUAL vector(3) columns
    # below. This does not pretend the import-time default 768 was migrated.
    monkeypatch.setattr(conversation_message, "CONVERSATION_MESSAGE_EMBEDDING_DIM", 3)
    db, service, pool = n.pg_storage.db, n.service, n.backend._pool
    original_providers = service.providers
    reached, originals, constructed = [], [], []

    @asynccontextmanager
    async def checkout(native):
        async with native as connection:
            await connection.execute(f'SET search_path TO "{n.schema}", public')
            yield connection

    n.backend._pool = PostgresPoolAdapter(pool, checkout)
    try:
        await _seed_job_corpus(n)
        route = service.providers[0]["name"]
        for key, value in (
            ("embedding_route", json.dumps(route)),
            (
                "embedding_model_overrides",
                json.dumps({route: {"model": "local-test-embedding", "dim": 3}}),
            ),
        ):
            await db.execute(
                "INSERT OR REPLACE INTO agent_metadata (agent_id,key,value) VALUES (?,?,?)",
                (n.did, key, value),
            )
        if door == "reindex-runtime-clear":
            secondary = dict(service.providers[0])
            secondary.update(
                name="openai:stale-local-fixture", route="stale-local-fixture"
            )
            secondary["capabilities"] = dict(secondary["capabilities"])
            # Same explicitly borrowed local client, not a second SDK owner.
            service.providers = [*service.providers, secondary]
            service.set_route_embedding_model(
                secondary["name"], "local-test-embedding", 3, persist=False
            )
        agent = _job_agent(n)
        monkeypatch.setattr(models, "get_agent", lambda request: agent)

        def fault():
            reached.append(door)
            original = None
            if kind == "native-control":
                try:
                    n.lease.require_live()
                except BaseException as error:
                    original = error
                    originals.append(error)
            if kind != "healthy":
                error_type = (
                    EmbeddingSpaceConflictError
                    if door == "http-model-conflict"
                    else ValueError
                    if door in {"http-route-value", "http-model-value"}
                    else OSError
                )
                if original is not None:
                    if error_type in {ValueError, EmbeddingSpaceConflictError}:
                        raise error_type("typed compatibility carrier") from original
                    raise original
                raise error_type("ordinary local callback unavailable")

        def wrap(obj, name, *, caller=None, asynchronous=False):
            actual = getattr(obj, name)

            def selected(frame):
                return caller is None or (
                    frame.f_code.co_name == caller
                    and frame.f_code.co_filename
                    in {models.__file__, cli_embeddings.__file__}
                )

            if asynchronous:

                async def observed(*args, **kwargs):
                    target = selected(inspect.currentframe().f_back)
                    result = await actual(*args, **kwargs)
                    if target:
                        fault()
                    return result
            else:

                def observed(*args, **kwargs):
                    target = selected(inspect.currentframe().f_back)
                    result = actual(*args, **kwargs)
                    if target:
                        fault()
                    return result

            monkeypatch.setattr(obj, name, observed)

        bindings = {
            "http-models-discover": (
                service,
                "discover_embedding_models",
                "get_embedding_models",
                True,
            ),
            "http-models-shared": (
                service,
                "shared_embedding_space_candidates",
                "get_embedding_models",
                True,
            ),
            "http-models-universal": (
                service,
                "universal_embedding_space_options",
                "get_embedding_models",
                True,
            ),
            "http-get-reconcile": (
                service,
                "reconcile_embedding_capabilities",
                "get_embedding_settings",
                True,
            ),
            "http-get-classify": (models, "_classify_stale_embedding_rows", None, True),
            "http-route-reconcile": (
                service,
                "reconcile_embedding_capabilities",
                "set_embedding_settings",
                True,
            ),
            "http-model-reconcile": (
                service,
                "reconcile_embedding_capabilities",
                "set_route_embedding_model",
                True,
            ),
            "http-route-value": (
                service,
                "aset_embedding_route",
                "set_embedding_settings",
                True,
            ),
            "http-model-value": (
                service,
                "aset_route_embedding_model",
                "set_route_embedding_model",
                True,
            ),
            "http-model-conflict": (
                service,
                "aset_route_embedding_model",
                "set_route_embedding_model",
                True,
            ),
            "http-route-service": (
                service,
                "get_embedding_service_for_route",
                "set_route_embedding_model",
                False,
            ),
            "http-parity": (
                service,
                "verify_embedding_space_parity",
                "verify_embedding_space",
                True,
            ),
            "reindex-runtime-read": (
                service,
                "get_route_embedding_model_overrides",
                "_resolve_reindex_target",
                False,
            ),
            "reindex-runtime-clear": (
                service,
                "set_route_embedding_model",
                "_resolve_reindex_target",
                False,
            ),
            "reindex-runtime-pin": (
                service,
                "set_route_embedding_model",
                "_resolve_reindex_target",
                False,
            ),
            "reindex-reconcile": (
                service,
                "reconcile_embedding_capabilities",
                "_resolve_reindex_target",
                True,
            ),
            "reindex-route": (
                service,
                "set_embedding_route",
                "_resolve_reindex_target",
                False,
            ),
            "reindex-resolve-provider": (
                service,
                "resolve_embedding_provider",
                "_resolve_reindex_target",
                False,
            ),
            "reindex-resolve-model": (
                service,
                "resolve_route_embedding_model",
                "_resolve_reindex_target",
                True,
            ),
            "cli-config-pin": (
                service,
                "set_route_embedding_model",
                "_apply_persisted_embedding_config",
                False,
            ),
            "cli-config-reconcile": (
                service,
                "reconcile_embedding_capabilities",
                "_apply_persisted_embedding_config",
                True,
            ),
            "cli-config-route": (
                service,
                "set_embedding_route",
                "_apply_persisted_embedding_config",
                False,
            ),
            "cli-target-route": (
                service,
                "get_embedding_route",
                "_resolve_target",
                False,
            ),
            "cli-target-service": (
                service,
                "get_embedding_service",
                "_resolve_target",
                False,
            ),
        }
        if door in bindings:
            obj, name, caller, asynchronous = bindings[door]
            wrap(obj, name, caller=caller, asynchronous=asynchronous)
        elif door == "cli-agent-import":
            actual_import = builtins.__import__

            def imported(name, globals=None, locals=None, fromlist=(), level=0):
                frame = inspect.currentframe().f_back
                result = actual_import(name, globals, locals, fromlist, level)
                if name == "kestrel_sovereign.llm.service" and frame.f_code.co_name == "resolve_agent_embedding":
                    fault()
                return result

            monkeypatch.setattr(builtins, "__import__", imported)
        elif door == "cli-agent-construct":
            actual_factory = service_module.LLMService

            def factory():
                # Explicit borrowed actual usage DB. This does not certify the
                # default production constructor's separate F63 ownership.
                created = actual_factory(
                    usage_db=db, agent_data_dir=tmp_path / "constructed"
                )
                constructed.append(created)
                fault()
                return created

            monkeypatch.setattr(service_module, "LLMService", factory)
        elif door == "cli-target-profile":
            actual_get = service.get_embedding_service
            wrapped = set()

            def getter():
                result = actual_get()
                if result is not None and id(result) not in wrapped:
                    wrapped.add(id(result))
                    wrap(result, "current_profile_id", caller="_resolve_target")
                return result

            monkeypatch.setattr(service, "get_embedding_service", getter)

        body = json.dumps(
            {
                "embedding_route": route,
                "route": route,
                "embedding_model": "local-test-embedding",
                "embedding_dim": 3,
            }
        ).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(
            {"type": "http", "method": "POST", "path": "/component"}, receive
        )
        if door in {"http-parity-json", "http-reindex-json"}:
            wrap(request, "json", asynchronous=True)
        caught = result = None
        try:
            if door.startswith("http-models-"):
                result = await models.get_embedding_models(request)
            elif door == "http-reindex-json":
                result = await models.reindex_embeddings(request)
            elif door.startswith("http-get"):
                result = await models.get_embedding_settings(request)
            elif door.startswith("http-route") and door != "http-route-service":
                result = await models.set_embedding_settings(request)
            elif door.startswith("http-model") or door == "http-route-service":
                result = await models.set_route_embedding_model(request)
            elif door in {"http-parity", "http-parity-json"}:
                result = await models.verify_embedding_space(request)
            elif door.startswith("reindex-"):
                result = await models._resolve_reindex_target(agent, db, n.did)
            elif door.startswith("cli-config"):
                result = await cli_embeddings._apply_persisted_embedding_config(
                    service, db, agent_id=n.did
                )
            elif door.startswith("cli-agent-"):
                result = await cli_embeddings.resolve_agent_embedding(db, n.did)
            else:
                result = cli_embeddings._resolve_target(service)
        except BaseException as error:
            caught = error
        assert reached, f"declared boundary not reached: {door} {caught!r} {result!r}"
        if kind == "native-control":
            assert (
                originals
                and caught is not None
                and originals[0] in tuple(_explicit_control_chain(caught))
            ), (
                f"original native denial became availability/validation: {door} {caught!r} {result!r}"
            )
            assert is_execution_control_error(caught)
        else:
            assert not is_execution_control_error(caught)
            if kind == "healthy":
                assert caught is None, (
                    f"healthy real operation failed: {door} {caught!r}"
                )
            elif caught is not None:
                assert isinstance(caught, HTTPException), (
                    f"ordinary compatibility changed: {door} {caught!r}"
                )
            if kind == "ordinary" and door in {
                "http-route-value",
                "http-model-value",
                "http-model-conflict",
            }:
                assert isinstance(caught, HTTPException)
                assert caught.status_code == (
                    409 if door == "http-model-conflict" else 400
                )
        assert all("local-test-embedding" == item.get("model") for item in n.requests)
    finally:
        service.providers = original_providers
        for created in constructed:
            await created.close()
            await created.wait_for_close_completion()
        n.backend._pool = pool
