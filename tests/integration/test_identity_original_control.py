"""Identity HTTP consumers preserve actual PostgreSQL release receipts.

Real initialized storage and native SQL are used. The partial-agent request
projection and a partial rename outcome are explicit component inputs, not
authenticated HTTP, full Agent genesis/READY or provider acceptance.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from fastapi import HTTPException, Request, Response

from kestrel_sovereign.endpoints import models
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.bootstrap.feature import RenameOutcome
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door",
    [
        "rename-validation",
        "description",
        "partial-readback",
        "get-node",
        "get-description",
    ],
)
@pytest.mark.parametrize("kind", ["native-control", "ordinary"])
async def test_identity_consumers_do_not_normalize_original_native_release(
    native_memory_owners, monkeypatch, door, kind
):
    n = native_memory_owners

    @dataclass
    class IdentityProjection:
        agent_id: str
        storage: object
        _raw_storage: object

        def resolve_effective_name(self, node):
            return "native-component"

    agent = IdentityProjection(n.did, n.pg_storage, n.pg_storage)
    request = Request({"type": "http", "state": {"agent": agent}})
    response = Response()
    pool = n.backend._pool
    original = OSError("original identity native release acknowledgement lost")
    ordinary = OSError("ordinary identity callback failure after acknowledged SQL")
    sources, fired = [], []
    armed = door != "get-description"

    @asynccontextmanager
    async def fail_release(native):
        async with native as connection:
            yield connection
        if armed and not fired:
            fired.append(True)
            raise original

    async def observe_native(operation, *args, **kwargs):
        nonlocal armed
        if kind == "native-control":
            armed = True
        try:
            result = await operation(*args, **kwargs)
        except BaseException as error:
            sources.append(error)
            raise
        if kind == "ordinary":
            fired.append(True)
            raise ordinary
        return result

    actual_get = n.pg_storage.get_node
    actual_fetch = n.pg_storage.db.fetchone
    actual_execute = n.pg_storage.db.execute
    if door in {"partial-readback", "get-node"}:

        async def get_node(*args, **kwargs):
            return await observe_native(actual_get, *args, **kwargs)

        monkeypatch.setattr(n.pg_storage, "get_node", get_node)
    elif door == "get-description":

        async def fetch(*args, **kwargs):
            if "SELECT value FROM agent_metadata" in args[0]:
                return await observe_native(actual_fetch, *args, **kwargs)
            return await actual_fetch(*args, **kwargs)

        monkeypatch.setattr(n.pg_storage.db, "fetchone", fetch)
    elif door == "description":

        async def execute(*args, **kwargs):
            return await observe_native(actual_execute, *args, **kwargs)

        monkeypatch.setattr(n.pg_storage.db, "execute", execute)

    if door in {"rename-validation", "partial-readback"}:

        async def rename(current, name):
            nonlocal armed
            if door == "rename-validation":
                try:
                    await observe_native(
                        actual_execute,
                        "INSERT INTO effects VALUES (?,?)",
                        (1, "actual original rename callback write"),
                    )
                except BaseException as error:
                    # Explicit compatibility-validator carrier around the
                    # actual native receipt, not a fabricated control marker.
                    raise ValueError("rename validator observed failure") from error
            # The declared partial rename result follows a real metadata write.
            # Do not inject release failure until the endpoint reads it back.
            armed = False
            await actual_execute(
                "INSERT INTO agent_metadata (agent_id,key,value) VALUES (?,?,?)",
                (n.did, "name", name),
            )
            return RenameOutcome(
                False, True, False, False, False, "declared partial rename"
            )

        monkeypatch.setattr(models, "rename_agent_core", rename)

    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    if kind == "native-control":
        n.backend._pool = PostgresPoolAdapter(pool, fail_release)
    failure = payload = None
    try:
        try:
            if door.startswith("get-"):
                payload = await models.get_identity(request)
            else:
                body = models.UpdateIdentityRequest(
                    **(
                        {"description": "actual new description"}
                        if door == "description"
                        else {"name": "NativeComponent"}
                    )
                )
                payload = await models.update_identity.__wrapped__(
                    request, response, body
                )
        except BaseException as error:
            failure = error

        assert fired, f"intended actual SQL/callback boundary not reached: {failure!r}"
        if kind == "native-control":
            assert sources and is_execution_control_error(sources[0])
            assert execution_commit_outcome(sources[0]) == "committed"
            assert original in tuple(_explicit_control_chain(sources[0]))
            assert failure is not None and is_execution_control_error(failure), (
                f"original native identity control became ordinary HTTP/partial output: "
                f"door={door} failure={failure!r} payload={payload!r} status={response.status_code}"
            )
            assert sources[0] in tuple(_explicit_control_chain(failure))
            assert execution_commit_outcome(failure) == "committed"
        elif door in {"partial-readback", "get-description"}:
            assert failure is None and payload is not None
            assert response.status_code == (207 if door == "partial-readback" else 200)
            if door == "partial-readback":
                assert payload["success"] is False
        else:
            assert isinstance(failure, HTTPException)
            assert failure.status_code == (422 if door == "rename-validation" else 500)
            assert not is_execution_control_error(failure)

        if door == "description":
            row = await n.observer.fetchval(
                f"SELECT value FROM \"{n.schema}\".agent_metadata WHERE agent_id=$1 AND key='description'",
                n.did,
            )
            assert row == "actual new description"
        elif door == "partial-readback":
            row = await n.observer.fetchval(
                f"SELECT value FROM \"{n.schema}\".agent_metadata WHERE agent_id=$1 AND key='name'",
                n.did,
            )
            assert row == "NativeComponent"
        elif door == "rename-validation":
            assert (
                await n.observer.fetchval(f'SELECT count(*) FROM "{n.schema}".effects')
                == 1
            )
        assert n.requests == [], "identity proof must not make any LLM request"
    finally:
        n.backend._pool = pool
        n.backend._execution_custody = None
