"""Actual metadata writers retain a separate original drain withdrawal.

Native PG SQL/COMMIT/RESET/release run before observer cancellation and lost
release ACK. Actual service/Agent persistence callbacks and endpoint bodies
execute; declared request/Agent fields are not full READY/authentication.
"""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import Request

from kestrel_sovereign import _async_ownership
from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin
from kestrel_sovereign.endpoints import models
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.llm import service as service_module
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
        "drain",
        "route-http",
        "model-http",
        "mandate",
        "close-completion",
        "bounded-close",
    ],
)
@pytest.mark.parametrize(
    "kind", ["native-control", "native-cancel", "ordinary", "healthy"]
)
async def test_actual_preference_drain_keeps_withdrawal_without_polluting_latch(
    native_memory_owners, monkeypatch, door, kind
):
    n = native_memory_owners

    class Owner(ModelPreferenceMixin):
        pass

    agent = Owner()
    agent.agent_id, agent.storage, agent._raw_storage = (
        n.did,
        n.pg_storage,
        n.pg_storage,
    )
    agent.llm_service = n.service
    service, db = n.service, n.pg_storage.db
    service.set_embedding_route_persistence_callback(agent._persist_embedding_route)
    service.set_route_embedding_model_persistence_callback(
        agent._persist_route_embedding_models
    )
    monkeypatch.setattr(models, "get_agent", lambda request: agent)
    key = (
        agent.EMBEDDING_MODEL_OVERRIDES_KEY
        if door == "model-http"
        else agent.EMBEDDING_ROUTE_KEY
    )
    pool, actual_execute, actual_wait = (
        n.backend._pool,
        db.execute,
        service_module.await_owned_task,
    )
    armed = False
    observer = None
    fired, sources, writers, withdrawals, entered = [], [], [], [], []
    original = OSError("actual preference release acknowledgement lost")
    actual_terminal = service_module.execution_terminal_error
    actual_harvest = service_module.harvest_llm_close_task
    harvest_withdrawn = False

    def harvest(service_owner, key, task):
        nonlocal harvest_withdrawn
        result = actual_harvest(service_owner, key, task)
        if (
            door == "bounded-close"
            and key == "preference-writers"
            and asyncio.current_task() is observer
            and not harvest_withdrawn
        ):
            harvest_withdrawn = True
            observer.cancel("actual preference observer withdrawn")
        return result

    def terminal(*errors, **kwargs):
        if asyncio.current_task() is observer:
            for error in errors:
                if (
                    isinstance(error, asyncio.CancelledError)
                    and error.args == ("actual preference observer withdrawn",)
                    and error not in withdrawals
                ):
                    withdrawals.append(error)
        return actual_terminal(*errors, **kwargs)

    @asynccontextmanager
    async def release(native):
        async with native as connection:
            yield connection
        if armed and not fired:
            fired.append(True)
            # Cancel THAT endpoint/drain observer, never the accepted writer.
            if door != "bounded-close":
                observer.cancel("actual preference observer withdrawn")
                await asyncio.sleep(0)
            if kind == "native-control":
                raise original
            if kind == "native-cancel":
                raise asyncio.CancelledError(
                    "actual release observation withdrawn"
                ) from original

    async def execute(sql, *args, **kwargs):
        nonlocal armed
        selected = (
            "INSERT OR REPLACE INTO agent_metadata" in sql
            and key in args[0]
            and not writers
        )
        if selected:
            writers.append(asyncio.current_task())
            armed = True
        try:
            result = await actual_execute(sql, *args, **kwargs)
        except BaseException as error:
            if selected:
                sources.append(error)
            raise
        if selected and kind == "ordinary":
            raise OSError("ordinary persistence callback unavailable after ACK")
        return result

    async def wait(task, *args, **kwargs):
        outcome = await actual_wait(task, *args, **kwargs)
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome  # SAME original outcome, not a fabricated ACK/Future.

    monkeypatch.setattr(db, "execute", execute)
    monkeypatch.setattr(service_module, "await_owned_task", wait)
    monkeypatch.setattr(service_module, "execution_terminal_error", terminal)
    monkeypatch.setattr(service_module, "harvest_llm_close_task", harvest)
    n.backend._pool = PostgresPoolAdapter(pool, release)
    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    payload = (
        {"route": service.providers[0]["name"], "embedding_model": None}
        if door == "model-http"
        else {"embedding_route": None}
    )

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps(payload).encode(),
            "more_body": False,
        }

    request = Request({"type": "http", "method": "POST", "path": "/component"}, receive)

    async def operation():
        nonlocal observer
        observer = asyncio.current_task()
        if door == "route-http":
            return await models.set_embedding_settings(request)
        if door == "model-http":
            return await models.set_route_embedding_model(request)
        service.set_embedding_route(None)
        if door == "close-completion":
            await service.wait_for_close_completion()
        elif door == "bounded-close":
            await service.close()
        elif door == "mandate":
            async with service.mandate_change():
                entered.append(True)
        else:
            await service.drain_preference_persistence()

    owner = None
    try:
        owner = _async_ownership.create_owned_task(
            operation(), name="actual-preference-observer", retain_originals=True
        )
        outcome = await _async_ownership.await_owned_task(owner)
        if door in {"close-completion", "bounded-close"} and not kind.startswith(
            "native"
        ):
            n.retirement_withdrawals.extend(withdrawals)
        assert (
            fired and writers and withdrawals and all(task.done() for task in writers)
        )
        assert outcome.error is not None and withdrawals[0] in tuple(
            _explicit_control_chain(outcome.error)
        ), (
            f"preference drain discarded its original caller withdrawal: {door} {kind} {outcome.error!r}"
        )
        assert not entered, "cancelled mandate observer entered a fresh change"
        assert (
            await n.observer.fetchval(
                f'SELECT value FROM "{n.schema}".agent_metadata WHERE agent_id=$1 AND key=$2',
                n.did,
                key,
            )
            is not None
        )
        latched = getattr(service, "_preference_persistence_control", None)
        if kind.startswith("native"):
            assert sources and execution_commit_outcome(sources[0]) == "committed"
            assert sources[0] in tuple(_explicit_control_chain(outcome.error))
            assert is_execution_control_error(outcome.error)
            assert latched is not None and withdrawals[0] not in tuple(
                _explicit_control_chain(latched)
            ), "finite observer cancellation polluted the long-lived writer latch"
            baseline = tuple(_explicit_control_chain(latched))
            for _ in range(3):
                try:
                    await service.drain_preference_persistence()
                except BaseException as later:
                    assert withdrawals[0] not in tuple(_explicit_control_chain(later))
                else:
                    pytest.fail(
                        "later reader erased the original native writer control"
                    )
                assert tuple(_explicit_control_chain(latched)) == baseline
        else:
            assert isinstance(outcome.error, asyncio.CancelledError)
            assert latched is None
            await service.drain_preference_persistence()
        assert not service._preference_persistence_tasks and not n.requests
    finally:
        if owner is not None:
            await _async_ownership.await_owned_task(owner)
        n.backend._pool = pool
