"""Unpublished provider joins preserve each actual original observation.

Actual LLMService, Agent metadata callbacks, PG writes, native Tasks, SDK close
and retained provider owners execute. Native control is an actual retired PG
lease denial after SQL at a declared persistence callback, not a fabricated
COMMIT ACK or full Agent startup/READY.
"""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.llm import lifecycle
from kestrel_sovereign.llm import service as service_module
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["outer", "inner", "upgrade"])
@pytest.mark.parametrize("carrier", ["native-control", "native-cancel"])
async def test_unpublished_provider_preserves_actual_outer_and_completion_originals(
    native_memory_owners, monkeypatch, door, carrier
):
    n = native_memory_owners
    service = n.service

    class AgentInput(ModelPreferenceMixin):
        pass

    agent = AgentInput()
    agent.agent_id, agent.storage, agent._raw_storage, agent.llm_service = (
        n.did,
        n.pg_storage,
        n.pg_storage,
        service,
    )
    second_started, release, wait_entered = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    source_errors, writers, withdrawals = [], [], []
    calls = 0
    owner = lifecycle.UnpublishedLLMOwner(service, ())
    actual_service_wait, actual_lifecycle_wait = (
        service_module.await_owned_task,
        lifecycle.await_owned_task,
    )
    actual_completion = type(service).wait_for_close_completion

    async def persist(route, *args, **kwargs):
        nonlocal calls
        calls += 1
        index = calls
        writers.append(asyncio.current_task())
        await agent._persist_embedding_route(route, *args, **kwargs)
        if index == 2:
            second_started.set()
            await release.wait()
        try:
            n.lease.require_live()
        except BaseException as original:
            if carrier == "native-cancel":
                cancellation = asyncio.CancelledError(
                    "actual native persistence callback withdrawal"
                )
                cancellation.__cause__ = original
                source_errors.append(cancellation)
                raise cancellation
            source_errors.append(original)
            raise

    async def observed_service_wait(task, *args, **kwargs):
        outcome = await actual_service_wait(task, *args, **kwargs)
        if (
            door == "inner"
            and asyncio.current_task() is owner.task
            and outcome.cancellation is not None
        ):
            withdrawals.append(outcome.cancellation)
        return outcome

    async def observed_lifecycle_wait(task, *args, **kwargs):
        outcome = await actual_lifecycle_wait(task, *args, **kwargs)
        if door != "inner" and task is owner.task and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome

    async def observed_completion(selected):
        if selected is service:
            wait_entered.set()
        return await actual_completion(selected)

    service.set_embedding_route_persistence_callback(persist)
    monkeypatch.setattr(service_module, "await_owned_task", observed_service_wait)
    monkeypatch.setattr(lifecycle, "await_owned_task", observed_lifecycle_wait)
    monkeypatch.setattr(type(service), "wait_for_close_completion", observed_completion)
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.01)
    operation = None
    try:
        service.set_embedding_route(None)
        (first,) = tuple(service._preference_persistence_tasks)
        first_outcome = await await_owned_task(first)
        assert first_outcome.error is source_errors[0] and is_execution_control_error(
            first_outcome.error
        )
        assert service._preference_persistence_control is source_errors[0]
        service.set_embedding_route(service.providers[0]["name"])
        await asyncio.wait_for(second_started.wait(), 5)
        assert len(writers) == 2 and not writers[-1].done()
        if door == "upgrade":
            with pytest.raises(BaseException) as initial:
                await owner.retire()
            assert source_errors[0] in tuple(_explicit_control_chain(initial.value))

        operation = create_owned_task(
            owner.retire(join_native=True),
            name="actual-unpublished-observer",
            retain_originals=True,
        )
        await asyncio.wait_for(wait_entered.wait(), 5)
        cancelled = owner.task if door == "inner" else operation
        assert cancelled is not None and not cancelled.done()
        cancelled.cancel("actual unpublished observer withdrawn")
        await asyncio.sleep(0)
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(operation), 5)
        assert outcome.error is not None and withdrawals
        chain = tuple(_explicit_control_chain(outcome.error))
        assert all(error in chain for error in source_errors), (
            "unpublished close replaced an original native persistence carrier"
        )
        assert all(error in chain for error in withdrawals), (
            "unpublished close dropped its actual caller/completion withdrawal"
        )
        assert all(task.done() for task in writers) and calls == 2
        assert service.providers[0]["client"].is_closed()
        assert (
            await n.observer.fetchval(
                f'SELECT value FROM "{n.schema}".agent_metadata WHERE agent_id=$1 AND key=$2',
                n.did,
                agent.EMBEDDING_ROUTE_KEY,
            )
            is not None
        )
        latched = service._preference_persistence_control
        assert latched is source_errors[0]
        assert all(
            error not in tuple(_explicit_control_chain(latched))
            for error in withdrawals
        )
        for _ in range(3):
            with pytest.raises(BaseException) as later:
                await service.drain_preference_persistence()
            assert later.value is latched
        assert not service.has_pending_close_owners() and not n.requests
    finally:
        release.set()
        if operation is not None:
            await await_owned_task(operation)
        for writer in writers:
            await await_owned_task(writer)


@pytest.mark.asyncio
async def test_unpublished_physical_join_keeps_actual_plain_withdrawal(native_pg):
    """Generic close protocol over an actual accepted PG owner, not an SDK ACK."""
    backend, observer, schema = native_pg
    await backend.execute(
        "CREATE TABLE unpublished_close_effect (id INTEGER PRIMARY KEY)"
    )
    entered, release = asyncio.Event(), asyncio.Event()
    withdrawals, close_calls = [], []

    async def physical_close():
        await backend.execute("INSERT INTO unpublished_close_effect VALUES (1)")
        entered.set()
        await release.wait()

    physical = create_owned_task(physical_close(), retain_originals=True)

    class OriginalProvider:
        async def close(self):
            close_calls.append(asyncio.current_task())
            if not physical.done():
                raise RuntimeError("original close owner remains pending")

        def has_pending_close_owners(self):
            return not physical.done()

        async def wait_for_close_completion(self):
            outcome = await await_owned_task(physical)
            if outcome.cancellation is not None:
                withdrawals.append(outcome.cancellation)
                raise outcome.cancellation

    provider = OriginalProvider()
    owner = lifecycle.UnpublishedLLMOwner(provider, ())
    operation = create_owned_task(owner.retire(join_native=True), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        # The actual provider waiter is running in this original close Task.
        while owner.task is None or not close_calls:
            await asyncio.sleep(0)
        owner.task.cancel("actual plain close observer withdrawal")
        await asyncio.sleep(0)
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(operation), 5)
        assert (
            withdrawals
            and outcome.error is not None
            and withdrawals[0] in tuple(_explicit_control_chain(outcome.error))
        ), "successful physical retirement erased its actual observer withdrawal"
        assert physical.done() and len(close_calls) == 2
        assert (
            await observer.fetchval(
                f'SELECT count(*) FROM "{schema}".unpublished_close_effect'
            )
            == 1
        )
    finally:
        release.set()
        await await_owned_task(operation)
        await await_owned_task(physical)


@pytest.mark.asyncio
async def test_successful_native_upgrade_retains_bounded_original_cancellation(
    native_memory_owners, monkeypatch
):
    """Real bounded service close C is not an obsolete pending-status error."""
    n = native_memory_owners
    service = n.service
    client = service.providers[0]["client"]
    actual_close = client.close
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def close():
        calls.append(asyncio.current_task())
        entered.set()
        await release.wait()
        await actual_close()

    monkeypatch.setattr(client, "close", close)
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 1.0)
    owner = lifecycle.UnpublishedLLMOwner(service, ())
    bounded = create_owned_task(owner.retire(), retain_originals=True)
    upgraded = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        original_owner = owner.task
        original_owner.cancel("actual bounded close caller withdrawal")
        first = await asyncio.wait_for(await_owned_task(bounded), 5)
        assert isinstance(first.error, asyncio.CancelledError)
        assert first.cancellation is None and owner.control is None
        assert owner.task is original_owner and not client.is_closed()
        assert service.has_pending_close_owners()
        original = first.error
        release.set()
        upgraded = create_owned_task(
            owner.retire(join_native=True), retain_originals=True
        )
        result = await asyncio.wait_for(await_owned_task(upgraded), 5)
        assert result.error is not None and original in tuple(
            _explicit_control_chain(result.error)
        ), "successful native upgrade erased original bounded-close cancellation"
        assert client.is_closed() and len(calls) == 1 and calls[0].done()
        assert not service.has_pending_close_owners() and not n.requests
    finally:
        release.set()
        await await_owned_task(bounded)
        if upgraded is not None:
            await await_owned_task(upgraded)
