"""Original PG receipt survives admission refusal of exact-token cleanup.

The reused delivery fixture owns real local PG rows and dispatcher cleanup.
Its routing doubles are not exercised: this probes the actual terminalizer,
not Agent genesis, provider invocation or a fabricated native disposition.
"""

import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review18_pg import join_owned, setup_delivery
from tests.unit.test_cleanup_task_admission import _operation_name
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["direct", "cancel"])
@pytest.mark.parametrize("active_unwind", [False, True])
@pytest.mark.parametrize("refusal_rounds", [1, 4])
async def test_native_cognition_terminal_admission_keeps_original_receipt_and_delivery(
    native_pg, carrier, active_unwind, refusal_rounds,
):
    backend, observer, _ = native_pg
    agent, dispatcher, delivery, _ = await setup_delivery(native_pg)
    async with backend.advisory_locks([(3569, 18017)]) as lease:
        lease.require_live()
    source = None
    entered = asyncio.Event()

    async def original_source():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as withdrawal:
            try:
                lease.require_live()
            except BaseException as native_error:
                withdrawal.__cause__ = native_error
                raise withdrawal

    if carrier == "cancel":
        source = create_owned_task(original_source(), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        source.cancel("actual original cognition observer withdrawal")
        original = (await await_owned_task(source)).error
    else:
        try:
            lease.require_live()
        except BaseException as native_error:
            original = native_error
    assert is_execution_control_error(original)
    original_graph = tuple(_explicit_control_chain(original))
    scope = ExecutionCustody(GenerationFence())
    agent._execution_custody = backend._execution_custody = scope
    scope.revoke("actual original dispatcher runtime retired")
    loop, factory = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused, refusals = [], []
    sibling = await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'")

    def allocate(loop, coroutine, **kwargs):
        if _operation_name(coroutine).endswith("._apply_cognition_control_terminal"):
            refused.append(coroutine)
            refusal = RuntimeError("actual original terminalizer Task allocation refused")
            refusals.append(refusal)
            raise refusal
        return factory(loop, coroutine, **kwargs) if factory else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def terminalize():
        if not active_unwind:
            return await dispatcher._terminalize_failed_cognition(delivery, original)
        try:
            raise original
        except BaseException as current:
            assert current is original
            return await dispatcher._terminalize_failed_cognition(delivery, current)

    try:
        loop.set_task_factory(allocate)
        with pytest.raises(BaseException) as caught:
            await terminalize()
        chain = tuple(_explicit_control_chain(caught.value))
        assert original in chain and refusals[-1] in chain, "terminalizer admission replaced original native receipt"
        assert tuple(_explicit_control_chain(original)) == original_graph, "native cognition source receipt graph was mutated"
        assert refused and all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused)
        assert any(resource is delivery and task is None for resource, task in execution_cleanup_debts(caught.value))
        held_delivery, held_error = dispatcher._retained_cognition_control_debt[delivery.delivery_id]
        assert held_delivery is delivery
        for snapshot in OriginalFailure.capture_sources(caught.value):
            snapshot.assert_retained(held_error)
        held_graph = tuple(_explicit_control_chain(held_error))
        for retry_index in range(refusal_rounds - 1):
            with pytest.raises(BaseException) as repeated:
                if retry_index % 2:
                    await dispatcher._terminalize_failed_cognition(delivery, original)
                else:
                    await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
            repeated_chain = tuple(_explicit_control_chain(repeated.value))
            assert original in repeated_chain and refusals[-1] in repeated_chain
            current_delivery, current_error = dispatcher._retained_cognition_control_debt[delivery.delivery_id]
            assert current_delivery is delivery and current_error is held_error, (
                "repeated refusal replaced the bounded original cleanup latch"
            )
            assert tuple(_explicit_control_chain(current_error)) == held_graph
        loop.set_task_factory(factory)
        assert len(refusals) == refusal_rounds
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused)
        row = await observer.fetchrow("SELECT status,lease_owner,lease_token FROM durable_signal_deliveries WHERE delivery_id='delivery1'")
        assert tuple(row) == ("leased", "dispatcher:original", delivery.lease_token)
        assert await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'") == sibling
        # A refused attempt never acquired a metadata Task. A later exact-token
        # cleanup may settle that same snapshot, without reviving ordinary work.
        await dispatcher._terminalize_failed_cognition(delivery, held_error)
        assert not dispatcher._retained_cognition_control_debt
        assert await observer.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "failed"
        with pytest.raises(BaseException) as denied:
            await backend.execute("INSERT INTO effects VALUES (180, 'forbidden fresh work')")
        assert is_execution_control_error(denied.value)
        assert await observer.fetchval("SELECT count(*) FROM effects") == 0
    finally:
        loop.set_task_factory(factory)
        for coroutine in refused:
            coroutine.close()
        await dispatcher._terminalize_failed_cognition(delivery, original)
        await join_owned(agent, dispatcher)
        if source is not None:
            await await_owned_task(source)
