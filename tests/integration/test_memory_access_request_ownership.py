"""Overlapping native rehearsal calls keep their own accepted children.

Actual PostgreSQL writes and native Task admission/cancellation are exercised.
Captured turn-scope inputs are conformance fixtures, not new Agent authority,
authenticated request admission or READY publication.
"""

import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    bind_execution_custody_snapshot,
    is_execution_control_error,
)
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door", ["independent", "admission", "withdrawal", "lifecycle", "scope-denial"]
)
async def test_overlapping_memory_calls_do_not_borrow_peer_retirement(
    native_memory_owners, monkeypatch, door
):
    n = native_memory_owners
    rows = [
        await n.pg_storage.db.fetchval(
            "INSERT INTO conversation_history (agent_id,role,content,metadata) VALUES (?,?,?,?) RETURNING id",
            (n.did, "assistant", f"overlapping native rehearsal {i}", "{}"),
        )
        for i in range(3)
    ]
    a, b, spare = rows
    runtime = ExecutionCustody(GenerationFence())
    n.backend._execution_custody = runtime
    retriever = n.pg_memory.retriever
    actual = retriever.update_access
    reached = {a: asyncio.Event(), b: asyncio.Event()}
    release_a, release_b = asyncio.Event(), asyncio.Event()
    withdrawals = {a: [], b: []}
    accepted, refused = [], []
    original_refusal = RuntimeError(
        "actual second request memory Task admission refused"
    )
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    owners = []
    first_scope = ExecutionCustody(GenerationFence())
    denied = []

    async def update_with_gate(message_id, agent_id):
        if door == "scope-denial" and message_id == a:
            reached[a].set()
            await release_a.wait()
            try:
                return await actual(message_id, agent_id)
            except BaseException as error:
                denied.append(error)
                raise
        await actual(message_id, agent_id)
        if message_id not in reached:
            return
        reached[message_id].set()
        try:
            if message_id == a:
                await release_a.wait()
            elif door == "withdrawal":
                await release_b.wait()
        except asyncio.CancelledError as error:
            withdrawals[message_id].append(error)
            raise

    monkeypatch.setattr(retriever, "update_access", update_with_gate)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        frame = getattr(operation, "cr_frame", None)
        target = (
            getattr(operation, "cr_code", None) is update_with_gate.__code__
            and frame is not None
            and frame.f_locals.get("message_id") in {b, spare}
        )
        if target and accepted:
            refused.append((coroutine, operation))
            raise original_refusal
        task = (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )
        if target:
            accepted.append(task)
        return task

    async def request(scope, ids, *, refuse=False):
        with bind_execution_custody_snapshot((scope,)):
            if refuse:
                loop.set_task_factory(factory)
            try:
                await retriever.record_accesses(ids, n.did)
            finally:
                if refuse:
                    loop.set_task_factory(previous)

    first = create_owned_task(request(first_scope, [a]), retain_originals=True)
    owners.append(first)
    try:
        await asyncio.wait_for(reached[a].wait(), 5)
        assert await n.observer.fetchval(
            f"SELECT (metadata::jsonb->>'access_count')::int FROM \"{n.schema}\".conversation_history WHERE id=$1",
            a,
        ) == (None if door == "scope-denial" else 1)
        if door == "lifecycle":
            await retriever.shutdown()
            assert withdrawals[a]
            result = await await_owned_task(first)
            assert result.error is withdrawals[a][0], (
                "component cancellation became successful request/context publication"
            )
        else:
            second = create_owned_task(
                request(
                    ExecutionCustody(GenerationFence()),
                    [b, spare] if door == "admission" else [b],
                    refuse=door == "admission",
                ),
                retain_originals=True,
            )
            owners.append(second)
            if door == "admission":
                done, _ = await asyncio.wait([second], timeout=1)
                assert second in done, (
                    "request refusal borrowed a peer's outstanding join"
                )
                result = await await_owned_task(second)
                assert result.error is original_refusal
                assert len(accepted) == 1 and len(refused) == 1
                assert all(task.done() for task in accepted)
                assert all(
                    inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                    for pair in refused
                    for coro in pair
                )
            else:
                await asyncio.wait_for(reached[b].wait(), 5)
                assert (
                    await n.observer.fetchval(
                        f"SELECT (metadata::jsonb->>'access_count')::int FROM \"{n.schema}\".conversation_history WHERE id=$1",
                        b,
                    )
                    == 1
                )
                if door == "withdrawal":
                    second.cancel("actual second request withdrawal")
                elif door == "scope-denial":
                    first_scope.revoke(
                        "declared first original turn withdrew before its write"
                    )
                    release_a.set()
                done, _ = await asyncio.wait([second], timeout=1)
                assert second in done, (
                    "independent request waited for another request's accepted native child"
                )
                result = await await_owned_task(second)
                if door == "withdrawal":
                    finite = tuple(_explicit_control_chain(result.error))
                    assert any(
                        isinstance(error, asyncio.CancelledError)
                        and error.args == ("actual second request withdrawal",)
                        for error in finite
                    )
                    assert withdrawals[b], (
                        "own child did not receive original caller withdrawal"
                    )
                    assert withdrawals[b][0] in finite
                else:
                    assert result.error is None

            if door == "scope-denial":
                original = (await await_owned_task(first)).error
                assert denied and is_execution_control_error(original)
                assert denied[0] in tuple(_explicit_control_chain(original))
                assert retriever._access_update_control is None
            else:
                assert not withdrawals[a] and not first.done(), (
                    "request retirement cancelled or completed another request's original child"
                )
                release_a.set()
                assert (await await_owned_task(first)).error is None
        assert n.requests == [], "native request isolation must not call any provider"
        assert n.backend._execution_custody is runtime
        runtime.require_work()
    finally:
        loop.set_task_factory(previous)
        release_a.set()
        release_b.set()
        for owner in owners:
            await await_owned_task(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
