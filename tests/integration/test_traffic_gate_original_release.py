"""Actual native work leaves no admission leak when gate release cannot spawn."""

import asyncio
from uuid import uuid4

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
)
from kestrel_sovereign.features.isolated_runtime import _TrafficGate
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_actual_native_gate_body_does_not_require_a_release_task(
    native_pg, monkeypatch, unknown
):
    backend, observer, schema = native_pg
    backend._execution_custody = ExecutionCustody(GenerationFence())
    gate, refused = _TrafficGate(), []
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    original = OSError("actual gated COMMIT ACK lost")
    marker = uuid4().hex

    def refuse_release(loop, coroutine, **kwargs):
        code = getattr(coroutine, "cr_code", None)
        if code is not None and code.co_qualname == "_TrafficGate._release_admission":
            refused.append(coroutine)
            raise RuntimeError("unnecessary gate release Task allocation refused")
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    caught = None
    try:
        loop.set_task_factory(refuse_release)
        if unknown:
            lose_native_ack(monkeypatch, boundary="commit", original=original)
        try:
            async with gate.admit():
                await backend.execute("INSERT INTO effects VALUES (41, ?)", (marker,))
        except BaseException as error:
            caught = error
        loop.set_task_factory(previous)
        assert not refused, (
            "in-memory admission release unnecessarily requires a fresh Task"
        )
        assert gate._active == 0, (
            "original native body leaked gate admission after Task refusal"
        )
        await asyncio.wait_for(gate.close_and_drain(), 1)
        assert (
            await observer.fetchval(f'SELECT value FROM "{schema}".effects WHERE id=41')
            == marker
        )
        if unknown:
            assert caught is not None and execution_commit_outcome(caught) == "unknown"
            assert carries(caught, original)
        else:
            assert caught is None, caught
    finally:
        loop.set_task_factory(previous)
        for coroutine in refused:
            coroutine.close()
        backend._execution_custody = None


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_original_gate_release_joins_through_repeated_caller_cancellation(
    native_pg, monkeypatch, unknown
):
    backend, observer, schema = native_pg
    backend._execution_custody = ExecutionCustody(GenerationFence())
    gate = _TrafficGate()
    body_done, leave_body = asyncio.Event(), asyncio.Event()
    original = OSError("actual gated COMMIT ACK lost before release contention")
    marker = uuid4().hex
    if unknown:
        lose_native_ack(monkeypatch, boundary="commit", original=original)

    async def native_body():
        async with gate.admit():
            error = None
            try:
                await backend.execute("INSERT INTO effects VALUES (42, ?)", (marker,))
            except BaseException as caught:
                error = caught
            body_done.set()
            await leave_body.wait()
            if error is not None:
                raise error

    caller = create_owned_task(native_body(), retain_originals=True)
    try:
        await asyncio.wait_for(body_done.wait(), 2)
        async with gate._condition:
            leave_body.set()
            await asyncio.sleep(0)
            for message in (
                "first gate release withdrawal",
                "second gate release withdrawal",
            ):
                caller.cancel(message)
                await asyncio.sleep(0)
                assert not caller.done() and gate._active == 1
        outcome = await await_owned_task(caller)
        assert gate._active == 0
        await asyncio.wait_for(gate.close_and_drain(), 1)
        assert (
            await observer.fetchval(f'SELECT value FROM "{schema}".effects WHERE id=42')
            == marker
        )
        assert outcome.error is not None
        if unknown:
            assert execution_commit_outcome(outcome.error) == "unknown"
            assert carries(outcome.error, original)
        else:
            assert isinstance(outcome.error, asyncio.CancelledError)
        # Both original withdrawals remain explicit evidence, not just the
        # native receipt or the last cancellation delivered to the caller.
        chain, pending = [], [outcome.error]
        while pending:
            error = pending.pop()
            if any(error is prior for prior in chain):
                continue
            chain.append(error)
            pending.extend(getattr(error, "exceptions", ()))
            if error.__cause__ is not None:
                pending.append(error.__cause__)
        assert {
            error.args[0]
            for error in chain
            if isinstance(error, asyncio.CancelledError)
        } >= {"first gate release withdrawal", "second gate release withdrawal"}
    finally:
        leave_body.set()
        await await_owned_task(caller)
        backend._execution_custody = None
