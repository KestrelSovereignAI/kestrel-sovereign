"""The await chain names the frame a suspended task is parked in (#4001).

A turn parks behind task boundaries and inside async context managers, where
``Task.get_stack()`` shows one outer frame. These pin that the walk reaches
the innermost frame through both.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from kestrel_sovereign import await_chain
from kestrel_sovereign.await_chain import format_await_chain, parked_frames

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def policy_like_span(gate: asyncio.Future):
    await wait_for_the_gate(gate)
    yield


async def wait_for_the_gate(gate: asyncio.Future):
    await gate


async def turn_body(gate: asyncio.Future):
    async with policy_like_span(gate):
        pass


async def outer_turn(gate: asyncio.Future):
    child = asyncio.create_task(turn_body(gate), name="invocation-turn:child")
    await child


async def _started(task: asyncio.Task) -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_the_chain_follows_a_child_task_into_an_async_context_manager():
    gate = asyncio.get_running_loop().create_future()
    outer = asyncio.create_task(outer_turn(gate), name="turn-outer")
    await _started(outer)
    try:
        chain = format_await_chain(outer)
        names = [frame.name for frame in parked_frames(outer)]
    finally:
        gate.set_result(None)
        await outer

    assert "task 'turn-outer':" in chain
    assert "task 'invocation-turn:child':" in chain
    for frame in ("outer_turn", "turn_body", "policy_like_span", "wait_for_the_gate"):
        assert frame in chain, chain
    assert names == ["wait_for_the_gate"]
    assert "awaiting <Future pending" in chain


async def test_the_chain_follows_every_task_behind_a_gather():
    first = asyncio.get_running_loop().create_future()
    second = asyncio.get_running_loop().create_future()

    async def gathered():
        await asyncio.gather(
            asyncio.create_task(wait_for_the_gate(first), name="branch-one"),
            asyncio.create_task(turn_body(second), name="branch-two"),
        )

    task = asyncio.create_task(gathered(), name="gathering")
    await _started(task)
    try:
        chain = format_await_chain(task)
    finally:
        first.set_result(None)
        second.set_result(None)
        await task

    assert "task 'branch-one':" in chain and "task 'branch-two':" in chain
    assert "policy_like_span" in chain


async def test_a_finished_task_says_so():
    task = asyncio.create_task(asyncio.sleep(0), name="done-already")
    await task
    assert format_await_chain(task) == "task 'done-already' has finished"


async def test_a_long_chain_is_cut_and_says_how_much(monkeypatch):
    monkeypatch.setattr(await_chain, "MAX_AWAIT_CHAIN_LINES", 3)
    gate = asyncio.get_running_loop().create_future()
    outer = asyncio.create_task(outer_turn(gate), name="turn-outer")
    await _started(outer)
    try:
        lines = format_await_chain(outer).splitlines()
    finally:
        gate.set_result(None)
        await outer

    assert len(lines) == 4
    assert lines[-1].strip().startswith("...") and "omitted" in lines[-1]


async def test_the_chain_follows_a_task_joined_through_asyncio_wait():
    gate = asyncio.get_running_loop().create_future()

    async def joining():
        child = asyncio.create_task(turn_body(gate), name="joined-by-wait")
        await asyncio.wait({child})

    task = asyncio.create_task(joining(), name="waiting-turn")
    await _started(task)
    try:
        chain = format_await_chain(task)
        names = [frame.name for frame in parked_frames(task)]
    finally:
        gate.set_result(None)
        await task

    assert "task 'joined-by-wait':" in chain, chain
    assert names == ["wait_for_the_gate"]


async def test_the_chain_follows_a_task_joined_by_await_owned_task():
    """How the effect-journal retirement joins the Tasks its turn created."""
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task

    gate = asyncio.get_running_loop().create_future()

    async def retiring():
        child = create_owned_task(turn_body(gate), name="admitted-dispatch")
        await await_owned_task(child)

    task = asyncio.create_task(retiring(), name="journal-retirement")
    await _started(task)
    try:
        chain = format_await_chain(task)
    finally:
        gate.set_result(None)
        await task

    assert "task 'admitted-dispatch':" in chain, chain
    assert "wait_for_the_gate" in chain


class _RaisingGen:
    @property
    def gen(self):
        raise RuntimeError("a diagnostic must not run this")


async def test_a_frame_whose_self_has_a_raising_gen_property_is_walked_safely():
    gate = asyncio.get_running_loop().create_future()

    class Holder(_RaisingGen):
        async def park(self):
            # Awaits a frameless future, where the walk looks for a driven
            # async generator among this frame's locals.
            await gate

    task = asyncio.create_task(Holder().park(), name="odd-self")
    await _started(task)
    try:
        chain = format_await_chain(task)
    finally:
        gate.set_result(None)
        await task

    assert "Holder.park" in chain and "awaiting <Future pending>" in chain


async def test_the_depth_limit_says_what_it_did_not_follow(monkeypatch):
    monkeypatch.setattr(await_chain, "MAX_AWAITED_TASK_DEPTH", 0)
    gate = asyncio.get_running_loop().create_future()
    outer = asyncio.create_task(outer_turn(gate), name="turn-outer")
    await _started(outer)
    try:
        chain = format_await_chain(outer)
    finally:
        gate.set_result(None)
        await outer

    assert "invocation-turn:child" not in chain
    assert "1 awaited task(s) beyond depth 0 not followed" in chain


class _ExplodingClass:
    """A lazy proxy: reading its ``__class__`` runs code, and here it raises."""

    @property
    def __class__(self):
        raise RuntimeError("a diagnostic must not run this")


async def _ticking(gate: asyncio.Future):
    await wait_for_the_gate(gate)
    yield


async def test_the_walk_reads_no_attribute_of_a_frame_local_it_does_not_own():
    """A walk inside an ``async for`` looks for the driven generator among the
    frame's locals; it must not run code on the other locals to do so."""
    gate = asyncio.get_running_loop().create_future()

    async def iterate():
        proxy = _ExplodingClass()
        generator = _ticking(gate)
        async for _ in generator:
            pass
        return proxy

    task = asyncio.create_task(iterate(), name="iterating")
    await _started(task)
    try:
        chain = format_await_chain(task)
    finally:
        gate.set_result(None)
        await task

    assert "_ticking" in chain and "wait_for_the_gate" in chain, chain
