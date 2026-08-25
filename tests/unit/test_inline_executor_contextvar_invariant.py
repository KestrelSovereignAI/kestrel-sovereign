"""Regression: the inline tool executor re-binds ``_current_signal`` (#3112).

Plus a live-context probe over the ContextVars this test publishes. This is
NOT the completeness invariant — see SCOPE below and #3114.

``OrchestratorEngineMixin._make_inline_tool_executor`` is the seam where a tool
runs on a task that is NOT the dispatching turn's task. The codex app-server
spawns its reader loop once and dispatches each ``item/tool/call`` on a
reader-spawned task, so that task carries a FROZEN pre-turn copy of the
context. Anything the turn published in a ContextVar reads as its default
inside the tool unless the executor explicitly re-presents it.

That mistake has been made four separate times in this one function:

1. ``_part_collector``            (#2081) - emitted parts silently dropped
2. transition-lock reentry token  (#2672) - durable-identity writes deadlocked
3. bound turn/session             (#2965) - restart wakes routed to nowhere
4. ``_current_signal``            (#3112) - the self_followup single-hop guard
   read ``None`` and stopped refusing, permitting an unbounded self-wake chain

SCOPE — read this before trusting a green run.

This test is NOT a completeness assertion, despite what an earlier draft of
this docstring claimed. It enforces two things:

1. A regression test for instance four specifically: ``_current_signal`` is
   re-bound, carrying the turn's value, and is NOT manufactured when the turn
   had none.
2. A live-context probe: any ContextVar that THIS TEST publishes on the turn
   and that the executor drops is caught, BY VALUE (so a re-bind carrying the
   wrong value also fails).

Method: ``contextvars.copy_context()`` enumerates every ContextVar *set* in a
context. We snapshot what the turn published, dispatch through the real
executor on a frozen-context task, snapshot again inside the tool, and require
the second to cover the first.

Why that is short of completeness: ``copy_context()`` only sees vars that are
actually SET, so the probe's coverage is exactly the set this test's setup
publishes — currently ``_current_signal`` and ``_part_collector``. Deleting
the transition-lock or turn/session bind from the executor does NOT fail this
test; that was verified by mutation. Publishing the other names here would buy
coverage of today's list and still miss tomorrow's, moving the enumeration
from the executor into this setup where it is harder to see.

Real completeness has to derive the turn-scoped set from the DECLARATION site
rather than from whatever a test happens to publish; the four live in four
different modules with no registry or marker. That is tracked in #3114, which
also records instances five and six (``_CURRENT_TURN_ID``, ``_CURRENT_CHAIN``)
as already present in the tree and NOT covered here.
"""
from __future__ import annotations

import asyncio
import contextvars

import pytest

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.parts import part_collector
from kestrel_sovereign.signals.context import (
    get_current_signal,
    reset_current_signal,
    set_current_signal,
)


class _FakeSignal:
    """Stand-in for an SDK Signal (guards only read ``source``)."""

    def __init__(self, source: str):
        self.source = source
        self.kind = source
        self.session_id = "session-under-test"

    def __eq__(self, other):  # value comparison for the invariant assert
        return getattr(other, "source", object()) == self.source

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"_FakeSignal(source={self.source!r})"


def _visible_contextvars() -> dict:
    """``ContextVar.name -> value`` for every var SET in this context.

    ``copy_context()`` excludes vars left at their default, so this is exactly
    the "published by the turn" set; an unset var never looks bound.
    """
    return {var.name: value for var, value in contextvars.copy_context().items()}


# Vars deliberately NOT carried across the boundary, with the reason. An
# exclusion list (not an inclusion list) makes "must be carried" the default
# for anything new - the safe direction.
_INTENTIONALLY_NOT_CARRIED: dict = {}


class _Agent(OrchestratorEngineMixin):
    """Host whose stubbed tool snapshots the context it actually runs in."""

    def __init__(self):
        self.seen: dict = {}

    async def execute_named_tool(self, name, args, *, session_id, source, _capture):
        _capture["effective_args"] = args
        self.seen = _visible_contextvars()
        # Record the semantic read the self_followup guard performs, so a
        # regression reads as "the guard saw None", not just a name diff.
        self.seen["__get_current_signal__"] = get_current_signal()
        return {"ok": True}


class _CodexReaderHarness:
    """Reproduce the codex reader-task topology (see #2081's regression test).

    The reader is spawned ONCE, before the turn publishes its ContextVars, so
    it freezes a pre-turn snapshot; each tool call runs on a task spawned from
    that reader - the topology in which an un-re-bound var reads its default.
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader = None

    async def ensure_started(self):
        if self._reader is None:
            self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        while True:
            item = await self._queue.get()
            if item[0] is None:
                return
            executor, name, args, done = item
            asyncio.create_task(self._handle(executor, name, args, done))

    async def _handle(self, executor, name, args, done):
        try:
            done.set_result(await executor(name, args))
        except Exception as exc:  # noqa: BLE001 - pragma: no cover, re-raised via future
            done.set_exception(exc)

    async def dispatch(self, executor, name, args):
        await self.ensure_started()
        done = asyncio.get_running_loop().create_future()
        await self._queue.put((executor, name, args, done))
        return await done

    async def stop(self):
        if self._reader is not None:
            await self._queue.put((None, None, None, None))
            await self._reader
            self._reader = None


@pytest.mark.asyncio
async def test_inline_executor_rebinds_every_turn_scoped_contextvar():
    """Every ContextVar the turn publishes is visible, with its value, in the tool."""
    agent = _Agent()
    harness = _CodexReaderHarness()

    # Reader spawned BEFORE the turn publishes: its tasks inherit pre-turn ctx.
    await harness.ensure_started()

    signal = _FakeSignal("cron.self_followup")
    token = set_current_signal(signal)
    try:
        with part_collector():
            published = _visible_contextvars()
            executor = agent._make_inline_tool_executor("session-under-test")
            _eff, result = await harness.dispatch(executor, "some_tool", {})
            assert result["ok"] is True
    finally:
        reset_current_signal(token)
        await harness.stop()

    missing = []
    stale = []
    for name, turn_value in published.items():
        if name in _INTENTIONALLY_NOT_CARRIED:
            continue
        if name not in agent.seen:
            missing.append(name)
        elif agent.seen[name] != turn_value:
            stale.append(name)

    assert not missing, (
        "The inline tool executor did not re-bind these turn-scoped "
        f"ContextVars across the reader-task boundary: {sorted(missing)}. "
        "Code inside an inline tool will read their DEFAULT value, so any "
        "guard or router keyed on them silently stops working. Add a capture "
        "on the turn task plus a bind_* in _make_inline_tool_executor (or, if "
        "the var is deliberately turn-local, record it in "
        "_INTENTIONALLY_NOT_CARRIED with the reason)."
    )
    assert not stale, (
        f"These ContextVars were re-bound with the WRONG value: {sorted(stale)}."
    )


@pytest.mark.asyncio
async def test_inline_executor_carries_current_signal_for_the_hop_guard():
    """The #3112 P1, in the terms the self_followup guard uses.

    ``get_current_signal()`` inside an inline tool must return the dispatching
    Signal - it is the only thing between a follow-up turn and scheduling
    another follow-up unbounded.
    """
    agent = _Agent()
    harness = _CodexReaderHarness()
    await harness.ensure_started()

    signal = _FakeSignal("cron.self_followup")
    token = set_current_signal(signal)
    try:
        with part_collector():
            executor = agent._make_inline_tool_executor("session-under-test")
            await harness.dispatch(executor, "schedule_add_deadline", {})
    finally:
        reset_current_signal(token)
        await harness.stop()

    seen_signal = agent.seen.get("__get_current_signal__")
    assert seen_signal is not None, (
        "get_current_signal() returned None inside an inline tool: the "
        "self_followup single-hop guard short-circuits and a follow-up can "
        "schedule a follow-up without bound (#3112 P1)."
    )
    assert getattr(seen_signal, "source", None) == "cron.self_followup"


@pytest.mark.asyncio
async def test_off_turn_executor_carries_no_signal():
    """No signal on the turn means no signal in the tool - ``None`` stays ``None``.

    The re-bind must not manufacture a Signal where there was none, or the
    guard starts refusing legitimate first-hop schedules.
    """
    agent = _Agent()
    harness = _CodexReaderHarness()
    await harness.ensure_started()

    with part_collector():
        executor = agent._make_inline_tool_executor("session-direct")
        await harness.dispatch(executor, "schedule_add_deadline", {})
    await harness.stop()

    assert agent.seen.get("__get_current_signal__") is None
