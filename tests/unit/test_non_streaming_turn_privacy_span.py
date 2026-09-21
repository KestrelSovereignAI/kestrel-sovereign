"""#3310 — the non-streaming turn must hold the privacy-transition mutex.

`KestrelAgent.process_input` is the entry point `SignalDispatcher` calls for
EVERY COGNITION wake (`signals/dispatcher.py` → `self._agent.process_input(...)`).
Before this fix it acquired CONVERSATION via `_turn_lifecycle()` and nothing
else, while `process_input_streaming` additionally held
`_privacy_transition_lock` for its whole body.

The consequence was general rather than feature-specific: on the non-streaming
path there was **no span a privacy-mode check could run inside and be
authoritative**. A check placed anywhere — at schedule creation, at fire time,
or at the last synchronous instant before handoff — is a read racing a
write-side transition, because every remaining `await` between the check and
the turn consuming its prompt is a window the transition can land in. Each
successive fix moved the check one door further along and the defect survived.

These tests pin the span itself:

* the whole non-streaming body runs under the mutex,
* a writer that serializes only on that mutex (the shape `bootstrap/service.py`
  and `storage/memory_consolidator.py` use via `optional_transition_lock`)
  cannot land between an in-turn privacy check and the turn consuming its
  prompt, and
* a streamed turn, a signal-driven non-streaming turn, and a privacy
  transition run concurrently without deadlocking.

The ordering is not optional. `docs/architecture/SIGNAL_DISPATCHER.md` pins
CONVERSATION as the highest-order acquisition system-wide, so the pair is
always CONVERSATION → privacy. Taking the transition lock first — here or in a
caller such as the dispatcher or the scheduler — is the AB-BA wedge reverted in
`9da78c16`; `test_privacy_transition_lock_order.py` covers why that deadlocks.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock

import pytest

from kestrel_sdk.signals import ResourceLock

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import optional_transition_lock


def _turn_agent(did: str) -> KestrelAgent:
    """A real agent stopped just short of the machinery a turn body needs.

    Everything `process_input` / `process_input_streaming` touch BEFORE the
    turn lifecycle is stubbed, so the test exercises the real lock acquisition
    sequence (`_turn_lifecycle` → `_get_privacy_transition_lock`) rather than a
    re-implementation of it.
    """
    agent = KestrelAgent(did=did, storage_path=":memory:")
    agent.storage = object()
    agent.context_manager = object()
    agent.bootstrap_service = None
    agent._safe_mode = False
    agent._maybe_audit = AsyncMock()
    agent._maybe_refresh_user_byok_resolver = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    return agent


@pytest.mark.asyncio
async def test_non_streaming_turn_body_holds_the_privacy_transition_lock():
    """The span exists: the turn body runs with the privacy mutex held."""
    agent = _turn_agent("did:test:3310-span")
    observed: list[bool] = []

    async def traced(user_input, *args, **kwargs):
        observed.append(agent._get_privacy_transition_lock().locked())
        return "ok"

    agent._process_input_traced_locked = traced

    assert not agent._get_privacy_transition_lock().locked()
    await agent.process_input("wake")
    assert observed == [True], (
        "process_input must hold _privacy_transition_lock for its body; "
        "without it no privacy check inside the turn can be authoritative"
    )
    # Released on exit, so the next turn and any transition can proceed.
    assert not agent._get_privacy_transition_lock().locked()


@pytest.mark.asyncio
async def test_transition_lock_writer_cannot_land_inside_the_turn():
    """A privacy check inside the turn stays valid until the prompt is used.

    The writer here takes ONLY the privacy mutex — the production shape of
    `optional_transition_lock`, which `bootstrap/service.py` and
    `storage/memory_consolidator.py` use to make a check-then-write pair
    atomic against a mode flip. Before #3310 the non-streaming turn held
    nothing that writer contends on, so it could run in any of the turn's
    `await` gaps: between the turn's own privacy check and the turn consuming
    its prompt.
    """
    agent = _turn_agent("did:test:3310-writer")
    checked = asyncio.Event()
    may_consume = asyncio.Event()
    writer_landed = False
    consumed: list[str] = []

    async def traced(user_input, *args, **kwargs):
        # The in-turn privacy check the span makes authoritative.
        agent._privacy_mode  # noqa: B018 - the read is the check being pinned
        checked.set()
        # Several suspension points between the check and the prompt being
        # consumed — exactly the windows rounds 1-3 kept moving the check past.
        for _ in range(5):
            await asyncio.sleep(0)
        await may_consume.wait()
        consumed.append(user_input)
        return "ok"

    agent._process_input_traced_locked = traced

    async def transition_lock_writer() -> None:
        nonlocal writer_landed
        await checked.wait()
        async with optional_transition_lock(agent._get_privacy_transition_lock()):
            writer_landed = True

    turn = asyncio.create_task(agent.process_input("wake"))
    writer = asyncio.create_task(transition_lock_writer())
    await asyncio.wait_for(checked.wait(), timeout=1)
    for _ in range(10):
        await asyncio.sleep(0)

    assert not writer_landed, (
        "a privacy-mutex writer landed between the in-turn check and the turn "
        "consuming its prompt — the #3310 span is missing or too narrow"
    )
    assert consumed == []

    may_consume.set()
    await asyncio.wait_for(turn, timeout=1)
    await asyncio.wait_for(writer, timeout=1)
    assert consumed == ["wake"]
    assert writer_landed, "the writer must proceed once the turn releases"


@pytest.mark.asyncio
async def test_in_turn_cross_task_tool_re_enters_the_span():
    """The hazard the new span introduces, and the plumbing that answers it.

    Holding the mutex across the whole non-streaming turn means a tool that
    itself performs a privacy transition now meets a lock its OWN turn holds.
    On the turn task that is same-task reentry; on a foreign task — the
    isolated batch `_execute_tool_batch_at_stop_boundary` creates, and the
    Codex app-server's per-tool reader task — it is only safe because the
    executor captures the span's reentry token and delegates the lock
    manager's ownership. This reproduces that exact seam: without it the tool
    would wait on the lock the turn holds while the turn waits on the tool.
    """
    import contextvars

    from kestrel_sovereign.storage.privacy_wrapper import (
        bind_transition_lock_reentry,
    )

    agent = _turn_agent("did:test:3310-reentry")
    reentered = asyncio.Event()

    async def traced(user_input, *args, **kwargs):
        token = agent._capture_transition_reentry_token()
        assert token is not None, (
            "the turn must own a privacy-transition span for an inline tool "
            "to re-enter cross-task"
        )

        async def tool_on_a_foreign_task() -> None:
            with bind_transition_lock_reentry(token):
                async with agent.privacy_transition():
                    reentered.set()

        batch_context = contextvars.copy_context()
        agent._get_lock_manager().delegate_current_task_ownership(batch_context)
        await asyncio.wait_for(
            asyncio.create_task(
                tool_on_a_foreign_task(), context=batch_context
            ),
            timeout=1,
        )
        return "ok"

    agent._process_input_traced_locked = traced
    await asyncio.wait_for(agent.process_input("wake"), timeout=5)
    assert reentered.is_set()


@pytest.mark.asyncio
async def test_privacy_transition_waits_for_an_in_flight_non_streaming_turn():
    """The production transition path serializes behind the whole turn.

    CONVERSATION alone already gave this much — `privacy_transition()` takes
    CONVERSATION before the mutex for an external caller — so this test does
    not fail without the fix. It is here because the span must not weaken it:
    the invariant the privacy mutex adds is for callers that hold ONLY that
    mutex (see the `optional_transition_lock` test above).
    """
    agent = _turn_agent("did:test:3310-transition")
    in_body = asyncio.Event()
    may_finish = asyncio.Event()
    applied = asyncio.Event()

    async def apply(_mode):
        applied.set()
        return object()

    agent._set_privacy_mode_with_effects_locked = apply

    async def traced(user_input, *args, **kwargs):
        in_body.set()
        await may_finish.wait()
        return "ok"

    agent._process_input_traced_locked = traced

    turn = asyncio.create_task(agent.process_input("wake"))
    await asyncio.wait_for(in_body.wait(), timeout=1)
    transition = asyncio.create_task(
        agent.set_privacy_mode_with_effects(PrivacyMode.EPHEMERAL)
    )
    for _ in range(10):
        await asyncio.sleep(0)
    assert not applied.is_set()

    may_finish.set()
    await asyncio.wait_for(turn, timeout=1)
    await asyncio.wait_for(transition, timeout=1)
    assert applied.is_set()


@pytest.mark.asyncio
async def test_streamed_turn_signal_turn_and_transition_do_not_deadlock():
    """The three-way concurrency the new span has to survive.

    The signal-driven turn is wrapped in a MEMORY acquisition to mirror the
    dispatcher's pipeline: it acquires the COGNITION source's registered
    resources first and the turn lifecycle then takes CONVERSATION inside. So
    the live orders are MEMORY → CONVERSATION → privacy (signal turn),
    CONVERSATION → privacy (streamed turn) and CONVERSATION → privacy
    (transition) — one global order, therefore no cycle.
    """
    agent = _turn_agent("did:test:3310-deadlock")
    order: list[str] = []

    async def traced(user_input, *args, **kwargs):
        for _ in range(3):
            await asyncio.sleep(0)
        order.append("signal-turn")
        return "ok"

    async def streamed(*args, **kwargs):
        for _ in range(3):
            await asyncio.sleep(0)
        order.append("streamed-turn")
        yield "chunk"

    async def apply(_mode):
        order.append("transition")
        return object()

    agent._process_input_traced_locked = traced
    agent._process_input_streaming_traced_locked = streamed
    agent._set_privacy_mode_with_effects_locked = apply

    async def signal_turn() -> None:
        # A COGNITION source that declares MEMORY; CONVERSATION is never in a
        # source's resource set (SIGNAL_DISPATCHER.md §Concern 2).
        async with agent._get_lock_manager().acquire({ResourceLock.MEMORY}):
            await agent.process_input("wake")

    async def streamed_turn() -> None:
        async for _ in agent.process_input_streaming("hello"):
            pass

    await asyncio.wait_for(
        asyncio.gather(
            signal_turn(),
            streamed_turn(),
            agent.set_privacy_mode_with_effects(PrivacyMode.EPHEMERAL),
        ),
        timeout=5,
    )

    assert sorted(order) == ["signal-turn", "streamed-turn", "transition"]
    assert not agent._get_privacy_transition_lock().locked()
    assert not agent._get_lock_manager().is_held(ResourceLock.CONVERSATION)


def test_process_input_takes_conversation_before_the_transition_lock():
    """Source-level guard on the acquisition ORDER (the AB-BA tripwire).

    A future edit that hoists `_get_privacy_transition_lock()` above the
    lifecycle entry — or moves it into a caller such as the dispatcher or the
    scheduler — reintroduces the inversion reverted in `9da78c16`. Ordering
    cannot be asserted from a passing concurrent run (the wedge needs a precise
    interleaving to show up), so it is pinned at the source.
    """
    src = inspect.getsource(KestrelAgent.process_input)

    assert "self._turn_lifecycle()" in src
    assert "self._get_privacy_transition_lock()" in src, (
        "process_input must acquire the privacy-transition lock for its body"
    )
    assert src.index("self._get_privacy_transition_lock()") > src.index(
        "self._turn_lifecycle()"
    ), (
        "CONVERSATION (via _turn_lifecycle) must be acquired BEFORE the "
        "privacy-transition lock — see SIGNAL_DISPATCHER.md and 9da78c16"
    )

    # Everything that consumes the prompt stays inside the span.
    transition_pos = src.index("self._get_privacy_transition_lock()")
    for marker in (
        "BOOTSTRAP CHECK",
        "Handle explicit commands",
        "_process_input_traced_locked",
    ):
        assert src.index(marker) > transition_pos, (
            f"{marker!r} must run inside the privacy-transition span"
        )
