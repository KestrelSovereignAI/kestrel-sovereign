"""Unit tests for the explicit, rollback-safe boot state machine (#2522).

These exercise the agent-agnostic primitives in
``kestrel_sovereign.agent.boot`` in isolation — no ``KestrelAgent`` surface —
so the sequencing/rollback contract is pinned independently of any phase body:

* phases run in declared order and the journal records commits;
* ``READY`` is set exactly once, only after every phase commits;
* a phase failure unwinds the rollback stack LIFO, sets ``FAILED``, and
  re-raises the ORIGINAL exception;
* resources a phase declares ``retained`` are never rolled back;
* rollback is best-effort — one stubborn undo can't strand the others;
* a resource leaves custody only when its undo reports ``RELEASED`` for an
  owner that reports a failed release truthfully;
* a boot cancelled mid-phase still releases every acquired resource before
  the ``CancelledError`` propagates.
"""

import asyncio

import pytest

from kestrel_sovereign.agent.boot import (
    BootContext,
    BootPhase,
    BootPhaseState,
    run_boot_sequence,
)
from kestrel_sovereign.agent.custody import ReleaseOutcome, ResourceCustody
from kestrel_sovereign.kestrel_agent import KestrelAgent


class _Recorder:
    """Collects a StateSetter's transitions for assertions."""

    def __init__(self) -> None:
        self.states: list[BootPhaseState] = []

    def __call__(self, state: BootPhaseState) -> None:
        self.states.append(state)


@pytest.mark.asyncio
async def test_successful_hosted_boot_retains_watchdog_until_manager_handoff(
    monkeypatch,
):
    """Active services cannot create an unowned TTL gap after initialize()."""

    agent = object.__new__(KestrelAgent)
    agent._boot_state = BootPhaseState.NOT_STARTED
    agent._boot_context = None
    agent._host_authority_boot_expired = False
    handle = asyncio.get_running_loop().call_later(60, lambda: None)
    agent._host_authority_boot_deadline_handle = handle

    async def successful_boot(_phases, _ctx, set_state):
        set_state(BootPhaseState.IN_PROGRESS)
        set_state(BootPhaseState.READY)

    monkeypatch.setattr(
        "kestrel_sovereign.kestrel_agent.run_boot_sequence",
        successful_boot,
    )

    await agent.initialize()

    assert handle.cancelled() is False
    assert agent._host_authority_boot_deadline_handle is handle
    agent._disarm_host_authority_boot_deadline()
    assert handle.cancelled() is True


# ---------------------------------------------------------------------------
# Happy path — order, journal, single READY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phases_run_in_declared_order():
    order: list[str] = []

    def make(name: str) -> BootPhase:
        async def body(ctx: BootContext) -> None:
            order.append(name)

        return BootPhase(name, body)

    ctx = BootContext()
    rec = _Recorder()
    await run_boot_sequence([make("a"), make("b"), make("c")], ctx, rec)

    assert order == ["a", "b", "c"]
    assert ctx.committed_phases == ["a", "b", "c"]
    assert rec.states[0] is BootPhaseState.IN_PROGRESS
    assert rec.states[-1] is BootPhaseState.READY


@pytest.mark.asyncio
async def test_ready_fires_only_after_all_phases_commit():
    seen_states: list[BootPhaseState] = []

    async def observe(ctx: BootContext) -> None:
        # While a phase body runs the machine must still be IN_PROGRESS —
        # readiness may never be observable mid-boot.
        seen_states.append(rec.states[-1])

    rec = _Recorder()
    ctx = BootContext()
    await run_boot_sequence(
        [BootPhase("p1", observe), BootPhase("p2", observe)], ctx, rec
    )

    assert seen_states == [BootPhaseState.IN_PROGRESS, BootPhaseState.IN_PROGRESS]
    # READY appears exactly once and only as the final transition.
    assert rec.states.count(BootPhaseState.READY) == 1
    assert rec.states[-1] is BootPhaseState.READY


# ---------------------------------------------------------------------------
# Failure — LIFO rollback, FAILED, original exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_unwinds_rollback_in_reverse_order():
    released: list[str] = []

    async def acquire(ctx: BootContext, label: str) -> None:
        async def undo(name: str = label) -> ReleaseOutcome:
            released.append(name)
            return ReleaseOutcome.RELEASED

        ctx.on_rollback(label, undo)

    def phase(name: str) -> BootPhase:
        async def body(ctx: BootContext) -> None:
            await acquire(ctx, name)

        return BootPhase(name, body)

    async def boom(ctx: BootContext) -> None:
        # This phase acquires nothing before failing.
        raise RuntimeError("phase-4 exploded")

    ctx = BootContext()
    rec = _Recorder()
    with pytest.raises(RuntimeError, match="phase-4 exploded"):
        await run_boot_sequence(
            [phase("r1"), phase("r2"), phase("r3"), BootPhase("boom", boom)],
            ctx,
            rec,
        )

    # LIFO: last-acquired released first.
    assert released == ["r3", "r2", "r1"]
    assert rec.states[-1] is BootPhaseState.FAILED
    # The failing phase never committed; the three before it did.
    assert ctx.committed_phases == ["r1", "r2", "r3"]


@pytest.mark.asyncio
async def test_failure_reraises_original_exception_type():
    class MyBootError(RuntimeError):
        pass

    async def boom(ctx: BootContext) -> None:
        raise MyBootError("specific")

    ctx = BootContext()
    with pytest.raises(MyBootError, match="specific"):
        await run_boot_sequence([BootPhase("boom", boom)], ctx, _Recorder())
    assert ctx.committed_phases == []


@pytest.mark.asyncio
async def test_rollback_labels_reflect_acquisition_order():
    async def body(ctx: BootContext) -> None:
        async def noop() -> ReleaseOutcome:
            return ReleaseOutcome.RELEASED

        ctx.on_rollback("first", noop)
        ctx.on_rollback("second", noop)

    ctx = BootContext()
    await run_boot_sequence([BootPhase("p", body)], ctx, _Recorder())
    # After a clean run nothing is rolled back, but the stack recorded order.
    assert ctx.rollback_labels == ["first", "second"]


# ---------------------------------------------------------------------------
# Retained resources are deliberately NOT rolled back
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retained_resources_are_not_rolled_back():
    released: list[str] = []

    async def durable_phase(ctx: BootContext) -> None:
        # A retained phase commits a durable resource. It registers NO undo
        # for the retained artifact (that is the whole point) but does open a
        # rollback-able side resource to prove the two are handled separately.
        async def undo_side() -> ReleaseOutcome:
            released.append("side")
            return ReleaseOutcome.RELEASED

        ctx.on_rollback("side", undo_side)

    async def boom(ctx: BootContext) -> None:
        raise RuntimeError("later failure")

    ctx = BootContext()
    rec = _Recorder()
    with pytest.raises(RuntimeError, match="later failure"):
        await run_boot_sequence(
            [
                BootPhase(
                    "identity",
                    durable_phase,
                    retained=("durable identity node",),
                ),
                BootPhase("boom", boom),
            ],
            ctx,
            rec,
        )

    # The retained artifact is recorded in the audit trail and never released.
    assert "durable identity node" in ctx.retained_resources
    assert "durable identity node" not in released
    # The side resource the same phase opened IS rolled back.
    assert released == ["side"]
    assert rec.states[-1] is BootPhaseState.FAILED


# ---------------------------------------------------------------------------
# Rollback is best-effort — a failing undo cannot strand the others
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollback_tolerates_a_failing_undo_step():
    released: list[str] = []

    async def acquire_ok(ctx: BootContext, label: str) -> None:
        async def undo(name: str = label) -> ReleaseOutcome:
            released.append(name)
            return ReleaseOutcome.RELEASED

        ctx.on_rollback(label, undo)

    async def acquire_bad(ctx: BootContext) -> None:
        async def undo_bad() -> None:
            raise RuntimeError("teardown blew up")

        ctx.on_rollback("bad", undo_bad)

    async def p1(ctx: BootContext) -> None:
        await acquire_ok(ctx, "bottom")

    async def p2(ctx: BootContext) -> None:
        await acquire_bad(ctx)

    async def p3(ctx: BootContext) -> None:
        await acquire_ok(ctx, "top")

    async def boom(ctx: BootContext) -> None:
        raise RuntimeError("go")

    ctx = BootContext()
    with pytest.raises(RuntimeError, match="go"):
        await run_boot_sequence(
            [
                BootPhase("p1", p1),
                BootPhase("p2", p2),
                BootPhase("p3", p3),
                BootPhase("boom", boom),
            ],
            ctx,
            _Recorder(),
        )

    # 'top' and 'bottom' both released despite 'bad' raising in between.
    assert "top" in released
    assert "bottom" in released


def _trusting(*owners: str) -> BootContext:
    """A context whose custody believes a release reported by ``owners``."""
    return BootContext(custody=ResourceCustody(truthful_owners=owners))


@pytest.mark.asyncio
async def test_run_rollback_reports_error_and_ok_steps():
    ctx = _trusting("ok", "err")

    async def undo_ok() -> ReleaseOutcome:
        return ReleaseOutcome.RELEASED

    async def undo_err() -> ReleaseOutcome:
        raise ValueError("nope")

    ctx.on_rollback("ok", undo_ok)
    ctx.on_rollback("err", undo_err)

    released = await ctx.run_rollback()
    # LIFO: err popped first (marked), then ok.
    assert released == ["err (error)", "ok"]
    # Stack drained.
    assert ctx.rollback_labels == []
    # The failed undo's resource is still held; the released one is not.
    assert ctx.custody.held == ("err",)
    assert ctx.retained_resources == ["err"]


# ---------------------------------------------------------------------------
# Custody — a resource is released only by a confirmed undo (#3522)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_undo_released_by_an_unconverted_owner_stays_held():
    """No owner is trusted yet: a clean undo cannot confirm its release."""
    ctx = BootContext()
    ran: list[str] = []

    async def undo() -> ReleaseOutcome:
        ran.append("storage")
        return ReleaseOutcome.RELEASED

    ctx.on_rollback("storage", undo)

    assert await ctx.run_rollback() == ["storage (unknown)"]
    assert ran == ["storage"]
    assert ctx.custody.held == ("storage",)
    assert ctx.retained_resources == ["storage"]


@pytest.mark.asyncio
async def test_an_undo_is_released_by_the_owner_it_was_registered_with():
    ctx = _trusting("feature")

    async def undo() -> ReleaseOutcome:
        return ReleaseOutcome.RELEASED

    ctx.on_rollback("feature:Talon", undo, owner="feature")
    ctx.on_rollback("storage", undo)

    assert await ctx.run_rollback() == ["storage (unknown)", "feature:Talon"]
    assert ctx.custody.held == ("storage",)


@pytest.mark.parametrize(
    "outcome, label",
    [
        (ReleaseOutcome.RETAINED, "retained"),
        (ReleaseOutcome.UNKNOWN, "unknown"),
        (None, "unknown"),
    ],
    ids=["retained", "unknown", "returns-nothing"],
)
@pytest.mark.asyncio
async def test_an_undo_that_does_not_report_released_leaves_its_resource_held(
    outcome, label
):
    ctx = _trusting("ok", "unclear")
    ran: list[str] = []

    async def undo_unclear():
        ran.append("unclear")
        return outcome

    async def undo_ok() -> ReleaseOutcome:
        ran.append("ok")
        return ReleaseOutcome.RELEASED

    ctx.on_rollback("ok", undo_ok)
    ctx.on_rollback("unclear", undo_unclear)
    assert ctx.custody.held == ("ok", "unclear")

    assert await ctx.run_rollback() == [f"unclear ({label})", "ok"]
    # The unwind went on past it, and only the unclear resource stays held.
    assert ran == ["unclear", "ok"]
    assert ctx.custody.held == ("unclear",)
    assert ctx.retained_resources == ["unclear"]


@pytest.mark.parametrize("failure", ["error", "cancelled"])
@pytest.mark.asyncio
async def test_an_undo_that_raises_leaves_its_resource_held(failure):
    ctx = _trusting("worker")

    async def undo() -> ReleaseOutcome:
        if failure == "error":
            raise RuntimeError("teardown failed")
        raise asyncio.CancelledError()

    ctx.on_rollback("worker", undo)

    assert await ctx.run_rollback() == [f"worker ({failure})"]
    assert ctx.custody.held == ("worker",)
    assert ctx.retained_resources == ["worker"]


@pytest.mark.asyncio
async def test_a_rollback_into_a_shared_custody_leaves_what_it_kept_for_later():
    """An agent passes its own custody, so a later shutdown sees what stayed."""
    custody = ResourceCustody(truthful_owners={"cache", "storage"})
    ctx = BootContext(custody=custody)

    async def released() -> ReleaseOutcome:
        return ReleaseOutcome.RELEASED

    async def kept() -> ReleaseOutcome:
        return ReleaseOutcome.RETAINED

    ctx.on_rollback("cache", released)
    ctx.on_rollback("storage", kept)
    await ctx.run_rollback()

    assert custody.held == ("storage",)


@pytest.mark.asyncio
async def test_an_unheld_undo_reports_its_outcome_but_holds_nothing_itself():
    ctx = BootContext()

    async def sweep() -> ReleaseOutcome:
        return ReleaseOutcome.RETAINED

    ctx.on_rollback_unheld("features", sweep)
    assert ctx.custody.held == ()

    assert await ctx.run_rollback() == ["features (retained)"]
    assert ctx.retained_resources == ["features"]
    assert ctx.custody.held == ()


# ---------------------------------------------------------------------------
# Cancellation — boot interrupted while an async phase owns resources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellation_mid_phase_still_unwinds_all_resources():
    released: list[str] = []
    hanging_started = asyncio.Event()

    async def acquire_phase(ctx: BootContext) -> None:
        async def undo() -> ReleaseOutcome:
            released.append("db-connection")
            return ReleaseOutcome.RELEASED

        ctx.on_rollback("db-connection", undo)

    async def hang_phase(ctx: BootContext) -> None:
        hanging_started.set()
        await asyncio.Event().wait()  # never completes; awaits cancellation

    ctx = BootContext()
    rec = _Recorder()
    task = asyncio.ensure_future(
        run_boot_sequence(
            [BootPhase("acquire", acquire_phase), BootPhase("hang", hang_phase)],
            ctx,
            rec,
        )
    )

    await asyncio.wait_for(hanging_started.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The resource acquired before the cancelled phase was released; the
    # machine reached the terminal FAILED state.
    assert released == ["db-connection"]
    assert rec.states[-1] is BootPhaseState.FAILED


@pytest.mark.asyncio
async def test_cancellation_completes_teardown_even_if_undo_awaits():
    """Rollback must finish even when an undo step yields to the loop.

    The unwind runs as a shielded task, so a cancellation delivered to the
    boot coroutine cannot abandon a teardown that is mid-await.
    """
    released: list[str] = []
    hanging_started = asyncio.Event()

    async def acquire_phase(ctx: BootContext) -> None:
        async def undo() -> ReleaseOutcome:
            # Yield to the loop mid-teardown to model a real async close().
            await asyncio.sleep(0)
            released.append("worker")
            return ReleaseOutcome.RELEASED

        ctx.on_rollback("worker", undo)

    async def hang_phase(ctx: BootContext) -> None:
        hanging_started.set()
        await asyncio.Event().wait()

    ctx = BootContext()
    task = asyncio.ensure_future(
        run_boot_sequence(
            [BootPhase("acquire", acquire_phase), BootPhase("hang", hang_phase)],
            ctx,
            _Recorder(),
        )
    )
    await asyncio.wait_for(hanging_started.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released == ["worker"]
