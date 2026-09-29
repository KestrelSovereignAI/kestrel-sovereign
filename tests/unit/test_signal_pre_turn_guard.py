"""#3310 — the source-declared pre-turn admission seam.

`tests/unit/test_non_streaming_turn_privacy_span.py` pins the SPAN: that
`KestrelAgent.process_input` holds CONVERSATION and then the privacy-transition
mutex for its whole body. This module pins the thing the span was opened FOR —
the one place a COGNITION source's precondition is evaluated.

The seam has three halves and each can fail silently on its own, which is why
each is pinned separately here:

* the registry (`signals/registry.py`) refuses a guard that could never run, or
  that could reintroduce a suspension point into the span;
* the dispatcher (`signals/dispatcher.py`) refuses to dispatch at all when the
  agent cannot evaluate the guard, rather than running the turn unadmitted; and
* the agent (`KestrelAgent._evaluate_pre_turn_guard`) runs it as the FIRST
  operation inside the span and turns a refusal into `PreTurnRefusal`.

The last test is the issue's own reproduction: a privacy transition that lands
inside the dispatch must not reach a turn carrying the persisted intent.
"""

from __future__ import annotations

import asyncio
import gc
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import (
    RedactionPolicy,
    ResourceLock,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
)

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.signals import (
    OrderedLockManager,
    PreTurnRefusal,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistrationWithPreTurnGuard,
    SourceRegistry,
)
from kestrel_sovereign.signals.registry import RegistrationError
from kestrel_sovereign.storage.db import SQLiteBackend


def _redaction() -> RedactionPolicy:
    return RedactionPolicy(summarize=lambda p: "<redacted>")


def _guard_reg(
    template: Path,
    guard,
    *,
    name: str = "guarded_src",
    modes=None,
    **overrides,
) -> SourceRegistrationWithPreTurnGuard:
    allowed = modes or frozenset({SignalMode.COGNITION})
    base = dict(
        name=name,
        schema=dict,
        default_mode=(
            SignalMode.COGNITION
            if SignalMode.COGNITION in allowed
            else next(iter(allowed))
        ),
        allowed_modes=allowed,
        prompt_template=template,
        log_redaction=_redaction(),
        pre_turn_guard=guard,
    )
    base.update(overrides)
    return SourceRegistrationWithPreTurnGuard(**base)


def _signal(source: str, *, target: str = "agent-test", payload=None) -> Signal:
    return Signal(
        source=source,
        kind="tick",
        mode=SignalMode.COGNITION,
        payload=payload if payload is not None else {},
        target_agent=target,
        causation_chain=[],
    )


@pytest.fixture
def template(tmp_path) -> Path:
    path = tmp_path / "guarded.md"
    path.write_text("payload: {payload}")
    return path


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_the_seam_is_exported_from_the_signals_package():
    """Every name the package re-exports must be in ``__all__``.

    `PreTurnRefusal` especially: the dispatcher maps it to
    `Status.DROPPED_VALIDATION`, so recognizing a refusal is part of the public
    contract, not an internal detail. Re-exporting a name but omitting it from
    ``__all__`` makes it invisible to `import *` and to consumers that check the
    declared surface, while still looking exported at the import site.
    """
    import kestrel_sovereign.signals as signals

    for name in (
        "BoundPreTurnGuard",
        "PreTurnGuard",
        "PreTurnRefusal",
        "SourceRegistrationWithPreTurnGuard",
    ):
        assert hasattr(signals, name), f"{name} is not importable from the package"
        assert name in signals.__all__, f"{name} is re-exported but not in __all__"


# ---------------------------------------------------------------------------
# Registry — a guard that could never run is a source contract error
# ---------------------------------------------------------------------------


def test_registry_rejects_a_non_callable_guard(template):
    with pytest.raises(RegistrationError, match="must be callable"):
        SourceRegistry().register(_guard_reg(template, "not-a-callable"))


def test_registry_rejects_a_guard_on_a_source_with_no_cognition(template):
    """ACTION / ARTIFACT have no turn to refuse, so the guard would no-op."""

    async def handler(payload):
        return "ok"

    with pytest.raises(RegistrationError, match="does not allow COGNITION"):
        SourceRegistry().register(
            _guard_reg(
                template,
                lambda signal: None,
                modes=frozenset({SignalMode.ACTION}),
                prompt_template=None,
                handler=handler,
            )
        )


def test_registry_rejects_a_coroutine_function_guard(template):
    """A guard that can await would put a suspension point back in the span."""

    async def async_guard(signal):
        return None

    with pytest.raises(RegistrationError, match="must be synchronous"):
        SourceRegistry().register(_guard_reg(template, async_guard))


def test_registry_accepts_a_synchronous_cognition_guard(template):
    registry = SourceRegistry()
    registry.register(_guard_reg(template, lambda signal: None))
    assert registry.get("guarded_src") is not None


def test_swapping_the_guard_is_a_contract_mismatch(template):
    """A re-registration that changes admission must not compare equivalent.

    Comparing them equivalent would keep the OLD admission decision in force
    behind a running dispatcher — the silent failure the whole seam exists to
    remove.
    """
    first = _guard_reg(template, lambda signal: None)
    second = _guard_reg(template, lambda signal: "refused")

    assert not SourceRegistry.contract_equivalent(first, second)
    # ... and an unguarded registration is not equivalent to a guarded one.
    unguarded = SourceRegistration(
        name="guarded_src",
        schema=dict,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=template,
        log_redaction=_redaction(),
    )
    assert not SourceRegistry.contract_equivalent(first, unguarded)


def test_the_same_guard_is_contract_equivalent(template):
    """The fingerprint must not produce a spurious mismatch on re-registration."""
    guard = lambda signal: None  # noqa: E731 - identity is the point
    assert SourceRegistry.contract_equivalent(
        _guard_reg(template, guard), _guard_reg(template, guard)
    )


# ---------------------------------------------------------------------------
# Agent — the evaluation that runs inside the span
# ---------------------------------------------------------------------------


def test_absent_guard_is_a_noop():
    KestrelAgent._evaluate_pre_turn_guard(None)


def test_guard_returning_none_admits_the_turn():
    KestrelAgent._evaluate_pre_turn_guard(lambda: None)


def test_guard_returning_a_reason_refuses_with_that_reason():
    with pytest.raises(PreTurnRefusal, match="privacy mode forbids this intent"):
        KestrelAgent._evaluate_pre_turn_guard(
            lambda: "privacy mode forbids this intent"
        )


def test_guard_returning_an_awaitable_fails_closed_without_warning():
    """An async guard is a contract violation, not a refusal — and must not
    leak an un-awaited coroutine as an unrelated RuntimeWarning at GC time."""

    async def sneaky():
        return None

    def guard():
        return sneaky()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(TypeError, match="must be synchronous"):
            KestrelAgent._evaluate_pre_turn_guard(guard)
        gc.collect()


def test_a_guard_that_raises_is_not_laundered_into_a_refusal():
    """A bug in the guard must surface as a bug, not as a policy decision."""

    def guard():
        raise ZeroDivisionError("boom")

    with pytest.raises(ZeroDivisionError):
        KestrelAgent._evaluate_pre_turn_guard(guard)


# ---------------------------------------------------------------------------
# Dispatcher — capability refusal and refusal encoding
# ---------------------------------------------------------------------------


class _GuardAwareAgent:
    """An agent that implements the in-span admission contract."""

    def __init__(self, did: str = "agent-test"):
        self._did = did
        self.background_tasks: list[asyncio.Task] = []
        self.process_input_calls: list[str] = []

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt: str, pre_turn_guard=None):
        # Mirror the real agent: evaluate first, then consume the prompt.
        KestrelAgent._evaluate_pre_turn_guard(pre_turn_guard)
        self.process_input_calls.append(prompt)
        return "ok"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


class _KwargsSwallowingAgent(_GuardAwareAgent):
    """An agent whose ``**kwargs`` would silently absorb the guard."""

    async def process_input(self, prompt: str, **kwargs):
        self.process_input_calls.append(prompt)
        return "ok"


class _UnawareAgent(_GuardAwareAgent):
    """An agent predating the contract entirely."""

    async def process_input(self, prompt: str):
        self.process_input_calls.append(prompt)
        return "ok"


async def _components(db_path, agent):
    backend = SQLiteBackend(db_path)
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    return SimpleNamespace(
        dispatcher=dispatcher,
        agent=agent,
        registry=registry,
        store=store,
        backend=backend,
    )


@pytest.fixture
async def guard_components(tmp_path, request):
    agent_cls = getattr(request, "param", _GuardAwareAgent)
    c = await _components(str(tmp_path / "signal_log.db"), agent_cls())
    yield c
    for task in list(c.agent.background_tasks):
        if not task.done():
            task.cancel()
    await asyncio.gather(*c.agent.background_tasks, return_exceptions=True)
    await c.backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "guard_components",
    [_UnawareAgent, _KwargsSwallowingAgent],
    indirect=True,
)
async def test_an_agent_that_cannot_evaluate_the_guard_is_refused(
    guard_components, template
):
    """No turn runs when the agent does not implement the contract.

    ``**kwargs`` is deliberately NOT accepted as implementing it: absorbing the
    guard silently would run the turn with the source's precondition never
    evaluated, which is exactly the failure the guard removes.
    """
    c = guard_components
    calls: list[Signal] = []
    c.registry.register(
        _guard_reg(template, lambda signal: calls.append(signal) or None)
    )

    result = await c.dispatcher.dispatch_signal(_signal("guarded_src"))

    assert result.status is Status.DROPPED_VALIDATION
    assert "does not implement the in-span admission contract" in (
        result.error or ""
    )
    assert c.agent.process_input_calls == []
    assert calls == [], "the guard must not run outside the span"


@pytest.mark.asyncio
async def test_a_refusing_guard_yields_no_turn_and_a_non_success_occurrence(
    guard_components, template
):
    c = guard_components
    c.registry.register(
        _guard_reg(template, lambda signal: "storage is none; intent refused")
    )

    result = await c.dispatcher.dispatch_signal(_signal("guarded_src"))

    assert result.status is Status.DROPPED_VALIDATION
    assert result.error == "storage is none; intent refused"
    assert result.turn_id is None
    assert c.agent.process_input_calls == []


@pytest.mark.asyncio
async def test_an_admitting_guard_runs_the_turn_once_with_its_signal(
    guard_components, template
):
    c = guard_components
    seen: list[Signal] = []

    def guard(signal):
        seen.append(signal)
        return None

    c.registry.register(_guard_reg(template, guard))

    result = await c.dispatcher.dispatch_signal(
        _signal("guarded_src", payload={"intent": "verify CI"})
    )

    assert result.status is Status.OK
    assert len(c.agent.process_input_calls) == 1
    assert len(seen) == 1, "the guard is evaluated exactly once, in the span"
    assert seen[0].source == "guarded_src"
    assert seen[0].payload == {"intent": "verify CI"}


# ---------------------------------------------------------------------------
# The real agent — the refusal has to survive the isolated-task boundary
# ---------------------------------------------------------------------------


def _real_turn_agent(did: str) -> KestrelAgent:
    """A real agent stopped just short of the machinery a turn body needs."""
    agent = KestrelAgent(did=did, storage_path=":memory:")
    agent.storage = object()
    agent.context_manager = object()
    agent.bootstrap_service = None
    agent._safe_mode = False

    async def _noop(*args, **kwargs):
        return None

    agent._maybe_audit = _noop
    agent._maybe_refresh_user_byok_resolver = _noop
    agent._genesis_audit_cognition_block = _noop
    return agent


@pytest.mark.asyncio
async def test_refusal_survives_the_real_process_input_task_boundary(
    tmp_path, template
):
    """`PreTurnRefusal` must reach the dispatcher intact from the real agent.

    The fake agents above call `_evaluate_pre_turn_guard` directly. The real
    `process_input` is wrapped by `bind_async_invocation(track_request_lifecycle=True)`,
    which runs the turn body in a SEPARATE isolated asyncio task and re-raises
    through `await isolated_operation`. A task boundary is exactly where a typed
    exception gets mangled into something generic, which would turn this policy
    decision into `Status.FAILED` with a traceback. Pin the whole path.
    """
    agent = _real_turn_agent("did:test:3310-real-refusal")
    turn_bodies: list[str] = []

    async def traced(user_input, *args, **kwargs):
        turn_bodies.append(user_input)
        return "ok"

    agent._process_input_traced_locked = traced

    # The issue's round 3 only reproduces with the monitoring hook in the loop:
    # `await_monitored_execution` then runs the turn in a SEPARATE task and
    # yields before `process_input` starts. A real KestrelAgent carries that
    # hook via EventManagerMixin, so assert the monitored branch is genuinely
    # taken rather than assuming it — if the hook ever stops being consulted,
    # this test silently degrades to the straight-line path it exists to avoid.
    monitored: list[object] = []
    real_monitor = agent.monitor_cognition_signal_execution

    async def counting_monitor(signal):
        monitored.append(signal)
        return await real_monitor(signal)

    agent.monitor_cognition_signal_execution = counting_monitor

    c = await _components(str(tmp_path / "real.db"), agent)
    c.registry.register(
        _guard_reg(
            template,
            lambda signal: "intent withdrawn",
            name="real_refusal_src",
        )
    )

    try:
        result = await asyncio.wait_for(
            c.dispatcher.dispatch_signal(
                _signal("real_refusal_src", target=agent.did)
            ),
            timeout=5,
        )

        assert result.status is Status.DROPPED_VALIDATION, (
            "a guard refusal must stay a policy decision across the isolated "
            f"task boundary, not become {result.status}: {result.error!r}"
        )
        assert result.error == "intent withdrawn"
        assert turn_bodies == [], "no part of the turn body may run"
        assert monitored, (
            "the monitoring hook was not consulted, so the turn did NOT run in "
            "a separate task — this test no longer covers the issue's round 3"
        )
        # A refusal unwinds BOTH locks, or the next turn and every privacy
        # transition wedge behind a turn that never ran.
        assert not agent._get_privacy_transition_lock().locked()
        assert not agent._get_lock_manager().is_held(ResourceLock.CONVERSATION)
    finally:
        await c.backend.close()


@pytest.mark.asyncio
async def test_real_process_input_admits_and_runs_the_turn(tmp_path, template):
    """The admitting case on the same real path, so the refusal test can fail."""
    agent = _real_turn_agent("did:test:3310-real-admit")
    turn_bodies: list[str] = []
    locked_in_body: list[bool] = []

    async def traced(user_input, *args, **kwargs):
        locked_in_body.append(agent._get_privacy_transition_lock().locked())
        turn_bodies.append(user_input)
        return "ok"

    agent._process_input_traced_locked = traced

    c = await _components(str(tmp_path / "real-admit.db"), agent)
    c.registry.register(
        _guard_reg(template, lambda signal: None, name="real_admit_src")
    )

    try:
        result = await asyncio.wait_for(
            c.dispatcher.dispatch_signal(
                _signal("real_admit_src", target=agent.did)
            ),
            timeout=5,
        )

        assert result.status is Status.OK
        assert len(turn_bodies) == 1
        assert locked_in_body == [True], (
            "the admitted turn must consume its prompt inside the span"
        )
        assert not agent._get_privacy_transition_lock().locked()
    finally:
        await c.backend.close()


# ---------------------------------------------------------------------------
# The issue's reproduction, end to end against a real agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_transition_inside_the_dispatch_cannot_reach_the_turn(
    tmp_path, template
):
    """#3310 acceptance: the reproduction yields no turn.

    A guard that reads privacy state is only worth anything if nothing can
    change the answer before the prompt is consumed. Here the transition is
    started while the dispatch is already in flight, so it lands in one of the
    dispatch's own suspension points — the exact interleaving that defeated a
    check at schedule creation, at fire time, and at the last synchronous
    instant before handoff.

    The transition serializes on CONVERSATION -> privacy, and the turn holds
    both for its whole body, so there are only two admissible outcomes and the
    test accepts either: the transition completes first and the guard refuses
    (no turn), or the turn's span wins and the guard admits a turn that ran
    entirely under the pre-transition policy. What must NOT happen is a turn
    that consumed the prompt after the flip.
    """
    from kestrel_sovereign.privacy import PrivacyMode

    agent = KestrelAgent(did="did:test:3310-repro", storage_path=":memory:")
    agent.storage = object()
    agent.context_manager = object()
    agent.bootstrap_service = None
    agent._safe_mode = False

    async def _noop(*args, **kwargs):
        return None

    agent._maybe_audit = _noop
    agent._maybe_refresh_user_byok_resolver = _noop
    agent._genesis_audit_cognition_block = _noop

    flipped = asyncio.Event()
    turn_prompts: list[str] = []
    policy_at_consumption: list[bool] = []

    async def traced(user_input, *args, **kwargs):
        # Several suspension points before the prompt is consumed, mirroring
        # the readiness awaits a real turn performs.
        for _ in range(5):
            await asyncio.sleep(0)
        policy_at_consumption.append(flipped.is_set())
        turn_prompts.append(user_input)
        return "ok"

    agent._process_input_traced_locked = traced

    async def apply(_mode):
        flipped.set()
        return object()

    agent._set_privacy_mode_with_effects_locked = apply

    c = await _components(str(tmp_path / "repro.db"), agent)
    c.dispatcher._agent = agent

    def guard(signal):
        # The persisted intent is only admissible while the mode still permits
        # it. `flipped` stands in for that durable policy read.
        return "privacy transition withdrew this intent" if flipped.is_set() else None

    c.registry.register(_guard_reg(template, guard, name="repro_src"))

    dispatch = asyncio.create_task(
        c.dispatcher.dispatch_signal(
            _signal("repro_src", target=agent.did, payload={"intent": "SENTINEL"})
        )
    )
    # Let the dispatch reach its first suspension, then start the transition so
    # it races the remaining handoff.
    await asyncio.sleep(0)
    transition = asyncio.create_task(
        agent.set_privacy_mode_with_effects(PrivacyMode.EPHEMERAL)
    )

    result = await asyncio.wait_for(dispatch, timeout=5)
    await asyncio.wait_for(transition, timeout=5)

    try:
        if result.status is Status.DROPPED_VALIDATION:
            assert result.error == "privacy transition withdrew this intent"
            assert turn_prompts == [], "a refused wake must run no turn"
        else:
            assert result.status is Status.OK
            assert policy_at_consumption == [False], (
                "the turn consumed its prompt after the privacy transition "
                "landed — the guard's answer was invalidated inside the span"
            )
        assert not agent._get_privacy_transition_lock().locked()
    finally:
        for task in list(agent.background_tasks if hasattr(
            agent, "background_tasks"
        ) else []):
            if not task.done():
                task.cancel()
        await c.backend.close()
