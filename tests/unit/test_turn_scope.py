"""Turn-scoped context crosses every task boundary through one registry (#3114).

The codex app-server dispatches each inline ``item/tool/call`` on a task spawned
from a reader loop that started BEFORE the turn published its ContextVars. The
inline tool executors used to re-present those values one hand-written bind at a
time, and each value forgotten became its own bug (#2081, #2672, #2965, #3112,
and here the turn id and causation chain). These tests pin the replacement:

* the set of carried values is declared at each ContextVar's definition site
  (``kestrel_sovereign.turn_scope``), and every turn-scoped var a realistic turn
  publishes is covered — derived from a live context, not from a list here;
* both executor doors (orchestrator and feature subagent) carry it, including
  across the nested second reader-task boundary;
* the real consumers the ticket names (``todo_add``'s ``turn_id`` stamp, the
  outbound-A2A causation chain) observe the owning turn's values;
* carrying the raw turn id confers no authority: the three live-turn gates
  admit a task paired with the live turn and refuse one holding only the id;
* no code outside a declaring module re-presents a turn-scoped value by hand.
"""
from __future__ import annotations

import ast
import asyncio
import contextvars
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sdk.signals import CausationFrame
from kestrel_sdk.tools.result import ToolResultStatus

import kestrel_sovereign
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.parts import current_part_collector, part_collector
from kestrel_sovereign.agent.turn_lifecycle import (
    _BOUND_TURN_SESSION,
    TurnLifecycleMixin,
    capture_turn_session_binding,
)
from kestrel_sovereign.auth import (
    AuthMethod,
    CallerContext,
    caller_context_scope,
    current_caller_context,
)
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.todo.feature import TodoFeature
from kestrel_sovereign.signals import OrderedLockManager
from kestrel_sovereign.signals.context import (
    get_current_signal,
    reset_current_signal,
    set_current_signal,
)
from kestrel_sovereign.storage.privacy_wrapper import (
    ReentrantTransitionLock,
    current_bound_reentry_token,
)
from kestrel_sovereign.telemetry import bind_current_turn_id, current_turn_id
from kestrel_sovereign.turn_scope import (
    capture_turn_scope,
    turn_scoped,
    turn_scoped_carriers,
    turn_scoped_variables,
)


class _FakeSignal:
    """Stand-in for an SDK Signal; the guards only read ``source``/payload."""

    def __init__(self, source: str):
        self.source = source
        self.payload = {}


class _Host(TurnLifecycleMixin, OrchestratorEngineMixin):
    """A real turn lifecycle plus the real orchestrator inline executor."""

    def __init__(self) -> None:
        self.did = "did:test:turn-scope"
        self.agent_name = "turn-scope"
        self._lock_manager = OrderedLockManager()
        self._privacy_transition_lock = ReentrantTransitionLock()
        self._live_turn_id = None
        self._live_turn_task = None
        self._active_session_id = None
        self.tool_impl = None
        self.seen: dict = {}

    def _get_privacy_transition_lock(self):
        return self._privacy_transition_lock

    async def execute_named_tool(self, name, args, *, session_id, source, _capture):
        _capture["effective_args"] = args
        return await self.tool_impl(name, args)


def _observe(host: _Host) -> dict:
    """What an inline tool sees, through the accessors its consumers use."""
    return {
        "turn_id": host.get_current_turn_id(),
        "chain": host._get_current_chain(),
        "signal": get_current_signal(),
        "collector": current_part_collector(),
        "caller": current_caller_context(),
        "session": host.get_turn_bound_session_id(),
        "reentry_token": current_bound_reentry_token(),
        "context": {
            var: value for var, value in contextvars.copy_context().items()
        },
    }


class _ReaderHarness:
    """The codex reader-task topology (see #2081's regression test).

    The reader is spawned ONCE, before the turn publishes anything, so every
    call it dispatches runs on a task carrying a frozen pre-turn context.
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader = None
        self.baseline: dict = {}

    async def start(self):
        self._reader = asyncio.create_task(self._read_loop())
        # Yield so the reader is actually running before the turn publishes.
        await asyncio.sleep(0)

    async def _read_loop(self):
        self.baseline = dict(contextvars.copy_context().items())
        while True:
            item = await self._queue.get()
            if item is None:
                return
            executor, name, args, done = item
            asyncio.create_task(self._handle(executor, name, args, done))

    @staticmethod
    async def _handle(executor, name, args, done):
        try:
            done.set_result(await executor(name, args))
        except Exception as exc:  # noqa: BLE001 - re-raised via the future
            done.set_exception(exc)

    async def dispatch(self, executor, name="tool", args=None):
        done = asyncio.get_running_loop().create_future()
        await self._queue.put((executor, name, args or {}, done))
        return await done

    async def stop(self):
        await self._queue.put(None)
        await self._reader


def _chain() -> list:
    return [
        CausationFrame(
            agent_id="did:test:peer",
            source="a2a.task_complete",
            signal_id="sig-1",
            turn_id="turn-upstream",
            depth=1,
            emitted_at=datetime(2026, 9, 22, tzinfo=timezone.utc),
        )
    ]


def _caller() -> CallerContext:
    return CallerContext.sovereign(AuthMethod.API_KEY, "operator")


async def _run_turn(
    host: _Host, harness: _ReaderHarness, build_executor, name="tool", args=None
):
    """Publish a realistic signal-driven turn, then dispatch via the reader.

    Mirrors production ordering: the dispatcher publishes the signal and chain
    on its task, the endpoint owns the caller, the turn lifecycle publishes the
    turn id and ownership, the streamed turn holds the transition lock and a
    part collector — and only then is the inline executor built.
    """
    signal = _FakeSignal("a2a.task_submitted")
    signal_token = set_current_signal(signal)
    chain_token = host._set_current_chain(_chain())
    try:
        with caller_context_scope(_caller()):
            async with host._turn_lifecycle() as turn_id:
                host._active_session_id = "chat-owner"
                async with host._privacy_transition_lock:
                    with part_collector():
                        published = dict(contextvars.copy_context().items())
                        executor = build_executor()
                        result = await harness.dispatch(executor, name, args)
                        lock_token = (
                            host._privacy_transition_lock.current_reentry_token()
                        )
                        collector = current_part_collector()
    finally:
        host._clear_current_chain(chain_token)
        reset_current_signal(signal_token)
    return {
        "turn_id": turn_id,
        "signal": signal,
        "published": published,
        "lock_token": lock_token,
        "collector": collector,
        "result": result,
    }


def _assert_owning_turn_observed(seen: dict, turn: dict) -> None:
    assert seen["turn_id"] == turn["turn_id"], (
        "inline tool read turn_id "
        f"{seen['turn_id']!r}; todo metadata / origin_turn_id would stamp it"
    )
    assert seen["chain"] == _chain(), (
        "inline tool lost the causation chain; signals and outbound A2A tasks "
        "it emits would restart lineage at depth 1"
    )
    assert seen["signal"] is turn["signal"]
    assert seen["collector"] is turn["collector"]
    assert seen["caller"] == _caller()
    assert seen["session"] == "chat-owner"
    assert seen["reentry_token"] is turn["lock_token"]


# ---------------------------------------------------------------------------
# Declaration-site registry
# ---------------------------------------------------------------------------


def test_every_turn_scoped_value_is_declared_at_its_definition_site():
    from kestrel_sovereign import auth, telemetry
    from kestrel_sovereign.agent import parts, turn_lifecycle
    from kestrel_sovereign.signals import context
    from kestrel_sovereign.storage import privacy_wrapper

    declared = {
        carrier.name: {var.name for var in carrier.variables}
        for carrier in turn_scoped_carriers()
    }
    expected = {
        "part_collector": {parts._part_collector.name},
        "transition_lock_reentry": {
            privacy_wrapper._transition_lock_reentry_token.name
        },
        "turn_session": {turn_lifecycle._BOUND_TURN_SESSION.name},
        "caller_context": {auth._current_caller_context.name},
        "current_signal": {context._current_signal.name},
        "turn_id": {telemetry._CURRENT_TURN_ID.name},
        "causation_chain": {turn_lifecycle._CURRENT_CHAIN.name},
    }
    for name, var_names in expected.items():
        assert declared.get(name) == var_names, name


def test_registry_refuses_two_writers_for_one_value():
    var = contextvars.ContextVar("kestrel_test_turn_scope_probe")

    turn_scoped(
        "test_probe",
        variables=(var,),
        capture=lambda _agent: None,
        bind=lambda _value: contextlib_null(),
    )
    try:
        with pytest.raises(ValueError, match="already declared"):
            turn_scoped(
                "test_probe",
                variables=(contextvars.ContextVar("kestrel_test_other"),),
                capture=lambda _agent: None,
                bind=lambda _value: contextlib_null(),
            )
        with pytest.raises(ValueError, match="already carried"):
            turn_scoped(
                "test_probe_second_owner",
                variables=(var,),
                capture=lambda _agent: None,
                bind=lambda _value: contextlib_null(),
            )
    finally:
        from kestrel_sovereign import turn_scope

        turn_scope._CARRIERS.pop("test_probe", None)


def contextlib_null():
    import contextlib

    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Orchestrator door
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_executor_carries_turn_id_and_chain_to_reader_task():
    """The #3114 reproduction: reader spawned before the turn publishes."""
    host = _Host()
    harness = _ReaderHarness()
    await harness.start()

    async def tool(_name, _args):
        host.seen = _observe(host)
        return {"ok": True}

    host.tool_impl = tool
    try:
        turn = await _run_turn(
            host, harness, lambda: host._make_inline_tool_executor("chat-owner")
        )
    finally:
        await harness.stop()

    assert harness.baseline.get(_BOUND_TURN_SESSION) is None
    assert current_turn_id() is None
    _assert_owning_turn_observed(host.seen, turn)


@pytest.mark.asyncio
async def test_every_var_the_turn_publishes_is_declared_and_carried():
    """Completeness, derived from a live context rather than a list.

    Every ContextVar whose value on the turn differs from what the reader task
    was born with is exactly the set an inline tool would otherwise read at a
    stale value. Each must be declared turn-scoped at its definition site, and
    must reach the tool carrying the turn's value.
    """
    host = _Host()
    harness = _ReaderHarness()
    await harness.start()

    async def tool(_name, _args):
        host.seen = _observe(host)
        return {"ok": True}

    host.tool_impl = tool
    try:
        turn = await _run_turn(
            host, harness, lambda: host._make_inline_tool_executor("chat-owner")
        )
    finally:
        await harness.stop()

    published = {
        var: value
        for var, value in turn["published"].items()
        if harness.baseline.get(var, _MISSING) is not value
    }
    assert published, "precondition: the turn published turn-scoped state"

    undeclared = sorted(
        var.name for var in published if var not in turn_scoped_variables()
    )
    assert not undeclared, (
        f"The turn publishes {undeclared} but no turn_scoped(...) declaration "
        "carries them, so an inline tool reads a stale value. Declare the "
        "carrier next to the ContextVar."
    )

    inside = host.seen["context"]
    stale = []
    for var, value in published.items():
        seen = inside.get(var, _MISSING)
        if var is _BOUND_TURN_SESSION:
            # The lifecycle's own binding is re-presented as the captured
            # explicit pair for the same agent and turn — a deliberate
            # transformation, checked semantically.
            if not (
                seen is not _MISSING
                and seen.agent is value.agent
                and seen.turn_id == value.turn_id
                and not seen.lifecycle
            ):
                stale.append(var.name)
        elif seen != value:
            stale.append(var.name)
    assert not stale, f"carried with the wrong value: {sorted(stale)}"


_MISSING = object()


@pytest.mark.asyncio
async def test_off_turn_executor_manufactures_nothing():
    """An executor built off-turn carries defaults, including explicit clears."""
    host = _Host()
    harness = _ReaderHarness()
    await harness.start()

    async def tool(_name, _args):
        host.seen = _observe(host)
        return {"ok": True}

    host.tool_impl = tool
    try:
        executor = host._make_inline_tool_executor("no-turn")
        await harness.dispatch(executor)
    finally:
        await harness.stop()

    assert host.seen["turn_id"] is None
    assert host.seen["chain"] is None
    assert host.seen["signal"] is None
    assert host.seen["collector"] is None
    assert host.seen["caller"] is None
    assert host.seen["session"] is None


@pytest.mark.asyncio
async def test_todo_add_on_reader_task_stamps_the_owning_turn_id():
    """The consumer the ticket names: the real TodoFeature's metadata stamp."""
    host = _Host()
    host.storage = MagicMock()
    host.storage.graph = MagicMock()
    host.storage.graph.add_node = AsyncMock()
    todo = TodoFeature(host)
    await todo.initialize()
    harness = _ReaderHarness()
    await harness.start()

    async def tool(_name, args):
        return await todo.todo_add(**args)

    host.tool_impl = tool
    try:
        turn = await _run_turn(
            host,
            harness,
            lambda: host._make_inline_tool_executor("chat-owner"),
            name="todo_add",
            args={"title": "watch CI"},
        )
    finally:
        await harness.stop()

    _effective_args, result = turn["result"]
    assert result.status == ToolResultStatus.OK, result
    written = host.storage.graph.add_node.await_args.args[0]
    assert written.properties["source_turn"] == {
        "turn_id": turn["turn_id"],
        "session_id": "chat-owner",
    }


@pytest.mark.asyncio
async def test_outbound_a2a_chain_provider_sees_the_turn_chain_on_reader_task():
    """The causation-chain consumer: TaskManager's provider reads the chain."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    host = _Host()
    host._provide_causation_chain = KestrelAgent._provide_causation_chain.__get__(
        host
    )
    harness = _ReaderHarness()
    await harness.start()

    async def tool(_name, _args):
        host.seen = {"provided": host._provide_causation_chain()}
        return {"ok": True}

    host.tool_impl = tool
    try:
        await _run_turn(
            host, harness, lambda: host._make_inline_tool_executor("chat-owner")
        )
    finally:
        await harness.stop()

    provided = host.seen["provided"]
    assert provided, "outbound A2A task would carry no causation chain"
    assert provided[0]["signal_id"] == "sig-1"


# ---------------------------------------------------------------------------
# Feature door (nested second reader-task boundary)
# ---------------------------------------------------------------------------


class _Subagent(Feature):
    name = "turn_scope_probe"

    async def initialize(self):
        return None

    def tool_description(self) -> str:
        return "probe"

    def get_tools(self):
        return []


@pytest.mark.asyncio
async def test_nested_feature_executor_carries_turn_scope_across_second_reader():
    """Parent inline tool builds a subagent executor on the parent reader task;
    the subagent's own tool runs on a SECOND pre-turn reader task."""
    host = _Host()
    host.hooks_manager = None
    feature = _Subagent(host)
    parent_reader = _ReaderHarness()
    nested_reader = _ReaderHarness()
    await parent_reader.start()
    await nested_reader.start()

    async def nested_tool(**_kwargs):
        host.seen = _observe(host)
        return {"ok": True}

    probe = MagicMock()
    probe.name = "probe"
    probe.execute = nested_tool

    async def parent_tool(_name, _args):
        # Built HERE, on the parent reader task, inside the parent's bind.
        sub_executor = feature._make_feature_inline_tool_executor(
            parts_sink=None, runtime_tools=[probe]
        )
        return await nested_reader.dispatch(sub_executor, "probe", {})

    host.tool_impl = parent_tool
    try:
        turn = await _run_turn(
            host,
            parent_reader,
            lambda: host._make_inline_tool_executor("chat-owner"),
        )
    finally:
        await parent_reader.stop()
        await nested_reader.stop()

    assert host.seen, "nested tool did not run"
    _assert_owning_turn_observed(host.seen, turn)


@pytest.mark.asyncio
async def test_feature_executor_built_on_turn_task_carries_the_held_reentry_token():
    """A subagent executor built on the turn task itself (no parent inline
    executor) must still carry the lock-held reentry token; reading only the
    bound ContextVar there captured ``None`` and deadlocked a nested write."""
    host = _Host()
    feature = _Subagent(host)
    async with host._turn_lifecycle():
        async with host._privacy_transition_lock:
            expected = host._privacy_transition_lock.current_reentry_token()
            snapshot = capture_turn_scope(feature.agent)
    captured = {carrier.name: value for carrier, value in snapshot.captured}
    assert expected is not None
    assert captured["transition_lock_reentry"] is expected


# ---------------------------------------------------------------------------
# Live-turn gates: owned vs. inherited
# ---------------------------------------------------------------------------


class _ValidSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def get_span_context(self):
        ctx = MagicMock()
        ctx.is_valid = True
        ctx.trace_id = 0x0123456789ABCDEF0123456789ABCDEF
        ctx.span_id = 0x0123456789ABCDEF
        return ctx


def _gates(host: _Host) -> dict:
    """Each gate probed independently of the others."""
    span = _ValidSpan()
    span_bound = host.bind_current_turn_span(span)
    return {
        "capture": capture_turn_session_binding(host).turn_id,
        "trace_identity": host.bind_current_turn_trace_identity(
            "0123456789abcdef0123456789abcdef", "0123456789abcdef"
        ),
        # bind_current_turn_span stamps the span BEFORE delegating to the
        # trace-identity gate, so its own gate is observed by the stamp.
        "span": span_bound and bool(span.attributes),
        "span_stamped": bool(span.attributes),
    }


@pytest.mark.asyncio
async def test_gates_admit_the_owning_turn():
    host = _Host()
    async with host._turn_lifecycle() as turn_id:
        assert _gates(host) == {
            "capture": turn_id,
            "trace_identity": True,
            "span": True,
            "span_stamped": True,
        }


@pytest.mark.asyncio
async def test_gates_refuse_a_task_holding_only_the_carried_turn_id():
    """Carrying the raw turn id onto a foreign task is observability only."""
    host = _Host()
    harness = _ReaderHarness()
    await harness.start()
    results = {}

    async def only_turn_id(turn_id):
        with bind_current_turn_id(turn_id):
            assert host.get_current_turn_id() == turn_id
            results.update(_gates(host))
        return None, None

    try:
        async with host._turn_lifecycle() as turn_id:
            await harness.dispatch(lambda _n, _a: only_turn_id(turn_id))
    finally:
        await harness.stop()

    assert results == {
        "capture": None,
        "trace_identity": False,
        "span": False,
        "span_stamped": False,
    }


@pytest.mark.asyncio
async def test_gates_admit_an_explicitly_re_presented_turn_scope():
    host = _Host()
    harness = _ReaderHarness()
    await harness.start()
    results = {}

    async def re_presented(snapshot):
        with snapshot.bind():
            results.update(_gates(host))
        return None, None

    try:
        async with host._turn_lifecycle() as turn_id:
            snapshot = capture_turn_scope(host)
            await harness.dispatch(lambda _n, _a: re_presented(snapshot))
    finally:
        await harness.stop()

    assert results == {
        "capture": turn_id,
        "trace_identity": True,
        "span": True,
        "span_stamped": True,
    }


@pytest.mark.asyncio
async def test_lifecycle_binding_inherited_by_a_child_is_not_a_lock_grant():
    """Genuine descendants inherit the lifecycle's binding for session and
    span attribution, but not the explicit grant that lets a callback skip
    CONVERSATION in ``privacy_transition``."""
    host = _Host()
    async with host._turn_lifecycle() as turn_id:
        host._active_session_id = "chat-owner"

        async def child():
            return (
                host.get_turn_bound_session_id(),
                capture_turn_session_binding(host).turn_id,
                host._caller_belongs_to_live_turn(),
            )

        assert await asyncio.create_task(child()) == (
            "chat-owner",
            turn_id,
            False,
        )


# ---------------------------------------------------------------------------
# Enforcement: nothing re-presents a turn-scoped value by hand
# ---------------------------------------------------------------------------

# Calls to a carrier's bind primitive outside its declaring module that are
# NOT re-presenting captured turn state, with the reason. Anything else must go
# through capture_turn_scope(...).bind().
_ALLOWED_DIRECT_BINDS = {
    # The invocation boundary publishes the endpoint's own caller lifetime on
    # each ``anext`` of an isolated invocation — it precedes any turn.
    ("agent/invocation.py", "caller_context_binding_scope"),
    # A subagent binds its LOCAL parts sink, not a captured turn collector.
    ("features/base.py", "bind_part_collector"),
}


def _declaring_file(func) -> str:
    module = getattr(func, "__module__", "")
    return module.replace("kestrel_sovereign.", "").replace(".", "/") + ".py"


def test_no_closure_re_presents_turn_scoped_state_by_hand():
    import kestrel_sovereign.signals.context  # noqa: F401 - lazily declared
    import kestrel_sovereign.kestrel_agent  # noqa: F401 - load every declaration

    bind_owners = {
        carrier.bind.__name__: _declaring_file(carrier.bind)
        for carrier in turn_scoped_carriers()
        if carrier.name
        in {
            "part_collector",
            "transition_lock_reentry",
            "turn_session",
            "caller_context",
            "current_signal",
            "turn_id",
            "causation_chain",
        }
    }
    assert "<lambda>" not in bind_owners, (
        "a core carrier's bind must be a named function so this scan can find "
        "hand-written re-presentations of it"
    )

    root = Path(kestrel_sovereign.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            owner = bind_owners.get(name)
            if owner is None or owner == rel:
                continue
            if (rel, name) in _ALLOWED_DIRECT_BINDS:
                continue
            offenders.append(f"{rel}:{node.lineno} {name}")

    assert not offenders, (
        "These re-present one turn-scoped value by hand; a closure crossing a "
        "task boundary must use capture_turn_scope(agent).bind() so it carries "
        f"every declared value: {offenders}"
    )
