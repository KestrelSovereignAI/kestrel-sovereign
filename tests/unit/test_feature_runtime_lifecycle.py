"""Canonical feature teardown / activation lifecycle (kestrel-sovereign#2522).

These drive the REAL agent runtime-lifecycle rails end to end — no endpoint-local
or test-only shims — and pin the three P1 contracts the GPT-5.6 Terra review found:

* **P1 #1 (production path):** the public disable/enable endpoints route through
  the ONE canonical teardown/activation on the agent, so a disabled feature has
  its signal sources, wait providers, hooks, A2A agent, AND dynamic tools all
  detached (the old light path left ``task:`` / ``talon:`` wait providers and
  dispatcher sources live), and enable recreates every one of them on the SAME
  loaded instance.
* **P1 #2:** ``_unregister_feature_runtime`` runs every INDEPENDENT inverse
  cleanup unconditionally even when ``on_disable`` raises, then re-raises — so a
  failing ``on_disable`` leaves no provider / source / hook / tool / feature
  behind.
* **P1 #3:** wait-provider ownership is identity-aware — shutdown restores the
  host provider a feature displaced only while the feature's own provider is
  still current, and leaves a newer provider untouched.
"""

from __future__ import annotations

import asyncio
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sdk.features import (
    FeaturePermissionDefaults,
    PermissionLevel as SDKPermissionLevel,
)
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.tools import Outcome, WaitStatus
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sovereign.endpoints.features import router as features_router
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.security.feature import SecurityFeature
from kestrel_sovereign.features.security.permissions import PermissionLevel
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.signals.registry import SourceRegistry
from kestrel_sovereign.waits import WaitRegistry


# ---------------------------------------------------------------------------
# Test doubles: a real Feature that acquires ONE of every runtime registration
# ---------------------------------------------------------------------------


def _fake_source_registration(name: str):
    from kestrel_sdk.signals import (
        RedactionPolicy,
        SignalMode,
        SourceRegistration,
        Trust,
    )

    async def handler(payload):  # pragma: no cover - never dispatched here
        return None

    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda p: ""),
    )


class _LifecycleHook(Hook):
    def __init__(self) -> None:
        super().__init__(name="lifecycle_hook", events=[HookEvent.SESSION_START])

    async def execute(self, input):  # noqa: A002 - SDK signature
        return HookOutput()


class _LifecycleSleepHook:
    """A distinct sleep-hook instance, tracked by identity in agent.sleep_hooks."""


class _FullFeature(Feature):
    """A feature that registers a signal source, a wait provider, a hook, and a
    tool — i.e. every kind of registration the canonical teardown must reverse.
    ``initialize()`` is idempotent so re-activation can re-run it (#2522)."""

    SOURCE = "lifecycle.job_complete"

    tool_name = "full_feature"
    tool_description = "feature exercising the full runtime lifecycle"

    def __init__(self, agent):
        super().__init__(agent)
        self._hook = _LifecycleHook()
        self._sleep_hook = _LifecycleSleepHook()
        self.on_disable_should_raise = False
        self.ready_calls = 0
        self.ready_should_raise = False
        # Owning a permanent background task is opt-in: it's exercised only by
        # the direct-call (single-loop) tests. The endpoint test drives disable
        # through Starlette's TestClient, which runs the handler in a SEPARATE
        # event loop — cancelling a task created in the pytest loop from there
        # is a test-only cross-loop artifact that never happens in production
        # (disable runs in the agent's serving loop, same loop as the task).
        self.own_background_task = False

    @property
    def promote_tools_on_startup(self) -> bool:
        # Promote the feature's direct tool at boot so teardown has a dynamic
        # tool to detach and activation has one to re-register.
        return True

    async def initialize(self):
        from kestrel_sovereign.signals import RegistrationPolicy

        registry = getattr(self.agent, "signal_registry", None)
        if registry is not None:
            # OPTIONAL policy is idempotent on a second initialize().
            self._register_signal_sources(
                _fake_source_registration(self.SOURCE),
                RegistrationPolicy.OPTIONAL,
            )

    def get_hooks(self):
        return [self._hook]

    async def on_disable(self):
        if self.on_disable_should_raise:
            raise RuntimeError("on_disable boom")

    async def _sweep_forever(self):
        # Stand-in for Peers' hourly expiry sweep: a permanent tracked loop that
        # must be cancelled by feature teardown, not just at full agent shutdown.
        while True:
            await asyncio.sleep(3600)

    async def post_all_features_loaded(self, agent):
        registry = getattr(agent, "wait_registry", None)
        if registry is not None:
            self._register_wait_provider(
                registry, _FullFeatureWaitable(), replace=True
            )
        # A permanent feature-owned background task (Peers analogue) and a sleep
        # hook (Memory analogue) — both must be torn down on disable / rollback.
        if self.own_background_task:
            self._track_owned_background_task(
                self._sweep_forever(), name="lifecycle_sweep"
            )
        self._register_sleep_hook(agent, self._sleep_hook)

    async def on_agent_ready(self, agent):
        # Ready-phase hook (RestartCoordinator analogue). Boot fires this after
        # services are live; runtime re-enable must fire it too (#2522 P2).
        self.ready_calls += 1
        if self.ready_should_raise:
            raise RuntimeError("on_agent_ready boom")

    @tool("full_do", "does a full-feature thing", ToolCategory.SYSTEM)
    async def full_do(self):  # pragma: no cover - never executed here
        return "done"


class _RuntimeDeniedFeature(Feature):
    """Contributes a hard default only when its runtime lifecycle is active."""

    tool_name = "runtime_denied_feature"
    tool_description = "feature exercising runtime permission registration"

    async def initialize(self):
        return None

    def get_feature_permission_defaults(self):
        return FeaturePermissionDefaults(
            feature_default=SDKPermissionLevel.DENY,
        )

    @tool("runtime_guarded", "guarded at runtime", ToolCategory.SYSTEM)
    async def runtime_guarded(self):  # pragma: no cover - never executed here
        return "blocked"


class _FullFeatureWaitable:
    kind = "lifecycle"
    signal = None

    async def poll(self, handle):  # pragma: no cover - never polled here
        raise NotImplementedError


def _agent(tmp_path) -> KestrelAgent:
    agent = KestrelAgent(
        did="did:test:lifecycle", storage_path=str(tmp_path / "lc.db")
    )
    agent.task_manager = _RecordingTaskManager()
    agent.signal_registry = SourceRegistry()
    agent.wait_registry = WaitRegistry()
    agent.features = {}
    return agent


class _RecordingTaskManager:
    """Minimal A2A TaskManager double recording agent (un)registration."""

    def __init__(self) -> None:
        self.agents: dict = {}

    def register_agent(self, *, agent_card, handler, command_prefixes):
        self.agents[agent_card.name] = handler

    def unregister_agent(self, name):
        self.agents.pop(name, None)


async def _boot_feature(agent, feature) -> None:
    """Register a feature exactly as boot does: per-feature registration then
    the post-load cross-feature wiring + startup-tool promotion."""
    await agent._register_feature(feature)
    await feature.post_all_features_loaded(agent)
    agent._promote_startup_feature_tools()


def _live_registrations(agent, feature):
    """Snapshot of the loop-safe registries the lifecycle touches.

    Background-task liveness is asserted separately in the single-loop tests —
    it can't ride through the endpoint test because Starlette's TestClient runs
    the handler in its own event loop (see ``_FullFeature.own_background_task``).
    """
    return {
        "source": _FullFeature.SOURCE in agent.signal_registry,
        "wait": agent.wait_registry.get("lifecycle") is not None,
        "hook": feature._hook
        in agent.hooks_manager.get_hooks(HookEvent.SESSION_START),
        "a2a": feature.tool_name in agent.task_manager.agents,
        "tool": any(
            owner == feature.tool_name
            for owner in agent._tool_to_feature.values()
        ),
        # Feature-owned sleep hook is wired on the agent (#2522 P1).
        "sleep_hook": feature._sleep_hook in getattr(agent, "sleep_hooks", []),
        "loaded": agent.features.get(feature.name) is feature,
    }


# ---------------------------------------------------------------------------
# P1 #1 — production path through the public endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_disable_detaches_everything_then_enable_recreates(tmp_path):
    agent = _agent(tmp_path)
    feature = _FullFeature(agent)
    await _boot_feature(agent, feature)

    # Sanity: a booted feature owns one of every registration.
    live = _live_registrations(agent, feature)
    assert all(live.values()), f"feature not fully live after boot: {live}"

    app = FastAPI()
    app.include_router(features_router)
    app.state.agent = agent

    with TestClient(app) as client:
        disable = client.post("/api/features/_FullFeature/disable")
    assert disable.status_code == 200
    assert disable.json()["status"] == "disabled"

    # Everything the feature owned is detached — including the wait provider and
    # dispatcher source the OLD light endpoint left live (#2522).
    after_disable = _live_registrations(agent, feature)
    assert after_disable["source"] is False
    assert after_disable["wait"] is False
    assert after_disable["hook"] is False
    assert after_disable["a2a"] is False
    assert after_disable["tool"] is False
    # The sleep hook the OLD teardown never touched is detached too (#2522 P1).
    assert after_disable["sleep_hook"] is False
    assert feature._sleep_hook not in agent.sleep_hooks
    # Soft-toggle: the SAME instance stays loaded so it can be re-enabled.
    assert after_disable["loaded"] is True
    assert feature.enabled is False

    with TestClient(app) as client:
        enable = client.post("/api/features/_FullFeature/enable")
    assert enable.status_code == 200
    assert enable.json()["status"] == "enabled"

    # Enable recreated every registration on the SAME instance.
    reenabled = _live_registrations(agent, feature)
    assert all(reenabled.values()), f"enable did not fully recreate: {reenabled}"
    assert feature.enabled is True
    assert agent.features["_FullFeature"] is feature


# ---------------------------------------------------------------------------
# P1 #2 — unconditional cleanup even when on_disable raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregister_runtime_cleans_everything_when_on_disable_raises(tmp_path):
    agent = _agent(tmp_path)
    feature = _FullFeature(agent)
    await _boot_feature(agent, feature)
    assert all(_live_registrations(agent, feature).values())

    feature.on_disable_should_raise = True

    # The failure is surfaced AFTER cleanup (full unload path).
    with pytest.raises(RuntimeError, match="on_disable boom"):
        await agent._unregister_feature_runtime(feature)

    # No provider / source / hook / A2A / tool / feature remains despite the
    # on_disable failure (#2522 P2).
    assert _FullFeature.SOURCE not in agent.signal_registry
    assert agent.wait_registry.get("lifecycle") is None
    assert feature._hook not in agent.hooks_manager.get_hooks(
        HookEvent.SESSION_START
    )
    assert feature.tool_name not in agent.task_manager.agents
    assert not any(
        owner == feature.tool_name for owner in agent._tool_to_feature.values()
    )
    assert feature.name not in agent.features


@pytest.mark.asyncio
async def test_unregister_runtime_cleans_everything_when_on_disable_cancels(tmp_path):
    """A hook-owned cancelled child is an operation failure, so canonical
    teardown still drains every independent registration before it propagates."""
    agent = _agent(tmp_path)
    feature = _FullFeature(agent)
    await _boot_feature(agent, feature)
    assert all(_live_registrations(agent, feature).values())

    async def cancel_from_child():
        child = asyncio.create_task(asyncio.sleep(3600))
        child.cancel()
        await child

    feature.on_disable = cancel_from_child

    with pytest.raises(asyncio.CancelledError):
        await agent._unregister_feature_runtime(feature)

    assert _FullFeature.SOURCE not in agent.signal_registry
    assert agent.wait_registry.get("lifecycle") is None
    assert feature._hook not in agent.hooks_manager.get_hooks(
        HookEvent.SESSION_START
    )
    assert feature.tool_name not in agent.task_manager.agents
    assert not any(
        owner == feature.tool_name for owner in agent._tool_to_feature.values()
    )
    assert feature.name not in agent.features


@pytest.mark.asyncio
async def test_soft_disable_cleans_everything_when_on_disable_raises(tmp_path):
    """Same guarantee on the soft-toggle path: the instance stays loaded but
    every OTHER registration is still detached, then the error surfaces."""
    agent = _agent(tmp_path)
    feature = _FullFeature(agent)
    await _boot_feature(agent, feature)
    feature.on_disable_should_raise = True

    with pytest.raises(RuntimeError, match="on_disable boom"):
        await agent._unregister_feature_runtime(feature, unload=False)

    assert _FullFeature.SOURCE not in agent.signal_registry
    assert agent.wait_registry.get("lifecycle") is None
    assert feature._hook not in agent.hooks_manager.get_hooks(
        HookEvent.SESSION_START
    )
    assert feature.tool_name not in agent.task_manager.agents
    assert not any(
        owner == feature.tool_name for owner in agent._tool_to_feature.values()
    )
    # Soft-toggle keeps it loaded + marked disabled.
    assert agent.features.get(feature.name) is feature
    assert feature.enabled is False


# ---------------------------------------------------------------------------
# P1 #3 — identity-aware wait-provider ownership at the Feature level
# ---------------------------------------------------------------------------


class _OwnerFeature(Feature):
    tool_name = "owner_feature"
    tool_description = "feature that takes over a wait-provider slot"

    async def initialize(self):
        pass

    def register(self, registry, provider):
        self._register_wait_provider(registry, provider, replace=True)


def _named(kind, tag):
    class _P:
        pass

    p = _P()
    p.kind = kind
    p.tag = tag
    p.signal = None

    async def poll(handle):  # pragma: no cover
        raise NotImplementedError

    p.poll = poll
    return p


@pytest.mark.asyncio
async def test_shutdown_restores_displaced_host_provider():
    registry = WaitRegistry()
    host = _named("task", "host")
    registry.register(host)  # host owns the slot first

    agent = type("A", (), {"wait_registry": registry})()
    feature = _OwnerFeature(agent)
    mine = _named("task", "feature")
    feature.register(registry, mine)
    assert registry.get("task") is mine

    # Feature shuts down while ITS provider is still current → host is restored.
    await feature.shutdown()
    assert registry.get("task") is host


@pytest.mark.asyncio
async def test_shutdown_removes_slot_when_it_was_empty():
    registry = WaitRegistry()
    agent = type("A", (), {"wait_registry": registry})()
    feature = _OwnerFeature(agent)
    mine = _named("task", "feature")
    feature.register(registry, mine)  # installed into an empty slot
    assert registry.get("task") is mine

    await feature.shutdown()
    assert registry.get("task") is None
    assert registry.kinds() == []


@pytest.mark.asyncio
async def test_shutdown_leaves_newer_provider_untouched():
    registry = WaitRegistry()
    host = _named("task", "host")
    registry.register(host)

    agent = type("A", (), {"wait_registry": registry})()
    feature = _OwnerFeature(agent)
    mine = _named("task", "feature")
    feature.register(registry, mine)

    # A newer owner replaces the feature's provider AFTER it registered.
    newer = _named("task", "newer")
    registry.register(newer, replace=True)
    assert registry.get("task") is newer

    # Feature shutdown must NOT evict the newer provider nor resurrect host.
    await feature.shutdown()
    assert registry.get("task") is newer


@pytest.mark.asyncio
async def test_stacked_features_disable_A_then_B_restores_host_not_disabled_A():
    """host → feature A → feature B; disable A then B must restore the HOST.

    The reviewer's exact scenario (#2522 P3): with the old single-slot design,
    feature B saved feature A as its ``previous`` at registration time, so
    tearing B down restored A — resurrecting a feature the operator had already
    disabled. The per-kind ownership stack removes A when A is disabled, so B's
    teardown can only fall through to the nearest STILL-LIVE predecessor (host).
    """
    registry = WaitRegistry()
    host = _named("task", "host")
    registry.register(host)  # host owns the slot first

    agent = type("A", (), {"wait_registry": registry})()

    feature_a = _OwnerFeature(agent)
    prov_a = _named("task", "A")
    feature_a.register(registry, prov_a)  # A replaces host

    feature_b = _OwnerFeature(agent)
    prov_b = _named("task", "B")
    feature_b.register(registry, prov_b)  # B replaces A
    assert registry.get("task") is prov_b

    # Disable A first — A is the MIDDLE of the stack (B is current). B stays.
    await feature_a.shutdown()
    assert registry.get("task") is prov_b

    # Disable B — the nearest still-live predecessor is the host, never the
    # already-disabled A.
    await feature_b.shutdown()
    assert registry.get("task") is host


@pytest.mark.asyncio
async def test_stacked_features_disable_B_then_A_walks_down_to_host():
    """Reverse teardown order still walks host ← A ← B cleanly."""
    registry = WaitRegistry()
    host = _named("task", "host")
    registry.register(host)
    agent = type("A", (), {"wait_registry": registry})()

    feature_a = _OwnerFeature(agent)
    prov_a = _named("task", "A")
    feature_a.register(registry, prov_a)
    feature_b = _OwnerFeature(agent)
    prov_b = _named("task", "B")
    feature_b.register(registry, prov_b)

    await feature_b.shutdown()
    assert registry.get("task") is prov_a
    await feature_a.shutdown()
    assert registry.get("task") is host


# ---------------------------------------------------------------------------
# P1 (Terra follow-up #1) — a soft-disabled feature is invisible to the LLM
# AND unresolvable by name; enable makes it reappear.
# ---------------------------------------------------------------------------


class _DispatchFeature(Feature):
    """A subagent-dispatchable feature that does NOT promote its tools at
    startup, so its dispatcher tool stays in ``_build_feature_tools()`` (an
    explored feature's dispatcher is suppressed in favour of its direct tools —
    orthogonal to the enabled gate we are exercising here)."""

    tool_name = "dispatch_feature"
    tool_description = "dispatchable feature for the visibility gate"

    async def initialize(self):
        pass

    @tool("dispatch_do", "does a dispatch thing", ToolCategory.SYSTEM)
    async def dispatch_do(self):  # pragma: no cover - never executed here
        return "done"


def _llm_visibility(agent, feature):
    """What the orchestrator LLM / external transports can see for ``feature``."""
    dispatcher_names = {
        tool_def["function"]["name"]
        for tool_def in agent._build_feature_tools()
    }
    resolved_tool, resolved_feature = agent._resolve_named_tool("dispatch_do")
    return {
        # Dispatcher tool advertised to the orchestrator LLM.
        "dispatcher_tool": feature.tool_name in dispatcher_names,
        # Subagent dispatch map (chat path + _resolve_named_subagent).
        "subagent_map": feature.tool_name in agent._visible_features_by_tool_name(),
        # Direct @tool resolution used by execute_named_tool / external transports.
        "named_tool": resolved_tool is not None and resolved_feature is feature,
        # Feature advertised in the system prompt section.
        "prompt": feature.name in agent._build_features_prompt_section(),
    }


@pytest.mark.asyncio
async def test_disabled_feature_is_invisible_to_llm_then_enable_restores(tmp_path):
    agent = _agent(tmp_path)
    feature = _DispatchFeature(agent)
    # Register exactly as boot does (no startup tool promotion for this one).
    await agent._register_feature(feature)
    await feature.post_all_features_loaded(agent)

    # Enabled: dispatcher tool, subagent map, named-tool resolution, and prompt
    # all see the feature.
    live = _llm_visibility(agent, feature)
    assert all(live.values()), f"feature not fully visible while enabled: {live}"

    app = FastAPI()
    app.include_router(features_router)
    app.state.agent = agent

    with TestClient(app) as client:
        assert client.post("/api/features/_DispatchFeature/disable").status_code == 200

    # Soft-disabled: the instance is still LOADED, but NOTHING resolves it as a
    # callable tool — dispatcher, subagent map, direct name, and prompt are all
    # gated on ``enabled`` (#2522 P1).
    assert agent.features.get("_DispatchFeature") is feature
    assert feature.enabled is False
    hidden = _llm_visibility(agent, feature)
    assert hidden == {
        "dispatcher_tool": False,
        "subagent_map": False,
        "named_tool": False,
        "prompt": False,
    }, hidden

    with TestClient(app) as client:
        assert client.post("/api/features/_DispatchFeature/enable").status_code == 200

    # Re-enabled: every surface sees the feature again.
    restored = _llm_visibility(agent, feature)
    assert all(restored.values()), f"enable did not restore visibility: {restored}"
    assert feature.enabled is True


# ---------------------------------------------------------------------------
# P1 (Terra follow-up #2) — feature-owned background tasks + sleep hooks are
# reversed by boot rollback and soft disable.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boot_rollback_after_post_load_cancels_owned_background_task(tmp_path):
    """A boot failure AFTER ``post_all_features_loaded`` triggers the feature
    rollback (``_boot_teardown_features``) — the exact ``ctx.on_rollback`` boot
    registers before the post-load wiring — which must cancel the permanent
    feature-owned sweep task, leaving NO task behind (#2522 P1)."""
    agent = _agent(tmp_path)
    feature = _FullFeature(agent)
    feature.own_background_task = True
    await _boot_feature(agent, feature)

    owned = list(feature._owned_background_tasks)
    assert owned, "feature should own its permanent sweep task after post-load"
    assert all(not task.done() for task in owned)
    assert all(task in agent._background_tasks for task in owned)

    # Drive the REAL boot rollback path a post-load failure would trigger.
    await agent._boot_teardown_features()

    # No task leaks: the owned task is cancelled, dropped from the feature's
    # ownership list AND from the agent's global reap set.
    for task in owned:
        assert task.done()
    assert feature._owned_background_tasks == []
    assert all(task not in agent._background_tasks for task in owned)
    # Full unload: boot rollback drops the feature entirely.
    assert agent.features == {}


@pytest.mark.asyncio
async def test_soft_disable_cancels_then_reenable_restarts_owned_task(tmp_path):
    """Runtime soft disable cancels the feature-owned task; re-enable starts a
    fresh one — all in one event loop, exactly as the agent's serving loop does
    (#2522 P1)."""
    agent = _agent(tmp_path)
    feature = _FullFeature(agent)
    feature.own_background_task = True
    await _boot_feature(agent, feature)

    first = list(feature._owned_background_tasks)
    assert first and all(not task.done() for task in first)

    # Soft disable (unload=False) — the runtime-disable teardown.
    await agent._unregister_feature_runtime(feature, unload=False)
    for task in first:
        assert task.done()
    assert feature._owned_background_tasks == []
    assert all(task not in agent._background_tasks for task in first)

    # Re-enable starts a brand-new tracked task.
    await agent._activate_feature_runtime(feature)
    second = list(feature._owned_background_tasks)
    assert second and all(not task.done() for task in second)
    assert all(task in agent._background_tasks for task in second)
    assert set(second).isdisjoint(first)

    # Cleanup so the loop doesn't leak the sweep task past the test.
    await agent._unregister_feature_runtime(feature, unload=True)


@pytest.mark.asyncio
async def test_disabling_memory_removes_reflection_sleep_hook():
    """Disabling MemoryFeature via the canonical teardown removes exactly the
    ReflectionSleepHook it appended to ``agent.sleep_hooks`` — and never a
    foreign hook (#2522 P1)."""
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.memory.feature import MemoryFeature
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    foreign_hook = object()  # another feature's / host's hook — must survive
    agent = MagicMock()
    agent.sleep_hooks = [foreign_hook]
    feature = MemoryFeature(agent)

    await feature.post_all_features_loaded(agent)

    reflection = [h for h in agent.sleep_hooks if isinstance(h, ReflectionSleepHook)]
    assert len(reflection) == 1, "Memory should register exactly one sleep hook"
    assert reflection[0] in feature._owned_sleep_hooks

    # Idempotent re-run (e.g. re-enable) must not stack a second hook.
    await feature.post_all_features_loaded(agent)
    assert [h for h in agent.sleep_hooks if isinstance(h, ReflectionSleepHook)] == reflection

    # Canonical teardown (what runtime disable / boot rollback call).
    await feature.shutdown()

    assert not any(isinstance(h, ReflectionSleepHook) for h in agent.sleep_hooks)
    assert feature._owned_sleep_hooks == []
    # The foreign hook is untouched.
    assert foreign_hook in agent.sleep_hooks
    # Idempotent: a second shutdown is a benign no-op.
    await feature.shutdown()
    assert foreign_hook in agent.sleep_hooks


# ---------------------------------------------------------------------------
# P2 (Terra follow-up #3) — runtime re-enable runs the ready-phase lifecycle.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reenable_fires_on_agent_ready(tmp_path):
    """``_activate_feature_runtime`` must invoke ``on_agent_ready`` after a
    successful activation, mirroring boot's ready phase (#2522 P2). Boot's
    ``_register_feature`` does NOT fire it (readiness is a later phase), so the
    counter proves it fires specifically on the re-enable path."""
    agent = _agent(tmp_path)
    feature = _FullFeature(agent)
    await _boot_feature(agent, feature)
    # Booting never runs the ready phase for this helper.
    assert feature.ready_calls == 0

    await agent._unregister_feature_runtime(feature, unload=False)
    assert feature.enabled is False

    await agent._activate_feature_runtime(feature)
    assert feature.enabled is True
    assert feature.ready_calls == 1


@pytest.mark.asyncio
async def test_reenable_on_agent_ready_failure_is_non_fatal(tmp_path):
    """The ready hook is best-effort (boot policy): a failing ``on_agent_ready``
    logs but must NOT roll back an already-live feature (#2522 P2)."""
    agent = _agent(tmp_path)
    feature = _FullFeature(agent)
    await _boot_feature(agent, feature)
    await agent._unregister_feature_runtime(feature, unload=False)

    feature.ready_should_raise = True
    # No exception surfaces despite on_agent_ready raising.
    await agent._activate_feature_runtime(feature)

    # Feature is fully live — ready-hook failure did not tear it down.
    assert feature.enabled is True
    assert feature.ready_calls == 1
    live = _live_registrations(agent, feature)
    assert all(live.values()), f"ready-hook failure wrongly rolled back: {live}"


@pytest.mark.asyncio
async def test_boot_ready_hook_waits_until_host_context_is_published(tmp_path):
    """A readiness hook may await cognition without circular startup waiting."""

    agent = _agent(tmp_path)
    gate = asyncio.Event()
    agent._host_context_publication_gate = gate
    entered = asyncio.Event()

    class CognitionReadyFeature(_FullFeature):
        async def on_agent_ready(self, ready_agent):
            async with ready_agent._turn_lifecycle():
                entered.set()

    feature = CognitionReadyFeature(agent)
    agent.features[feature.name] = feature

    await asyncio.wait_for(agent._run_or_defer_agent_ready_hooks(), timeout=1)
    assert not entered.is_set()

    gate.set()
    await asyncio.wait_for(agent.complete_deferred_agent_readiness(), timeout=1)
    assert entered.is_set()


@pytest.mark.asyncio
async def test_deferred_readiness_completion_is_exactly_once(tmp_path):
    """Registration and the server snapshot may race the same ready pass."""

    agent = _agent(tmp_path)
    gate = asyncio.Event()
    gate.set()
    agent._host_context_publication_gate = gate
    agent._agent_ready_hooks_deferred = True
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class BlockingReadyFeature(_FullFeature):
        async def on_agent_ready(self, ready_agent):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()

    feature = BlockingReadyFeature(agent)
    agent.features[feature.name] = feature

    first = asyncio.create_task(agent.complete_deferred_agent_readiness())
    await asyncio.wait_for(started.wait(), timeout=1)
    second = asyncio.create_task(agent.complete_deferred_agent_readiness())
    await asyncio.sleep(0)
    assert calls == 1

    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

    assert calls == 1
    assert agent._agent_ready_hooks_deferred is False


@pytest.mark.asyncio
async def test_runtime_enable_registers_contributed_hard_permission_immediately(
    tmp_path,
):
    """Runtime activation consumes the newly-live declaration before exposure."""
    agent = _agent(tmp_path)
    security = SecurityFeature(agent)
    await security.initialize()
    agent.features[security.name] = security

    feature = _RuntimeDeniedFeature(agent)
    feature.enabled = False
    agent.features[feature.name] = feature
    await security.permission_store.register_tool(
        feature.name,
        "runtime_guarded",
        PermissionLevel.ALLOW,
    )

    await agent._activate_feature_runtime(feature)

    assert feature.enabled is True
    assert (
        await security.permission_store.get_permission(
            feature.name,
            "runtime_guarded",
        )
        is PermissionLevel.DENY
    )


@pytest.mark.asyncio
async def test_runtime_permission_registration_failure_rolls_back_activation(
    tmp_path,
    monkeypatch,
):
    """Permission registration is an atomic activation gate, not an afterthought."""
    agent = _agent(tmp_path)
    security = SecurityFeature(agent)
    await security.initialize()
    agent.features[security.name] = security

    feature = _RuntimeDeniedFeature(agent)
    feature.enabled = False
    agent.features[feature.name] = feature

    async def fail_registration(*args, **kwargs):
        raise RuntimeError("permission registration failed")

    monkeypatch.setattr(
        security,
        "register_feature_tools",
        fail_registration,
    )

    with pytest.raises(RuntimeError, match="permission registration failed"):
        await agent._activate_feature_runtime(feature)

    assert feature.enabled is False
    assert not agent.feature_contribution_runtime.is_active(feature)
    assert feature.tool_name not in agent.task_manager.agents


# ---------------------------------------------------------------------------
# P1 (Terra follow-up) — feature REMOVAL routes through the canonical teardown.
# The old /remove path unregistered only hooks + set enabled=False, leaving
# signal sources, wait providers, the A2A agent, owned tasks/sleep hooks, AND
# the promoted direct tools live; the direct tools stayed EXECUTABLE because
# resolution gates on the feature's ``enabled`` flag, not on the tool map
# (#2522 P1).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_remove_drains_every_registration_and_direct_tool(tmp_path):
    """Removal drains EVERY registration (source, wait provider, hook, A2A agent,
    promoted direct tool, sleep hook), stops the @tool methods resolving, unloads
    the instance, and runs ``on_remove`` only AFTER the runtime is fully drained
    (#2522 P1)."""
    from unittest.mock import MagicMock, patch

    from kestrel_sovereign.feature_registry import FeaturePackageInfo

    agent = _agent(tmp_path)
    feature = _FullFeature(agent)
    await _boot_feature(agent, feature)

    live = _live_registrations(agent, feature)
    assert all(live.values()), f"feature not fully live after boot: {live}"
    # The promoted direct tool resolves by name before removal — proving the
    # feature's @tool methods are callable while loaded.
    resolved_tool, resolved_feature = agent._resolve_named_tool("full_do")
    assert resolved_tool is not None and resolved_feature is feature

    # Capture the live-registration snapshot AT on_remove time so we can prove
    # on_remove runs after — not before — the runtime drained.
    on_remove_snapshots: list[dict] = []

    async def _record_on_remove():
        on_remove_snapshots.append(_live_registrations(agent, feature))

    feature.on_remove = _record_on_remove

    pkg = FeaturePackageInfo(
        name="full-pkg", package="kestrel-feature-full", git="",
        features=["_FullFeature"], description="full", core=False,
    )

    app = FastAPI()
    app.include_router(features_router)
    app.state.agent = agent

    with patch(
        "kestrel_sovereign.endpoints.features.get_package_for_feature",
        return_value=pkg,
    ), patch(
        "kestrel_sovereign.endpoints.features.subprocess.run",
        return_value=MagicMock(returncode=0, stdout="", stderr=""),
    ):
        with TestClient(app) as client:
            resp = client.post("/api/features/_FullFeature/remove")

    assert resp.status_code == 200
    assert resp.json()["status"] == "removed"

    # Every runtime registration the OLD hooks-only path left live is drained.
    assert _FullFeature.SOURCE not in agent.signal_registry
    assert agent.wait_registry.get("lifecycle") is None
    assert feature._hook not in agent.hooks_manager.get_hooks(HookEvent.SESSION_START)
    assert feature.tool_name not in agent.task_manager.agents
    assert feature._sleep_hook not in agent.sleep_hooks
    # The promoted direct tool is gone from the map AND no longer resolves, so a
    # removed feature's @tool methods can never be executed (the exact bug: the
    # map, not ``enabled``, gates execution).
    assert not any(
        owner == feature.tool_name for owner in agent._tool_to_feature.values()
    )
    assert agent._resolve_named_tool("full_do") == (None, None)
    # The instance is fully unloaded, and pip uninstall ran.
    assert "_FullFeature" not in agent.features

    # on_remove ran exactly once, and the runtime was ALREADY fully drained when
    # it did — so unloading never robbed on_remove of live feature state.
    assert len(on_remove_snapshots) == 1
    assert not any(on_remove_snapshots[0].values()), (
        "on_remove ran before the canonical teardown drained the runtime: "
        f"{on_remove_snapshots[0]}"
    )


# ---------------------------------------------------------------------------
# P1 (Terra follow-up) — a boot registration that fails AFTER a partial wiring
# must drain the partial registration, not strand it. ``_register_feature`` wires
# a feature in stages (initialize → hooks → on_enable → A2A); a failure in a late
# stage used to leave the earlier stages' registrations live AND drop the feature
# from ``self.features`` so boot rollback could no longer find it (#2522 P1).
# ---------------------------------------------------------------------------


class _OnEnableFailFeature(_FullFeature):
    """Registration fails at ``on_enable`` — AFTER ``initialize`` registered the
    signal source and ``_wire_feature_hooks`` registered the hook."""

    tool_name = "on_enable_fail_feature"

    async def on_enable(self):
        raise RuntimeError("on_enable boom (post-hook)")


class _A2AFailFeature(_FullFeature):
    """Registration fails INSIDE ``_wire_feature_a2a`` AFTER ``register_agent``
    already recorded the A2A agent — ``set_task_manager`` raises."""

    tool_name = "a2a_fail_feature"

    def set_task_manager(self, task_manager):
        raise RuntimeError("a2a wiring boom (post-register)")


def _no_registration_survives(agent, feature) -> None:
    """Assert NOTHING the failed registration touched is left behind."""
    assert _FullFeature.SOURCE not in agent.signal_registry
    assert agent.wait_registry.get("lifecycle") is None
    assert feature._hook not in agent.hooks_manager.get_hooks(HookEvent.SESSION_START)
    assert feature.tool_name not in agent.task_manager.agents
    assert not any(
        owner == feature.tool_name for owner in agent._tool_to_feature.values()
    )
    assert feature.name not in agent.features


@pytest.mark.asyncio
async def test_register_feature_on_enable_failure_strands_nothing(tmp_path):
    """``_register_feature`` fails at ``on_enable`` with the feature's hook and
    signal source already registered. The failed-feature rollback must drain BOTH
    — the old path called only ``feature.shutdown()`` (which reverses the signal
    source but NOT the agent-side hook), stranding the hook on a dead agent — and
    drop the feature so boot rollback isn't left a phantom (#2522 P1)."""
    agent = _agent(tmp_path)
    feature = _OnEnableFailFeature(agent)

    # Sanity: the original registration error surfaces (not a teardown error).
    with pytest.raises(RuntimeError, match="on_enable boom"):
        await agent._register_feature(feature)

    _no_registration_survives(agent, feature)


@pytest.mark.asyncio
async def test_register_feature_a2a_failure_strands_nothing(tmp_path):
    """``_register_feature`` fails INSIDE ``_wire_feature_a2a`` AFTER the A2A
    agent was registered, with the hook and signal source live too. The rollback
    must undo the A2A registration, the hook, AND the source (the old
    ``shutdown()``-only path undid none of the agent-side wiring) and drop the
    feature (#2522 P1)."""
    agent = _agent(tmp_path)
    feature = _A2AFailFeature(agent)

    # Sanity: register_agent recorded the A2A agent before set_task_manager blew
    # up — so this genuinely exercises the "failed AFTER registration" path.
    original_register = agent.task_manager.register_agent
    registered_names: list[str] = []

    def _spy_register(*, agent_card, handler, command_prefixes):
        registered_names.append(agent_card.name)
        return original_register(
            agent_card=agent_card, handler=handler, command_prefixes=command_prefixes
        )

    agent.task_manager.register_agent = _spy_register

    with pytest.raises(RuntimeError, match="a2a wiring boom"):
        await agent._register_feature(feature)

    assert registered_names == ["a2a_fail_feature"], (
        "A2A agent must have been registered before the failure for this test to "
        "exercise the post-registration strand"
    )
    _no_registration_survives(agent, feature)


# --- the ONE door a feature registers signal sources through (#3074) --------
#
# Ownership is stated at registration now, so `owner=None` means the host and
# nothing infers a holder from the policy. These pin the two halves the
# registry cannot see: what a feature does with a registry that predates the
# claims ledger, and what it does when it registers a source and then refuses
# it.


class _NoClaimsRegistry:
    """An embedder-supplied registry from before the claims ledger.

    No ``release_all``, so :meth:`Feature._registry_holds_claims` is False and
    the base class must track names itself — otherwise this feature's sources
    are never torn down at all, which is worse than the ledger it replaced.
    Deliberately REJECTS ``owner=``: a registry that cannot hold claims has no
    parameter for one, and passing it anyway is the failure this branch exists
    to avoid.
    """

    def __init__(self):
        self.sources = {}
        self.unregistered = []

    def register_with_policy(self, registration, policy):
        from kestrel_sovereign.signals import RegistrationOutcome, RegistrationState

        if registration.name in self.sources:
            return RegistrationOutcome(
                registration.name, RegistrationState.ALREADY_EQUIVALENT
            )
        self.sources[registration.name] = registration
        return RegistrationOutcome(registration.name, RegistrationState.REGISTERED)

    def unregister(self, name):
        self.unregistered.append(name)
        return self.sources.pop(name, None) is not None


class _MinimalFeature(Feature):
    tool_name = "minimal_owner"
    tool_description = "feature used to exercise the registration seam"

    async def initialize(self):
        # The seam is driven directly by these tests, so boot does nothing.
        return None


def _seam_feature(registry):
    return _MinimalFeature(types.SimpleNamespace(signal_registry=registry))


async def test_a_registry_without_the_claims_api_is_torn_down_by_name():
    """A registry that cannot hold a claim still has to be cleaned up.

    The names are tracked on the feature for THIS case only. It is not the
    second ledger #3053 removed: there is one record per registry, and which
    one is decided by what the registry can actually do.
    """
    from kestrel_sovereign.signals import RegistrationPolicy

    registry = _NoClaimsRegistry()
    feature = _seam_feature(registry)

    feature._register_signal_sources(
        _fake_source_registration("seam.byname"), RegistrationPolicy.OPTIONAL,
    )
    assert "seam.byname" in registry.sources

    await feature._unregister_owned_signal_sources()

    assert registry.unregistered == ["seam.byname"]
    assert "seam.byname" not in registry.sources


async def test_a_registry_without_the_claims_api_keeps_what_it_did_not_create():
    """Riding an incumbent is not owning it.

    This path has no claims to express shared use, so a source it merely found
    already registered must not be removed by its teardown — the peer that
    created it is still dispatching on it.
    """
    from kestrel_sovereign.signals import RegistrationPolicy

    registry = _NoClaimsRegistry()
    incumbent = _seam_feature(registry)
    incumbent._register_signal_sources(
        _fake_source_registration("seam.shared"), RegistrationPolicy.OPTIONAL,
    )
    rider = _seam_feature(registry)
    rider._register_signal_sources(
        _fake_source_registration("seam.shared"), RegistrationPolicy.OPTIONAL,
    )

    await rider._unregister_owned_signal_sources()

    assert registry.unregistered == []
    assert "seam.shared" in registry.sources


async def test_a_feature_that_registers_a_source_and_refuses_it_hands_the_claim_back():
    """Registering and declining are two halves of one door.

    The claim is taken AT registration now, so a feature that verifies the
    installed contract afterwards and rejects it has to give the claim back —
    and giving back the last claim removes the source, which is the right
    answer: nothing needs a source its only holder refused.
    """
    from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry

    registry = SourceRegistry()
    feature = _seam_feature(registry)

    outcomes = feature._register_signal_sources(
        _fake_source_registration("seam.refused"), RegistrationPolicy.OPTIONAL,
    )
    assert registry.get("seam.refused") is not None

    feature._disown_signal_sources(outcomes)

    assert registry.get("seam.refused") is None
    assert registry.owners_of("seam.refused") == ()


async def test_disowning_a_source_a_peer_still_holds_leaves_it_registered():
    """The invariant is per-holder, not per-name.

    A feature that rode an equivalent incumbent and then refused it drops its
    own claim only; the peer that created it keeps the source alive.
    """
    from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry

    registry = SourceRegistry()
    incumbent = _seam_feature(registry)
    rider = _seam_feature(registry)
    incumbent._register_signal_sources(
        _fake_source_registration("seam.peer"), RegistrationPolicy.OPTIONAL,
    )
    outcomes = rider._register_signal_sources(
        _fake_source_registration("seam.peer"), RegistrationPolicy.OPTIONAL,
    )

    rider._disown_signal_sources(outcomes)

    assert registry.get("seam.peer") is not None
    assert registry.owners_of("seam.peer") == (incumbent,)


class _RolelessClaimsRegistry:
    """A registry from the generation BETWEEN the two: claims, but no role.

    `release_all` and `owner=` shipped together; `role=` came a commit later.
    An embedder written against that first shape passes the "can it hold
    claims" question and then raises `TypeError` on the keyword, mid
    `initialize()` — which is why the capability question has to ask about the
    whole contract, not the half that happens to be older.
    """

    def __init__(self):
        self.sources = {}
        self.claims = {}
        self.unregistered = []

    def register_with_policy(self, registration, policy, *, owner=None):
        from kestrel_sovereign.signals import RegistrationOutcome, RegistrationState

        if registration.name in self.sources:
            return RegistrationOutcome(
                registration.name, RegistrationState.ALREADY_EQUIVALENT
            )
        self.sources[registration.name] = registration
        self.claims.setdefault(registration.name, []).append(owner)
        return RegistrationOutcome(registration.name, RegistrationState.REGISTERED)

    def release_all(self, owner, role=None):
        raise AssertionError(
            "teardown must not reach a registry the register call could not "
            "claim through"
        )

    def unregister(self, name):
        self.unregistered.append(name)
        return self.sources.pop(name, None) is not None


async def test_a_claims_registry_without_the_role_keyword_falls_back_by_name():
    """Having `release_all` is not having the contract a claim needs.

    Passing `role=` to a registry that predates it is a TypeError inside
    `initialize()` — Channels would degrade ingress permanently and Scheduler
    and Restart Coordinator would fail to start. Such a registry gets the
    name-tracking path: no claims, but correct teardown, which is the half that
    cannot be skipped.
    """
    from kestrel_sovereign.signals import RegistrationPolicy

    registry = _RolelessClaimsRegistry()
    feature = _seam_feature(registry)

    assert feature._registry_holds_claims(registry) is False

    feature._register_signal_sources(
        _fake_source_registration("seam.roleless"), RegistrationPolicy.OPTIONAL,
    )
    assert "seam.roleless" in registry.sources
    # Registered WITHOUT an owner, because this registry has no role to put it
    # in — the fallback is a name list, not a half-stated claim.
    assert registry.claims["seam.roleless"] == [None]

    await feature._unregister_owned_signal_sources()

    assert registry.unregistered == ["seam.roleless"]


async def test_declining_on_the_fallback_path_unregisters_what_it_created():
    """Declining and teardown must do the SAME thing to a source.

    Dropping only the bookkeeping left the refused source registered and no
    longer tracked: still dispatchable, and beyond the reach of the shutdown
    that would otherwise have removed it. Strictly worse than either branch.
    """
    from kestrel_sovereign.signals import RegistrationPolicy

    registry = _NoClaimsRegistry()
    feature = _seam_feature(registry)

    outcomes = feature._register_signal_sources(
        _fake_source_registration("seam.declined"), RegistrationPolicy.OPTIONAL,
    )
    assert "seam.declined" in registry.sources

    feature._disown_signal_sources(outcomes)

    assert registry.unregistered == ["seam.declined"]
    assert "seam.declined" not in registry.sources
    assert feature._owned_signal_source_names == []


async def test_declining_on_the_fallback_path_keeps_a_peers_source():
    """A ridden incumbent is not this feature's to remove, declining included.

    The name path has no claims to express shared use, so the rule is the same
    one its teardown follows: remove only what this feature created.
    """
    from kestrel_sovereign.signals import RegistrationPolicy

    registry = _NoClaimsRegistry()
    incumbent = _seam_feature(registry)
    incumbent._register_signal_sources(
        _fake_source_registration("seam.declined.peer"), RegistrationPolicy.OPTIONAL,
    )
    rider = _seam_feature(registry)
    outcomes = rider._register_signal_sources(
        _fake_source_registration("seam.declined.peer"), RegistrationPolicy.OPTIONAL,
    )

    rider._disown_signal_sources(outcomes)

    assert registry.unregistered == []
    assert "seam.declined.peer" in registry.sources


class _HalfRoleAwareRegistry(_RolelessClaimsRegistry):
    """Role-aware for one source, pre-role for a batch.

    The shape a subclass takes when it inherits the new single-source method
    and keeps its own `register_batch` override. Scheduler registers its cron
    sources as a BATCH, so passing `role=` to it raises TypeError inside
    `initialize()` — and refusing the whole registry over that pushes it onto
    the name path it does not need. It keeps its claims and registers one at a
    time instead.
    """

    def __init__(self):
        super().__init__()
        self.claimed = []

    def register_with_policy(self, registration, policy, *, owner=None, role=None):
        outcome = super().register_with_policy(registration, policy)
        self.claimed.append((registration.name, owner, role))
        return outcome

    def register_batch(self, registrations, policy, *, owner=None):
        raise AssertionError(
            "a batch that cannot take the role must not be used for a claim"
        )


class _ForwardingProxyRegistry:
    """An embedder proxy that forwards everything to a real registry.

    `(*args, **kwargs)` cannot be inspected for a `role` parameter, and
    refusing it would push a proxy onto the name path — whose teardown removes
    by name and cannot see a peer's claim in the registry behind it. A leak is
    the better failure than deleting a source something else dispatches on.
    """

    def __init__(self, inner):
        self._inner = inner

    def register_with_policy(self, *args, **kwargs):
        return self._inner.register_with_policy(*args, **kwargs)

    def register_batch(self, *args, **kwargs):
        return self._inner.register_batch(*args, **kwargs)

    def release_all(self, *args, **kwargs):
        return self._inner.release_all(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self._inner.release(*args, **kwargs)


async def test_a_batch_that_cannot_take_the_role_is_not_used_for_a_claim():
    """The batch is an optimisation, not a capability.

    A registry whose batch predates the role can still hold claims one source
    at a time. Passing `role=` to that batch raises inside `initialize()`;
    refusing the registry over it pushes a claims-capable registry onto the
    name path, whose teardown removes by name and cannot see a peer's claim.
    Neither is necessary — register one at a time and keep the claims.

    Atomicity is what that costs, and it costs nothing here: every imperative
    site registers OPTIONAL, where each source is independent by definition.
    """
    from kestrel_sovereign.signals import CLAIM_IMPERATIVE, RegistrationPolicy

    registry = _HalfRoleAwareRegistry()
    feature = _seam_feature(registry)

    assert feature._registry_holds_claims(registry) is True

    feature._register_signal_sources(
        [
            _fake_source_registration("seam.half.one"),
            _fake_source_registration("seam.half.two"),
        ],
        RegistrationPolicy.OPTIONAL,
    )

    assert set(registry.sources) == {"seam.half.one", "seam.half.two"}
    # Claimed as this feature, in the imperative role, one call each.
    assert registry.claimed == [
        ("seam.half.one", feature, CLAIM_IMPERATIVE),
        ("seam.half.two", feature, CLAIM_IMPERATIVE),
    ]


async def test_a_forwarding_proxy_keeps_the_claims_path():
    """`**kwargs` counts: a proxy must not lose shared-claim safety.

    On the name path a rider's teardown removes by name, so it would delete a
    source the peer behind the proxy still holds. The claims path keeps that
    peer's claim visible.
    """
    from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry

    inner = SourceRegistry()
    proxy = _ForwardingProxyRegistry(inner)
    incumbent = _seam_feature(proxy)
    rider = _seam_feature(proxy)

    assert rider._registry_holds_claims(proxy) is True

    incumbent._register_signal_sources(
        _fake_source_registration("seam.proxy"), RegistrationPolicy.OPTIONAL,
    )
    rider._register_signal_sources(
        _fake_source_registration("seam.proxy"), RegistrationPolicy.OPTIONAL,
    )

    await rider._unregister_owned_signal_sources()

    # The peer still holds it, so it survives the rider going away.
    assert inner.get("seam.proxy") is not None
    assert inner.owners_of("seam.proxy") == (incumbent,)


class _NoBatchClaimsRegistry(_RolelessClaimsRegistry):
    """Role-aware, with release APIs, and no `register_batch` at all.

    A valid embedder shape for a feature that registers one source at a time.
    Requiring a batch method refused it over a call it never makes.
    """

    def __init__(self, inner):
        super().__init__()
        self._inner = inner

    def register_with_policy(self, registration, policy, *, owner=None, role=None):
        return self._inner.register_with_policy(
            registration, policy, owner=owner, role=role,
        )

    def release_all(self, owner, role=None):
        return self._inner.release_all(owner, role)

    def release(self, name, owner, role=None):
        return self._inner.release(name, owner, role)


async def test_a_registry_without_a_batch_method_still_holds_claims():
    """No batch is not "no claims".

    Refusing such a registry pushed it onto the name path, where the creator's
    shutdown removes by name — deleting a source a peer that rode it is still
    dispatching on.
    """
    from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry

    inner = SourceRegistry()
    registry = _NoBatchClaimsRegistry(inner)
    incumbent = _seam_feature(registry)
    rider = _seam_feature(registry)

    assert rider._registry_holds_claims(registry) is True

    incumbent._register_signal_sources(
        _fake_source_registration("seam.nobatch"), RegistrationPolicy.OPTIONAL,
    )
    rider._register_signal_sources(
        _fake_source_registration("seam.nobatch"), RegistrationPolicy.OPTIONAL,
    )

    await incumbent._unregister_owned_signal_sources()

    # The rider still holds it, so the creator's teardown does not take it.
    assert inner.get("seam.nobatch") is not None
    assert inner.owners_of("seam.nobatch") == (rider,)
