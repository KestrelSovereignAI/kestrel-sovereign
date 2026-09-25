"""The restart idle gate against a REALLY booted, idle agent (#3347).

Every ``idle_agents_only`` restart on the production host waited the full
``MAX_IDLE_ONLY_DEFERRAL_SECONDS`` because a co-hosted agent that had run no
cognition for weeks always reported "background task(s) in flight". The idle
gate's exclusions are a hand-kept list of task-name prefixes, and every earlier
test of it hand-built ``_background_tasks`` from names the test author already
knew about — so a permanent task nobody listed could never fail one.

These tests boot a real ``KestrelAgent`` through ``initialize()`` with the
mandatory feature set (the features EVERY agent carries), let it settle, and ask
the coordinator about the set that boot actually produced. A new permanent task
started by boot and not classified as infrastructure fails here, by name.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.features import discover_features
from kestrel_sovereign.features.restart_coordinator.feature import (
    RestartCoordinatorFeature,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from tests.unit.test_agent_boot_phases import _boot_mocks, _cleanup, _make_agent


def _mandatory_features_only(agent, allowed_features=None):
    # An empty allowlist still loads every MANDATORY feature — the floor every
    # production agent shares, including one that has done nothing for weeks.
    return discover_features(agent, allowed_features=set())


@contextlib.asynccontextmanager
async def _booted_idle_agent(tmp_path):
    agent = _make_agent(tmp_path)
    agent.reconcile_a2a_cognition_wakes = AsyncMock()
    try:
        with _boot_mocks(), patch(
            "kestrel_sovereign.kestrel_agent.discover_features",
            side_effect=_mandatory_features_only,
        ):
            await agent.initialize()
        # Let boot-spawned tasks reach their steady state (a permanent loop is
        # parked in its sleep; one-shot boot work has finished).
        for _ in range(5):
            await asyncio.sleep(0)
        yield agent
    finally:
        await _cleanup(agent)


def _live_task_names(agent: KestrelAgent) -> list[str]:
    return sorted(t.get_name() for t in agent._background_tasks if not t.done())


async def _never() -> None:
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_booted_idle_agent_is_idle_to_the_restart_gate(tmp_path):
    async with _booted_idle_agent(tmp_path) as agent:
        # Non-vacuous: boot really did leave the permanent wait-reconcile
        # driver in the agent's task set. This is the task that made every
        # agent busy forever from #2729 on.
        assert "wait_fallback_reconcile" in _live_task_names(agent)

        state = RestartCoordinatorFeature(agent)._agent_appears_idle()

        assert state["idle"] is True, (state, _live_task_names(agent))


@pytest.mark.asyncio
async def test_an_owner_heartbeat_tick_does_not_make_the_agent_busy(tmp_path):
    """The durable owner heartbeat is a fresh task every ~40s forever, so the
    gate sees one in flight on a fraction of its checks."""
    async with _booted_idle_agent(tmp_path) as agent:
        agent.dispatcher._start_runtime_owner_heartbeat()
        heartbeat = agent.dispatcher._runtime_owner_heartbeat_task
        assert heartbeat is not None and not heartbeat.done()
        assert heartbeat in agent._background_tasks
        assert heartbeat.get_name().startswith("durable_signal_owner_heartbeat:")

        state = RestartCoordinatorFeature(agent)._agent_appears_idle()

        assert state["idle"] is True, (state, _live_task_names(agent))


@pytest.mark.asyncio
async def test_real_work_on_a_booted_agent_still_defers(tmp_path):
    async with _booted_idle_agent(tmp_path) as agent:
        work = agent._track_background_task(
            _never(), name="signal_dispatch:channel.telegram:sig-private-1",
        )
        try:
            state = RestartCoordinatorFeature(agent)._agent_appears_idle()

            assert state["idle"] is False
            # The requester's own reason keeps the full name (#2665) and
            # lists only the real blocker, not the infrastructure beside it.
            assert state["reason"].startswith(
                "1 background task(s) in flight: "
                "signal_dispatch:channel.telegram:sig-private-1 ("
            )
            assert "wait_fallback_reconcile" not in state["reason"]
        finally:
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work


@pytest.mark.asyncio
async def test_cohosted_booted_agent_names_blocking_kind_and_age(tmp_path):
    """A busy sibling's deferral names what is blocking and for how long, by
    kind only — never the per-signal tail."""
    async with _booted_idle_agent(tmp_path) as sibling:
        requester = SimpleNamespace(
            did="did:test:requester",
            name="Requester",
            dispatcher=SimpleNamespace(),
            _active_request_ids=set(),
            _background_tasks=set(),
        )
        requester._cohosted_agents_provider = lambda: [requester, sibling]
        coordinator = RestartCoordinatorFeature(requester)

        # Idle sibling: the fleet is idle — the sibling's permanent tasks do
        # not hold the host.
        assert coordinator._fleet_idle()["idle"] is True

        # What AgentManager does at registration. A real agent has no
        # ``name`` attribute, which is why the live reason showed a bare DID.
        sibling._set_display_name("Meridian")
        work = sibling._track_background_task(
            _never(), name="signal_dispatch:channel.telegram:sig-private-1",
        )
        # Backdate the creation stamp so the age is a real, checkable value.
        work._kestrel_started_at -= 3 * 3600
        try:
            state = coordinator._fleet_idle()

            assert state["idle"] is False
            assert state["blocker"]["scope"] == "cohosted_agent"
            assert state["reason"] == (
                f"co-hosted agent Meridian ({sibling.did}) busy "
                "(1 background task(s) in flight: signal_dispatch (3h))"
            )
            assert state["blocker"]["summary"] == "signal_dispatch (3h)"
            assert state["blocker"]["count"] == 1
            assert state["blocker"]["oldest_age_seconds"] >= 3 * 3600
            assert "sig-private-1" not in state["reason"]
            assert "channel.telegram" not in state["reason"]
        finally:
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work
