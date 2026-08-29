"""Turn-start Hold is unconditional, typed, and source-independent (#3162)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kestrel_sovereign.hold import (
    EffectiveHoldState,
    HoldEnforcementUnavailableError,
    HoldScope,
    HoldState,
    HoldTurnRefusal,
    require_context_hold_store,
    require_turn_start_allowed,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent


def _latch(scope: HoldScope, receipt_id: str, *, target: str) -> HoldState:
    return HoldState(
        scope=scope,
        target_id=target,
        reason=f"reason:{receipt_id}",
        actor_id=f"actor:{receipt_id}",
        set_at="2026-08-28T12:00:00+00:00",
        hold_receipt_id=receipt_id,
        revision=7,
    )


class _Store:
    def __init__(self, effective: EffectiveHoldState) -> None:
        self.effective = effective
        self.calls: list[str] = []

    async def get_effective(self, agent_id: str) -> EffectiveHoldState:
        self.calls.append(agent_id)
        return self.effective


def _bare_agent(store: _Store) -> KestrelAgent:
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.did = "did:test:held"
    agent._hold_store = store
    return agent


@pytest.mark.asyncio
async def test_process_input_refuses_at_unconditional_hold_seam() -> None:
    """Mutation tripwire: deleting the non-streaming check must run onward."""

    host = _latch(HoldScope.HOST, "hold:host", target="host")
    agent_hold = _latch(
        HoldScope.AGENT, "hold:agent", target="did:test:held"
    )
    store = _Store(EffectiveHoldState(host=host, agent=agent_hold))
    agent = _bare_agent(store)  # deliberately has no hooks manager or turn state

    with pytest.raises(HoldTurnRefusal) as caught:
        await agent.process_input("do not begin")

    assert store.calls == ["did:test:held"]
    assert caught.value.effective_state.host is host
    assert caught.value.effective_state.agent is agent_hold
    assert caught.value.metadata["host_hold"] is host
    assert caught.value.metadata["agent_hold"] is agent_hold


@pytest.mark.asyncio
async def test_streaming_input_refuses_before_first_yield_or_side_effect() -> None:
    """Mutation tripwire: deleting the streaming check must run onward."""

    agent_hold = _latch(
        HoldScope.AGENT, "hold:stream", target="did:test:held"
    )
    store = _Store(EffectiveHoldState(host=None, agent=agent_hold))
    agent = _bare_agent(store)

    stream = agent.process_input_streaming("do not stream")
    with pytest.raises(HoldTurnRefusal) as caught:
        await anext(stream)

    assert store.calls == ["did:test:held"]
    assert caught.value.host_hold is None
    assert caught.value.agent_hold is agent_hold


@pytest.mark.asyncio
async def test_hold_snapshot_is_the_turn_admission_linearization_point() -> None:
    observed = _latch(
        HoldScope.AGENT, "hold:observed", target="did:test:held"
    )

    class _ReleaseAfterReadStore(_Store):
        async def get_effective(self, agent_id: str) -> EffectiveHoldState:
            result = await super().get_effective(agent_id)
            self.effective = EffectiveHoldState(host=None, agent=None)
            return result

    store = _ReleaseAfterReadStore(EffectiveHoldState(host=None, agent=observed))
    agent = _bare_agent(store)

    with pytest.raises(HoldTurnRefusal) as caught:
        await require_turn_start_allowed(agent)

    # A later release cannot erase the exact latch that refused this turn.
    assert caught.value.agent_hold is observed
    assert store.effective.held is False


@pytest.mark.asyncio
async def test_unheld_snapshot_admits_turn_at_the_same_boundary() -> None:
    store = _Store(EffectiveHoldState(host=None, agent=None))
    agent = _bare_agent(store)

    effective = await require_turn_start_allowed(agent)

    assert effective == EffectiveHoldState(host=None, agent=None)
    assert store.calls == ["did:test:held"]


def test_production_context_without_hold_store_fails_closed() -> None:
    context = SimpleNamespace(hold_store=None, backend_error="database locked")

    with pytest.raises(HoldEnforcementUnavailableError, match="database locked"):
        require_context_hold_store(context)


def test_held_refusal_cannot_be_built_from_an_unheld_snapshot() -> None:
    with pytest.raises(ValueError, match="active latch"):
        HoldTurnRefusal(
            agent_id="did:test:held",
            effective_state=EffectiveHoldState(host=None, agent=None),
        )
