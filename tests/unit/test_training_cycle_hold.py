"""Standalone training retains Hold authority through deferred shutdown."""

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_training_joins_deferred_shutdown_before_closing_hold(monkeypatch):
    from scripts import training_cycle
    from kestrel_sovereign import hold

    events: list[str] = []
    context = object()

    class Agent:
        async def shutdown(self):
            events.append("shutdown-returned")

        async def wait_for_shutdown_completion(self):
            events.append("deferred-shutdown-joined")

    agent = Agent()

    async def run_cycle(**kwargs):
        kwargs["lifecycle"].update(agent=agent, hold_context=context)
        return "complete"

    async def close_context(received):
        assert received is context
        events.append("hold-closed")

    monkeypatch.setattr(training_cycle, "_run_training_cycle", run_cycle)
    monkeypatch.setattr(
        hold,
        "close_bound_host_context",
        AsyncMock(side_effect=close_context),
    )

    assert await training_cycle.run_training_cycle("unused.db") == "complete"
    assert events == [
        "shutdown-returned",
        "deferred-shutdown-joined",
        "hold-closed",
    ]
