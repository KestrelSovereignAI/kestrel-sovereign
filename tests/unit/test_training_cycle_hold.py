"""Standalone training launcher Hold lifecycle coverage."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_training_cycle_binds_hold_before_initialize_and_closes_it(
    tmp_path, monkeypatch
):
    from scripts import training_cycle
    from kestrel_sovereign import hold
    from kestrel_sovereign import kestrel_agent
    from kestrel_sovereign.llm import service as llm_service_module

    events = []
    hold_store = object()
    hold_context = SimpleNamespace(hold_store=hold_store)
    reflection = SimpleNamespace(
        reflect=AsyncMock(return_value={"actions": []})
    )

    class FakeAgent:
        def __init__(self, **_kwargs):
            self.features = {"ReflectionFeature": reflection}

        async def initialize(self):
            assert self._hold_store is hold_store
            events.append("initialize")

        async def shutdown(self):
            events.append("shutdown")

    async def bind(agent, **_kwargs):
        events.append("bind")
        agent._hold_store = hold_store
        return hold_context

    async def close(context):
        assert context is hold_context
        events.append("close")

    database = tmp_path / "training.db"
    database.touch()
    monkeypatch.setattr(
        training_cycle,
        "get_agent_did",
        AsyncMock(return_value="did:agent:training"),
    )
    monkeypatch.setattr(kestrel_agent, "KestrelAgent", FakeAgent)
    monkeypatch.setattr(llm_service_module, "LLMService", lambda: object())
    monkeypatch.setattr(hold, "build_bound_host_context", bind)
    monkeypatch.setattr(hold, "close_bound_host_context", close)

    await training_cycle.run_training_cycle(
        str(database), max_iterations=1, wait_between=0
    )

    assert events == ["bind", "initialize", "shutdown", "close"]

