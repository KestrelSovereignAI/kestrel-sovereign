"""Real fixture cleanup keeps backend reservations and restores lifecycle guards."""

import pytest

from kestrel_sovereign import kestrel_agent as agent_module
from kestrel_sovereign.storage.db import sqlite
from tests.shared.booted_agent import booted_agent


@pytest.mark.parametrize("worker_budget", [1.0, 5.0])
@pytest.mark.parametrize("body_error", [False, True])
async def test_fixture_cleanup_closes_real_workers_and_restores_guards(
    tmp_path, monkeypatch, worker_budget, body_error,
):
    monkeypatch.setattr(sqlite, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", worker_budget)
    original = (
        agent_module.SHUTDOWN_TIMEOUT,
        agent_module.KESTREL_AGENT_SHUTDOWN_TIMEOUT_S,
    )
    resources, observed = [], []
    opening = RuntimeError("fixture body failed")

    async def exercise():
        async with booted_agent(
            tmp_path, name="Cleanup", provenance="test:native_fixture_cleanup",
            features=set(),
        ) as agent:
            backend = agent._raw_storage.db.backend
            connection = backend._connection
            resources.append((backend, getattr(connection, "_thread", connection)))
            shutdown = agent.shutdown

            async def observe_shutdown():
                tail = agent._durable_tail_minimum_budget(
                    agent_module._minimum_storage_close_timeout(agent.storage),
                    agent_module._minimum_storage_potential_preclose_timeout(agent.storage),
                )
                observed.append((
                    agent_module.SHUTDOWN_TIMEOUT,
                    agent_module.KESTREL_AGENT_SHUTDOWN_TIMEOUT_S,
                    agent_module._resolve_shutdown_budget(tail),
                    tail,
                ))
                await shutdown()

            monkeypatch.setattr(agent, "shutdown", observe_shutdown)
            if body_error:
                raise opening

    if body_error:
        with pytest.raises(RuntimeError) as caught:
            await exercise()
        assert caught.value is opening
    else:
        await exercise()

    assert len(resources) == len(observed) == 1
    backend, worker = resources[0]
    assert backend._connection is None
    assert not worker.is_alive()
    outer, internal, (prefix, reserved), tail = observed[0]
    assert prefix > 0 and reserved >= tail
    if worker_budget == 1.0:
        assert (outer, internal) == original, "coherent production guards changed"
    else:
        assert prefix >= max(original)
        assert outer == internal <= 30
    assert (
        agent_module.SHUTDOWN_TIMEOUT,
        agent_module.KESTREL_AGENT_SHUTDOWN_TIMEOUT_S,
    ) == original
