"""A staged real Agent remains rooted when its original initializer withdraws."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_execution_custody_review29 import native_owner
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
async def test_staging_withdrawal_retains_original_private_cleanup_and_name(
    native_pg, tmp_path, monkeypatch,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    agent, database = await native_owner(tmp_path)
    manager = AgentManager(base_data_dir=tmp_path)
    returned, beta_release, cleanup_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    stops = []
    actual_shutdown = agent.shutdown
    async def initialize(name, *_args, **_kwargs):
        if name == "Alpha":
            returned.set()
            return agent
        await beta_release.wait()
        raise RuntimeError("declared independent initializer failure")
    async def shutdown():
        stops.append(asyncio.current_task())
        cleanup_entered.set()
        await actual_shutdown()
        raise source.error
    monkeypatch.setattr(manager, "_initialize_agent", initialize)
    monkeypatch.setattr(agent, "shutdown", shutdown)
    config = MultiAgentConfig(agents={
        name: LocalAgentConfig(data_dir=name.lower(), port=8811 + i)
        for i, name in enumerate(("Alpha", "Beta"))
    })
    caller = create_owned_task(manager.load_from_config(config), retain_originals=True)
    try:
        await asyncio.wait_for(returned.wait(), 5)
        admission = manager._agent_operations["alpha"]
        original = admission.owner_task
        assert original is not None and not original.done()
        original.cancel("actual staged initializer withdrew")
        await asyncio.wait_for(cleanup_entered.wait(), 5)
        beta_release.set()
        outcome = await asyncio.wait_for(await_owned_task(caller), 5)
        cleanup = vars(agent).get("_unpublished_cleanup_owner")
        assert cleanup is not None, "staging withdrawal bypassed canonical private cleanup"
        source.assert_retained(outcome.error, resource=agent, task=cleanup.admission.task)
        assert original.done() and stops == [cleanup.admission.task]
        assert manager._agent_operations.get("alpha") is admission and admission.child is agent
        assert not manager.list_agents() and database.backend._connection is None
    finally:
        beta_release.set()
        if not caller.done():
            caller.cancel()
        await await_owned_task(caller)
        await database.close()
