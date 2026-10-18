"""A cancellation count is not a newly delivered lifecycle withdrawal."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.multi_agent.agent_manager import AgentManager


@pytest.mark.asyncio
async def test_name_release_does_not_replay_an_already_handled_cancellation(tmp_path):
    manager = AgentManager(base_data_dir=tmp_path)
    async def invoke():
        admission, created = await manager._admit_agent_operation("Handled", kind="load")
        assert created
        asyncio.current_task().cancel("already handled before retirement")
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        assert asyncio.current_task().cancelling() == 1
        await manager._release_agent_operation(admission)
        return True
    owner = create_owned_task(invoke(), retain_originals=True)
    result = await await_owned_task(owner)
    assert result.error is None and result.result is True
    assert not manager._agent_operations
