"""Failed initialization retains its original private candidate and name.

A real incepted Agent/storage is supplied at the manager constructor seam.
Initialization/shutdown hooks perform real SQLite writes then fail with
declared/original PG receipts. This is not full boot/READY acceptance.
"""
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_failed_initializer_cleanup_retains_original_candidate_before_withdrawal(native_pg, onboarding_owner, tmp_path, monkeypatch, source_cancel):
    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    opening = OriginalFailure.capture(OSError("actual initialized candidate hook failed"))
    admissions, initialized, shutdown = [], [], []
    actual_admit = manager._admit_agent_operation
    async def admit(*args, **kwargs):
        admission, owns = await actual_admit(*args, **kwargs)
        admissions.append(admission)
        return admission, owns
    async def initialize():
        initialized.append(True)
        await database.execute("INSERT INTO host_rollback_effects VALUES (101)")
        raise opening.error
    async def retire():
        shutdown.append(True)
        await database.execute("INSERT INTO host_rollback_effects VALUES (102)")
        raise source.error
    monkeypatch.setattr(manager, "_admit_agent_operation", admit)
    monkeypatch.setattr(manager_module, "KestrelAgent", lambda **kwargs: agent)
    monkeypatch.setattr(manager_module, "LLMService", lambda **kwargs: agent.llm_service)
    monkeypatch.setattr(agent, "initialize", initialize)
    monkeypatch.setattr(agent, "shutdown", retire)
    owner = create_owned_task(manager.load_agent("Original", LocalAgentConfig(data_dir=tmp_path / "onboarding", port=8831)), retain_originals=True)
    result = await await_owned_task(owner)
    assert initialized == shutdown == [True], "actual initialization/retirement boundaries were not reached"
    source.assert_retained(result.error)
    opening.assert_retained(result.error)
    admission = admissions[0]
    assert manager._agent_operations.get("original") is admission and admission.child is agent
    assert not manager._initializing_agents and not manager.list_agents()
    assert any(resource is agent and task is not None and task.done()
               for resource, task in execution_cleanup_debts(result.error))
    assert database.backend._connection is not None
    assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id IN (101,102)") == 2
    repeated = await await_owned_task(create_owned_task(manager._release_agent_operations([admission]), retain_originals=True))
    source.assert_retained(repeated.error)
    assert shutdown == [True] and manager._agent_operations.get("original") is admission
