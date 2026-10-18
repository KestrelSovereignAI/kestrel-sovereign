"""Single load/preconstruction keep independent actual PG originals finite."""
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.multi_agent import agent_manager as manager_module
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _cleanup_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

cleanup_pg = _cleanup_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("preconstruction", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_single_load_keeps_opening_and_cleanup_original_graphs(
    native_pg, cleanup_pg, onboarding_owner, tmp_path, monkeypatch,
    preconstruction, source_cancel,
):
    manager, agent, database = onboarding_owner
    opening = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    cleanup = OriginalFailure.capture(await _original_control(cleanup_pg[0], monkeypatch))
    effects = []
    actual_admit = manager._admit_agent_operation
    actual_close = agent.llm_service.close
    async def cleanup_agent():
        await database.execute("INSERT INTO host_rollback_effects VALUES (263)")
        effects.append(True)
        raise cleanup.error
    async def cleanup_service():
        await database.execute("INSERT INTO host_rollback_effects VALUES (263)")
        effects.append(True)
        await actual_close()
        raise cleanup.error
    async def before_publish(_agent):
        raise opening.error
    async def admit(*args, **kwargs):
        admission, owns = await actual_admit(*args, **kwargs)
        admission.before_publish = before_publish
        return admission, owns
    async def candidate(*args, **kwargs):
        return agent
    def refuse_constructor(**kwargs):
        raise opening.error
    # Restore the physical service method BEFORE the owning fixture's final
    # retirement. The measured failure is outside its successful native close.
    with monkeypatch.context() as patch:
        if preconstruction:
            patch.setattr(manager_module, "KestrelAgent", refuse_constructor)
            patch.setattr(manager_module, "LLMService", lambda **kwargs: agent.llm_service)
            patch.setattr(agent.llm_service, "close", cleanup_service)
        else:
            patch.setattr(manager, "_initialize_agent", candidate)
            patch.setattr(manager, "_admit_agent_operation", admit)
            patch.setattr(agent, "shutdown", cleanup_agent)
        owner = create_owned_task(manager.load_agent("Original",
            LocalAgentConfig(data_dir=tmp_path / "onboarding", port=8863)), retain_originals=True)
        outcome = await await_owned_task(owner)
        assert effects == [True], "original physical cleanup boundary was not reached"
        opening.assert_retained(outcome.error)
        cleanup.assert_retained(outcome.error)
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=263") == 1
