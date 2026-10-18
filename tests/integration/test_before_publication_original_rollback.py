"""Before-publication rollback retains the original named private candidate.

An incepted actual Agent is supplied at the existing initializer seam. This
is a lifecycle/SQLite component check, not fresh boot, signed spawn or funds.
"""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_before_publication_refusal_keeps_opening_candidate_and_name(native_pg, onboarding_owner, monkeypatch, source_cancel):
    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    opening = OriginalFailure.capture(OSError("declared before-publication failure"))
    actual_admit = manager._admit_agent_operation
    admissions, refused = [], []
    async def before(_agent):
        raise opening.error
    async def rollback(_agent):
        await database.execute("INSERT INTO host_rollback_effects VALUES (91)")
    async def admit(*args, **kwargs):
        admission, owns = await actual_admit(*args, **kwargs)
        admission.before_publish = before
        admission.before_publish_rollback = rollback
        admissions.append(admission)
        return admission, owns
    async def existing_candidate(*_args, **_kwargs):
        return agent
    monkeypatch.setattr(manager, "_admit_agent_operation", admit)
    monkeypatch.setattr(manager, "_initialize_agent", existing_candidate)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is rollback.__code__:
            refused.append((coroutine, operation))
            raise source.error
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = create_owned_task(manager.load_agent("Original", LocalAgentConfig(data_dir="declared-candidate", port=8827)), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        assert len(refused) == 1, "actual original rollback admission was not exercised"
        source.assert_retained(outcome.error)
        opening.assert_retained(outcome.error)
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        admission = admissions[0]
        assert manager._agent_operations.get("original") is admission and admission.child is agent
        assert any(resource is admission and task is None for resource, task in execution_cleanup_debts(outcome.error))
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=91") == 0
        assert database.backend._connection is not None
        repeated = await await_owned_task(create_owned_task(manager._release_agent_operations([admission]), retain_originals=True))
        source.assert_retained(repeated.error)
        assert len(refused) == 1, "failed before-publication rollback acquired a replacement attempt"
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
