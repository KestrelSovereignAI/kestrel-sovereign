"""Best-effort published readiness cannot consume genuine native control."""
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_published_ready_boundary_does_not_consume_native_source(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    manager, agent, _database = onboarding_owner
    source = OriginalFailure.capture(
        await _original_control(native_pg[0], monkeypatch, source_cancel)
    )
    admission, owns = await manager._admit_agent_operation("NativeReady", kind="load")
    assert owns
    manager._register_agent("NativeReady", agent)
    admission.published = True

    async def ready(_agent):
        raise source.error

    monkeypatch.setattr(manager, "_run_hosted_agent_ready_hooks", ready)
    owner = create_owned_task(
        manager._run_hosted_agent_ready_hooks_before_mandate_expiry(
            "NativeReady", agent, deadline=None, admission=admission,
        ), retain_originals=True,
    )
    try:
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        assert owner.done()
    finally:
        await await_owned_task(owner)
        manager._withdraw_initialized_agent("NativeReady", agent)
        await manager._release_agent_operation(admission)
