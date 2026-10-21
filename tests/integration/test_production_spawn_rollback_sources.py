"""Real _do_spawn rollback must return the original source, not a fresh C."""
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.spawn.mandate import SpawnMandate
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_agent_manager import _make_mock_agent
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_production_spawn_body_rollback_preserves_native_source(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    manager, child, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    parent = _make_mock_agent("did:test:production-rollback-parent")
    parent.features = {}

    async def create(name, **_kwargs):
        admission = manager._agent_operations[manager._canonical_agent_name(name)]
        admission.child = child
        manager._register_agent(name, child)
        raise source.error

    async def remove(name, **_kwargs):
        await database.execute("INSERT INTO host_rollback_effects VALUES (320)")
        manager._withdraw_initialized_agent(name, child)
        return True

    monkeypatch.setattr(manager, "create_agent", create)
    monkeypatch.setattr(manager, "remove_agent", remove)
    owner = create_owned_task(manager._do_spawn(
        "OriginalRollback", parent, SpawnMandate(parent_did=parent.agent_id),
    ), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=320") == 1
        assert manager.get_agent("OriginalRollback") is None and not manager._agent_operations
    finally:
        await await_owned_task(owner)
