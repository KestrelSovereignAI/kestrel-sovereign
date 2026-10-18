"""Startup preparation/rollback share runtime native-integrity fencing."""
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _opening_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

opening_pg = _opening_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["initialize", "config", "enable", "preflight", "native-initialize"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_startup_native_cleanup_or_preflight_prevents_ready_and_reactivation(
    native_pg, opening_pg, onboarding_owner, monkeypatch, stage, source_cancel,
):
    _, agent, database = onboarding_owner
    cleanup = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    opening = OriginalFailure.capture(await _original_control(opening_pg[0], monkeypatch))
    events = []
    class NativeStartupFeature(SDKFixtureFeature):
        contribution_prefix = "native-startup-205"
        async def initialize(self):
            events.append("initialize")
            if stage == "initialize":
                raise OSError("declared ordinary startup preparation failure")
            if stage == "native-initialize":
                raise opening.error
            await super().initialize()
        async def on_enable(self):
            events.append("enable")
            if stage == "enable":
                raise OSError("declared ordinary startup enablement failure")
        async def on_disable(self):
            events.append("disable")
            await database.execute("INSERT INTO host_rollback_effects VALUES (273)")
            raise cleanup.error
        def get_context_clause_registrations(self):
            if stage == "preflight":
                raise cleanup.error
            return super().get_context_clause_registrations()
    feature = NativeStartupFeature(agent)
    if stage == "config":
        async def reject_config(_feature):
            raise OSError("declared ordinary startup config failure")
        monkeypatch.setattr(agent, "_apply_host_feature_config", reject_config)
    async def invoke():
        if stage == "preflight":
            return agent._prepare_feature_contribution_transition((feature,))
        return await agent._register_startup_feature(feature)
    outcome = await await_owned_task(create_owned_task(invoke(), retain_originals=True))
    cleanup.assert_retained(outcome.error)
    if stage == "native-initialize":
        opening.assert_retained(outcome.error)
    assert vars(agent).get("_feature_lifecycle_integrity_uncertain") is True
    assert agent._safe_mode is True
    if stage != "preflight":
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=273") == 1
    before = tuple(events)
    for retry in (agent._activate_feature_runtime(feature), agent._notify_feature_runtime_ready(feature)):
        rejected = await await_owned_task(create_owned_task(retry, retain_originals=True))
        cleanup.assert_retained(rejected.error)
    assert tuple(events) == before
