"""Native context and registry failures cannot become suppression success.

These are canonical lifecycle components on an incepted Agent with actual
SQLite contributions and an original PG receipt, not a deployed privacy boot.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["sync-context", "async-context", "quarantine-wait", "quarantine-operator"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_context_and_quarantine_keep_actual_control_before_recovery(native_pg, onboarding_owner, monkeypatch, surface, source_cancel):
    _manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    class NativeContextFeature(SDKFixtureFeature):
        contribution_prefix = "native-context-control"
        fault = False
        starts = 0
        async def initialize(self):
            self.starts += 1
            await super().initialize()
        def _render_context_clause(self):
            if self.fault and surface.endswith("context"):
                raise source.error
            return super()._render_context_clause()
    feature = NativeContextFeature(agent)
    await agent._activate_feature_runtime(feature)
    runtime = agent.feature_contribution_runtime
    await database.execute("INSERT INTO host_rollback_effects VALUES (62)")
    if surface.startswith("quarantine"):
        registry = runtime.wait_registry if surface == "quarantine-wait" else runtime.operator_registry
        name = "deregister" if surface == "quarantine-wait" else "quarantine_registration_set"
        original = getattr(type(registry), name)
        calls = []
        def retire(actual_registry, *args, **kwargs):
            result = original(actual_registry, *args, **kwargs)
            if actual_registry is registry and not calls:
                calls.append(args)
                raise source.error
            return result
        monkeypatch.setattr(type(registry), name, retire)
        async def invoke():
            runtime.quarantine(feature)
    else:
        feature.fault = True
        async def invoke():
            if surface == "sync-context":
                return agent.refresh_all_feature_context_clauses(fail_closed=True)
            return await agent.prepare_and_refresh_all_feature_context_clauses(fail_closed=True)
    result = await await_owned_task(create_owned_task(invoke(), retain_originals=True))
    source.assert_retained(result.error)
    assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=62") == 1
    if surface.startswith("quarantine"):
        assert calls
        assert not runtime.wait_registry.contains(feature.wait_registration.name, feature.wait_provider)
        assert runtime.setup_step_registry.get(feature.setup_registration.name) is None
        assert runtime.is_active(feature), "native quarantine failure falsely certified retirement"
    else:
        assert agent._feature_lifecycle_integrity_uncertain is True and agent._safe_mode is True
        assert all(clause.body == "" for clause in runtime.active_context_clauses())
        repeated = await await_owned_task(create_owned_task(agent._activate_feature_runtime(feature), retain_originals=True))
        source.assert_retained(repeated.error)
        assert feature.starts == 1
