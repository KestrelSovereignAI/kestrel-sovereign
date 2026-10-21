"""Native uncertainty cannot reactivate a feature generation or its ready hook.

Direct canonical lifecycle/endpoint helpers on an incepted real agent/storage;
this is not authenticated HTTP, whole-agent genesis or provider acceptance.
"""
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.endpoints.features import (
    ConfigUpdateRequest, _disable_feature_locked, _enable_feature_locked,
    _remove_feature_locked, _update_feature_config_locked,
)
from kestrel_sovereign.feature_registry import FeaturePackageInfo
from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["disable", "remove", "config-setter", "config-refresh", "config-rollback", "enable-cleanup", "ready", "activate-preflight", "enable-preflight", "register-preflight"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_disable_native_failure_does_not_reopen_the_retained_generation(native_pg, onboarding_owner, monkeypatch, surface, source_cancel):
    _manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    await database.execute("CREATE TABLE lifecycle_config (id INTEGER PRIMARY KEY, value INTEGER)")
    await database.execute("INSERT INTO lifecycle_config VALUES (1, 0)")
    events = []
    class NativeUncertainFeature(SDKFixtureFeature):
        contribution_prefix = "native-uncertain"
        starts = 0
        ready = 0
        fault = False
        setters = 0
        renders = 0
        async def initialize(self):
            self.starts += 1
            await super().initialize()
        async def on_agent_ready(self, _agent):
            self.ready += 1
            if self.fault and surface == "ready":
                raise source.error
        async def on_disable(self):
            events.append("disable")
            await database.execute("UPDATE lifecycle_config SET value=value+1 WHERE id=1")
            if self.fault and surface in {"disable", "remove", "enable-cleanup"}:
                raise source.error
        async def on_enable(self):
            events.append("enable")
            if self.fault and surface == "enable-cleanup":
                raise OSError("ordinary feature activation failure")
        async def get_config(self):
            return {"value": await database.fetchval("SELECT value FROM lifecycle_config WHERE id=1")}
        async def set_config(self, config):
            self.setters += 1
            await database.execute("UPDATE lifecycle_config SET value=? WHERE id=1", (config["value"],))
            if self.fault and (surface == "config-setter" or surface == "config-rollback" and self.setters == 2):
                raise source.error
        def get_context_clause_registrations(self):
            if self.fault and surface in {"activate-preflight", "enable-preflight", "register-preflight"}:
                raise source.error
            return super().get_context_clause_registrations()
        def _render_context_clause(self):
            self.renders += 1
            if self.fault and surface == "config-refresh":
                raise source.error
            if self.fault and surface == "config-rollback":
                raise OSError("ordinary context refresh failure")
            return super()._render_context_clause()
    feature = NativeUncertainFeature(agent)
    await agent._activate_feature_runtime(feature)
    if surface in {"enable-cleanup", "activate-preflight", "enable-preflight", "register-preflight"}:
        await agent._unregister_feature_runtime(feature, unload=False)
    before = feature.starts, feature.ready
    feature.fault = True
    if surface == "disable":
        operation = _disable_feature_locked(agent, feature.name)
    elif surface == "remove":
        operation = _remove_feature_locked(agent, FeaturePackageInfo(
            name="native-uncertain", package="never-uninstall-fixture", git="", features=[feature.name], description="native lifecycle test",
        ))
    elif surface == "enable-cleanup":
        operation = _enable_feature_locked(agent, feature.name)
    elif surface == "ready":
        operation = agent._notify_feature_runtime_ready(feature)
    elif surface == "activate-preflight":
        operation = agent._activate_feature_runtime(feature)
    elif surface == "enable-preflight":
        operation = _enable_feature_locked(agent, feature.name)
    elif surface == "register-preflight":
        operation = agent._register_feature(feature)
    else:
        operation = _update_feature_config_locked(agent, feature, feature.name, ConfigUpdateRequest(config={"value": 41}))
    outcome = await await_owned_task(create_owned_task(operation, retain_originals=True))
    source.assert_retained(outcome.error)
    expected = before[0] + (surface == "enable-cleanup"), before[1] + (surface == "ready")
    assert (feature.starts, feature.ready) == expected, "native teardown uncertainty reactivated the generation or reopened its ready seam"
    assert feature.enabled is False
    assert agent._feature_lifecycle_integrity_uncertain is True and agent._safe_mode is True
    if surface.startswith("config"):
        assert feature.setters == (2 if surface == "config-rollback" else 1), "native uncertainty replayed an unproven config mutation"
    for retry in (agent._activate_feature_runtime(feature), agent._notify_feature_runtime_ready(feature)):
        repeated = await await_owned_task(create_owned_task(retry, retain_originals=True))
        source.assert_retained(repeated.error)
    assert (feature.starts, feature.ready) == expected, "retained native control admitted later activation/READY"
