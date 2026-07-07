"""Tests for UI capability derivation from enabled features (#2041)."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.endpoints.features import router as features_router
from kestrel_sovereign.feature_registry import FeaturePackageInfo, FeatureStatus
from kestrel_sovereign.ui_capabilities import (
    compute_feature_capabilities,
    render_multi_agent_host_config_script,
    render_ui_config_script,
)


def _registry(*, voice_enabled=False, spawn_enabled=True):
    return {
        "voice": FeaturePackageInfo(
            name="voice",
            package="kestrel-feature-voice",
            git="https://example.com/voice.git",
            features=["VoiceFeature"],
            description="Voice",
            core=False,
            status=FeatureStatus.ENABLED if voice_enabled else FeatureStatus.AVAILABLE,
        ),
        "spawn": FeaturePackageInfo(
            name="spawn",
            package="kestrel-sovereign",
            git="https://example.com/core.git",
            features=["SpawnFeature"],
            description="Spawn",
            core=True,
            status=FeatureStatus.ENABLED if spawn_enabled else FeatureStatus.DISABLED,
        ),
    }


def _make_feature(enabled=True):
    feature = MagicMock()
    feature.enabled = enabled
    feature.on_enable = AsyncMock()
    feature.on_disable = AsyncMock()
    return feature


def _make_agent(features=None):
    agent = MagicMock()
    agent.features = features or {}
    return agent


# ---------------------------------------------------------------------------
# compute_feature_capabilities
# ---------------------------------------------------------------------------


class TestComputeFeatureCapabilities:
    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_enabled_feature_is_true_disabled_is_false(self, mock_registry):
        mock_registry.return_value = _registry(voice_enabled=True, spawn_enabled=False)
        agent = _make_agent({"VoiceFeature": _make_feature()})

        caps = compute_feature_capabilities(agent)

        assert caps == {"voice": True, "spawn": False}

    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_uninstalled_feature_reported_false_not_omitted(self, mock_registry):
        # voice not loaded → AVAILABLE → reported as False so the frontend can
        # treat it as authoritatively off (a force-true override is ignored).
        mock_registry.return_value = _registry(voice_enabled=False, spawn_enabled=True)
        agent = _make_agent({"SpawnFeature": _make_feature()})

        caps = compute_feature_capabilities(agent)

        assert caps["voice"] is False
        assert caps["spawn"] is True

    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_runtime_disabled_feature_drops_from_active_set(self, mock_registry):
        # A loaded-but-runtime-disabled feature (enabled flag False) must not be
        # passed to the registry as an enabled class → its capability is False.
        captured = {}

        def _capture(enabled_class_names=None):
            captured["enabled"] = set(enabled_class_names or set())
            return _registry(voice_enabled=False, spawn_enabled=True)

        mock_registry.side_effect = _capture
        agent = _make_agent(
            {
                "VoiceFeature": _make_feature(enabled=False),
                "SpawnFeature": _make_feature(enabled=True),
            }
        )

        compute_feature_capabilities(agent)

        assert "VoiceFeature" not in captured["enabled"]
        assert "SpawnFeature" in captured["enabled"]

    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_ui_capabilities_override_short_name(self, mock_registry):
        # When a feature declares ui_capabilities, those keys are emitted (not
        # the short name) so the derived set lines up with the frontend gates.
        mock_registry.return_value = {
            "observability": FeaturePackageInfo(
                name="observability",
                package="kestrel-feature-observability",
                git="https://example.com/obs.git",
                features=["ObservabilityFeature"],
                description="Metrics",
                status=FeatureStatus.ENABLED,
                ui_capabilities=["metrics"],
            ),
            "security": FeaturePackageInfo(
                name="security",
                package="kestrel-sovereign",
                git="https://example.com/core.git",
                features=["SecurityFeature"],
                description="Security",
                status=FeatureStatus.DISABLED,
                ui_capabilities=["audit", "permissions"],
            ),
        }
        agent = _make_agent({"ObservabilityFeature": _make_feature()})

        caps = compute_feature_capabilities(agent)

        # Frontend keys are present; the registry short names are not emitted.
        assert caps == {"metrics": True, "audit": False, "permissions": False}
        assert "observability" not in caps
        assert "security" not in caps

    def test_real_registry_maps_diverging_keys(self):
        # Against the bundled registry (no mock): the panels whose registry name
        # diverges from their frontend hasCapability() key must emit the UI key.
        agent = _make_agent()  # nothing enabled → all features report False

        caps = compute_feature_capabilities(agent)

        # observability → metrics; security → audit + permissions (PANEL_CAPABILITIES).
        for ui_key in ("metrics", "audit", "permissions"):
            assert ui_key in caps, f"expected derived UI key {ui_key!r}"
        # The bare registry short names must NOT leak in as capability keys.
        assert "observability" not in caps
        assert "security" not in caps

    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_missing_enabled_attr_defaults_to_active(self, mock_registry):
        captured = {}

        def _capture(enabled_class_names=None):
            captured["enabled"] = set(enabled_class_names or set())
            return {}

        mock_registry.side_effect = _capture
        feature = MagicMock(spec=[])  # no `enabled` attribute
        agent = _make_agent({"SpawnFeature": feature})

        compute_feature_capabilities(agent)

        assert "SpawnFeature" in captured["enabled"]


# ---------------------------------------------------------------------------
# render_ui_config_script
# ---------------------------------------------------------------------------


class TestRenderUiConfigScript:
    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_emits_inline_script_seeding_feature_capabilities(self, mock_registry):
        mock_registry.return_value = _registry(voice_enabled=True, spawn_enabled=True)
        agent = _make_agent({"VoiceFeature": _make_feature()})

        script = render_ui_config_script(agent)

        assert script.startswith("<script>")
        assert script.endswith("</script>")
        assert "window.KESTREL_UI_CONFIG" in script
        assert "featureCapabilities" in script
        assert '"voice": true' in script

    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_escapes_angle_brackets_to_prevent_script_breakout(self, mock_registry):
        mock_registry.return_value = {
            "we</script>ird": FeaturePackageInfo(
                name="we</script>ird",
                package="x",
                git="x",
                features=["X"],
                description="x",
                status=FeatureStatus.ENABLED,
            ),
        }
        agent = _make_agent({"X": _make_feature()})

        script = render_ui_config_script(agent)

        assert "</script>ird" not in script
        assert "\\u003c" in script


# ---------------------------------------------------------------------------
# render_multi_agent_host_config_script
# ---------------------------------------------------------------------------


class TestRenderMultiAgentHostConfigScript:
    def test_seeds_multi_agent_host_flag(self):
        script = render_multi_agent_host_config_script()

        assert script.startswith("<script>")
        assert script.endswith("</script>")
        assert "window.KESTREL_UI_CONFIG" in script
        assert '"multiAgentHost": true' in script
        # No agent is resolvable in this mode, so no capability map is seeded.
        assert "featureCapabilities" not in script


# ---------------------------------------------------------------------------
# GET /api/ui/capabilities
# ---------------------------------------------------------------------------


def _make_app(agent):
    app = FastAPI()
    app.include_router(features_router)
    app.state.agent = agent
    return app


class TestUiCapabilitiesEndpoint:
    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_returns_capability_map(self, mock_registry):
        mock_registry.return_value = _registry(voice_enabled=True, spawn_enabled=False)
        agent = _make_agent({"VoiceFeature": _make_feature()})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/ui/capabilities")

        assert resp.status_code == 200
        assert resp.json() == {"capabilities": {"voice": True, "spawn": False}}

    def test_returns_503_without_agent(self):
        app = FastAPI()
        app.include_router(features_router)
        with TestClient(app) as client:
            resp = client.get("/api/ui/capabilities")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# enable / disable include the recomputed capability map
# ---------------------------------------------------------------------------


class TestLifecyclePushesCapabilities:
    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_disable_flips_capability_and_returns_map(self, mock_registry):
        feature = _make_feature(enabled=True)
        feature.get_hooks.return_value = []

        # After disable the feature's `enabled` flag flips False; the registry
        # then reports it disabled.
        def _registry_for(enabled_class_names=None):
            voice_active = "VoiceFeature" in (enabled_class_names or set())
            return _registry(voice_enabled=voice_active, spawn_enabled=True)

        mock_registry.side_effect = _registry_for
        agent = _make_agent({"VoiceFeature": feature})
        agent.hooks_manager = None
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/VoiceFeature/disable")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "disabled"
        assert body["capabilities"]["voice"] is False
        assert feature.enabled is False

    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_enable_sets_flag_and_returns_map(self, mock_registry):
        feature = _make_feature(enabled=False)
        feature.get_hooks.return_value = []

        def _registry_for(enabled_class_names=None):
            voice_active = "VoiceFeature" in (enabled_class_names or set())
            return _registry(voice_enabled=voice_active, spawn_enabled=True)

        mock_registry.side_effect = _registry_for
        agent = _make_agent({"VoiceFeature": feature})
        agent.hooks_manager = None
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/VoiceFeature/enable")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "enabled"
        assert body["capabilities"]["voice"] is True
        assert feature.enabled is True

    @patch("kestrel_sovereign.ui_capabilities.get_registry")
    def test_enable_rolls_back_when_on_enable_fails(self, mock_registry):
        # A failing on_enable() must leave the feature authoritatively OFF: the
        # `enabled` flag stays False and the hooks registered for the attempt are
        # unwound, so the next capability computation doesn't see it as enabled.
        feature = _make_feature(enabled=False)
        hook = MagicMock()
        hook.name = "voice-hook"
        feature.get_hooks.return_value = [hook]
        feature.on_enable = AsyncMock(side_effect=RuntimeError("boom"))

        hooks_manager = MagicMock()
        mock_registry.return_value = _registry(voice_enabled=False, spawn_enabled=True)
        agent = _make_agent({"VoiceFeature": feature})
        agent.hooks_manager = hooks_manager
        app = _make_app(agent)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/features/VoiceFeature/enable")

        assert resp.status_code == 500
        assert feature.enabled is False
        hooks_manager.register.assert_called_once_with(hook)
        hooks_manager.unregister.assert_called_once_with(hook)
