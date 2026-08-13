"""Tests for the Feature Store API endpoints (endpoints/features.py)."""

import shlex
import sys
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.endpoints.features import router as features_router
from kestrel_sovereign.feature_registry import (
    FeaturePackageInfo,
    FeatureStatus,
    SkillInfo,
)
from tests.utils.fake_uv import CORE, FakeUv, use_fake_uv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tool(name="test_tool", description="A test tool", category="system", parameters=None, command_prefix=None):
    """Create a mock AgentTool."""
    schema = MagicMock()
    schema.name = name
    schema.description = description
    schema.category = category
    schema.parameters = parameters or {"type": "object", "properties": {}}
    schema.command_prefix = command_prefix
    tool = MagicMock()
    tool.schema = schema
    tool.name = name
    return tool


def _make_feature(
    name="TestFeature",
    tool_name="test_feature",
    tool_description="A test feature",
    tools=None,
    hooks=None,
    config_schema=None,
    config=None,
    enabled=True,
):
    """Create a mock Feature instance."""
    feature = MagicMock()
    feature.name = name
    feature.tool_name = tool_name
    feature.tool_description = tool_description
    feature.get_tools.return_value = tools or []
    feature.get_hooks.return_value = hooks or []
    type(feature).config_schema = PropertyMock(return_value=config_schema)
    feature.get_config = AsyncMock(return_value=config or {})
    feature.set_config = AsyncMock()
    feature.on_enable = AsyncMock()
    feature.on_disable = AsyncMock()
    feature.on_remove = AsyncMock()
    # Async lifecycle surface the canonical activation/teardown drive
    # (KestrelAgent._activate_feature_runtime / _unregister_feature_runtime).
    feature.initialize = AsyncMock()
    feature.shutdown = AsyncMock()
    feature.post_all_features_loaded = AsyncMock()
    # A concrete bool (not an auto-truthy MagicMock) so activation's
    # startup-tool-promotion branch is deterministically skipped.
    feature.promote_tools_on_startup = False
    feature.enabled = enabled
    return feature


def _make_app(agent=None):
    """Create a FastAPI app with the features router mounted."""
    app = FastAPI()
    app.include_router(features_router)
    if agent is not None:
        app.state.agent = agent
    return app


def _make_agent(features=None):
    """Create a mock agent with optional features dict."""
    agent = MagicMock()
    agent.features = features or {}
    return agent


def _lifecycle_agent(features=None):
    """A REAL KestrelAgent for the enable/disable production path (#2522).

    The enable/disable endpoints delegate per-feature work to the agent's
    canonical ``_activate_feature_runtime`` / ``_unregister_feature_runtime`` —
    there is no endpoint-local teardown to mock, so these tests must drive the
    real methods. Construction is cheap (``__init__`` opens no DB); the runtime
    registries the teardown touches (signal / wait) are attached empty and the
    A2A task manager left unset so mock features need no agent-card wiring.
    """
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.signals.registry import SourceRegistry
    from kestrel_sovereign.waits import WaitRegistry

    agent = KestrelAgent(did="did:test:features", storage_path=":memory:")
    agent.task_manager = None
    agent.signal_registry = SourceRegistry()
    agent.wait_registry = WaitRegistry()
    agent.features = features or {}
    return agent


FAKE_REGISTRY = {
    "test-pkg": FeaturePackageInfo(
        name="test-pkg",
        package="kestrel-feature-test",
        git="https://github.com/example/test.git",
        features=["TestFeature"],
        description="Test feature package",
        tags=["test", "demo"],
        icon="flask",
        core=False,
        skills=[
            SkillInfo(name="do_thing", description="Does a thing", category="system", tags=["test"]),
        ],
    ),
    "core-pkg": FeaturePackageInfo(
        name="core-pkg",
        package="kestrel-sovereign",
        git="https://github.com/example/core.git",
        features=["CoreFeature"],
        description="Core feature",
        tags=["core"],
        icon="star",
        core=True,
        skills=[],
        status=FeatureStatus.INSTALLED,
    ),
}


# ---------------------------------------------------------------------------
# GET /api/features
# ---------------------------------------------------------------------------


class TestListFeatures:
    def test_returns_503_without_agent(self):
        app = _make_app()
        with TestClient(app) as client:
            resp = client.get("/api/features")
        assert resp.status_code == 503

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_returns_catalog(self, mock_registry):
        mock_registry.return_value = dict(FAKE_REGISTRY)
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features")

        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert data["count"] == 2
        by_name = {item["name"]: item for item in data["features"]}
        assert by_name["test-pkg"]["boundary"] == "feature-package"
        assert by_name["test-pkg"]["installable"] is True
        assert by_name["core-pkg"]["boundary"] == "bundled"
        assert by_name["core-pkg"]["installable"] is False

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_filter_by_tag(self, mock_registry):
        mock_registry.return_value = dict(FAKE_REGISTRY)
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features?tag=core")

        data = resp.json()
        assert data["count"] == 1
        assert data["features"][0]["name"] == "core-pkg"

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_disabled_loaded_feature_is_not_reported_enabled(self, mock_registry):
        mock_registry.return_value = {}
        feature = _make_feature(enabled=False)
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            client.get("/api/features")

        assert mock_registry.call_args.kwargs["enabled_class_names"] == set()


# ---------------------------------------------------------------------------
# GET /api/features/installed
# ---------------------------------------------------------------------------


class TestListInstalledFeatures:
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    def test_returns_loaded_features_with_tools(self, mock_pkg):
        mock_pkg.return_value = FAKE_REGISTRY["test-pkg"]
        tool = _make_tool()
        feature = _make_feature(tools=[tool])
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/installed")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["features"][0]["name"] == "TestFeature"
        assert data["features"][0]["boundary"] == "feature-package"
        assert len(data["features"][0]["tools"]) == 1
        assert data["features"][0]["tools"][0]["name"] == "test_tool"


# ---------------------------------------------------------------------------
# GET /api/features/{name}
# ---------------------------------------------------------------------------


class TestGetFeatureDetail:
    def test_loaded_feature_returns_detail(self):
        tool = _make_tool()
        feature = _make_feature(
            tools=[tool],
            config_schema={"type": "object", "properties": {"key": {"type": "string"}}},
        )
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/TestFeature")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TestFeature"
        assert data["status"] == "enabled"
        assert len(data["tools"]) == 1
        assert data["config_schema"] is not None

    def test_loaded_disabled_feature_reports_disabled(self):
        feature = _make_feature(enabled=False)
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/TestFeature")

        assert resp.status_code == 200
        assert resp.json()["status"] == "disabled"

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_unloaded_feature_from_registry(self, mock_registry):
        mock_registry.return_value = dict(FAKE_REGISTRY)
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/TestFeature")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-pkg"

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_unknown_feature_returns_404(self, mock_registry):
        mock_registry.return_value = {}
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/NonexistentFeature")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/features/{name}/enable
# ---------------------------------------------------------------------------


class TestEnableFeature:
    def test_enable_calls_on_enable(self):
        feature = _make_feature(enabled=False)
        agent = _lifecycle_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/TestFeature/enable")

        assert resp.status_code == 200
        assert resp.json()["status"] == "enabled"
        feature.on_enable.assert_awaited_once()
        # Canonical activation re-runs the full registration on the SAME
        # instance (#2522): initialize (signal sources) + post_all_features_loaded
        # (wait providers) ran too, and the feature is live again.
        feature.initialize.assert_awaited_once()
        feature.post_all_features_loaded.assert_awaited_once()
        assert feature.enabled is True

    def test_enable_unknown_feature_returns_404(self):
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/Unknown/enable")

        assert resp.status_code == 404

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_enable_accepts_package_stable_id(self, mock_registry):
        first = _make_feature(name="FirstFeature", enabled=False)
        second = _make_feature(name="SecondFeature", enabled=False)
        info = FeaturePackageInfo(
            name="multi-pkg", package="kestrel-feature-multi", git="",
            features=["FirstFeature", "SecondFeature"], description="multi",
        )
        mock_registry.return_value = {"multi-pkg": info}
        agent = _lifecycle_agent(
            features={"FirstFeature": first, "SecondFeature": second}
        )
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/multi-pkg/enable")

        assert resp.status_code == 200
        assert resp.json()["features"] == ["FirstFeature", "SecondFeature"]
        first.on_enable.assert_awaited_once()
        second.on_enable.assert_awaited_once()
        assert first.enabled is True and second.enabled is True

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_package_enable_rolls_back_when_a_member_fails(self, mock_registry):
        first = _make_feature(name="FirstFeature", enabled=False)
        second = _make_feature(name="SecondFeature", enabled=False)
        second.on_enable.side_effect = RuntimeError("boom")
        info = FeaturePackageInfo(
            name="multi-pkg", package="kestrel-feature-multi", git="",
            features=["FirstFeature", "SecondFeature"], description="multi",
        )
        mock_registry.return_value = {"multi-pkg": info}
        agent = _lifecycle_agent(
            features={"FirstFeature": first, "SecondFeature": second}
        )
        app = _make_app(agent)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/features/multi-pkg/enable")

        assert resp.status_code == 500
        # first was fully activated then rolled back (soft-disabled); second's
        # atomic activation tore its own partial state down on the failed
        # on_enable — so both members ran on_disable and both end disabled.
        first.on_disable.assert_awaited_once()
        second.on_disable.assert_awaited_once()
        assert first.enabled is False and second.enabled is False

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_enable_rollback_continues_when_cleanup_fails(self, mock_registry):
        first = _make_feature(name="FirstFeature", enabled=False)
        first.on_disable.side_effect = RuntimeError("cleanup failed")
        second = _make_feature(name="SecondFeature", enabled=False)
        second.on_enable.side_effect = RuntimeError("enable failed")
        second.on_disable.side_effect = RuntimeError("second cleanup failed")
        info = FeaturePackageInfo(
            name="multi-pkg", package="kestrel-feature-multi", git="",
            features=["FirstFeature", "SecondFeature"], description="multi",
        )
        mock_registry.return_value = {"multi-pkg": info}
        agent = _lifecycle_agent(
            features={"FirstFeature": first, "SecondFeature": second}
        )
        app = _make_app(agent)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/features/multi-pkg/enable")

        assert resp.status_code == 500
        # Even when the teardown lifecycle itself raises, the unconditional
        # cleanup still flips ``enabled`` false on both members (#2522 P2).
        assert first.enabled is False and second.enabled is False


# ---------------------------------------------------------------------------
# POST /api/features/{name}/disable
# ---------------------------------------------------------------------------


class TestDisableFeature:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "ConstitutionFeature",
            "IdentityFeature",
            "PeersFeature",
            "SecurityFeature",
            "WaitFeature",
        ],
    )
    def test_mandatory_feature_disable_is_rejected_before_lifecycle(
        self, feature_name
    ):
        feature = _make_feature(name=feature_name)
        agent = _make_agent(features={feature_name: feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            response = client.post(f"/api/features/{feature_name}/disable")

        assert response.status_code == 409
        assert feature_name in response.json()["detail"]
        feature.on_disable.assert_not_awaited()
        assert feature.enabled is True

    def test_disable_calls_on_disable(self):
        feature = _make_feature()
        agent = _lifecycle_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/TestFeature/disable")

        assert resp.status_code == 200
        assert resp.json()["status"] == "disabled"
        feature.on_disable.assert_awaited_once()
        # Canonical teardown detaches the feature's owned resources too (#2522):
        # shutdown() (signal sources + wait providers) ran, and the SAME
        # instance stays loaded (soft-toggle) so /enable can restore it.
        feature.shutdown.assert_awaited_once()
        assert feature.enabled is False
        assert agent.features.get("TestFeature") is feature

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_disable_accepts_package_stable_id(self, mock_registry):
        first = _make_feature(name="FirstFeature")
        second = _make_feature(name="SecondFeature")
        info = FeaturePackageInfo(
            name="multi-pkg", package="kestrel-feature-multi", git="",
            features=["FirstFeature", "SecondFeature"], description="multi",
        )
        mock_registry.return_value = {"multi-pkg": info}
        agent = _lifecycle_agent(
            features={"FirstFeature": first, "SecondFeature": second}
        )
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/multi-pkg/disable")

        assert resp.status_code == 200
        assert resp.json()["features"] == ["FirstFeature", "SecondFeature"]
        first.on_disable.assert_awaited_once()
        second.on_disable.assert_awaited_once()
        assert first.enabled is False and second.enabled is False

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_package_disable_rolls_back_when_a_member_fails(self, mock_registry):
        first = _make_feature(name="FirstFeature")
        second = _make_feature(name="SecondFeature")
        second.on_disable.side_effect = RuntimeError("boom")
        info = FeaturePackageInfo(
            name="multi-pkg", package="kestrel-feature-multi", git="",
            features=["FirstFeature", "SecondFeature"], description="multi",
        )
        mock_registry.return_value = {"multi-pkg": info}
        agent = _lifecycle_agent(
            features={"FirstFeature": first, "SecondFeature": second}
        )
        app = _make_app(agent)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/features/multi-pkg/disable")

        assert resp.status_code == 500
        # Group transaction rolled back: every attempted member (first, which
        # tore down cleanly, AND second, whose teardown raised) is re-activated,
        # so both are enabled again.
        first.on_enable.assert_awaited_once()
        second.on_enable.assert_awaited_once()
        assert first.enabled is True and second.enabled is True

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_disable_rollback_reenable_failure_leaves_member_disabled(
        self, mock_registry
    ):
        # When the group rollback's re-enable ITSELF fails, the canonical
        # activation is atomic: the member whose on_enable raised is torn back
        # down rather than falsely reported enabled (#2522). This supersedes the
        # old "restore the enabled flag regardless" behavior — an ``enabled``
        # flag out of sync with a failed on_enable was a lie about live state.
        first = _make_feature(name="FirstFeature")
        first.on_enable.side_effect = RuntimeError("rollback failed")
        second = _make_feature(name="SecondFeature")
        second.on_disable.side_effect = RuntimeError("disable failed")
        info = FeaturePackageInfo(
            name="multi-pkg", package="kestrel-feature-multi", git="",
            features=["FirstFeature", "SecondFeature"], description="multi",
        )
        mock_registry.return_value = {"multi-pkg": info}
        agent = _lifecycle_agent(
            features={"FirstFeature": first, "SecondFeature": second}
        )
        app = _make_app(agent)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/features/multi-pkg/disable")

        assert resp.status_code == 500
        # second re-enabled cleanly; first's re-enable failed → left disabled.
        assert second.enabled is True
        assert first.enabled is False


# ---------------------------------------------------------------------------
# POST /api/features/{name}/install
# ---------------------------------------------------------------------------


class TestInstallFeature:
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_install_core_returns_400(self, mock_registry):
        mock_registry.return_value = dict(FAKE_REGISTRY)
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/CoreFeature/install")

        assert resp.status_code == 400
        assert "core" in resp.json()["detail"].lower()

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_install_unknown_returns_404(self, mock_registry):
        mock_registry.return_value = {}
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/Unknown/install")

        assert resp.status_code == 404

    # -- core install guard (#2949) -----------------------------------------
    #
    # Installing from the console is not a safer path than installing from the
    # CLI: the package depends on kestrel-sovereign, so an unguarded install can
    # resolve core from the index and replace the running editable core. Same
    # venv/resolver double as the CLI tests (tests/utils/fake_uv.py) — the two
    # surfaces claim identical behaviour, so they are held to one model.

    @staticmethod
    def _venv(monkeypatch, **kw):
        venv = FakeUv(feature="kestrel-feature-test", core_checkout="/src/core", **kw)
        use_fake_uv(monkeypatch, venv)
        return venv

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_install_pins_core_to_the_editable_checkout(self, mock_registry, monkeypatch):
        """The regression, over HTTP: a feature requiring core > the checkout's
        version fails loudly instead of replacing the editable install."""
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = self._venv(monkeypatch)  # editable core 0.52.0; feature wants >=0.53

        with TestClient(_make_app(_make_agent()), raise_server_exceptions=False) as client:
            resp = client.post("/api/features/test-pkg/install")

        assert resp.status_code == 500
        assert "No solution found" in resp.json()["detail"]
        assert venv.pins == ["==0.52.0"]  # the pin reached the resolver
        assert venv.editable[CORE] == "/src/core"  # link intact
        assert "kestrel-feature-test" not in venv.installed

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_install_succeeds_when_the_checkout_satisfies_the_feature(
        self, mock_registry, monkeypatch
    ):
        """The pin must not manufacture failures."""
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = self._venv(monkeypatch, feature_requires=">=0.52")

        with TestClient(_make_app(_make_agent())) as client:
            resp = client.post("/api/features/test-pkg/install")

        assert resp.status_code == 200
        assert resp.json()["status"] == "installed"
        assert venv.installed["kestrel-feature-test"] == "0.4.0"
        assert venv.editable[CORE] == "/src/core"

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_install_reports_and_restores_a_replaced_core(self, mock_registry, monkeypatch):
        """An install that bypassed the pin cannot return a clean 'installed'."""
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = self._venv(monkeypatch, honours_constraints=False)

        with TestClient(_make_app(_make_agent())) as client:
            resp = client.post("/api/features/test-pkg/install")

        body = resp.json()
        assert resp.status_code == 200  # the package really did install...
        assert body["status"] == "installed_with_core_drift"  # ...but say so
        assert body["core_restored"] is True
        assert "expected: editable → /src/core" in body["core_drift"]
        assert venv.editable[CORE] == "/src/core"  # actually re-linked
        assert venv.installed[CORE] == "0.52.0"

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_failed_install_still_verifies_and_restores_core(
        self, mock_registry, monkeypatch
    ):
        """A non-zero install is not a no-op.

        pip resolves and installs dependencies BEFORE the requested package, so
        a build failure can leave core already swapped. Returning the install
        error without checking would leave that swap in place, unnamed.
        """
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = self._venv(
            monkeypatch, honours_constraints=False, feature_install_fails=True,
        )

        with TestClient(_make_app(_make_agent()), raise_server_exceptions=False) as client:
            resp = client.post("/api/features/test-pkg/install")

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "Installation failed" in detail
        assert "was replaced during the install batch" in detail
        assert (
            f"restored: uv pip install --python {shlex.quote(sys.executable)} "
            "-e /src/core"
        ) in detail
        assert venv.editable[CORE] == "/src/core"  # repaired despite the failure

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_timed_out_install_still_verifies_and_restores_core(
        self, mock_registry, monkeypatch
    ):
        """A killed install leaves whatever it had already written — including
        a swapped core. The timeout response must not skip the check."""
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = self._venv(
            monkeypatch, honours_constraints=False, feature_install_times_out=True,
        )

        with TestClient(_make_app(_make_agent()), raise_server_exceptions=False) as client:
            resp = client.post("/api/features/test-pkg/install")

        assert resp.status_code == 504
        detail = resp.json()["detail"]
        assert "timed out" in detail
        assert "was replaced during the install batch" in detail
        assert venv.editable[CORE] == "/src/core"  # repaired despite the timeout

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_install_fails_closed_when_core_cannot_be_restored(
        self, mock_registry, monkeypatch
    ):
        """The worst case: the package installed, core was replaced, and the
        re-link failed. The host is running a core nobody declared — that is
        not a 2xx, whatever happened to the package."""
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = self._venv(
            monkeypatch, honours_constraints=False, repair_fails=True,
        )

        with TestClient(_make_app(_make_agent()), raise_server_exceptions=False) as client:
            resp = client.post("/api/features/test-pkg/install")

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "was replaced during the install batch" in detail
        # The operator's command, verbatim — the response is the only place
        # they will see it.
        assert (
            "RESTORE FAILED — run `uv pip install --python "
            f"{shlex.quote(sys.executable)} -e /src/core` by hand."
        ) in detail
        assert venv.editable.get(CORE) is None  # still swapped — reported, not hidden
        assert venv.installed["kestrel-feature-test"] == "0.4.0"


# ---------------------------------------------------------------------------
# POST /api/features/{name}/remove
# ---------------------------------------------------------------------------


class TestRemoveFeature:
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    def test_remove_core_returns_400(self, mock_pkg):
        mock_pkg.return_value = FeaturePackageInfo(
            name="core",
            package="kestrel-sovereign",
            git="https://example.com",
            features=["CoreFeature"],
            description="Core",
            core=True,
        )
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/CoreFeature/remove")

        assert resp.status_code == 400
        assert "core" in resp.json()["detail"].lower()

    @pytest.mark.parametrize(
        "feature_name",
        ["ConstitutionFeature", "PeersFeature", "SecurityFeature"],
    )
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    def test_remove_mandatory_feature_is_rejected_before_teardown(
        self, mock_pkg, feature_name
    ):
        # A non-core package that (hypothetically) declares a mandatory feature
        # must be refused with 409 BEFORE any teardown / on_remove runs — the
        # canonical unload teardown would otherwise cripple the agent (#2522).
        mock_pkg.return_value = FeaturePackageInfo(
            name="rogue-pkg",
            package="kestrel-feature-rogue",
            git="",
            features=[feature_name],
            description="rogue package claiming a mandatory feature",
            core=False,
        )
        feature = _make_feature(name=feature_name)
        agent = _lifecycle_agent(features={feature_name: feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post(f"/api/features/{feature_name}/remove")

        assert resp.status_code == 409
        assert feature_name in resp.json()["detail"]
        # Nothing was torn down and no stored-data cleanup ran.
        feature.shutdown.assert_not_awaited()
        feature.on_remove.assert_not_awaited()
        assert agent.features.get(feature_name) is feature

    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    def test_remove_unknown_returns_404(self, mock_pkg):
        mock_pkg.return_value = None
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/Unknown/remove")

        assert resp.status_code == 404

    @patch("kestrel_sovereign.endpoints.features.subprocess.run")
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    def test_remove_accepts_package_stable_id(self, mock_pkg, mock_registry, mock_run):
        # REAL agent: removal delegates its per-member drain to the canonical
        # ``_unregister_feature_runtime`` (#2522 P1), so this must drive the real
        # method rather than a mocked-away teardown.
        mock_pkg.return_value = None
        info = FeaturePackageInfo(
            name="multi-pkg", package="kestrel-feature-multi", git="",
            features=["FirstFeature", "SecondFeature"], description="multi",
        )
        mock_registry.return_value = {"multi-pkg": info}
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        first = _make_feature(name="FirstFeature")
        second = _make_feature(name="SecondFeature")
        agent = _lifecycle_agent(
            features={"FirstFeature": first, "SecondFeature": second}
        )
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/multi-pkg/remove")

        assert resp.status_code == 200
        assert resp.json()["features"] == ["FirstFeature", "SecondFeature"]
        # Canonical teardown drained each member (shutdown ran), the instances
        # were unloaded, and on_remove ran AFTER teardown on the same instance.
        first.shutdown.assert_awaited_once()
        second.shutdown.assert_awaited_once()
        first.on_remove.assert_awaited_once()
        second.on_remove.assert_awaited_once()
        assert "FirstFeature" not in agent.features
        assert "SecondFeature" not in agent.features
        command = mock_run.call_args.args[0]
        assert command[-2:] == ["-y", "kestrel-feature-multi"]

    @patch("kestrel_sovereign.endpoints.features.subprocess.run")
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    def test_remove_by_class_cleans_every_loaded_package_member(self, mock_pkg, mock_run):
        info = FeaturePackageInfo(
            name="multi-pkg", package="kestrel-feature-multi", git="",
            features=["FirstFeature", "SecondFeature"], description="multi",
        )
        mock_pkg.return_value = info
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        first = _make_feature(name="FirstFeature")
        second = _make_feature(name="SecondFeature")
        agent = _lifecycle_agent(
            features={"FirstFeature": first, "SecondFeature": second}
        )
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/FirstFeature/remove")

        assert resp.status_code == 200
        # Both package members drained (canonical teardown) then cleaned up.
        first.shutdown.assert_awaited_once()
        second.shutdown.assert_awaited_once()
        first.on_remove.assert_awaited_once()
        second.on_remove.assert_awaited_once()
        assert first.enabled is False and second.enabled is False
        assert agent.features == {}


# ---------------------------------------------------------------------------
# GET /api/features/{name}/config
# ---------------------------------------------------------------------------


class TestGetFeatureConfig:
    def test_returns_config_and_schema(self):
        schema = {"type": "object", "properties": {"enabled": {"type": "boolean"}}}
        feature = _make_feature(config_schema=schema, config={"enabled": True})
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/TestFeature/config")

        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["enabled"] is True
        assert data["config_schema"] is not None

    def test_unknown_feature_returns_404(self):
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/Unknown/config")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /api/features/{name}/config
# ---------------------------------------------------------------------------


class TestUpdateFeatureConfig:
    def test_updates_config(self):
        feature = _make_feature(config={"enabled": True})
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"enabled": False}},
            )

        assert resp.status_code == 200
        feature.set_config.assert_awaited_once_with({"enabled": False})

    def test_validates_required_fields(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        feature = _make_feature(config_schema=schema)
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {}},
            )

        assert resp.status_code == 422
        assert "name" in resp.json()["detail"]

    def test_validates_field_types(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        feature = _make_feature(config_schema=schema)
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"count": "not-a-number"}},
            )

        assert resp.status_code == 422
        assert "count" in resp.json()["detail"]

    def test_validates_minimum(self):
        schema = {
            "type": "object",
            "properties": {"risk": {"type": "integer", "minimum": 0, "maximum": 10}},
        }
        feature = _make_feature(config_schema=schema)
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"risk": -1}},
            )

        assert resp.status_code == 422
        assert ">=" in resp.json()["detail"]

    def test_validates_maximum(self):
        schema = {
            "type": "object",
            "properties": {"risk": {"type": "integer", "minimum": 0, "maximum": 10}},
        }
        feature = _make_feature(config_schema=schema)
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"risk": 11}},
            )

        assert resp.status_code == 422
        assert "<=" in resp.json()["detail"]

    def test_validates_enum(self):
        schema = {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
        }
        feature = _make_feature(config_schema=schema)
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"mode": "turbo"}},
            )

        assert resp.status_code == 422
        assert "one of" in resp.json()["detail"]

    def test_valid_enum_passes(self):
        schema = {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
        }
        feature = _make_feature(config_schema=schema, config={"mode": "fast"})
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"mode": "slow"}},
            )

        assert resp.status_code == 200

    def test_valid_min_max_passes(self):
        schema = {
            "type": "object",
            "properties": {"risk": {"type": "integer", "minimum": 0, "maximum": 10}},
        }
        feature = _make_feature(config_schema=schema, config={"risk": 5})
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"risk": 5}},
            )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Secret masking / UI hints (#2045)
# ---------------------------------------------------------------------------


SECRET_SCHEMA = {
    "type": "object",
    "properties": {
        "api_key": {"type": "string", "writeOnly": True, "format": "password"},
        "status": {"type": "string", "readOnly": True},
        "enabled": {"type": "boolean"},
    },
}


class TestSecretMasking:
    def test_get_config_strips_secret_and_reports_presence(self):
        feature = _make_feature(
            config_schema=SECRET_SCHEMA,
            config={"api_key": "sk-super-secret", "status": "Connected", "enabled": True},
        )
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/TestFeature/config")

        assert resp.status_code == 200
        data = resp.json()
        # Secret value never returned in plaintext.
        assert "api_key" not in data["config"]
        # Presence is surfaced for the UI.
        assert data["secrets_set"]["api_key"] is True
        # Non-secret fields pass through.
        assert data["config"]["status"] == "Connected"
        assert data["config"]["enabled"] is True

    def test_get_config_reports_unset_secret(self):
        feature = _make_feature(
            config_schema=SECRET_SCHEMA,
            config={"status": "Not configured"},
        )
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/TestFeature/config")

        data = resp.json()
        assert data["secrets_set"]["api_key"] is False

    def test_patch_omitted_secret_preserves_stored_value(self):
        feature = _make_feature(
            config_schema=SECRET_SCHEMA,
            config={"api_key": "stored-key", "enabled": True},
        )
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"enabled": False}},
            )

        assert resp.status_code == 200
        # The stored secret is re-injected so it is not cleared.
        saved = feature.set_config.await_args.args[0]
        assert saved["api_key"] == "stored-key"
        assert saved["enabled"] is False

    def test_patch_delegates_isolated_secret_preservation_to_atomic_feature_method(self):
        """Hosted isolated features own preservation at their stage CAS boundary."""

        feature = _make_feature(
            config_schema=SECRET_SCHEMA,
            config={"api_key": "stored-key", "enabled": False},
        )
        feature.set_config_with_secret_preservation = AsyncMock()
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"enabled": True}},
            )

        assert resp.status_code == 200
        feature.set_config_with_secret_preservation.assert_awaited_once()
        saved, secret_fields, validate = feature.set_config_with_secret_preservation.await_args.args
        assert saved == {"enabled": True}
        assert secret_fields == {"api_key"}
        # The endpoint does not read/re-inject a secret before delegating. The
        # runtime invokes this validation only after it merged the stage-CAS
        # snapshot's current secret.
        validate({"enabled": True, "api_key": "atomic-key"})
        feature.set_config.assert_not_awaited()

    def test_patch_new_secret_overrides(self):
        feature = _make_feature(
            config_schema=SECRET_SCHEMA,
            config={"api_key": "old-key"},
        )
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"api_key": "new-key"}},
            )

        assert resp.status_code == 200
        saved = feature.set_config.await_args.args[0]
        assert saved["api_key"] == "new-key"

    def test_patch_response_does_not_echo_secret(self):
        feature = _make_feature(
            config_schema=SECRET_SCHEMA,
            config={"api_key": "stored-key", "enabled": True},
        )
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"enabled": True}},
            )

        assert resp.status_code == 200
        assert "api_key" not in resp.json()["config"]


# ---------------------------------------------------------------------------
# GET /api/features/{name}/skills
# ---------------------------------------------------------------------------


class TestGetFeatureSkills:
    def test_live_skills_from_loaded_feature(self):
        tool = _make_tool(name="my_skill", description="Does stuff")
        feature = _make_feature(tools=[tool])
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/TestFeature/skills")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "live"
        assert data["count"] == 1
        assert data["skills"][0]["name"] == "my_skill"

    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    def test_registry_skills_for_unloaded_feature(self, mock_pkg):
        mock_pkg.return_value = FAKE_REGISTRY["test-pkg"]
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/TestFeature/skills")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "registry"
        assert data["count"] == 1
        assert data["skills"][0]["name"] == "do_thing"

    @patch("kestrel_sovereign.endpoints.features.get_skills_for_package")
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    def test_unknown_feature_returns_404(self, mock_pkg, mock_skills):
        mock_pkg.return_value = None
        mock_skills.return_value = []
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/Unknown/skills")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/skills
# ---------------------------------------------------------------------------


class TestListAllSkills:
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_returns_live_and_registry_skills(self, mock_registry):
        mock_registry.return_value = dict(FAKE_REGISTRY)
        tool = _make_tool(name="live_tool")
        feature = _make_feature(tools=[tool])
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/skills")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        names = {s["name"] for s in data["skills"]}
        assert "live_tool" in names

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_filter_by_category(self, mock_registry):
        mock_registry.return_value = {}
        tool = _make_tool(name="sys_tool", category="system")
        feature = _make_feature(tools=[tool])
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/skills?category=system")

        data = resp.json()
        assert data["count"] == 1
        assert data["skills"][0]["name"] == "sys_tool"

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_filter_excludes_non_matching(self, mock_registry):
        mock_registry.return_value = {}
        tool = _make_tool(name="sys_tool", category="system")
        feature = _make_feature(tools=[tool])
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/skills?category=voice")

        data = resp.json()
        assert data["count"] == 0


# ---------------------------------------------------------------------------
# GET /api/skills/{skill_id}/schema
# ---------------------------------------------------------------------------


class TestGetSkillSchema:
    def test_returns_function_calling_schema(self):
        tool = _make_tool(
            name="my_skill",
            description="Does stuff",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        feature = _make_feature(tools=[tool])
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/skills/my_skill/schema")

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "function"
        assert data["function"]["name"] == "my_skill"
        assert data["function"]["description"] == "Does stuff"
        assert "x" in data["function"]["parameters"]["properties"]
        assert data["feature"] == "TestFeature"

    @patch("kestrel_sovereign.endpoints.features.get_all_skills")
    def test_falls_back_to_registry(self, mock_skills):
        mock_skills.return_value = [
            SkillInfo(name="reg_skill", description="From registry", category="system"),
        ]
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/skills/reg_skill/schema")

        assert resp.status_code == 200
        data = resp.json()
        assert data["function"]["name"] == "reg_skill"
        assert data["source"] == "registry"

    @patch("kestrel_sovereign.endpoints.features.get_all_skills")
    def test_unknown_skill_returns_404(self, mock_skills):
        mock_skills.return_value = []
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/skills/nonexistent/schema")

        assert resp.status_code == 404
