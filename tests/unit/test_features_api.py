"""Tests for the Feature Store API endpoints (endpoints/features.py)."""

from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from endpoints.features import router as features_router
from kestrel_sovereign.feature_registry import (
    FeaturePackageInfo,
    FeatureStatus,
    SkillInfo,
)


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

    @patch("endpoints.features.get_registry")
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

    @patch("endpoints.features.get_registry")
    def test_filter_by_tag(self, mock_registry):
        mock_registry.return_value = dict(FAKE_REGISTRY)
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features?tag=core")

        data = resp.json()
        assert data["count"] == 1
        assert data["features"][0]["name"] == "core-pkg"


# ---------------------------------------------------------------------------
# GET /api/features/installed
# ---------------------------------------------------------------------------


class TestListInstalledFeatures:
    @patch("endpoints.features.get_package_for_feature")
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

    @patch("endpoints.features.get_registry")
    def test_unloaded_feature_from_registry(self, mock_registry):
        mock_registry.return_value = dict(FAKE_REGISTRY)
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/features/TestFeature")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-pkg"

    @patch("endpoints.features.get_registry")
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
        feature = _make_feature()
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/TestFeature/enable")

        assert resp.status_code == 200
        assert resp.json()["status"] == "enabled"
        feature.on_enable.assert_awaited_once()

    def test_enable_unknown_feature_returns_404(self):
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/Unknown/enable")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/features/{name}/disable
# ---------------------------------------------------------------------------


class TestDisableFeature:
    def test_disable_calls_on_disable(self):
        feature = _make_feature()
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/TestFeature/disable")

        assert resp.status_code == 200
        assert resp.json()["status"] == "disabled"
        feature.on_disable.assert_awaited_once()


# ---------------------------------------------------------------------------
# POST /api/features/{name}/install
# ---------------------------------------------------------------------------


class TestInstallFeature:
    @patch("endpoints.features.get_registry")
    def test_install_core_returns_400(self, mock_registry):
        mock_registry.return_value = dict(FAKE_REGISTRY)
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/CoreFeature/install")

        assert resp.status_code == 400
        assert "core" in resp.json()["detail"].lower()

    @patch("endpoints.features.get_registry")
    def test_install_unknown_returns_404(self, mock_registry):
        mock_registry.return_value = {}
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/Unknown/install")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/features/{name}/remove
# ---------------------------------------------------------------------------


class TestRemoveFeature:
    @patch("endpoints.features.get_package_for_feature")
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

    @patch("endpoints.features.get_package_for_feature")
    def test_remove_unknown_returns_404(self, mock_pkg):
        mock_pkg.return_value = None
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.post("/api/features/Unknown/remove")

        assert resp.status_code == 404


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

    @patch("endpoints.features.get_package_for_feature")
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

    @patch("endpoints.features.get_skills_for_package")
    @patch("endpoints.features.get_package_for_feature")
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
    @patch("endpoints.features.get_registry")
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

    @patch("endpoints.features.get_registry")
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

    @patch("endpoints.features.get_registry")
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

    @patch("endpoints.features.get_all_skills")
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

    @patch("endpoints.features.get_all_skills")
    def test_unknown_skill_returns_404(self, mock_skills):
        mock_skills.return_value = []
        agent = _make_agent()
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.get("/api/skills/nonexistent/schema")

        assert resp.status_code == 404
