"""Tests for the Feature Store API endpoints (endpoints/features.py)."""

import asyncio
import shlex
import sys
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from kestrel_sovereign.auth import AuthMethod, CallerContext, CallerRole

from kestrel_sovereign import cli
from kestrel_sovereign.endpoints import features as features_endpoint
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


_UNSET = object()


def _make_app(agent=None, caller=_UNSET):
    """Create a FastAPI app with the features router mounted.

    ``caller`` stands in for the auth middleware that populates
    ``request.state.caller`` in ``server.py``. It defaults to a sovereign
    caller because install and remove now require sovereign authority
    (#3214) and the tests below are about install mechanics, not about
    who may ask. Pass an explicit caller — including ``None`` for "no
    middleware ran" — to exercise the authority gate itself.
    """
    app = FastAPI()

    resolved = CallerContext.sovereign() if caller is _UNSET else caller

    @app.middleware("http")
    async def _attach_caller(request, call_next):
        request.state.caller = resolved
        return await call_next(request)

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


async def _propagate_cancelled_child(*_args):
    """Raise cancellation created inside feature-owned async work."""

    child = asyncio.create_task(asyncio.sleep(0))
    child.cancel()
    await child


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
    @pytest.mark.asyncio
    async def test_isolated_reenable_settles_declared_config_before_opening_ingress(
        self,
    ):
        """Re-enable cannot drain a callback queued on its conversation lock."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class IsolatedOrderingFeature(SDKFixtureFeature):
            contribution_prefix = "isolated-enable-config-order"
            config_schema = {"type": "object", "additionalProperties": True}
            _apply_host_config_before_initialize = True

            def __init__(self, agent):
                super().__init__(agent)
                self.config = {"mode": "old"}
                self.ingress_open = False
                self.ingress_started = asyncio.Event()
                self.ingress_finished = asyncio.Event()
                self.drain_waited_on_conversation = asyncio.Event()
                self.rescue_deadlock = asyncio.Event()
                self.ingress_task = None
                self.call_order = []

            async def initialize(self):
                self.call_order.append("initialize")
                self.ingress_open = True

                async def inbound_callback():
                    self.ingress_started.set()
                    async with self.agent.feature_config_transition():
                        self.ingress_finished.set()

                self.ingress_task = asyncio.create_task(inbound_callback())
                await self.ingress_started.wait()
                # Let the callback queue on the transition's CONVERSATION lock
                # before initialize returns to the activation sequence.
                await asyncio.sleep(0)

            async def get_config(self):
                return dict(self.config)

            async def set_config(self, config):
                self.call_order.append("set_config")
                if self.ingress_open and not self.ingress_finished.is_set():
                    # A real proxy drains its admitted callback here. The bounded
                    # rescue makes the pre-fix lock cycle observable without
                    # leaving a permanently hung pytest task.
                    self.drain_waited_on_conversation.set()
                    await self.rescue_deadlock.wait()
                self.config = dict(config)

        agent = _lifecycle_agent()
        feature = IsolatedOrderingFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature
        agent._declared_feature_config = lambda _name: {"mode": "declared"}
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        async def bounded_rescue():
            await asyncio.sleep(0.05)
            feature.rescue_deadlock.set()

        rescue = asyncio.create_task(bounded_rescue())
        try:
            response = await asyncio.wait_for(
                features_endpoint.enable_feature(request, feature.name),
                timeout=1,
            )
            assert response["status"] == "enabled"
            assert feature.call_order == ["set_config", "initialize"]
            assert not feature.drain_waited_on_conversation.is_set()
            await asyncio.wait_for(feature.ingress_task, timeout=1)
            assert feature.ingress_finished.is_set()
        finally:
            feature.rescue_deadlock.set()
            await rescue
            if feature.ingress_task is not None and not feature.ingress_task.done():
                feature.ingress_task.cancel()
                with suppress(asyncio.CancelledError):
                    await feature.ingress_task

    @pytest.mark.asyncio
    async def test_enable_and_config_use_ingress_before_conversation_lock(self):
        """A config PATCH cannot deadlock a concurrent runtime re-enable."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class IngressFeature(SDKFixtureFeature):
            contribution_prefix = "enable-config-lock-order"
            config_schema = {"type": "object", "additionalProperties": True}

            def __init__(self, agent):
                super().__init__(agent)
                self._config_ingress_lock = asyncio.Lock()
                self.activation_started = asyncio.Event()
                self.release_activation = asyncio.Event()
                self.config = {"mode": "old"}
                self._active_lease = None
                self._authorized_task = None

            async def initialize(self):
                self.activation_started.set()
                await self.release_activation.wait()

            @asynccontextmanager
            async def config_transition_ingress_fence(self):
                async with self._config_ingress_lock:
                    lease = object()
                    self._active_lease = lease
                    try:
                        yield lease
                    finally:
                        self._active_lease = None
                        self._authorized_task = None

            def claim_config_transition_ingress_fence(self, lease):
                if lease is not self._active_lease:
                    return False
                self._authorized_task = asyncio.current_task()
                return True

            async def get_config(self):
                return dict(self.config)

            async def set_config(self, config):
                if asyncio.current_task() is self._authorized_task:
                    self.config = dict(config)
                    return
                async with self._config_ingress_lock:
                    self.config = dict(config)

        agent = _lifecycle_agent()
        feature = IngressFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature
        agent._declared_feature_config = lambda _name: {"mode": "declared"}
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        enable = asyncio.create_task(
            features_endpoint.enable_feature(request, feature.name)
        )
        await asyncio.wait_for(feature.activation_started.wait(), timeout=1)
        update = asyncio.create_task(
            features_endpoint.update_feature_config(
                request,
                feature.name,
                features_endpoint.ConfigUpdateRequest(config={"mode": "patched"}),
            )
        )
        # Give the competing request a chance to take the wrong first lock.
        await asyncio.sleep(0.05)
        feature.release_activation.set()

        _done, pending = await asyncio.wait({enable, update}, timeout=0.5)
        deadlocked = bool(pending)
        if deadlocked:
            # A queued PATCH owns ingress while enable owns CONVERSATION. Cancel
            # the queued PATCH so its fence unwinds and the test can fail cleanly.
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(update, timeout=1)
            await asyncio.wait_for(enable, timeout=1)

        assert not deadlocked
        assert (await enable)["status"] == "enabled"
        assert (await update)["config"] == {"mode": "patched"}

    @pytest.mark.asyncio
    async def test_queued_cognition_rechecks_safe_mode_after_transition_lock(self):
        """A turn admitted before a quarantine latch cannot run afterward."""

        agent = _lifecycle_agent()
        agent.storage = object()
        agent.context_manager = object()
        agent.bootstrap_service = None
        agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
        agent._maybe_audit = AsyncMock()
        agent._process_input_traced_locked = AsyncMock(return_value="MODEL-RAN")
        real_turn_lifecycle = agent._turn_lifecycle
        lock_attempted = asyncio.Event()

        @asynccontextmanager
        async def observed_turn_lifecycle():
            lock_attempted.set()
            async with real_turn_lifecycle():
                yield

        agent._turn_lifecycle = observed_turn_lifecycle

        async with real_turn_lifecycle():
            turn = asyncio.create_task(agent.process_input("queued cognition"))
            await asyncio.wait_for(lock_attempted.wait(), timeout=1)
            assert not turn.done()
            agent._safe_mode = True
            agent._safe_mode_reason = "feature contribution quarantine failed"

        response = await asyncio.wait_for(turn, timeout=1)

        assert "SAFE MODE ACTIVE" in response
        agent._process_input_traced_locked.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_queued_stream_rechecks_safe_mode_after_transition_lock(self):
        """The streaming entry point shares the post-lock cognition guard."""

        agent = _lifecycle_agent()
        agent.storage = object()
        agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
        agent._maybe_audit = AsyncMock()
        model_started = False
        real_turn_lifecycle = agent._turn_lifecycle
        lock_attempted = asyncio.Event()

        @asynccontextmanager
        async def observed_turn_lifecycle():
            lock_attempted.set()
            async with real_turn_lifecycle():
                yield

        agent._turn_lifecycle = observed_turn_lifecycle

        async def stream_model(*_args, **_kwargs):
            nonlocal model_started
            model_started = True
            yield "MODEL-RAN"

        agent._process_input_streaming_traced_locked = stream_model

        async def collect():
            return [
                chunk
                async for chunk in agent.process_input_streaming(
                    "queued streaming cognition"
                )
            ]

        async with real_turn_lifecycle():
            turn = asyncio.create_task(collect())
            await asyncio.wait_for(lock_attempted.wait(), timeout=1)
            assert not turn.done()
            agent._safe_mode = True
            agent._safe_mode_reason = "feature contribution quarantine failed"

        chunks = await asyncio.wait_for(turn, timeout=1)

        assert len(chunks) == 1
        assert "SAFE MODE ACTIVE" in chunks[0]
        assert model_started is False

    @pytest.mark.asyncio
    async def test_enable_hook_can_reenter_privacy_transition(self):
        """The owned mutation task must own the conversation boundary.

        A feature hook is allowed to perform a privacy-governed write.  If the
        HTTP parent owns ``CONVERSATION`` while the hook runs in the shielded
        child task, the child waits on its parent forever.
        """

        feature = _make_feature(enabled=False)
        agent = _lifecycle_agent(features={"TestFeature": feature})
        entered = asyncio.Event()

        async def enable_with_privacy_transition():
            async with agent.privacy_transition():
                entered.set()

        feature.on_enable.side_effect = enable_with_privacy_transition
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        response = await asyncio.wait_for(
            features_endpoint.enable_feature(request, "TestFeature"),
            timeout=1,
        )

        assert response["status"] == "enabled"
        assert entered.is_set()

    @pytest.mark.asyncio
    async def test_enable_hook_cognition_is_rejected_before_partial_publication(self):
        """A pre-commit hook cannot inspect a half-published generation."""

        feature = _make_feature(enabled=False)
        agent = _lifecycle_agent(features={"TestFeature": feature})
        observed = []

        async def enable_with_cognition():
            async with agent._turn_lifecycle():
                observed.append(feature.enabled)

        feature.on_enable.side_effect = enable_with_cognition
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(
            RuntimeError,
            match="feature transition generation is fully committed",
        ):
            await asyncio.wait_for(
                features_endpoint.enable_feature(request, "TestFeature"),
                timeout=1,
            )

        assert observed == []
        assert feature.enabled is False

    @pytest.mark.asyncio
    async def test_enable_hook_child_cognition_is_rejected_without_deadlock(self):
        """Transition ancestry must fail closed across an inherited task context."""

        feature = _make_feature(enabled=False)
        agent = _lifecycle_agent(features={"TestFeature": feature})
        child_started = asyncio.Event()
        cognition_task = None

        async def child_cognition():
            child_started.set()
            async with agent._turn_lifecycle():
                raise AssertionError("pre-commit child cognition was admitted")

        async def enable_with_child_cognition():
            nonlocal cognition_task
            cognition_task = asyncio.create_task(child_cognition())
            await cognition_task

        feature.on_enable.side_effect = enable_with_child_cognition
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )
        operation = asyncio.create_task(
            features_endpoint.enable_feature(request, "TestFeature")
        )

        try:
            await asyncio.wait_for(child_started.wait(), timeout=1)
            with pytest.raises(
                RuntimeError,
                match="feature transition generation is fully committed",
            ):
                await asyncio.wait_for(asyncio.shield(operation), timeout=0.2)
        finally:
            if cognition_task is not None and not cognition_task.done():
                cognition_task.cancel()
                await asyncio.gather(cognition_task, return_exceptions=True)
            if not operation.done():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)

        assert feature.enabled is False

    @pytest.mark.asyncio
    async def test_enable_hook_detached_child_cannot_outlive_transition_ancestry(self):
        """A descendant cannot wait for commit and reuse stale hook authority."""

        feature = _make_feature(enabled=False)
        agent = _lifecycle_agent(features={"TestFeature": feature})
        release_child = asyncio.Event()
        child_started = asyncio.Event()
        cognition_entered = False
        detached_task = None

        async def delayed_child_cognition():
            nonlocal cognition_entered
            child_started.set()
            await release_child.wait()
            async with agent._turn_lifecycle():
                cognition_entered = True

        async def enable_with_detached_child():
            nonlocal detached_task
            detached_task = asyncio.create_task(delayed_child_cognition())
            await child_started.wait()

        feature.on_enable.side_effect = enable_with_detached_child
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        response = await features_endpoint.enable_feature(request, "TestFeature")
        assert response["status"] == "enabled"
        assert detached_task is not None

        release_child.set()
        with pytest.raises(RuntimeError, match="expired feature transition"):
            await asyncio.wait_for(detached_task, timeout=1)

        assert cognition_entered is False

    @pytest.mark.asyncio
    async def test_committed_enable_ready_hook_can_await_cognition(self):
        """The explicit post-commit ready seam remains cognition-capable."""

        feature = _make_feature(enabled=False)
        agent = _lifecycle_agent(features={"TestFeature": feature})
        observed = []

        async def ready_with_cognition(_agent):
            async with agent._turn_lifecycle():
                observed.append(feature.enabled)

        feature.on_agent_ready = AsyncMock(side_effect=ready_with_cognition)
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        response = await asyncio.wait_for(
            features_endpoint.enable_feature(request, "TestFeature"),
            timeout=1,
        )

        assert response["status"] == "enabled"
        assert observed == [True]

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    async def test_package_ready_hook_observes_complete_enabled_generation(
        self, mock_registry
    ):
        """No package ready hook can enter cognition between member commits."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class FirstReadyFeature(SDKFixtureFeature):
            contribution_prefix = "ready-first"

        class SecondReadyFeature(SDKFixtureFeature):
            contribution_prefix = "ready-second"

        agent = _lifecycle_agent()
        first = FirstReadyFeature(agent)
        second = SecondReadyFeature(agent)
        first.enabled = False
        second.enabled = False
        observations = []

        async def observe_complete_generation(_agent):
            async with agent._turn_lifecycle():
                observations.append(
                    (
                        first.enabled,
                        second.enabled,
                        len(
                            agent.feature_contribution_runtime.active_context_clauses()
                        ),
                    )
                )

        first.on_agent_ready = observe_complete_generation
        agent.features = {first.name: first, second.name: second}
        info = FeaturePackageInfo(
            name="ready-pkg",
            package="kestrel-feature-ready",
            git="",
            features=[first.name, second.name],
            description="ready generation fixture",
        )
        mock_registry.return_value = {info.name: info}
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        response = await asyncio.wait_for(
            features_endpoint.enable_feature(request, info.name),
            timeout=1,
        )

        assert response["status"] == "enabled"
        assert observations == [(True, True, 2)]

    @pytest.mark.asyncio
    async def test_enable_waits_for_active_turn_before_publication(self):
        """No contribution can become prompt-visible mid-turn."""

        feature = _make_feature(enabled=False)
        agent = _lifecycle_agent(features={"TestFeature": feature})
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        async with agent._turn_lifecycle():
            enable = asyncio.create_task(
                features_endpoint.enable_feature(request, "TestFeature")
            )
            await asyncio.sleep(0)
            feature.initialize.assert_not_awaited()
            assert feature.enabled is False

        response = await asyncio.wait_for(enable, timeout=1)
        assert response["status"] == "enabled"
        feature.initialize.assert_awaited_once()
        assert feature.enabled is True

    @pytest.mark.asyncio
    async def test_cancelled_enable_settles_published_generation_before_unlock(self):
        """A disconnect after publication cannot expose a half-enabled feature."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_enable():
            entered.set()
            await release.wait()

        feature.on_enable = slow_enable
        agent.features[feature.name] = feature
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        enable = asyncio.create_task(
            features_endpoint.enable_feature(request, feature.name)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert feature.enabled is False
        assert agent.feature_contribution_runtime.active_context_clauses()

        enable.cancel()
        await asyncio.sleep(0)
        assert not enable.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(enable, timeout=1)

        assert feature.enabled is True
        assert agent.features[feature.name] is feature
        assert agent.feature_contribution_runtime.active_context_clauses()

    @pytest.mark.asyncio
    async def test_hook_cancelled_error_rolls_back_enable_generation(self):
        """Feature-owned cancellation is a failed activation, not a disconnect."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        feature.on_enable = _propagate_cancelled_child
        agent.features[feature.name] = feature
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(asyncio.CancelledError):
            await features_endpoint.enable_feature(request, feature.name)

        assert feature.enabled is False
        assert agent.features[feature.name] is feature
        assert not agent.feature_contribution_runtime.active_context_clauses()

    @pytest.mark.asyncio
    async def test_failed_activation_quarantines_context_after_teardown_drift(self):
        """A failed enable cannot retain prompt bytes when exact cleanup drifts."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class DriftedActivationFeature(SDKFixtureFeature):
            contribution_prefix = "drifted-activation-fixture"

        agent = _lifecycle_agent()
        feature = DriftedActivationFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature

        async def drift_then_fail():
            agent.wait_registry.deregister(
                feature.wait_provider.kind,
                feature.wait_provider,
            )
            raise RuntimeError("activation failed after contribution drift")

        feature.on_enable = drift_then_fail

        with pytest.raises(
            RuntimeError,
            match="activation failed after contribution drift",
        ):
            await agent._activate_feature_runtime(feature)

        runtime = agent.feature_contribution_runtime
        assert feature.enabled is False
        assert not runtime.is_active(feature)
        assert runtime.active_context_clauses() == ()

    @pytest.mark.asyncio
    async def test_failed_activation_latches_safe_mode_if_quarantine_fails(
        self, monkeypatch
    ):
        """Unrepairable activation ownership blocks later cognition."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class UnrepairableActivationFeature(SDKFixtureFeature):
            contribution_prefix = "unrepairable-activation-fixture"

        agent = _lifecycle_agent()
        feature = UnrepairableActivationFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature

        async def drift_then_fail():
            agent.wait_registry.deregister(
                feature.wait_provider.kind,
                feature.wait_provider,
            )
            raise RuntimeError("activation failure must be superseded")

        feature.on_enable = drift_then_fail
        monkeypatch.setattr(
            agent,
            "_quarantine_feature_contributions",
            MagicMock(side_effect=RuntimeError("private quarantine failure")),
        )

        with pytest.raises(
            RuntimeError,
            match="failed activation contributions could not be quarantined",
        ):
            await agent._activate_feature_runtime(feature)

        assert agent._safe_mode is True
        assert "quarantine failed" in agent._safe_mode_reason
        assert agent._safe_mode_cause == "feature_lifecycle_uncertain"
        assert agent._feature_lifecycle_integrity_uncertain is True

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    async def test_package_hook_cancellation_rolls_back_prior_enabled_member(
        self, mock_registry
    ):
        """A later member's cancellation cannot strand an earlier member live."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class CancellingFeature(SDKFixtureFeature):
            contribution_prefix = "cancelling-agent-fixture"

        agent = _lifecycle_agent()
        first = SDKFixtureFeature(agent)
        second = CancellingFeature(agent)
        first.enabled = False
        second.enabled = False
        second.on_enable = _propagate_cancelled_child
        agent.features = {first.name: first, second.name: second}
        mock_registry.return_value = {
            "fixture-pkg": FeaturePackageInfo(
                name="fixture-pkg",
                package="kestrel-feature-fixture",
                git="",
                features=[first.name, second.name],
                description="fixture",
            )
        }
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(asyncio.CancelledError):
            await features_endpoint.enable_feature(request, "fixture-pkg")

        assert first.enabled is False
        assert second.enabled is False
        assert not agent.feature_contribution_runtime.active_context_clauses()

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    async def test_package_enable_quarantines_prior_member_after_rollback_drift(
        self, mock_registry
    ):
        """A failed package enable cannot retain a disabled member's context."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class FirstPackageFeature(SDKFixtureFeature):
            contribution_prefix = "rollback-drift-first"

        class SecondPackageFeature(SDKFixtureFeature):
            contribution_prefix = "rollback-drift-second"

        agent = _lifecycle_agent()
        first = FirstPackageFeature(agent)
        second = SecondPackageFeature(agent)
        first.enabled = False
        second.enabled = False

        async def drift_first_then_fail():
            agent.wait_registry.deregister(
                first.wait_provider.kind,
                first.wait_provider,
            )
            raise RuntimeError("later package activation failed")

        second.on_enable = drift_first_then_fail
        agent.features = {first.name: first, second.name: second}
        mock_registry.return_value = {
            "rollback-drift-pkg": FeaturePackageInfo(
                name="rollback-drift-pkg",
                package="kestrel-feature-rollback-drift",
                git="",
                features=[first.name, second.name],
                description="rollback drift fixture",
            )
        }
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(RuntimeError, match="later package activation failed"):
            await features_endpoint.enable_feature(request, "rollback-drift-pkg")

        runtime = agent.feature_contribution_runtime
        assert first.enabled is False
        assert second.enabled is False
        assert not runtime.is_active(first)
        assert not runtime.is_active(second)
        assert runtime.active_context_clauses() == ()

    @pytest.mark.asyncio
    async def test_quarantine_safe_mode_helper_latches_after_entry_failure(self):
        """Package rollback has a synchronous fail-closed compatibility latch."""

        agent = SimpleNamespace(_safe_mode=False)

        async def fail_entry(_reason):
            raise RuntimeError("safe-mode persistence unavailable")

        agent.enter_safe_mode = fail_entry
        await features_endpoint._enter_feature_quarantine_safe_mode(
            agent,
            "package quarantine failed",
        )

        assert agent._safe_mode is True
        assert agent._safe_mode_reason == "package quarantine failed"
        assert agent._safe_mode_cause == "feature_lifecycle_uncertain"
        assert agent._feature_lifecycle_integrity_uncertain is True

    @pytest.mark.asyncio
    async def test_ready_hook_cancelled_error_remains_best_effort(self):
        """The committed generation survives optional ready-hook cancellation."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        feature.on_agent_ready = _propagate_cancelled_child
        agent.features[feature.name] = feature
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        response = await features_endpoint.enable_feature(request, feature.name)

        assert response["status"] == "enabled"
        assert feature.enabled is True
        assert agent.features[feature.name] is feature
        assert agent.feature_contribution_runtime.active_context_clauses()

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
    @pytest.mark.asyncio
    async def test_disable_waits_for_active_turn_before_teardown(self):
        """A live turn retains one stable feature/context generation."""

        feature = _make_feature()
        agent = _lifecycle_agent(features={"TestFeature": feature})
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        async with agent._turn_lifecycle():
            disable = asyncio.create_task(
                features_endpoint.disable_feature(request, "TestFeature")
            )
            await asyncio.sleep(0)
            feature.on_disable.assert_not_awaited()
            assert feature.enabled is True

        response = await asyncio.wait_for(disable, timeout=1)
        assert response["status"] == "disabled"
        feature.on_disable.assert_awaited_once()
        assert feature.enabled is False

    @pytest.mark.asyncio
    async def test_cancelled_disable_settles_removed_generation_before_unlock(self):
        """A disconnect after depublication cannot expose a half-disabled feature."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature
        await agent._activate_feature_runtime(feature)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_disable():
            entered.set()
            await release.wait()

        feature.on_disable = slow_disable
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        disable = asyncio.create_task(
            features_endpoint.disable_feature(request, feature.name)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert feature.enabled is True
        assert not agent.feature_contribution_runtime.active_context_clauses()

        disable.cancel()
        await asyncio.sleep(0)
        assert not disable.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(disable, timeout=1)

        assert feature.enabled is False
        assert agent.features[feature.name] is feature
        assert not agent.feature_contribution_runtime.active_context_clauses()

    @pytest.mark.asyncio
    async def test_hook_cancelled_error_rolls_back_disable_generation(self):
        """Feature-owned cancellation cannot leave teardown half-published."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature
        await agent._activate_feature_runtime(feature)
        feature.on_disable = _propagate_cancelled_child
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(asyncio.CancelledError):
            await features_endpoint.disable_feature(request, feature.name)

        assert feature.enabled is True
        assert agent.features[feature.name] is feature
        assert agent.feature_contribution_runtime.active_context_clauses()

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    async def test_disable_failure_restores_cross_feature_setup_batch(
        self, mock_registry
    ):
        """Disable rollback retains one valid interdependent generation."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class FirstFeature(SDKFixtureFeature):
            contribution_prefix = "disable-first"

        class SecondFeature(SDKFixtureFeature):
            contribution_prefix = "disable-second"

        agent = _lifecycle_agent()
        first = FirstFeature(agent)
        second = SecondFeature(agent)
        second.setup_registration = replace(
            second.setup_registration,
            after=(first.setup_registration.name,),
        )
        transition = agent._prepare_feature_contribution_transition((first, second))
        for feature, prepared in transition.activatable((first, second)):
            await agent._activate_feature_runtime(
                feature,
                prepared_contributions=prepared,
            )

        original_shutdown = second.shutdown
        shutdown_calls = 0

        async def fail_first_shutdown():
            nonlocal shutdown_calls
            shutdown_calls += 1
            if shutdown_calls == 1:
                raise RuntimeError("second disable failed")
            await original_shutdown()

        second.shutdown = fail_first_shutdown
        observations = []

        async def observe_complete_rollback(_agent):
            async with agent._turn_lifecycle():
                observations.append(
                    (
                        first.enabled,
                        second.enabled,
                        len(
                            agent.feature_contribution_runtime.active_context_clauses()
                        ),
                    )
                )

        first.on_agent_ready = observe_complete_rollback
        info = FeaturePackageInfo(
            name="dependent-pkg",
            package="kestrel-feature-dependent",
            git="",
            features=[first.name, second.name],
            description="cross-feature setup dependency fixture",
        )
        mock_registry.return_value = {info.name: info}
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(RuntimeError, match="second disable failed"):
            await features_endpoint.disable_feature(request, info.name)

        assert agent.features[first.name] is first
        assert agent.features[second.name] is second
        assert first.enabled is True
        assert second.enabled is True
        assert agent.setup_step_registry.get(first.setup_registration.name) is (
            first.setup_registration
        )
        assert agent.setup_step_registry.get(second.setup_registration.name) is (
            second.setup_registration
        )
        assert len(agent.feature_contribution_runtime.active_context_clauses()) == 2
        assert observations == [(True, True, 2)]

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    async def test_disable_rollback_repairs_drifted_active_member(
        self, mock_registry
    ):
        """One failed exact inverse cannot block restoration of its package."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class FirstDriftFeature(SDKFixtureFeature):
            contribution_prefix = "drift-first"

        class SecondDriftFeature(SDKFixtureFeature):
            contribution_prefix = "drift-second"

        agent = _lifecycle_agent()
        first = FirstDriftFeature(agent)
        second = SecondDriftFeature(agent)
        first.enabled = False
        second.enabled = False
        transition = agent._prepare_feature_contribution_transition((first, second))
        for feature, prepared in transition.activatable((first, second)):
            await agent._activate_feature_runtime(
                feature,
                prepared_contributions=prepared,
                notify_ready=False,
            )
        agent.wait_registry.deregister(
            second.wait_provider.kind, second.wait_provider
        )
        info = FeaturePackageInfo(
            name="drift-pkg",
            package="kestrel-feature-drift",
            git="",
            features=[first.name, second.name],
            description="drifted teardown fixture",
        )
        mock_registry.return_value = {info.name: info}
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(RuntimeError, match="wait-provider"):
            await features_endpoint.disable_feature(request, info.name)

        assert first.enabled is True
        assert second.enabled is True
        assert agent.feature_contribution_runtime.is_active(first)
        assert agent.feature_contribution_runtime.is_active(second)
        assert agent.wait_registry.contains(
            second.wait_provider.kind, second.wait_provider
        )
        assert len(agent.feature_contribution_runtime.active_context_clauses()) == 2

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    async def test_disable_rollback_latches_safe_mode_on_foreign_context_drift(
        self, mock_registry
    ):
        """An unprovable rollback blocks cognition over the surviving clause."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature
        await agent._activate_feature_runtime(feature)
        runtime = agent.feature_contribution_runtime
        original = runtime.active_context_clauses()[0]
        foreign = replace(original, body="foreign replacement prompt bytes")
        runtime.context_clause_registry._clauses[original.identity] = foreign
        info = FeaturePackageInfo(
            name="foreign-drift-pkg",
            package="kestrel-feature-foreign-drift",
            git="",
            features=[feature.name],
            description="foreign context drift fixture",
        )
        mock_registry.return_value = {info.name: info}
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(RuntimeError, match="complete feature generation") as error:
            await features_endpoint.disable_feature(request, info.name)

        assert error.value.__cause__ is None
        assert agent._safe_mode is True
        assert "rollback" in agent._safe_mode_reason.lower()
        assert feature.enabled is False
        assert runtime.is_active(feature)
        assert runtime.active_context_clauses() == (foreign,)

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
        """An install that bypassed the pin cannot return a clean 'installed'.

        This feature requires core `>=0.53`, so the very act of installing it
        pulled 0.53 in and the repair put 0.52 back — which leaves the freshly
        installed package unable to load. The endpoint has a branch for exactly
        that, and until the requirement reading was shared (#3080) it could not
        fire here: it read the metadata of a package that only exists in the
        double, got nothing, and reported the completed install over an
        environment that cannot load it. The check was blind in the one
        scenario it was written for.

        There is no companion test for "drift, but the restored core satisfies
        the package", because that state cannot occur on this path: the swap
        happens precisely BECAUSE the installed core does not satisfy the
        requirement, so restoring it always leaves the requirement unmet. The
        "no drift" arm is
        `test_install_succeeds_when_the_checkout_satisfies_the_feature`.
        """
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = self._venv(monkeypatch, honours_constraints=False)

        with TestClient(_make_app(_make_agent()), raise_server_exceptions=False) as client:
            resp = client.post("/api/features/test-pkg/install")

        detail = resp.json()["detail"]
        assert resp.status_code == 500
        assert "installed and then moved core" in detail
        assert "present but cannot load" in detail
        assert f"{CORE}>=0.53" in detail  # names the requirement that cannot be met
        assert venv.editable[CORE] == "/src/core"  # core was actually re-linked
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

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_a_hung_repair_is_bounded_not_a_wedged_request(
        self, mock_registry, monkeypatch
    ):
        """The repair runs a SECOND installer, and it must be bounded too.

        The endpoint caps the install so a hung resolve becomes a 504. The
        repair that follows a swap resolves against the same index through the
        same resolver, so whatever hung the install hangs it as well — left
        unbounded it holds the request open forever and the 504 never arrives.
        ``FakeUv(repair_hangs=True)`` refuses to fake a return for an unbounded
        call (``UnboundedInstall``), so this fails loudly if the bound is lost.
        """
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = self._venv(
            monkeypatch,
            honours_constraints=False,
            feature_install_times_out=True,
            repair_hangs=True,
        )

        with TestClient(_make_app(_make_agent()), raise_server_exceptions=False) as client:
            resp = client.post("/api/features/test-pkg/install")

        # The install's own verdict still stands...
        assert resp.status_code == 504
        detail = resp.json()["detail"]
        assert "Installation timed out" in detail
        # ...and the core it left swapped is named, with the manual command,
        # because the automatic restore did not finish either.
        assert "was replaced during the install batch" in detail
        assert (
            "RESTORE FAILED — run `uv pip install --python "
            f"{shlex.quote(sys.executable)} -e /src/core` by hand."
        ) in detail
        assert "timed out after 300s" in detail  # why the repair's tail stops
        assert venv.editable.get(CORE) is None  # still swapped — reported, not hidden

    def test_the_cli_repair_stays_unbounded(self, monkeypatch):
        """The bound is the HTTP surface's, not a new default.

        An operator watching a terminal can interrupt a slow install; capping
        the CLI's restore would abandon a genuinely slow-but-working reinstall
        of core, which is worse than waiting. So the guard's default must stay
        unlimited — asserted here rather than left to whoever reads the
        signature next.
        """
        from kestrel_sovereign.cli_features import CoreInstallGuard

        venv = FakeUv(core_checkout="/src/core")
        use_fake_uv(monkeypatch, venv)
        guard = CoreInstallGuard.snapshot()
        venv.editable.pop(CORE)  # something swapped core out from under us
        seen = []
        real_run = venv.run

        def record(cmd, **kw):
            seen.append(kw.get("timeout"))
            return real_run(cmd, **kw)

        monkeypatch.setattr("kestrel_sovereign.cli.subprocess.run", record)

        assert guard.verify() == 1  # drift reported...
        assert venv.editable[CORE] == "/src/core"  # ...and repaired
        assert seen == [None]


# ---------------------------------------------------------------------------
# POST /api/features/{name}/remove
# ---------------------------------------------------------------------------


class TestRemoveFeature:
    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.subprocess.run")
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    async def test_hook_cancelled_error_rolls_back_remove_generation(
        self, mock_pkg, mock_run
    ):
        """Removal rolls back before its irreversible data/package boundary."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature
        await agent._activate_feature_runtime(feature)
        feature.on_disable = _propagate_cancelled_child
        mock_pkg.return_value = FeaturePackageInfo(
            name="fixture-pkg",
            package="kestrel-feature-fixture",
            git="",
            features=[feature.name],
            description="fixture",
        )
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(asyncio.CancelledError):
            await features_endpoint.remove_feature(request, feature.name)

        assert feature.enabled is True
        assert agent.features[feature.name] is feature
        assert agent.feature_contribution_runtime.active_context_clauses()
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.subprocess.run")
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    async def test_remove_failure_restores_cross_feature_setup_batch(
        self, mock_pkg, mock_run
    ):
        """Rollback restores a package whose setup steps depend on each other."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        class FirstFeature(SDKFixtureFeature):
            contribution_prefix = "remove-first"

        class SecondFeature(SDKFixtureFeature):
            contribution_prefix = "remove-second"

        agent = _lifecycle_agent()
        first = FirstFeature(agent)
        second = SecondFeature(agent)
        second.setup_registration = replace(
            second.setup_registration,
            after=(first.setup_registration.name,),
        )
        transition = agent._prepare_feature_contribution_transition((first, second))
        for feature, prepared in transition.activatable((first, second)):
            await agent._activate_feature_runtime(
                feature,
                prepared_contributions=prepared,
            )

        original_shutdown = second.shutdown
        shutdown_calls = 0

        async def fail_first_shutdown():
            nonlocal shutdown_calls
            shutdown_calls += 1
            if shutdown_calls == 1:
                raise RuntimeError("second teardown failed")
            await original_shutdown()

        second.shutdown = fail_first_shutdown
        mock_pkg.return_value = FeaturePackageInfo(
            name="dependent-pkg",
            package="kestrel-feature-dependent",
            git="",
            features=[first.name, second.name],
            description="cross-feature setup dependency fixture",
        )
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(RuntimeError, match="second teardown failed"):
            await features_endpoint.remove_feature(request, first.name)

        assert agent.features[first.name] is first
        assert agent.features[second.name] is second
        assert first.enabled is True
        assert second.enabled is True
        assert agent.setup_step_registry.get(first.setup_registration.name) is (
            first.setup_registration
        )
        assert agent.setup_step_registry.get(second.setup_registration.name) is (
            second.setup_registration
        )
        assert len(agent.feature_contribution_runtime.active_context_clauses()) == 2
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.subprocess.run")
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    async def test_cancelled_queued_remove_performs_no_later_mutation(
        self, mock_pkg, mock_run
    ):
        """Cancellation before the turn boundary means removal never starts."""

        feature = _make_feature()
        agent = _lifecycle_agent(features={"TestFeature": feature})
        mock_pkg.return_value = FeaturePackageInfo(
            name="fixture-pkg",
            package="kestrel-feature-fixture",
            git="",
            features=["TestFeature"],
            description="fixture",
        )
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        async with agent._turn_lifecycle():
            removal = asyncio.create_task(
                features_endpoint.remove_feature(request, "TestFeature")
            )
            await asyncio.sleep(0)
            removal.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(removal, timeout=1)

        # There is no orphaned owned task waiting to mutate after the old turn
        # releases its lock.
        await asyncio.sleep(0.05)
        feature.on_disable.assert_not_awaited()
        feature.on_remove.assert_not_awaited()
        assert agent.features["TestFeature"] is feature
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.subprocess.run")
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    async def test_remove_waits_for_active_turn_before_teardown(
        self, mock_pkg, mock_run
    ):
        """Removal preserves the complete runtime generation for a live turn."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature
        await agent._activate_feature_runtime(feature)
        teardown_entered = asyncio.Event()

        async def record_disable():
            teardown_entered.set()

        feature.on_disable = record_disable
        mock_pkg.return_value = FeaturePackageInfo(
            name="fixture-pkg",
            package="kestrel-feature-fixture",
            git="",
            features=[feature.name],
            description="fixture",
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        async with agent._turn_lifecycle():
            removal = asyncio.create_task(
                features_endpoint.remove_feature(request, feature.name)
            )
            # Give an incorrectly unserialized teardown ample time to enter;
            # the correct path is still queued on the active turn boundary.
            await asyncio.sleep(0.05)
            assert not teardown_entered.is_set()
            assert feature.name in agent.features
            assert agent.feature_contribution_runtime.active_context_clauses()
            assert not feature.disabled

        response = await asyncio.wait_for(removal, timeout=1)
        assert response["status"] == "removed"
        assert feature.name not in agent.features
        assert not agent.feature_contribution_runtime.active_context_clauses()
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.endpoints.features.subprocess.run")
    @patch("kestrel_sovereign.endpoints.features.get_package_for_feature")
    async def test_cancelled_remove_finishes_before_releasing_turn_boundary(
        self, mock_pkg, mock_run
    ):
        """Once removal starts, disconnect cannot expose its partial state."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature
        await agent._activate_feature_runtime(feature)
        entered = asyncio.Event()
        release = asyncio.Event()
        turn_entered = asyncio.Event()

        async def slow_remove():
            entered.set()
            await release.wait()

        async def enter_turn():
            async with agent._turn_lifecycle():
                turn_entered.set()

        feature.on_remove = slow_remove
        mock_pkg.return_value = FeaturePackageInfo(
            name="fixture-pkg",
            package="kestrel-feature-fixture",
            git="",
            features=[feature.name],
            description="fixture",
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        removal = asyncio.create_task(
            features_endpoint.remove_feature(request, feature.name)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        removal.cancel()
        await asyncio.sleep(0)
        assert not removal.done()

        contender = asyncio.create_task(enter_turn())
        await asyncio.sleep(0)
        assert not turn_entered.is_set()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(removal, timeout=1)
        await asyncio.wait_for(contender, timeout=1)

        assert feature.name not in agent.features
        assert not agent.feature_contribution_runtime.active_context_clauses()
        assert turn_entered.is_set()
        mock_run.assert_called_once()

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
    @pytest.mark.asyncio
    async def test_late_isolated_ingress_generation_change_returns_retryable_409(
        self,
    ):
        """A lease lost under the lifecycle lock is a conflict, not a 500."""

        from kestrel_sovereign.features.isolated_runtime import (
            IsolatedRuntimeConfigGenerationChanged,
        )

        lease = object()

        class LateGenerationChangeFeature:
            name = "TestFeature"
            enabled = True
            config_schema = None

            async def get_config(self):
                return {"mode": "old"}

            async def set_config(self, _config):
                raise IsolatedRuntimeConfigGenerationChanged(
                    "private generation detail"
                )

            @asynccontextmanager
            async def config_transition_ingress_fence(self):
                yield lease

            def claim_config_transition_ingress_fence(self, candidate):
                return candidate is lease

        feature = LateGenerationChangeFeature()
        agent = _lifecycle_agent(features={feature.name: feature})
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(HTTPException) as error:
            await features_endpoint.update_feature_config(
                request,
                feature.name,
                features_endpoint.ConfigUpdateRequest(config={"mode": "next"}),
            )

        assert error.value.status_code == 409
        assert "retry against the current generation" in error.value.detail
        assert "private generation detail" not in error.value.detail
        agent.refresh_feature_context_clauses.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_turn_can_call_isolated_tool_after_config_fence_closes(
        self, monkeypatch, tmp_path
    ):
        """A queued config transition cannot strand its current live turn."""

        from kestrel_sovereign.features.isolated_runtime import (
            InstalledFeatureRuntime,
            ProxyFeature,
        )

        class Storage:
            def __init__(self, agent_id):
                self.agent_id = agent_id
                self.nodes = {}

            async def get_node(self, node_id):
                return self.nodes.get(node_id)

            async def compare_and_swap_node(self, node_id, expected, node):
                current = self.nodes.get(node_id)
                properties = None if current is None else current.properties
                if properties != expected:
                    return "predicate_failed"
                self.nodes[node_id] = node
                return "swapped"

        tool_called = asyncio.Event()

        class Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.event_handler = None

            async def start(self):
                return None

            async def stop(self):
                return None

            async def health(self):
                return True

            @property
            def capabilities(self):
                return {"inbound_producer": False}

            async def list_tools(self):
                return [
                    {
                        "name": "ping",
                        "description": "Ping",
                        "category": "utility",
                        "input_schema": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                        },
                    }
                ]

            async def call_tool(self, name, args):
                tool_called.set()
                return {"name": name, "args": dict(args)}

            def on_event(self, handler):
                self.event_handler = handler

        agent = _lifecycle_agent()
        agent.storage = Storage(agent.did)
        agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        runtime = InstalledFeatureRuntime(
            class_name="DelayedToolFeature",
            entry_point="test_pkg.feature:DelayedToolFeature",
            distribution="test-pkg",
            runtime="isolated-venv",
            service="test_service",
            description="Delayed tool fixture",
        )
        monkeypatch.setenv(
            "KESTREL_FEATURE_DELAYEDTOOLFEATURE_BIN", "/bin/test-service"
        )
        feature = ProxyFeature(agent, runtime, client_factory=Client)
        await feature.initialize()
        agent.features[feature.name] = feature
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )
        update = None
        detached = None
        rescue = None
        blocked = False
        try:
            async with agent._turn_lifecycle():
                update = asyncio.create_task(
                    features_endpoint.update_feature_config(
                        request,
                        feature.name,
                        features_endpoint.ConfigUpdateRequest(
                            config={"enabled": True}
                        ),
                    )
                )
                for _ in range(100):
                    if feature._traffic_gate.closed:
                        break
                    await asyncio.sleep(0.01)
                assert feature._traffic_gate.closed is True

                # A detached child inherits the ambient turn id, but it does
                # not own the live turn and must remain behind the fence.
                detached = asyncio.create_task(
                    feature.call_isolated_tool("ping", {"message": "detached"})
                )
                await asyncio.sleep(0.02)
                assert detached.done() is False
                detached.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await detached

                async def release_deadlock() -> None:
                    nonlocal blocked
                    await asyncio.sleep(0.05)
                    if not tool_called.is_set():
                        blocked = True
                        # Cancel the queued config child so the gate reopens and
                        # the live turn can report the cycle without leaking it.
                        update.cancel()

                rescue = asyncio.create_task(release_deadlock())
                result = await feature.call_isolated_tool(
                    "ping", {"message": "late"}
                )
                assert result == {
                    "success": True,
                    "tool": "ping",
                    "result": {
                        "name": "ping",
                        "args": {"message": "late"},
                    },
                }
                if not rescue.done():
                    rescue.cancel()
                with suppress(asyncio.CancelledError):
                    await rescue

            if not blocked:
                response = await asyncio.wait_for(update, timeout=1)
                assert response["config"] == {"enabled": True}
                assert feature._traffic_gate.closed is False
                assert feature._config_ingress_live_turn_bypass_active is False
            else:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(update, timeout=1)
        finally:
            if update is not None and not update.done():
                update.cancel()
            if detached is not None and not detached.done():
                detached.cancel()
            if rescue is not None and not rescue.done():
                rescue.cancel()
            await feature.shutdown()

        assert not blocked

    @pytest.mark.asyncio
    async def test_isolated_ingress_drains_before_config_takes_turn_lock(self):
        """An admitted callback cannot invert ingress-drain and turn locks."""

        state = {"mode": "old"}
        release_cognition = asyncio.Event()
        cognition_done = asyncio.Event()
        fence_entered = False

        class AdmittedIngressFeature:
            name = "TestFeature"
            enabled = True
            config_schema = None

            async def get_config(self):
                return dict(state)

            async def set_config(self, config):
                # The old endpoint first takes CONVERSATION, then reaches the
                # isolated setter's traffic drain. The admitted callback below
                # needs that same turn lock, reproducing the production cycle.
                release_cognition.set()
                await asyncio.wait_for(cognition_done.wait(), timeout=0.05)
                state.clear()
                state.update(config)

            @asynccontextmanager
            async def config_transition_ingress_fence(self):
                nonlocal fence_entered
                fence_entered = True
                release_cognition.set()
                await asyncio.wait_for(cognition_done.wait(), timeout=1)
                yield

        feature = AdmittedIngressFeature()
        agent = _lifecycle_agent(features={feature.name: feature})
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        async def admitted_cognition():
            await release_cognition.wait()
            async with agent._turn_lifecycle():
                cognition_done.set()

        cognition = asyncio.create_task(admitted_cognition())
        try:
            response = await asyncio.wait_for(
                features_endpoint.update_feature_config(
                    request,
                    feature.name,
                    features_endpoint.ConfigUpdateRequest(
                        config={"mode": "new"}
                    ),
                ),
                timeout=1,
            )
        finally:
            release_cognition.set()
            await asyncio.wait_for(cognition, timeout=1)

        assert fence_entered is True
        assert state == {"mode": "new"}
        assert response["config"] == {"mode": "new"}

    @pytest.mark.asyncio
    async def test_config_rejects_fence_from_superseded_feature_lifecycle(self):
        """The generation fenced before CONVERSATION must still own ingress."""

        fence_entered = asyncio.Event()
        state = {"mode": "old"}

        class LifecycleBoundFeature:
            name = "TestFeature"
            enabled = True
            config_schema = None

            def __init__(self):
                self.generation = 0
                self.set_calls = 0

            async def get_config(self):
                return dict(state)

            async def set_config(self, config):
                self.set_calls += 1
                state.clear()
                state.update(config)

            @asynccontextmanager
            async def config_transition_ingress_fence(self):
                lease = (self, self.generation)
                fence_entered.set()
                yield lease

            def claim_config_transition_ingress_fence(self, lease):
                return lease == (self, self.generation)

        feature = LifecycleBoundFeature()
        agent = _lifecycle_agent(features={feature.name: feature})
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        async with agent.feature_config_transition():
            update = asyncio.create_task(
                features_endpoint.update_feature_config(
                    request,
                    feature.name,
                    features_endpoint.ConfigUpdateRequest(
                        config={"mode": "new"}
                    ),
                )
            )
            await asyncio.wait_for(fence_entered.wait(), timeout=1)
            feature.generation += 1

        with pytest.raises(HTTPException) as error:
            await asyncio.wait_for(update, timeout=1)

        assert error.value.status_code == 409
        assert "changed while configuration was queued" in error.value.detail
        assert feature.set_calls == 0
        assert state == {"mode": "old"}

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
        agent.refresh_feature_context_clauses.assert_called_once_with(feature)

    def test_refresh_failure_restores_config_and_previous_context_snapshot(self):
        """Tools and cached prompt policy roll back as one failed transition."""

        state = {"mode": "old"}
        feature = _make_feature(config=state)

        async def get_config():
            return dict(state)

        async def set_config(config):
            state.clear()
            state.update(config)

        feature.get_config.side_effect = get_config
        feature.set_config.side_effect = set_config
        agent = _make_agent(features={"TestFeature": feature})
        agent.refresh_feature_context_clauses.side_effect = [
            RuntimeError("new clause renderer failed"),
            None,
        ]
        app = _make_app(agent)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"mode": "new"}},
            )

        assert resp.status_code == 500
        assert state == {"mode": "old"}
        assert [call.args[0] for call in feature.set_config.await_args_list] == [
            {"mode": "new"},
            {"mode": "old"},
        ]
        assert agent.refresh_feature_context_clauses.call_count == 2

    def test_refresh_failure_uses_generation_owned_rollback_when_available(self):
        """Hosted rollback receives the exact commit receipt, not a blind write."""

        receipt = object()
        feature = _make_feature(config={"mode": "old"})
        feature.set_config.return_value = receipt
        feature.rollback_config_transition = AsyncMock()
        agent = _make_agent(features={"TestFeature": feature})
        agent.refresh_feature_context_clauses.side_effect = [
            RuntimeError("new clause renderer failed"),
            None,
        ]
        app = _make_app(agent)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"mode": "new"}},
            )

        assert resp.status_code == 500
        feature.rollback_config_transition.assert_awaited_once_with(receipt)
        feature.set_config.assert_awaited_once_with({"mode": "new"})

    @pytest.mark.asyncio
    async def test_endpoint_config_transition_waits_for_active_turn(self):
        """The HTTP transition shares the production conversation lock."""

        feature = _make_feature(config={"mode": "old"})
        applied = asyncio.Event()

        async def set_config(_config):
            applied.set()

        feature.set_config.side_effect = set_config
        agent = _lifecycle_agent(features={"TestFeature": feature})
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        async with agent._turn_lifecycle():
            update = asyncio.create_task(
                features_endpoint.update_feature_config(
                    request,
                    "TestFeature",
                    features_endpoint.ConfigUpdateRequest(
                        config={"mode": "new"}
                    ),
                )
            )
            await asyncio.sleep(0)
            assert not applied.is_set()

        response = await asyncio.wait_for(update, timeout=1)
        assert response["config"] == {"mode": "old"}
        assert applied.is_set()

    @pytest.mark.asyncio
    async def test_queued_config_update_re_resolves_after_feature_removal(self):
        """A waiter cannot mutate the stale feature generation it first saw."""

        stale = _make_feature(config={"mode": "old"})
        agent = _lifecycle_agent(features={"TestFeature": stale})
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        async with agent._turn_lifecycle():
            update = asyncio.create_task(
                features_endpoint.update_feature_config(
                    request,
                    "TestFeature",
                    features_endpoint.ConfigUpdateRequest(
                        config={"mode": "new"}
                    ),
                )
            )
            await asyncio.sleep(0)
            assert not update.done()
            assert agent.features.pop("TestFeature") is stale

        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(update, timeout=1)

        assert exc_info.value.status_code == 404
        stale.set_config.assert_not_awaited()
        agent.refresh_feature_context_clauses.assert_not_called()

    @pytest.mark.asyncio
    async def test_config_setter_can_reenter_privacy_transition(self):
        """A hosted setter cannot deadlock on its mutation's own turn lock."""

        feature = _make_feature(config={"mode": "old"})
        agent = _lifecycle_agent(features={"TestFeature": feature})
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        entered = asyncio.Event()

        async def set_config(_config):
            async with agent.privacy_transition():
                entered.set()

        feature.set_config.side_effect = set_config
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        await asyncio.wait_for(
            features_endpoint.update_feature_config(
                request,
                "TestFeature",
                features_endpoint.ConfigUpdateRequest(config={"mode": "new"}),
            ),
            timeout=1,
        )

        assert entered.is_set()

    @pytest.mark.asyncio
    async def test_config_setter_cognition_is_rejected_before_context_refresh(self):
        """A setter cannot expose new live config beside old cached context."""

        state = {"mode": "old"}
        feature = _make_feature(config=state)
        feature.get_config.side_effect = lambda: dict(state)
        agent = _lifecycle_agent(features={"TestFeature": feature})
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        observed = []

        async def set_config(config):
            state.clear()
            state.update(config)
            async with agent._turn_lifecycle():
                observed.append(dict(state))

        feature.set_config.side_effect = set_config
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(
            RuntimeError,
            match="feature transition generation is fully committed",
        ):
            await asyncio.wait_for(
                features_endpoint.update_feature_config(
                    request,
                    "TestFeature",
                    features_endpoint.ConfigUpdateRequest(
                        config={"mode": "new"}
                    ),
                ),
                timeout=1,
            )

        assert observed == []
        assert state == {"mode": "new"}
        agent.refresh_feature_context_clauses.assert_called_once_with(feature)

    @pytest.mark.asyncio
    async def test_config_updates_are_not_serialized_across_agents(self):
        """A slow tenant setter must not block an unrelated tenant."""

        first_started = asyncio.Event()
        release_first = asyncio.Event()
        first = _make_feature(config={"mode": "old"})
        second = _make_feature(config={"mode": "old"})

        async def block_first(_config):
            first_started.set()
            await release_first.wait()

        first.set_config.side_effect = block_first
        first_agent = _lifecycle_agent(features={"TestFeature": first})
        second_agent = _lifecycle_agent(features={"TestFeature": second})
        first_agent.refresh_feature_context_clauses = MagicMock(return_value=())
        second_agent.refresh_feature_context_clauses = MagicMock(return_value=())

        def request_for(agent):
            return SimpleNamespace(
                state=SimpleNamespace(agent=agent),
                app=SimpleNamespace(state=SimpleNamespace(agent=None)),
            )

        first_update = asyncio.create_task(
            features_endpoint.update_feature_config(
                request_for(first_agent),
                "TestFeature",
                features_endpoint.ConfigUpdateRequest(config={"mode": "first"}),
            )
        )
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second_update = asyncio.create_task(
            features_endpoint.update_feature_config(
                request_for(second_agent),
                "TestFeature",
                features_endpoint.ConfigUpdateRequest(config={"mode": "second"}),
            )
        )
        try:
            done, _pending = await asyncio.wait({second_update}, timeout=0.5)
            assert second_update in done
            assert second_update.result()["config"] == {"mode": "old"}
        finally:
            release_first.set()
            await asyncio.gather(first_update, second_update)

    @pytest.mark.asyncio
    async def test_cancellation_after_commit_waits_for_context_reconciliation(self):
        """A disconnected PATCH cannot strand new config with old clauses."""

        state = {"mode": "old"}
        committed = asyncio.Event()
        release_setter = asyncio.Event()
        refreshed = asyncio.Event()
        feature = _make_feature(config=state)

        async def get_config():
            return dict(state)

        async def set_config(config):
            state.clear()
            state.update(config)
            committed.set()
            await release_setter.wait()

        def refresh(_feature):
            refreshed.set()

        feature.get_config.side_effect = get_config
        feature.set_config.side_effect = set_config
        agent = _lifecycle_agent(features={"TestFeature": feature})
        agent.refresh_feature_context_clauses = refresh
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        update = asyncio.create_task(
            features_endpoint.update_feature_config(
                request,
                "TestFeature",
                features_endpoint.ConfigUpdateRequest(config={"mode": "new"}),
            )
        )
        await asyncio.wait_for(committed.wait(), timeout=1)
        update.cancel()
        await asyncio.sleep(0)
        assert not update.done()
        assert not refreshed.is_set()
        update.cancel()
        await asyncio.sleep(0)
        assert not update.done()

        release_setter.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(update, timeout=1)

        assert state == {"mode": "new"}
        assert refreshed.is_set()

    @pytest.mark.asyncio
    async def test_setter_owned_cancellation_reconciles_committed_context(self):
        """A setter's cancelled child cannot split live config from clauses."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        state = {"mode": "old"}
        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        feature.context_text = "context:old"
        agent.features[feature.name] = feature
        await agent._activate_feature_runtime(feature)

        async def get_config():
            return dict(state)

        async def set_config(config):
            state.clear()
            state.update(config)
            feature.context_text = f"context:{state['mode']}"
            await _propagate_cancelled_child()

        feature.get_config = get_config
        feature.set_config = set_config
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(asyncio.CancelledError):
            await features_endpoint.update_feature_config(
                request,
                feature.name,
                features_endpoint.ConfigUpdateRequest(config={"mode": "new"}),
            )

        assert state == {"mode": "new"}
        assert feature.enabled is True
        clauses = agent.feature_contribution_runtime.active_context_clauses()
        assert [(clause.name, clause.body) for clause in clauses] == [
            ("agent-fixture-context", "context:new")
        ]

    @pytest.mark.asyncio
    async def test_setter_owned_cancellation_disables_when_reconcile_fails(self):
        """Ambiguous commit plus failed republish removes tools and clauses."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        state = {"mode": "old"}
        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        feature.context_text = "context:old"
        agent.features[feature.name] = feature
        await agent._activate_feature_runtime(feature)

        async def get_config():
            return dict(state)

        async def set_config(config):
            state.clear()
            state.update(config)
            feature.context_text = f"context:{state['mode']}"
            await _propagate_cancelled_child()

        feature.get_config = get_config
        feature.set_config = set_config
        agent.refresh_feature_context_clauses = MagicMock(
            side_effect=RuntimeError("private renderer detail")
        )
        # Simulate ownership-ledger drift before the config transition must
        # fail closed. The retained exact source must still be withdrawn.
        del agent.signal_registry._claims[feature.source.name]
        agent.wait_registry.deregister(
            feature.wait_provider.kind, feature.wait_provider
        )
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(RuntimeError, match="feature was disabled") as error:
            await features_endpoint.update_feature_config(
                request,
                feature.name,
                features_endpoint.ConfigUpdateRequest(config={"mode": "new"}),
            )

        assert error.value.__cause__ is None
        assert state == {"mode": "new"}
        assert feature.enabled is False
        assert agent.features[feature.name] is feature
        assert not agent.feature_contribution_runtime.is_active(feature)
        assert not agent.feature_contribution_runtime.active_context_clauses()
        assert agent.signal_registry.get(feature.source.name) is None

    @pytest.mark.asyncio
    async def test_config_quarantine_failure_latches_safe_mode(self):
        """A failed config repair cannot leave prompt authority cognitive."""

        from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

        state = {"mode": "old"}
        agent = _lifecycle_agent()
        feature = SDKFixtureFeature(agent)
        feature.enabled = False
        agent.features[feature.name] = feature
        await agent._activate_feature_runtime(feature)
        runtime = agent.feature_contribution_runtime
        original = runtime.active_context_clauses()[0]
        foreign = replace(original, body="foreign config replacement")
        runtime.context_clause_registry._clauses[original.identity] = foreign

        async def get_config():
            return dict(state)

        async def set_config(config):
            state.clear()
            state.update(config)
            await _propagate_cancelled_child()

        feature.get_config = get_config
        feature.set_config = set_config
        agent.refresh_feature_context_clauses = MagicMock(
            side_effect=RuntimeError("private renderer detail")
        )
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(
            RuntimeError,
            match="feature contributions could not be quarantined",
        ) as error:
            await features_endpoint.update_feature_config(
                request,
                feature.name,
                features_endpoint.ConfigUpdateRequest(config={"mode": "new"}),
            )

        assert error.value.__cause__ is None
        assert agent._safe_mode is True
        assert "quarantine failed" in agent._safe_mode_reason
        assert agent._safe_mode_cause == "feature_lifecycle_uncertain"
        assert agent._feature_lifecycle_integrity_uncertain is True
        assert feature.enabled is False
        assert runtime.is_active(feature)
        assert runtime.active_context_clauses() == (foreign,)

    def test_failed_refresh_and_rollback_disables_feature_runtime(self):
        """A doubly-failed transition is quarantined instead of split-brain."""

        state = {"mode": "old"}
        feature = _make_feature(config=state)

        async def get_config():
            return dict(state)

        async def set_config(config):
            if config == {"mode": "old"}:
                raise RuntimeError("durable rollback failed")
            state.clear()
            state.update(config)

        feature.get_config.side_effect = get_config
        feature.set_config.side_effect = set_config
        agent = _make_agent(features={"TestFeature": feature})
        agent.refresh_feature_context_clauses.side_effect = RuntimeError(
            "new clause renderer failed"
        )
        agent._unregister_feature_runtime = AsyncMock()
        app = _make_app(agent)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"mode": "new"}},
            )

        assert resp.status_code == 500
        assert feature.enabled is False
        agent._unregister_feature_runtime.assert_awaited_once_with(
            feature, unload=False
        )

    def test_updates_disabled_feature_without_refreshing_inactive_context(self):
        feature = _make_feature(config={"enabled": False}, enabled=False)
        agent = _make_agent(features={"TestFeature": feature})
        app = _make_app(agent)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/features/TestFeature/config",
                json={"config": {"enabled": True}},
            )

        assert resp.status_code == 200
        feature.set_config.assert_awaited_once_with({"enabled": True})
        agent.refresh_feature_context_clauses.assert_not_called()

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
    @pytest.mark.asyncio
    async def test_atomic_secret_setter_cancellation_reconciles_context(self):
        """The hosted atomic setter shares the ambiguous-commit boundary."""

        state = {"api_key": "stored-key", "enabled": True}
        feature = _make_feature(config_schema=SECRET_SCHEMA, config=state)

        async def get_config():
            return dict(state)

        async def atomic_update(incoming, secret_fields, validate):
            assert secret_fields == {"api_key"}
            effective = {**state, **incoming}
            validate(effective)
            state.clear()
            state.update(effective)
            await _propagate_cancelled_child()

        feature.get_config.side_effect = get_config
        feature.set_config_with_secret_preservation = AsyncMock(
            side_effect=atomic_update
        )
        agent = _lifecycle_agent(features={"TestFeature": feature})
        agent.refresh_feature_context_clauses = MagicMock(return_value=())
        request = SimpleNamespace(
            state=SimpleNamespace(agent=agent),
            app=SimpleNamespace(state=SimpleNamespace(agent=None)),
        )

        with pytest.raises(asyncio.CancelledError):
            await features_endpoint.update_feature_config(
                request,
                "TestFeature",
                features_endpoint.ConfigUpdateRequest(config={"enabled": False}),
            )

        assert state == {"api_key": "stored-key", "enabled": False}
        agent.refresh_feature_context_clauses.assert_called_once_with(feature)
        feature.set_config.assert_not_awaited()

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


class TestConcurrentInstallSerialization:
    """Two overlapping installs are one transaction each, not two interleaved.

    The endpoint runs snapshot -> install -> resolve in worker threads. Without
    a lock both halves break: concurrent pip writes to one environment are
    unsupported (the no-uv fallback is a multi-pass sequence), and each request
    snapshots core before its own install and compares after — so B's install
    lands inside A's window, A reports it as drift and "repairs" a core nobody
    moved, and B then sees THAT as drift. Two correct installs manufacture two
    spurious CORE_UNSAFE verdicts between them.
    """

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_installs_do_not_interleave(self, mock_registry, monkeypatch):
        import asyncio as _asyncio
        import threading

        from kestrel_sovereign.endpoints import features as features_ep

        mock_registry.return_value = dict(FAKE_REGISTRY)

        # Record entry/exit of the guarded section from the worker threads. If
        # the lock holds, the sequence is strictly paired: enter,exit,enter,exit.
        events: list = []
        lock = threading.Lock()
        overlap = threading.Event()

        class _Guard:
            @classmethod
            def snapshot(cls, *a, **kw):
                with lock:
                    events.append("enter")
                    if events.count("enter") > events.count("exit") + 1:
                        overlap.set()
                return cls()

            def run(self, *a, **kw):
                # Long enough that a second request would overlap if unlocked.
                import time
                time.sleep(0.05)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def resolve(self, *a, **kw):
                with lock:
                    events.append("exit")
                return SimpleNamespace(
                    drift=None, conforming=True, describe=lambda: "",
                )

        monkeypatch.setattr(
            "kestrel_sovereign.cli_features.CoreInstallGuard", _Guard,
        )

        async def _drive():
            features_ep._INSTALL_LOCK = _asyncio.Lock()
            app = _make_app(_make_agent())
            import httpx

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://t",
            ) as client:
                return await _asyncio.gather(
                    client.post("/api/features/test-pkg/install"),
                    client.post("/api/features/test-pkg/install"),
                    return_exceptions=True,
                )

        _asyncio.run(_drive())

        assert not overlap.is_set(), f"installs interleaved: {events}"
        # Strictly paired: no enter follows an enter without an exit between.
        assert events == ["enter", "exit", "enter", "exit"], events

    def test_a_cancelled_request_does_not_release_the_lock_early(self, monkeypatch):
        """Cancelling the request must not hand the venv to the next install.

        Cancelling an `asyncio.to_thread` await does NOT stop the worker thread
        or the installer subprocess it is running. Inline, a cancelled request
        unwinds `async with` and frees the lock immediately while that abandoned
        installer keeps writing — so the next request snapshots a venv still
        being mutated. The work is shielded precisely so the lock outlives the
        request that started it.
        """
        import asyncio as _asyncio
        import threading

        from kestrel_sovereign.endpoints import features as features_ep

        released_while_running = threading.Event()
        finished = threading.Event()
        started = _asyncio.Event()

        async def _slow_guarded():
            async with features_ep._INSTALL_LOCK:
                started.set()
                await _asyncio.sleep(0.15)   # the "installer" still running
                finished.set()

        async def _drive():
            features_ep._INSTALL_LOCK = _asyncio.Lock()
            inner = _asyncio.create_task(_slow_guarded())
            task = _asyncio.ensure_future(_asyncio.shield(inner))
            await started.wait()
            task.cancel()                     # client hangs up mid-install
            try:
                await task
            except _asyncio.CancelledError:
                pass
            # The shielded work is still going; the lock must still be held.
            if not features_ep._INSTALL_LOCK.locked() and not finished.is_set():
                released_while_running.set()
            # Let the shielded task finish so the lock is released by IT.
            await _asyncio.sleep(0.25)
            return features_ep._INSTALL_LOCK.locked()

        still_locked_after = _asyncio.run(_drive())

        assert not released_while_running.is_set(), (
            "lock was released while the abandoned worker was still running"
        )
        assert finished.is_set(), "shielded work did not run to completion"
        assert not still_locked_after, "lock was never released by the task itself"

    def test_cancelling_while_queued_prevents_the_install_entirely(self):
        """A request cancelled while WAITING must not install later.

        Shielding the lock wait as well as the transaction meant a request
        queued behind another install survived its own cancellation, kept
        waiting, and then installed a package nobody was asking for any more —
        a mutation with no live request behind it. Cancellation while waiting
        has to prevent the work; only cancellation once the installer is
        already running is unstoppable.
        """
        import asyncio as _asyncio

        from kestrel_sovereign.endpoints import features as features_ep

        installed: list = []

        async def _txn(tag):
            try:
                installed.append(tag)
                await _asyncio.sleep(0.05)
            finally:
                features_ep._INSTALL_LOCK.release()

        async def _request(tag):
            # Mirrors the endpoint: acquire OUTSIDE the shield, shield the work.
            await features_ep._INSTALL_LOCK.acquire()
            return await _asyncio.shield(_asyncio.create_task(_txn(tag)))

        async def _drive():
            features_ep._INSTALL_LOCK = _asyncio.Lock()
            first = _asyncio.create_task(_request("first"))
            await _asyncio.sleep(0.01)          # first holds the lock
            second = _asyncio.create_task(_request("second"))
            await _asyncio.sleep(0.01)          # second is queued on acquire()
            second.cancel()                     # client hangs up while WAITING
            try:
                await second
            except _asyncio.CancelledError:
                pass
            await first
            await _asyncio.sleep(0.1)           # give a phantom install time to land
            return list(installed)

        done = _asyncio.run(_drive())

        assert done == ["first"], f"a cancelled-while-queued request installed: {done}"
        assert not features_ep._INSTALL_LOCK.locked()


class TestPostRepairRevalidation:
    """A restored core can be exactly the version the new package rejected."""

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_orphaned_feature_after_repair_is_not_reported_as_installed(
        self, mock_registry, monkeypatch,
    ):
        """The package that MOVED core is usually the reason it moved.

        A feature requiring core >=0.53 pulls 0.53 in; the repair puts 0.52
        back; the freshly installed feature now has an unsatisfied dependency
        and cannot load. Returning 200 "installed, restart the agent" there is a
        completed-install report over an environment that cannot run it.
        """
        mock_registry.return_value = dict(FAKE_REGISTRY)
        # Core moved and was restored to 0.52.0, which the package rejects.
        venv = FakeUv(
            feature="kestrel-feature-test", core_checkout="/src/core",
            honours_constraints=False,
        )
        use_fake_uv(monkeypatch, venv)
        # Stated on the seam the code reads requirements through, so the double
        # and the test agree about what the artifact on disk declares.
        monkeypatch.setattr(
            cli, "_installed_requirements", lambda name: ("kestrel-sovereign>=0.53",),
        )

        with TestClient(
            _make_app(_make_agent()), raise_server_exceptions=False,
        ) as client:
            resp = client.post("/api/features/test-pkg/install")

        assert resp.status_code == 500, resp.json()
        detail = resp.json()["detail"]
        assert "cannot load" in detail
        assert "0.52.0" in detail

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_an_inactive_conditional_core_requirement_is_not_reported_unmet(
        self, mock_registry, monkeypatch,
    ):
        """A requirement this interpreter does not have is not an unmet one.

        `Requires-Dist: kestrel-sovereign>=0.60; python_version < "3.10"` says
        nothing about a 3.13 host. Checking its specifier regardless turned a
        healthy install into a 500 — the guard inventing the very failure it
        exists to report.
        """
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = FakeUv(
            feature="kestrel-feature-test", core_checkout="/src/core",
            honours_constraints=False,
        )
        use_fake_uv(monkeypatch, venv)
        # The SAME specifier the test above proves is reported when it applies —
        # only the marker differs, so the marker is what this test isolates.
        monkeypatch.setattr(
            cli,
            "_installed_requirements",
            lambda name: ('kestrel-sovereign>=0.53; python_version < "3.10"',),
        )

        with TestClient(
            _make_app(_make_agent()), raise_server_exceptions=False,
        ) as client:
            resp = client.post("/api/features/test-pkg/install")

        assert resp.status_code == 200, resp.json()
        # The drift itself is still reported — only the false unmet-requirement
        # claim is gone.
        assert resp.json()["status"] == "installed_with_core_drift"

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_a_legal_alternate_spelling_of_core_is_still_core(
        self, mock_registry, monkeypatch,
    ):
        """`Requirement.name` keeps whatever spelling the metadata used.

        `Kestrel_Sovereign` and `kestrel-sovereign` are the same distribution
        to every installer, so a check that reads the rendered sentence misses
        this one and returns 200 over a core the package cannot accept.
        """
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = FakeUv(
            feature="kestrel-feature-test", core_checkout="/src/core",
            honours_constraints=False,
        )
        use_fake_uv(monkeypatch, venv)
        monkeypatch.setattr(
            cli, "_installed_requirements", lambda name: ("Kestrel_Sovereign>=0.53",),
        )

        with TestClient(
            _make_app(_make_agent()), raise_server_exceptions=False,
        ) as client:
            resp = client.post("/api/features/test-pkg/install")

        assert resp.status_code == 500, resp.json()
        assert "cannot load" in resp.json()["detail"]

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_a_different_distribution_that_starts_with_cores_name_is_not_core(
        self, mock_registry, monkeypatch,
    ):
        """The mirror image: `kestrel-sovereign-sdk` is not `kestrel-sovereign`.

        Matching on the sentence makes an unmet SDK requirement read as an
        unmet CORE requirement, and the response then blames the repair for a
        conflict the repair had nothing to do with.
        """
        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = FakeUv(
            feature="kestrel-feature-test", core_checkout="/src/core",
            honours_constraints=False,
        )
        use_fake_uv(monkeypatch, venv)
        monkeypatch.setattr(
            cli,
            "_installed_requirements",
            lambda name: ("kestrel-sovereign-sdk>=0.99",),
        )

        with TestClient(_make_app(_make_agent())) as client:
            resp = client.post("/api/features/test-pkg/install")

        # The core drift is still reported; it is just not blamed for this.
        assert resp.status_code == 200, resp.json()
        assert resp.json()["status"] == "installed_with_core_drift"

    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_core_revalidation_runs_while_the_install_lock_is_held(
        self, mock_registry, monkeypatch,
    ):
        """Reading live core version outside the lock reads a passing state.

        A queued install is waiting to mutate the very environment this reads.
        Outside the lock, this can observe a core the next request puts up and
        takes back down, and answer about an environment that never outlived the
        read. The earlier fix moved snapshot/install/resolve inside the lock;
        this check was added afterwards and was left outside it.
        """
        import importlib.metadata as md

        from kestrel_sovereign.endpoints import features as features_ep

        mock_registry.return_value = dict(FAKE_REGISTRY)
        venv = FakeUv(
            feature="kestrel-feature-test", core_checkout="/src/core",
            honours_constraints=False,
        )
        use_fake_uv(monkeypatch, venv)

        class _Meta(dict):
            def get_all(self, key):
                return self.get(key, [])

        monkeypatch.setattr(
            md, "metadata",
            lambda name: _Meta({"Requires-Dist": ["kestrel-sovereign>=0.53"]}),
        )

        held = []
        real = features_ep._core_requirement_unsatisfied

        def _observe(package_spec):
            # Observed AT THE MOMENT OF THE CALL, not afterwards: the question
            # is whether this read is inside the critical section.
            held.append(features_ep._INSTALL_LOCK.locked())
            return real(package_spec)

        monkeypatch.setattr(
            features_ep, "_core_requirement_unsatisfied", _observe,
        )

        with TestClient(
            _make_app(_make_agent()), raise_server_exceptions=False,
        ) as client:
            resp = client.post("/api/features/test-pkg/install")

        assert resp.status_code == 500, resp.json()   # the check really ran...
        assert held == [True]                          # ...and ran holding the lock


# =============================================================================
# Sovereign authority on the shared-environment routes (#3214)
# =============================================================================


class TestSharedEnvironmentRoutesRequireSovereignAuthority:
    """Install and remove run pip against the interpreter every agent on the
    host is loaded from, so they are host administration rather than the
    routed agent's own business.

    The claim was already in the docstring — "Requires a sovereign agent —
    governed agents cannot install packages" — while the handler resolved
    the routed agent and installed. The predicate that would have enforced
    it existed and guarded ``POST /api/agents``; these routes just never
    used it.
    """

    @pytest.mark.parametrize("route", ["install", "remove"])
    @pytest.mark.parametrize(
        "caller,label",
        [
            (CallerContext(role=CallerRole.AUTHENTICATED), "an OAuth/JWT user"),
            (CallerContext.a2a_transport(), "a peer admitted by transport"),
            (CallerContext.anonymous(), "an anonymous caller"),
            (None, "no auth middleware at all"),
        ],
    )
    def test_a_caller_without_sovereign_authority_is_refused(
        self, route, caller, label
    ):
        agent = _make_agent()
        app = _make_app(agent, caller=caller)

        with TestClient(app) as client:
            response = client.post(f"/api/features/test-pkg/{route}")

        assert response.status_code == 403, (label, route, response.text)
        assert "Sovereign authority is required." in response.text

    @pytest.mark.parametrize("route", ["install", "remove"])
    @patch("kestrel_sovereign.endpoints.features.get_registry")
    def test_refusal_does_not_reveal_whether_the_package_exists(
        self, mock_registry, route
    ):
        """403 before 404, so the refusal is not an existence oracle.

        The guard is a route dependency for this reason: a check inside
        the handler would run after the registry lookup, and an
        unauthorized caller could tell a real package from an invented
        one by whether they got 404 or 403.

        The registry has to be the real fake one for this to mean
        anything. Written first against a bare `_make_agent()`, both
        names resolved identically, so the two requests could not
        differ however the guard was placed — and a mutant that moved
        the check after the lookup passed. `test-pkg` is in the catalog
        below and the invented name is not, so the oracle exists to be
        closed.
        """
        mock_registry.return_value = dict(FAKE_REGISTRY)
        agent = _make_agent()
        app = _make_app(agent, caller=CallerContext.anonymous())

        with TestClient(app) as client:
            known = client.post(f"/api/features/test-pkg/{route}")
            invented = client.post(
                f"/api/features/no-such-package-anywhere/{route}"
            )

        assert known.status_code == invented.status_code == 403
        assert known.text == invented.text

    @pytest.mark.parametrize("route", ["install", "remove"])
    def test_a_sovereign_caller_is_admitted_past_the_gate(self, route):
        """The positive control.

        Without it, a guard that refused everyone would satisfy every
        assertion above. A sovereign caller must get *past* authority —
        whatever the route then does about an unknown package is the
        route's business, but it must not be 403.
        """
        agent = _make_agent()
        app = _make_app(agent, caller=CallerContext.sovereign(AuthMethod.API_KEY))

        with TestClient(app) as client:
            response = client.post(f"/api/features/no-such-package-anywhere/{route}")

        assert response.status_code != 403, response.text
