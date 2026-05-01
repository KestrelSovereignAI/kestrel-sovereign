"""
E2E tests for feature lifecycle: enable/disable/remove cycle with hook management.

Issue #496: Verify the full feature lifecycle including hook registration,
unregistration, config persistence, and cleanup.

Tests the complete cycle:
1. Install feature with hooks → hooks registered with HooksManager
2. Disable feature via agent → hooks unregistered (no stale hooks)
3. Re-enable feature → hooks re-registered
4. Remove feature → on_remove() cleanup runs
5. Config persists across enable/disable cycles
"""

import pytest
import pytest_asyncio
import logging
from typing import Dict, List, Optional
from unittest.mock import MagicMock

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.hooks import Hook, HookEvent, HookInput, HookOutput, HooksManager
from kestrel_sovereign.hooks.base import PermissionDecision
from kestrel_sovereign.tools.base import ToolCategory
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode

logger = logging.getLogger(__name__)


# =============================================================================
# Test Feature & Hook implementations
# =============================================================================


class TransactionSecurityHookStub(Hook):
    """Stub of TransactionSecurityHook for lifecycle testing.

    Tracks registration/unregistration via call_count and fires on PRE_TOOL_USE.
    Uses unique names to avoid collision with real features loaded by the agent.
    """

    def __init__(self):
        super().__init__(
            name="lifecycle_test_tx_security",
            events=[HookEvent.PRE_TOOL_USE],
            priority=10,
        )
        self.call_count = 0

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        return HookOutput.allow("transaction approved")


class AuditLogHookStub(Hook):
    """Stub audit hook that fires on POST_TOOL_USE."""

    def __init__(self):
        super().__init__(
            name="lifecycle_test_audit_log",
            events=[HookEvent.POST_TOOL_USE],
            priority=100,
        )
        self.logged_events: list = []

    async def execute(self, input: HookInput) -> HookOutput:
        self.logged_events.append(input.tool_name)
        return HookOutput.allow()


class LifecycleTestFeature(Feature):
    """Feature with hooks, config, and lifecycle tracking for E2E testing.

    Simulates a wallet-like feature with:
    - TransactionSecurityHook (PRE_TOOL_USE)
    - AuditLogHook (POST_TOOL_USE)
    - Config schema with persistence
    - Lifecycle event tracking
    """

    def __init__(self, agent):
        super().__init__(agent)
        self.tx_hook = TransactionSecurityHookStub()
        self.audit_hook = AuditLogHookStub()
        self._config = {"max_amount": 1000, "require_approval": True}

        # Lifecycle tracking
        self.initialize_count = 0
        self.enable_count = 0
        self.disable_count = 0
        self.remove_count = 0
        self.remove_cleanup_data: Optional[dict] = None

    @property
    def tool_description(self) -> str:
        return "Lifecycle test feature with hooks and config"

    async def initialize(self):
        self.initialize_count += 1

    def get_hooks(self) -> List[Hook]:
        return [self.tx_hook, self.audit_hook]

    async def on_enable(self):
        self.enable_count += 1

    async def on_disable(self):
        self.disable_count += 1

    async def on_remove(self):
        self.remove_count += 1
        self.remove_cleanup_data = {"cleaned": True, "removed_at": "test"}

    @property
    def config_schema(self) -> Optional[Dict]:
        return {
            "type": "object",
            "properties": {
                "max_amount": {"type": "integer", "minimum": 0},
                "require_approval": {"type": "boolean"},
            },
        }

    async def get_config(self) -> Dict:
        return self._config.copy()

    async def set_config(self, config: Dict) -> None:
        self._config.update(config)

    @tool("lifecycle_test_action", "Test action", ToolCategory.SYSTEM)
    async def lifecycle_test_action(self, amount: int = 100) -> dict:
        return {"success": True, "amount": amount}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hooks_manager():
    """Fresh HooksManager instance."""
    return HooksManager()


@pytest.fixture
def mock_agent(hooks_manager):
    """Mock agent with a real HooksManager for lifecycle testing."""
    agent = MagicMock()
    agent.hooks_manager = hooks_manager
    agent.features = {}
    agent.task_manager = None
    agent.storage = None
    return agent


@pytest.fixture
def lifecycle_feature(mock_agent):
    """LifecycleTestFeature bound to mock_agent."""
    return LifecycleTestFeature(mock_agent)


# =============================================================================
# Test: Full Lifecycle E2E (the main test)
# =============================================================================


class TestFeatureLifecycleE2E:
    """
    Full enable/disable/remove lifecycle with hook verification.

    This is the primary E2E test for issue #496.
    """

    @pytest.mark.asyncio
    async def test_full_lifecycle_enable_disable_remove(self, mock_agent, lifecycle_feature):
        """
        Complete lifecycle: register → disable → re-enable → remove.

        Acceptance criteria:
        - Hooks correctly register on enable
        - Hooks correctly unregister on disable (no stale hooks)
        - on_remove() cleanup runs before uninstall
        - Feature config persists across enable/disable cycles
        """
        manager = mock_agent.hooks_manager

        # ── Phase 1: Register feature (simulating _register_feature) ──
        await lifecycle_feature.initialize()
        mock_agent.features[lifecycle_feature.name] = lifecycle_feature

        for hook in lifecycle_feature.get_hooks():
            manager.register(hook)
        await lifecycle_feature.on_enable()

        # Verify hooks are registered
        pre_hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        post_hooks = manager.get_hooks(HookEvent.POST_TOOL_USE)
        assert any(h.name == "lifecycle_test_tx_security" for h in pre_hooks), \
            "TransactionSecurityHook should be registered for PRE_TOOL_USE"
        assert any(h.name == "lifecycle_test_audit_log" for h in post_hooks), \
            "AuditLogHook should be registered for POST_TOOL_USE"

        # Verify hooks are active by executing them
        hook_input = HookInput(
            session_id="lifecycle-test",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="send_payment",
        )
        result = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert result.continue_execution is True
        assert lifecycle_feature.tx_hook.call_count == 1

        # Set config before disable (should persist)
        await lifecycle_feature.set_config({"max_amount": 500, "require_approval": False})

        # ── Phase 2: Disable feature (simulating _disable_feature) ──
        await lifecycle_feature.on_disable()
        for hook in lifecycle_feature.get_hooks():
            manager.unregister(hook)

        assert lifecycle_feature.disable_count == 1

        # Verify hooks are unregistered — NO stale hooks
        pre_hooks_after = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        post_hooks_after = manager.get_hooks(HookEvent.POST_TOOL_USE)
        assert not any(h.name == "lifecycle_test_tx_security" for h in pre_hooks_after), \
            "TransactionSecurityHook should be unregistered after disable"
        assert not any(h.name == "lifecycle_test_audit_log" for h in post_hooks_after), \
            "AuditLogHook should be unregistered after disable"

        # Verify hook does NOT fire after disable
        result = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert lifecycle_feature.tx_hook.call_count == 1, \
            "Hook call_count should not increase after disable"

        # Verify config persisted across disable
        config = await lifecycle_feature.get_config()
        assert config["max_amount"] == 500, "Config should persist across disable"
        assert config["require_approval"] is False

        # ── Phase 3: Re-enable feature ──
        for hook in lifecycle_feature.get_hooks():
            manager.register(hook)
        await lifecycle_feature.on_enable()

        assert lifecycle_feature.enable_count == 2  # Initial + re-enable

        # Verify hooks are re-registered
        pre_hooks_reenable = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        post_hooks_reenable = manager.get_hooks(HookEvent.POST_TOOL_USE)
        assert any(h.name == "lifecycle_test_tx_security" for h in pre_hooks_reenable), \
            "TransactionSecurityHook should be re-registered after enable"
        assert any(h.name == "lifecycle_test_audit_log" for h in post_hooks_reenable), \
            "AuditLogHook should be re-registered after enable"

        # Verify hook fires again after re-enable
        result = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert lifecycle_feature.tx_hook.call_count == 2, \
            "Hook should fire again after re-enable"

        # Verify config still persisted
        config = await lifecycle_feature.get_config()
        assert config["max_amount"] == 500

        # ── Phase 4: Remove feature ──
        # Unregister hooks before removal
        for hook in lifecycle_feature.get_hooks():
            manager.unregister(hook)
        await lifecycle_feature.on_remove()

        assert lifecycle_feature.remove_count == 1
        assert lifecycle_feature.remove_cleanup_data == {"cleaned": True, "removed_at": "test"}

        # Verify hooks are gone after removal
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0
        assert len(manager.get_hooks(HookEvent.POST_TOOL_USE)) == 0


# =============================================================================
# Test: Hook registration guarantees
# =============================================================================


class TestHookRegistrationGuarantees:
    """Verify hook registration/unregistration invariants."""

    @pytest.mark.asyncio
    async def test_no_duplicate_hooks_on_double_enable(self, mock_agent, lifecycle_feature):
        """Registering hooks twice should not create duplicates."""
        manager = mock_agent.hooks_manager

        await lifecycle_feature.initialize()

        # Register hooks twice
        for hook in lifecycle_feature.get_hooks():
            manager.register(hook)
        for hook in lifecycle_feature.get_hooks():
            manager.register(hook)

        # Should still have exactly 1 of each
        pre_hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        post_hooks = manager.get_hooks(HookEvent.POST_TOOL_USE)
        tx_hooks = [h for h in pre_hooks if h.name == "lifecycle_test_tx_security"]
        audit_hooks = [h for h in post_hooks if h.name == "lifecycle_test_audit_log"]
        assert len(tx_hooks) == 1, "No duplicate hooks on double register"
        assert len(audit_hooks) == 1, "No duplicate hooks on double register"

    @pytest.mark.asyncio
    async def test_unregister_idempotent(self, mock_agent, lifecycle_feature):
        """Unregistering already-unregistered hooks should not error."""
        manager = mock_agent.hooks_manager

        for hook in lifecycle_feature.get_hooks():
            manager.register(hook)

        # Unregister twice — should not raise
        for hook in lifecycle_feature.get_hooks():
            manager.unregister(hook)
        for hook in lifecycle_feature.get_hooks():
            manager.unregister(hook)  # Should be a no-op

        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0

    @pytest.mark.asyncio
    async def test_hook_execution_stops_after_unregister(self, mock_agent, lifecycle_feature):
        """After unregistering, hooks must not fire on execute_hooks."""
        manager = mock_agent.hooks_manager

        for hook in lifecycle_feature.get_hooks():
            manager.register(hook)

        hook_input = HookInput(
            session_id="test",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="any_tool",
        )

        # Fire once
        await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert lifecycle_feature.tx_hook.call_count == 1

        # Unregister
        for hook in lifecycle_feature.get_hooks():
            manager.unregister(hook)

        # Fire again — should not increment
        await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert lifecycle_feature.tx_hook.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_features_hooks_independent(self, mock_agent):
        """Disabling one feature's hooks should not affect another feature's hooks."""
        manager = mock_agent.hooks_manager

        feature_a = LifecycleTestFeature(mock_agent)
        feature_b = LifecycleTestFeature(mock_agent)

        # Give feature_b unique hook names to distinguish
        feature_b.tx_hook = TransactionSecurityHookStub()
        feature_b.tx_hook.name = "lifecycle_test_tx_security_b"
        feature_b.audit_hook = AuditLogHookStub()
        feature_b.audit_hook.name = "lifecycle_test_audit_log_b"

        # Register both
        for hook in feature_a.get_hooks():
            manager.register(hook)
        for hook in feature_b.get_hooks():
            manager.register(hook)

        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 2

        # Unregister feature_a's hooks only
        for hook in feature_a.get_hooks():
            manager.unregister(hook)

        # Feature B's hooks should still be there
        remaining = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        assert len(remaining) == 1
        assert remaining[0].name == "lifecycle_test_tx_security_b"


# =============================================================================
# Test: Lifecycle callbacks ordering
# =============================================================================


class TestLifecycleCallbackOrdering:
    """Verify lifecycle callbacks are called in the correct order."""

    @pytest.mark.asyncio
    async def test_initialize_before_enable(self, mock_agent, lifecycle_feature):
        """initialize() must be called before on_enable()."""
        assert lifecycle_feature.initialize_count == 0
        assert lifecycle_feature.enable_count == 0

        await lifecycle_feature.initialize()
        assert lifecycle_feature.initialize_count == 1

        await lifecycle_feature.on_enable()
        assert lifecycle_feature.enable_count == 1

    @pytest.mark.asyncio
    async def test_disable_before_unregister(self, mock_agent, lifecycle_feature):
        """on_disable() should be called before hooks are unregistered
        (matching _disable_feature behavior)."""
        manager = mock_agent.hooks_manager

        for hook in lifecycle_feature.get_hooks():
            manager.register(hook)

        # Simulate _disable_feature: on_disable first, then unregister
        await lifecycle_feature.on_disable()
        assert lifecycle_feature.disable_count == 1
        # Hooks still registered at this point (feature can do final hook work)
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 1

        for hook in lifecycle_feature.get_hooks():
            manager.unregister(hook)
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0

    @pytest.mark.asyncio
    async def test_on_remove_called_before_pip_uninstall(self, mock_agent, lifecycle_feature):
        """on_remove() should run cleanup before the package is removed."""
        manager = mock_agent.hooks_manager

        for hook in lifecycle_feature.get_hooks():
            manager.register(hook)

        # Unregister hooks, then call on_remove (matching remove endpoint)
        for hook in lifecycle_feature.get_hooks():
            manager.unregister(hook)
        await lifecycle_feature.on_remove()

        assert lifecycle_feature.remove_count == 1
        assert lifecycle_feature.remove_cleanup_data is not None
        assert lifecycle_feature.remove_cleanup_data["cleaned"] is True


# =============================================================================
# Test: Config persistence across enable/disable cycles
# =============================================================================


class TestConfigPersistenceAcrossCycles:
    """Verify feature config survives enable/disable cycles."""

    @pytest.mark.asyncio
    async def test_config_survives_disable_enable(self, mock_agent, lifecycle_feature):
        """Config values set before disable should be available after re-enable."""
        await lifecycle_feature.set_config({"max_amount": 9999, "require_approval": False})

        # Disable
        await lifecycle_feature.on_disable()

        # Config should still be accessible on the feature object
        config = await lifecycle_feature.get_config()
        assert config["max_amount"] == 9999
        assert config["require_approval"] is False

        # Re-enable
        await lifecycle_feature.on_enable()

        config_after = await lifecycle_feature.get_config()
        assert config_after["max_amount"] == 9999
        assert config_after["require_approval"] is False

    @pytest.mark.asyncio
    async def test_config_schema_available_after_re_enable(self, mock_agent, lifecycle_feature):
        """Config schema should be available after re-enable."""
        await lifecycle_feature.on_disable()
        await lifecycle_feature.on_enable()

        schema = lifecycle_feature.config_schema
        assert schema is not None
        assert "max_amount" in schema["properties"]

    @pytest.mark.asyncio
    async def test_multiple_disable_enable_cycles(self, mock_agent, lifecycle_feature):
        """Config and hooks survive multiple disable/enable cycles."""
        manager = mock_agent.hooks_manager

        for cycle in range(3):
            # Enable
            for hook in lifecycle_feature.get_hooks():
                manager.register(hook)
            await lifecycle_feature.on_enable()

            # Set different config each cycle
            await lifecycle_feature.set_config({"max_amount": (cycle + 1) * 100})

            # Verify hooks active
            assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 1

            # Disable
            await lifecycle_feature.on_disable()
            for hook in lifecycle_feature.get_hooks():
                manager.unregister(hook)

            # Verify hooks gone
            assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0

            # Verify config persisted
            config = await lifecycle_feature.get_config()
            assert config["max_amount"] == (cycle + 1) * 100

        assert lifecycle_feature.enable_count == 3
        assert lifecycle_feature.disable_count == 3


# =============================================================================
# Test: Agent-level lifecycle (using real _register_feature/_disable_feature)
# =============================================================================


class TestAgentLevelLifecycle:
    """Test lifecycle via KestrelAgent._register_feature/_disable_feature."""

    @pytest_asyncio.fixture
    async def agent_with_feature(self, temp_db):
        """Create a real KestrelAgent and register our test feature."""
        llm_service = LLMService()
        agent = KestrelAgent(
            did="did:test:lifecycle-e2e",
            storage_path=str(temp_db),
            llm_service=llm_service,
            privacy_mode=PrivacyMode.NORMAL,
        )
        await agent.initialize()

        # SecurityHook permissions for the registered features are
        # auto-granted by the _auto_grant_security_permissions
        # autouse fixture in tests/integration/conftest.py.  But
        # this test fires HookInput(tool_name="test_tool") with no
        # explicit feature_name, so SecurityHook resolves it as
        # ("unknown", "test_tool") — a synthetic pair that isn't in
        # the registered-tools tree.  Grant it ALLOW directly.
        from kestrel_sovereign.features.security.permissions import PermissionLevel
        security_feature = agent.get_feature("SecurityFeature")
        if security_feature and security_feature.permission_store:
            await security_feature.permission_store.set_permission(
                feature_name="unknown",
                tool_name="test_tool",
                level=PermissionLevel.ALLOW,
                scope="always",
                reason="feature-lifecycle-e2e synthetic hook input",
            )

        feature = LifecycleTestFeature(agent)

        yield agent, feature

        await agent.shutdown()
        await llm_service.close()

    @pytest.mark.asyncio
    async def test_register_feature_registers_hooks(self, agent_with_feature):
        """_register_feature should register all hooks from get_hooks()."""
        agent, feature = agent_with_feature

        await agent._register_feature(feature)

        # Hooks should be in the agent's hooks_manager
        manager = agent.hooks_manager
        pre_hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        post_hooks = manager.get_hooks(HookEvent.POST_TOOL_USE)

        assert any(h.name == "lifecycle_test_tx_security" for h in pre_hooks)
        assert any(h.name == "lifecycle_test_audit_log" for h in post_hooks)
        assert feature.initialize_count == 1
        assert feature.enable_count == 1

    @pytest.mark.asyncio
    async def test_disable_feature_unregisters_hooks(self, agent_with_feature):
        """_disable_feature should unregister all hooks — no stale hooks."""
        agent, feature = agent_with_feature

        await agent._register_feature(feature)
        manager = agent.hooks_manager

        # Verify hooks are there
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) >= 1

        # Disable
        await agent._disable_feature(feature.name)

        # Verify hooks are gone
        tx_hooks = [h for h in manager.get_hooks(HookEvent.PRE_TOOL_USE)
                     if h.name == "lifecycle_test_tx_security"]
        audit_hooks = [h for h in manager.get_hooks(HookEvent.POST_TOOL_USE)
                        if h.name == "lifecycle_test_audit_log"]
        assert len(tx_hooks) == 0, "No stale transaction_security hooks"
        assert len(audit_hooks) == 0, "No stale audit_log hooks"
        assert feature.disable_count == 1

    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, agent_with_feature):
        """Full lifecycle via agent methods: register → disable → re-enable → verify."""
        agent, feature = agent_with_feature
        manager = agent.hooks_manager

        # Register
        await agent._register_feature(feature)
        assert feature.name in agent.features

        # Execute hook to prove it's active
        hook_input = HookInput(
            session_id="agent-lifecycle-test",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="test_tool",
        )
        result = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert feature.tx_hook.call_count == 1

        # Disable
        await agent._disable_feature(feature.name)
        result = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert feature.tx_hook.call_count == 1, "Hook should not fire after disable"

        # Re-enable (manually, since _register_feature would re-initialize)
        for hook in feature.get_hooks():
            manager.register(hook)
        await feature.on_enable()

        result = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert feature.tx_hook.call_count == 2, "Hook should fire after re-enable"


# =============================================================================
# Test: API endpoint lifecycle (using FastAPI test client)
# =============================================================================


class TestAPIEndpointLifecycle:
    """Test enable/disable/remove via the FastAPI endpoints."""

    @pytest.fixture
    def app_with_feature(self):
        """Create a FastAPI app with the features router and a mock agent."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from endpoints.features import router

        app = FastAPI()
        app.include_router(router)

        # Create mock agent with real HooksManager
        manager = HooksManager()
        agent = MagicMock()
        agent.hooks_manager = manager
        agent.features = {}
        agent.storage = None
        agent.task_manager = None

        feature = LifecycleTestFeature(agent)
        agent.features["LifecycleTestFeature"] = feature

        # Register hooks initially (simulating _register_feature)
        for hook in feature.get_hooks():
            manager.register(hook)

        app.state.agent = agent

        client = TestClient(app)
        return client, agent, feature, manager

    def test_disable_endpoint_unregisters_hooks(self, app_with_feature):
        """POST /api/features/{name}/disable should unregister hooks."""
        client, agent, feature, manager = app_with_feature

        # Verify hooks are registered
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 1

        # Disable via API
        response = client.post("/api/features/LifecycleTestFeature/disable")
        assert response.status_code == 200
        assert response.json()["status"] == "disabled"

        # Verify hooks are unregistered
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0
        assert len(manager.get_hooks(HookEvent.POST_TOOL_USE)) == 0
        assert feature.disable_count == 1

    def test_enable_endpoint_registers_hooks(self, app_with_feature):
        """POST /api/features/{name}/enable should register hooks."""
        client, agent, feature, manager = app_with_feature

        # First disable to clear hooks
        client.post("/api/features/LifecycleTestFeature/disable")
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0

        # Re-enable via API
        response = client.post("/api/features/LifecycleTestFeature/enable")
        assert response.status_code == 200
        assert response.json()["status"] == "enabled"

        # Verify hooks are re-registered
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 1
        assert len(manager.get_hooks(HookEvent.POST_TOOL_USE)) == 1
        assert feature.enable_count == 1  # Only the re-enable call

    def test_disable_enable_cycle_no_stale_hooks(self, app_with_feature):
        """Repeated disable/enable cycles should never leave stale hooks."""
        client, agent, feature, manager = app_with_feature

        for _ in range(5):
            # Disable
            resp = client.post("/api/features/LifecycleTestFeature/disable")
            assert resp.status_code == 200
            assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0

            # Enable
            resp = client.post("/api/features/LifecycleTestFeature/enable")
            assert resp.status_code == 200
            pre_hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
            assert len(pre_hooks) == 1, f"Expected 1 PRE_TOOL_USE hook, got {len(pre_hooks)}"
            post_hooks = manager.get_hooks(HookEvent.POST_TOOL_USE)
            assert len(post_hooks) == 1, f"Expected 1 POST_TOOL_USE hook, got {len(post_hooks)}"

    def test_disable_enable_preserves_config(self, app_with_feature):
        """Config should persist across API disable/enable cycles."""
        client, agent, feature, manager = app_with_feature

        # Set config via the feature directly (as the API would via PATCH)
        import asyncio
        asyncio.run(feature.set_config({"max_amount": 777}))

        # Disable
        client.post("/api/features/LifecycleTestFeature/disable")

        # Enable
        client.post("/api/features/LifecycleTestFeature/enable")

        # Config should be preserved
        config = asyncio.run(feature.get_config())
        assert config["max_amount"] == 777

    def test_feature_detail_shows_hooks(self, app_with_feature):
        """GET /api/features/{name} should list registered hooks."""
        client, agent, feature, manager = app_with_feature

        response = client.get("/api/features/LifecycleTestFeature")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "enabled"
        hook_names = [h["name"] for h in data["hooks"]]
        assert "lifecycle_test_tx_security" in hook_names
        assert "lifecycle_test_audit_log" in hook_names

        # Verify events are serialized correctly
        for hook_info in data["hooks"]:
            assert "events" in hook_info
            assert isinstance(hook_info["events"], list)

    def test_enable_nonexistent_feature_404(self, app_with_feature):
        """Enabling a feature that's not loaded returns 404."""
        client, agent, feature, manager = app_with_feature

        response = client.post("/api/features/NonExistent/enable")
        assert response.status_code == 404

    def test_disable_nonexistent_feature_404(self, app_with_feature):
        """Disabling a feature that's not loaded returns 404."""
        client, agent, feature, manager = app_with_feature

        response = client.post("/api/features/NonExistent/disable")
        assert response.status_code == 404
