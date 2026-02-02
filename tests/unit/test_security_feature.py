"""
Unit Tests for Kestrel Security Feature.

Tests the security infrastructure:
- PermissionStore
- ApprovalQueue
- SecurityHook
- SecurityFeature
"""

import asyncio
import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Registry to track stores for cleanup
_stores_to_close = []


@pytest.fixture(autouse=True)
def _cleanup_stores():
    """Clear store registry before each test."""
    _stores_to_close.clear()
    yield


@pytest_asyncio.fixture(autouse=True)
async def _async_cleanup_stores():
    """Async cleanup for stores after each test."""
    yield
    # Cancel any pending tasks that might be waiting
    for task in asyncio.all_tasks():
        if task.get_name().startswith('Task-') and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
    for store in _stores_to_close:
        try:
            if hasattr(store, 'close'):
                await store.close()
            elif hasattr(store, '_db') and store._db:
                await store._db.close()
        except Exception:
            pass
    _stores_to_close.clear()


def track_store(store):
    """Register a store for cleanup after the test."""
    _stores_to_close.append(store)
    return store


from kestrel_sovereign.features.security.permissions import (
    PermissionLevel,
    PermissionStore,
    ToolPermission,
    FeaturePermissions,
)
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue, ApprovalRequest
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.hooks import HookInput, HookEvent, PermissionDecision


# === PermissionStore Tests ===

class TestPermissionStore:
    """Tests for PermissionStore."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a temporary database path."""
        return str(tmp_path / "test_permissions.db")

    @pytest.fixture
    async def store(self, db_path):
        """Create and initialize a permission store."""
        store = track_store(PermissionStore(db_path))
        await store.initialize()
        yield store

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, db_path):
        store = track_store(PermissionStore(db_path))
        await store.initialize()
        assert os.path.exists(db_path)

    @pytest.mark.asyncio
    async def test_register_tool(self, store):
        await store.register_tool(
            feature_name="WalletAgent",
            tool_name="get_balance",
            default_level=PermissionLevel.ASK,
        )

        level = await store.get_permission("WalletAgent", "get_balance")
        assert level == PermissionLevel.ASK

    @pytest.mark.asyncio
    async def test_set_permission(self, store):
        # Register first
        await store.register_tool("WalletAgent", "get_balance", PermissionLevel.ASK)

        # Change permission
        await store.set_permission(
            feature_name="WalletAgent",
            tool_name="get_balance",
            level=PermissionLevel.ALLOW,
        )

        level = await store.get_permission("WalletAgent", "get_balance")
        assert level == PermissionLevel.ALLOW

    @pytest.mark.asyncio
    async def test_set_permission_with_scope(self, store):
        await store.register_tool("WalletAgent", "send_tokens", PermissionLevel.ASK)

        # Set with session scope
        await store.set_permission(
            feature_name="WalletAgent",
            tool_name="send_tokens",
            level=PermissionLevel.ALLOW,
            scope="session",
        )

        level = await store.get_permission("WalletAgent", "send_tokens")
        assert level == PermissionLevel.ALLOW

    @pytest.mark.asyncio
    async def test_session_overrides(self, store):
        await store.register_tool("WalletAgent", "get_balance", PermissionLevel.ASK)

        # Set session override
        await store.set_permission(
            feature_name="WalletAgent",
            tool_name="get_balance",
            level=PermissionLevel.ALLOW,
            scope="session",
        )

        # Session override should take effect
        level = await store.get_permission("WalletAgent", "get_balance")
        assert level == PermissionLevel.ALLOW

        # Clear session overrides
        store.clear_session_overrides()

        # Should fall back to stored permission
        level = await store.get_permission("WalletAgent", "get_balance")
        assert level == PermissionLevel.ASK

    @pytest.mark.asyncio
    async def test_set_feature_permission(self, store):
        # Register multiple tools
        await store.register_tool("WalletAgent", "get_balance", PermissionLevel.ASK)
        await store.register_tool("WalletAgent", "send_tokens", PermissionLevel.ASK)
        await store.register_tool("WalletAgent", "get_history", PermissionLevel.ASK)

        # Set all to ALLOW
        await store.set_feature_permission("WalletAgent", PermissionLevel.ALLOW)

        # Check all
        assert await store.get_permission("WalletAgent", "get_balance") == PermissionLevel.ALLOW
        assert await store.get_permission("WalletAgent", "send_tokens") == PermissionLevel.ALLOW
        assert await store.get_permission("WalletAgent", "get_history") == PermissionLevel.ALLOW

    @pytest.mark.asyncio
    async def test_get_permission_tree(self, store):
        # Register tools
        await store.register_tool("WalletAgent", "get_balance", PermissionLevel.ALLOW)
        await store.register_tool("WalletAgent", "send_tokens", PermissionLevel.DENY)
        await store.register_tool("SearchFeature", "web_search", PermissionLevel.ASK)

        tree = await store.get_permission_tree()

        assert len(tree) == 2  # Two features
        wallet_feature = next(f for f in tree if f.feature_name == "WalletAgent")
        assert len(wallet_feature.tools) == 2
        assert wallet_feature.rollup_state == "mixed"

        search_feature = next(f for f in tree if f.feature_name == "SearchFeature")
        assert len(search_feature.tools) == 1
        assert search_feature.rollup_state == "ask_all"

    @pytest.mark.asyncio
    async def test_log_decision(self, store):
        await store.log_decision(
            feature_name="WalletAgent",
            tool_name="get_balance",
            action="tool_execution",
            decision="auto_allowed",
        )

        logs = await store.get_audit_log(limit=10)
        assert len(logs) == 1
        assert logs[0]["feature"] == "WalletAgent"
        assert logs[0]["tool"] == "get_balance"
        assert logs[0]["decision"] == "auto_allowed"

    @pytest.mark.asyncio
    async def test_get_audit_log_limit(self, store):
        # Log multiple decisions
        for i in range(15):
            await store.log_decision(
                feature_name=f"Feature{i}",
                tool_name="tool",
                action="test",
                decision="auto_allowed",
            )

        logs = await store.get_audit_log(limit=10)
        assert len(logs) == 10


# === ApprovalQueue Tests ===

class TestApprovalQueue:
    """Tests for ApprovalQueue."""

    @pytest.fixture
    def queue(self):
        """Create an approval queue."""
        return ApprovalQueue()

    @pytest.mark.asyncio
    async def test_request_approval_and_approve(self, queue):
        # Start approval request in background
        async def approve_later():
            await asyncio.sleep(0.1)
            # Find the request
            pending = queue.pending_requests
            assert len(pending) == 1
            queue.submit_decision(pending[0].id, True, "session")

        asyncio.create_task(approve_later())

        approved, scope = await queue.request_approval(
            feature_name="WalletAgent",
            tool_name="get_balance",
            tool_args={"test": True},
        )

        assert approved is True
        assert scope == "session"

    @pytest.mark.asyncio
    async def test_request_approval_and_deny(self, queue):
        async def deny_later():
            await asyncio.sleep(0.1)
            pending = queue.pending_requests
            assert len(pending) == 1
            queue.submit_decision(pending[0].id, False, "denied")

        asyncio.create_task(deny_later())

        approved, scope = await queue.request_approval(
            feature_name="WalletAgent",
            tool_name="send_tokens",
            tool_args={"amount": 100},
        )

        assert approved is False
        assert scope == "denied"

    @pytest.mark.asyncio
    async def test_request_approval_timeout(self, queue):
        # Use very short timeout to test timeout behavior
        approved, scope = await queue.request_approval(
            feature_name="WalletAgent",
            tool_name="slow_operation",
            tool_args={},
            timeout=0.1,  # Pass timeout as parameter
        )

        assert approved is False
        assert scope == "timeout"

    @pytest.mark.asyncio
    async def test_cancel_request(self, queue):
        # Start a request
        request_task = asyncio.create_task(
            queue.request_approval("Feature", "tool", {})
        )

        await asyncio.sleep(0.05)
        pending = queue.pending_requests
        assert len(pending) == 1

        # Cancel it
        result = queue.cancel_request(pending[0].id)
        assert result is True

        approved, scope = await request_task
        assert approved is False

    @pytest.mark.asyncio
    async def test_cancel_all(self, queue):
        # Start multiple requests
        tasks = [
            asyncio.create_task(queue.request_approval(f"Feature{i}", "tool", {}))
            for i in range(3)
        ]

        await asyncio.sleep(0.05)
        assert len(queue.pending_requests) == 3

        # Cancel all
        count = queue.cancel_all()
        assert count == 3

        # All should return denied
        for task in tasks:
            approved, scope = await task
            assert approved is False

    @pytest.mark.asyncio
    async def test_callback_on_request(self):
        callback_calls = []

        async def callback(request: ApprovalRequest):
            callback_calls.append(request)

        queue = ApprovalQueue(on_request_added=callback)

        # Start request but cancel quickly
        request_task = asyncio.create_task(
            queue.request_approval("Feature", "tool", {"arg": "value"})
        )

        await asyncio.sleep(0.05)
        queue.cancel_all()
        await request_task

        assert len(callback_calls) == 1
        assert callback_calls[0].feature_name == "Feature"
        assert callback_calls[0].tool_args == {"arg": "value"}


# === SecurityHook Tests ===

class TestSecurityHook:
    """Tests for SecurityHook."""

    @pytest.fixture
    async def permission_store(self, tmp_path):
        """Create a permission store."""
        store = track_store(PermissionStore(str(tmp_path / "test.db")))
        await store.initialize()
        return store

    @pytest.fixture
    def approval_queue(self):
        """Create an approval queue."""
        return ApprovalQueue()

    @pytest.fixture
    async def hook(self, permission_store, approval_queue):
        """Create a security hook."""
        return SecurityHook(permission_store, approval_queue)

    @pytest.mark.asyncio
    async def test_auto_allow(self, hook, permission_store):
        await permission_store.register_tool(
            "WalletAgent", "get_balance", PermissionLevel.ALLOW
        )

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="get_balance",
            feature_name="WalletAgent",
            tool_input={},
        )

        output = await hook.execute(input)
        assert output.continue_execution is True
        assert output.permission_decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_auto_deny(self, hook, permission_store):
        await permission_store.register_tool(
            "WalletAgent", "delete_everything", PermissionLevel.DENY
        )

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="delete_everything",
            feature_name="WalletAgent",
            tool_input={},
        )

        output = await hook.execute(input)
        assert output.continue_execution is False
        assert output.permission_decision == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_ask_mode_approve(self, hook, permission_store, approval_queue):
        await permission_store.register_tool(
            "WalletAgent", "send_tokens", PermissionLevel.ASK
        )

        # Approve in background
        async def approve_later():
            await asyncio.sleep(0.1)
            pending = approval_queue.pending_requests
            if pending:
                approval_queue.submit_decision(pending[0].id, True, "once")

        asyncio.create_task(approve_later())

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="send_tokens",
            feature_name="WalletAgent",
            tool_input={"amount": 50},
        )

        output = await hook.execute(input)
        assert output.continue_execution is True
        assert output.permission_decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_ask_mode_deny(self, hook, permission_store, approval_queue):
        await permission_store.register_tool(
            "WalletAgent", "send_tokens", PermissionLevel.ASK
        )

        # Deny in background
        async def deny_later():
            await asyncio.sleep(0.1)
            pending = approval_queue.pending_requests
            if pending:
                approval_queue.submit_decision(pending[0].id, False, "denied")

        asyncio.create_task(deny_later())

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="send_tokens",
            feature_name="WalletAgent",
            tool_input={"amount": 50},
        )

        output = await hook.execute(input)
        assert output.continue_execution is False
        assert output.permission_decision == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_ask_approve_always_updates_permission(
        self, hook, permission_store, approval_queue
    ):
        await permission_store.register_tool(
            "WalletAgent", "get_balance", PermissionLevel.ASK
        )

        # Approve with "always" scope
        async def approve_always():
            await asyncio.sleep(0.1)
            pending = approval_queue.pending_requests
            if pending:
                approval_queue.submit_decision(pending[0].id, True, "always")

        asyncio.create_task(approve_always())

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="get_balance",
            feature_name="WalletAgent",
            tool_input={},
        )

        await hook.execute(input)

        # Permission should now be ALLOW
        level = await permission_store.get_permission("WalletAgent", "get_balance")
        assert level == PermissionLevel.ALLOW

    @pytest.mark.asyncio
    async def test_mask_sensitive_data(self, hook):
        sensitive_data = {
            "password": "secret123",
            "api_key": "sk-abc123",
            "user": "john",
            "nested": {"token": "tok-xyz"},
        }

        masked = hook._mask_sensitive(sensitive_data)

        assert masked["password"] == "***MASKED***"
        assert masked["api_key"] == "***MASKED***"
        assert masked["user"] == "john"  # Not sensitive
        assert masked["nested"]["token"] == "***MASKED***"


# === Integration Test ===

class TestSecurityIntegration:
    """Integration tests for the security system."""

    @pytest.fixture
    async def setup(self, tmp_path):
        """Set up the full security stack."""
        db_path = str(tmp_path / "test.db")
        store = track_store(PermissionStore(db_path))
        await store.initialize()

        queue = ApprovalQueue()
        hook = SecurityHook(store, queue)

        return store, queue, hook

    @pytest.mark.asyncio
    async def test_full_approval_flow(self, setup):
        store, queue, hook = setup

        # Register tool with ASK permission
        await store.register_tool("MCPAgent", "execute_server", PermissionLevel.ASK)

        # Start approval flow in background
        async def user_approves():
            await asyncio.sleep(0.1)
            pending = queue.pending_requests
            if pending:
                queue.submit_decision(pending[0].id, True, "session")

        asyncio.create_task(user_approves())

        # Execute hook
        input = HookInput(
            session_id="test-session",
            hook_event_name="PreToolUse",
            tool_name="execute_server",
            feature_name="MCPAgent",
            tool_input={"server": "test-server"},
        )

        output = await hook.execute(input)
        assert output.continue_execution is True

        # Check audit log
        logs = await store.get_audit_log(limit=1)
        assert logs[0]["decision"] == "user_approved"
        assert logs[0]["user_choice"] == "session"


# === Run tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
