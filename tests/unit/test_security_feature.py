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
from datetime import datetime, timedelta, timezone

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
from kestrel_sovereign.features.security.approval_queue import (
    ApprovalQueue,
    ApprovalRequest,
    DecisionResult,
)
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.feature import SecurityFeature
from kestrel_sdk.hooks.base import HookInput, HookEvent, PermissionDecision


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
    async def test_global_auto_mode_overrides_non_denied_permissions(self, store):
        await store.register_tool("WalletAgent", "get_balance", PermissionLevel.ASK)
        await store.register_tool("WalletAgent", "delete_everything", PermissionLevel.DENY)
        await store.register_tool("ShellFeature", "rm", PermissionLevel.ALWAYS_ASK)
        await store.register_tool("SearchFeature", "web_search", PermissionLevel.ALLOW)

        store.set_global_auto_mode(True)

        assert await store.get_permission("WalletAgent", "get_balance") == PermissionLevel.AUTO
        assert await store.get_permission("SearchFeature", "web_search") == PermissionLevel.AUTO
        assert await store.get_permission("NewFeature", "new_tool") == PermissionLevel.AUTO
        assert await store.get_permission("WalletAgent", "delete_everything") == PermissionLevel.DENY
        assert await store.get_permission("ShellFeature", "rm") == PermissionLevel.ALWAYS_ASK

    @pytest.mark.asyncio
    async def test_clear_session_overrides_disables_global_auto(self, store):
        await store.register_tool("WalletAgent", "get_balance", PermissionLevel.ASK)
        store.set_global_auto_mode(True)

        store.clear_session_overrides()

        assert store.get_global_auto_mode() is False
        assert await store.get_permission("WalletAgent", "get_balance") == PermissionLevel.ASK

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
    async def test_auto_permission_rolls_up(self, store):
        await store.register_tool("ComputeFeature", "run_script", PermissionLevel.AUTO)
        await store.register_tool("ComputeFeature", "list_scripts", PermissionLevel.AUTO)

        tree = await store.get_permission_tree()

        compute_feature = next(f for f in tree if f.feature_name == "ComputeFeature")
        assert compute_feature.rollup_state == "auto_all"
        assert {tool.level for tool in compute_feature.tools} == {PermissionLevel.AUTO}

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
            await queue.submit_decision(pending[0].id, True, "session")

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
            await queue.submit_decision(pending[0].id, False, "denied")

        asyncio.create_task(deny_later())

        approved, scope = await queue.request_approval(
            feature_name="WalletAgent",
            tool_name="send_tokens",
            tool_args={"amount": 100},
        )

        assert approved is False
        assert scope == "denied"

    @pytest.mark.asyncio
    async def test_submit_decision_is_idempotent(self, queue):
        """A second decision on the same request must be rejected.

        Closes the race that flaked test_fs_write_requires_real_approval
        on the v0.10.0 release-sign rerun: submit_decision() resolves
        resume_event but does not pop _pending — that pop happens in
        request_approval()'s finally-block on the awaiter's next tick.
        Without this guard, a polling responder (UI double-click,
        background poller) silently overwrites the user's first
        decision on every loop iteration that lands inside the race
        window.
        """
        async def respond_twice():
            await asyncio.sleep(0.05)
            pending = queue.pending_requests
            assert len(pending) == 1
            req_id = pending[0].id
            # First decision: approve once. Must succeed.
            result = await queue.submit_decision(req_id, True, "once")
            assert result.in_memory is True
            assert result.persisted is True
            # Second decision: approve again with a different scope.
            # Must be rejected — same request, already decided.
            result = await queue.submit_decision(req_id, True, "session")
            assert result.in_memory is False
            # Even an opposite decision must be rejected.
            result = await queue.submit_decision(req_id, False, "denied")
            assert result.in_memory is False

        asyncio.create_task(respond_twice())

        approved, scope = await queue.request_approval(
            feature_name="ComputerUseFeature",
            tool_name="fs_write",
            tool_args={"path": "/tmp/x"},
        )

        # First decision wins; second/third are no-ops.
        assert approved is True
        assert scope == "once"

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

    @pytest.mark.asyncio
    async def test_withdrawn_callback_fires_on_timeout(self):
        # #877: when ``request_approval`` exits via timeout (no user
        # submit), ``_on_request_withdrawn`` MUST fire so the UI can close
        # any modal showing this request — otherwise the user clicks
        # ""Approve"" on a stale modal and submits to a now-empty queue,
        # getting a 404 with a misleading ""expired"" message.
        added: list = []
        withdrawn: list = []

        async def on_added(req: ApprovalRequest):
            added.append(req)

        async def on_withdrawn(req: ApprovalRequest, reason: str):
            withdrawn.append((req, reason))

        queue = ApprovalQueue(
            on_request_added=on_added,
            on_request_withdrawn=on_withdrawn,
        )

        approved, scope = await queue.request_approval(
            "Feature", "tool", {}, timeout=0.05,
        )
        assert (approved, scope) == (False, "timeout")
        assert len(withdrawn) == 1
        assert withdrawn[0][0].id == added[0].id
        assert withdrawn[0][1] == "timeout"

    @pytest.mark.asyncio
    async def test_cancellation_keeps_request_alive_for_user(self):
        # New contract (replaces the old #877 spackle): when the
        # calling task is cancelled (HTTP stream dropped, browser
        # closed, user switched to a different agent in the
        # multi_agent), the request MUST survive in ``_pending`` so the
        # user's modal stays interactive. No ``approval_withdrawn``
        # event fires, because the user has not abandoned the
        # decision — only the agent's read of it.
        withdrawn: list = []

        async def on_withdrawn(req: ApprovalRequest, reason: str):
            withdrawn.append((req, reason))

        queue = ApprovalQueue(on_request_withdrawn=on_withdrawn)

        request_task = asyncio.create_task(
            queue.request_approval("Feature", "tool", {})
        )
        await asyncio.sleep(0.05)
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

        # Request stayed alive. No withdrawal event fired.
        assert len(queue._pending) == 1
        assert withdrawn == []

        # The user can still submit a decision against the surviving
        # request — exactly the "user comes back later" path that
        # the old behavior denied them.
        request_id = next(iter(queue._pending.keys()))
        result = await queue.submit_decision(
            request_id, approved=True, scope="once",
        )
        assert result.in_memory is True

    @pytest.mark.asyncio
    async def test_sweep_stale_reaps_old_pending(self):
        # The cancellation-leaves-request-alive contract means
        # ``_pending`` grows when many agent tasks die before users
        # decide. ``sweep_stale(cutoff)`` is the operator's GC: pass
        # an explicit age cutoff, anything older gets reaped.
        from datetime import timedelta

        withdrawn: list = []

        async def on_withdrawn(req: ApprovalRequest, reason: str):
            withdrawn.append((req, reason))

        queue = ApprovalQueue(on_request_withdrawn=on_withdrawn)

        fresh = ApprovalRequest(
            id="fresh", feature_name="f", tool_name="t", tool_args={},
            created_at=datetime.now(timezone.utc),
        )
        ancient = ApprovalRequest(
            id="ancient", feature_name="f", tool_name="t", tool_args={},
            created_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        queue._pending[fresh.id] = fresh
        queue._pending[ancient.id] = ancient

        # 24h cutoff: ancient is older, fresh isn't.
        reaped = await queue.sweep_stale(older_than_seconds=24 * 3600)
        assert reaped == 1
        assert "fresh" in queue._pending
        assert "ancient" not in queue._pending
        assert len(withdrawn) == 1
        assert withdrawn[0][0].id == "ancient"
        assert withdrawn[0][1] == "timeout"

    @pytest.mark.asyncio
    async def test_withdrawn_callback_does_not_fire_on_user_submit(self):
        # User-submitted decisions resolve via ``submit_decision`` which
        # sets ``resume_event``. The normal path runs, ``request_approval``
        # returns. The withdrawal callback must NOT fire — the modal
        # already closed when the user clicked.
        withdrawn: list = []

        async def on_withdrawn(req: ApprovalRequest, reason: str):
            withdrawn.append((req, reason))

        queue = ApprovalQueue(on_request_withdrawn=on_withdrawn)

        request_task = asyncio.create_task(
            queue.request_approval("Feature", "tool", {})
        )
        await asyncio.sleep(0.05)
        # Find the queued request and resolve it as if the user clicked.
        request_id = next(iter(queue._pending.keys()))
        result = await queue.submit_decision(
            request_id, approved=True, scope="once"
        )
        assert result.in_memory is True
        approved, scope = await request_task
        assert approved is True
        assert scope == "once"
        assert withdrawn == []  # no spurious withdrawal event


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
    async def approval_queue(self, permission_store):
        """Create an approval queue wired to the permission store.

        Mirrors production wiring (SecurityFeature passes the store in)
        so the queue can persist scope choices and write audit rows
        centrally — see #785.
        """
        return ApprovalQueue(permission_store=permission_store)

    @pytest.fixture
    async def hook(self, permission_store, approval_queue):
        """Create a security hook."""
        return SecurityHook(permission_store, approval_queue)

    def test_security_hook_is_marked_awaits_user_input(
        self, permission_store, approval_queue,
    ):
        """SecurityHook blocks on a human decision, so the hook
        manager must NOT wrap it in ``asyncio.wait_for``. The
        ``awaits_user_input=True`` flag tells the manager to skip
        the watchdog. Without it, the manager's default 5s timeout
        cancelled the queue's await before the user could click —
        the actual driver of the "modal disappears in ~5 seconds"
        bug. Bumping the timeout to a bigger arbitrary number was
        spackle; the structural answer is to take the manager's
        clock off this hook entirely.
        """
        hook = SecurityHook(permission_store, approval_queue)
        assert hook.awaits_user_input is True, (
            "SecurityHook must declare awaits_user_input=True so the "
            "hook manager doesn't impose a timeout on a human "
            "response."
        )

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
    async def test_auto_mode_allows_without_queue_and_audits_distinctly(
        self, hook, permission_store, approval_queue
    ):
        await permission_store.register_tool(
            "WalletAgent", "get_balance", PermissionLevel.AUTO
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
        assert approval_queue.pending_requests == []

        logs = await permission_store.get_audit_log(limit=1)
        assert logs[0]["decision"] == "auto_mode_allowed"
        assert logs[0]["user_choice"] == "constitutional_honesty_unflagged"

    @pytest.mark.asyncio
    async def test_always_ask_prompts_under_global_auto_mode(
        self, hook, permission_store, approval_queue
    ):
        await permission_store.register_tool(
            "ShellFeature", "rm", PermissionLevel.ALWAYS_ASK
        )
        permission_store.set_global_auto_mode(True)
        request_was_pending = False

        async def approve_later():
            nonlocal request_was_pending
            await asyncio.sleep(0.05)
            pending = approval_queue.pending_requests
            request_was_pending = bool(pending)
            if pending:
                await approval_queue.submit_decision(pending[0].id, True, "once")

        asyncio.create_task(approve_later())
        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="rm",
            feature_name="ShellFeature",
            tool_input={"command": "rm /tmp/outside-workspace"},
        )

        output = await hook.execute(input)

        assert output.continue_execution is True
        assert output.permission_decision == PermissionDecision.ALLOW
        assert request_was_pending is True

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
                await approval_queue.submit_decision(pending[0].id, True, "once")

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
                await approval_queue.submit_decision(pending[0].id, False, "denied")

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
                await approval_queue.submit_decision(pending[0].id, True, "always")

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
        """Set up the full security stack with the queue wired to the store
        (matches production wiring per #785 — the queue owns scope
        persistence and audit-row writes)."""
        db_path = str(tmp_path / "test.db")
        store = track_store(PermissionStore(db_path))
        await store.initialize()

        queue = ApprovalQueue(permission_store=store)
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
                await queue.submit_decision(pending[0].id, True, "session")

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


class TestApprovalQueueScopePersistence:
    """Regression suite for #785.

    Six features (code_edit, compute, keys, reflection.*) call
    ``ApprovalQueue.request_approval`` directly without going through
    ``SecurityHook``.  Before #785 they all discarded the user's scope,
    so "This Session" / "Always" never stuck.  These tests pin that the
    queue itself now persists scope and writes audit rows whenever a
    ``permission_store`` is wired in.
    """

    @pytest.fixture
    async def store(self, tmp_path):
        from kestrel_sovereign.features.security.permissions import PermissionStore
        s = PermissionStore(str(tmp_path / "scope.db"))
        await s.initialize()
        return s

    @pytest.fixture
    def queue_with_store(self, store):
        return ApprovalQueue(permission_store=store)

    @pytest.fixture
    def queue_without_store(self):
        return ApprovalQueue()

    @pytest.mark.asyncio
    async def test_global_auto_mode_direct_request_approval_does_not_prompt(
        self, store
    ):
        """Direct ApprovalQueue callers must honor global Auto too.

        Several features call ``request_approval`` directly instead of going
        through SecurityHook. Global Auto is only global if this path bypasses
        the human queue as well.
        """
        request_added = False

        async def on_request_added(_request):
            nonlocal request_added
            request_added = True

        queue = ApprovalQueue(
            on_request_added=on_request_added,
            permission_store=store,
        )
        await store.register_tool("ComputeFeature", "run_script", PermissionLevel.ASK)
        store.set_global_auto_mode(True)

        approved, scope = await queue.request_approval(
            feature_name="ComputeFeature",
            tool_name="run_script",
            tool_args={"script_id": "s-auto"},
        )

        assert approved is True
        assert scope == "auto"
        assert request_added is False
        assert queue.pending_requests == []

        logs = await store.get_audit_log(limit=1)
        assert logs[0]["decision"] == "auto_mode_allowed"
        assert logs[0]["user_choice"] == "constitutional_honesty_unflagged"

    @pytest.mark.asyncio
    async def test_global_auto_mode_direct_request_approval_respects_deny(
        self, store
    ):
        queue = ApprovalQueue(permission_store=store)
        await store.register_tool(
            "WalletAgent",
            "delete_everything",
            PermissionLevel.DENY,
        )
        store.set_global_auto_mode(True)

        approved, scope = await queue.request_approval(
            feature_name="WalletAgent",
            tool_name="delete_everything",
            tool_args={},
        )

        assert approved is False
        assert scope == "denied"
        assert queue.pending_requests == []

        logs = await store.get_audit_log(limit=1)
        assert logs[0]["decision"] == "auto_denied"

    @pytest.mark.asyncio
    async def test_global_auto_mode_direct_request_approval_always_ask_prompts(
        self, store
    ):
        request_added = False

        async def on_request_added(_request):
            nonlocal request_added
            request_added = True

        auto_policy = MagicMock()
        auto_policy.evaluate = AsyncMock(
            side_effect=AssertionError("ALWAYS_ASK must not reach auto-approve")
        )
        queue = ApprovalQueue(
            on_request_added=on_request_added,
            permission_store=store,
            auto_approve_policy=auto_policy,
        )
        await store.register_tool(
            "codex_native",
            "commandExecution",
            PermissionLevel.ALWAYS_ASK,
        )
        store.set_global_auto_mode(True)

        approved, scope = await queue.request_approval(
            feature_name="codex_native",
            tool_name="commandExecution",
            tool_args={"command": "rm /etc/passwd"},
            timeout=0.01,
        )

        assert approved is False
        assert scope == "timeout"
        assert request_added is True
        assert queue.pending_requests == []
        auto_policy.evaluate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_scope_persists_via_direct_caller(self, queue_with_store, store):
        """A direct ``request_approval`` call (the path used by code_edit,
        compute, keys, reflection.*) with scope='session' must produce a
        SESSION-level permission so the next call doesn't re-prompt."""
        async def approve_session():
            await asyncio.sleep(0.05)
            pending = queue_with_store.pending_requests
            assert len(pending) == 1
            await queue_with_store.submit_decision(pending[0].id, True, "session")

        asyncio.create_task(approve_session())
        approved, scope = await queue_with_store.request_approval(
            feature_name="code_edit",
            tool_name="code_edit",
            tool_args={"path": "foo.py"},
        )

        assert approved is True
        assert scope == "session"
        # Permission persisted as ALLOW so the second call short-circuits.
        level = await store.get_permission("code_edit", "code_edit")
        from kestrel_sovereign.features.security.permissions import PermissionLevel
        assert level == PermissionLevel.ALLOW

    @pytest.mark.asyncio
    async def test_always_scope_persists_via_direct_caller(self, queue_with_store, store):
        async def approve_always():
            await asyncio.sleep(0.05)
            pending = queue_with_store.pending_requests
            await queue_with_store.submit_decision(pending[0].id, True, "always")

        asyncio.create_task(approve_always())
        approved, scope = await queue_with_store.request_approval(
            feature_name="compute",
            tool_name="execute_script",
            tool_args={"script": "x.py"},
        )

        assert approved is True
        assert scope == "always"
        from kestrel_sovereign.features.security.permissions import PermissionLevel
        # ALWAYS goes to the SQLite row (not just the in-memory session dict).
        # We can verify via raw SQL — bypassing the in-memory session cache.
        import aiosqlite
        async with aiosqlite.connect(store.db_path) as db:
            cursor = await db.execute(
                "SELECT level FROM security_permissions WHERE feature_name=? AND tool_name=?",
                ("compute", "execute_script"),
            )
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "allow"

    @pytest.mark.asyncio
    async def test_once_scope_does_not_persist(self, queue_with_store, store):
        """``scope='once'`` should grant the current call but NOT persist —
        the user explicitly asked for one-time approval."""
        async def approve_once():
            await asyncio.sleep(0.05)
            pending = queue_with_store.pending_requests
            await queue_with_store.submit_decision(pending[0].id, True, "once")

        asyncio.create_task(approve_once())
        approved, _ = await queue_with_store.request_approval(
            feature_name="keys",
            tool_name="set_key",
            tool_args={"provider": "openai"},
        )

        assert approved is True
        from kestrel_sovereign.features.security.permissions import PermissionLevel
        level = await store.get_permission("keys", "set_key")
        assert level == PermissionLevel.ASK  # Default — nothing persisted.

    @pytest.mark.asyncio
    async def test_every_decision_writes_audit_row(self, queue_with_store, store):
        """Audit log MUST capture every popup decision regardless of scope
        or which caller fired the request — the empty audit log on real
        agents was the most visible symptom of #785."""
        async def approve_session():
            await asyncio.sleep(0.05)
            pending = queue_with_store.pending_requests
            await queue_with_store.submit_decision(pending[0].id, True, "session")

        asyncio.create_task(approve_session())
        await queue_with_store.request_approval(
            feature_name="reflection",
            tool_name="self_model",
            tool_args={},
        )

        logs = await store.get_audit_log(limit=5)
        assert any(
            log["decision"] == "user_approved"
            and log["user_choice"] == "session"
            and log["feature"] == "reflection"
            and log["tool"] == "self_model"
            for log in logs
        )

    @pytest.mark.asyncio
    async def test_submit_decision_reports_persistence_failure(
        self, queue_with_store, store
    ):
        async def fail_set_permission(*args, **kwargs):
            raise RuntimeError("permission store write failed")

        store.set_permission = fail_set_permission

        request_task = asyncio.create_task(
            queue_with_store.request_approval(
                feature_name="reflection",
                tool_name="self_model",
                tool_args={},
            )
        )
        await asyncio.sleep(0.05)
        pending = queue_with_store.pending_requests

        result = await queue_with_store.submit_decision(
            pending[0].id, True, "session"
        )
        approved, scope = await request_task

        assert result.in_memory is True
        assert result.persisted is False
        assert "permission store write failed" in result.error
        assert approved is True
        assert scope == "session"

    @pytest.mark.asyncio
    async def test_user_denied_writes_audit_row(self, queue_with_store, store):
        async def deny():
            await asyncio.sleep(0.05)
            pending = queue_with_store.pending_requests
            await queue_with_store.submit_decision(pending[0].id, False, "denied")

        asyncio.create_task(deny())
        approved, _ = await queue_with_store.request_approval(
            feature_name="code_edit",
            tool_name="code_commit",
            tool_args={},
        )

        assert approved is False
        logs = await store.get_audit_log(limit=5)
        assert any(
            log["decision"] == "user_denied" and log["feature"] == "code_edit"
            for log in logs
        )

    @pytest.mark.asyncio
    async def test_timeout_writes_audit_row(self, queue_with_store, store):
        approved, scope = await queue_with_store.request_approval(
            feature_name="code_edit",
            tool_name="code_test",
            tool_args={},
            timeout=0.05,
        )

        assert approved is False
        assert scope == "timeout"
        logs = await store.get_audit_log(limit=5)
        assert any(log["decision"] == "timeout" for log in logs)

    @pytest.mark.asyncio
    async def test_queue_without_store_remains_compatible(self, queue_without_store):
        """ApprovalQueue must remain usable without a permission_store
        (legacy callers / standalone tests).  The queue silently skips
        persistence when no store is configured."""
        async def approve():
            await asyncio.sleep(0.05)
            pending = queue_without_store.pending_requests
            await queue_without_store.submit_decision(pending[0].id, True, "session")

        asyncio.create_task(approve())
        approved, scope = await queue_without_store.request_approval(
            feature_name="x", tool_name="y", tool_args={},
        )
        assert approved is True
        assert scope == "session"  # Returned to caller; just nothing persisted.


class TestSecurityFeature:
    """Tests for SecurityFeature command contracts."""

    @pytest.mark.asyncio
    async def test_register_all_tools_seeds_codex_native_as_always_ask(self, tmp_path):
        agent = MagicMock()
        agent.features = {}
        feature = SecurityFeature(agent)
        feature.permission_store = PermissionStore(str(tmp_path / "security.db"))
        await feature.permission_store.initialize()

        await feature._register_all_tools()

        assert (
            await feature.permission_store.get_permission(
                "codex_native", "commandExecution"
            )
        ) == PermissionLevel.ALWAYS_ASK
        assert (
            await feature.permission_store.get_permission("codex_native", "fileChange")
        ) == PermissionLevel.ALWAYS_ASK

    @pytest.mark.asyncio
    async def test_pending_approvals_uses_wall_clock_age(self):
        from kestrel_sdk.tools.result import ToolResultStatus
        agent = MagicMock()
        feature = SecurityFeature(agent)
        feature.approval_queue = ApprovalQueue()

        request = ApprovalRequest(
            id="req-12345678",
            feature_name="WalletAgent",
            tool_name="send_tokens",
            tool_args={"amount": 5},
            created_at=datetime.now(timezone.utc) - timedelta(seconds=12),
        )
        feature.approval_queue._pending[request.id] = request

        result = await feature.pending_approvals()

        assert result.status is ToolResultStatus.OK
        assert "Pending Approvals (1):" in result.confirmation
        assert "[req-1234] WalletAgent.send_tokens" in result.confirmation
        assert (
            "(12s ago)" in result.confirmation
            or "(11s ago)" in result.confirmation
            or "(13s ago)" in result.confirmation
        )
        # Structured data also surfaces the request
        assert result.data["count"] == 1
        assert result.data["requests"][0]["id"] == "req-12345678"
        assert result.data["requests"][0]["feature_name"] == "WalletAgent"

    @pytest.mark.asyncio
    async def test_approve_once_returns_ok(self):
        """scope='once' has no durable permission row — OK is honest."""
        from kestrel_sdk.tools.result import ToolResultStatus
        agent = MagicMock()
        feature = SecurityFeature(agent)
        feature.approval_queue = MagicMock()
        feature.approval_queue.submit_decision = AsyncMock(
            return_value=DecisionResult(in_memory=True, persisted=True)
        )
        feature.approval_queue.pending_requests = [
            ApprovalRequest(
                id="req-once-12345678",
                feature_name="WalletAgent",
                tool_name="get_balance",
                tool_args={},
                created_at=datetime.now(timezone.utc),
            )
        ]

        result = await feature.approve_request("req-once-1", scope="once")
        assert result.status is ToolResultStatus.OK
        assert result.data["scope"] == "once"
        assert result.data["scope_persisted"] is True

    @pytest.mark.asyncio
    async def test_approve_with_durable_scope_returns_ok_when_persisted(self):
        """scope='session'/'always' returns OK when the queue confirms
        durable persistence succeeded."""
        from kestrel_sdk.tools.result import ToolResultStatus
        agent = MagicMock()
        feature = SecurityFeature(agent)
        feature.approval_queue = MagicMock()
        feature.approval_queue.submit_decision = AsyncMock(
            return_value=DecisionResult(in_memory=True, persisted=True)
        )
        feature.approval_queue.pending_requests = [
            ApprovalRequest(
                id="req-session-12345678",
                feature_name="WalletAgent",
                tool_name="send_tokens",
                tool_args={},
                created_at=datetime.now(timezone.utc),
            )
        ]

        for scope in ("session", "always"):
            result = await feature.approve_request("req-session-1", scope=scope)
            assert result.status is ToolResultStatus.OK, scope
            assert result.data["scope"] == scope
            assert result.data["scope_persisted"] is True

    @pytest.mark.asyncio
    async def test_approve_with_durable_scope_is_partial_when_persistence_fails(self):
        """If the queue accepts the decision but reports a store failure,
        the tool reports PARTIAL instead of claiming the scope stuck."""
        from kestrel_sdk.tools.result import ToolResultStatus
        agent = MagicMock()
        feature = SecurityFeature(agent)
        feature.approval_queue = MagicMock()
        feature.approval_queue.submit_decision = AsyncMock(
            return_value=DecisionResult(
                in_memory=True,
                persisted=False,
                error="database is locked",
            )
        )
        feature.approval_queue.pending_requests = [
            ApprovalRequest(
                id="req-session-12345678",
                feature_name="WalletAgent",
                tool_name="send_tokens",
                tool_args={},
                created_at=datetime.now(timezone.utc),
            )
        ]

        result = await feature.approve_request("req-session-1", scope="session")
        assert result.status is ToolResultStatus.PARTIAL
        assert "database is locked" in result.error
        assert result.data["scope"] == "session"
        assert result.data["scope_persisted"] is False

    @pytest.mark.asyncio
    async def test_approve_when_request_withdrawn_is_error(self):
        """When the queue withdrew the request between lookup and
        submit, surface ERROR with a specific framing."""
        from kestrel_sdk.tools.result import ToolResultStatus
        agent = MagicMock()
        feature = SecurityFeature(agent)
        feature.approval_queue = MagicMock()
        # submit reports in_memory=False — withdrawal race
        feature.approval_queue.submit_decision = AsyncMock(
            return_value=DecisionResult(
                in_memory=False,
                persisted=False,
                error="request not found or expired",
            )
        )
        feature.approval_queue.pending_requests = [
            ApprovalRequest(
                id="req-race-12345678",
                feature_name="WalletAgent",
                tool_name="send_tokens",
                tool_args={},
                created_at=datetime.now(timezone.utc),
            )
        ]

        result = await feature.approve_request("req-race-1", scope="once")
        assert result.status is ToolResultStatus.ERROR
        assert "no longer pending" in result.error.lower()

    @pytest.mark.asyncio
    async def test_security_audit_filters_sensitive_fields_from_data(self):
        """Round 1 codex finding: security_audit's data.entries goes
        back into the LLM context. Audit rows carry ``args_summary``
        (sometimes unmasked direct-ApprovalQueue callers) with paths,
        tokens, request payloads. The pre-fix str output deliberately
        omitted args; the new ToolResult must do the same in
        ``data.entries``."""
        from kestrel_sdk.tools.result import ToolResultStatus
        agent = MagicMock()
        feature = SecurityFeature(agent)
        # Mock permission_store.get_audit_log to return rows with sensitive fields
        feature.permission_store = MagicMock()
        feature.permission_store.get_audit_log = AsyncMock(return_value=[
            {
                "feature": "WalletAgent",
                "tool": "send_tokens",
                "decision": "user_approved",
                "user_choice": "always",
                "timestamp": "2026-05-07T22:00:00",
                # Fields that MUST NOT be exposed to the LLM
                "args_summary": '{"recipient": "0xSECRET", "amount": 9999, "token": "FIL"}',
                "raw_args": {"private_key": "0xDEADBEEF"},
            },
        ])

        result = await feature.security_audit(limit=10)

        assert result.status is ToolResultStatus.OK
        # data.entries must NOT include args_summary or raw_args
        for entry in result.data["entries"]:
            assert "args_summary" not in entry, (
                "args_summary leaked into LLM context — "
                "would expose tool arguments to the model"
            )
            assert "raw_args" not in entry
            # Safe fields are present
            assert "feature" in entry
            assert "tool" in entry
            assert "decision" in entry
        # Confirmation text already omitted args (pre-fix) — preserve that
        assert "0xSECRET" not in result.confirmation
        assert "0xDEADBEEF" not in result.confirmation

    @pytest.mark.asyncio
    async def test_pending_approvals_handles_legacy_naive_timestamps(self):
        from kestrel_sdk.tools.result import ToolResultStatus
        agent = MagicMock()
        feature = SecurityFeature(agent)
        feature.approval_queue = ApprovalQueue()

        request = ApprovalRequest(
            id="req-legacy",
            feature_name="WalletAgent",
            tool_name="get_balance",
            tool_args={},
            created_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).replace(tzinfo=None),
        )
        feature.approval_queue._pending[request.id] = request

        result = await feature.pending_approvals()

        assert result.status is ToolResultStatus.OK
        assert "WalletAgent.get_balance" in result.confirmation
        assert "ago)" in result.confirmation


# === Run tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
