"""
Integration tests for orchestrator permission enforcement.

Tests verify that the _execute_tool_with_hooks helper method correctly
enforces security hooks (ALLOW, DENY, ASK) before tool execution and
logs audit entries for every decision.

These tests use real PermissionStore, ApprovalQueue, SecurityHook, and
HooksManager instances -- only the tool execution itself is mocked.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
from kestrel_sovereign.hooks import HooksManager


class MockAgent:
    """Minimal mock agent with hooks_manager for testing _execute_tool_with_hooks."""

    def __init__(self, hooks_manager: HooksManager):
        self.hooks_manager = hooks_manager


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
async def permission_store(temp_db):
    """Create and initialize a real PermissionStore."""
    store = PermissionStore(temp_db)
    await store.initialize()
    return store


@pytest.fixture
def approval_queue():
    """Create a real ApprovalQueue."""
    return ApprovalQueue()


@pytest.fixture
def security_hook(permission_store, approval_queue):
    """Create a real SecurityHook."""
    return SecurityHook(permission_store, approval_queue)


@pytest.fixture
def hooks_manager(security_hook):
    """Create a HooksManager with the security hook registered."""
    manager = HooksManager()
    manager.register(security_hook)
    return manager


@pytest.fixture
def mock_agent(hooks_manager):
    """Create a mock agent with the hooks manager."""
    return MockAgent(hooks_manager)


async def _execute_tool_with_hooks(
    hooks_manager: HooksManager,
    tool_name: str,
    feature_name: str,
    args: dict,
    session_id: str,
    execute_fn,
) -> dict:
    """
    Standalone version of KestrelAgent._execute_tool_with_hooks for testing.

    Mirrors the implementation in kestrel_agent.py exactly, so that tests
    validate the same logic without needing a full KestrelAgent instance.
    """
    import time

    from kestrel_sovereign.hooks import HookEvent, HookInput, PermissionDecision

    hook_input = HookInput(
        session_id=session_id,
        hook_event_name=HookEvent.PRE_TOOL_USE.value,
        tool_name=tool_name,
        tool_input=args,
        feature_name=feature_name,
    )

    hook_output = await hooks_manager.execute_hooks(
        HookEvent.PRE_TOOL_USE,
        hook_input,
    )

    if hook_output.permission_decision == PermissionDecision.DENY:
        reason = hook_output.permission_reason or "Blocked by security policy"
        return {"success": False, "error": f"Permission denied: {reason}"}

    if hook_output.updated_input:
        args = hook_output.updated_input

    exec_start = time.time()
    result = await execute_fn()
    exec_duration_ms = int((time.time() - exec_start) * 1000)

    post_hook_input = HookInput(
        session_id=session_id,
        hook_event_name=HookEvent.POST_TOOL_USE.value,
        tool_name=tool_name,
        tool_input=args,
        feature_name=feature_name,
        tool_response=result if isinstance(result, dict) else {"result": str(result)},
        execution_time_ms=exec_duration_ms,
    )
    await hooks_manager.execute_hooks_parallel(
        HookEvent.POST_TOOL_USE,
        post_hook_input,
    )

    return result


class TestDenyBlocksTool:
    """Test that DENY permission prevents tool execution."""

    @pytest.mark.asyncio
    async def test_deny_blocks_tool(self, hooks_manager, permission_store):
        """When a tool is set to DENY, execution is blocked and the tool is never called."""
        await permission_store.register_tool("TestFeature", "dangerous_tool")
        await permission_store.set_permission(
            "TestFeature", "dangerous_tool", PermissionLevel.DENY
        )

        tool_executed = False

        async def mock_execute():
            nonlocal tool_executed
            tool_executed = True
            return {"success": True, "data": "should not appear"}

        result = await _execute_tool_with_hooks(
            hooks_manager=hooks_manager,
            tool_name="dangerous_tool",
            feature_name="TestFeature",
            args={"param": "value"},
            session_id="test-session-deny",
            execute_fn=mock_execute,
        )

        assert tool_executed is False, "Tool should NOT have been executed"
        assert result["success"] is False
        assert "Permission denied" in result["error"]


class TestAllowPassesThrough:
    """Test that ALLOW permission lets tool execute normally."""

    @pytest.mark.asyncio
    async def test_allow_passes_through(self, hooks_manager, permission_store):
        """When a tool is set to ALLOW, it executes and returns its result."""
        await permission_store.register_tool("TestFeature", "safe_tool")
        await permission_store.set_permission(
            "TestFeature", "safe_tool", PermissionLevel.ALLOW
        )

        async def mock_execute():
            return {"success": True, "data": "hello"}

        result = await _execute_tool_with_hooks(
            hooks_manager=hooks_manager,
            tool_name="safe_tool",
            feature_name="TestFeature",
            args={},
            session_id="test-session-allow",
            execute_fn=mock_execute,
        )

        assert result["success"] is True
        assert result["data"] == "hello"


class TestAskBlocksUntilApproved:
    """Test that ASK permission blocks until user approval is submitted."""

    @pytest.mark.asyncio
    async def test_ask_blocks_until_approved(
        self, hooks_manager, permission_store, approval_queue
    ):
        """When a tool is ASK, execution blocks until the user approves via the queue."""
        await permission_store.register_tool("TestFeature", "ask_tool")
        # ASK is the default, but set it explicitly for clarity
        await permission_store.set_permission(
            "TestFeature", "ask_tool", PermissionLevel.ASK
        )

        tool_executed = False

        async def mock_execute():
            nonlocal tool_executed
            tool_executed = True
            return {"success": True, "data": "approved result"}

        # Start the tool call in a task (it will block on approval)
        task = asyncio.create_task(
            _execute_tool_with_hooks(
                hooks_manager=hooks_manager,
                tool_name="ask_tool",
                feature_name="TestFeature",
                args={"key": "val"},
                session_id="test-session-ask",
                execute_fn=mock_execute,
            )
        )

        # Wait briefly for the approval request to appear
        await asyncio.sleep(0.1)
        assert tool_executed is False, "Tool should still be waiting for approval"
        assert approval_queue.pending_count == 1

        # Submit approval
        pending = approval_queue.pending_requests[0]
        approval_queue.submit_decision(pending.id, approved=True, scope="once")

        # Now the task should complete
        result = await asyncio.wait_for(task, timeout=5.0)

        assert tool_executed is True
        assert result["success"] is True
        assert result["data"] == "approved result"


class TestAskDenyBlocksTool:
    """Test that ASK followed by user denial blocks tool execution."""

    @pytest.mark.asyncio
    async def test_ask_deny_blocks_tool(
        self, hooks_manager, permission_store, approval_queue
    ):
        """When the user denies an ASK request, the tool is not executed."""
        await permission_store.register_tool("TestFeature", "ask_deny_tool")
        await permission_store.set_permission(
            "TestFeature", "ask_deny_tool", PermissionLevel.ASK
        )

        tool_executed = False

        async def mock_execute():
            nonlocal tool_executed
            tool_executed = True
            return {"success": True}

        task = asyncio.create_task(
            _execute_tool_with_hooks(
                hooks_manager=hooks_manager,
                tool_name="ask_deny_tool",
                feature_name="TestFeature",
                args={},
                session_id="test-session-ask-deny",
                execute_fn=mock_execute,
            )
        )

        await asyncio.sleep(0.1)
        assert approval_queue.pending_count == 1

        pending = approval_queue.pending_requests[0]
        approval_queue.submit_decision(pending.id, approved=False, scope="once")

        result = await asyncio.wait_for(task, timeout=5.0)

        assert tool_executed is False, "Tool should NOT have executed after denial"
        assert result["success"] is False
        assert "Permission denied" in result["error"] or "denied" in result["error"].lower()


class TestSessionScopePersists:
    """Test that session-scoped approval persists across calls."""

    @pytest.mark.asyncio
    async def test_session_scope_persists(
        self, hooks_manager, permission_store, approval_queue
    ):
        """After approving with session scope, subsequent calls auto-allow without queuing."""
        await permission_store.register_tool("TestFeature", "session_tool")
        await permission_store.set_permission(
            "TestFeature", "session_tool", PermissionLevel.ASK
        )

        call_count = 0

        async def mock_execute():
            nonlocal call_count
            call_count += 1
            return {"success": True, "call": call_count}

        # First call: blocks on approval
        task = asyncio.create_task(
            _execute_tool_with_hooks(
                hooks_manager=hooks_manager,
                tool_name="session_tool",
                feature_name="TestFeature",
                args={},
                session_id="test-session-scope",
                execute_fn=mock_execute,
            )
        )

        await asyncio.sleep(0.1)
        assert approval_queue.pending_count == 1

        pending = approval_queue.pending_requests[0]
        approval_queue.submit_decision(pending.id, approved=True, scope="session")

        result1 = await asyncio.wait_for(task, timeout=5.0)
        assert result1["success"] is True
        assert result1["call"] == 1

        # Second call: should auto-allow (no approval queue interaction)
        result2 = await _execute_tool_with_hooks(
            hooks_manager=hooks_manager,
            tool_name="session_tool",
            feature_name="TestFeature",
            args={},
            session_id="test-session-scope",
            execute_fn=mock_execute,
        )

        assert result2["success"] is True
        assert result2["call"] == 2
        # No new pending requests should have been created
        assert approval_queue.pending_count == 0


class TestAuditLogPopulated:
    """Test that audit log entries are created for all decision types."""

    @pytest.mark.asyncio
    async def test_audit_log_populated(
        self, hooks_manager, permission_store, approval_queue
    ):
        """Verify audit log entries for ALLOW, DENY, and ASK-approved decisions."""
        # --- ALLOW decision ---
        await permission_store.register_tool("AuditFeature", "allowed_tool")
        await permission_store.set_permission(
            "AuditFeature", "allowed_tool", PermissionLevel.ALLOW
        )

        async def mock_execute():
            return {"success": True}

        await _execute_tool_with_hooks(
            hooks_manager=hooks_manager,
            tool_name="allowed_tool",
            feature_name="AuditFeature",
            args={"x": 1},
            session_id="test-audit",
            execute_fn=mock_execute,
        )

        # --- DENY decision ---
        await permission_store.register_tool("AuditFeature", "denied_tool")
        await permission_store.set_permission(
            "AuditFeature", "denied_tool", PermissionLevel.DENY
        )

        await _execute_tool_with_hooks(
            hooks_manager=hooks_manager,
            tool_name="denied_tool",
            feature_name="AuditFeature",
            args={},
            session_id="test-audit",
            execute_fn=mock_execute,
        )

        # --- ASK-approved decision ---
        await permission_store.register_tool("AuditFeature", "ask_audit_tool")
        await permission_store.set_permission(
            "AuditFeature", "ask_audit_tool", PermissionLevel.ASK
        )

        task = asyncio.create_task(
            _execute_tool_with_hooks(
                hooks_manager=hooks_manager,
                tool_name="ask_audit_tool",
                feature_name="AuditFeature",
                args={},
                session_id="test-audit",
                execute_fn=mock_execute,
            )
        )

        await asyncio.sleep(0.1)
        pending = approval_queue.pending_requests[0]
        approval_queue.submit_decision(pending.id, approved=True, scope="once")
        await asyncio.wait_for(task, timeout=5.0)

        # --- Verify audit log ---
        audit_log = await permission_store.get_audit_log(limit=20)

        # Find entries by tool name
        allowed_entries = [e for e in audit_log if e["tool"] == "allowed_tool"]
        denied_entries = [e for e in audit_log if e["tool"] == "denied_tool"]
        ask_entries = [e for e in audit_log if e["tool"] == "ask_audit_tool"]

        assert len(allowed_entries) >= 1, f"Expected ALLOW audit entry, got: {audit_log}"
        assert allowed_entries[0]["decision"] == "auto_allowed"

        assert len(denied_entries) >= 1, f"Expected DENY audit entry, got: {audit_log}"
        assert denied_entries[0]["decision"] == "auto_denied"

        assert len(ask_entries) >= 1, f"Expected ASK-approved audit entry, got: {audit_log}"
        assert ask_entries[0]["decision"] == "user_approved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
