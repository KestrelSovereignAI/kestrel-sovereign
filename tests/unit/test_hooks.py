"""
Unit Tests for Kestrel Hooks System.

Tests the core hooks infrastructure:
- HookEvent enum
- HookInput/HookOutput dataclasses
- Hook base class
- HooksManager registration and execution
"""

import asyncio
import pytest
from datetime import datetime

from kestrel_sovereign.hooks import (
    Hook,
    HookEvent,
    HookInput,
    HookOutput,
    HooksManager,
    PermissionDecision,
)


# === Test Hook Implementations ===

class AllowAllHook(Hook):
    """Test hook that always allows."""

    def __init__(self, name: str = "allow_all", priority: int = 100):
        super().__init__(name=name, events=[HookEvent.PRE_TOOL_USE], priority=priority)
        self.call_count = 0

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        return HookOutput.allow("Test allow")


class DenyAllHook(Hook):
    """Test hook that always denies."""

    def __init__(self, name: str = "deny_all", priority: int = 100):
        super().__init__(name=name, events=[HookEvent.PRE_TOOL_USE], priority=priority)
        self.call_count = 0

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        return HookOutput.deny("Test deny")


class RegexMatcherHook(Hook):
    """Test hook with regex matching."""

    def __init__(self, matcher: str, priority: int = 100):
        super().__init__(
            name="regex_matcher",
            events=[HookEvent.PRE_TOOL_USE],
            matcher=matcher,
            priority=priority,
        )
        self.matched_tools = []

    async def execute(self, input: HookInput) -> HookOutput:
        self.matched_tools.append(input.tool_name)
        return HookOutput.allow()


class TimeoutHook(Hook):
    """Test hook that times out."""

    def __init__(self, delay: float = 10.0):
        super().__init__(
            name="timeout_hook",
            events=[HookEvent.PRE_TOOL_USE],
            timeout=0.1,  # Very short timeout
        )
        self.delay = delay

    async def execute(self, input: HookInput) -> HookOutput:
        await asyncio.sleep(self.delay)
        return HookOutput.allow()


class FailingHook(Hook):
    """Test hook that raises an exception."""

    def __init__(self):
        super().__init__(name="failing_hook", events=[HookEvent.PRE_TOOL_USE])

    async def execute(self, input: HookInput) -> HookOutput:
        raise ValueError("Test error")


# === HookInput Tests ===

class TestHookInput:
    """Tests for HookInput dataclass."""

    def test_basic_creation(self):
        input = HookInput(
            session_id="test-session",
            hook_event_name="PreToolUse",
        )
        assert input.session_id == "test-session"
        assert input.hook_event_name == "PreToolUse"

    def test_tool_context(self):
        input = HookInput(
            session_id="test-session",
            hook_event_name="PreToolUse",
            tool_name="web_search",
            tool_input={"query": "test"},
            feature_name="SearchFeature",
        )
        assert input.tool_name == "web_search"
        assert input.tool_input == {"query": "test"}
        assert input.feature_name == "SearchFeature"

    def test_to_dict(self):
        input = HookInput(
            session_id="test-session",
            hook_event_name="PreToolUse",
            tool_name="web_search",
        )
        d = input.to_dict()
        assert d["session_id"] == "test-session"
        assert d["hook_event_name"] == "PreToolUse"
        assert d["tool_name"] == "web_search"


# === HookOutput Tests ===

class TestHookOutput:
    """Tests for HookOutput dataclass."""

    def test_allow(self):
        output = HookOutput.allow("Test reason")
        assert output.continue_execution is True
        assert output.permission_decision == PermissionDecision.ALLOW
        assert output.permission_reason == "Test reason"

    def test_deny(self):
        output = HookOutput.deny("Blocked")
        assert output.continue_execution is False
        assert output.permission_decision == PermissionDecision.DENY
        assert output.permission_reason == "Blocked"
        assert output.stop_reason == "Blocked"

    def test_ask(self):
        output = HookOutput.ask("approval-123", "Need approval")
        assert output.continue_execution is False
        assert output.permission_decision == PermissionDecision.ASK
        assert output.approval_id == "approval-123"

    def test_modify(self):
        new_args = {"modified": True}
        output = HookOutput.modify(new_args, "Modified args")
        assert output.continue_execution is True
        assert output.permission_decision == PermissionDecision.ALLOW
        assert output.updated_input == new_args

    def test_to_dict(self):
        output = HookOutput.allow("Test")
        d = output.to_dict()
        assert d["continue_execution"] is True
        assert d["permission_decision"] == "allow"


# === Hook Base Class Tests ===

class TestHookBaseClass:
    """Tests for Hook base class."""

    def test_regex_matcher_valid(self):
        hook = AllowAllHook()
        hook.matcher = "web_.*"
        hook._compiled_matcher = None
        # Re-initialize to compile matcher
        hook2 = RegexMatcherHook(matcher="web_.*")
        assert hook2.matches("web_search")
        assert hook2.matches("web_fetch")
        assert not hook2.matches("file_read")

    def test_regex_matcher_none_matches_all(self):
        hook = AllowAllHook()
        assert hook.matches("anything")
        assert hook.matches("web_search")

    def test_invalid_regex_raises(self):
        with pytest.raises(ValueError, match="Invalid matcher regex"):
            RegexMatcherHook(matcher="[invalid")


# === HooksManager Tests ===

class TestHooksManager:
    """Tests for HooksManager."""

    def test_register(self):
        manager = HooksManager()
        hook = AllowAllHook()
        manager.register(hook)

        hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        assert len(hooks) == 1
        assert hooks[0] == hook

    def test_unregister(self):
        manager = HooksManager()
        hook = AllowAllHook()
        manager.register(hook)
        manager.unregister(hook)

        hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        assert len(hooks) == 0

    def test_unregister_by_name(self):
        manager = HooksManager()
        hook = AllowAllHook(name="test_hook")
        manager.register(hook)

        result = manager.unregister_by_name("test_hook")
        assert result is True
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0

        # Unregister non-existent
        result = manager.unregister_by_name("nonexistent")
        assert result is False

    def test_priority_ordering(self):
        manager = HooksManager()
        hook1 = AllowAllHook(name="low_priority", priority=200)
        hook2 = AllowAllHook(name="high_priority", priority=50)
        hook3 = AllowAllHook(name="mid_priority", priority=100)

        manager.register(hook1)
        manager.register(hook2)
        manager.register(hook3)

        hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        assert hooks[0].name == "high_priority"
        assert hooks[1].name == "mid_priority"
        assert hooks[2].name == "low_priority"

    def test_enable_disable(self):
        manager = HooksManager()
        hook = AllowAllHook(name="test_hook")
        manager.register(hook)

        # Disable
        result = manager.set_hook_enabled("test_hook", False)
        assert result is True
        assert len(manager.get_enabled_hooks(HookEvent.PRE_TOOL_USE)) == 0

        # Re-enable
        result = manager.set_hook_enabled("test_hook", True)
        assert result is True
        assert len(manager.get_enabled_hooks(HookEvent.PRE_TOOL_USE)) == 1

    @pytest.mark.asyncio
    async def test_execute_hooks_allow(self):
        manager = HooksManager()
        hook = AllowAllHook()
        manager.register(hook)

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="test_tool",
        )

        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input)
        assert output.continue_execution is True
        assert hook.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_hooks_deny_stops_chain(self):
        manager = HooksManager()
        deny_hook = DenyAllHook(name="deny", priority=50)  # Runs first
        allow_hook = AllowAllHook(name="allow", priority=100)  # Runs second

        manager.register(deny_hook)
        manager.register(allow_hook)

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="test_tool",
        )

        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input)
        assert output.continue_execution is False
        assert output.permission_decision == PermissionDecision.DENY
        assert deny_hook.call_count == 1
        assert allow_hook.call_count == 0  # Chain was stopped

    @pytest.mark.asyncio
    async def test_execute_hooks_no_hooks_allows(self):
        manager = HooksManager()

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="test_tool",
        )

        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input)
        assert output.continue_execution is True
        assert output.permission_decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_execute_hooks_regex_matching(self):
        manager = HooksManager()
        hook = RegexMatcherHook(matcher="web_.*")
        manager.register(hook)

        # Matching tool
        input1 = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="web_search",
        )
        await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input1)

        # Non-matching tool
        input2 = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="file_read",
        )
        await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input2)

        assert hook.matched_tools == ["web_search"]  # Only matched first

    @pytest.mark.asyncio
    async def test_execute_hooks_timeout_skips(self):
        manager = HooksManager()
        timeout_hook = TimeoutHook(delay=10.0)
        allow_hook = AllowAllHook(priority=200)

        manager.register(timeout_hook)
        manager.register(allow_hook)

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="test_tool",
        )

        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input)
        # Should still allow (timeout hook skipped)
        assert output.continue_execution is True
        assert allow_hook.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_hooks_exception_skips(self):
        manager = HooksManager()
        failing_hook = FailingHook()
        allow_hook = AllowAllHook(priority=200)

        manager.register(failing_hook)
        manager.register(allow_hook)

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="test_tool",
        )

        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input)
        # Should still allow (failing hook skipped)
        assert output.continue_execution is True
        assert allow_hook.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_hooks_parallel(self):
        manager = HooksManager()
        hook1 = AllowAllHook(name="hook1")
        hook2 = AllowAllHook(name="hook2")
        hook3 = AllowAllHook(name="hook3")

        manager.register(hook1)
        manager.register(hook2)
        manager.register(hook3)

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="test_tool",
        )

        outputs = await manager.execute_hooks_parallel(HookEvent.PRE_TOOL_USE, input)
        assert len(outputs) == 3
        assert hook1.call_count == 1
        assert hook2.call_count == 1
        assert hook3.call_count == 1


# === PRE_SUBAGENT_CALL Tests ===


class SubagentAllowHook(Hook):
    """Test hook for PRE_SUBAGENT_CALL that always allows."""

    def __init__(self, name: str = "subagent_allow", priority: int = 100):
        super().__init__(name=name, events=[HookEvent.PRE_SUBAGENT_CALL], priority=priority)
        self.call_count = 0
        self.last_input = None

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        self.last_input = input
        return HookOutput.allow("Subagent allowed")


class SubagentDenyHook(Hook):
    """Test hook for PRE_SUBAGENT_CALL that always denies."""

    def __init__(self, name: str = "subagent_deny", priority: int = 100):
        super().__init__(name=name, events=[HookEvent.PRE_SUBAGENT_CALL], priority=priority)
        self.call_count = 0
        self.last_input = None

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        self.last_input = input
        return HookOutput.deny("Subagent blocked by test policy")


class DualEventHook(Hook):
    """Test hook registered for both PRE_TOOL_USE and PRE_SUBAGENT_CALL (like SecurityHook)."""

    def __init__(self, name: str = "dual_event", priority: int = 10):
        super().__init__(
            name=name,
            events=[HookEvent.PRE_TOOL_USE, HookEvent.PRE_SUBAGENT_CALL],
            priority=priority,
        )
        self.calls = []  # List of (event_name, tool_name) tuples

    async def execute(self, input: HookInput) -> HookOutput:
        self.calls.append((input.hook_event_name, input.tool_name))
        return HookOutput.allow("Allowed by dual hook")


class TestPreSubagentCall:
    """Tests for PRE_SUBAGENT_CALL hook event."""

    @pytest.mark.asyncio
    async def test_subagent_hook_receives_correct_input(self):
        """Verify PRE_SUBAGENT_CALL hook receives tool_name, tool_input, feature_name."""
        manager = HooksManager()
        hook = SubagentAllowHook()
        manager.register(hook)

        hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
            tool_name="mcp_search",
            tool_input={"task": "search for files", "context": "user request"},
            feature_name="MCPFeature",
        )

        output = await manager.execute_hooks(HookEvent.PRE_SUBAGENT_CALL, hook_input)

        assert output.continue_execution is True
        assert hook.call_count == 1
        assert hook.last_input.tool_name == "mcp_search"
        assert hook.last_input.tool_input == {"task": "search for files", "context": "user request"}
        assert hook.last_input.feature_name == "MCPFeature"
        assert hook.last_input.hook_event_name == "PreSubagentCall"

    @pytest.mark.asyncio
    async def test_subagent_deny_blocks_execution(self):
        """Verify DENY on PRE_SUBAGENT_CALL blocks and returns correct output."""
        manager = HooksManager()
        hook = SubagentDenyHook()
        manager.register(hook)

        hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
            tool_name="wallet_transfer",
            tool_input={"task": "send 100 USDC"},
            feature_name="WalletFeature",
        )

        output = await manager.execute_hooks(HookEvent.PRE_SUBAGENT_CALL, hook_input)

        assert output.continue_execution is False
        assert output.permission_decision == PermissionDecision.DENY
        assert "blocked by test policy" in output.permission_reason
        assert hook.call_count == 1

    @pytest.mark.asyncio
    async def test_dual_event_hook_receives_both_events(self):
        """Verify a hook registered for both events receives both independently."""
        manager = HooksManager()
        hook = DualEventHook()
        manager.register(hook)

        # Fire PRE_SUBAGENT_CALL
        subagent_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
            tool_name="compute_provision",
            tool_input={"task": "provision GPU"},
            feature_name="ComputeFeature",
        )
        await manager.execute_hooks(HookEvent.PRE_SUBAGENT_CALL, subagent_input)

        # Fire PRE_TOOL_USE
        tool_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="compute_provision",
            tool_input={"task": "provision GPU"},
            feature_name="ComputeFeature",
        )
        await manager.execute_hooks(HookEvent.PRE_TOOL_USE, tool_input)

        assert len(hook.calls) == 2
        assert hook.calls[0] == ("PreSubagentCall", "compute_provision")
        assert hook.calls[1] == ("PreToolUse", "compute_provision")

    @pytest.mark.asyncio
    async def test_subagent_deny_stops_chain(self):
        """Verify DENY on PRE_SUBAGENT_CALL stops the hook chain."""
        manager = HooksManager()
        deny_hook = SubagentDenyHook(name="deny_first", priority=10)
        allow_hook = SubagentAllowHook(name="allow_second", priority=100)

        manager.register(deny_hook)
        manager.register(allow_hook)

        hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
            tool_name="dangerous_tool",
            tool_input={"task": "do something risky"},
            feature_name="RiskyFeature",
        )

        output = await manager.execute_hooks(HookEvent.PRE_SUBAGENT_CALL, hook_input)

        assert output.permission_decision == PermissionDecision.DENY
        assert deny_hook.call_count == 1
        assert allow_hook.call_count == 0  # Chain was stopped

    @pytest.mark.asyncio
    async def test_no_subagent_hooks_allows(self):
        """Verify no registered PRE_SUBAGENT_CALL hooks results in ALLOW."""
        manager = HooksManager()

        hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
            tool_name="any_tool",
            tool_input={"task": "anything"},
            feature_name="AnyFeature",
        )

        output = await manager.execute_hooks(HookEvent.PRE_SUBAGENT_CALL, hook_input)

        assert output.continue_execution is True
        assert output.permission_decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_subagent_hook_independent_of_tool_hooks(self):
        """Verify PRE_SUBAGENT_CALL hooks don't fire for PRE_TOOL_USE events and vice versa."""
        manager = HooksManager()
        subagent_hook = SubagentAllowHook(name="subagent_only")
        tool_hook = AllowAllHook(name="tool_only")

        manager.register(subagent_hook)
        manager.register(tool_hook)

        # Fire PRE_TOOL_USE — only tool_hook should fire
        tool_input = HookInput(
            session_id="test",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="some_tool",
        )
        await manager.execute_hooks(HookEvent.PRE_TOOL_USE, tool_input)

        assert tool_hook.call_count == 1
        assert subagent_hook.call_count == 0

        # Fire PRE_SUBAGENT_CALL — only subagent_hook should fire
        subagent_input = HookInput(
            session_id="test",
            hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
            tool_name="some_feature",
        )
        await manager.execute_hooks(HookEvent.PRE_SUBAGENT_CALL, subagent_input)

        assert tool_hook.call_count == 1  # Unchanged
        assert subagent_hook.call_count == 1


# === Run tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
