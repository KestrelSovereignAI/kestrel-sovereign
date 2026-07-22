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

from kestrel_sovereign.hooks import HooksManager
from kestrel_sdk.hooks.base import (
    Hook,
    HookEvent,
    HookInput,
    HookOutput,
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


class SessionStartHook(Hook):
    """Test hook that listens on SESSION_START."""

    def __init__(self, name: str = "session_start_hook", priority: int = 100):
        super().__init__(name=name, events=[HookEvent.SESSION_START], priority=priority)
        self.call_count = 0
        self.received_input: "HookInput | None" = None

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        self.received_input = input
        return HookOutput.allow("Session started")


class FailingHook(Hook):
    """Test hook that raises an exception."""

    def __init__(self):
        super().__init__(name="failing_hook", events=[HookEvent.PRE_TOOL_USE])

    async def execute(self, input: HookInput) -> HookOutput:
        raise ValueError("Test error")


class AwaitsUserInputHook(Hook):
    """Test hook that blocks for longer than the watchdog would allow.

    Used to verify the hook manager's ``awaits_user_input`` opt-out:
    despite the parent class having ``timeout=0.1``, the manager
    must NOT cancel this hook because it represents an
    approval-style human wait.
    """

    def __init__(self, delay: float = 0.5):
        super().__init__(
            name="awaits_user_hook",
            events=[HookEvent.PRE_TOOL_USE],
            timeout=0.1,  # would normally cancel a 0.5s sleep
            awaits_user_input=True,
        )
        self.delay = delay
        self.completed = False

    async def execute(self, input: HookInput) -> HookOutput:
        await asyncio.sleep(self.delay)
        self.completed = True
        return HookOutput.allow("approved by human")


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
    async def test_awaits_user_input_hook_not_subject_to_watchdog(self):
        """Hooks that block on a human decision (approvals, prompts)
        opt out of the manager's ``asyncio.wait_for`` watchdog by
        setting ``awaits_user_input=True``. The watchdog's clock is
        designed to bound deterministic hooks (audit, telemetry,
        validation) — applying it to a human wait is the bug that
        made the SecurityHook's approval modals "disappear in
        ~5 seconds." This test asserts the watchdog is genuinely
        skipped, not just bumped.
        """
        manager = HooksManager()
        # Hook sleeps 0.5s but its declared per-hook timeout is 0.1s.
        # Without the opt-out, the manager would cancel it.
        human_hook = AwaitsUserInputHook(delay=0.5)
        manager.register(human_hook)

        input = HookInput(
            session_id="test",
            hook_event_name="PreToolUse",
            tool_name="test_tool",
        )

        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input)
        # The hook ran to completion. If the watchdog had fired, the
        # manager would log "skipping" before the sleep finished and
        # ``human_hook.completed`` would be False.
        assert output.continue_execution is True
        assert human_hook.completed is True, (
            "watchdog cancelled awaits_user_input hook — opt-out failed"
        )

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
    async def test_enforcing_hook_exception_fails_closed(self):
        """#1723: a deny-capable (fail_closed) hook that RAISES must resolve to
        DENY, not allow — a crashed enforcing hook can't silently pass."""
        manager = HooksManager()

        class EnforcingFailingHook(Hook):
            def __init__(self):
                super().__init__(name="enforcing_fail", events=[HookEvent.PRE_TOOL_USE], priority=10)
                self.fail_closed = True

            async def execute(self, input):
                raise ValueError("backend exploded")

        manager.register(EnforcingFailingHook())
        manager.register(AllowAllHook(priority=200))
        input = HookInput(session_id="t", hook_event_name="PreToolUse", tool_name="x")
        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input)
        assert output.continue_execution is False
        assert output.permission_decision == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_enforcing_hook_timeout_fails_closed(self):
        """#1723: an enforcing hook that TIMES OUT must resolve to DENY."""
        manager = HooksManager()

        class EnforcingTimeoutHook(Hook):
            def __init__(self):
                super().__init__(
                    name="enforcing_timeout", events=[HookEvent.PRE_TOOL_USE],
                    priority=10, timeout=0.05,
                )
                self.fail_closed = True

            async def execute(self, input):
                await asyncio.sleep(5.0)
                return HookOutput.allow()

        manager.register(EnforcingTimeoutHook())
        input = HookInput(session_id="t", hook_event_name="PreToolUse", tool_name="x")
        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input)
        assert output.continue_execution is False
        assert output.permission_decision == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_awaits_user_input_hook_exception_fails_closed(self):
        """#1723: an approval hook (awaits_user_input) whose queue raises must
        DENY — we never got approval, so the safe resolution is deny."""
        manager = HooksManager()

        class CrashingApprovalHook(Hook):
            def __init__(self):
                super().__init__(
                    name="approval", events=[HookEvent.PRE_TOOL_USE],
                    priority=10, awaits_user_input=True,
                )

            async def execute(self, input):
                raise RuntimeError("approval queue down")

        manager.register(CrashingApprovalHook())
        input = HookInput(session_id="t", hook_event_name="PreToolUse", tool_name="x")
        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, input)
        assert output.permission_decision == PermissionDecision.DENY

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

    @pytest.mark.asyncio
    async def test_session_start_fires_with_correct_input(self):
        """Test that SESSION_START hook fires via execute_hooks_parallel with correct HookInput."""
        manager = HooksManager()
        hook = SessionStartHook()
        manager.register(hook)

        hook_input = HookInput(
            session_id="agent_init",
            hook_event_name=HookEvent.SESSION_START.value,
        )

        outputs = await manager.execute_hooks_parallel(
            HookEvent.SESSION_START, hook_input
        )

        assert len(outputs) == 1
        assert hook.call_count == 1
        assert hook.received_input.session_id == "agent_init"
        assert hook.received_input.hook_event_name == "SessionStart"
        assert hook.received_input.tool_name is None
        assert hook.received_input.user_message is None

    @pytest.mark.asyncio
    async def test_session_start_multiple_hooks_parallel(self):
        """Test that multiple SESSION_START hooks execute in parallel."""
        manager = HooksManager()
        hook1 = SessionStartHook(name="hook1")
        hook2 = SessionStartHook(name="hook2")
        manager.register(hook1)
        manager.register(hook2)

        hook_input = HookInput(
            session_id="agent_init",
            hook_event_name=HookEvent.SESSION_START.value,
        )

        outputs = await manager.execute_hooks_parallel(
            HookEvent.SESSION_START, hook_input
        )

        assert len(outputs) == 2
        assert hook1.call_count == 1
        assert hook2.call_count == 1


# === USER_PROMPT_SUBMIT Hook Tests ===


class UserPromptSubmitAllowHook(Hook):
    """Test hook for USER_PROMPT_SUBMIT that allows."""

    def __init__(self, name: str = "prompt_allow", priority: int = 100):
        super().__init__(name=name, events=[HookEvent.USER_PROMPT_SUBMIT], priority=priority)
        self.call_count = 0
        self.received_message = None

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        self.received_message = input.user_message
        return HookOutput.allow("Prompt allowed")


class UserPromptSubmitDenyHook(Hook):
    """Test hook for USER_PROMPT_SUBMIT that denies."""

    def __init__(self, name: str = "prompt_deny", priority: int = 100):
        super().__init__(name=name, events=[HookEvent.USER_PROMPT_SUBMIT], priority=priority)
        self.call_count = 0

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        return HookOutput.deny("Content policy violation")


class UserPromptSubmitModifyHook(Hook):
    """Test hook for USER_PROMPT_SUBMIT that modifies user_message."""

    def __init__(self, replacement: str, name: str = "prompt_modify", priority: int = 100):
        super().__init__(name=name, events=[HookEvent.USER_PROMPT_SUBMIT], priority=priority)
        self.replacement = replacement
        self.call_count = 0

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        return HookOutput.modify({"user_message": self.replacement}, "Input sanitized")


class TestUserPromptSubmitHooks:
    """Tests for USER_PROMPT_SUBMIT hook event."""

    @pytest.mark.asyncio
    async def test_fires_with_correct_user_message(self):
        """Verify USER_PROMPT_SUBMIT fires with correct user_message in HookInput."""
        manager = HooksManager()
        hook = UserPromptSubmitAllowHook()
        manager.register(hook)

        hook_input = HookInput(
            session_id="test-session",
            hook_event_name=HookEvent.USER_PROMPT_SUBMIT.value,
            user_message="Hello, agent!",
        )

        output = await manager.execute_hooks(HookEvent.USER_PROMPT_SUBMIT, hook_input)
        assert output.continue_execution is True
        assert hook.call_count == 1
        assert hook.received_message == "Hello, agent!"

    @pytest.mark.asyncio
    async def test_deny_blocks_processing(self):
        """Test DENY blocks processing and returns rejection."""
        manager = HooksManager()
        deny_hook = UserPromptSubmitDenyHook(priority=50)
        allow_hook = UserPromptSubmitAllowHook(priority=100)

        manager.register(deny_hook)
        manager.register(allow_hook)

        hook_input = HookInput(
            session_id="test-session",
            hook_event_name=HookEvent.USER_PROMPT_SUBMIT.value,
            user_message="bad input",
        )

        output = await manager.execute_hooks(HookEvent.USER_PROMPT_SUBMIT, hook_input)
        assert output.continue_execution is False
        assert output.permission_decision == PermissionDecision.DENY
        assert output.permission_reason == "Content policy violation"
        assert deny_hook.call_count == 1
        assert allow_hook.call_count == 0  # Chain stopped before allow hook

    @pytest.mark.asyncio
    async def test_modify_updates_user_input(self):
        """Test MODIFY updates user_input before context building.

        The HooksManager applies updated_input to hook_input.tool_input,
        so callers check hook_input.tool_input["user_message"] after execution.
        """
        manager = HooksManager()
        hook = UserPromptSubmitModifyHook(replacement="sanitized input")
        manager.register(hook)

        hook_input = HookInput(
            session_id="test-session",
            hook_event_name=HookEvent.USER_PROMPT_SUBMIT.value,
            user_message="original input",
        )

        output = await manager.execute_hooks(HookEvent.USER_PROMPT_SUBMIT, hook_input)
        assert output.continue_execution is True
        assert hook.call_count == 1
        # Manager applies updated_input to hook_input.tool_input
        assert hook_input.tool_input == {"user_message": "sanitized input"}

    @pytest.mark.asyncio
    async def test_modify_threads_empty_dict_rewrite(self):
        """An empty-dict rewrite from a MODIFY hook is a legitimate
        decision — a redactor / constraint hook clearing all sensitive
        fields.  Truthiness checks (``if output.updated_input``) would
        silently drop the rewrite and downstream callers would see the
        original payload.  Regression guard for codex round-5 on
        #1314.
        """
        class EmptyRewriteHook(Hook):
            def __init__(self):
                super().__init__(name="empty_rewrite", events=[HookEvent.PRE_TOOL_USE], priority=100)

            async def execute(self, input: HookInput) -> HookOutput:
                return HookOutput.modify(updated_input={}, reason="clear all args")

        manager = HooksManager()
        manager.register(EmptyRewriteHook())

        hook_input = HookInput(
            session_id="test",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="send_email",
            tool_input={"to": "x@example.com", "body": "secret"},
        )

        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert output.continue_execution is True
        # The empty rewrite REACHES the threaded input — downstream
        # callers must see ``{}``, not the original sensitive payload.
        assert hook_input.tool_input == {}

    @pytest.mark.asyncio
    async def test_no_hooks_allows(self):
        """No registered hooks should allow processing to continue."""
        manager = HooksManager()

        hook_input = HookInput(
            session_id="test-session",
            hook_event_name=HookEvent.USER_PROMPT_SUBMIT.value,
            user_message="any input",
        )

        output = await manager.execute_hooks(HookEvent.USER_PROMPT_SUBMIT, hook_input)
        assert output.continue_execution is True
        assert output.permission_decision == PermissionDecision.ALLOW


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


# === POST_SUBAGENT_CALL Tests ===

class PostSubagentHook(Hook):
    """Test hook that captures POST_SUBAGENT_CALL inputs."""

    def __init__(self, name: str = "post_subagent_hook"):
        super().__init__(
            name=name,
            events=[HookEvent.POST_SUBAGENT_CALL],
            priority=100,
        )
        self.received_inputs = []

    async def execute(self, input: HookInput) -> HookOutput:
        self.received_inputs.append(input)
        return HookOutput.allow()


class TestPostSubagentCall:
    """Tests for POST_SUBAGENT_CALL hook event."""

    @pytest.mark.asyncio
    async def test_post_subagent_call_fires_on_success(self):
        """POST_SUBAGENT_CALL hook receives correct input on success."""
        manager = HooksManager()
        hook = PostSubagentHook()
        manager.register(hook)

        hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
            tool_name="memory_search",
            tool_input={"task": "find old conversations"},
            feature_name="MemoryFeature",
            tool_response={"success": True, "results": ["item1"]},
            execution_time_ms=150,
        )

        outputs = await manager.execute_hooks_parallel(
            HookEvent.POST_SUBAGENT_CALL, hook_input
        )

        assert len(outputs) == 1
        assert len(hook.received_inputs) == 1
        received = hook.received_inputs[0]
        assert received.hook_event_name == "PostSubagentCall"
        assert received.tool_name == "memory_search"
        assert received.feature_name == "MemoryFeature"
        assert received.tool_response == {"success": True, "results": ["item1"]}
        assert received.execution_time_ms == 150

    @pytest.mark.asyncio
    async def test_post_subagent_call_fires_on_failure(self):
        """POST_SUBAGENT_CALL hook receives error info on failure."""
        manager = HooksManager()
        hook = PostSubagentHook()
        manager.register(hook)

        hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
            tool_name="memory_search",
            tool_input={"task": "find old conversations"},
            feature_name="MemoryFeature",
            tool_response={"success": False, "error": "Connection refused"},
            execution_time_ms=50,
        )

        outputs = await manager.execute_hooks_parallel(
            HookEvent.POST_SUBAGENT_CALL, hook_input
        )

        assert len(outputs) == 1
        received = hook.received_inputs[0]
        assert received.tool_response["success"] is False
        assert "Connection refused" in received.tool_response["error"]

    @pytest.mark.asyncio
    async def test_post_subagent_call_parallel_execution(self):
        """Multiple POST_SUBAGENT_CALL hooks execute in parallel."""
        manager = HooksManager()
        hook1 = PostSubagentHook(name="observer_1")
        hook2 = PostSubagentHook(name="observer_2")
        hook3 = PostSubagentHook(name="observer_3")

        manager.register(hook1)
        manager.register(hook2)
        manager.register(hook3)

        hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
            tool_name="web_search",
            feature_name="SearchFeature",
            tool_response={"success": True},
            execution_time_ms=200,
        )

        outputs = await manager.execute_hooks_parallel(
            HookEvent.POST_SUBAGENT_CALL, hook_input
        )

        assert len(outputs) == 3
        assert len(hook1.received_inputs) == 1
        assert len(hook2.received_inputs) == 1
        assert len(hook3.received_inputs) == 1

    @pytest.mark.asyncio
    async def test_post_subagent_call_does_not_trigger_pre_hooks(self):
        """POST_SUBAGENT_CALL does not trigger PRE_TOOL_USE hooks."""
        manager = HooksManager()
        pre_hook = AllowAllHook(name="pre_only")
        post_hook = PostSubagentHook(name="post_only")

        manager.register(pre_hook)
        manager.register(post_hook)

        hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
            tool_name="test_tool",
            tool_response={"success": True},
        )

        outputs = await manager.execute_hooks_parallel(
            HookEvent.POST_SUBAGENT_CALL, hook_input
        )

        assert len(outputs) == 1
        assert pre_hook.call_count == 0
        assert len(post_hook.received_inputs) == 1


# === POST_RESPONSE MODIFY propagation (#2033) ===


class PostResponseModifyHook(Hook):
    """POST_RESPONSE hook that rewrites the response text via MODIFY.

    Mirrors response_audit's warn mode, which appends an audit
    annotation through ``HookOutput.modify(updated_input={"response_text": ...})``.
    """

    def __init__(self, suffix: str, name: str = "post_response_modify", priority: int = 100):
        super().__init__(name=name, events=[HookEvent.POST_RESPONSE], priority=priority)
        self.suffix = suffix
        self.call_count = 0
        self.seen_response_text: "str | None" = None

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        self.seen_response_text = input.response_text
        return HookOutput.modify(
            updated_input={"response_text": (input.response_text or "") + self.suffix},
            reason="audit annotation",
        )


class PostResponseDenyHook(Hook):
    """POST_RESPONSE hook that always DENIES — mirrors response_audit strict mode
    blocking a risky turn."""

    def __init__(self, name: str = "post_response_deny", priority: int = 50):
        super().__init__(name=name, events=[HookEvent.POST_RESPONSE], priority=priority)
        self.call_count = 0

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        return HookOutput.deny("blocked by audit")


class TestPostResponseModify:
    """Regression tests for #2033: POST_RESPONSE MODIFY must round-trip
    its ``updated_input`` back to the caller, not be dropped on the
    final ALLOW."""

    @pytest.mark.asyncio
    async def test_post_response_modify_propagates_to_returned_output(self):
        """A POST_RESPONSE hook returning modify(updated_input={"response_text": "X"})
        must surface on execute_hooks(...).updated_input["response_text"]."""
        manager = HooksManager()
        manager.register(PostResponseModifyHook(suffix=" [warned]"))

        hook_input = HookInput(
            session_id="test",
            hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text="original",
        )

        output = await manager.execute_hooks(HookEvent.POST_RESPONSE, hook_input)
        assert output.continue_execution is True
        assert output.updated_input is not None
        assert output.updated_input["response_text"] == "original [warned]"

    @pytest.mark.asyncio
    async def test_post_response_modify_chains_across_hooks(self):
        """A second hook in the chain audits the rewritten text, and the
        final accumulated rewrite reaches the caller."""
        manager = HooksManager()
        manager.register(PostResponseModifyHook(suffix=" [first]", name="first", priority=50))
        second = PostResponseModifyHook(suffix=" [second]", name="second", priority=100)
        manager.register(second)

        hook_input = HookInput(
            session_id="test",
            hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text="base",
        )

        output = await manager.execute_hooks(HookEvent.POST_RESPONSE, hook_input)
        # Second hook saw the first hook's rewrite threaded into response_text.
        assert second.seen_response_text == "base [first]"
        assert output.updated_input["response_text"] == "base [first] [second]"


class TestExecuteHooksSnapshot:
    """#2674: ``execute_hooks_snapshot`` runs a CALLER-PINNED hook list, so the
    strict streaming audit enforces exactly the POST_RESPONSE set captured at
    turn start — independent of anything the turn's tools register/enable/disable
    at the live registry mid-turn."""

    @pytest.mark.asyncio
    async def test_snapshot_ignores_hook_registered_after_capture(self):
        """A hook registered AFTER the snapshot is not in the pinned list, so it
        does not run — even though it is live in the registry now (the
        skip→enable-strict fail-open case)."""
        manager = HooksManager()
        hook_input = HookInput(
            session_id="t",
            hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text="base",
        )
        # Snapshot taken while the registry is empty.
        snapshot = manager.get_enabled_hooks(HookEvent.POST_RESPONSE)
        assert snapshot == []

        # A denier is registered live afterwards.
        denier = PostResponseDenyHook()
        manager.register(denier)

        out = await manager.execute_hooks_snapshot(
            HookEvent.POST_RESPONSE, hook_input, snapshot,
        )
        # Snapshot was empty → nothing enforced, despite the live denier.
        assert out.permission_decision == PermissionDecision.ALLOW
        assert denier.call_count == 0
        # Sanity: the LIVE registry WOULD have denied.
        live = await manager.execute_hooks(HookEvent.POST_RESPONSE, hook_input)
        assert live.permission_decision == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_snapshot_runs_hook_disabled_after_capture(self):
        """A hook captured while enabled still runs from the snapshot even after
        it is disabled — ignoring the current ``enabled`` flag (the
        strict→disable release-without-audit case)."""
        manager = HooksManager()
        denier = PostResponseDenyHook()
        manager.register(denier)
        hook_input = HookInput(
            session_id="t",
            hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text="base",
        )
        snapshot = manager.get_enabled_hooks(HookEvent.POST_RESPONSE)
        assert snapshot == [denier]

        # Disable it after capture, as ``audit_disable`` would.
        denier.enabled = False
        # Live execution now finds nothing enabled → ALLOW.
        live = await manager.execute_hooks(HookEvent.POST_RESPONSE, hook_input)
        assert live.permission_decision == PermissionDecision.ALLOW

        # Snapshot execution ignores the disable and still enforces.
        out = await manager.execute_hooks_snapshot(
            HookEvent.POST_RESPONSE, hook_input, snapshot,
        )
        assert out.permission_decision == PermissionDecision.DENY
        assert denier.call_count == 1

    @pytest.mark.asyncio
    async def test_snapshot_preserves_priority_order(self):
        """The pinned list is executed in priority order regardless of the order
        it was passed in."""
        manager = HooksManager()
        first = PostResponseModifyHook(suffix=" [first]", name="first", priority=10)
        second = PostResponseModifyHook(suffix=" [second]", name="second", priority=90)
        hook_input = HookInput(
            session_id="t",
            hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text="base",
        )
        # Passed high-priority-first; the manager must still run first→second.
        out = await manager.execute_hooks_snapshot(
            HookEvent.POST_RESPONSE, hook_input, [second, first],
        )
        assert second.seen_response_text == "base [first]"
        assert out.updated_input["response_text"] == "base [first] [second]"


class _FlippableEnforcingHook(Hook):
    """A hook whose enforcement flag (``fail_closed`` / ``awaits_user_input``)
    can be flipped AFTER the turn-start snapshot, and whose ``execute`` can be
    scripted to raise or hang (#2674 P0-1). Models a generic (mode-less)
    POST_RESPONSE gate whose live flag drifts from its captured turn-start value.
    """

    def __init__(self, *, fail_closed=False, awaits_user_input=False,
                 behavior="allow", priority=10, timeout=0.05):
        super().__init__(
            name="flippable", events=[HookEvent.PRE_TOOL_USE],
            priority=priority, timeout=timeout,
            awaits_user_input=awaits_user_input,
        )
        # A plain mutable attribute (NOT the ResponseAuditHook's mode-derived
        # property) — the exact shape the P0-1 repro flips mid-turn.
        self.fail_closed = fail_closed
        self.behavior = behavior
        self.executed = 0

    async def execute(self, input):
        self.executed += 1
        if self.behavior == "raise":
            raise RuntimeError("enforcing backend exploded")
        if self.behavior == "hang":
            await asyncio.sleep(5.0)
        return HookOutput.allow()


class TestSnapshotEnforcementOverrides:
    """#2674 P0-1: ``enforcement_overrides`` pins the CAPTURED (turn-start)
    fail-closed verdict for a caller-pinned run, so a hook whose ``fail_closed``
    / ``awaits_user_input`` flips mid-turn cannot change whether its crash /
    timeout denies. Without the override the live read wins and the two
    directions diverge — the raw-streamed / block-persisted fail-open split."""

    def _pre_tool_input(self):
        return HookInput(
            session_id="t", hook_event_name="PreToolUse", tool_name="x",
        )

    @pytest.mark.asyncio
    async def test_override_keeps_flipped_hook_fail_closed_on_crash(self):
        """Captured enforcing=True + live flip to False + crash → override wins →
        DENY. The control (no override, live read) skips and ALLOWs — the exact
        fail-open the override closes."""
        manager = HooksManager()
        hook = _FlippableEnforcingHook(fail_closed=True, behavior="raise")
        # Enforcement captured True at snapshot, then flipped False mid-turn.
        overrides = {id(hook): True}
        hook.fail_closed = False

        out = await manager.execute_hooks_snapshot(
            HookEvent.PRE_TOOL_USE, self._pre_tool_input(), [hook],
            enforcement_overrides=overrides,
        )
        assert out.permission_decision == PermissionDecision.DENY
        assert out.continue_execution is False

        # Control: same flipped hook, NO override → live read (False) → skip →
        # ALLOW (the pre-fix fail-open).
        hook2 = _FlippableEnforcingHook(fail_closed=True, behavior="raise")
        hook2.fail_closed = False
        live = await manager.execute_hooks_snapshot(
            HookEvent.PRE_TOOL_USE, self._pre_tool_input(), [hook2],
        )
        assert live.permission_decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_override_keeps_flipped_hook_fail_closed_on_timeout(self):
        """Same as the crash case but the enforcing hook TIMES OUT: the captured
        override still fails it closed even though its live ``fail_closed`` is
        now False."""
        manager = HooksManager()
        hook = _FlippableEnforcingHook(
            fail_closed=True, behavior="hang", timeout=0.05,
        )
        overrides = {id(hook): True}
        hook.fail_closed = False

        out = await manager.execute_hooks_snapshot(
            HookEvent.PRE_TOOL_USE, self._pre_tool_input(), [hook],
            enforcement_overrides=overrides,
        )
        assert out.permission_decision == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_override_keeps_nonenforcing_hook_advisory_on_crash(self):
        """Inverse transition: captured enforcing=False + live flip to True +
        crash → override wins → SKIP (ALLOW), staying advisory for the turn its
        buffering decision never accounted for. The control (live read, no
        override) would DENY off the flipped flag."""
        manager = HooksManager()
        hook = _FlippableEnforcingHook(fail_closed=False, behavior="raise")
        overrides = {id(hook): False}
        hook.fail_closed = True  # flip to enforcing AFTER capture

        out = await manager.execute_hooks_snapshot(
            HookEvent.PRE_TOOL_USE, self._pre_tool_input(), [hook],
            enforcement_overrides=overrides,
        )
        assert out.permission_decision == PermissionDecision.ALLOW

        # Control: the live read (no override) sees the flipped True → DENY.
        hook2 = _FlippableEnforcingHook(fail_closed=False, behavior="raise")
        hook2.fail_closed = True
        live = await manager.execute_hooks_snapshot(
            HookEvent.PRE_TOOL_USE, self._pre_tool_input(), [hook2],
        )
        assert live.permission_decision == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_override_captures_awaits_user_input_enforcement(self):
        """``_hook_is_enforcing`` = ``fail_closed OR awaits_user_input``. An
        approval hook captured enforcing via ``awaits_user_input`` that drops the
        flag mid-turn (and crashes) must still DENY off the captured value."""
        manager = HooksManager()
        hook = _FlippableEnforcingHook(
            awaits_user_input=True, behavior="raise", timeout=0.05,
        )
        overrides = {id(hook): True}
        hook.awaits_user_input = False  # flip after capture

        out = await manager.execute_hooks_snapshot(
            HookEvent.PRE_TOOL_USE, self._pre_tool_input(), [hook],
            enforcement_overrides=overrides,
        )
        assert out.permission_decision == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_observer_override_skips_crashed_flipped_hook(self):
        """``execute_post_response_observers`` honors the captured override too:
        an observer captured non-enforcing that flips ``fail_closed`` True and
        crashes SKIPS (never re-blocks an approved release off a live flag the
        turn's buffering never accounted for)."""
        manager = HooksManager()
        obs = _FlippableEnforcingHook(
            fail_closed=False, behavior="raise", priority=90,
        )
        hook_input = HookInput(
            session_id="t", hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text="reviewed release",
        )
        obs.fail_closed = True  # flip to enforcing after capture

        out = await manager.execute_post_response_observers(
            HookEvent.POST_RESPONSE, hook_input, [obs],
            enforcement_overrides={id(obs): False},
        )
        # Crashed observer skipped → the approved release stands (ALLOW).
        assert out.permission_decision == PermissionDecision.ALLOW


# === Run tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
