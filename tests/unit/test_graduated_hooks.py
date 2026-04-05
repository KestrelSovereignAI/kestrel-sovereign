"""Tests for graduated hook decisions (warning fields on HookOutput)."""

import asyncio
import pytest

from kestrel_sdk.hooks.base import (
    Hook,
    HookEvent,
    HookInput,
    HookOutput,
    PermissionDecision,
)
from kestrel_sovereign.hooks.manager import HooksManager


class AllowHook(Hook):
    """Hook that always allows."""
    def __init__(self, name="allow_hook"):
        super().__init__(name=name, events=[HookEvent.PRE_TOOL_USE])

    async def execute(self, input: HookInput) -> HookOutput:
        return HookOutput.allow()


class WarnHook(Hook):
    """Hook that returns a warning but allows."""
    def __init__(self, name="warn_hook", message="Advisory", severity="warning"):
        super().__init__(name=name, events=[HookEvent.PRE_TOOL_USE])
        self._message = message
        self._severity = severity

    async def execute(self, input: HookInput) -> HookOutput:
        return HookOutput.warn(self._message, self._severity)


class DenyHook(Hook):
    """Hook that denies execution."""
    def __init__(self, name="deny_hook"):
        super().__init__(name=name, events=[HookEvent.PRE_TOOL_USE])

    async def execute(self, input: HookInput) -> HookOutput:
        return HookOutput.deny("Blocked by policy")


def _make_input() -> HookInput:
    return HookInput(
        session_id="test-session",
        hook_event_name="PreToolUse",
        tool_name="test_tool",
    )


class TestHookOutputWarnFactory:
    """Test the HookOutput.warn() class method."""

    def test_warn_creates_allow_with_warning(self):
        output = HookOutput.warn("Advisory message", "warning")
        assert output.continue_execution is True
        assert output.permission_decision == PermissionDecision.ALLOW
        assert output.warning_message == "Advisory message"
        assert output.warning_severity == "warning"

    def test_warn_default_severity(self):
        output = HookOutput.warn("Message")
        assert output.warning_severity == "warning"

    def test_warn_info_severity(self):
        output = HookOutput.warn("Info", "info")
        assert output.warning_severity == "info"

    def test_warn_critical_severity(self):
        output = HookOutput.warn("Critical", "critical")
        assert output.warning_severity == "critical"


class TestHookOutputSerialization:
    """Test that warning fields serialize correctly."""

    def test_to_dict_includes_warning_fields(self):
        output = HookOutput.warn("Advisory", "warning")
        d = output.to_dict()
        assert d["warning_message"] == "Advisory"
        assert d["warning_severity"] == "warning"

    def test_to_dict_warning_fields_none_by_default(self):
        output = HookOutput.allow()
        d = output.to_dict()
        assert d["warning_message"] is None
        assert d["warning_severity"] is None


class TestWarningAccumulation:
    """Test that HooksManager accumulates warnings across the hook chain."""

    @pytest.mark.asyncio
    async def test_single_warning_passes_through(self):
        mgr = HooksManager()
        mgr.register(WarnHook(name="w1", message="Watch out"))
        result = await mgr.execute_hooks(HookEvent.PRE_TOOL_USE, _make_input())
        assert result.continue_execution is True
        assert result.warning_message == "Watch out"
        assert result.warning_severity == "warning"

    @pytest.mark.asyncio
    async def test_multiple_warnings_accumulate(self):
        mgr = HooksManager()
        mgr.register(WarnHook(name="w1", message="First warning", severity="info"))
        mgr.register(WarnHook(name="w2", message="Second warning", severity="warning"))
        result = await mgr.execute_hooks(HookEvent.PRE_TOOL_USE, _make_input())
        assert result.continue_execution is True
        assert "First warning" in result.warning_message
        assert "Second warning" in result.warning_message
        # Max severity should be "warning" (higher than "info")
        assert result.warning_severity == "warning"

    @pytest.mark.asyncio
    async def test_warning_does_not_block_execution(self):
        mgr = HooksManager()
        mgr.register(WarnHook(name="w1", message="Advisory"))
        mgr.register(AllowHook(name="a1"))
        result = await mgr.execute_hooks(HookEvent.PRE_TOOL_USE, _make_input())
        assert result.continue_execution is True
        assert result.permission_decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_warnings_attached_to_deny(self):
        mgr = HooksManager()
        mgr.register(WarnHook(name="w1", message="Advisory before deny", severity="info"))
        mgr.register(DenyHook(name="d1"))
        result = await mgr.execute_hooks(HookEvent.PRE_TOOL_USE, _make_input())
        assert result.continue_execution is False
        assert result.permission_decision == PermissionDecision.DENY
        # Warning should still be attached even though denied
        assert "Advisory before deny" in result.warning_message

    @pytest.mark.asyncio
    async def test_no_warnings_when_none_returned(self):
        mgr = HooksManager()
        mgr.register(AllowHook(name="a1"))
        mgr.register(AllowHook(name="a2"))
        result = await mgr.execute_hooks(HookEvent.PRE_TOOL_USE, _make_input())
        assert result.warning_message is None
        assert result.warning_severity is None

    @pytest.mark.asyncio
    async def test_max_severity_escalates(self):
        mgr = HooksManager()
        mgr.register(WarnHook(name="w1", message="Info", severity="info"))
        mgr.register(WarnHook(name="w2", message="Critical", severity="critical"))
        mgr.register(WarnHook(name="w3", message="Warning", severity="warning"))
        result = await mgr.execute_hooks(HookEvent.PRE_TOOL_USE, _make_input())
        assert result.warning_severity == "critical"

    @pytest.mark.asyncio
    async def test_no_hooks_returns_allow_no_warnings(self):
        mgr = HooksManager()
        result = await mgr.execute_hooks(HookEvent.PRE_TOOL_USE, _make_input())
        assert result.continue_execution is True
        assert result.warning_message is None
