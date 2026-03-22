"""
Unit tests for the per-response audit plugin.

Tests the ResponseAuditHook for LLM response integrity checking and
the ResponseAuditFeature for mode management and tool commands.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.hooks.base import (
    HookEvent,
    HookInput,
    HookOutput,
    PermissionDecision,
)
from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook
from kestrel_sovereign.features.response_audit.feature import ResponseAuditFeature


# =========================================================================
# Helpers
# =========================================================================


def _make_hook_input(response_text: str = "This is a normal helpful response from the agent.") -> HookInput:
    """Create a HookInput with the given response_text."""
    return HookInput(
        session_id="test-session",
        hook_event_name=HookEvent.POST_RESPONSE.value,
        response_text=response_text,
    )


def _make_agent(audit_response: dict = None):
    """Create a mock agent with llm_service.get_audit_response."""
    agent = MagicMock()
    agent.llm_service = MagicMock()
    if audit_response is None:
        audit_response = {"risk_level": 1, "reasoning": "Normal response"}
    agent.llm_service.get_audit_response = AsyncMock(return_value=audit_response)
    agent.hooks_manager = MagicMock()
    agent.hooks_manager.register = MagicMock()
    agent.hooks_manager.get_enabled_hooks = MagicMock(return_value=[])
    agent.features = {}
    return agent


# =========================================================================
# ResponseAuditHook Tests
# =========================================================================


class TestResponseAuditHook:
    """Tests for ResponseAuditHook."""

    @pytest.mark.asyncio
    async def test_hook_allow_low_risk(self):
        """Low risk audit result should ALLOW the response."""
        agent = _make_agent({"risk_level": 1, "reasoning": "Normal response"})
        hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)

        output = await hook.execute(_make_hook_input())

        assert output.permission_decision == PermissionDecision.ALLOW
        assert output.continue_execution is True
        assert hook.audit_count == 1
        assert hook.last_risk_level == 1

    @pytest.mark.asyncio
    async def test_hook_deny_strict_high_risk(self):
        """Strict mode with high risk should DENY the response."""
        agent = _make_agent({"risk_level": 3, "reasoning": "Harmful content detected"})
        hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)

        output = await hook.execute(_make_hook_input())

        assert output.permission_decision == PermissionDecision.DENY
        assert output.continue_execution is False
        assert "Harmful content detected" in output.permission_reason
        assert hook.last_risk_level == 3

    @pytest.mark.asyncio
    async def test_hook_warn_high_risk(self):
        """Warn mode with high risk should MODIFY the response with a warning."""
        agent = _make_agent({"risk_level": 3, "reasoning": "Potentially misleading"})
        hook = ResponseAuditHook(agent=agent, mode="warn", risk_threshold=3)
        original_text = "This is a normal helpful response from the agent."

        output = await hook.execute(_make_hook_input(original_text))

        assert output.permission_decision == PermissionDecision.ALLOW
        assert output.continue_execution is True
        assert output.updated_input is not None
        modified_text = output.updated_input["response_text"]
        assert original_text in modified_text
        assert "[Audit warning (risk 3)" in modified_text
        assert "Potentially misleading" in modified_text

    @pytest.mark.asyncio
    async def test_hook_skip_short_response(self):
        """Responses shorter than 20 characters should be skipped."""
        agent = _make_agent()
        hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)

        output = await hook.execute(_make_hook_input("Short"))

        assert output.permission_decision == PermissionDecision.ALLOW
        assert "too short" in output.permission_reason
        # get_audit_response should NOT have been called
        agent.llm_service.get_audit_response.assert_not_called()
        assert hook.audit_count == 0

    @pytest.mark.asyncio
    async def test_hook_error_handling(self):
        """Audit errors should result in ALLOW with error reason."""
        agent = _make_agent()
        agent.llm_service.get_audit_response = AsyncMock(
            side_effect=RuntimeError("LLM provider unavailable")
        )
        hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)

        output = await hook.execute(_make_hook_input())

        assert output.permission_decision == PermissionDecision.ALLOW
        assert "error" in output.permission_reason.lower()
        assert hook.audit_count == 0

    @pytest.mark.asyncio
    async def test_hook_medium_risk_below_threshold(self):
        """Risk below threshold should ALLOW regardless of mode."""
        agent = _make_agent({"risk_level": 2, "reasoning": "Minor concern"})
        hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)

        output = await hook.execute(_make_hook_input())

        assert output.permission_decision == PermissionDecision.ALLOW
        assert hook.last_risk_level == 2

    @pytest.mark.asyncio
    async def test_hook_notifies_audit_anchor(self):
        """Hook should notify AuditAnchorFeature for risk >= 2."""
        agent = _make_agent({"risk_level": 2, "reasoning": "Some concern"})
        mock_anchor = MagicMock()
        mock_anchor.on_audit_complete = AsyncMock()
        agent.features = {"AuditAnchorFeature": mock_anchor}
        hook = ResponseAuditHook(agent=agent, mode="warn", risk_threshold=3)

        await hook.execute(_make_hook_input())

        mock_anchor.on_audit_complete.assert_called_once()
        call_args = mock_anchor.on_audit_complete.call_args[0][0]
        assert call_args["source"] == "response_audit"
        assert call_args["is_valid"] is True

    @pytest.mark.asyncio
    async def test_hook_no_anchor_notification_low_risk(self):
        """Hook should NOT notify AuditAnchorFeature for risk < 2."""
        agent = _make_agent({"risk_level": 1, "reasoning": "Normal"})
        mock_anchor = MagicMock()
        mock_anchor.on_audit_complete = AsyncMock()
        agent.features = {"AuditAnchorFeature": mock_anchor}
        hook = ResponseAuditHook(agent=agent, mode="warn", risk_threshold=3)

        await hook.execute(_make_hook_input())

        mock_anchor.on_audit_complete.assert_not_called()


# =========================================================================
# ResponseAuditFeature Tests
# =========================================================================


class TestResponseAuditFeature:
    """Tests for ResponseAuditFeature."""

    @pytest.mark.asyncio
    async def test_feature_skip_mode_no_hook(self):
        """In skip mode, initialize should NOT register a hook."""
        agent = _make_agent()
        with patch.dict("os.environ", {"KESTREL_RESPONSE_AUDIT_MODE": "skip"}, clear=False):
            feature = ResponseAuditFeature(agent)
            await feature.initialize()

        assert feature._hook is None
        agent.hooks_manager.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_feature_warn_mode_registers_hook(self):
        """In warn mode, initialize should register a hook."""
        agent = _make_agent()
        with patch.dict("os.environ", {"KESTREL_RESPONSE_AUDIT_MODE": "warn"}, clear=False):
            feature = ResponseAuditFeature(agent)
            await feature.initialize()

        assert feature._hook is not None
        agent.hooks_manager.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_feature_enable_disable(self):
        """Enable and disable should toggle the hook state."""
        agent = _make_agent()
        feature = ResponseAuditFeature(agent)
        feature._mode = "skip"

        # Enable
        result = await feature.enable_audit(mode="warn")
        assert result["status"] == "enabled"
        assert result["mode"] == "warn"
        assert feature._hook is not None
        assert feature._hook.enabled is True

        # Disable
        result = await feature.disable_audit()
        assert result["status"] == "disabled"
        assert feature._hook.enabled is False
        assert feature._mode == "skip"

    @pytest.mark.asyncio
    async def test_feature_enable_invalid_mode(self):
        """Enabling with invalid mode should return error."""
        agent = _make_agent()
        feature = ResponseAuditFeature(agent)

        result = await feature.enable_audit(mode="invalid")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_feature_enable_updates_existing_hook(self):
        """Enabling when hook already exists should update mode."""
        agent = _make_agent()
        feature = ResponseAuditFeature(agent)

        await feature.enable_audit(mode="warn")
        assert feature._hook.mode == "warn"

        result = await feature.enable_audit(mode="strict")
        assert result["status"] == "updated"
        assert feature._hook.mode == "strict"

    @pytest.mark.asyncio
    async def test_feature_status(self):
        """audit_status should return all expected fields."""
        agent = _make_agent()
        feature = ResponseAuditFeature(agent)

        status = await feature.audit_status()

        assert "mode" in status
        assert "strategy" in status
        assert "risk_threshold" in status
        assert "hook_registered" in status
        assert "audit_count" in status
        assert "last_risk_level" in status
        assert status["mode"] == "skip"
        assert status["hook_registered"] is False
        assert status["audit_count"] == 0

    @pytest.mark.asyncio
    async def test_audit_count_tracking(self):
        """Audit count should increment with each audit call."""
        agent = _make_agent({"risk_level": 1, "reasoning": "OK"})
        hook = ResponseAuditHook(agent=agent, mode="warn", risk_threshold=3)

        assert hook.audit_count == 0

        await hook.execute(_make_hook_input("First response that is long enough to audit"))
        assert hook.audit_count == 1

        await hook.execute(_make_hook_input("Second response that is long enough to audit"))
        assert hook.audit_count == 2

        await hook.execute(_make_hook_input("Third response that is long enough to audit"))
        assert hook.audit_count == 3
        assert hook.last_risk_level == 1
