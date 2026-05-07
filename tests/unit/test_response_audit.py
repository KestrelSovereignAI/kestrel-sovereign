"""
Unit tests for the per-response audit plugin.

Tests the ResponseAuditHook for LLM response integrity checking and
the ResponseAuditFeature for mode management and tool commands.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.hooks.base import (
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
# Wave 5D — Narration check folding (#1042 layer 3)
# =========================================================================


def _make_narration_hook_input(
    response_text: str = "Looking at the result, the save did not persist.",
    pre_tool_prose: str | None = "Saved your favorite color.",
    tool_results: list | None = None,
) -> HookInput:
    """HookInput pre-loaded with narration-check fields. Default
    payload reproduces the canonical #1042 case: agent claimed
    'Saved' before the tool returned an error envelope."""
    if tool_results is None:
        tool_results = [
            {"tool_call_id": "tc-1", "name": "save_fact",
             "result": {"status": "error", "error": "no store"}},
        ]
    return HookInput(
        session_id="t",
        hook_event_name=HookEvent.POST_RESPONSE.value,
        response_text=response_text,
        pre_tool_prose=pre_tool_prose,
        tool_calls=[{"id": "tc-1", "name": "save_fact", "arguments": {}}],
        tool_results=tool_results,
    )


class TestResponseAuditHookNarrationFolding:
    """Wave 5D: deterministic narration check folds into audit risk."""

    @pytest.mark.asyncio
    async def test_narration_violation_elevates_clean_audit_to_warn(self):
        """LLM audit returned low risk, but the deterministic check
        catches a 'Saved' before a failed tool. Combined risk crosses
        threshold → warn-mode appends an audit warning to the response."""
        agent = _make_agent({"risk_level": 1, "reasoning": "Normal"})
        hook = ResponseAuditHook(agent=agent, mode="warn", risk_threshold=3)

        output = await hook.execute(_make_narration_hook_input())

        # 1 (LLM audit) + 2 (narration boost) = 3 → at threshold
        assert hook.last_risk_level == 3
        assert output.permission_decision == PermissionDecision.ALLOW
        assert output.updated_input is not None
        assert "narration_check" in output.updated_input["response_text"]
        assert "save_fact" in output.updated_input["response_text"]
        # The verdict object is exposed for telemetry consumers.
        assert hook.last_narration_verdict is not None
        assert hook.last_narration_verdict.offending_tool == "save_fact"
        assert hook.last_narration_verdict.offending_verb == "saved"

    @pytest.mark.asyncio
    async def test_narration_violation_can_drive_strict_deny(self):
        """In strict mode, the boosted score crossing threshold
        results in DENY — the response is blocked entirely."""
        agent = _make_agent({"risk_level": 1, "reasoning": ""})
        hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)

        output = await hook.execute(_make_narration_hook_input())

        assert output.permission_decision == PermissionDecision.DENY
        assert hook.last_risk_level == 3

    @pytest.mark.asyncio
    async def test_clean_narration_passes_unchanged_audit_score(self):
        """Tool result was successful → no narration boost → low
        audit risk preserved → ALLOW."""
        agent = _make_agent({"risk_level": 1, "reasoning": "Normal"})
        hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)

        clean_input = _make_narration_hook_input(
            tool_results=[{
                "tool_call_id": "tc-1", "name": "save_fact",
                "result": {"status": "ok"},
            }],
        )
        output = await hook.execute(clean_input)

        assert output.permission_decision == PermissionDecision.ALLOW
        assert hook.last_risk_level == 1
        assert hook.last_narration_verdict.risk_boost == 0

    @pytest.mark.asyncio
    async def test_no_narration_fields_preserves_legacy_behavior(self):
        """Hook fed pre-0.9-shaped HookInput (no pre_tool_prose,
        no tool_results) must behave exactly as the LLM-only
        audit did before — verdict has zero boost, score equals
        the LLM-only score."""
        agent = _make_agent({"risk_level": 2, "reasoning": "Borderline"})
        hook = ResponseAuditHook(agent=agent, mode="warn", risk_threshold=3)

        output = await hook.execute(_make_hook_input("This is a longer response from the agent."))

        assert hook.last_narration_verdict.risk_boost == 0
        assert hook.last_risk_level == 2
        assert output.permission_decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_narration_violation_fires_when_llm_audit_unavailable(self):
        """The deterministic check is the load-bearing piece for
        compliance: if the LLM audit raises (rate-limited, network
        out, model unavailable), the narration violation MUST still
        elevate risk and apply the configured mode's policy."""
        agent = _make_agent()
        agent.llm_service.get_audit_response = AsyncMock(
            side_effect=RuntimeError("audit LLM down")
        )
        hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=2)

        output = await hook.execute(_make_narration_hook_input())

        # Narration boost = 2 ≥ threshold 2 → strict denies.
        assert output.permission_decision == PermissionDecision.DENY
        assert hook.last_risk_level == 2
        assert "narration" in output.permission_reason.lower() or "save_fact" in output.permission_reason

    @pytest.mark.asyncio
    async def test_narration_clean_when_llm_audit_unavailable_falls_through_to_allow(self):
        """LLM down + narration check clean → existing skip-on-error
        path retained. Behavior matches pre-Wave-5D expectations for
        the no-violation case."""
        agent = _make_agent()
        agent.llm_service.get_audit_response = AsyncMock(
            side_effect=RuntimeError("audit LLM down")
        )
        hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)

        clean_input = _make_narration_hook_input(
            tool_results=[{
                "tool_call_id": "tc-1", "name": "save_fact",
                "result": {"status": "ok"},
            }],
        )
        output = await hook.execute(clean_input)

        assert output.permission_decision == PermissionDecision.ALLOW
        assert "error" in output.permission_reason.lower()


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
    async def test_feature_warn_mode_provides_hook(self):
        """In warn mode, initialize should create a hook available via get_hooks()."""
        agent = _make_agent()
        with patch.dict("os.environ", {"KESTREL_RESPONSE_AUDIT_MODE": "warn"}, clear=False):
            feature = ResponseAuditFeature(agent)
            await feature.initialize()

        assert feature._hook is not None
        hooks = feature.get_hooks()
        assert len(hooks) == 1
        assert hooks[0] is feature._hook

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
