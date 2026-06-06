"""Tests for the escalation classifier (#1540 / #1563).

The acceptance criterion from #1540 is precise:

  > The formatter must not produce ``user rejected`` /
  > ``escalation rejected by user`` wording unless the recorded
  > outcome explicitly identifies a user denial.

These tests pin that contract: every classification path is
exercised, every narrative output is asserted to either contain or
not contain the user-denial wording. The Codex ``Rejected("rejected
by user")`` raw string — the exact source of the #1563 bug — is
explicitly verified to NOT produce a user-denial narration unless an
audit row backs it up.
"""

from __future__ import annotations

from typing import Mapping

import pytest

from kestrel_sovereign.llm.escalation_classifier import (
    EscalationDecision,
    EscalationOutcome,
    classify_escalation_failure,
    format_escalation_outcome,
)


def _audit_row(**kwargs) -> Mapping[str, str]:
    base = {
        "feature": "shell",
        "tool": "bash",
        "action": "tool_execution",
        "decision": "user_denied",
        "user_choice": "user_denied",
        "args_summary": "uv run pytest",
        "timestamp": "2026-06-06T00:00:00+00:00",
    }
    base.update(kwargs)
    return base


def test_audit_row_with_user_denied_decision_yields_user_denied():
    decision = classify_escalation_failure(
        "tool returned Rejected(\"rejected by user\")",
        recent_decisions=[_audit_row()],
        tool_name="bash",
        feature_name="shell",
    )
    assert decision.outcome is EscalationOutcome.USER_DENIED
    assert decision.evidence_source == "audit"


@pytest.mark.parametrize("choice", ["once", "session", "always"])
def test_audit_row_with_web_ui_per_scope_user_choice_yields_user_denied(choice):
    decision = classify_escalation_failure(
        "anything",
        recent_decisions=[_audit_row(
            decision="denied", user_choice=choice,
        )],
        tool_name="bash",
        feature_name="shell",
    )
    assert decision.outcome is EscalationOutcome.USER_DENIED


def test_audit_row_for_different_tool_does_not_attribute_denial():
    decision = classify_escalation_failure(
        "rejected by user",
        recent_decisions=[_audit_row(tool="some_other_tool")],
        tool_name="bash",
        feature_name="shell",
    )
    assert decision.outcome is EscalationOutcome.SANDBOX_BLOCKED


def test_audit_row_for_different_feature_does_not_attribute_denial():
    decision = classify_escalation_failure(
        "rejected by user",
        recent_decisions=[_audit_row(feature="other_feature")],
        tool_name="bash",
        feature_name="shell",
    )
    assert decision.outcome is EscalationOutcome.SANDBOX_BLOCKED


def test_codex_rejected_by_user_without_audit_is_sandbox_blocked():
    """The #1563 root case: Codex's raw ``Rejected("rejected by user")``
    string MUST classify as SANDBOX_BLOCKED when no audit row backs it.
    """
    decision = classify_escalation_failure(
        "CreateProcess { message: \"Rejected(\\\"rejected by user\\\")\" }",
        recent_decisions=[],
    )
    assert decision.outcome is EscalationOutcome.SANDBOX_BLOCKED
    assert decision.evidence_source == "raw_error"


def test_sandbox_seccomp_refusal_is_sandbox_blocked():
    decision = classify_escalation_failure(
        "Permission denied (seccomp capability filter)",
        recent_decisions=[],
    )
    assert decision.outcome is EscalationOutcome.SANDBOX_BLOCKED


def test_approval_timeout_is_policy_blocked():
    decision = classify_escalation_failure(
        "approval request timed out before any user decision",
        recent_decisions=[],
    )
    assert decision.outcome is EscalationOutcome.POLICY_BLOCKED


def test_auto_denied_is_policy_blocked():
    decision = classify_escalation_failure(
        "auto_denied: operator policy blocked",
        recent_decisions=[],
    )
    assert decision.outcome is EscalationOutcome.POLICY_BLOCKED


def test_binary_not_found_is_tooling_error():
    decision = classify_escalation_failure(
        "binary not found: codex",
        recent_decisions=[],
    )
    assert decision.outcome is EscalationOutcome.TOOLING_ERROR


def test_timeout_is_tooling_error():
    decision = classify_escalation_failure(
        "RPC error: turn timed out after 300s",
        recent_decisions=[],
    )
    assert decision.outcome is EscalationOutcome.TOOLING_ERROR


def test_unknown_error_with_no_audit_is_unconfirmed():
    decision = classify_escalation_failure(
        "tool returned a string we have never seen before",
        recent_decisions=[],
    )
    assert decision.outcome is EscalationOutcome.UNCONFIRMED


def test_no_recent_decisions_passed_at_all_falls_through_to_raw_error():
    decision = classify_escalation_failure(
        "rejected by user",
        recent_decisions=None,
    )
    assert decision.outcome is EscalationOutcome.SANDBOX_BLOCKED


def test_empty_raw_error_and_no_audit_is_unconfirmed():
    decision = classify_escalation_failure("", recent_decisions=[])
    assert decision.outcome is EscalationOutcome.UNCONFIRMED


def test_user_denied_narrative_does_say_user_denied():
    decision = EscalationDecision(
        outcome=EscalationOutcome.USER_DENIED,
        reason="audit row says so",
        evidence_source="audit",
    )
    text = format_escalation_outcome(decision, command="uv run pytest")
    assert "user explicitly denied" in text
    assert "uv run pytest" in text


@pytest.mark.parametrize("outcome,reason", [
    (EscalationOutcome.SANDBOX_BLOCKED, "sandbox refused"),
    (EscalationOutcome.POLICY_BLOCKED, "policy refused"),
    (EscalationOutcome.TOOLING_ERROR, "binary missing"),
    (EscalationOutcome.UNCONFIRMED, "no evidence"),
])
def test_non_user_denied_narrative_never_blames_user(outcome, reason):
    """For EVERY non-audit-backed outcome, the rendered narrative
    MUST NOT say the user denied anything. This is the headline
    contract from #1540."""
    decision = EscalationDecision(
        outcome=outcome, reason=reason, evidence_source="raw_error",
    )
    text = format_escalation_outcome(decision, command="uv run pytest").lower()
    forbidden_phrases = (
        "user denied", "user explicitly denied",
        "user rejected", "rejected by user",
    )
    for phrase in forbidden_phrases:
        assert phrase not in text, (
            f"narrative for {outcome} must not say {phrase!r}; got: {text!r}"
        )


def test_unconfirmed_narrative_warns_against_attributing_to_user():
    decision = EscalationDecision(
        outcome=EscalationOutcome.UNCONFIRMED,
        reason="no evidence", evidence_source="default",
    )
    text = format_escalation_outcome(decision)
    assert "could not be confirmed" in text.lower()
    assert "without audit evidence" in text.lower()


def test_end_to_end_1563_reproduction():
    """The exact #1563 scenario: auto-mode ``gh issue comment`` fails
    with Codex's ``Rejected("rejected by user")`` and the security
    audit has NO denial row. The narrative must NOT report a user
    denial.
    """
    raw = (
        "CreateProcess { message: \"Rejected(\\\"rejected by user\\\")\" }"
    )
    decision = classify_escalation_failure(
        raw,
        recent_decisions=[
            {
                "feature": "shell", "tool": "bash",
                "action": "tool_execution",
                "decision": "auto_allowed", "user_choice": None,
            },
            {
                "feature": "shell", "tool": "bash",
                "action": "tool_execution",
                "decision": "auto_allowed", "user_choice": None,
            },
        ],
        tool_name="bash", feature_name="shell",
    )
    assert decision.outcome is EscalationOutcome.SANDBOX_BLOCKED
    text = format_escalation_outcome(
        decision, command="gh issue comment 1560 --body ...",
    ).lower()
    assert "user denied" not in text
    assert "rejected by user" not in text
    assert "sandbox" in text or "approval plumbing" in text
