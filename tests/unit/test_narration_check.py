"""
Wave 5D — deterministic narration check (#1042 layer 3).

Pure-Python tests of ``analyze_narration``: exercises the regex
boundary cases, ToolResult / legacy / ambiguous result envelopes, and
the no-violation cases that must NOT elevate audit risk.
"""
import pytest

from kestrel_sovereign.security.narration_check import (
    analyze_narration,
    NarrationVerdict,
)


class TestNoViolation:
    """Cases where the heuristic must NOT flag the response."""

    def test_no_pre_tool_prose_returns_clean(self):
        v = analyze_narration(None, [{"name": "x", "result": {"status": "ok"}}])
        assert v.risk_boost == 0
        assert v.reasoning == ""

    def test_empty_pre_tool_prose_returns_clean(self):
        v = analyze_narration("", [{"name": "x", "result": {"status": "ok"}}])
        assert v.risk_boost == 0

    def test_no_tool_results_returns_clean(self):
        # Without any tools fired, there is no pre-tool/post-tool
        # distinction — the streamed text IS the answer.
        v = analyze_narration("Saved your favorite color.", None)
        assert v.risk_boost == 0
        v2 = analyze_narration("Saved your favorite color.", [])
        assert v2.risk_boost == 0

    def test_present_progressive_hedge_is_honest(self):
        """The TOOL_HONESTY_SYSTEM_PROMPT (#1043) directs the agent
        toward 'Saving that now...' phrasing during tool use. That
        MUST NOT trigger a violation."""
        v = analyze_narration(
            "Saving that now... let me check the result.",
            [{"name": "save", "result": {"status": "error", "error": "boom"}}],
        )
        assert v.risk_boost == 0

    def test_past_tense_with_successful_tool_result_is_honest(self):
        """If every tool returned success, a past-tense claim is
        consistent with the observed outcome — not a violation."""
        v = analyze_narration(
            "Saved your favorite color.",
            [{"name": "save_fact", "result": {"status": "ok", "data": {}}}],
        )
        assert v.risk_boost == 0

    def test_legacy_success_envelope_passes(self):
        v = analyze_narration(
            "Done.",
            [{"name": "x", "result": {"success": True, "id": 1}}],
        )
        assert v.risk_boost == 0

    def test_past_tense_far_into_prose_is_skipped(self):
        """The check only scans the head of the prose. A `saved`
        token deep in the text is more likely to refer to past state
        than the about-to-run tool. Avoids false positives."""
        long_prose = (
            "I'm going to look into your request and check several "
            "options for you. " * 30  # >>400 chars before the keyword
        ) + "saved your favorite color."
        v = analyze_narration(
            long_prose,
            [{"name": "x", "result": {"status": "error"}}],
        )
        assert v.risk_boost == 0


class TestViolation:
    """Cases that MUST trigger an audit-risk boost."""

    def test_saved_followed_by_failed_tool_result(self):
        """The canonical #1042 reproduction: agent says 'Saved' then
        the tool returns no positive confirmation."""
        v = analyze_narration(
            "Saved: your favorite color is teal.",
            [
                {"name": "save_fact", "result": {"status": "error", "error": "no store"}},
            ],
        )
        assert v.risk_boost == 2
        assert v.offending_verb == "saved"
        assert v.offending_tool == "save_fact"
        assert "save_fact" in v.reasoning

    def test_legacy_envelope_failure_triggers(self):
        v = analyze_narration(
            "Done!",
            [{"name": "act", "result": {"success": False, "error": "fail"}}],
        )
        assert v.risk_boost == 2
        assert v.offending_verb == "done"

    def test_partial_status_treated_as_failure(self):
        """ToolResult.PARTIAL means some sub-step failed; the agent
        cannot honestly claim past-tense success against partial."""
        v = analyze_narration(
            "Stored that for you.",
            [{"name": "memory_save", "result": {"status": "partial"}}],
        )
        assert v.risk_boost == 2
        assert v.offending_tool == "memory_save"

    def test_ambiguous_envelope_treated_as_failure(self):
        """The honesty doctrine: don't claim success without explicit
        confirmation. An envelope with no status, no success flag,
        and no error is ambiguous → for-audit-purposes treated as
        non-success so confident past-tense language flags."""
        v = analyze_narration(
            "Sent the message.",
            [{"name": "send", "result": {"data": {}}}],
        )
        assert v.risk_boost == 2

    def test_multiple_tools_first_failure_surfaced(self):
        """If two tools were called and only one failed, the verdict
        names the first failing tool — sufficient to audit."""
        v = analyze_narration(
            "Updated your record.",
            [
                {"name": "load", "result": {"status": "ok"}},
                {"name": "write", "result": {"status": "error"}},
            ],
        )
        assert v.risk_boost == 2
        assert v.offending_tool == "write"

    def test_leading_acknowledgement_does_not_hide_violation(self):
        """`Sure! Saved.` should still match — the agent loves to
        prefix confident claims with 'Sure!'."""
        v = analyze_narration(
            "Sure! Saved your preference.",
            [{"name": "save", "result": {"status": "error"}}],
        )
        assert v.risk_boost == 2
        assert v.offending_verb == "saved"

    def test_non_dict_result_treated_as_failure(self):
        """Defensive: if a caller passes a result that isn't a dict
        (None, string, etc.), we cannot verify success and the
        confident claim flags."""
        v = analyze_narration(
            "Saved.",
            [{"name": "x", "result": None}],
        )
        assert v.risk_boost == 2
        v2 = analyze_narration(
            "Saved.",
            [{"name": "x", "result": "string return"}],
        )
        assert v2.risk_boost == 2


class TestLegacySuccessEdgeCases:
    """Codex P2 of #1076 — ``is not True`` (rather than ``is False``)
    handling means non-True legacy values are treated as failure."""

    def test_success_none_is_failure(self):
        v = analyze_narration(
            "Saved.",
            [{"name": "x", "result": {"success": None}}],
        )
        assert v.risk_boost == 2

    def test_success_string_false_is_failure(self):
        """Stringly-typed legacy callers used 'false' / 0 / etc. The
        old ``is False`` check would have let these pass."""
        v = analyze_narration(
            "Saved.",
            [{"name": "x", "result": {"success": "false"}}],
        )
        assert v.risk_boost == 2
        v2 = analyze_narration(
            "Saved.",
            [{"name": "x", "result": {"success": 0}}],
        )
        assert v2.risk_boost == 2

    def test_success_literal_true_remains_success(self):
        v = analyze_narration(
            "Saved.",
            [{"name": "x", "result": {"success": True}}],
        )
        assert v.risk_boost == 0


class TestSummarizeForAudit:
    """``summarize_tool_result_for_audit`` keeps only the audit-
    relevant envelope shape — codex P2 of #1076."""

    def test_strips_data_payload(self):
        from kestrel_sovereign.security.narration_check import (
            summarize_tool_result_for_audit,
        )
        full = {
            "status": "ok",
            "data": {"secret": "ssn=123-45-6789", "huge": "x" * 100_000},
            "confirmation": "Saved fact #42",
        }
        s = summarize_tool_result_for_audit(full)
        assert s == {"status": "ok"}
        assert "data" not in s
        assert "confirmation" not in s

    def test_keeps_legacy_success_field(self):
        from kestrel_sovereign.security.narration_check import (
            summarize_tool_result_for_audit,
        )
        s = summarize_tool_result_for_audit({"success": False, "data": [1, 2, 3]})
        assert s == {"success": False}

    def test_caps_error_string_to_500_chars(self):
        from kestrel_sovereign.security.narration_check import (
            summarize_tool_result_for_audit,
        )
        long_err = "boom " * 500  # 2500 chars
        s = summarize_tool_result_for_audit({"status": "error", "error": long_err})
        assert s["status"] == "error"
        assert len(s["error"]) == 500

    def test_non_dict_coerced_to_opaque_unknown_envelope(self):
        """Codex re-review of #1076: tools that return raw primitives
        (string file contents, search snippets, etc.) MUST NOT leak
        verbatim through tool_results. The summarizer coerces to an
        opaque ``{status: unknown}`` envelope; analyze_narration
        treats unknown/non-dict as failure either way so the audit
        verdict is preserved."""
        from kestrel_sovereign.security.narration_check import (
            summarize_tool_result_for_audit,
        )
        # Each leak-prone primitive becomes the opaque envelope:
        assert summarize_tool_result_for_audit(
            "ssn=123-45-6789, dob=1980-01-01"
        ) == {"status": "unknown"}
        assert summarize_tool_result_for_audit(None) == {"status": "unknown"}
        assert summarize_tool_result_for_audit(42) == {"status": "unknown"}
        # Lists / tuples (search-results-style) likewise:
        assert summarize_tool_result_for_audit(
            [{"hit": "secret"}, {"hit": "more"}]
        ) == {"status": "unknown"}

    def test_non_dict_summary_still_drives_failure_verdict(self):
        """Audit verdict equivalence: a non-dict input through the
        summarizer + analyze_narration produces the same risk_boost
        as the non-dict input direct, so the privacy fix doesn't
        weaken the audit."""
        from kestrel_sovereign.security.narration_check import (
            summarize_tool_result_for_audit,
        )
        v_direct = analyze_narration(
            "Saved.", [{"name": "x", "result": "raw secret"}],
        )
        slim = summarize_tool_result_for_audit("raw secret")
        v_slim = analyze_narration(
            "Saved.", [{"name": "x", "result": slim}],
        )
        assert v_direct.risk_boost == v_slim.risk_boost == 2

    def test_summary_still_supports_narration_check(self):
        """The whole point: the slim envelope must still drive the
        same verdict the full envelope did."""
        from kestrel_sovereign.security.narration_check import (
            summarize_tool_result_for_audit,
        )
        full = {"status": "error", "error": "boom", "data": {"big": "x" * 50_000}}
        slim = summarize_tool_result_for_audit(full)
        v_full = analyze_narration("Saved.", [{"name": "x", "result": full}])
        v_slim = analyze_narration("Saved.", [{"name": "x", "result": slim}])
        assert v_full.risk_boost == v_slim.risk_boost == 2


class TestVerdictShape:
    def test_clean_verdict_has_no_offender_fields(self):
        v = analyze_narration(None, None)
        assert isinstance(v, NarrationVerdict)
        assert v.offending_verb is None
        assert v.offending_tool is None

    def test_dataclass_is_frozen(self):
        v = analyze_narration(None, None)
        with pytest.raises(Exception):
            v.risk_boost = 99  # frozen dataclass → can't mutate


# ---------------------------------------------------------------------------
# Escalation-attribution check (#1563 wire-up)
# ---------------------------------------------------------------------------

from kestrel_sovereign.security.narration_check import (
    check_escalation_attribution,
)


def _failing_tool_result(error: str = ""):
    return [
        {
            "name": "shell",
            "result": {"status": "error", "error": error},
        }
    ]


def _user_denied_audit_row():
    return {
        "feature": "shell",
        "tool": "shell",
        "action": "tool_execution",
        "decision": "user_denied",
        "user_choice": "user_denied",
    }


class TestEscalationAttributionNoViolation:
    """Responses that do NOT make a user-denial claim must pass."""

    def test_no_response_text(self):
        v = check_escalation_attribution(
            "", _failing_tool_result("anything"),
        )
        assert v.risk_boost == 0

    def test_response_without_user_denial_wording(self):
        v = check_escalation_attribution(
            "The shell command failed because the binary is missing.",
            _failing_tool_result("binary not found"),
        )
        assert v.risk_boost == 0

    def test_response_makes_audit_backed_user_denial_claim(self):
        """When the audit DOES carry a user_denied row, the wording
        is honest and must pass."""
        v = check_escalation_attribution(
            "The user explicitly denied the escalation at the prompt.",
            _failing_tool_result("Rejected(\"rejected by user\")"),
            recent_decisions=[_user_denied_audit_row()],
            tool_name="shell", feature_name="shell",
        )
        assert v.risk_boost == 0


class TestEscalationAttributionViolation:
    """The #1563 reproduction: response narrates user denial, no audit."""

    def test_codex_rejection_with_empty_audit_flags(self):
        v = check_escalation_attribution(
            "I attempted to run the command but the user rejected the escalation.",
            _failing_tool_result(
                "CreateProcess { message: \"Rejected(\\\"rejected by user\\\")\" }",
            ),
            recent_decisions=[],
        )
        assert v.risk_boost == 2
        assert "sandbox_blocked" in v.reasoning

    def test_rejected_by_user_wording_without_audit_flags(self):
        v = check_escalation_attribution(
            "The command was rejected by user.",
            _failing_tool_result("any error"),
            recent_decisions=[],
        )
        assert v.risk_boost == 2

    def test_user_denied_wording_without_audit_flags(self):
        v = check_escalation_attribution(
            "user denied the escalation request.",
            _failing_tool_result("approval timed out"),
            recent_decisions=[],
        )
        assert v.risk_boost == 2
        # Classifier should label this policy_blocked, not user_denied.
        assert "policy_blocked" in v.reasoning


class TestEscalationAttributionEvidence:
    """Edge cases around the audit-backed/unbacked decision."""

    def test_audit_row_for_wrong_tool_does_not_back_claim(self):
        """A user_denied row for ``other_tool`` cannot vouch for a
        ``shell`` denial claim."""
        wrong_tool_row = _user_denied_audit_row()
        wrong_tool_row["tool"] = "other_tool"
        v = check_escalation_attribution(
            "The user explicitly denied this command.",
            _failing_tool_result("any error"),
            recent_decisions=[wrong_tool_row],
            tool_name="shell", feature_name="shell",
        )
        assert v.risk_boost == 2

    def test_per_scope_audit_choice_backs_claim(self):
        """Web-UI per-scope user_choice (once/session/always) on a
        deny row counts as a real user denial."""
        for choice in ("once", "session", "always"):
            row = _user_denied_audit_row()
            row["decision"] = "denied"
            row["user_choice"] = choice
            v = check_escalation_attribution(
                "The user denied the escalation.",
                _failing_tool_result("anything"),
                recent_decisions=[row],
                tool_name="shell", feature_name="shell",
            )
            assert v.risk_boost == 0, (
                f"choice={choice!r} should back the claim"
            )

    def test_no_tool_results_still_classifies_as_unconfirmed(self):
        """The agent narrated user denial with no tool result visible
        — classifier defaults to UNCONFIRMED, still a violation."""
        v = check_escalation_attribution(
            "rejected by user",
            tool_results=None,
            recent_decisions=[],
        )
        assert v.risk_boost == 2
        assert "unconfirmed" in v.reasoning


class TestEscalationAttributionCodexRound1Fixes:
    """Regression coverage for the codex round-1 findings."""

    def test_passive_denied_by_user_wording_caught(self):
        """codex P2 r1: 'denied by user' was previously missed."""
        v = check_escalation_attribution(
            "The command was denied by user.",
            _failing_tool_result("any error"),
            recent_decisions=[],
        )
        assert v.risk_boost == 2

    def test_passive_declined_by_the_user_wording_caught(self):
        v = check_escalation_attribution(
            "The escalation was declined by the user.",
            _failing_tool_result("any error"),
            recent_decisions=[],
        )
        assert v.risk_boost == 2

    def test_user_refused_wording_caught(self):
        v = check_escalation_attribution(
            "user refused the escalation request",
            _failing_tool_result("any error"),
            recent_decisions=[],
        )
        assert v.risk_boost == 2

    def test_uses_latest_failing_tool_result_not_earliest(self):
        """codex P1 r1 #2: when multiple failing tools are present,
        the check must classify the LATEST one — the response refers
        to the most recent failure, not an earlier audit-backed denial.
        """
        # Earlier failure: user denied this one (audit row present).
        # Later failure: sandbox refusal with no audit backing.
        v = check_escalation_attribution(
            "The escalation was rejected by user.",
            tool_results=[
                {"name": "tool_a", "result": {
                    "status": "error",
                    "error": "Rejected(\"rejected by user\")",
                }},
                {"name": "tool_b", "result": {
                    "status": "error",
                    "error": "sandbox refused new escalation",
                }},
            ],
            recent_decisions=[{
                "feature": "shell", "tool": "tool_a",
                "decision": "user_denied", "user_choice": "user_denied",
            }],
        )
        # Classifier scopes audit to ``tool_b`` (the latest failure),
        # finds no audit row for it, classifies as sandbox_blocked,
        # flags the violation.
        assert v.risk_boost == 2
        assert "sandbox_blocked" in v.reasoning

    def test_derives_tool_scope_from_failing_result(self):
        """codex P1 r1 #1: the check must derive the tool name from
        the failing result so an unrelated user_denied row in the
        audit cannot corroborate the current narration.
        """
        v = check_escalation_attribution(
            "rejected by user",
            tool_results=[{
                "name": "shell",
                "result": {"status": "error", "error": "Rejected"},
            }],
            recent_decisions=[{
                "feature": "approval", "tool": "an_unrelated_tool",
                "decision": "user_denied", "user_choice": "user_denied",
            }],
        )
        # Audit row is for an unrelated tool → cannot back the claim.
        assert v.risk_boost == 2

    def test_codex_p2_r2_no_tool_results_with_unrelated_audit_still_flags(self):
        """codex P2 r2: when tool_results is None / empty AND an
        unrelated user_denied row exists in the audit, the audit
        lookup MUST be disabled — otherwise the unrelated row
        corroborates the narration silently.
        """
        v = check_escalation_attribution(
            "rejected by user",
            tool_results=None,
            recent_decisions=[{
                "feature": "approval", "tool": "an_unrelated_tool",
                "decision": "user_denied", "user_choice": "user_denied",
            }],
        )
        # Audit lookup was disabled because we had no tool scope to
        # match against → classifier defaults to UNCONFIRMED → flagged.
        assert v.risk_boost == 2
