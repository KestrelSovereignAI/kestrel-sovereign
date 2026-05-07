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
