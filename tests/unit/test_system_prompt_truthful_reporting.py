"""Regression: the base system prompt must carry a GENERAL truthful-tool-
reporting rule, not one quarantined to cryptographic data (#1925).

Dogfooding showed agents confabulate tool success for unavailable/disabled
tools (a fabricated `echo` stdout; a made-up GitHub "network escalation").
The anti-fabrication rule existed but was scoped under a "CRYPTOGRAPHIC DATA"
heading, so the model didn't apply it to ordinary tool output. This test locks
in the generalized rule.
"""

from pathlib import Path

import kestrel_sovereign

_PROMPT = (
    Path(kestrel_sovereign.__file__).parent / "prompts" / "system_prompt.md"
)


def _text() -> str:
    return _PROMPT.read_text(encoding="utf-8")


def test_base_prompt_has_general_truthful_tool_reporting_rule():
    text = _text().lower()
    # The crisp, tool-agnostic instruction must be present...
    assert "report tool outcomes faithfully" in text
    # ...and explicitly scoped to EVERY tool, not just sensitive/crypto ones.
    assert "every" in text and "not only sensitive" in text


def test_fabrication_rule_is_not_crypto_scoped_only():
    text = _text()
    # The hard-rule heading must be generalized to tool results, not limited
    # to cryptographic data (which is now framed as the worst case, not the
    # boundary).
    assert "NEVER FABRICATE TOOL RESULTS" in text
    assert "NEVER FABRICATE CRYPTOGRAPHIC DATA" not in text
    # The unavailable/disabled/error path must be called out so a readiness
    # failure or missing tool is reported, not narrated as success.
    lowered = text.lower()
    assert "unavailable" in lowered and "disabled" in lowered
