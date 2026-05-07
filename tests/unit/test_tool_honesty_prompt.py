"""Tests for the tool-honesty system prompt addendum (issue #1042 Fix 1).

Asserts:
1. The constant exists, is non-trivial, and contains the load-bearing
   rules (regex assertions on stable phrases — minor copy edits are
   tolerated, but the rule semantics must remain).
2. ``append_security_addendum`` produces the documented order:
   base → INPUT SECURITY → HONESTY ABOUT TOOL USE.
3. Both production code paths (streaming + non-streaming) actually
   use the helper, so the addendum reaches every assembled system
   prompt. This is enforced via static import-graph + source
   inspection rather than booting an agent.

Cache-stability rationale: see
``input_guardrails.append_security_addendum`` docstring + issue #703.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kestrel_sovereign.security.input_guardrails import (
    ANTI_INJECTION_SYSTEM_PROMPT,
    TOOL_HONESTY_SYSTEM_PROMPT,
    append_security_addendum,
)


# Repo root, used by the static-source assertions below.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestToolHonestyPromptConstant:
    """The TOOL_HONESTY_SYSTEM_PROMPT constant is well-formed and
    contains the load-bearing rules from #1042."""

    def test_constant_is_non_trivial(self):
        """Constant exists, is a string, has reasonable length."""
        assert isinstance(TOOL_HONESTY_SYSTEM_PROMPT, str)
        # Empirically ~1.5 KB; assert >= 500 chars to catch accidental
        # truncation. Upper bound left soft so future rule clarifications
        # don't fail the test.
        assert len(TOOL_HONESTY_SYSTEM_PROMPT) > 500

    def test_constant_has_section_markers(self):
        """Begins with the section header so an LLM can recognize it
        as a distinct rule block; ends with the matching close marker."""
        assert "--- HONESTY ABOUT TOOL USE ---" in TOOL_HONESTY_SYSTEM_PROMPT
        assert "--- END HONESTY ABOUT TOOL USE ---" in TOOL_HONESTY_SYSTEM_PROMPT

    @pytest.mark.parametrize(
        "rule_keyword",
        [
            # Rule 1: don't claim success before observing.
            r"observed",
            r"runtime result",
            # Rule 2: hedged present-progressive language. The term may
            # be hyphenated and/or wrapped across lines in the rendered
            # prompt, so allow any dash/whitespace combination between
            # the two halves.
            r"present[-\s]+progressive",
            # Rule 3: ground reply in actual result, name silent failures.
            r"silent",
            # Rule 4: surface contradictions explicitly.
            r"contradict",
            # Rule 5: confident lies are worse than uncertainty.
            r"uncertainty",
        ],
    )
    def test_load_bearing_rule_keywords_present(self, rule_keyword):
        """Each rule's load-bearing phrase appears in the prompt.

        Phrasing can evolve; these regex assertions target stable
        keywords so minor copy edits don't break the test, but
        accidentally deleting a whole rule does.
        """
        assert re.search(
            rule_keyword,
            TOOL_HONESTY_SYSTEM_PROMPT,
            flags=re.IGNORECASE,
        ), f"missing load-bearing keyword '{rule_keyword}' in TOOL_HONESTY_SYSTEM_PROMPT"

    def test_does_not_reference_specific_customers(self):
        """The honesty layer is universal. No customer-specific text."""
        # Lower-case scan — guard against accidental future references.
        lowered = TOOL_HONESTY_SYSTEM_PROMPT.lower()
        assert "caprock" not in lowered
        assert "frinz" not in lowered


class TestAppendSecurityAddendum:
    """The helper that assembles the security + honesty addenda."""

    def test_includes_anti_injection_block(self):
        """ANTI_INJECTION_SYSTEM_PROMPT is appended."""
        out = append_security_addendum("BASE")
        assert ANTI_INJECTION_SYSTEM_PROMPT in out

    def test_includes_tool_honesty_block(self):
        """TOOL_HONESTY_SYSTEM_PROMPT is appended."""
        out = append_security_addendum("BASE")
        assert TOOL_HONESTY_SYSTEM_PROMPT in out

    def test_preserves_base_prompt(self):
        """The base system prompt appears at the start of the result."""
        base = "You are a constitutional agent. Identity: Kestrel."
        out = append_security_addendum(base)
        assert out.startswith(base)

    def test_order_is_base_then_anti_injection_then_honesty(self):
        """Order is load-bearing for prompt-cache stability (#703).

        Cache invalidates on prefix changes; we must keep the order
        ``base → INPUT SECURITY → HONESTY`` so adding future addenda
        only affects bytes after the existing prefix.
        """
        out = append_security_addendum("BASE")
        idx_base = out.index("BASE")
        idx_anti = out.index(ANTI_INJECTION_SYSTEM_PROMPT)
        idx_honesty = out.index(TOOL_HONESTY_SYSTEM_PROMPT)
        assert idx_base < idx_anti < idx_honesty, (
            "addendum order must be base → ANTI_INJECTION → TOOL_HONESTY "
            "for prompt-cache stability"
        )

    def test_idempotent_in_shape(self):
        """Calling with the same base returns byte-identical output —
        cache stability across turns within a session depends on this."""
        base = "stable base prompt"
        a = append_security_addendum(base)
        b = append_security_addendum(base)
        assert a == b


class TestProductionCallSites:
    """Both production paths assemble system prompts via the helper.

    Without these tests, a future refactor could quietly drop the
    honesty addendum from one path while leaving the other intact —
    a regression that's hard to spot in code review.
    """

    @pytest.mark.parametrize(
        "module_path",
        [
            "kestrel_sovereign/agent/streaming.py",
            "kestrel_sovereign/kestrel_agent.py",
        ],
    )
    def test_uses_append_security_addendum(self, module_path):
        """Both code paths import + call ``append_security_addendum``,
        and neither path directly references ``ANTI_INJECTION_SYSTEM_PROMPT``
        (the helper now owns that detail)."""
        src = (_REPO_ROOT / module_path).read_text(encoding="utf-8")
        assert "append_security_addendum" in src, (
            f"{module_path} should call append_security_addendum to assemble "
            "the system prompt's security + honesty addenda"
        )
        # The helper now owns the inline f"{...}\n{ANTI_INJECTION_SYSTEM_PROMPT}"
        # composition — call sites should not reference the constant directly.
        assert "ANTI_INJECTION_SYSTEM_PROMPT" not in src, (
            f"{module_path} references ANTI_INJECTION_SYSTEM_PROMPT directly; "
            "use append_security_addendum() instead — the helper is the "
            "single source of truth for assembly order (#703 cache stability)"
        )

    @pytest.mark.parametrize(
        "module_path",
        [
            "kestrel_sovereign/agent/streaming.py",
            "kestrel_sovereign/kestrel_agent.py",
        ],
    )
    def test_does_not_reassemble_addendum_inline(self, module_path):
        """Catch a future refactor that copies the f-string back inline."""
        src = (_REPO_ROOT / module_path).read_text(encoding="utf-8")
        # No call site should be inlining the anti-injection f-string.
        assert "ANTI_INJECTION_SYSTEM_PROMPT}" not in src
        assert "TOOL_HONESTY_SYSTEM_PROMPT}" not in src
