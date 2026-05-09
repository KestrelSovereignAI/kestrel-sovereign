"""Unit tests for the priority-ordered system-prompt assembler.

Pin the priority/drop semantics declared in
`docs/architecture/CONSTITUTION_INJECTION.md` §7 and the clause-name
contract used by `signal_log.injected_clauses_json` /
`dropped_clauses_json` (kestrel-sovereign#1137 chunk 1E).

The assembler is the pure-text core; `ContextBuilder.build_system_prompt_with_tracking`
delegates to it and is exercised separately.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

from kestrel_sovereign.agent.system_prompt_assembler import (
    AGENTS_FILENAME,
    CLAUSE_ADDITIONAL_CONTEXT,
    CLAUSE_KESTREL_CONSTITUTION,
    CLAUSE_PROMPT_ADAPTATION,
    CLAUSE_SESSION_BRIEFING,
    CLAUSE_STATE_OF_MIND,
    CLAUSE_STYLE_REMINDER,
    PRIORITY_ANCHORED_DOCTRINE,
    PRIORITY_CONSTITUTION,
    PRIORITY_OTHER_BOOTSTRAP,
    PRIORITY_SOUL,
    PRIORITY_STATE_OF_MIND,
    PRIORITY_STYLE_REMINDER,
    PRIORITY_TORTOISE_DOCTRINE,
    SystemPromptResult,
    TORTOISE_DOCTRINE_FILENAME,
    assemble_system_prompt,
    section_name_for_anchored_file,
)


# ---------------------------------------------------------------------------
# section_name_for_anchored_file
# ---------------------------------------------------------------------------


def test_section_name_strips_md_and_uppercases():
    assert section_name_for_anchored_file("AGENTS.md") == "AGENTS"
    assert (
        section_name_for_anchored_file("TORTOISE_DOCTRINE.md")
        == "TORTOISE DOCTRINE"
    )
    assert (
        section_name_for_anchored_file("docs/principles/KESTREL.md")
        == "KESTREL"
    )


# ---------------------------------------------------------------------------
# Minimal assembly
# ---------------------------------------------------------------------------


def test_assemble_minimal_has_constitution_only():
    """Bare assembly with no extras: just the constitution clause."""
    result = assemble_system_prompt(
        constitution="hello",
        bootstrap_files=OrderedDict(),
    )
    assert isinstance(result, SystemPromptResult)
    assert result.injected_clauses == [CLAUSE_KESTREL_CONSTITUTION]
    assert result.dropped_clauses == []
    assert "GOVERNING CONSTITUTION" in result.prompt
    assert "hello" in result.prompt


def test_assemble_full_combination_clause_order():
    """All clause types present — verify naming and emission order."""
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict(
            [
                ("SOUL.md", "soul"),
                ("TOOLS.md", "tools"),
                ("USER.md", "user"),
            ]
        ),
        anchored_doctrine=OrderedDict(
            [
                ("TORTOISE_DOCTRINE.md", "T"),
                ("AGENTS.md", "A"),
            ]
        ),
        session_briefing="briefing body",
        prompt_adaptation_preamble="preamble body",
        state_of_mind_block="--- STATE OF MIND ---\nbody\n--- END ---",
        style_reminder="--- REMINDER ---\nbody\n--- END ---",
        additional_context="extra",
    )
    assert result.injected_clauses == [
        "SOUL.md",
        "TOOLS.md",
        "USER.md",
        CLAUSE_SESSION_BRIEFING,
        CLAUSE_PROMPT_ADAPTATION,
        TORTOISE_DOCTRINE_FILENAME,
        AGENTS_FILENAME,
        CLAUSE_KESTREL_CONSTITUTION,
        CLAUSE_STATE_OF_MIND,
        CLAUSE_STYLE_REMINDER,
        CLAUSE_ADDITIONAL_CONTEXT,
    ]
    assert result.dropped_clauses == []


def test_assemble_skips_heartbeat_md():
    """HEARTBEAT.md is loaded by the heartbeat runner separately.
    The assembler must not emit it."""
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict(
            [("HEARTBEAT.md", "heartbeat body"), ("SOUL.md", "soul")]
        ),
    )
    assert "HEARTBEAT.md" not in result.injected_clauses
    assert "heartbeat body" not in result.prompt


def test_assemble_skips_agents_when_anchored_supplies_it():
    """AGENTS.md anchored doctrine takes precedence over the bootstrap
    loader's copy. Avoids duplicating the section."""
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict([("AGENTS.md", "from bootstrap")]),
        anchored_doctrine=OrderedDict([("AGENTS.md", "from anchor")]),
    )
    # AGENTS.md from bootstrap is skipped — only the anchored copy
    # survives. The clause appears exactly once in injected_clauses.
    assert result.injected_clauses.count("AGENTS.md") == 1
    assert "from bootstrap" not in result.prompt
    assert "from anchor" in result.prompt


def test_assemble_includes_agents_from_bootstrap_when_no_anchor():
    """Backwards-compat: if no anchored doctrine is supplied,
    AGENTS.md from bootstrap iteration is still emitted."""
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict([("AGENTS.md", "from bootstrap")]),
    )
    assert "AGENTS.md" in result.injected_clauses
    assert "from bootstrap" in result.prompt


# ---------------------------------------------------------------------------
# Fence labels
# ---------------------------------------------------------------------------


def test_soul_fence_uses_your_identity_label():
    """Legacy convention: SOUL.md → `--- YOUR IDENTITY ---`."""
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict([("SOUL.md", "soul body")]),
    )
    assert "--- YOUR IDENTITY ---" in result.prompt
    assert "--- END YOUR IDENTITY ---" in result.prompt


def test_other_bootstrap_fence_uses_uppercase_basename():
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict([("TOOLS.md", "tools body")]),
    )
    assert "--- TOOLS ---" in result.prompt
    assert "--- END TOOLS ---" in result.prompt


def test_anchored_doctrine_fence_uses_basename_with_underscores_replaced():
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict(),
        anchored_doctrine=OrderedDict(
            [("TORTOISE_DOCTRINE.md", "tortoise body")]
        ),
    )
    assert "--- TORTOISE DOCTRINE ---" in result.prompt
    assert "--- END TORTOISE DOCTRINE ---" in result.prompt


# ---------------------------------------------------------------------------
# Budget-aware truncation
# ---------------------------------------------------------------------------


def test_no_truncation_when_under_budget():
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict(),
        budget_bytes=10_000,
    )
    assert result.dropped_clauses == []


def test_no_truncation_when_budget_is_none():
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict([("TOOLS.md", "x" * 5000)]),
        budget_bytes=None,
    )
    assert result.dropped_clauses == []


def test_truncation_drops_highest_priority_number_first():
    """Drop priority 7 (style reminder, additional context) before 6."""
    result = assemble_system_prompt(
        constitution="C" * 50,
        bootstrap_files=OrderedDict([("TOOLS.md", "T" * 200)]),
        style_reminder="--- REM ---\n" + ("R" * 200),
        additional_context="A" * 200,
        budget_bytes=400,
    )
    # Style reminder + additional context (priority 7) drop first.
    # TOOLS.md (priority 6) drops only if still over budget.
    dropped_set = set(result.dropped_clauses)
    assert CLAUSE_STYLE_REMINDER in dropped_set
    assert CLAUSE_ADDITIONAL_CONTEXT in dropped_set
    # Constitution survives.
    assert CLAUSE_KESTREL_CONSTITUTION in result.injected_clauses
    assert len(result.prompt.encode("utf-8")) <= 400


def test_truncation_within_priority_drops_latest_emit_first():
    """Two entries at the same priority (other bootstrap): the later-
    emitted one drops first. Pin the deterministic order."""
    big = "X" * 1000
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict(
            [
                ("TOOLS.md", big),
                ("USER.md", big),
            ]
        ),
        budget_bytes=1200,  # fits ~one big bootstrap clause + constitution
    )
    # USER.md (later in emission order) drops before TOOLS.md.
    assert "USER.md" in result.dropped_clauses
    assert "TOOLS.md" in result.injected_clauses


def test_constitution_never_dropped_even_when_oversized():
    """If the constitution alone exceeds budget, keep it. The design
    treats integrity as load-bearing; an oversized constitution is
    an operator-config problem, not something to silently truncate."""
    result = assemble_system_prompt(
        constitution="C" * 10_000,
        bootstrap_files=OrderedDict([("TOOLS.md", "x" * 100)]),
        budget_bytes=1000,
    )
    assert CLAUSE_KESTREL_CONSTITUTION in result.injected_clauses
    assert "TOOLS.md" in result.dropped_clauses
    assert len(result.prompt.encode("utf-8")) > 1000


def test_drop_order_in_dropped_clauses_is_drop_time_order():
    """`dropped_clauses` reflects the order the budget loop removed
    items (highest priority number first), not emission order. Useful
    forensics: an auditor sees which clause would have been included
    next if the budget were larger."""
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict([("TOOLS.md", "x" * 500)]),
        style_reminder="x" * 500,
        additional_context="x" * 500,
        budget_bytes=200,
    )
    # Priority 7 entries drop first, in reverse emission order.
    # additional_context emitted after style_reminder in our order →
    # pops first.
    style_idx = result.dropped_clauses.index(CLAUSE_STYLE_REMINDER)
    addl_idx = result.dropped_clauses.index(CLAUSE_ADDITIONAL_CONTEXT)
    assert addl_idx < style_idx
    # TOOLS.md (priority 6) drops AFTER both priority-7 entries.
    tools_idx = result.dropped_clauses.index("TOOLS.md")
    assert style_idx < tools_idx


def test_truncation_records_byte_size_correctly_with_unicode():
    """Multi-byte UTF-8 characters count by encoded length, not str
    length, so the budget is enforced against actual transmitted bytes."""
    # 100 emoji = 400 UTF-8 bytes (4 bytes each) but only 100 codepoints.
    big_unicode = "🐢" * 100
    result = assemble_system_prompt(
        constitution="C",
        bootstrap_files=OrderedDict([("TOOLS.md", big_unicode)]),
        budget_bytes=300,
    )
    # TOOLS.md drops because its encoded size > 300 even though
    # str-len would fit under 300.
    assert "TOOLS.md" in result.dropped_clauses
    assert len(result.prompt.encode("utf-8")) <= 300


# ---------------------------------------------------------------------------
# Priority value sanity (regression guard against accidental shuffling)
# ---------------------------------------------------------------------------


def test_priority_ladder_is_strictly_increasing():
    """The drop algorithm depends on priorities being a strict
    ladder: lower number = kept first. Pin the exact constants so a
    reorder doesn't silently flip drop semantics."""
    assert (
        PRIORITY_CONSTITUTION
        < PRIORITY_TORTOISE_DOCTRINE
        < PRIORITY_ANCHORED_DOCTRINE
        < PRIORITY_SOUL
        < PRIORITY_STATE_OF_MIND
        < PRIORITY_OTHER_BOOTSTRAP
        < PRIORITY_STYLE_REMINDER
    )
    # Constitution is canonically priority 1 — never dropped.
    assert PRIORITY_CONSTITUTION == 1
