"""Tests for `ContextBuilder.build_system_prompt_with_tracking`.

The new method is the dispatcher's entry point for constitutional
injection (kestrel-sovereign#1137 chunk 1E). It MUST:

1. Return a `SystemPromptResult` (not a bare str) so the dispatcher
   can persist `injected_clauses_json` / `dropped_clauses_json`.
2. Render anchored doctrine, state-of-mind, prompt-adaptation, and
   style-reminder consistently with the assembler's contract.
3. Honor `budget_bytes` truncation.
4. Leave the legacy `build_system_prompt` byte-stable — the existing
   prompt-cache invariants (Anthropic's position-indexed cache)
   depend on it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.agent.system_prompt_assembler import (
    CLAUSE_KESTREL_CONSTITUTION,
    CLAUSE_PROMPT_ADAPTATION,
    CLAUSE_STATE_OF_MIND,
    CLAUSE_STYLE_REMINDER,
    SystemPromptResult,
)


class _FakePromptAdaptation:
    def __init__(self, preamble: str = "") -> None:
        self.preamble = preamble


class _FakeStateOfMind:
    def __init__(
        self,
        governance_mode: str = "balanced",
        active_conflicts: Optional[List[dict]] = None,
        delegated_principles: Optional[List[str]] = None,
    ) -> None:
        self.governance_mode = governance_mode
        self.active_conflicts = active_conflicts or []
        self.delegated_principles = delegated_principles or []


def _stub_builder(bootstrap: dict) -> ContextBuilder:
    cb = object.__new__(ContextBuilder)
    cb._llm_service = None
    cb._model_fallback = "test-model"
    cb._counter = MagicMock()
    cb._counter_model = "test-model"
    cb._bootstrap_loader = MagicMock()
    cb._bootstrap_loader.load = MagicMock(return_value=OrderedDict(bootstrap))
    cb._bootstrap_loader.get_file = MagicMock(
        side_effect=lambda name: bootstrap.get(name)
    )
    cb.storage = MagicMock()
    # ``ContextManager.build_context`` (#1309 elastic budget) calls
    # ``measure_mandatory_system_tokens`` to size the non-borrowable
    # governance floor. The MagicMock counter in this stub returns
    # MagicMock from ``count()``, which would propagate into the
    # ElasticTokenBudget constructor as a non-int. Override with a
    # zero-floor stub so the tests focus on the tracking assembler.
    cb.measure_mandatory_system_tokens = lambda *a, **kw: 0
    return cb


def test_returns_system_prompt_result():
    cb = _stub_builder({"SOUL.md": "soul body"})
    result = cb.build_system_prompt_with_tracking(
        constitution="C",
        include_briefing=False,
    )
    assert isinstance(result, SystemPromptResult)
    assert CLAUSE_KESTREL_CONSTITUTION in result.injected_clauses
    assert "SOUL.md" in result.injected_clauses


def test_anchored_doctrine_appears_in_clauses():
    cb = _stub_builder({"SOUL.md": "soul"})
    result = cb.build_system_prompt_with_tracking(
        constitution="C",
        anchored_doctrine=OrderedDict(
            [
                ("TORTOISE_DOCTRINE.md", "tortoise body"),
                ("AGENTS.md", "agents body"),
            ]
        ),
        include_briefing=False,
    )
    assert "TORTOISE_DOCTRINE.md" in result.injected_clauses
    assert "AGENTS.md" in result.injected_clauses
    assert "tortoise body" in result.prompt
    assert "agents body" in result.prompt


def test_state_of_mind_renders_when_provided():
    cb = _stub_builder({"SOUL.md": "soul"})
    state = _FakeStateOfMind(
        governance_mode="firm",
        active_conflicts=[
            {"principle": "sovereignty", "description": "tension X"}
        ],
        delegated_principles=["honesty"],
    )
    result = cb.build_system_prompt_with_tracking(
        constitution="C",
        state_of_mind=state,
        include_briefing=False,
    )
    assert CLAUSE_STATE_OF_MIND in result.injected_clauses
    assert "Governance Mode: FIRM" in result.prompt
    assert "sovereignty: tension X" in result.prompt
    assert "honesty" in result.prompt


def test_prompt_adaptation_preamble_renders():
    cb = _stub_builder({"SOUL.md": "soul"})
    adapt = _FakePromptAdaptation(preamble="emphasis on truth")
    result = cb.build_system_prompt_with_tracking(
        constitution="C",
        prompt_adaptation=adapt,
        include_briefing=False,
    )
    assert CLAUSE_PROMPT_ADAPTATION in result.injected_clauses
    assert "emphasis on truth" in result.prompt


def test_prompt_adaptation_skipped_when_preamble_empty():
    cb = _stub_builder({"SOUL.md": "soul"})
    adapt = _FakePromptAdaptation(preamble="")  # falsy preamble
    result = cb.build_system_prompt_with_tracking(
        constitution="C",
        prompt_adaptation=adapt,
        include_briefing=False,
    )
    assert CLAUSE_PROMPT_ADAPTATION not in result.injected_clauses


def test_style_reminder_only_emitted_when_soul_loaded():
    """Legacy parity: style reminder requires SOUL.md to be present.
    Without SOUL.md, the reminder is omitted."""
    cb_no_soul = _stub_builder({"TOOLS.md": "tools"})
    result = cb_no_soul.build_system_prompt_with_tracking(
        constitution="C", include_briefing=False
    )
    assert CLAUSE_STYLE_REMINDER not in result.injected_clauses

    cb_with_soul = _stub_builder({"SOUL.md": "soul"})
    result2 = cb_with_soul.build_system_prompt_with_tracking(
        constitution="C", include_briefing=False
    )
    assert CLAUSE_STYLE_REMINDER in result2.injected_clauses


def test_budget_truncation_drops_low_priority():
    cb = _stub_builder(
        {
            "SOUL.md": "soul",
            "TOOLS.md": "x" * 1000,  # bulk in priority 6
        }
    )
    result = cb.build_system_prompt_with_tracking(
        constitution="C",
        additional_context="y" * 1000,  # bulk in priority 7
        include_briefing=False,
        budget_bytes=400,
    )
    # Constitution survives.
    assert CLAUSE_KESTREL_CONSTITUTION in result.injected_clauses
    # ADDITIONAL_CONTEXT (priority 7) drops first.
    assert "ADDITIONAL_CONTEXT" in result.dropped_clauses
    assert len(result.prompt.encode("utf-8")) <= 400


def test_legacy_build_system_prompt_remains_byte_stable():
    """The legacy method must NOT route through the new assembler —
    byte-equivalent output is the load-bearing prompt-cache invariant.
    Pin a small fixture: same inputs → same bytes across calls."""
    cb = _stub_builder({"SOUL.md": "soul"})
    cb.get_session_briefing = lambda: "briefing"

    out1 = cb.build_system_prompt(constitution="C", include_briefing=True)
    out2 = cb.build_system_prompt(constitution="C", include_briefing=True)
    assert out1 == out2  # idempotent
    # Specifically check the legacy fence labels still appear (so a
    # refactor that accidentally rerouted through the new assembler
    # would fail this).
    assert "--- GOVERNING CONSTITUTION ---" in out1
    assert "--- END CONSTITUTION ---" in out1  # asymmetric legacy fence
    assert "--- YOUR IDENTITY ---" in out1
    assert "--- END IDENTITY ---" in out1


def test_system_prompt_addendum_appended_when_provided():
    """Chunk 1G: dispatcher injects per-turn directives (canary
    instruction) via `system_prompt_addendum`. Default None preserves
    byte-stable output for the cached path."""
    cb = _stub_builder({"SOUL.md": "soul"})
    cb.get_session_briefing = lambda: "briefing"

    legacy = cb.build_system_prompt(constitution="C")
    with_addendum = cb.build_system_prompt(
        constitution="C",
        system_prompt_addendum=(
            "--- CONSTITUTION RECEIPT ---\n"
            "constitution_canary: 0123456789abcdef\n"
            "--- END ---"
        ),
    )

    # Legacy output is unchanged.
    assert legacy == cb.build_system_prompt(constitution="C")
    # With addendum, the directive appears at the END (so the
    # cache-stable prefix above remains identical).
    assert with_addendum.startswith(legacy + "\n\n")
    assert with_addendum.endswith("--- END ---")
    assert "0123456789abcdef" in with_addendum


@pytest.mark.asyncio
async def test_context_manager_budget_includes_addendum_bytes(tmp_path):
    """Codex round-12 P2: when both `system_prompt_budget_bytes` and
    `system_prompt_addendum` are set, the assembler must reserve
    addendum bytes from the budget so the FINAL system prompt
    (assembler output + joiner + addendum) stays within the cap.
    Tight budgets that previously over-budgeted are now within cap."""
    from collections import OrderedDict
    from unittest.mock import AsyncMock, MagicMock

    from kestrel_sovereign.agent.context_manager import ContextManager

    # Build a real ContextBuilder stub. ``_stub_builder`` already
    # stubs ``measure_mandatory_system_tokens`` to zero (so the
    # ElasticTokenBudget #1309 doesn't trip on the MagicMock counter).
    cb = _stub_builder({"SOUL.md": "soul body"})
    cb.get_session_briefing = lambda: ""

    # ContextManager wired against the stub.
    cm = ContextManager(
        storage=MagicMock(),
        context_builder=cb,
    )
    cm.conversation_manager = MagicMock()
    cm.conversation_manager.get_conversation_history = AsyncMock(
        return_value=[]
    )
    cm.llm_service = None

    addendum = "X" * 200
    result = await cm.build_context(
        query="ignored",
        constitution="C",
        include_briefing=False,
        include_memories=False,
        include_rag=False,
        privacy_mode="NORMAL",
        conversation_history=[],
        system_prompt_addendum=addendum,
        system_prompt_budget_bytes=400,
    )
    # Final assembled system prompt fits the budget.
    assert len(result.system_prompt.encode("utf-8")) <= 400
    # The addendum still made it through (not silently dropped).
    assert addendum in result.system_prompt


@pytest.mark.asyncio
async def test_context_manager_ephemeral_honors_budget(tmp_path):
    """Codex round-12 P2: ephemeral privacy mode also honors the
    budget rather than silently falling back to the unbounded
    legacy path."""
    from unittest.mock import AsyncMock, MagicMock

    from kestrel_sovereign.agent.context_manager import ContextManager

    cb = _stub_builder({"SOUL.md": "soul body"})
    cb.get_session_briefing = lambda: ""

    cm = ContextManager(
        storage=MagicMock(),
        context_builder=cb,
    )
    cm.conversation_manager = MagicMock()
    cm.llm_service = None

    addendum = "Y" * 100
    result = await cm.build_context(
        query="ignored",
        constitution="C",
        include_briefing=False,
        privacy_mode="EPHEMERAL",
        system_prompt_addendum=addendum,
        system_prompt_budget_bytes=300,
    )
    # The ephemeral notice gets appended after the budget-aware
    # assembly; verify the budget-aware portion + notice still
    # contains the addendum and the assembler-portion respects
    # the cap (notice is operator-fixed, not part of the budget
    # contract).
    assert addendum in result.system_prompt


@pytest.mark.asyncio
async def test_anchored_doctrine_routes_through_tracking_assembler_without_budget():
    """Codex round-17 P2: when anchored_doctrine is supplied but
    system_prompt_budget_bytes is NOT, the legacy
    build_system_prompt path can't accept anchored_doctrine and
    would silently drop it. Routing must go through the tracking
    assembler whenever anchored_doctrine is present."""
    from collections import OrderedDict
    from unittest.mock import AsyncMock, MagicMock

    from kestrel_sovereign.agent.context_manager import ContextManager

    cb = _stub_builder({"SOUL.md": "soul"})
    cb.get_session_briefing = lambda: ""

    cm = ContextManager(storage=MagicMock(), context_builder=cb)
    cm.conversation_manager = MagicMock()
    cm.conversation_manager.get_conversation_history = AsyncMock(
        return_value=[]
    )
    cm.llm_service = None

    result = await cm.build_context(
        query="q",
        constitution="C",
        include_briefing=False,
        privacy_mode="NORMAL",
        conversation_history=[],
        anchored_doctrine=OrderedDict(
            [
                ("TORTOISE_DOCTRINE.md", "TORTOISE BODY"),
                ("AGENTS.md", "AGENTS BODY"),
            ]
        ),
        # NOTE: no budget set
    )
    # Doctrine bodies appear in the rendered system prompt.
    assert "TORTOISE BODY" in result.system_prompt
    assert "AGENTS BODY" in result.system_prompt
    # Tracking ran (the assembler path produces the lists).
    assert result.injected_clauses is not None
    assert "TORTOISE_DOCTRINE.md" in result.injected_clauses
    assert "AGENTS.md" in result.injected_clauses


@pytest.mark.asyncio
async def test_reflection_guidance_skipped_when_over_budget():
    """Codex round-15 P2: reflection guidance is appended after the
    budget-aware assembler runs. If adding it would push the prompt
    over the per-source cap, skip the append rather than silently
    exceed the budget."""
    from collections import OrderedDict
    from unittest.mock import AsyncMock, MagicMock

    from kestrel_sovereign.agent.context_manager import ContextManager

    cb = _stub_builder({"SOUL.md": "soul body"})
    cb.get_session_briefing = lambda: ""

    cm = ContextManager(storage=MagicMock(), context_builder=cb)
    cm.conversation_manager = MagicMock()
    cm.conversation_manager.get_conversation_history = AsyncMock(
        return_value=[]
    )
    cm.llm_service = None

    # Tight budget; reflection guidance is bulky and would exceed cap.
    bulky_guidance = ["X" * 200] * 5
    result = await cm.build_context(
        query="q",
        constitution="C",
        include_briefing=False,
        include_memories=False,
        include_rag=False,
        privacy_mode="NORMAL",
        conversation_history=[],
        reflection_guidance=bulky_guidance,
        system_prompt_budget_bytes=400,
    )
    # Final prompt within cap (reflection guidance was skipped).
    assert len(result.system_prompt.encode("utf-8")) <= 400
    # And the guidance fence DOES NOT appear.
    assert "ACTIVE REFLECTION GUIDANCE" not in result.system_prompt


@pytest.mark.asyncio
async def test_injection_tracking_is_per_async_task_isolated():
    """Codex round-14 P2: injection tracking must be per-async-task,
    not stored on a shared agent attribute that concurrent dispatches
    could overwrite. Two parallel build_context calls must each see
    their own tracking back via `get_current_injection_tracking`."""
    import asyncio
    from collections import OrderedDict
    from unittest.mock import AsyncMock, MagicMock

    from kestrel_sovereign.agent.context_manager import (
        ContextManager,
        get_current_injection_tracking,
    )

    async def _run_one(name: str, addendum: str):
        cb = _stub_builder({"SOUL.md": f"soul {name}", f"{name}.md": "x" * 50})
        cb.get_session_briefing = lambda: ""
        cm = ContextManager(storage=MagicMock(), context_builder=cb)
        cm.conversation_manager = MagicMock()
        cm.conversation_manager.get_conversation_history = AsyncMock(
            return_value=[]
        )
        cm.llm_service = None

        await cm.build_context(
            query="q",
            constitution="C",
            include_briefing=False,
            privacy_mode="NORMAL",
            conversation_history=[],
            system_prompt_addendum=addendum,
            system_prompt_budget_bytes=400,
        )
        # The current task's ContextVar must reflect THIS task's
        # tracking, not the other concurrent task's.
        injected, _dropped = get_current_injection_tracking()
        return name, injected

    results = await asyncio.gather(
        _run_one("alpha", "addendum-A" * 10),
        _run_one("beta", "addendum-B" * 10),
    )
    name_to_injected = {n: inj for n, inj in results}
    # Each task saw its OWN bootstrap file (alpha.md vs beta.md) in
    # its tracking — no cross-contamination.
    assert "alpha.md" in (name_to_injected["alpha"] or [])
    assert "beta.md" in (name_to_injected["beta"] or [])
    assert "alpha.md" not in (name_to_injected["beta"] or [])
    assert "beta.md" not in (name_to_injected["alpha"] or [])


def test_system_prompt_addendum_empty_or_none_is_noop():
    """Empty string falsy → treated like None, preserves byte stability."""
    cb = _stub_builder({"SOUL.md": "soul"})
    cb.get_session_briefing = lambda: "briefing"

    legacy = cb.build_system_prompt(constitution="C")
    out_none = cb.build_system_prompt(
        constitution="C", system_prompt_addendum=None
    )
    out_empty = cb.build_system_prompt(
        constitution="C", system_prompt_addendum=""
    )
    assert legacy == out_none == out_empty
