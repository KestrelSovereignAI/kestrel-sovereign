"""Equivalence + concurrency coverage for the canonical context-build plan.

#2523 decomposed ``ContextManager.build_context`` into stages and moved the
section-content vocabulary into ``kestrel_sovereign.agent.context_stages`` so
that production and dry-run status builds consume one definition and cannot
drift (#2534). These tests pin:

- the shared vocabulary's exact bytes (golden),
- the typed ``ContextAssembly`` / ``SectionResult`` per-build state,
- production ⇄ compatibility-adapter byte parity for dynamic wrapping,
- table-driven ``build_context`` behavior across EPHEMERAL / trivial /
  episodes / memories / RAG / none,
- durable-salvage success and fail-closed paths,
- concurrent builds not sharing mutable counters/results or injection tracking.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.agent import context_stages as cs
from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.agent.context_manager import (
    ContextManager,
    ContextResult,
    get_current_injection_tracking,
    reset_injection_tracking,
)
from kestrel_sovereign.agent.context_stages import (
    ContextBuildMode,
    SectionStatus,
)
from kestrel_sovereign.agent.salvage import SalvageWriteError


# ---------------------------------------------------------------------------
# Shared vocabulary — golden bytes (single definition consumed by both paths)
# ---------------------------------------------------------------------------


def test_wrap_memories_golden():
    assert cs.wrap_memories("BODY") == "<memories>\nBODY\n</memories>"


def test_wrap_documents_golden():
    assert cs.wrap_documents("BODY") == "<documents>\nBODY\n</documents>"


def test_assemble_dynamic_user_context_empty_is_blank():
    assert cs.assemble_dynamic_user_context([]) == ""


def test_assemble_dynamic_user_context_single_and_multi():
    one = cs.assemble_dynamic_user_context(["<memories>\nX\n</memories>"])
    assert one == "<retrieved_context>\n<memories>\nX\n</memories>\n</retrieved_context>"
    two = cs.assemble_dynamic_user_context(
        ["<memories>\nX\n</memories>", "<documents>\nY\n</documents>"]
    )
    assert two.startswith("<retrieved_context>\n")
    assert two.endswith("\n</retrieved_context>")
    assert "<memories>\nX\n</memories>\n<documents>\nY\n</documents>" in two


def test_empty_envelope_constant_matches_assembly_overhead():
    # The overhead measure_context_breakdown charges once must equal the
    # envelope assemble_dynamic_user_context produces with no inner bytes.
    assert cs.RETRIEVED_CONTEXT_EMPTY_ENVELOPE == "<retrieved_context>\n\n</retrieved_context>"


def test_count_helpers():
    assert cs.count_memory_blocks("[Memory 1] a [Memory 2] b") == 2
    assert cs.count_rag_chunks("[Document 1] a [Document 2] b") == 2
    # Falls back to Source: when no [Document markers present.
    assert cs.count_rag_chunks("Source: a\nSource: b") == 2
    assert cs.count_rag_chunks("nothing here") == 0


def test_reflection_guidance_block_golden():
    block = cs.build_reflection_guidance_block(["one", "two"])
    assert block == (
        "\n--- ACTIVE REFLECTION GUIDANCE ---\n"
        "Based on self-reflection, keep these insights in mind:\n"
        "- one\n- two\n"
        "--- END GUIDANCE ---"
    )


def test_ephemeral_notice_golden():
    assert cs.EPHEMERAL_NOTICE == (
        "--- EPHEMERAL MODE ACTIVE ---\n"
        "This conversation is not being recorded. "
        "No history or memories are available.\n"
        "--- END NOTICE ---"
    )


# ---------------------------------------------------------------------------
# Typed per-build state
# ---------------------------------------------------------------------------


def test_context_assembly_dynamic_user_context_is_computed_view():
    a = cs.ContextAssembly()
    assert a.dynamic_user_context == ""
    a.dynamic_blocks.append("<memories>\nM\n</memories>")
    assert a.dynamic_user_context == (
        "<retrieved_context>\n<memories>\nM\n</memories>\n</retrieved_context>"
    )
    # Dynamic context is a derived view — it never lives inside system_prompt.
    assert "M" not in a.system_prompt


def test_context_assembly_instances_are_independent():
    a = cs.ContextAssembly()
    b = cs.ContextAssembly()
    a.dynamic_blocks.append("x")
    a.warnings.append("w")
    a.memory_count = 3
    assert b.dynamic_blocks == []
    assert b.warnings == []
    assert b.memory_count == 0
    assert a.dynamic_blocks is not b.dynamic_blocks


def test_context_manager_has_no_shared_assembly_class_state():
    # Concurrency safety is structural: the mutable per-build state lives on
    # a per-call ContextAssembly local, never as ContextManager class state.
    for attr in ("system_prompt", "dynamic_blocks", "formatted_history"):
        assert not hasattr(ContextManager, attr)


# ---------------------------------------------------------------------------
# Shared section producers — the single definition the canonical planner
# consumes for both live and dry-run plans.
# ---------------------------------------------------------------------------


def test_build_memory_section_golden():
    sec = cs.build_memory_section(
        "[Memory 1] a\n[Memory 2] b", count_tokens=lambda s: len(s)
    )
    assert sec.name == "memories"
    assert sec.destination is cs.SectionDestination.DYNAMIC
    # tokens is the RAW block cost — the byte both budget gates charge.
    assert sec.tokens == len("[Memory 1] a\n[Memory 2] b")
    assert sec.items == 2
    assert sec.dynamic_block == "<memories>\n[Memory 1] a\n[Memory 2] b\n</memories>"
    # Defaults for the production-only rehearsal fields.
    assert sec.message_ids == ()
    assert sec.is_memory_block is False


def test_build_memory_section_passes_through_rehearsal_fields():
    sec = cs.build_memory_section(
        "m", count_tokens=lambda s: 1, message_ids=(7, 9), is_memory_block=True
    )
    assert sec.message_ids == (7, 9)
    assert sec.is_memory_block is True


def test_build_rag_section_golden():
    sec = cs.build_rag_section("[Document A] x", count_tokens=lambda s: len(s))
    assert sec.name == "rag"
    assert sec.destination is cs.SectionDestination.DYNAMIC
    assert sec.tokens == len("[Document A] x")
    assert sec.items == 1
    assert sec.dynamic_block == "<documents>\n[Document A] x\n</documents>"


def test_build_episode_section_golden():
    sec = cs.build_episode_section("EPBLOCK", 3, count_tokens=lambda s: len(s))
    assert sec.name == "episodes"
    assert sec.destination is cs.SectionDestination.SYSTEM
    assert sec.tokens == len("EPBLOCK")
    # items is the true len(episodes) count — never the legacy "**" heuristic.
    assert sec.items == 3
    assert sec.append_text == "EPBLOCK"


# ---------------------------------------------------------------------------
# Pure stage primitives
# ---------------------------------------------------------------------------


def test_finalize_section_is_noop_without_marker():
    # Legacy budgets lacking mark_section_finalized must not raise.
    cs.finalize_section(object(), "history")


def test_finalize_section_calls_marker():
    budget = MagicMock()
    cs.finalize_section(budget, "episodes")
    budget.mark_section_finalized.assert_called_once_with("episodes")


def test_compute_lumpy_anchor_zero_when_fits():
    history = [{"content": "hi", "role": "user"}]
    anchor = cs.compute_lumpy_anchor(
        history, 10_000, prune_target_frac=0.75, count_msg_tokens=lambda m: 1
    )
    assert anchor == 0


def test_compute_lumpy_anchor_keeps_at_least_one():
    history = [{"content": str(i), "role": "user"} for i in range(50)]
    anchor = cs.compute_lumpy_anchor(
        history, 4, prune_target_frac=0.75, count_msg_tokens=lambda m: 100
    )
    assert 0 <= anchor <= len(history) - 1
    assert len(history[anchor:]) >= 1


def test_compute_pruned_span_none_when_nothing_dropped():
    history = [{"id": 1, "content": "a"}]
    assert cs.compute_pruned_span(history, list(history), len) is None


def test_compute_pruned_span_identifies_dropped_head():
    history = [
        {"id": 1, "content": "aaaa", "metadata": {"session_id": "s1"}},
        {"id": 2, "content": "bbbb"},
        {"id": 3, "content": "cccc"},
    ]
    formatted = history[1:]  # oldest dropped
    span = cs.compute_pruned_span(history, formatted, lambda s: len(s))
    assert span is not None
    assert span.dropped_ids == [1]
    assert span.session_id == "s1"
    # token_estimate = len("aaaa") + _MESSAGE_OVERHEAD
    assert span.token_estimate == 4 + 4


def test_compute_pruned_span_reports_unmappable_rows_when_no_ids():
    history = [{"content": "a"}, {"content": "b"}]  # no id fields
    span = cs.compute_pruned_span(history, history[1:], len)
    assert span is not None
    assert span.dropped_messages == []
    assert span.dropped_ids == []
    assert span.token_estimate == 0
    assert span.total_dropped_count == 1
    assert span.unmappable_count == 1


def test_compute_pruned_span_separates_mappable_and_idless_rows():
    history = [
        {"content": "idless", "metadata": {"session_id": "s1"}},
        {"id": 2, "content": "aaaa"},
        {"id": 3, "content": "kept"},
    ]
    span = cs.compute_pruned_span(history, history[2:], len)

    assert span is not None
    assert span.dropped_messages == [history[1]]
    assert span.dropped_ids == [2]
    assert span.token_estimate == len("aaaa") + 4
    assert span.total_dropped_count == 2
    assert span.unmappable_count == 1
    assert span.session_id == "s1"


def test_microcompact_clears_old_tool_results_and_keeps_recent():
    history = [{"role": "tool", "content": f"result {i}"} for i in range(8)]
    cleared = cs.microcompact_tool_results(history, keep_recent=3)
    assert cleared == 5
    # last 3 untouched
    for msg in history[-3:]:
        assert not msg["content"].startswith('{"cleared":')
    # earlier ones cleared
    assert history[0]["content"].startswith('{"cleared":')


def test_microcompact_respects_protection_flags():
    history = [
        {"role": "tool", "content": "protected", "metadata": {"context_priority": "protected"}},
        {"role": "tool", "content": "excluded", "metadata": {"excluded_from_context": True}},
        {"role": "tool", "content": "decay", "metadata": {"decay_protected": True}},
        {"role": "tool", "content": "a"},
        {"role": "tool", "content": "b"},
    ]
    cs.microcompact_tool_results(history, keep_recent=1)
    assert history[0]["content"] == "protected"
    assert history[1]["content"] == "excluded"
    assert history[2]["content"] == "decay"


# ---------------------------------------------------------------------------
# ContextManager test double
# ---------------------------------------------------------------------------


def _make_cm(
    *,
    memories_result="",
    rag_result="",
    history=None,
    injected_clauses=None,
    dropped_clauses=None,
    budget_bytes=None,
    conv_store=MagicMock,
    format_side_effect=None,
):
    """Build a ContextManager with all dependencies mocked.

    Token counts are stubbed to len(text)//4 so budget accounting still
    functions; these tests care about section composition, not token math.
    """
    cm = object.__new__(ContextManager)
    cm.MICROCOMPACT_KEEP_RECENT = 5
    cm.EPISODE_THRESHOLD_MESSAGES = 20
    cm.PRUNE_TARGET_FRAC = 0.75
    cm.agent_id = "test-agent"
    cm.storage = MagicMock()

    counter = MagicMock()
    counter.count = lambda s: max(1, len(s) // 4) if s else 0
    counter.count_messages = lambda msgs: sum(
        (len(m.get("content", "")) // 4) + 4 for m in msgs
    )
    cm._counter = counter
    cm._counter_model = "test-model"

    cm._llm_service = None
    cm.llm_service = None
    cm._model_fallback = "test-model"

    cm.conversation_manager = MagicMock()
    cm.conversation_manager.get_conversation_history = AsyncMock(
        return_value=history or []
    )
    cm.conversation_manager._get_conversation_store = MagicMock(
        return_value=(conv_store() if callable(conv_store) else conv_store)
    )

    cm.memory_retriever = MagicMock() if memories_result else None
    if cm.memory_retriever is not None:
        cm.memory_retriever.record_accesses = AsyncMock()
    cm.memory_manager = MagicMock()
    cm.memory_manager.retrieve_memories = AsyncMock(return_value=memories_result)

    cm.context_builder = MagicMock()
    cm.context_builder.measure_mandatory_system_tokens = MagicMock(return_value=0)
    cm.context_builder.build_system_prompt = MagicMock(return_value="SYSTEM_PROMPT_BASE")
    cm.context_builder.build_system_prompt_with_subsections = MagicMock(
        return_value=(
            "SYSTEM_PROMPT_BASE",
            [("assembled_system_prompt", "SYSTEM_PROMPT_BASE")],
        )
    )
    if injected_clauses is not None or dropped_clauses is not None:
        cm.context_builder.build_system_prompt_with_tracking = MagicMock(
            return_value=SimpleNamespace(
                prompt="SYSTEM_PROMPT_TRACKED",
                injected_clauses=injected_clauses or [],
                dropped_clauses=dropped_clauses or [],
                subsections=[
                    ("assembled_system_prompt", "SYSTEM_PROMPT_TRACKED")
                ],
            )
        )
    cm.context_builder.retrieve_context = AsyncMock(return_value=rag_result)
    cm.context_builder.format_conversation_history = MagicMock(
        side_effect=format_side_effect or (lambda history, max_tokens: list(history))
    )

    cm.consolidator = None
    cm.salvage_worker = None
    return cm


# ---------------------------------------------------------------------------
# Production ⇄ compatibility-adapter byte parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_production_and_measurement_wrap_dynamic_context_identically():
    """build_context and measure_context_breakdown must wrap the same
    retrieved memory + RAG content into byte-identical dynamic user context.
    """
    memory_block = "[Memory 1] shared fact"
    rag_block = "[Document A] shared doc"

    # Production path.
    cm = _make_cm(memories_result=memory_block, rag_result=rag_block)
    prod = await cm.build_context(query="what is the weather", constitution="C")

    # Read-only measurement path (shares the same wrappers via context_stages).
    storage = MagicMock()
    storage.search_chunks = AsyncMock(
        return_value=[{"document_name": "A", "content": "shared doc", "created_at": ""}]
    )
    builder = ContextBuilder(storage)
    measured = await builder.measure_context_breakdown(
        query="what is the weather",
        history=[],
        constitution="C",
        include_rag=True,
        memory_retriever=AsyncMock(return_value=memory_block),
    )

    expected = cs.assemble_dynamic_user_context(
        [cs.wrap_memories(memory_block), cs.wrap_documents(rag_block)]
    )
    assert prod.dynamic_user_context == expected
    # The measurement artifact wraps the SAME memory block identically; the
    # RAG body differs only because the two feed different stores, so compare
    # the memory wrapper explicitly.
    assert cs.wrap_memories(memory_block) in measured["_artifacts"]["dynamic_user_context"]
    assert "<retrieved_context>" in measured["_artifacts"]["dynamic_user_context"]


@pytest.mark.asyncio
async def test_retrieval_inclusion_full_section_parity():
    """Full-result parity (not just wrapper bytes): given identical retrieved
    memory + RAG content, ``build_context`` and ``measure_context_breakdown``
    coordinate the SAME shared section plan — identical item counts AND
    byte-identical assembled dynamic user context.
    """
    memory_block = "[Memory 1] a\n[Memory 2] b"
    rag_block = "[Document A] x\n[Document B] y"

    # Production path.
    cm = _make_cm(memories_result=memory_block, rag_result=rag_block)
    prod = await cm.build_context(query="tell me about phoenix", constitution="C")

    # Read-only measurement path — force the SAME retrieved content on both
    # dynamic sections so any divergence is a plan divergence, not an input one.
    builder = ContextBuilder(MagicMock())
    with patch.object(builder, "retrieve_context", AsyncMock(return_value=rag_block)):
        measured = await builder.measure_context_breakdown(
            query="tell me about phoenix",
            history=[],
            constitution="C",
            include_rag=True,
            memory_retriever=AsyncMock(return_value=memory_block),
        )

    # Same section plan → same (counter-independent) item counts.
    assert prod.memory_count == measured["sections"]["memories"]["count"] == 2
    assert prod.rag_chunks == measured["sections"]["rag"]["chunks"] == 2

    # Byte-identical dynamic user context: one shared <retrieved_context>
    # envelope wrapping the same <memories> then <documents> blocks.
    expected = cs.assemble_dynamic_user_context(
        [cs.wrap_memories(memory_block), cs.wrap_documents(rag_block)]
    )
    assert prod.dynamic_user_context == expected
    assert measured["_artifacts"]["dynamic_user_context"] == expected


@pytest.mark.asyncio
async def test_trivial_query_uses_the_production_relevance_gate():
    """The compatibility adapter must not retrieve content production skips."""
    memory_block = "[Memory 1] a"
    rag_block = "[Document A] y"

    # Production: 'hi' is trivial → no retrieval calls, empty dynamic context.
    cm = _make_cm(memories_result=memory_block, rag_result=rag_block)
    prod = await cm.build_context(query="hi", constitution="C")
    assert prod.dynamic_user_context == ""
    cm.memory_manager.retrieve_memories.assert_not_called()
    cm.context_builder.retrieve_context.assert_not_called()

    # The read-only adapter follows the same gate and performs no retrieval.
    builder = ContextBuilder(MagicMock())
    retrieve = AsyncMock(return_value=rag_block)
    memories = AsyncMock(return_value=memory_block)
    with patch.object(builder, "retrieve_context", retrieve):
        measured = await builder.measure_context_breakdown(
            query="hi",
            history=[],
            constitution="C",
            include_rag=True,
            memory_retriever=memories,
        )
    assert measured["sections"]["memories"]["status"] == "skipped"
    assert measured["sections"]["rag"]["status"] == "skipped"
    assert measured["_artifacts"]["dynamic_user_context"] == ""
    memories.assert_not_awaited()
    retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_dynamic_sections_stable_under_elastic_slack_and_history_pressure():
    """Elastic-slack / history-pressure parity: production's elastic borrowing
    and severe history pruning (with salvage) must not perturb the shared
    dynamic-section plan. The memory/RAG wrapping stays byte-identical to the
    context_stages builders regardless of how hard history is pruned — proving
    the section plan is cleanly separated from the budget/pressure policy.
    """
    memory_block = "[Memory 1] a"
    rag_block = "[Document A] y"
    history = [
        {"id": i, "role": "user", "content": "x" * 400} for i in range(20)
    ]
    cm = _make_cm(
        memories_result=memory_block,
        rag_result=rag_block,
        history=history,
        format_side_effect=lambda history, max_tokens: list(history)[-2:],
    )
    result = await cm.build_context(query="tell me about phoenix", constitution="C")

    # History was pruned hard under pressure...
    assert len(result.messages) == 2
    assert any("History truncated" in w for w in result.warnings)

    # ...yet the dynamic sections are exactly the shared-builder assembly.
    assert result.memory_count == 1
    assert result.rag_chunks == 1
    assert result.dynamic_user_context == cs.assemble_dynamic_user_context(
        [cs.wrap_memories(memory_block), cs.wrap_documents(rag_block)]
    )


# ---------------------------------------------------------------------------
# Table-driven build_context behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,kwargs,build_kwargs,expect",
    [
        (
            "ephemeral",
            {},
            {"privacy_mode": "EPHEMERAL"},
            {"messages_empty": True, "dynamic": "", "system_has": "EPHEMERAL"},
        ),
        (
            "trivial_turn_skips_retrieval",
            {"memories_result": "[Memory 1] x", "rag_result": "[Document A] y"},
            {"query": "hi"},
            {"dynamic": "", "no_memory_call": True, "no_rag_call": True},
        ),
        (
            "memories_only",
            {"memories_result": "[Memory 1] x"},
            {"query": "tell me about phoenix"},
            {"has_memories": True, "no_documents": True},
        ),
        (
            "rag_only",
            {"rag_result": "[Document A] y"},
            {"query": "tell me about phoenix"},
            {"has_documents": True, "no_memories": True},
        ),
        (
            "both_single_envelope",
            {"memories_result": "[Memory 1] x", "rag_result": "[Document A] y"},
            {"query": "tell me about phoenix"},
            {"has_memories": True, "has_documents": True, "single_envelope": True},
        ),
        (
            "none",
            {},
            {"query": "tell me about phoenix"},
            {"dynamic": ""},
        ),
    ],
)
async def test_build_context_table(name, kwargs, build_kwargs, expect):
    cm = _make_cm(**kwargs)
    result = await cm.build_context(constitution="C", **{"query": "q", **build_kwargs})
    assert isinstance(result, ContextResult)

    if expect.get("messages_empty"):
        assert result.messages == []
    if "dynamic" in expect:
        assert result.dynamic_user_context == expect["dynamic"]
    if expect.get("system_has"):
        assert expect["system_has"] in result.system_prompt
    if expect.get("has_memories"):
        assert "<memories>" in result.dynamic_user_context
    if expect.get("no_memories"):
        assert "<memories>" not in result.dynamic_user_context
    if expect.get("has_documents"):
        assert "<documents>" in result.dynamic_user_context
    if expect.get("no_documents"):
        assert "<documents>" not in result.dynamic_user_context
    if expect.get("single_envelope"):
        assert result.dynamic_user_context.count("<retrieved_context>") == 1
    if expect.get("no_memory_call"):
        cm.memory_manager.retrieve_memories.assert_not_called()
    if expect.get("no_rag_call"):
        cm.context_builder.retrieve_context.assert_not_called()
    # Dynamic context never leaks into the (cacheable) system prefix.
    assert "<retrieved_context>" not in result.system_prompt


@pytest.mark.asyncio
async def test_build_context_golden_no_retrieval():
    """Representative full ContextResult for a bare build (golden fields)."""
    cm = _make_cm()
    result = await cm.build_context(query="tell me about phoenix", constitution="C")
    assert result.system_prompt == "SYSTEM_PROMPT_BASE"
    assert result.dynamic_user_context == ""
    assert result.messages == []
    assert result.degraded_mode is False
    assert result.injected_clauses is None
    assert result.dropped_clauses is None
    assert result.mandatory_system_tokens == 0


@pytest.mark.asyncio
async def test_build_context_includes_episodes_for_long_conversations():
    history = [{"role": "user", "content": f"m{i}"} for i in range(25)]
    cm = _make_cm(history=history)
    cm.consolidator = MagicMock()
    episodes = [
        {"title": "T", "timespan": "x", "summary": "s", "emotional_arc": "a"}
    ]
    cm.context_builder.get_episodes_for_context = AsyncMock(return_value=episodes)
    cm.context_builder.format_episodes_for_context = MagicMock(
        return_value="--- CONVERSATION EPISODES ---\nblock\n--- END EPISODES ---"
    )
    result = await cm.build_context(query="tell me about phoenix", constitution="C")
    assert result.episode_count == 1
    assert "CONVERSATION EPISODES" in result.system_prompt


@pytest.mark.asyncio
async def test_build_context_budget_addendum_and_doctrine_tracking():
    from collections import OrderedDict

    cm = _make_cm(
        injected_clauses=["KESTREL_CONSTITUTION", "TORTOISE_DOCTRINE.md"],
        dropped_clauses=[],
    )
    result = await cm.build_context(
        query="tell me about phoenix",
        constitution="C",
        system_prompt_budget_bytes=10_000,
        system_prompt_addendum="CANARY-DIRECTIVE",
        anchored_doctrine=OrderedDict({"TORTOISE_DOCTRINE.md": "body"}),
    )
    # Addendum appended at the END so the cache prefix stays put.
    assert result.system_prompt.endswith("CANARY-DIRECTIVE")
    assert result.injected_clauses == ["KESTREL_CONSTITUTION", "TORTOISE_DOCTRINE.md"]


@pytest.mark.asyncio
async def test_build_context_severe_history_pressure_truncates():
    history = [{"id": i, "role": "user", "content": "x" * 400} for i in range(20)]
    cm = _make_cm(
        history=history,
        format_side_effect=lambda history, max_tokens: list(history)[-2:],
    )
    result = await cm.build_context(query="tell me about phoenix", constitution="C")
    assert len(result.messages) == 2
    assert any("History truncated" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_system_prompt_stable_across_differing_retrieval():
    """Golden cache invariant: differing memory/RAG must not shift a byte of
    the system prompt."""
    cm1 = _make_cm(memories_result="[Memory 1] a", rag_result="[Document A] a")
    r1 = await cm1.build_context(query="what is the weather", constitution="C")
    cm2 = _make_cm(memories_result="[Memory 2] totally different", rag_result="[Document B] different")
    r2 = await cm2.build_context(query="who are you", constitution="C")
    assert r1.system_prompt == r2.system_prompt


@pytest.mark.asyncio
async def test_injection_tracking_published_on_success():
    cm = _make_cm(injected_clauses=["KESTREL_CONSTITUTION"], dropped_clauses=["ADDITIONAL_CONTEXT"])
    reset_injection_tracking()
    result = await cm.build_context(
        query="what is the weather",
        constitution="C",
        system_prompt_budget_bytes=10_000,
    )
    assert result.injected_clauses == ["KESTREL_CONSTITUTION"]
    assert result.dropped_clauses == ["ADDITIONAL_CONTEXT"]
    assert get_current_injection_tracking() == (
        ["KESTREL_CONSTITUTION"],
        ["ADDITIONAL_CONTEXT"],
    )


# ---------------------------------------------------------------------------
# Durable salvage — success + fail-closed boundary
# ---------------------------------------------------------------------------


def _drop_one(history, max_tokens):
    # Simulate the formatter pruning the oldest message.
    return list(history)[1:]


@pytest.mark.asyncio
async def test_salvage_success_records_warning_not_degraded():
    history = [{"id": i, "content": "x" * 20, "metadata": {"session_id": "s1"}} for i in range(3)]
    cm = _make_cm(history=history, format_side_effect=_drop_one)
    salvage_result = SimpleNamespace(salvage_id=42, pointer_only_terminal=False)
    with patch(
        "kestrel_sovereign.agent.context_manager.is_durable_salvage_enabled",
        return_value=True,
    ), patch(
        "kestrel_sovereign.agent.context_manager.get_pending_count",
        new=AsyncMock(return_value=0),
    ), patch(
        "kestrel_sovereign.agent.context_manager.salvage_messages",
        new=AsyncMock(return_value=salvage_result),
    ):
        result = await cm.build_context(query="what is the weather", constitution="C")
    assert result.degraded_mode is False
    assert any("context-salvage: 1 messages" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_salvage_write_failure_is_fail_closed_degraded():
    history = [{"id": i, "content": "x" * 20, "metadata": {"session_id": "s1"}} for i in range(3)]
    cm = _make_cm(history=history, format_side_effect=_drop_one)
    with patch(
        "kestrel_sovereign.agent.context_manager.is_durable_salvage_enabled",
        return_value=True,
    ), patch(
        "kestrel_sovereign.agent.context_manager.get_pending_count",
        new=AsyncMock(return_value=0),
    ), patch(
        "kestrel_sovereign.agent.context_manager.salvage_messages",
        new=AsyncMock(side_effect=SalvageWriteError("disk full")),
    ):
        result = await cm.build_context(query="what is the weather", constitution="C")
    assert result.degraded_mode is True
    assert result.messages == []
    assert result.budget_summary["mode"] == "degraded"
    assert any("durable salvage write failed" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_salvage_unavailable_store_is_fail_closed_degraded():
    history = [{"id": i, "content": "x" * 20, "metadata": {"session_id": "s1"}} for i in range(3)]
    cm = _make_cm(history=history, format_side_effect=_drop_one, conv_store=lambda: None)
    with patch(
        "kestrel_sovereign.agent.context_manager.is_durable_salvage_enabled",
        return_value=True,
    ):
        result = await cm.build_context(query="what is the weather", constitution="C")
    assert result.degraded_mode is True
    assert result.budget_summary["reason"] == "salvage-conv-store-unavailable"


# ---------------------------------------------------------------------------
# Concurrency — builds do not share counters/results or injection tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_builds_do_not_share_state():
    """Two concurrent build_context calls must produce independent results
    and independent per-task injection tracking.
    """
    cm_a = _make_cm(
        memories_result="[Memory 1] a",
        rag_result="[Document A] a",
        injected_clauses=["A_CLAUSE"],
    )
    cm_b = _make_cm(injected_clauses=["B_CLAUSE"])  # trivial-free, no retrieval content

    async def run(cm, query):
        reset_injection_tracking()
        result = await cm.build_context(
            query=query, constitution="C", system_prompt_budget_bytes=10_000
        )
        # Read tracking back inside the SAME task — must be this build's value.
        return result, get_current_injection_tracking()

    (res_a, track_a), (res_b, track_b) = await asyncio.gather(
        run(cm_a, "what is the weather"),
        run(cm_b, "who are you"),
    )

    # Independent counters/results — A retrieved, B did not.
    assert res_a.memory_count == 1
    assert res_a.rag_chunks == 1
    assert res_b.memory_count == 0
    assert res_b.rag_chunks == 0
    assert res_a.dynamic_user_context != res_b.dynamic_user_context

    # Per-task injection tracking did not cross.
    assert track_a == (["A_CLAUSE"], [])
    assert track_b == (["B_CLAUSE"], [])
    assert res_a.injected_clauses == ["A_CLAUSE"]
    assert res_b.injected_clauses == ["B_CLAUSE"]


# ---------------------------------------------------------------------------
# #2534 canonical live/dry plan parity
# ---------------------------------------------------------------------------


def _assert_live_dry_plan_equivalent(live, dry):
    assert live.assembly.system_prompt == dry.assembly.system_prompt
    assert live.assembly.formatted_history == dry.assembly.formatted_history
    assert live.assembly.dynamic_user_context == dry.assembly.dynamic_user_context
    assert live.total_tokens == dry.total_tokens
    assert live.total_budget == dry.total_budget
    assert live.budget_summary == dry.budget_summary
    assert live.warnings == dry.warnings
    assert live.memory_access_ids == dry.memory_access_ids
    assert live.salvage_requirement == dry.salvage_requirement
    assert live.microcompacted_tool_results == dry.microcompacted_tool_results
    assert set(live.sections) == set(dry.sections)
    for name in live.sections:
        left = live.sections[name]
        right = dry.sections[name]
        assert left.status == right.status, name
        assert left.tokens == right.tokens, name
        assert left.raw_tokens == right.raw_tokens, name
        assert left.items == right.items, name
        assert left.budget == right.budget, name
        assert left.provenance == right.provenance, name
        assert left.reason == right.reason, name
        assert left.details == right.details, name


async def _build_live_dry_plans(cm, **kwargs):
    live = await cm.build_context_plan(
        constitution="C",
        mode=ContextBuildMode.LIVE,
        **{"query": "explain the phoenix migration", **kwargs},
    )
    dry = await cm.build_context_plan(
        constitution="C",
        mode=ContextBuildMode.DRY_RUN,
        **{"query": "explain the phoenix migration", **kwargs},
    )
    _assert_live_dry_plan_equivalent(live, dry)
    return live, dry


@pytest.mark.asyncio
async def test_real_status_acquisition_matches_live_render_and_defers_writes():
    """Drive the endpoint acquisition and live wrapper through real policy.

    Unlike the table-driven parity cases below, this test does not replace
    system assembly, history formatting, retrieval normalization, budgeting,
    or pruning with mocks.  The only fakes are the external stores.
    """
    from kestrel_sovereign.endpoints.agent import (
        _acquire_context_status_measurement,
    )

    route_model = "openai:plan/gpt-4o"
    llm_service = SimpleNamespace(
        get_active_model_selection=lambda: {"model": route_model},
        get_active_model_id=lambda: route_model,
    )
    storage = MagicMock()
    storage.search_chunks = AsyncMock(
        return_value=[
            {
                "document_name": "migration.md",
                "content": "Phoenix uses an append-only migration journal.",
                "created_at": "2026-07-24T10:30:00",
            }
        ]
    )
    memory_retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": 701,
                    "role": "user",
                    "content": "Keep migration evidence linked to its source.",
                    "created_at": "2026-07-23T09:00:00",
                    "metadata": {
                        "importance": 0.9,
                        "emotional_valence": 0.1,
                    },
                }
            ]
        ),
        record_accesses=AsyncMock(),
    )
    builder = ContextBuilder(
        storage,
        llm_service=llm_service,
        model=route_model,
    )
    cm = ContextManager(
        storage,
        agent_id="did:test:status-live-parity",
        memory_retriever=memory_retriever,
        llm_service=llm_service,
        context_builder=builder,
    )
    history = [
        {
            "id": 11,
            "role": "user",
            "content": "What changed in Phoenix?",
            "created_at": "2026-07-24T10:00:00",
        },
        {
            "id": 12,
            "role": "assistant",
            "content": "The migration became append-only.",
            "model": "gpt-4o",
            "provider": "openai",
            "created_at": "2026-07-24T10:01:00",
        },
        {
            "id": 13,
            "role": "user",
            "content": "Explain the Phoenix migration evidence path.",
            "created_at": "2026-07-24T10:02:00",
        },
    ]
    constitution = "Protect operator intent and preserve evidence provenance."
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_migration",
                "description": "Read migration evidence.",
            },
        }
    ]
    constitution_calls = []

    async def get_governing_constitution(*, allow_lazy_anchor=True):
        constitution_calls.append(allow_lazy_anchor)
        return constitution

    privacy_agent = SimpleNamespace(
        privacy_mode="NORMAL",
        get_conversation_history=AsyncMock(return_value=history),
    )
    agent = SimpleNamespace(
        storage=storage,
        privacy_agent=privacy_agent,
        context_manager=cm,
        _privacy_mode="NORMAL",
        _session_briefed=False,
        _get_governing_constitution=get_governing_constitution,
        _build_all_tools=lambda: tools,
        features={},
    )

    # Capture plans while delegating to the unmodified production planner.
    real_plan_builder = cm.build_context_plan
    captured_plans = []

    async def capture_plan(**kwargs):
        plan = await real_plan_builder(**kwargs)
        captured_plans.append(plan)
        return plan

    cm.build_context_plan = capture_plan
    measurement = await _acquire_context_status_measurement(
        agent,
        "session-real-parity",
        full=True,
    )
    dry = captured_plans[-1]

    assert constitution_calls == [False]
    assert dry.mode is ContextBuildMode.DRY_RUN
    assert measurement.breakdown == dry.to_breakdown()
    assert measurement.current_model == route_model
    assert measurement.model_identity == {
        "model": "gpt-4o",
        "provider": "openai",
        "context_model": "openai/gpt-4o",
        "model_source": "assistant_turn",
    }
    assert dry.sections["memories"].provenance == (
        "memory_retriever",
        "query_relevance_gate",
        "elastic_budget_gate",
    )
    assert dry.sections["rag"].provenance == (
        "rag_store",
        "query_relevance_gate",
        "elastic_budget_gate",
    )
    assert sum(
        section.tokens or 0 for section in dry.sections.values()
    ) == dry.total_tokens
    memory_retriever.record_accesses.assert_not_awaited()

    live_result = await cm.build_context(
        query="Explain the Phoenix migration evidence path.",
        constitution=constitution,
        include_briefing=True,
        include_memories=True,
        include_rag=True,
        privacy_mode="NORMAL",
        conversation_history=history,
        tools=tools,
    )
    live = captured_plans[-1]

    assert live.mode is ContextBuildMode.LIVE
    _assert_live_dry_plan_equivalent(live, dry)
    assert live_result.system_prompt == dry.assembly.system_prompt
    assert live_result.messages == dry.assembly.formatted_history
    assert live_result.dynamic_user_context == dry.assembly.dynamic_user_context
    assert live_result.total_tokens == dry.total_tokens
    assert live_result.budget_summary == dry.budget_summary
    memory_retriever.record_accesses.assert_awaited_once_with(
        (701,),
        "did:test:status-live-parity",
    )


@pytest.mark.asyncio
async def test_status_refuses_missing_governing_policy_before_planning():
    from kestrel_sovereign.endpoints.agent import (
        _acquire_context_status_measurement,
    )

    plan_builder = AsyncMock(
        side_effect=AssertionError("must not plan with empty governing policy")
    )

    async def missing_governing_constitution(*, allow_lazy_anchor=True):
        assert allow_lazy_anchor is False
        return "Error: Governing constitution is not anchored."

    agent = SimpleNamespace(
        storage=SimpleNamespace(),
        privacy_agent=SimpleNamespace(
            privacy_mode="NORMAL",
            get_conversation_history=AsyncMock(
                return_value=[
                    {
                        "role": "user",
                        "content": "explain the migration",
                    }
                ]
            ),
        ),
        context_manager=SimpleNamespace(
            build_context_plan=plan_builder,
        ),
        _get_governing_constitution=missing_governing_constitution,
    )

    with pytest.raises(
        RuntimeError,
        match="governing constitution is unavailable",
    ):
        await _acquire_context_status_measurement(
            agent,
            "session-missing-policy",
            full=True,
        )

    plan_builder.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_dry_plan_equivalence_for_retrieval_and_tools():
    from kestrel_sovereign.agent.memory_manager import RetrievedMemoryBlock

    cm = _make_cm(
        memories_result="[Memory 1] a\n[Memory 2] b",
        rag_result="[Document A] x",
    )
    cm.memory_manager.retrieve_memories.return_value = RetrievedMemoryBlock(
        text="[Memory 1] a\n[Memory 2] b",
        message_ids=(41, 42),
    )
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    live, dry = await _build_live_dry_plans(cm, tools=tools)

    assert dry.sections["memories"].status is SectionStatus.INCLUDED
    assert dry.memory_access_ids == (41, 42)
    assert dry.sections["rag"].status is SectionStatus.INCLUDED
    assert dry.sections["tools"].tokens > 0
    assert (
        dry.sections["history"].budget
        > dry.budget_summary["allocations"]["history"]["budget"]
    ), "finalized section slack must reach the history effective budget"
    assert sum(
        section.tokens or 0 for section in dry.sections.values()
    ) == dry.total_tokens
    # Planning is read-only: rehearsal access commits only in build_context.
    cm.memory_retriever.record_accesses.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_dry_plan_equivalence_for_trivial_and_ephemeral_turns():
    cm = _make_cm(
        memories_result="[Memory 1] should not appear",
        rag_result="[Document A] should not appear",
    )
    live, dry = await _build_live_dry_plans(cm, query="hi")
    assert dry.sections["memories"].status is SectionStatus.SKIPPED
    assert dry.sections["memories"].tokens is None
    assert dry.sections["rag"].status is SectionStatus.SKIPPED
    assert dry.sections["rag"].tokens is None
    assert dry.assembly.dynamic_user_context == ""
    cm.memory_manager.retrieve_memories.assert_not_called()
    cm.context_builder.retrieve_context.assert_not_called()

    ephemeral_live, ephemeral_dry = await _build_live_dry_plans(
        cm,
        privacy_mode="EPHEMERAL",
    )
    assert ephemeral_dry.assembly.formatted_history == []
    assert "EPHEMERAL MODE ACTIVE" in ephemeral_dry.assembly.system_prompt
    assert ephemeral_dry.sections["history"].tokens is None
    assert ephemeral_live.total_tokens == ephemeral_dry.total_tokens


@pytest.mark.asyncio
async def test_live_dry_plan_equivalence_for_degraded_mandatory_floor():
    cm = _make_cm(
        memories_result="[Memory 1] must not be retrieved",
        rag_result="[Document A] must not be retrieved",
    )
    cm.context_builder.measure_mandatory_system_tokens.return_value = 10_000_000

    live, dry = await _build_live_dry_plans(cm)

    assert live.degraded_mode is True
    assert dry.degraded_mode is True
    assert dry.sections == {}
    assert dry.total_tokens == 0
    assert dry.budget_summary["mode"] == "degraded"
    assert "mandatory governance floor" in dry.degraded_reason
    cm.memory_manager.retrieve_memories.assert_not_awaited()
    cm.context_builder.retrieve_context.assert_not_awaited()

    result = await cm.build_context(
        query="explain the phoenix migration",
        constitution="C",
    )
    assert result.degraded_mode is True
    assert result.messages == []
    assert result.system_prompt == ""
    cm.memory_retriever.record_accesses.assert_not_awaited()


@pytest.mark.asyncio
async def test_cheap_plan_labels_expensive_sections_unknown_not_zero():
    cm = _make_cm(
        memories_result="[Memory 1] not fetched",
        rag_result="[Document A] not fetched",
    )
    plan = await cm.build_context_plan(
        query="explain the phoenix migration",
        constitution="C",
        mode=ContextBuildMode.DRY_RUN,
        measure_expensive_sections=False,
    )

    for name in ("memories", "rag"):
        assert plan.sections[name].status is SectionStatus.UNKNOWN
        assert plan.sections[name].tokens is None
        assert name not in plan.budget_summary["finalized_sections"]
    assert plan.measurement_complete is False
    cm.memory_manager.retrieve_memories.assert_not_called()
    cm.context_builder.retrieve_context.assert_not_called()


@pytest.mark.asyncio
async def test_live_dry_plan_equivalence_for_doctrine_reflection_and_episodes():
    from collections import OrderedDict

    history = [{"role": "user", "content": f"m{i}"} for i in range(25)]
    cm = _make_cm(
        history=history,
        injected_clauses=["KESTREL_CONSTITUTION", "TORTOISE_DOCTRINE.md"],
    )
    cm.consolidator = MagicMock()
    episodes = [
        {"title": "T", "timespan": "x", "summary": "s", "emotional_arc": "a"}
    ]
    cm.context_builder.get_episodes_for_context = AsyncMock(
        return_value=episodes
    )
    cm.context_builder.format_episodes_for_context = MagicMock(
        return_value="--- CONVERSATION EPISODES ---\nblock\n--- END EPISODES ---"
    )

    _, dry = await _build_live_dry_plans(
        cm,
        reflection_guidance=["retain provenance"],
        system_prompt_addendum="CANARY-DIRECTIVE",
        system_prompt_budget_bytes=10_000,
        anchored_doctrine=OrderedDict(
            {"TORTOISE_DOCTRINE.md": "body"}
        ),
    )
    assert dry.sections["episodes"].status is SectionStatus.INCLUDED
    assert dry.assembly.episode_count == 1
    assert "ACTIVE REFLECTION GUIDANCE" in dry.assembly.system_prompt
    assert "--- END GUIDANCE ---" in dry.assembly.system_prompt
    assert "CANARY-DIRECTIVE" in dry.assembly.system_prompt
    assert "system_prompt_addendum" in dry.sections["system"].provenance
    assert "reflection_guidance" in dry.sections["system"].provenance
    assert (
        "anchored_doctrine:TORTOISE_DOCTRINE.md"
        in dry.sections["system"].provenance
    )


@pytest.mark.asyncio
async def test_live_dry_plan_equivalence_for_doctrine_addendum_cap_exclusions():
    """Late optional sections obey the same byte cap in both plan modes."""

    from collections import OrderedDict

    history = [{"role": "user", "content": f"m{i}"} for i in range(25)]
    cm = _make_cm(history=history)
    cm.consolidator = MagicMock()
    cm.context_builder.get_episodes_for_context = AsyncMock(
        return_value=[
            {
                "title": "T",
                "timespan": "x",
                "summary": "long episode summary",
                "emotional_arc": "a",
            }
        ]
    )
    cm.context_builder.format_episodes_for_context = MagicMock(
        return_value="--- CONVERSATION EPISODES ---\nblock\n--- END EPISODES ---"
    )

    _, dry = await _build_live_dry_plans(
        cm,
        reflection_guidance=["this optional guidance cannot fit"],
        system_prompt_addendum="A",
        system_prompt_budget_bytes=30,
        anchored_doctrine=OrderedDict(
            {"TORTOISE_DOCTRINE.md": "body"}
        ),
    )

    assert dry.assembly.system_prompt.endswith("\n\nA")
    assert "ACTIVE REFLECTION GUIDANCE" not in dry.assembly.system_prompt
    assert dry.sections["episodes"].status is SectionStatus.EXCLUDED
    assert any("reflection guidance skipped" in warning for warning in dry.warnings)
    assert any("episode context skipped" in warning for warning in dry.warnings)


@pytest.mark.asyncio
async def test_reflection_guidance_rejected_by_token_budget_is_not_appended():
    """Optional guidance rejected by accounting never reaches prompt bytes."""

    from kestrel_sovereign.agent.context_stages import ContextAssembly
    from kestrel_sovereign.agent.token_budget import ElasticTokenBudget

    cm = _make_cm()
    assembly = ContextAssembly(system_prompt="BASE-PROMPT")
    budget = ElasticTokenBudget(
        "test-model", message_count=0, mandatory_system_tokens=0
    )
    system = budget.allocations["system"]
    system.used = system.budget
    before = budget.total_used

    included = cm._apply_reflection_guidance(
        assembly,
        budget,
        ["OPTIONAL-GUIDANCE-CANARY"],
        system_prompt_budget_bytes=None,
    )

    assert included is False
    assert assembly.system_prompt == "BASE-PROMPT"
    assert budget.total_used == before
    assert any(
        "reflection guidance skipped" in warning
        for warning in assembly.warnings
    )


@pytest.mark.asyncio
async def test_reflection_guidance_charges_joiner_token_before_append():
    """Optional guidance is skipped when only its body, not its joiner, fits."""

    from kestrel_sovereign.agent.context_stages import (
        ContextAssembly,
        build_reflection_guidance_block,
    )
    from kestrel_sovereign.agent.token_budget import ElasticTokenBudget

    cm = _make_cm()
    assembly = ContextAssembly(system_prompt="BASE-PROMPT")
    guidance = ["OPTIONAL-GUIDANCE-CANARY"]
    guidance_text = build_reflection_guidance_block(guidance)
    guidance_tokens = cm.counter.count(guidance_text)
    marginal_tokens = (
        cm.counter.count(f"{assembly.system_prompt}\n\n{guidance_text}")
        - cm.counter.count(assembly.system_prompt)
    )
    assert marginal_tokens > guidance_tokens

    budget = ElasticTokenBudget(
        "test-model", message_count=0, mandatory_system_tokens=0
    )
    system = budget.allocations["system"]
    system.used = system.budget - guidance_tokens
    before = budget.total_used

    included = cm._apply_reflection_guidance(
        assembly,
        budget,
        guidance,
        system_prompt_budget_bytes=None,
    )

    assert included is False
    assert assembly.system_prompt == "BASE-PROMPT"
    assert budget.total_used == before


@pytest.mark.asyncio
async def test_live_dry_plan_equivalence_under_lumpy_microcompact_pressure():
    history = [
        {
            "id": index,
            "role": "tool" if index < 8 else "user",
            "content": "tool-output-" + ("x" * 400),
            "created_at": "2026-01-01T00:00:00Z",
            "metadata": {"session_id": "s1"},
        }
        for index in range(8)
    ]
    original = [dict(message) for message in history]
    cm = _make_cm(
        history=history,
        format_side_effect=lambda history, max_tokens: list(history)[-3:],
    )
    live, dry = await _build_live_dry_plans(cm)

    assert live.microcompacted_tool_results == 3
    assert live.assembly.formatted_history == dry.assembly.formatted_history
    assert len(dry.assembly.formatted_history) == 3
    assert any("History truncated" in warning for warning in dry.warnings)
    # The planner copied its acquisition input; status reads do not mutate it.
    assert history == original


@pytest.mark.asyncio
async def test_live_dry_plan_equivalence_for_exact_final_payload_pruning():
    """Tool/wrapper costs participate in the final history prune."""

    history = [
        {
            "id": index,
            "role": "user",
            "content": "x" * 4_000,
            "metadata": {"session_id": "s1"},
        }
        for index in range(20)
    ]
    cm = _make_cm(history=history)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "large_schema",
                "description": "x" * 96_000,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    live, dry = await _build_live_dry_plans(cm, tools=tools)

    assert len(dry.assembly.formatted_history) < len(history)
    assert dry.total_tokens <= dry.total_budget
    assert dry.budget_summary["external_reserved_tokens"] > 0
    assert live.sections["history"].tokens == dry.sections["history"].tokens


@pytest.mark.asyncio
async def test_dry_plan_models_salvage_without_writing_success_or_failure():
    history = [
        {
            "id": index,
            "role": "user",
            "content": "x" * 20,
            "metadata": {"session_id": "s1"},
        }
        for index in range(3)
    ]
    cm = _make_cm(history=history, format_side_effect=_drop_one)
    with patch(
        "kestrel_sovereign.agent.context_manager.is_durable_salvage_enabled",
        return_value=True,
    ), patch(
        "kestrel_sovereign.agent.context_manager.get_pending_count",
        new=AsyncMock(return_value=0),
    ) as pending, patch(
        "kestrel_sovereign.agent.context_manager.salvage_messages",
        new=AsyncMock(
            return_value=SimpleNamespace(
                salvage_id=42,
                pointer_only_terminal=True,
            )
        ),
    ) as salvage:
        live, dry = await _build_live_dry_plans(cm)
        assert dry.salvage_requirement is not None
        assert dry.to_breakdown()["salvage"]["status"] == (
            "required_not_committed"
        )
        pending.assert_not_awaited()
        salvage.assert_not_awaited()

        result = await cm.build_context(
            query="explain the phoenix migration",
            constitution="C",
        )
        assert result.degraded_mode is False
        salvage.assert_awaited_once()

    failing_cm = _make_cm(history=history, format_side_effect=_drop_one)
    with patch(
        "kestrel_sovereign.agent.context_manager.is_durable_salvage_enabled",
        return_value=True,
    ), patch(
        "kestrel_sovereign.agent.context_manager.get_pending_count",
        new=AsyncMock(return_value=0),
    ), patch(
        "kestrel_sovereign.agent.context_manager.salvage_messages",
        new=AsyncMock(side_effect=SalvageWriteError("disk full")),
    ) as failing_salvage:
        _, dry_failure = await _build_live_dry_plans(failing_cm)
        assert dry_failure.salvage_requirement is not None
        failing_salvage.assert_not_awaited()
        failed = await failing_cm.build_context(
            query="explain the phoenix migration",
            constitution="C",
        )
        assert failed.degraded_mode is True


@pytest.mark.asyncio
async def test_dry_plan_reports_idless_salvage_gap_without_writing():
    history = [
        {"role": "user", "content": "idless"},
        {"role": "assistant", "content": "kept"},
    ]
    cm = _make_cm(history=history, format_side_effect=_drop_one)
    with patch(
        "kestrel_sovereign.agent.context_manager.is_durable_salvage_enabled",
        return_value=True,
    ), patch(
        "kestrel_sovereign.agent.context_manager.get_pending_count",
        new=AsyncMock(return_value=0),
    ) as pending, patch(
        "kestrel_sovereign.agent.context_manager.salvage_messages",
        new=AsyncMock(),
    ) as salvage:
        plan = await cm.build_context_plan(
            query="explain the migration",
            constitution="C",
            mode=ContextBuildMode.DRY_RUN,
        )
        rendered = plan.to_breakdown()["salvage"]

        assert plan.salvage_requirement is None
        assert rendered == {
            "feature_enabled": True,
            "required": False,
            "status": "unavailable_no_persistent_ids",
            "message_count": 0,
            "pruned_message_count": 1,
            "unmappable_message_count": 1,
            "token_estimate": 0,
            "silent_prune_possible": True,
        }
        pending.assert_not_awaited()
        salvage.assert_not_awaited()


@pytest.mark.asyncio
async def test_mixed_salvage_span_commits_only_durably_linkable_rows():
    history = [
        {"id": 7, "role": "user", "content": "persistent"},
        {"role": "assistant", "content": "idless"},
        {"id": 9, "role": "user", "content": "kept"},
    ]
    cm = _make_cm(
        history=history,
        format_side_effect=lambda history, max_tokens: list(history)[2:],
    )
    salvage_result = SimpleNamespace(
        salvage_id=42,
        pointer_only_terminal=True,
    )
    with patch(
        "kestrel_sovereign.agent.context_manager.is_durable_salvage_enabled",
        return_value=True,
    ), patch(
        "kestrel_sovereign.agent.context_manager.get_pending_count",
        new=AsyncMock(return_value=0),
    ), patch(
        "kestrel_sovereign.agent.context_manager.salvage_messages",
        new=AsyncMock(return_value=salvage_result),
    ) as salvage:
        plan = await cm.build_context_plan(
            query="explain the migration",
            constitution="C",
            mode=ContextBuildMode.DRY_RUN,
        )
        disposition = plan.to_breakdown()["salvage"]
        assert disposition["status"] == "partial_required_not_committed"
        assert disposition["message_count"] == 1
        assert disposition["pruned_message_count"] == 2
        assert disposition["unmappable_message_count"] == 1
        salvage.assert_not_awaited()

        result = await cm.build_context(
            query="explain the migration",
            constitution="C",
        )

    assert result.degraded_mode is False
    committed_rows = salvage.await_args.kwargs["original_messages"]
    assert committed_rows == [history[0]]
    assert all(isinstance(row.get("id"), int) for row in committed_rows)


@pytest.mark.asyncio
async def test_read_only_governing_constitution_does_not_lazy_anchor():
    from kestrel_sovereign.agent.constitution import ConstitutionMixin

    storage = SimpleNamespace(
        get_node=AsyncMock(return_value=SimpleNamespace(properties={})),
        store_file=AsyncMock(),
        add_node=AsyncMock(),
    )
    host = SimpleNamespace(storage=storage, agent_id="did:test:context-status")

    result = await ConstitutionMixin._get_governing_constitution(
        host,
        allow_lazy_anchor=False,
    )

    assert result.startswith("Error: Governing constitution is not anchored")
    storage.store_file.assert_not_awaited()
    storage.add_node.assert_not_awaited()
