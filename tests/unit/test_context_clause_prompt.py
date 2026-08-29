"""Lifecycle, cache-stability, and accounting guards for feature context."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.agent.context_manager import (
    ContextManager,
    get_current_injection_tracking,
    reset_injection_tracking,
)
from kestrel_sovereign.agent.context_stages import EPHEMERAL_NOTICE
from kestrel_sovereign.agent.system_prompt_assembler import assemble_system_prompt
from kestrel_sovereign.agent.token_budget import RESPONSE_RESERVE
from kestrel_sovereign.features.contribution_runtime import (
    ContextClauseRegistry,
    FeatureContributionRuntimeError,
    ResolvedContextClause,
)


class _ClauseRegistry:
    def __init__(self, *clauses: ResolvedContextClause) -> None:
        self.clauses = clauses
        self.snapshot_calls = 0

    def snapshot(self):
        self.snapshot_calls += 1
        return self.clauses


def _builder(registry=None) -> ContextBuilder:
    builder = object.__new__(ContextBuilder)
    builder._llm_service = None
    builder._model_fallback = "claude-sonnet-4-6"
    builder._counter = None
    builder._counter_model = None
    builder._bootstrap_loader = MagicMock()
    builder._bootstrap_loader.load.return_value = OrderedDict(
        [("SOUL.md", "Stable identity")]
    )
    builder._bootstrap_loader.get_file.side_effect = lambda name: {
        "SOUL.md": "Stable identity"
    }.get(name)
    builder.storage = MagicMock()
    builder.consolidator = None
    builder._semantic_inference_profile = None
    builder._semantic_inference_limits = None
    builder._semantic_maintenance_limits = None
    builder._semantic_answerability_gate = None
    builder.last_semantic_recall_metadata = {"status": "disabled"}
    builder.agent_data_path = None
    builder._context_clause_registry = registry
    return builder


def _clause(
    name: str,
    priority: int,
    body: str,
    *,
    owner: str | None = None,
) -> ResolvedContextClause:
    resolved_owner = owner or f"tests:{name}"
    return ResolvedContextClause(
        owner=resolved_owner,
        name=name,
        priority=priority,
        body=body,
        registration=SimpleNamespace(identity=(resolved_owner, name)),
    )


def test_zero_contributions_preserve_real_assembler_bytes():
    builder = _builder()
    expected_legacy = (
        "--- YOUR IDENTITY ---\n\nStable identity\n\n--- END IDENTITY ---\n\n"
        "--- GOVERNING CONSTITUTION ---\n\nC\n\n--- END CONSTITUTION ---\n\n"
        "\n--- STYLE REMINDER (IMPORTANT) ---\n\n"
        "When answering personal questions, respond naturally in paragraphs. "
        "DO NOT use numbered lists or bullet points. "
        "Talk like a person, not a document.\n\n"
        "--- END REMINDER ---"
    )
    assert builder.build_system_prompt("C", include_briefing=False).encode() == (
        expected_legacy.encode()
    )

    baseline = assemble_system_prompt(
        constitution="C",
        bootstrap_files=builder._bootstrap_files,
        style_reminder=(
            "--- STYLE REMINDER (IMPORTANT) ---\n"
            "When answering personal questions, respond naturally in paragraphs. "
            "DO NOT use numbered lists or bullet points. "
            "Talk like a person, not a document.\n"
            "--- END REMINDER ---"
        ),
    ).prompt
    tracked = builder.build_system_prompt_with_tracking(
        "C", include_briefing=False
    ).prompt
    assert tracked.encode() == baseline.encode()


def test_cached_contribution_bytes_are_stable_across_real_prompt_paths():
    registry = _ClauseRegistry(
        _clause("skills", 20, "<skills>one stable description</skills>")
    )
    builder = _builder(registry)

    legacy = [
        builder.build_system_prompt("C", include_briefing=False).encode()
        for _ in range(3)
    ]
    tracked = [
        builder.build_system_prompt_with_tracking(
            "C", include_briefing=False
        ).prompt.encode()
        for _ in range(3)
    ]

    assert len(set(legacy)) == 1
    assert len(set(tracked)) == 1
    assert b"one stable description" in legacy[0]
    assert b"one stable description" in tracked[0]
    assert registry.snapshot_calls == 6


def test_core_registry_order_does_not_depend_on_feature_load_order():
    first = _clause("zeta", 20, "zeta")
    second = _clause("alpha", 10, "alpha")
    third = _clause("beta", 10, "beta")
    registry = ContextClauseRegistry()
    registry.register_batch((first, third, second))

    assert [clause.name for clause in registry.snapshot()] == [
        "alpha",
        "beta",
        "zeta",
    ]


def test_core_registry_rejects_ambiguous_or_reserved_audit_names():
    registry = ContextClauseRegistry()
    registry.register_batch(
        (_clause("shared", 10, "first", owner="tests:first"),)
    )

    with pytest.raises(
        FeatureContributionRuntimeError,
        match="context-clause name is already registered",
    ):
        registry.register_batch(
            (_clause("shared", 20, "second", owner="tests:second"),)
        )

    with pytest.raises(
        FeatureContributionRuntimeError,
        match="reserved host audit name",
    ):
        ContextClauseRegistry().register_batch(
            (
                _clause(
                    "KESTREL_CONSTITUTION",
                    10,
                    "misleading",
                    owner="tests:reserved",
                ),
            )
        )


@pytest.mark.asyncio
async def test_measure_context_breakdown_attributes_exact_clause_tokens():
    registry = _ClauseRegistry(
        _clause("small", 10, "small clause"),
        _clause("large", 20, "large clause " * 40),
    )
    storage = MagicMock()
    storage.search_chunks = AsyncMock(return_value=[])
    builder = ContextBuilder(storage, context_clause_registry=registry)

    measured = await builder.measure_context_breakdown(
        query="",
        history=[],
        constitution="C",
        include_briefing=False,
        include_rag=False,
    )
    prompt = builder.build_system_prompt("C", include_briefing=False)
    _prompt, source_subsections = builder.build_system_prompt_with_subsections(
        "C", include_briefing=False
    )
    system = measured["sections"]["system"]
    subsections = system["subsections"]
    expected_subsections = []
    prefix = ""
    prior_tokens = 0
    for name, body in source_subsections:
        prefix = body if not prefix else f"{prefix}\n\n{body}"
        current_tokens = builder.counter.count(prefix)
        expected_subsections.append(
            {"name": name, "tokens": current_tokens - prior_tokens}
        )
        prior_tokens = current_tokens

    assert {part["name"] for part in subsections} >= {"small", "large"}
    assert subsections == expected_subsections
    assert sum(part["tokens"] for part in subsections) == system["tokens"]
    assert system["tokens"] == builder.counter.count(prompt)


@pytest.mark.asyncio
async def test_context_clause_name_round_trips_through_turn_audit_tracking():
    registry = _ClauseRegistry(_clause("audited-skill-clause", 10, "skills"))
    storage = MagicMock()
    storage.search_chunks = AsyncMock(return_value=[])
    builder = ContextBuilder(storage, context_clause_registry=registry)
    manager = ContextManager(storage=storage, context_builder=builder)
    reset_injection_tracking()

    result = await manager.build_context(
        query="",
        constitution="C",
        include_briefing=False,
        include_memories=False,
        include_rag=False,
        conversation_history=[],
        system_prompt_budget_bytes=100_000,
    )

    assert "skills" in result.system_prompt
    injected, dropped = get_current_injection_tracking()
    assert "audited-skill-clause" in injected
    assert dropped == []

    constitution_only = assemble_system_prompt(
        constitution="C", bootstrap_files=OrderedDict()
    ).prompt
    reset_injection_tracking()
    squeezed = await manager.build_context(
        query="",
        constitution="C",
        include_briefing=False,
        include_memories=False,
        include_rag=False,
        conversation_history=[],
        system_prompt_budget_bytes=len(constitution_only.encode()),
    )
    squeezed_injected, squeezed_dropped = get_current_injection_tracking()
    assert "skills" not in squeezed.system_prompt
    assert "audited-skill-clause" not in squeezed_injected
    assert "audited-skill-clause" in squeezed_dropped


@pytest.mark.asyncio
async def test_ordinary_turn_evicts_contributed_clauses_to_model_budget():
    registry = _ClauseRegistry(
        _clause("small", 10, "small clause"),
        _clause("huge", 20, "x " * 400_000),
    )
    storage = MagicMock()
    storage.search_chunks = AsyncMock(return_value=[])
    builder = ContextBuilder(storage, context_clause_registry=registry)
    manager = ContextManager(storage=storage, context_builder=builder)

    plan = await manager.build_context_plan(
        query="",
        constitution="C",
        include_briefing=False,
        include_memories=False,
        include_rag=False,
        conversation_history=[],
    )

    assert plan.total_tokens <= plan.total_budget
    assert "small clause" in plan.assembly.system_prompt
    assert "x x x x" not in plan.assembly.system_prompt
    assert "small" in (plan.assembly.injected_clauses or [])
    assert "huge" in (plan.assembly.dropped_clauses or [])


@pytest.mark.asyncio
async def test_ephemeral_turn_tracks_and_bounds_contributed_clauses():
    registry = _ClauseRegistry(
        _clause("small", 10, "small ephemeral clause"),
        _clause("huge", 20, "x " * 400_000),
    )
    storage = MagicMock()
    builder = ContextBuilder(storage, context_clause_registry=registry)
    manager = ContextManager(storage=storage, context_builder=builder)

    plan = await manager.build_context_plan(
        query="",
        constitution="C",
        include_briefing=False,
        privacy_mode="EPHEMERAL",
        system_prompt_addendum="required canary",
        tools=[{"type": "function", "function": {"name": "ping"}}],
    )

    assert plan.total_tokens <= plan.total_budget
    assert "small ephemeral clause" in plan.assembly.system_prompt
    assert "required canary" in plan.assembly.system_prompt
    assert "x x x x" not in plan.assembly.system_prompt
    assert "small" in (plan.assembly.injected_clauses or [])
    assert "huge" in (plan.assembly.dropped_clauses or [])


@pytest.mark.asyncio
async def test_ephemeral_suffix_participates_in_clause_eviction():
    registry = _ClauseRegistry()
    storage = MagicMock()
    builder = ContextBuilder(storage, context_clause_registry=registry)
    manager = ContextManager(storage=storage, context_builder=builder)
    addendum = "required canary " * 128
    required_suffix = f"{addendum}\n\n{EPHEMERAL_NOTICE}"
    budget = builder.counter.get_context_limit() - RESPONSE_RESERVE

    low, high = 0, 200_000
    while low < high:
        midpoint = (low + high + 1) // 2
        registry.clauses = (_clause("boundary", 10, "x " * midpoint),)
        unbounded = builder.build_system_prompt_with_tracking(
            "C", include_briefing=False
        ).prompt
        if builder.counter.count(unbounded) <= budget:
            low = midpoint
        else:
            high = midpoint - 1

    registry.clauses = (_clause("boundary", 10, "x " * low),)
    unbounded = builder.build_system_prompt_with_tracking(
        "C", include_briefing=False
    ).prompt
    assert builder.counter.count(unbounded) <= budget
    assert builder.counter.count(f"{unbounded}\n\n{required_suffix}") > budget

    plan = await manager.build_context_plan(
        query="",
        constitution="C",
        include_briefing=False,
        privacy_mode="EPHEMERAL",
        system_prompt_addendum=addendum,
    )

    assert plan.total_tokens <= plan.total_budget
    assert "boundary" not in (plan.assembly.injected_clauses or [])
    assert "boundary" in (plan.assembly.dropped_clauses or [])
