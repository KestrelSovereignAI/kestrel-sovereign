"""End-to-end regression tests for #1923: agent-facing @tool params validated
against a fixed set now normalize the synonyms LLMs reach for, while genuine
typos still fail with a value-listing error.

Covers a representative sample across features; the shared helper itself is
unit-tested in test_enum_coerce.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus


# ---------------------------------------------------------------------------
# save.item_type — synonyms/plurals map onto canonical SavedItemType values
# ---------------------------------------------------------------------------
class TestSaveItemType:
    def test_synonyms_and_plurals_normalize(self):
        from kestrel_sovereign.features.save.feature import SaveFeature

        cases = {
            "files": "file", "Document": "file", "doc": "file",
            "excerpts": "excerpt", "Snippet": "excerpt",
            "stashes": "stash",
            "records": "structured", "json": "structured", "STRUCTURED": "structured",
        }
        for raw, canon in cases.items():
            assert SaveFeature._normalize_item_type(raw) == canon, raw

    def test_typo_passes_through_for_validator(self):
        from kestrel_sovereign.features.save.feature import SaveFeature

        norm = SaveFeature._normalize_item_type("widget")
        assert norm == "widget"
        assert norm not in SaveFeature._VALID_ITEM_TYPES


# ---------------------------------------------------------------------------
# strategic_memory.strategy_add_blocker — severity synonyms (canonical middle
# value here is ``medium``, NOT todo's ``normal`` — proves per-domain maps)
# ---------------------------------------------------------------------------
def _make_strategic():
    from pathlib import Path

    from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature
    from kestrel_sovereign.features.strategic_memory.feature import _SaveOutcome

    feat = StrategicMemoryFeature(agent=MagicMock())
    feat._data = {}
    feat._strategy_path = Path("/tmp/kestrel-test/STRATEGY.yaml")
    feat._save = MagicMock(return_value=_SaveOutcome(persisted=True))
    return feat


class TestStrategicSeverity:
    @pytest.mark.asyncio
    async def test_moderate_normalizes_to_medium(self):
        feat = _make_strategic()
        result = await feat.strategy_add_blocker(issue="X", title="t", severity="moderate")
        assert result.status is ToolResultStatus.OK, result
        assert feat._data["blockers"][-1]["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_medium_is_canonical_and_accepted(self):
        feat = _make_strategic()
        result = await feat.strategy_add_blocker(issue="X", title="t", severity="MEDIUM")
        assert result.status is ToolResultStatus.OK
        assert feat._data["blockers"][-1]["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_genuine_typo_rejected_with_listing(self):
        feat = _make_strategic()
        result = await feat.strategy_add_blocker(issue="X", title="t", severity="sev->9")
        assert result.status is ToolResultStatus.ERROR
        assert "Must be one of: low, medium, high, critical" in (result.error or "")


# ---------------------------------------------------------------------------
# memory.recall_action_items — status synonyms (completed -> done)
# ---------------------------------------------------------------------------
def _make_memory():
    from kestrel_sovereign.features.memory.feature import MemoryFeature

    agent = MagicMock()
    agent.agent_id = "did:test:mem"
    agent.did = agent.agent_id
    agent.storage = MagicMock()
    agent.storage.graph = MagicMock()
    agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(return_value=[])
    feat = MemoryFeature(agent)
    feat.agent_id = agent.agent_id  # normally set in initialize()
    return feat


class TestMemoryActionItemStatus:
    @pytest.mark.asyncio
    async def test_completed_synonym_accepted(self):
        feat = _make_memory()
        result = await feat.recall_action_items(status="completed")
        # 'completed' normalizes to 'done' -> passes validation -> OK (empty set)
        assert result.status is not ToolResultStatus.ERROR, result

    @pytest.mark.asyncio
    async def test_typo_status_rejected_with_listing(self):
        feat = _make_memory()
        result = await feat.recall_action_items(status="finishedd")
        assert result.status is ToolResultStatus.ERROR
        assert "status must be one of: pending, done, cancelled" in (result.error or "")


# ---------------------------------------------------------------------------
# Per-domain alias maps stay distinct — the same synonym means different things
# in different features (the bug class #1923 is built on).
# ---------------------------------------------------------------------------
class TestPerDomainAliasesDoNotCollide:
    def test_medium_maps_differently_per_domain(self):
        from kestrel_sovereign.features.enum_coerce import (
            LOW_NORMAL_HIGH_URGENT_ALIASES,
            normalize_choice,
        )
        from kestrel_sovereign.features.strategic_memory.feature import _SEVERITY_ALIASES

        # todo/restart priority taxonomy: medium -> normal
        assert normalize_choice("medium", LOW_NORMAL_HIGH_URGENT_ALIASES) == "normal"
        # strategic severity taxonomy: medium IS canonical -> stays medium
        assert normalize_choice("medium", _SEVERITY_ALIASES) == "medium"

    def test_identity_merge_mode_aliases(self):
        from kestrel_sovereign.features.enum_coerce import normalize_choice
        from kestrel_sovereign.features.identity.feature import _MERGE_MODE_ALIASES

        assert normalize_choice("overwrite", _MERGE_MODE_ALIASES) == "replace"
        assert normalize_choice("skip", _MERGE_MODE_ALIASES) == "skip_existing"
        assert normalize_choice("combine", _MERGE_MODE_ALIASES) == "merge"
