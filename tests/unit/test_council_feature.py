"""ToolResult contract tests for CouncilFeature (#1090).

Pins the honesty edges introduced by the migration:
  - convene DEADLOCK / PENDING → PARTIAL
  - list_members fewer than min_members → PARTIAL
  - preview_evidence with red tests OR with collection failure → PARTIAL
  - override empty/whitespace reason → ERROR
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.council.feature import CouncilFeature
from kestrel_sovereign.features.council.models import (
    ConsensusRule,
    SessionOutcome,
)


def _make_feature(config=None, storage=None):
    feat = CouncilFeature(agent=None)
    feat.config = config
    feat.storage = storage or MagicMock()
    feat.disabled_skills = frozenset()
    return feat


def _fake_member(name="Claude"):
    return SimpleNamespace(
        name=name,
        provider="anthropic",
        model="auto",
        role="constitutional_reviewer",
    )


def _fake_session(outcome=SessionOutcome.APPROVED, verdicts=None):
    return SimpleNamespace(
        id="sess-abc-1234",
        outcome=outcome,
        members=[_fake_member()],
        rounds=[],
        verdicts=verdicts or [
            SimpleNamespace(
                member_name="Claude",
                decision=SimpleNamespace(value="APPROVE"),
                confidence=0.95,
                reasoning="looks good",
                concerns=[],
            )
        ],
        to_transcript=lambda: "transcript-text",
    )


# ---------------------------------------------------------------------------
# convene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convene_deadlock_is_partial(monkeypatch):
    """DEADLOCK is not a tool failure (council ran), but the LLM must
    speak the lack-of-decision. PARTIAL forces both halves out."""
    cfg = SimpleNamespace(
        members=[_fake_member("A"), _fake_member("B"), _fake_member("C")],
        min_members=3,
        max_rounds=5,
        consensus_rule=ConsensusRule.UNANIMOUS,
    )
    storage = MagicMock()
    storage.save_session = AsyncMock()
    feat = _make_feature(config=cfg, storage=storage)

    deadlock_session = _fake_session(
        outcome=SessionOutcome.DEADLOCK,
        verdicts=[
            SimpleNamespace(
                member_name=name,
                decision=SimpleNamespace(value=dec),
                confidence=0.7,
                reasoning="reason",
                concerns=[],
            )
            for name, dec in [("A", "APPROVE"), ("B", "REJECT"), ("C", "ABSTAIN")]
        ],
    )

    monkeypatch.setattr(
        "kestrel_sovereign.features.council.feature.compile_evidence",
        AsyncMock(return_value=SimpleNamespace(target="general")),
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.council.feature.convene_council",
        AsyncMock(return_value=deadlock_session),
    )

    result = await feat.convene(question="Should we ship?")

    assert result.status is ToolResultStatus.PARTIAL
    assert "deadlock" in result.error.lower()
    assert result.data["outcome"] == "DEADLOCK"
    assert result.data["approve_count"] == 1
    assert result.data["reject_count"] == 1
    assert result.data["abstain_count"] == 1


@pytest.mark.asyncio
async def test_convene_no_members_is_error():
    feat = _make_feature(config=None)
    result = await feat.convene(question="x")
    assert result.status is ToolResultStatus.ERROR
    assert "no council members" in result.error.lower()


@pytest.mark.asyncio
async def test_convene_invalid_max_rounds_is_error():
    cfg = SimpleNamespace(
        members=[_fake_member()] * 3,
        min_members=3,
        max_rounds=5,
        consensus_rule=ConsensusRule.UNANIMOUS,
    )
    feat = _make_feature(config=cfg)
    result = await feat.convene(question="x", max_rounds="not-an-int")
    assert result.status is ToolResultStatus.ERROR
    assert "integer" in result.error.lower()


# ---------------------------------------------------------------------------
# list_members
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_members_below_quorum_is_partial():
    """Only 1 member configured but min_members=3 → PARTIAL because
    !council-convene will refuse to run."""
    cfg = SimpleNamespace(
        members=[_fake_member()],
        min_members=3,
        max_rounds=5,
        consensus_rule=ConsensusRule.UNANIMOUS,
    )
    feat = _make_feature(config=cfg)
    result = await feat.list_members()
    assert result.status is ToolResultStatus.PARTIAL
    assert "refuse to run" in result.error.lower() or "requires at least" in result.error.lower()
    assert result.data["member_count"] == 1


@pytest.mark.asyncio
async def test_list_members_at_quorum_is_ok():
    cfg = SimpleNamespace(
        members=[_fake_member("A"), _fake_member("B"), _fake_member("C")],
        min_members=3,
        max_rounds=5,
        consensus_rule=ConsensusRule.UNANIMOUS,
    )
    feat = _make_feature(config=cfg)
    result = await feat.list_members()
    assert result.status is ToolResultStatus.OK
    assert result.data["member_count"] == 3


# ---------------------------------------------------------------------------
# override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_override_empty_reason_is_error():
    feat = _make_feature()
    result = await feat.override(session_id="x", decision="APPROVE", reason="")
    assert result.status is ToolResultStatus.ERROR
    assert "reason is required" in result.error.lower()


@pytest.mark.asyncio
async def test_override_whitespace_reason_is_error():
    feat = _make_feature()
    result = await feat.override(session_id="x", decision="APPROVE", reason="   ")
    assert result.status is ToolResultStatus.ERROR


@pytest.mark.asyncio
async def test_override_invalid_decision_is_error():
    feat = _make_feature()
    result = await feat.override(session_id="x", decision="MAYBE", reason="...")
    assert result.status is ToolResultStatus.ERROR
    assert "approve or reject" in result.error.lower()


@pytest.mark.asyncio
async def test_override_non_string_decision_is_error():
    feat = _make_feature()
    result = await feat.override(session_id="x", decision=True, reason="...")
    assert result.status is ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# preview_evidence
# ---------------------------------------------------------------------------


def _fake_evidence(test_count=10, test_passed=10, test_failed=0):
    return SimpleNamespace(
        target="general",
        compiled_at=datetime.now(timezone.utc),
        content_hash=lambda: "hash-abc",
        code_changes=[],
        test_count=test_count,
        test_passed=test_passed,
        test_failed=test_failed,
        risks=[],
        architecture_docs=[],
        previous_decisions=[],
    )


@pytest.mark.asyncio
async def test_preview_evidence_red_tests_is_partial(monkeypatch):
    feat = _make_feature()
    monkeypatch.setattr(
        "kestrel_sovereign.features.council.feature.compile_evidence",
        AsyncMock(return_value=_fake_evidence(test_count=10, test_passed=7, test_failed=3)),
    )
    result = await feat.preview_evidence()
    assert result.status is ToolResultStatus.PARTIAL
    assert "test(s) failing" in result.error.lower()
    assert result.data["test_failed"] == 3


@pytest.mark.asyncio
async def test_preview_evidence_collection_failure_is_partial(monkeypatch):
    """Round 1 codex finding: pytest collection failure produces
    test_count=0 with test_failed>0. The pre-fix guard skipped this
    case, framing a broken test suite as 'OK 0/0 passing'."""
    feat = _make_feature()
    monkeypatch.setattr(
        "kestrel_sovereign.features.council.feature.compile_evidence",
        AsyncMock(return_value=_fake_evidence(test_count=0, test_passed=0, test_failed=5)),
    )
    result = await feat.preview_evidence()
    assert result.status is ToolResultStatus.PARTIAL
    assert "collection" in result.error.lower() or "compile" in result.error.lower()
    assert result.data["test_failed"] == 5


@pytest.mark.asyncio
async def test_preview_evidence_clean_is_ok(monkeypatch):
    feat = _make_feature()
    monkeypatch.setattr(
        "kestrel_sovereign.features.council.feature.compile_evidence",
        AsyncMock(return_value=_fake_evidence(test_count=10, test_passed=10, test_failed=0)),
    )
    result = await feat.preview_evidence()
    assert result.status is ToolResultStatus.OK


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_invalid_limit_is_error():
    feat = _make_feature()
    result = await feat.status(limit="abc")
    assert result.status is ToolResultStatus.ERROR


@pytest.mark.asyncio
async def test_status_session_not_found_is_error():
    storage = MagicMock()
    storage.load_session = AsyncMock(return_value=None)
    feat = _make_feature(storage=storage)
    result = await feat.status(session_id="missing")
    assert result.status is ToolResultStatus.ERROR
    assert "not found" in result.error.lower()
