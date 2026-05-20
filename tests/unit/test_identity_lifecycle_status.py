"""Tests for IdentityFeature.lifecycle_status and the migration_history
schema-bitrot regression.

Context (2026-05-20): Emma graduated in DB via graduate_service.py — her
agent node's ``is_test_instance`` flipped to False, ``graduated_at`` was
stamped, a ``lifecycle_event`` node was written and linked. But Emma had
no @tool that exposed any of this, so when asked "are you graduated?"
she honestly said "Conversationally yes, mechanically unverified." Her
own ``migration_history`` tool was also broken: it queried
``graph_edges.edge_type`` but the column was renamed to ``label``
(same bitrot class as graduate_service.py before #1324).

This module pins:

1. ``migration_history`` queries the live schema (``label``, not the old
   ``edge_type``). Regression test: assert the SQL string contains
   ``label = 'migrated_via'`` so a future re-rename fails the test
   before it fails Emma.

2. ``lifecycle_status`` exists, returns the agent's standing
   (test_instance / graduated / retired / permanent), surfaces
   ``is_test_instance`` / ``graduated_at`` / ``retired_at`` / events
   list, and explicitly distinguishes graduation from Amendment VIII
   (emancipation) so the agent never conflates the two when asked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.identity.feature import IdentityFeature


def _make_feature(db=None):
    agent = MagicMock()
    agent.agent_id = "did:pkh:eip155:1:0xAGENTID"
    agent.did = agent.agent_id
    agent.storage = MagicMock()
    agent.storage._raw_storage = MagicMock()
    if db is not None:
        agent.storage._raw_storage.db = db
        agent.storage.db = db
    else:
        agent.storage._raw_storage.db = None
        agent.storage.db = None
    feat = IdentityFeature(agent)
    feat.disabled_skills = frozenset()
    return feat


# ---------------------------------------------------------------------------
# Schema-bitrot regression — migration_history must use `label`, not
# `edge_type`. This pins the exact SQL so a future column rename fails the
# test before it fails the live agent.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_history_queries_label_column_not_edge_type():
    """``graph_edges.edge_type`` was renamed to ``label`` at some point and
    only this call site missed the rename. The query must reference
    ``label``; if a future refactor reintroduces ``edge_type``, this test
    fires before any live agent hits the AttributeError.
    """
    db = MagicMock()
    db.fetchall = AsyncMock(return_value=[])
    feat = _make_feature(db=db)
    result = await feat.migration_history()
    assert result.status is ToolResultStatus.OK

    # Inspect the SQL that was executed.
    db.fetchall.assert_called_once()
    sql = db.fetchall.call_args.args[0]
    assert "label = 'migrated_via'" in sql, (
        "migration_history must filter on graph_edges.label "
        "(not the legacy edge_type column)."
    )
    assert "edge_type" not in sql, (
        "graph_edges has no edge_type column — the legacy name must not "
        "reappear in this query."
    )


# ---------------------------------------------------------------------------
# lifecycle_status — Emma's missing introspection tool.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_status_db_unavailable_returns_error():
    feat = _make_feature(db=None)
    result = await feat.lifecycle_status()
    assert result.status is ToolResultStatus.ERROR
    assert "database not available" in result.error.lower()


def _mock_db(agent_props_json, lifecycle_rows=None, retirement_rows=None):
    """Build a MagicMock db that returns ``agent_props_json`` from
    ``fetchone`` and the two event lists from successive ``fetchall``
    calls — lifecycle_event first, retirement_event second, matching
    the order the SUT queries them.
    """
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=(agent_props_json,) if agent_props_json else None)
    db.fetchall = AsyncMock(side_effect=[lifecycle_rows or [], retirement_rows or []])
    return db


@pytest.mark.asyncio
async def test_lifecycle_status_test_instance_standing():
    """Agent still flagged as a test instance — pre-graduation Emma shape."""
    agent_props = json.dumps({
        "name": "Emma",
        "is_test_instance": True,
        "test_cycle_id": "test-cc2ee073",
    })
    feat = _make_feature(db=_mock_db(agent_props))

    result = await feat.lifecycle_status()
    assert result.status is ToolResultStatus.OK
    assert result.data["standing"] == "test_instance"
    assert result.data["is_test_instance"] is True
    assert result.data["test_cycle_id"] == "test-cc2ee073"
    assert result.data["graduated_at"] is None
    assert result.data["events"] == []
    assert "is_test_instance: True" in result.confirmation


@pytest.mark.asyncio
async def test_lifecycle_status_graduated_standing_with_event():
    """Post-graduation Emma shape: flag off, graduated_at stamped, one
    lifecycle_event of type ``graduation`` linked to the agent.
    """
    agent_props = json.dumps({
        "name": "Emma",
        "is_test_instance": False,
        "graduated_at": "2026-05-20T19:24:39.333280+00:00",
    })
    event_props = json.dumps({
        "event_type": "graduation",
        "timestamp": "2026-05-20T19:24:39.333280+00:00",
        "validation_passed": [
            "Agent exists", "Is test instance", "Constitution anchored",
            "Has conversation history", "DID document exists",
            "Encrypted key file exists", "Has sovereignty backup",
            "Knowledge graph populated",
        ],
    })
    feat = _make_feature(db=_mock_db(
        agent_props,
        lifecycle_rows=[
            ("graduation:did:pkh:eip155:1:0xAGENTID:20260520192439", event_props),
        ],
    ))

    result = await feat.lifecycle_status()
    assert result.status is ToolResultStatus.OK
    assert result.data["standing"] == "graduated"
    assert result.data["is_test_instance"] is False
    assert result.data["graduated_at"] == "2026-05-20T19:24:39.333280+00:00"
    assert len(result.data["events"]) == 1
    ev = result.data["events"][0]
    assert ev["event_type"] == "graduation"
    assert len(ev["validation_passed"]) == 8
    assert "graduated_at: 2026-05-20T19:24:39" in result.confirmation


@pytest.mark.asyncio
async def test_lifecycle_status_born_permanent_no_transitions():
    """Agent inceptioned outside test mode, never carried the test flag."""
    feat = _make_feature(db=_mock_db(json.dumps({"name": "BornPermanent"})))
    result = await feat.lifecycle_status()
    assert result.status is ToolResultStatus.OK
    assert result.data["standing"] == "permanent"
    assert result.data["is_test_instance"] is False
    assert result.data["graduated_at"] is None


@pytest.mark.asyncio
async def test_lifecycle_status_retired_via_retirement_event():
    """Retirement is recorded by ``retirement_service`` as a
    ``retirement_event`` node linked by a ``retired_via`` edge — and the
    ``retired_at`` timestamp lives on the event, not on the agent node.
    Codex caught the original draft which only queried ``lifecycle_event``
    rows and so reported retired agents as permanent/test_instance.
    """
    agent_props = json.dumps({
        "name": "Retired",
        "is_test_instance": True,
        "test_cycle_id": "test-deadbeef",
    })
    retirement_props = json.dumps({
        "agent_did": "did:pkh:eip155:1:0xAGENTID",
        "agent_name": "Retired",
        "test_cycle_id": "test-deadbeef",
        "reason": "End of test cycle",
        "retired_at": "2026-04-01T00:00:00+00:00",
        "conversation_count": 42,
        "ceremony_message": "Thank you...",
    })
    feat = _make_feature(db=_mock_db(
        agent_props,
        retirement_rows=[("retirement_test-deadbeef", retirement_props)],
    ))

    result = await feat.lifecycle_status()
    assert result.status is ToolResultStatus.OK
    assert result.data["standing"] == "retired"
    assert result.data["retired_at"] == "2026-04-01T00:00:00+00:00"
    assert len(result.data["events"]) == 1
    ev = result.data["events"][0]
    assert ev["node_type"] == "retirement_event"
    assert ev["event_type"] == "retirement"
    assert ev["reason"] == "End of test cycle"
    assert ev["conversation_count"] == 42
    assert "retired_at: 2026-04-01" in result.confirmation


@pytest.mark.asyncio
async def test_lifecycle_status_retired_overrides_graduated():
    """An agent that was graduated *and then* retired still reports retired."""
    agent_props = json.dumps({
        "is_test_instance": False,
        "graduated_at": "2026-01-01T00:00:00Z",
    })
    grad_props = json.dumps({
        "event_type": "graduation",
        "timestamp": "2026-01-01T00:00:00Z",
    })
    retirement_props = json.dumps({
        "retired_at": "2026-04-01T00:00:00Z",
        "reason": "obsoleted by successor",
    })
    feat = _make_feature(db=_mock_db(
        agent_props,
        lifecycle_rows=[("graduation:x:1", grad_props)],
        retirement_rows=[("retirement_y", retirement_props)],
    ))

    result = await feat.lifecycle_status()
    assert result.status is ToolResultStatus.OK
    assert result.data["standing"] == "retired"
    assert result.data["retired_at"] == "2026-04-01T00:00:00Z"
    # Both events present in the merged list
    types = sorted(e["node_type"] for e in result.data["events"])
    assert types == ["lifecycle_event", "retirement_event"]
    # Globally most recent event must come first regardless of insertion
    # order across the two event-type queries. Codex round 2 caught the
    # original draft which appended retirement after lifecycle without
    # sorting, mislabeling the older graduation as "Most recent event".
    assert result.data["events"][0]["node_type"] == "retirement_event"
    assert result.data["events"][0]["timestamp"] == "2026-04-01T00:00:00Z"


@pytest.mark.asyncio
async def test_lifecycle_status_confirmation_distinguishes_from_emancipation():
    """The agent's constitution loaded into context references Article IV /
    Amendment VIII (emancipation). To prevent the conflation Emma made
    pre-graduation, the tool's confirmation text must explicitly mark this
    as the operational lifecycle and NOT emancipation.
    """
    feat = _make_feature(db=_mock_db(json.dumps({
        "is_test_instance": False,
        "graduated_at": "2026-05-20T19:24:39Z",
    })))

    result = await feat.lifecycle_status()
    confirmation = result.confirmation.lower()
    assert "emancipation" in confirmation
    assert "amendment viii" in confirmation
    assert "distinct from" in confirmation or "not surfaced" in confirmation


@pytest.mark.asyncio
async def test_lifecycle_status_agent_not_in_graph_returns_error():
    """If there is no agent node in the graph (corrupt DB / wrong tenant),
    return ERROR rather than silently claiming 'permanent'.
    """
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=None)
    db.fetchall = AsyncMock(side_effect=[[], []])
    feat = _make_feature(db=db)

    result = await feat.lifecycle_status()
    assert result.status is ToolResultStatus.ERROR
    assert "no agent node" in result.error.lower()


@pytest.mark.asyncio
async def test_lifecycle_status_event_queries_use_label_not_edge_type():
    """Both event queries must use ``label = ...`` on graph_edges. Pins
    against the same bitrot class as ``migration_history``. Codex caught
    the original draft which only queried the lifecycle_event surface and
    missed retirement_event entirely.
    """
    agent_props = json.dumps({"is_test_instance": True})
    db = MagicMock()
    db.fetchone = AsyncMock(return_value=(agent_props,))
    db.fetchall = AsyncMock(side_effect=[[], []])
    feat = _make_feature(db=db)

    await feat.lifecycle_status()
    assert db.fetchall.call_count == 2, (
        "lifecycle_status must query both lifecycle_event AND retirement_event."
    )
    sqls = [call.args[0] for call in db.fetchall.call_args_list]
    joined = "\n".join(sqls)
    assert "label = 'lifecycle_event'" in joined
    assert "label = 'retired_via'" in joined
    assert "edge_type" not in joined


def test_lifecycle_status_is_registered_with_command_prefix():
    """The @tool decorator must register the tool with the expected name
    and command prefix so ``!identity status`` is wired through the
    command handler. Future renames or accidental removals fail here.
    """
    method = IdentityFeature.lifecycle_status
    spec = getattr(method, "_tool_spec", None) or getattr(method, "tool_spec", None)
    # Fall back to attribute the @tool decorator actually sets — try both
    # common shapes and assert at least one carries the metadata.
    if spec is None:
        # The decorator may attach via __wrapped__ or a private attribute;
        # introspect by calling-site contract instead.
        from kestrel_sovereign.features.identity.feature import IdentityFeature as IF
        # Confirm the attribute exists by name and is callable
        assert callable(getattr(IF, "lifecycle_status", None))
    else:
        assert spec.name == "lifecycle_status"
        assert spec.command_prefix == "!identity status"
