"""Regression tests for ReflectionFeature handler wiring.

When you call ``reflection.create_improvement_ticket(...)`` and get
``"Ticket handler not available - check configuration"``, that means
``self._ticket_handler`` is None. There are two reasons it can be
None:

1. Real configuration miss — no GitHub token, no database. Expected.
2. Init order — ``_init_database`` tried to wire the handler at a
   point where ``_init_optional_services`` had not yet built
   ``_ticket_creator``. THAT is a wiring bug, not a config issue,
   and it's what Nellie hit today.

These tests construct a ReflectionFeature with everything the
handler needs (db, GitHub feature, GITHUB_TOKEN) and assert
``_ticket_handler`` ends up populated. They would have failed
against the broken init order.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.features.reflection.feature import ReflectionFeature
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.async_database import AsyncDatabase


def _make_github_feature():
    """A GitHubFeature stub whose ``client`` quacks well enough for
    ``TicketCreator(client)`` to accept it.
    """
    client = MagicMock()
    client._configured = True
    client.create_issue = AsyncMock()
    return SimpleNamespace(name="GitHubFeature", tool_name="github", client=client)


@pytest.fixture
async def agent_with_storage(tmp_path, monkeypatch):
    """Build an agent stub that satisfies ReflectionFeature.initialize.

    The agent exposes ``_raw_storage.db`` (so resolve_feature_database
    finds a real AsyncDatabase), a registered SecurityFeature, and a
    GitHubFeature with a usable client. GITHUB_TOKEN is set so
    TicketCreator's config check passes.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token_for_wiring_tests")

    db = await AsyncDatabase.sqlite(str(tmp_path / "reflection.db"))

    sec = SimpleNamespace(name="SecurityFeature", tool_name="security")
    gh = _make_github_feature()

    agent = SimpleNamespace(
        features={"SecurityFeature": sec, "GitHubFeature": gh},
        did="did:test:wiring",
        agent_id="did:test:wiring",
        _raw_storage=SimpleNamespace(db=db),
    )
    agent.get_feature = lambda name: KestrelAgent.get_feature(agent, name)

    yield agent

    await db.close()


@pytest.mark.asyncio
async def test_ticket_handler_wired_when_dependencies_present(agent_with_storage):
    feat = ReflectionFeature(agent=agent_with_storage)
    await feat.initialize()

    assert feat._db_helper is not None, "db_helper must be built"
    assert feat._ticket_creator is not None, (
        "ticket_creator must be built when GitHub feature + token are present"
    )
    assert feat._ticket_handler is not None, (
        "ticket_handler must be wired AFTER ticket_creator and db_helper exist"
    )


@pytest.mark.asyncio
async def test_create_improvement_ticket_reaches_handler(agent_with_storage):
    """End-to-end: with a properly wired handler, the entry point
    delegates instead of returning ``Ticket handler not available``.

    Reaching the handler is the bar — we don't drive a real GitHub
    write, just assert we got past the wiring guard.
    """
    feat = ReflectionFeature(agent=agent_with_storage)
    await feat.initialize()

    with patch.object(
        feat._ticket_handler, "create_improvement_ticket",
        new=AsyncMock(return_value={"success": True, "issue_url": "ok"}),
    ) as mock_create:
        result = await feat.create_improvement_ticket("insight-id-stub")

    mock_create.assert_awaited_once_with("insight-id-stub")
    assert result == {"success": True, "issue_url": "ok"}


@pytest.mark.asyncio
async def test_ticket_handler_skipped_when_github_missing(tmp_path, monkeypatch):
    """If GitHub isn't configured, the handler is intentionally not
    wired — that's the legitimate ``check configuration`` case, not a
    bug. The error message in that case is fine.
    """
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)

    db = await AsyncDatabase.sqlite(str(tmp_path / "reflection.db"))
    try:
        sec = SimpleNamespace(name="SecurityFeature", tool_name="security")
        agent = SimpleNamespace(
            features={"SecurityFeature": sec},
            did="did:test:no-github",
            agent_id="did:test:no-github",
            _raw_storage=SimpleNamespace(db=db),
        )
        agent.get_feature = lambda name: KestrelAgent.get_feature(agent, name)

        feat = ReflectionFeature(agent=agent)
        await feat.initialize()

        assert feat._db_helper is not None
        assert feat._ticket_creator is None
        assert feat._ticket_handler is None

        result = await feat.create_improvement_ticket("any")
        assert result["success"] is False
        assert "not available" in result["error"].lower()
    finally:
        await db.close()
