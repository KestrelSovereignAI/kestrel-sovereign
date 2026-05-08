"""Cross-seam regression test for the propose → approve → ticket flow.

The bug Nellie flagged: ``propose_improvement`` returns a proposal
ID, but ``create_improvement_ticket`` only accepted insight IDs, so
the user-visible workflow ``propose → approve → ticket`` silently
broke at the seam between two features inside Reflection.

Per Nellie's review feedback: "the bug is in the seam between
features, so unit tests inside only one feature will miss it."
This test covers the full path with a real database, real
ReflectionFeature, real GitHub-client stub, and a SecurityFeature
whose approval queue grants automatically.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.reflection.feature import ReflectionFeature
from kestrel_sovereign.features.reflection.models import ChangeType
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.async_database import AsyncDatabase


def _make_security(auto_approve: bool = True):
    sec = SimpleNamespace()
    sec.name = "SecurityFeature"
    sec.approval_queue = SimpleNamespace(
        request_approval=AsyncMock(return_value=(auto_approve, "user")),
    )
    return sec


def _make_github_feature(issue_url: str):
    client = MagicMock()
    client._configured = True
    client.create_issue = AsyncMock(
        return_value={"html_url": issue_url, "number": 999, "id": 12345},
    )
    return SimpleNamespace(name="GitHubFeature", client=client)


@pytest.fixture
async def wired_feature(tmp_path, monkeypatch):
    """Build a real ReflectionFeature with the dependencies it needs."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_propose_to_ticket_test_token")

    db = await AsyncDatabase.sqlite(str(tmp_path / "reflection.db"))
    sec = _make_security(auto_approve=True)
    gh = _make_github_feature("https://github.com/x/y/issues/999")

    agent = SimpleNamespace(
        features={"SecurityFeature": sec, "GitHubFeature": gh},
        did="did:test:propose-to-ticket",
        agent_id="did:test:propose-to-ticket",
        _raw_storage=SimpleNamespace(db=db),
    )
    agent.get_feature = lambda name: KestrelAgent.get_feature(agent, name)

    feat = ReflectionFeature(agent=agent)
    await feat.initialize()

    yield feat, sec, gh

    await db.close()


@pytest.mark.asyncio
async def test_propose_then_create_ticket_with_proposal_id(wired_feature):
    """The full transcript path: propose, approve, file ticket using
    the proposal ID. Pre-fix, the ticket call returned ``Insight not
    found`` because proposals weren't stored as insights.
    """
    feat, sec, gh = wired_feature

    propose_result = await feat.propose_improvement(
        title="Fix the proposal-to-ticket seam",
        description="Cross-feature contract bug surfaced in transcript review",
        change_type=ChangeType.BEHAVIOR.value,
        proposed_change="Allow create_improvement_ticket to consume proposal IDs",
    )

    assert propose_result.status is ToolResultStatus.OK
    assert propose_result.data["approved"] is True
    proposal_id = propose_result.data["proposal_id"]
    assert proposal_id

    ticket_result = await feat.create_improvement_ticket(proposal_id)

    assert ticket_result.status is ToolResultStatus.OK, ticket_result
    assert ticket_result.data["state"] == "ticket_created"
    assert ticket_result.data["source"] == "proposal"
    assert ticket_result.data["proposal_id"] == proposal_id
    assert ticket_result.data["issue_url"] == "https://github.com/x/y/issues/999"

    gh.client.create_issue.assert_awaited_once()
    issued_kwargs = gh.client.create_issue.await_args.kwargs
    assert "Agent Proposal" in issued_kwargs["title"]
    assert proposal_id in issued_kwargs["body"]


@pytest.mark.asyncio
async def test_unapproved_proposal_blocks_ticket(tmp_path, monkeypatch):
    """If the proposal was rejected, create_improvement_ticket must
    refuse — separation of self-modification approval from the
    external-write approval was deliberate per Nellie's feedback.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    db = await AsyncDatabase.sqlite(str(tmp_path / "reflection.db"))
    try:
        sec = _make_security(auto_approve=False)
        gh = _make_github_feature("https://example.invalid/should-not-be-used")
        agent = SimpleNamespace(
            features={"SecurityFeature": sec, "GitHubFeature": gh},
            did="did:test:rejected", agent_id="did:test:rejected",
            _raw_storage=SimpleNamespace(db=db),
        )
        agent.get_feature = lambda name: KestrelAgent.get_feature(agent, name)
        feat = ReflectionFeature(agent=agent)
        await feat.initialize()

        proposal_result = await feat.propose_improvement(
            title="Will be denied",
            description="x",
            change_type=ChangeType.BEHAVIOR.value,
            proposed_change="x",
        )
        # Rejected proposal → PARTIAL (recorded but not applied; agent
        # must speak the rejection per #1042 honesty layer 4).
        assert proposal_result.status is ToolResultStatus.PARTIAL
        assert proposal_result.data["approved"] is False
        proposal_id = proposal_result.data["proposal_id"]

        ticket_result = await feat.create_improvement_ticket(proposal_id)

        assert ticket_result.status is ToolResultStatus.ERROR
        assert ticket_result.data["state"] == "proposal_not_approved"
        assert ticket_result.data["proposal_id"] == proposal_id
        gh.client.create_issue.assert_not_awaited()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unknown_id_returns_clear_not_found(wired_feature):
    feat, _sec, _gh = wired_feature

    result = await feat.create_improvement_ticket("not-a-real-id")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["state"] == "not_found"
    assert "not-a-real-id" in result.error
