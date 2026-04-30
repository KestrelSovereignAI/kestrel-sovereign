"""Unit tests for GitHubFeature.create_github_issue_comment.

Closes the asymmetric-tool-seam gap: GitHubFeature could read issues
but had no way to leave durable feedback. The new tool is approval
gated and audit-friendly. These tests deliberately wire the agent
the way KestrelAgent does (``features={"SecurityFeature": ...}``) so
the security lookup is exercised — the original code_edit/reflection
tests stubbed approval at the function level and missed an entire
class of bugs (#TBD).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.features.github.client import GitHubClientError
from kestrel_sovereign.features.github.feature import GitHubFeature
from kestrel_sovereign.kestrel_agent import KestrelAgent


def _make_agent(approve: bool = True):
    """Build an agent stub with a SecurityFeature whose approval_queue
    grants or denies based on the ``approve`` flag.
    """
    sec = SimpleNamespace(
        name="SecurityFeature",
        tool_name="security",
        approval_queue=SimpleNamespace(
            request_approval=AsyncMock(return_value=(approve, "user")),
        ),
    )
    agent = SimpleNamespace(features={"SecurityFeature": sec})
    agent.get_feature = lambda name: KestrelAgent.get_feature(agent, name)
    return agent, sec


@pytest.fixture
def feature_factory():
    def _make(approve: bool = True):
        agent, sec = _make_agent(approve=approve)
        feat = GitHubFeature(agent=agent)
        # Avoid touching the real httpx client; the tests stub
        # `client.create_issue_comment` directly.
        feat._client = MagicMock()
        feat._client.create_issue_comment = AsyncMock(
            return_value={
                "html_url": "https://github.com/x/y/issues/1#issuecomment-99",
                "id": 99,
            },
        )
        return feat, sec

    return _make


@pytest.mark.asyncio
async def test_empty_body_refused_before_approval(feature_factory):
    feat, sec = feature_factory()

    result = await feat.create_github_issue_comment(
        issue_number=1, body="   ", repo="x/y",
    )

    assert result["success"] is False
    assert "empty" in result["error"].lower()
    sec.approval_queue.request_approval.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_body_refused_before_approval(feature_factory):
    feat, sec = feature_factory()

    result = await feat.create_github_issue_comment(
        issue_number=1, body="a" * 60_001, repo="x/y",
    )

    assert result["success"] is False
    assert "too long" in result["error"].lower()
    sec.approval_queue.request_approval.assert_not_awaited()


@pytest.mark.asyncio
async def test_approval_denied_does_not_post(feature_factory):
    feat, sec = feature_factory(approve=False)

    result = await feat.create_github_issue_comment(
        issue_number=1, body="hello", repo="x/y",
    )

    assert result["success"] is False
    assert result["requires_approval"] is True
    assert result["body_sha256"]
    sec.approval_queue.request_approval.assert_awaited_once()
    feat._client.create_issue_comment.assert_not_awaited()


@pytest.mark.asyncio
async def test_dry_run_does_not_post(feature_factory):
    feat, sec = feature_factory(approve=True)

    result = await feat.create_github_issue_comment(
        issue_number=1, body="preview please", repo="x/y", dry_run=True,
    )

    assert result["success"] is True
    assert result["preview"] is True
    assert result["body"] == "preview please"
    feat._client.create_issue_comment.assert_not_awaited()


@pytest.mark.asyncio
async def test_approved_call_posts_and_returns_url(feature_factory):
    feat, _sec = feature_factory(approve=True)

    result = await feat.create_github_issue_comment(
        issue_number=42, body="ship it", repo="x/y",
    )

    assert result["success"] is True
    assert result["html_url"].endswith("issuecomment-99")
    assert result["id"] == 99
    assert result["repo"] == "x/y"
    assert result["issue_number"] == 42
    feat._client.create_issue_comment.assert_awaited_once_with(
        "x/y", 42, "ship it",
    )


@pytest.mark.asyncio
async def test_self_repo_alias_resolves(feature_factory):
    feat, _sec = feature_factory(approve=True)

    with patch(
        "kestrel_sovereign.features.github.feature.GITHUB_SELF_REPO",
        "KestrelSovereignAI/kestrel-sovereign",
    ):
        result = await feat.create_github_issue_comment(
            issue_number=750, body="design note", repo="self",
        )

    assert result["success"] is True
    assert result["repo"] == "KestrelSovereignAI/kestrel-sovereign"
    posted_repo, _, _ = feat._client.create_issue_comment.await_args.args
    assert posted_repo == "KestrelSovereignAI/kestrel-sovereign"


@pytest.mark.asyncio
async def test_client_error_surfaces(feature_factory):
    feat, _sec = feature_factory(approve=True)
    feat._client.create_issue_comment = AsyncMock(
        side_effect=GitHubClientError("Issue not found", 404),
    )

    result = await feat.create_github_issue_comment(
        issue_number=999, body="hi", repo="x/y",
    )

    assert result["success"] is False
    assert "Issue not found" in result["error"]
    assert result["body_sha256"]


@pytest.mark.asyncio
async def test_no_security_feature_blocks_post():
    """If SecurityFeature is missing, the tool refuses rather than
    posting unapproved.
    """
    agent = SimpleNamespace(features={})
    agent.get_feature = lambda name: KestrelAgent.get_feature(agent, name)
    feat = GitHubFeature(agent=agent)
    feat._client = MagicMock()
    feat._client.create_issue_comment = AsyncMock()

    result = await feat.create_github_issue_comment(
        issue_number=1, body="hello", repo="x/y",
    )

    assert result["success"] is False
    assert result["requires_approval"] is True
    feat._client.create_issue_comment.assert_not_awaited()
