"""Tests for the ``ci:`` Waitable provider (#2729).

A ``ci:<owner/repo#N>`` wait watches a GitHub PR's merge/CI-check state so a
merge/check wait can survive restart. The classification logic is pure
(``classify_ci_state`` / ``_check_verdict``) and tested here without a
network; ``poll`` is exercised with a stubbed fetch + token.
"""

from __future__ import annotations

import pytest

from kestrel_sdk.tools import Outcome

from kestrel_sovereign.features.scheduler.ci_wait_provider import (
    CIWaitable,
    _check_verdict,
    classify_ci_state,
    parse_ci_handle,
)


# ---------------------------------------------------------------------------
# parse_ci_handle
# ---------------------------------------------------------------------------


def test_parse_ci_handle_ok():
    assert parse_ci_handle("owner/repo#123") == ("owner/repo", 123)
    assert parse_ci_handle("  KestrelSovereignAI/kestrel-sovereign#2729  ") == (
        "KestrelSovereignAI/kestrel-sovereign",
        2729,
    )


@pytest.mark.parametrize(
    "bad",
    [
        "c3f404fb77df4b79b0508a68ea46bbb7",  # a bare A2A task id
        "owner/repo",  # no number
        "repo#12",  # no owner/
        "owner/repo#",  # empty number
        "owner/repo#abc",  # non-numeric
        "",
    ],
)
def test_parse_ci_handle_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_ci_handle(bad)


# ---------------------------------------------------------------------------
# _check_verdict
# ---------------------------------------------------------------------------


def test_check_verdict_none_when_no_checks():
    assert _check_verdict() == "none"
    assert _check_verdict({}, {}) == "none"


def test_check_verdict_pending_on_incomplete_run():
    assert _check_verdict({"check_runs": [{"name": "ci", "status": "in_progress"}]}) == "pending"


def test_check_verdict_pending_on_combined_pending():
    assert _check_verdict(None, {"state": "pending", "statuses": []}) == "pending"


def test_check_verdict_success():
    runs = {"check_runs": [
        {"name": "ci", "status": "completed", "conclusion": "success"},
        {"name": "lint", "status": "completed", "conclusion": "skipped"},
    ]}
    assert _check_verdict(runs, {"state": "success", "statuses": []}) == "success"


def test_check_verdict_failure_from_conclusion():
    runs = {"check_runs": [
        {"name": "ci", "status": "completed", "conclusion": "failure"},
    ]}
    assert _check_verdict(runs) == "failure"


def test_check_verdict_failure_from_legacy_status():
    assert _check_verdict(
        None, {"state": "failure", "statuses": [{"context": "cov", "state": "failure"}]}
    ) == "failure"


# ---------------------------------------------------------------------------
# classify_ci_state
# ---------------------------------------------------------------------------


def test_classify_merged_is_done():
    st = classify_ci_state({"state": "closed", "merged": True}, repo="o/r", number=1)
    assert st.outcome is Outcome.DONE
    assert st.data["merged"] is True


def test_classify_closed_unmerged_is_failed():
    st = classify_ci_state({"state": "closed", "merged": False}, repo="o/r", number=1)
    assert st.outcome is Outcome.FAILED


def test_classify_open_passing_checks_is_done():
    st = classify_ci_state(
        {"state": "open", "merged": False},
        check_runs={"check_runs": [
            {"name": "ci", "status": "completed", "conclusion": "success"},
        ]},
        combined_status={"state": "success"},
        repo="o/r", number=2,
    )
    assert st.outcome is Outcome.DONE


def test_classify_open_failing_checks_is_failed():
    st = classify_ci_state(
        {"state": "open", "merged": False},
        check_runs={"check_runs": [
            {"name": "ci", "status": "completed", "conclusion": "failure"},
        ]},
        repo="o/r", number=2,
    )
    assert st.outcome is Outcome.FAILED


def test_classify_open_running_checks_is_pending():
    st = classify_ci_state(
        {"state": "open", "merged": False},
        check_runs={"check_runs": [
            {"name": "ci", "status": "in_progress"},
        ]},
        repo="o/r", number=2,
    )
    assert st.outcome is Outcome.PENDING


def test_classify_open_no_ci_is_pending():
    """An open PR with no CI configured is not terminal — keep watching for a
    merge/close rather than fabricating a verdict."""
    st = classify_ci_state({"state": "open", "merged": False}, repo="o/r", number=2)
    assert st.outcome is Outcome.PENDING


# ---------------------------------------------------------------------------
# CIWaitable.owns_handle + poll
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_owns_handle_rejects_non_pr_ref():
    provider = CIWaitable(feature=None)
    assert await provider.owns_handle("c3f404fb77df4b79b0508a68ea46bbb7") is False


@pytest.mark.asyncio
async def test_owns_handle_allows_well_formed_ref():
    provider = CIWaitable(feature=None)
    # Well-formed → None (can't verify existence offline, fail open).
    assert await provider.owns_handle("owner/repo#12") is None


@pytest.mark.asyncio
async def test_poll_no_token_stays_pending(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    provider = CIWaitable(feature=None)
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.PENDING
    assert status.data["blocked"] == "auth"


@pytest.mark.asyncio
async def test_poll_network_error_stays_pending(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.signals.sources.github_pr_watch import PRWatchNetworkError

    provider = CIWaitable(feature=None)

    async def boom(repo, number, token):
        raise PRWatchNetworkError("dns down")

    provider._fetch = boom
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.PENDING
    assert status.data["blocked"] == "network"


@pytest.mark.asyncio
async def test_poll_merged_is_done(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    provider = CIWaitable(feature=None)

    async def fake_fetch(repo, number, token):
        return {"state": "closed", "merged": True}, None, None

    provider._fetch = fake_fetch
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.DONE
    assert status.data["repo"] == "owner/repo"
    assert status.data["number"] == 7
