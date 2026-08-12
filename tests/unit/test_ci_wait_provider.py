"""Tests for the ``ci:`` Waitable provider (#2729).

A ``ci:<owner/repo#N>`` wait watches a GitHub PR's merge/CI-check state so a
merge/check wait can survive restart. The classification logic is pure
(``classify_ci_state`` / ``_check_verdict``) and tested here without a
network; ``poll`` is exercised with a stubbed fetch + token.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sdk.tools import Outcome

from kestrel_sovereign.features.scheduler.ci_wait_provider import (
    CIWaitable,
    _check_verdict,
    _rollup_summary,
    classify_ci_state,
    latest_check_runs,
    parse_ci_handle,
)


def _aged_pr(**extra):
    """An open PR whose last change is well outside the empty-rollup grace."""
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    return {"state": "open", "merged": False, "updated_at": old, **extra}


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
    runs = {"check_runs": [{"name": "ci", "status": "in_progress"}]}
    assert _check_verdict(runs) == "pending"


def test_check_verdict_pending_on_combined_pending():
    """A combined state of pending counts only when a status backs it."""
    assert _check_verdict(
        None,
        {"state": "pending", "statuses": [{"context": "cov", "state": "pending"}]},
    ) == "pending"


def test_check_verdict_ignores_combined_pending_with_no_statuses():
    """#2939 root cause: GitHub returns ``state: pending`` with an EMPTY
    statuses list for any commit that carries no legacy commit statuses —
    i.e. every Actions-only repo. Reading that as a real verdict pinned the
    rollup to pending forever (observed for ~3h, and in the terminal
    payload, on kestrel-sovereign#2934)."""
    assert _check_verdict(
        None, {"state": "pending", "statuses": [], "total_count": 0}
    ) == "none"
    # ...and it must not out-vote a fully completed check-run rollup.
    runs = {"check_runs": [
        {"name": "ci", "status": "completed", "conclusion": "success"},
    ]}
    assert _check_verdict(
        runs, {"state": "pending", "statuses": [], "total_count": 0}
    ) == "success"


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


@pytest.mark.parametrize("conclusion", ["success", "skipped", "neutral"])
def test_check_verdict_completed_is_terminal_for_passing_conclusions(conclusion):
    """``completed`` is terminal for every non-blocking conclusion (#2939)."""
    runs = {"check_runs": [
        {"name": "ci", "status": "completed", "conclusion": conclusion},
    ]}
    assert _check_verdict(runs) == "success"


def test_check_verdict_cancelled_latest_run_still_fails():
    """``cancelled`` is an absence of evidence, not a pass — it stays a
    failure once dedup has dropped superseded auto-cancelled runs."""
    runs = {"check_runs": [
        {"name": "ci", "status": "completed", "conclusion": "cancelled"},
    ]}
    assert _check_verdict(runs) == "failure"


# ---------------------------------------------------------------------------
# Superseded-run resolution (latest workflow run per check name)
# ---------------------------------------------------------------------------


def _run(name, *, suite, started, run_id, status="completed", conclusion="success"):
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started_at": started,
        "id": run_id,
        "check_suite": {"id": suite},
        "app": {"slug": "github-actions"},
    }


# The rollup shape observed on kestrel-sovereign#2934 (head 362b6c0d): two
# Kestrel CI workflow runs on the same head SHA, three SKIPPED conclusions,
# duplicate check names across the two runs — and a combined status endpoint
# that reports ``pending`` with zero statuses.
PR_2934_CHECK_RUNS = {"check_runs": [
    _run("cancel-expensive-tiers-when-unit-fails", suite=85448015702,
         started="2026-08-11T14:48:25Z", run_id=93819371322, conclusion="skipped"),
    _run("unit-tests", suite=85448015702,
         started="2026-08-11T14:30:18Z", run_id=93813697997),
    _run("dependency-review", suite=85448015702,
         started="2026-08-11T14:29:13Z", run_id=93813351563),
    _run("lint-and-imports", suite=85448015702,
         started="2026-08-11T14:29:19Z", run_id=93813350865),
    # ...superseded run 31500405381, including a SKIPPED dependency-review.
    _run("cancel-expensive-tiers-when-unit-fails", suite=85443943592,
         started="2026-08-11T14:21:02Z", run_id=93810816042, conclusion="skipped"),
    _run("unit-tests", suite=85443943592,
         started="2026-08-11T14:14:47Z", run_id=93808888919),
    _run("dependency-review", suite=85443943592,
         started="2026-08-11T14:13:49Z", run_id=93808608794, conclusion="skipped"),
    _run("lint-and-imports", suite=85443943592,
         started="2026-08-11T14:13:52Z", run_id=93808608038),
]}

PR_2934_WORKFLOW_RUNS = {"workflow_runs": [
    {"id": 31501778694, "workflow_id": 42, "check_suite_id": 85448015702,
     "run_number": 2, "created_at": "2026-08-11T14:29:00Z"},
    {"id": 31500405381, "workflow_id": 42, "check_suite_id": 85443943592,
     "run_number": 1, "created_at": "2026-08-11T14:13:00Z"},
]}

PR_2934_COMBINED_STATUS = {"state": "pending", "statuses": [], "total_count": 0}


def test_pr_2934_rollup_is_terminal_success():
    """Regression for #2939: skipped conclusions + duplicate check names
    across two workflow runs + an empty legacy-status list must read as a
    terminal success, not a forever-pending rollup."""
    assert _check_verdict(
        PR_2934_CHECK_RUNS, PR_2934_COMBINED_STATUS, PR_2934_WORKFLOW_RUNS
    ) == "success"
    # Same verdict without the workflow-run listing (enrichment degraded).
    assert _check_verdict(PR_2934_CHECK_RUNS, PR_2934_COMBINED_STATUS) == "success"

    st = classify_ci_state(
        _aged_pr(mergeable=True, mergeable_state="clean"),
        check_runs=PR_2934_CHECK_RUNS,
        combined_status=PR_2934_COMBINED_STATUS,
        workflow_runs=PR_2934_WORKFLOW_RUNS,
        repo="KestrelSovereignAI/kestrel-sovereign", number=2934,
    )
    assert st.outcome is Outcome.DONE
    assert st.data["checks"] == "success"
    assert "contradiction" not in st.data
    # 4 distinct checks survive; the 4 superseded entries are reported.
    assert st.data["checks_total"] == 4
    assert st.data["checks_superseded"] == 4


def test_latest_check_runs_keeps_newest_per_name():
    resolved = latest_check_runs(PR_2934_CHECK_RUNS, PR_2934_WORKFLOW_RUNS)
    assert resolved.superseded == 4
    # Workflow identity was available, so no collapse was a guess.
    assert resolved.ambiguous == []
    kept = {r["name"]: r["id"] for r in resolved.runs}
    assert kept == {
        "cancel-expensive-tiers-when-unit-fails": 93819371322,
        "unit-tests": 93813697997,
        "dependency-review": 93813351563,
        "lint-and-imports": 93813350865,
    }


def test_superseded_cancelled_run_does_not_fail_the_rollup():
    """A concurrency-cancelled run from a superseded workflow run must not
    make the rollup read as a failure."""
    check_runs = {"check_runs": [
        _run("unit-tests", suite=2, started="2026-08-11T14:30:00Z", run_id=200),
        _run("unit-tests", suite=1, started="2026-08-11T14:10:00Z", run_id=100,
             conclusion="cancelled"),
    ]}
    workflow_runs = {"workflow_runs": [
        {"id": 20, "workflow_id": 7, "check_suite_id": 2, "run_number": 2,
         "created_at": "2026-08-11T14:29:00Z"},
        {"id": 10, "workflow_id": 7, "check_suite_id": 1, "run_number": 1,
         "created_at": "2026-08-11T14:09:00Z"},
    ]}
    assert _check_verdict(check_runs, None, workflow_runs) == "success"


def test_superseded_queued_run_does_not_pin_the_rollup_pending():
    check_runs = {"check_runs": [
        _run("unit-tests", suite=2, started="2026-08-11T14:30:00Z", run_id=200),
        # A stale entry from the superseded run that never left the queue.
        _run("unit-tests", suite=1, started="2026-08-11T14:10:00Z", run_id=100,
             status="queued", conclusion=None),
    ]}
    workflow_runs = {"workflow_runs": [
        {"id": 20, "workflow_id": 7, "check_suite_id": 2, "run_number": 2,
         "created_at": "2026-08-11T14:29:00Z"},
        {"id": 10, "workflow_id": 7, "check_suite_id": 1, "run_number": 1,
         "created_at": "2026-08-11T14:09:00Z"},
    ]}
    assert _check_verdict(check_runs, None, workflow_runs) == "success"


def test_same_name_in_two_workflows_is_not_collapsed():
    """Dedup groups by (workflow, name): two DIFFERENT workflows that both
    define a job called ``test`` are distinct checks, so a failure in the
    older-started one still blocks."""
    check_runs = {"check_runs": [
        _run("test", suite=2, started="2026-08-11T14:30:00Z", run_id=200),
        _run("test", suite=1, started="2026-08-11T14:10:00Z", run_id=100,
             conclusion="failure"),
    ]}
    workflow_runs = {"workflow_runs": [
        {"id": 20, "workflow_id": 7, "check_suite_id": 2, "run_number": 1,
         "created_at": "2026-08-11T14:29:00Z"},
        {"id": 10, "workflow_id": 9, "check_suite_id": 1, "run_number": 1,
         "created_at": "2026-08-11T14:09:00Z"},
    ]}
    resolved = latest_check_runs(check_runs, workflow_runs)
    assert resolved.superseded == 0
    assert len(resolved.runs) == 2
    assert _check_verdict(check_runs, None, workflow_runs) == "failure"


# ---------------------------------------------------------------------------
# Rule 5: a collapse the provider cannot prove must not decide the verdict
# ---------------------------------------------------------------------------


# Two DIFFERENT workflows on one head SHA that both define a job called
# ``test`` — the newer one passing, the older one failing. With the workflow
# listing this is unambiguous (see
# ``test_same_name_in_two_workflows_is_not_collapsed``); without it, grouping
# by check app merges them and the pass overwrites the failure.
TWO_WORKFLOWS_SHARED_NAME = {"check_runs": [
    _run("test", suite=2, started="2026-08-11T14:30:00Z", run_id=200),
    _run("test", suite=1, started="2026-08-11T14:10:00Z", run_id=100,
         conclusion="failure"),
]}


def test_unprovable_collapse_hiding_a_failure_is_indeterminate():
    """Regression for the #2939 review: a fine-grained token can hold Checks
    access without Actions read, so the workflow listing — the only proof
    that two same-named runs are attempts of ONE job — can be missing. The
    app-level fallback would collapse two different workflows' ``test`` jobs
    and let the newer pass bury the older failure. Report indeterminate."""
    summary = _rollup_summary(TWO_WORKFLOWS_SHARED_NAME)
    assert summary["verdict"] == "indeterminate"
    assert summary["unresolved"] == ["ambiguous:test"]
    # The proof restores the real verdict — the failure blocks.
    assert _check_verdict(TWO_WORKFLOWS_SHARED_NAME, None, {"workflow_runs": [
        {"id": 20, "workflow_id": 7, "check_suite_id": 2, "run_number": 1,
         "created_at": "2026-08-11T14:29:00Z"},
        {"id": 10, "workflow_id": 9, "check_suite_id": 1, "run_number": 1,
         "created_at": "2026-08-11T14:09:00Z"},
    ]}) == "failure"


def test_unprovable_collapse_hiding_a_pending_run_is_indeterminate():
    check_runs = {"check_runs": [
        _run("test", suite=2, started="2026-08-11T14:30:00Z", run_id=200),
        _run("test", suite=1, started="2026-08-11T14:10:00Z", run_id=100,
             status="in_progress", conclusion=None),
    ]}
    assert _check_verdict(check_runs) == "indeterminate"


def test_indeterminate_rollup_is_pending_not_done(caplog):
    """Never DONE on evidence the provider does not hold — a false pass is
    the polarity that lets a merge proceed on checks nobody read."""
    with caplog.at_level(logging.WARNING):
        st = classify_ci_state(
            _aged_pr(),
            check_runs=TWO_WORKFLOWS_SHARED_NAME,
            repo="o/r", number=2,
        )
    assert st.outcome is Outcome.PENDING
    assert st.data["checks"] == "indeterminate"
    assert st.data["checks_unresolved"] == ["ambiguous:test"]
    assert "indeterminate" in caplog.text


def test_unprovable_collapse_that_cannot_change_the_verdict_is_kept():
    """Ambiguity only blocks when it MATTERS. #2934's degraded rollup has
    duplicate names across two suites with no workflow listing, but every run
    is completed and passing, so collapsing them cannot hide anything —
    reporting indeterminate there would re-create the false-pending stall."""
    summary = _rollup_summary(PR_2934_CHECK_RUNS, PR_2934_COMBINED_STATUS)
    assert summary["verdict"] == "success"
    assert summary["unresolved"] == []


def test_repeated_attempts_in_one_suite_are_not_ambiguous():
    """Two runs sharing a check suite are one workflow run's attempts however
    the group was formed — the suite id is the proof the workflow listing
    would have supplied, so no listing is needed."""
    check_runs = {"check_runs": [
        _run("flaky", suite=5, started="2026-08-11T14:30:00Z", run_id=200),
        _run("flaky", suite=5, started="2026-08-11T14:10:00Z", run_id=100,
             conclusion="failure"),
    ]}
    resolved = latest_check_runs(check_runs)
    assert resolved.ambiguous == []
    assert _check_verdict(check_runs) == "success"


# ---------------------------------------------------------------------------
# Rule 4: a terminal pass requires the complete listing
# ---------------------------------------------------------------------------


def test_truncated_check_run_page_is_not_a_pass():
    """Regression for the #2939 review: GitHub caps ``per_page`` at 100, so
    ``total_count`` above the rows in hand proves a check was not read. The
    101st could be the queued or failed one."""
    payload = {
        "total_count": 101,
        "check_runs": [
            {"name": f"job-{i}", "status": "completed", "conclusion": "success"}
            for i in range(100)
        ],
    }
    summary = _rollup_summary(payload)
    assert summary["verdict"] == "indeterminate"
    assert summary["unresolved"] == ["incomplete:check_runs"]


def test_truncated_status_page_is_not_a_pass():
    summary = _rollup_summary(
        {"check_runs": [
            {"name": "ci", "status": "completed", "conclusion": "success"},
        ]},
        {"state": "success", "total_count": 3,
         "statuses": [{"context": "cov", "state": "success"}]},
    )
    assert summary["verdict"] == "indeterminate"
    assert summary["unresolved"] == ["incomplete:statuses"]


def test_truncated_page_does_not_mask_a_visible_failure():
    """More pages cannot un-fail a check, so a failure seen on page 1 is a
    sound terminal even though the listing is short."""
    payload = {
        "total_count": 200,
        "check_runs": [
            {"name": "ci", "status": "completed", "conclusion": "failure"},
        ],
    }
    assert _check_verdict(payload) == "failure"


def test_truncated_page_with_a_pending_run_stays_pending():
    payload = {
        "total_count": 200,
        "check_runs": [{"name": "ci", "status": "in_progress"}],
    }
    assert _check_verdict(payload) == "pending"


def test_complete_page_matching_total_count_is_a_pass():
    payload = {
        "total_count": 2,
        "check_runs": [
            {"name": "ci", "status": "completed", "conclusion": "success"},
            {"name": "lint", "status": "completed", "conclusion": "skipped"},
        ],
    }
    assert _check_verdict(payload) == "success"


def test_fetcher_truncation_marker_blocks_a_pass():
    """``_truncated`` is how the fetcher reports a listing it could not page
    to the end when the endpoint gave it no ``total_count`` to check."""
    payload = {
        "check_runs": [
            {"name": "ci", "status": "completed", "conclusion": "success"},
        ],
        "_truncated": True,
    }
    assert _check_verdict(payload) == "indeterminate"


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


def test_classify_open_no_ci_is_terminal_none():
    """An empty rollup on a settled PR is TERMINAL, not pending (#2939):
    nothing ran, so there is nothing to wait for. Reported as ``none``, never
    ``success`` — the waiter must be able to tell "everything passed" from
    "nothing ran"."""
    st = classify_ci_state(_aged_pr(), repo="o/r", number=2)
    assert st.outcome is Outcome.DONE
    assert st.data["checks"] == "none"
    assert st.data["checks_total"] == 0
    assert "awaiting_ci_registration" not in st.data


def test_classify_empty_rollup_within_grace_is_pending():
    """Bounded exception: right after a push, CI may not have registered its
    check runs yet — that window is provably still progressing, so it must
    not resolve DONE on an empty rollup."""
    fresh = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    st = classify_ci_state(
        {"state": "open", "merged": False, "updated_at": fresh},
        repo="o/r", number=2,
    )
    assert st.outcome is Outcome.PENDING
    assert st.data["checks"] == "none"
    assert st.data["awaiting_ci_registration"] is True


def test_classify_empty_rollup_grace_expires():
    """The grace is bounded by the clock, so the state always converges."""
    updated = "2026-08-11T14:00:00Z"
    pr = {"state": "open", "merged": False, "updated_at": updated}
    inside = classify_ci_state(
        pr, repo="o/r", number=2,
        now=datetime(2026, 8, 11, 14, 1, 0, tzinfo=timezone.utc),
    )
    assert inside.outcome is Outcome.PENDING
    outside = classify_ci_state(
        pr, repo="o/r", number=2,
        now=datetime(2026, 8, 11, 14, 10, 0, tzinfo=timezone.utc),
    )
    assert outside.outcome is Outcome.DONE
    assert outside.data["checks"] == "none"


def test_classify_clean_but_pending_surfaces_contradiction(caplog):
    """A PR GitHub calls clean/mergeable while the rollup says a check is
    outstanding is two reads disagreeing. Surface it — but never promote it
    to a terminal DONE, because ``clean`` is also what GitHub reports when a
    repo has no *required* checks and CI is merely queued (#2939)."""
    with caplog.at_level(logging.WARNING):
        st = classify_ci_state(
            _aged_pr(mergeable=True, mergeable_state="clean"),
            check_runs={"check_runs": [{"name": "ci", "status": "in_progress"}]},
            repo="o/r", number=2,
        )
    assert st.outcome is Outcome.PENDING
    assert st.data["contradiction"] == "clean_but_pending"
    assert "mergeable_state=clean" in caplog.text
    assert st.data["mergeable"] is True
    assert st.data["mergeable_state"] == "clean"
    assert st.data["checks_pending"] == ["ci"]


def test_classify_blocked_pending_has_no_contradiction():
    st = classify_ci_state(
        _aged_pr(mergeable=True, mergeable_state="blocked"),
        check_runs={"check_runs": [{"name": "ci", "status": "queued"}]},
        repo="o/r", number=2,
    )
    assert st.outcome is Outcome.PENDING
    assert "contradiction" not in st.data


def test_classify_merged_payload_reports_resolved_checks():
    """The terminal payload must not carry a stale ``checks: pending``: the
    field evidence on #2934 showed ``merged: True`` alongside a check verdict
    that had never resolved."""
    st = classify_ci_state(
        {"state": "closed", "merged": True, "updated_at": "2026-08-11T18:00:00Z"},
        check_runs=PR_2934_CHECK_RUNS,
        combined_status=PR_2934_COMBINED_STATUS,
        workflow_runs=PR_2934_WORKFLOW_RUNS,
        repo="KestrelSovereignAI/kestrel-sovereign", number=2934,
    )
    assert st.outcome is Outcome.DONE
    assert st.data["checks"] == "success"


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
        return {"state": "closed", "merged": True}, None, None, None

    provider._fetch = fake_fetch
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.DONE
    assert status.data["repo"] == "owner/repo"
    assert status.data["number"] == 7


@pytest.mark.asyncio
async def test_poll_open_pr_with_completed_checks_is_done(monkeypatch):
    """End-to-end shape of the #2939 regression through ``poll``."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    provider = CIWaitable(feature=None)

    async def fake_fetch(repo, number, token):
        return (
            _aged_pr(mergeable=True, mergeable_state="clean"),
            PR_2934_CHECK_RUNS,
            PR_2934_COMBINED_STATUS,
            PR_2934_WORKFLOW_RUNS,
        )

    provider._fetch = fake_fetch
    status = await provider.poll("KestrelSovereignAI/kestrel-sovereign#2934")
    assert status.outcome is Outcome.DONE
    assert status.data["checks"] == "success"


@pytest.mark.asyncio
async def test_poll_indeterminate_rollup_stays_pending(monkeypatch):
    """End-to-end: an unprovable collapse must not wake the wait as DONE."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    provider = CIWaitable(feature=None)

    async def fake_fetch(repo, number, token):
        return _aged_pr(), TWO_WORKFLOWS_SHARED_NAME, None, None

    provider._fetch = fake_fetch
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.PENDING
    assert status.data["checks"] == "indeterminate"


# ---------------------------------------------------------------------------
# Pagination (#2939 review): one request is not a rollup
# ---------------------------------------------------------------------------


def _page_of(name_prefix, start, count, **overrides):
    return [
        {"name": f"{name_prefix}-{i}", "status": "completed",
         "conclusion": "success", "id": i,
         "check_suite": {"id": 1}, "app": {"slug": "github-actions"},
         **overrides}
        for i in range(start, start + count)
    ]


def _paging_github_get(pages, calls):
    """Serve ``pages[endpoint][page_number]``; record every URL requested."""
    async def fake_get(url, *, token, timeout, ref):
        calls.append(url)
        if url.startswith("https://api.github.com/repos/owner/repo/pulls/"):
            return {
                "state": "open", "merged": False,
                "updated_at": "2026-08-11T14:00:00Z",
                "head": {"sha": "deadbeef"},
            }
        page = int(url.split("page=")[-1])
        for endpoint, by_page in pages.items():
            if endpoint in url:
                return by_page.get(page, by_page["empty"])
        raise AssertionError(f"unexpected URL {url}")

    return fake_get


@pytest.mark.asyncio
async def test_poll_follows_pagination_and_sees_a_second_page_failure(monkeypatch):
    """A failing check on page 2 must decide the verdict. Before this, only
    page 1 was fetched and a full first page of successes read as DONE."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.signals.sources import github_pr_watch

    calls = []
    monkeypatch.setattr(github_pr_watch, "_github_get", _paging_github_get({
        "check-runs": {
            1: {"total_count": 101, "check_runs": _page_of("job", 0, 100)},
            2: {"total_count": 101, "check_runs": [
                {"name": "late-job", "status": "completed",
                 "conclusion": "failure", "id": 999,
                 "check_suite": {"id": 1}, "app": {"slug": "github-actions"}},
            ]},
            "empty": {"total_count": 101, "check_runs": []},
        },
        "/status": {"empty": {"state": "pending", "statuses": [],
                              "total_count": 0}},
        "/actions/runs": {"empty": {"total_count": 0, "workflow_runs": []}},
    }, calls))

    provider = CIWaitable(feature=None)
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.FAILED
    assert status.data["checks_failed"] == ["late-job"]
    assert status.data["checks_total"] == 101
    assert any("page=2" in u for u in calls)


@pytest.mark.asyncio
async def test_poll_follows_pagination_and_sees_a_second_page_pending(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.signals.sources import github_pr_watch

    calls = []
    monkeypatch.setattr(github_pr_watch, "_github_get", _paging_github_get({
        "check-runs": {
            1: {"total_count": 101, "check_runs": _page_of("job", 0, 100)},
            2: {"total_count": 101, "check_runs": [
                {"name": "late-job", "status": "queued", "id": 999,
                 "check_suite": {"id": 1}, "app": {"slug": "github-actions"}},
            ]},
            "empty": {"total_count": 101, "check_runs": []},
        },
        "/status": {"empty": {"state": "pending", "statuses": [],
                              "total_count": 0}},
        "/actions/runs": {"empty": {"total_count": 0, "workflow_runs": []}},
    }, calls))

    provider = CIWaitable(feature=None)
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.PENDING
    assert status.data["checks_pending"] == ["late-job"]


@pytest.mark.asyncio
async def test_poll_stops_paging_on_a_short_page(monkeypatch):
    """A page shorter than ``per_page`` is the last one — no wasted request."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.signals.sources import github_pr_watch

    calls = []
    monkeypatch.setattr(github_pr_watch, "_github_get", _paging_github_get({
        "check-runs": {
            1: {"total_count": 2, "check_runs": _page_of("job", 0, 2)},
            "empty": {"total_count": 2, "check_runs": []},
        },
        "/status": {"empty": {"state": "success", "statuses": [],
                              "total_count": 0}},
        "/actions/runs": {"empty": {"total_count": 0, "workflow_runs": []}},
    }, calls))

    provider = CIWaitable(feature=None)
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.DONE
    assert status.data["checks"] == "success"
    assert not any("page=2" in u for u in calls)


@pytest.mark.asyncio
async def test_poll_paging_cap_reports_indeterminate_not_success(monkeypatch):
    """A listing that never ends is bounded by the page cap — and the rollup
    it produced is explicitly incomplete rather than a fabricated pass."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.features.scheduler import ci_wait_provider
    from kestrel_sovereign.signals.sources import github_pr_watch

    monkeypatch.setattr(ci_wait_provider, "_MAX_PAGES", 3)
    calls = []
    endless = {"check_runs": _page_of("job", 0, 100)}  # always a full page
    monkeypatch.setattr(github_pr_watch, "_github_get", _paging_github_get({
        "check-runs": {"empty": endless},
        "/status": {"empty": {"state": "pending", "statuses": [],
                              "total_count": 0}},
        "/actions/runs": {"empty": {"total_count": 0, "workflow_runs": []}},
    }, calls))

    provider = CIWaitable(feature=None)
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.PENDING
    assert status.data["checks"] == "indeterminate"
    assert status.data["checks_unresolved"] == ["incomplete:check_runs"]
    assert len([u for u in calls if "check-runs" in u]) == 3


@pytest.mark.asyncio
async def test_status_pagination_preserves_the_combined_state(monkeypatch):
    """Merging pages must keep page 1's non-list fields — the combined
    ``state`` sits beside the ``statuses`` list, not inside it."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.features.scheduler.ci_wait_provider import (
        _fetch_all_pages,
    )
    from kestrel_sovereign.signals.sources import github_pr_watch

    statuses = [{"context": f"cov-{i}", "state": "success"} for i in range(100)]
    pages = {
        1: {"state": "failure", "sha": "deadbeef", "total_count": 101,
            "statuses": statuses},
        2: {"state": "failure", "sha": "deadbeef", "total_count": 101,
            "statuses": [{"context": "late", "state": "failure"}]},
    }

    async def fake_get(url, *, token, timeout, ref):
        return pages[int(url.split("page=")[-1])]

    monkeypatch.setattr(github_pr_watch, "_github_get", fake_get)
    merged = await _fetch_all_pages(
        "https://api.github.com/repos/o/r/commits/deadbeef/status",
        key="statuses", token="t", ref="o/r#1 status",
    )
    assert merged["state"] == "failure"
    assert merged["sha"] == "deadbeef"
    assert len(merged["statuses"]) == 101
    assert merged["_truncated"] is False
    assert _check_verdict(None, merged) == "failure"


@pytest.mark.asyncio
async def test_paged_to_the_end_beats_a_disagreeing_total_count(monkeypatch, caplog):
    """Anti-stall guard: GitHub counts and serves different sets on the
    check-runs endpoint (pages apply ``filter=latest``; the count documents
    nothing). Once pagination has reached the end of the listing, a larger
    ``total_count`` must not veto the verdict forever — that would be the
    #2939 stall in a new shape. The disagreement is logged, not buried."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.features.scheduler.ci_wait_provider import (
        _fetch_all_pages,
    )
    from kestrel_sovereign.signals.sources import github_pr_watch

    async def fake_get(url, *, token, timeout, ref):
        # A short first page — the end of the listing — with an inflated count.
        return {"total_count": 400, "check_runs": _page_of("job", 0, 3)}

    monkeypatch.setattr(github_pr_watch, "_github_get", fake_get)
    with caplog.at_level(logging.WARNING):
        merged = await _fetch_all_pages(
            "https://api.github.com/repos/o/r/commits/deadbeef/check-runs",
            key="check_runs", token="t", ref="o/r#1 check-runs",
        )
    assert merged["_truncated"] is False
    assert merged["total_count"] == 400
    assert "total_count=400" in caplog.text
    assert _check_verdict(merged) == "success"


@pytest.mark.asyncio
async def test_workflow_run_listing_keeps_partial_pages(monkeypatch):
    """Enrichment: a mid-pagination failure keeps the suites already
    identified (each one is an ambiguous group avoided) and marks the rest
    truncated, instead of throwing the whole listing away."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.signals.sources import github_pr_watch

    runs = [{"id": i, "workflow_id": 7, "check_suite_id": i,
             "run_number": 1, "created_at": "2026-08-11T14:00:00Z"}
            for i in range(100)]

    async def fake_get(url, *, token, timeout, ref):
        if url.split("page=")[-1] == "1":
            return {"total_count": 150, "workflow_runs": runs}
        raise github_pr_watch.PRWatchNetworkError("rate limited")

    monkeypatch.setattr(github_pr_watch, "_github_get", fake_get)
    provider = CIWaitable(feature=None)
    listing = await provider._fetch_workflow_runs(
        "https://api.github.com/repos/owner/repo", "deadbeef", "t", "owner/repo#7"
    )
    assert len(listing["workflow_runs"]) == 100
    assert listing["_truncated"] is True


@pytest.mark.asyncio
async def test_workflow_run_listing_first_page_failure_degrades_to_none(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.signals.sources import github_pr_watch

    async def fake_get(url, *, token, timeout, ref):
        raise github_pr_watch.PRWatchAuthError("no actions:read")

    monkeypatch.setattr(github_pr_watch, "_github_get", fake_get)
    provider = CIWaitable(feature=None)
    assert await provider._fetch_workflow_runs(
        "https://api.github.com/repos/owner/repo", "deadbeef", "t", "owner/repo#7"
    ) is None


@pytest.mark.asyncio
async def test_check_run_fetch_failure_blocks_the_poll(monkeypatch):
    """Unlike the workflow listing, the checks themselves are evidence — a
    failed fetch is ``blocked``, never an empty (and therefore terminal)
    rollup."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.signals.sources import github_pr_watch

    async def fake_get(url, *, token, timeout, ref):
        if url.endswith("/pulls/7"):
            return {"state": "open", "merged": False,
                    "updated_at": "2026-08-11T14:00:00Z",
                    "head": {"sha": "deadbeef"}}
        raise github_pr_watch.PRWatchAuthError("403")

    monkeypatch.setattr(github_pr_watch, "_github_get", fake_get)
    provider = CIWaitable(feature=None)
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.PENDING
    assert status.data["blocked"] == "auth"


@pytest.mark.asyncio
async def test_fetch_degrades_when_workflow_run_listing_fails(monkeypatch):
    """The workflow-run listing is enrichment: losing it must not block the
    poll, only degrade superseded-run grouping."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    from kestrel_sovereign.signals.sources import github_pr_watch

    calls = []

    async def fake_get(url, *, token, timeout, ref):
        calls.append(url)
        if "/actions/runs" in url:
            raise github_pr_watch.PRWatchNetworkError("rate limited")
        if url.endswith("/pulls/7"):
            return {
                "state": "open", "merged": False,
                "updated_at": "2026-08-11T14:00:00Z",
                "head": {"sha": "deadbeef"},
            }
        if "check-runs" in url:
            return PR_2934_CHECK_RUNS
        return PR_2934_COMBINED_STATUS

    monkeypatch.setattr(github_pr_watch, "_github_get", fake_get)
    provider = CIWaitable(feature=None)
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.DONE
    assert status.data["checks"] == "success"
    assert any("/actions/runs" in u for u in calls)
