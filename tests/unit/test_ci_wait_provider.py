"""Tests for the ``ci:`` Waitable provider (#2729).

A ``ci:<owner/repo#N>`` wait watches a GitHub PR's merge/CI-check state so a
merge/check wait can survive restart. The classification logic is pure
(``classify_ci_state`` / ``_check_verdict``) and tested here without a
network; ``poll`` is exercised with a stubbed fetch + token.

The ``#2939`` section covers the stale ``checks: "pending"`` stall, using the
real captured rollup of ``kestrel-sovereign#2934`` — 18 check runs, all
completed, three of them SKIPPED, with duplicate check names across two
workflow runs on the same head SHA.
"""

from __future__ import annotations

import pytest

from kestrel_sdk.tools import Outcome

from kestrel_sovereign.signals.sources.github_pr_watch import (
    CHECKS_SOURCE_CHECK_RUNS,
    CHECKS_SOURCE_WORKFLOW_RUNS,
    CheckRollup,
    _workflow_runs_as_check_runs,
)
from kestrel_sovereign.features.scheduler.ci_wait_provider import (
    CIWaitable,
    CITerminalEvent,
    _check_verdict,
    ci_terminal_event,
    classify_ci_state,
    parse_ci_handle,
)
from kestrel_sovereign.waits.engine import (
    TERMINAL_EVENT_DETAIL_KEY,
    TERMINAL_EVENT_FINAL_KEY,
    TERMINAL_EVENT_KEY,
    TERMINAL_EVENT_VIEW_KEY,
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


def test_check_verdict_unknown_when_rollup_not_read():
    """No payload at all is an evidence gap, not a claim about the checks."""
    assert _check_verdict() == "unknown"
    assert _check_verdict(None, None) == "unknown"


def test_check_verdict_none_when_rollup_read_and_empty():
    assert _check_verdict({}, {}) == "none"
    assert _check_verdict({"total_count": 0, "check_runs": []}) == "none"


def test_check_verdict_pending_on_incomplete_run():
    assert _check_verdict({"check_runs": [{"name": "ci", "status": "in_progress"}]}) == "pending"


def test_check_verdict_ignores_combined_pending_with_zero_statuses():
    """#2939: GitHub reports combined ``state: "pending"`` for a commit with
    ZERO legacy statuses — the shape of every Actions-only repo. Reading that
    as a running check pinned the verdict at pending forever."""
    assert _check_verdict(None, {"state": "pending", "total_count": 0, "statuses": []}) == "none"
    assert (
        _check_verdict(
            {"check_runs": [{"name": "ci", "status": "completed", "conclusion": "success"}]},
            {"state": "pending", "total_count": 0, "statuses": []},
        )
        == "success"
    )


def test_check_verdict_honors_combined_pending_with_real_statuses():
    """A combined ``pending`` backed by an actual legacy status is real
    evidence and must still block."""
    assert (
        _check_verdict(
            {"check_runs": [{"name": "ci", "status": "completed", "conclusion": "success"}]},
            {"state": "pending", "total_count": 1,
             "statuses": [{"context": "cov", "state": "pending"}]},
        )
        == "pending"
    )


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


def test_classify_open_unread_rollup_is_pending():
    """An unread rollup is an evidence gap — keep watching rather than
    fabricating a verdict from payloads we never fetched."""
    st = classify_ci_state({"state": "open", "merged": False}, repo="o/r", number=2)
    assert st.outcome is Outcome.PENDING
    assert st.data["checks"] == "unknown"


def test_classify_open_unread_runs_on_an_empty_page_is_pending_not_partial():
    """``total_count`` above an empty page means gates the read never saw.
    That is an evidence gap, not "no CI ran": PARTIAL is terminal and a
    mode="signal" wait on it would fire once, reporting no CI for a head the
    same payload says has 45 gates."""
    st = classify_ci_state(
        {"state": "open", "merged": False},
        check_runs={"total_count": 45, "check_runs": []},
        combined_status={"state": "pending", "total_count": 0, "statuses": []},
        repo="o/r", number=7,
    )
    assert st.outcome is Outcome.PENDING
    assert st.data["checks"] == "pending"


# ---------------------------------------------------------------------------
# #2939 regressions — the stale ``checks: "pending"`` stall
# ---------------------------------------------------------------------------


def _pr_2934_check_runs():
    """The real check-run rollup of ``kestrel-sovereign#2934`` head 362b6c0d.

    Captured from ``GET /commits/{sha}/check-runs``: 18 runs, all COMPLETED
    (15 success, 3 skipped), spanning two Kestrel CI runs on the same head —
    so ``unit-tests``/``dependency-review``/``cancel-expensive-tiers-when-unit-fails``
    each appear twice, and ``dependency-review`` has a *different* conclusion
    in each run (success, then skipped).
    """
    names_run_a = [
        ("cancel-expensive-tiers-when-unit-fails", "skipped"),
        ("unit-tests", "success"),
        ("llm-tests", "success"),
        ("integration-tests", "success"),
        ("python-314-install-check", "success"),
        ("clean-install (ubuntu-latest, sync)", "success"),
        ("clean-install (ubuntu-latest, wheel)", "success"),
        ("dependency-review", "success"),
        ("lint-and-imports", "success"),
        ("select-matrix", "success"),
        ("okf", "success"),
    ]
    names_run_b = [
        ("cancel-expensive-tiers-when-unit-fails", "skipped"),
        ("python-314-install-check", "success"),
        ("unit-tests", "success"),
        ("llm-tests", "success"),
        ("integration-tests", "success"),
        ("dependency-review", "skipped"),
        ("lint-and-imports", "success"),
    ]
    runs = [
        {"name": n, "status": "completed", "conclusion": c,
         "check_suite": {"id": 31500405381}}
        for n, c in names_run_a
    ] + [
        {"name": n, "status": "completed", "conclusion": c,
         "check_suite": {"id": 31501778694}}
        for n, c in names_run_b
    ]
    return {"total_count": len(runs), "check_runs": runs}


def _combined_status_actions_only():
    """The real combined status for that head SHA.

    Captured from ``GET /commits/{sha}/status``: an Actions-only repo has zero
    legacy commit statuses, and GitHub reports the *combined* state of zero
    statuses as ``"pending"``.
    """
    return {"state": "pending", "total_count": 0, "statuses": []}


def test_pr_2934_fully_completed_rollup_is_terminal():
    """The reported defect: every check run on the head SHA had completed
    (incl. SKIPPED, and duplicate names across two workflow runs) while the
    provider reported ``checks: "pending"`` for ~3h."""
    assert (
        _check_verdict(_pr_2934_check_runs(), _combined_status_actions_only())
        == "success"
    )
    st = classify_ci_state(
        {"state": "open", "merged": False, "mergeable": True,
         "mergeable_state": "clean"},
        check_runs=_pr_2934_check_runs(),
        combined_status=_combined_status_actions_only(),
        repo="KestrelSovereignAI/kestrel-sovereign",
        number=2934,
    )
    assert st.outcome is Outcome.DONE
    assert st.data["checks"] == "success"
    assert "contradiction" not in st.data


def test_merged_pr_reports_real_check_verdict_not_stale_pending():
    """The terminal ``wait.complete`` payload for #2934 carried
    ``checks: 'pending'`` ~22h after CI finished. Terminality came from
    ``merged``, and the check verdict was simply wrong."""
    st = classify_ci_state(
        {"state": "closed", "merged": True},
        check_runs=_pr_2934_check_runs(),
        combined_status=_combined_status_actions_only(),
        repo="KestrelSovereignAI/kestrel-sovereign",
        number=2934,
    )
    assert st.outcome is Outcome.DONE
    assert st.data["checks"] == "success"


@pytest.mark.parametrize("conclusion", ["skipped", "neutral"])
def test_skipped_and_neutral_never_hold_the_rollup_open(conclusion):
    """COMPLETED/<conclusion> is terminal and must not count toward pending."""
    runs = {"check_runs": [
        {"name": "gate", "status": "completed", "conclusion": conclusion},
    ]}
    assert _check_verdict(runs, _combined_status_actions_only()) == "success"


def test_cancelled_is_still_a_failure():
    """A cancelled check was stopped before it could tell us anything —
    absence of evidence, not a pass. It stays terminal FAILED."""
    runs = {"check_runs": [
        {"name": "unit", "status": "completed", "conclusion": "success"},
        {"name": "e2e", "status": "completed", "conclusion": "cancelled"},
    ]}
    assert _check_verdict(runs, _combined_status_actions_only()) == "failure"
    st = classify_ci_state(
        {"state": "open", "merged": False}, check_runs=runs, repo="o/r", number=3
    )
    assert st.outcome is Outcome.FAILED


def test_open_pr_with_empty_rollup_is_terminal_none():
    """Zero check runs + zero statuses: nothing ever ran, so there is nothing
    to wait for. Terminal PARTIAL — never DONE, because "nothing ran" must not
    read as "everything passed"."""
    st = classify_ci_state(
        {"state": "open", "merged": False, "updated_at": "2024-01-01T00:00:00Z"},
        check_runs={"total_count": 0, "check_runs": []},
        combined_status=_combined_status_actions_only(),
        repo="o/r", number=4,
    )
    assert st.outcome is Outcome.PARTIAL
    assert st.outcome.is_terminal()
    assert st.data["checks"] == "none"
    assert st.data["caveat"]


def test_empty_rollup_is_terminal_on_a_just_touched_pr():
    """No grace period for check runs GitHub might still be creating: a
    freshly-updated PR with an empty rollup is terminal on the first read,
    exactly like a stale one."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    st = classify_ci_state(
        {"state": "open", "merged": False, "updated_at": now,
         "created_at": now},
        check_runs={"total_count": 0, "check_runs": []},
        combined_status=_combined_status_actions_only(),
        repo="o/r", number=4,
    )
    assert st.outcome is Outcome.PARTIAL
    assert st.outcome.is_terminal()
    assert st.data["checks"] == "none"


def test_empty_rollup_terminal_is_not_deferred_by_mutable_pr_activity():
    """Regression: repeated reconciliations of the SAME checkless head SHA must
    each be terminal, even as unrelated PR edits keep refreshing ``updated_at``.

    A settle window anchored on that mutable timestamp would re-arm on every
    poll and never converge — the #2939 indefinite stall in a new shape. The
    verdict must depend only on the observed rollup.
    """
    from datetime import datetime, timedelta, timezone

    head = {"sha": "362b6c0d85573b25ee4fba32278d6ad2f1cbf80c"}
    outcomes = []
    for minute in range(10):
        # Same head SHA throughout; only the mutable activity stamp moves, as
        # a comment/label/automation touch would move it in production.
        touched = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
            + timedelta(minutes=minute)
        ).isoformat().replace("+00:00", "Z")
        st = classify_ci_state(
            {"state": "open", "merged": False, "head": head,
             "created_at": touched, "updated_at": touched},
            check_runs={"total_count": 0, "check_runs": []},
            combined_status=_combined_status_actions_only(),
            repo="o/r", number=4,
        )
        outcomes.append(st.outcome)

    assert outcomes == [Outcome.PARTIAL] * 10
    assert all(o.is_terminal() for o in outcomes)


def test_empty_rollup_with_unreadable_timestamp_is_terminal():
    """The verdict never depends on parsing a PR timestamp at all."""
    st = classify_ci_state(
        {"state": "open", "merged": False, "updated_at": "not-a-date"},
        check_runs={"total_count": 0, "check_runs": []},
        repo="o/r", number=4,
    )
    assert st.outcome is Outcome.PARTIAL
    assert st.data["checks"] == "none"


def test_clean_but_pending_surfaces_contradiction_without_fabricating_terminal():
    """A genuinely-pending rollup on a PR GitHub calls clean stays PENDING —
    the contradiction is surfaced for diagnosis, never resolved to DONE."""
    st = classify_ci_state(
        {"state": "open", "merged": False, "mergeable": True,
         "mergeable_state": "clean"},
        check_runs={"check_runs": [{"name": "ci", "status": "in_progress"}]},
        combined_status=_combined_status_actions_only(),
        repo="o/r", number=5,
    )
    assert st.outcome is Outcome.PENDING
    assert st.data["checks"] == "pending"
    assert st.data["contradiction"] == "clean_but_pending"
    assert st.data["mergeable_state"] == "clean"
    assert st.data["mergeable"] is True


def test_pending_on_blocked_pr_is_not_a_contradiction():
    st = classify_ci_state(
        {"state": "open", "merged": False, "mergeable": True,
         "mergeable_state": "blocked"},
        check_runs={"check_runs": [{"name": "ci", "status": "queued"}]},
        combined_status=_combined_status_actions_only(),
        repo="o/r", number=6,
    )
    assert st.outcome is Outcome.PENDING
    assert "contradiction" not in st.data
    assert st.data["mergeable_state"] == "blocked"


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
        return {"state": "closed", "merged": True}, CheckRollup()

    provider._fetch = fake_fetch
    status = await provider.poll("owner/repo#7")
    assert status.outcome is Outcome.DONE
    assert status.data["repo"] == "owner/repo"
    assert status.data["number"] == 7


@pytest.mark.asyncio
async def test_poll_through_real_fetch_resolves_completed_rollup(monkeypatch):
    """End-to-end through the real ``_fetch`` URL routing (not a stubbed
    ``_fetch``): the #2934 payloads must resolve to a terminal verdict."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    import kestrel_sovereign.signals.sources.github_pr_watch as prw

    seen = []

    async def fake_get(url, *, token, timeout, ref):
        seen.append(url)
        if url.endswith("/pulls/2934"):
            return {
                "state": "open", "merged": False, "mergeable": True,
                "mergeable_state": "clean",
                "head": {"sha": "362b6c0d85573b25ee4fba32278d6ad2f1cbf80c"},
            }
        if "/check-runs" in url:
            return _pr_2934_check_runs()
        if url.endswith("/status"):
            return _combined_status_actions_only()
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(prw, "_github_get", fake_get)
    provider = CIWaitable(feature=None)
    status = await provider.poll("KestrelSovereignAI/kestrel-sovereign#2934")

    assert status.outcome is Outcome.DONE
    assert status.data["checks"] == "success"
    assert any("/check-runs?per_page=100" in u for u in seen)


@pytest.mark.asyncio
async def test_poll_without_head_sha_stays_pending_unknown(monkeypatch):
    """No head SHA means the rollup was never read — an evidence gap that must
    stay PENDING rather than resolving to the terminal ``none``."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    import kestrel_sovereign.signals.sources.github_pr_watch as prw

    async def fake_get(url, *, token, timeout, ref):
        return {"state": "open", "merged": False}

    monkeypatch.setattr(prw, "_github_get", fake_get)
    provider = CIWaitable(feature=None)
    status = await provider.poll("owner/repo#7")

    assert status.outcome is Outcome.PENDING
    assert status.data["checks"] == "unknown"


# ---------------------------------------------------------------------------
# #3399: every terminal status names its terminal event
# ---------------------------------------------------------------------------

_SHA = "362b6c0d85573b25ee4fba32278d6ad2f1cbf80c"
_OPEN = {"state": "open", "merged": False, "head": {"sha": _SHA}}


def _runs(*runs):
    return {"total_count": len(runs), "check_runs": list(runs)}


def _failed_run(run_id, name="unit-tests"):
    return {"id": run_id, "name": name, "status": "completed", "conclusion": "failure"}


def _event_of(pr, check_runs=None, combined_status=None, **kwargs):
    """``(outcome, identity)``: the identity is the event, its detail and the
    detail's view as the poll reports them, or ``None`` when it names none."""
    status = classify_ci_state(
        pr, check_runs=check_runs, combined_status=combined_status,
        repo="o/r", number=1, **kwargs,
    )
    if TERMINAL_EVENT_KEY not in status.data:
        assert TERMINAL_EVENT_DETAIL_KEY not in status.data
        return status.outcome, None
    return status.outcome, CITerminalEvent(
        status.data[TERMINAL_EVENT_KEY],
        status.data.get(TERMINAL_EVENT_DETAIL_KEY),
        status.data.get(TERMINAL_EVENT_VIEW_KEY),
        status.data.get(TERMINAL_EVENT_FINAL_KEY) is True,
    )


def test_the_same_records_always_name_the_same_event():
    """Stable across polls and independent of the order GitHub pages in."""
    a, b = _failed_run(1, "lint"), _failed_run(2, "unit-tests")
    assert _event_of(_OPEN, _runs(a, b)) == _event_of(_OPEN, _runs(b, a))
    outcome, identity = _event_of(_OPEN, _runs(a, b))
    assert outcome is Outcome.FAILED
    assert identity.event == f"head@{_SHA}"
    assert identity.view == CHECKS_SOURCE_CHECK_RUNS


def test_a_rerun_that_fails_again_is_a_new_event_with_the_same_outcome():
    first = _event_of(_OPEN, _runs(_failed_run(101)))
    rerun = _event_of(_OPEN, _runs(_failed_run(102)))
    assert first[0] is rerun[0] is Outcome.FAILED
    assert first[1] != rerun[1]
    assert first[1].view == rerun[1].view, "one view, so the details compare"


def test_a_workflow_run_attempt_is_part_of_the_event():
    def attempt(n):
        return _event_of(
            _OPEN,
            _runs({**_failed_run(9001), "run_attempt": n}),
            checks_source=CHECKS_SOURCE_WORKFLOW_RUNS,
            unreadable=("check-runs",),
        )[1]

    assert attempt(1) != attempt(2)
    assert attempt(1).view == CHECKS_SOURCE_WORKFLOW_RUNS


def test_a_new_head_commit_is_a_new_event():
    moved = {**_OPEN, "head": {"sha": "f" * 40}}
    assert _event_of(_OPEN, _runs(_failed_run(1)))[1].event != (
        _event_of(moved, _runs(_failed_run(1)))[1].event
    )


def test_a_settled_legacy_status_is_part_of_the_execution_set():
    """A status re-reported under a new id is a new execution of that gate,
    the same as a check run re-run: the head SHA (the event) is unchanged,
    and the execution set it was read in differs."""
    def status(status_id):
        return {
            "state": "failure", "total_count": 1,
            "statuses": [{"id": status_id, "context": "cov", "state": "failure"}],
        }

    first = _event_of(_OPEN, {}, status(1))[1]
    rerun = _event_of(_OPEN, {}, status(2))[1]
    assert first.event == rerun.event == f"head@{_SHA}"
    assert first.view == rerun.view
    assert first.detail != rerun.detail


def test_a_pending_rerun_names_no_event():
    outcome, identity = _event_of(
        _OPEN, _runs({"id": 102, "name": "unit-tests", "status": "queued"})
    )
    assert outcome is Outcome.PENDING
    assert identity is None


def test_the_event_ignores_the_classification():
    """#3390: the same records read completely (DONE) or through the Actions
    fallback (PARTIAL) name one event and one execution set. Only the verdict
    about them differs, and the view that says which path read them."""
    green = _runs({"id": 5, "name": "ci", "status": "completed", "conclusion": "success"})
    done = _event_of(_OPEN, green)
    partial = _event_of(
        _OPEN, green,
        checks_source=CHECKS_SOURCE_WORKFLOW_RUNS, unreadable=("check-runs",),
    )
    assert (done[0], partial[0]) == (Outcome.DONE, Outcome.PARTIAL)
    assert (done[1].event, done[1].detail) == (partial[1].event, partial[1].detail)
    assert (done[1].view, partial[1].view) == (
        CHECKS_SOURCE_CHECK_RUNS, CHECKS_SOURCE_WORKFLOW_RUNS,
    )


def test_one_execution_read_through_checks_and_actions_is_one_event():
    """The review finding on #3399: the Checks API names a GitHub Actions
    execution by its job-level check-run ids, the Actions fallback by its
    workflow run's id and attempt. The ids cannot agree, so the execution set
    is scoped to its view; the event both paths name alike — the head SHA —
    is the same."""
    status = {
        "state": "failure", "total_count": 1,
        "statuses": [{"id": 77, "context": "cov", "state": "failure"}],
    }
    via_checks = _event_of(
        _OPEN,
        _runs({**_failed_run(31337, "unit-tests")}, _failed_run(31338, "lint")),
        status,
    )
    via_actions = _event_of(
        _OPEN,
        _workflow_runs_as_check_runs({
            "total_count": 1,
            "workflow_runs": [{
                "id": 9001, "run_attempt": 1, "name": "CI",
                "status": "completed", "conclusion": "failure",
            }],
        }),
        status,
        checks_source=CHECKS_SOURCE_WORKFLOW_RUNS,
        unreadable=("check-runs",),
    )

    assert via_checks[0] is via_actions[0] is Outcome.FAILED
    assert via_checks[1].event == via_actions[1].event
    assert via_checks[1].detail != via_actions[1].detail
    assert via_checks[1].view != via_actions[1].view


def test_merge_and_close_name_their_own_events():
    merged = {**_OPEN, "state": "closed", "merged": True}
    assert _event_of(merged, _runs(_failed_run(1)))[1] == CITerminalEvent(
        f"merged@{_SHA}", final=True
    )
    assert _event_of(merged)[1] == CITerminalEvent(f"merged@{_SHA}", final=True), (
        "checks do not move a merge, and a merge has no view-scoped detail"
    )

    closed_once = {**_OPEN, "state": "closed", "closed_at": "2026-09-29T10:00:00Z"}
    closed_again = {**closed_once, "closed_at": "2026-09-30T10:00:00Z"}
    assert _event_of(closed_once)[1] != _event_of(closed_again)[1]


def test_only_a_merge_is_final():
    """Nothing follows a merge. A close can be reopened, and an open PR can
    be pushed to or re-run, so neither is final."""
    def final(pr, check_runs=None):
        status = classify_ci_state(pr, check_runs=check_runs, repo="o/r", number=1)
        assert status.outcome.is_terminal()
        return status.data.get(TERMINAL_EVENT_FINAL_KEY)

    assert final({**_OPEN, "state": "closed", "merged": True}) is True
    assert final({**_OPEN, "state": "closed"}) is None
    assert final(_OPEN, _runs(_failed_run(1))) is None


def test_an_empty_rollup_is_an_event_and_its_first_run_is_another():
    assert _event_of(_OPEN, _runs())[0] is Outcome.PARTIAL
    assert _event_of(_OPEN, _runs())[1] != _event_of(_OPEN, _runs(_failed_run(1)))[1]


def test_ci_terminal_event_tolerates_runs_without_ids():
    by_name = ci_terminal_event(_OPEN, check_runs=_runs(
        {"name": "ci", "status": "completed", "conclusion": "failure"}
    ))
    assert by_name == ci_terminal_event(_OPEN, check_runs=_runs(
        {"name": "ci", "status": "completed", "conclusion": "failure"}
    ))
    assert by_name != ci_terminal_event(_OPEN, check_runs=_runs(
        {"name": "ci", "status": "completed", "conclusion": "success"}
    ))


def test_the_actions_projection_carries_each_runs_id_and_attempt():
    projected = _workflow_runs_as_check_runs({
        "total_count": 1,
        "workflow_runs": [{
            "id": 9001, "run_attempt": 2, "name": "CI",
            "status": "completed", "conclusion": "failure",
        }],
    })
    [run] = projected["check_runs"]
    assert (run["id"], run["run_attempt"]) == (9001, 2)
