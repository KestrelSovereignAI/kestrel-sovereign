"""A credential that cannot read the Checks API must still produce a verdict.

Measured against the live fine-grained PAT on 2026-09-08
(``KestrelSovereignAI/kestrel-feature-talon@399763c``)::

    GET /repos                                  200
    GET /commits/<sha>/status                   200   statuses=read
    GET /actions/runs?head_sha=<sha>            200   actions=read
    GET /commits/<sha>/check-runs               403   checks=read
    GET /pulls                                  200

The 403 is permanent, not a misconfiguration: ``checks`` is a GitHub *App*
permission with no entry in the fine-grained repository permission list at
all, so the response header names a permission that token type can never
hold. The same SHA's six check runs are all ``app=github-actions`` — the jobs
of one workflow run — and ``/actions/runs`` returns that run to the very same
token.

So the gate degrades instead of going blind, and these tests pin both halves
of that: the fallback recovers the verdict, and the recovered verdict never
claims more than it saw. Every test here drives the real fetch path with only
the HTTP layer faked. Stubbing ``_fetch`` — which is what the #3248 tests did
— would let the entire fallback be deleted with the suite still green.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import kestrel_sovereign.signals.sources.github_pr_watch as prw
from kestrel_sovereign.signals.sources.github_pr_watch import (
    CHECKS_SOURCE_CHECK_RUNS,
    CHECKS_SOURCE_WORKFLOW_RUNS,
    PRWatchAuthError,
    PRWatchNetworkError,
    fetch_check_rollup,
    summarize_checks,
)
from kestrel_sovereign.features.scheduler import ci_wait_provider as mod
from kestrel_sovereign.features.scheduler.ci_wait_provider import CIWaitable, Outcome

BASE = "https://api.github.com/repos/o/r"
SHA = "399763c306a9257dc3f46944b9260a9ba5d86080"
HANDLE = "o/r#20"

# The rollup as the live token sees it: Checks refused, one Actions run.
GREEN_RUN = {"name": "kestrel-feature-talon CI", "status": "completed",
             "conclusion": "success"}
EMPTY_STATUS = {"state": "pending", "total_count": 0, "statuses": []}


def _router(routes, seen=None):
    """A fake ``_github_get`` that dispatches on a substring of the URL.

    A route value that is an exception is raised; anything else is returned.
    ``seen`` collects every requested URL so a test can assert on a request
    that was *not* made — the only way to show the fallback stayed unused.
    """

    async def fake_get(url, *, token, timeout, ref):
        if seen is not None:
            seen.append(url)
        for fragment, value in routes.items():
            if fragment in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unrouted URL in test: {url}")

    return fake_get


def _live_shape(*, runs=(GREEN_RUN,), status=EMPTY_STATUS, pr=None):
    """The routes measured on 2026-09-08: Checks 403, Actions 200, status 200."""
    return {
        "/pulls/": pr if pr is not None else {"state": "open", "merged": False,
                                              "head": {"sha": SHA}},
        "/check-runs": PRWatchAuthError("GitHub returned 403 for check-runs"),
        "/actions/runs": {"total_count": len(runs), "workflow_runs": list(runs)},
        "/status": status,
    }


# ---------------------------------------------------------------------------
# fetch_check_rollup: what was read, and what it admits it did not read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_checks_api_falls_back_to_the_actions_api(monkeypatch):
    monkeypatch.setattr(prw, "_github_get", _router(_live_shape()))

    rollup = await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert rollup.source == CHECKS_SOURCE_WORKFLOW_RUNS
    assert rollup.unreadable == ("check-runs",)
    assert rollup.complete is False
    # Projected onto the check-runs shape so every downstream reducer is reused.
    assert prw._check_verdict(rollup.check_runs, rollup.combined_status) == "success"


@pytest.mark.asyncio
async def test_the_caveat_names_the_class_of_gate_that_stays_invisible(monkeypatch):
    monkeypatch.setattr(prw, "_github_get", _router(_live_shape()))

    rollup = await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    caveat = rollup.caveat()
    assert "other than GitHub Actions" in caveat
    # An operator has to be able to tell whether it matters for their repo,
    # which takes naming the hole, not the failing endpoint.
    assert "cannot see" in caveat


@pytest.mark.asyncio
async def test_a_complete_rollup_has_no_caveat_at_all(monkeypatch):
    """Control. If the caveat rode on every rollup it would say nothing."""
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": {"total_count": 1, "check_runs": [GREEN_RUN]},
        "/status": EMPTY_STATUS,
    }))

    rollup = await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert rollup.source == CHECKS_SOURCE_CHECK_RUNS
    assert rollup.unreadable == ()
    assert rollup.complete is True
    assert rollup.caveat() == ""


@pytest.mark.asyncio
async def test_a_network_failure_on_checks_does_not_take_the_fallback(monkeypatch):
    """The auth/network distinction is the difference between "a human must
    act" and "wait and it clears". A fallback that swallowed transport
    failures would erase it and settle a verdict off half a rollup during a
    GitHub blip."""
    seen: list = []
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": PRWatchNetworkError("connection reset"),
        "/actions/runs": {"total_count": 1, "workflow_runs": [GREEN_RUN]},
        "/status": EMPTY_STATUS,
    }, seen=seen))

    with pytest.raises(PRWatchNetworkError):
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert not [u for u in seen if "/actions/runs" in u]


@pytest.mark.asyncio
async def test_every_endpoint_refused_is_still_a_hard_auth_failure(monkeypatch):
    """Nothing readable is not a caveated verdict — it is a credential to fix."""
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": PRWatchAuthError("403"),
        "/actions/runs": PRWatchAuthError("403"),
        "/status": PRWatchAuthError("403"),
    }))

    with pytest.raises(PRWatchAuthError):
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")


@pytest.mark.asyncio
async def test_a_refused_status_alone_keeps_the_check_runs_it_did_read(monkeypatch):
    """One refused endpoint must not discard the evidence from the others."""
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": {"total_count": 1, "check_runs": [GREEN_RUN]},
        "/status": PRWatchAuthError("403"),
    }))

    rollup = await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert rollup.source == CHECKS_SOURCE_CHECK_RUNS
    assert rollup.unreadable == ("status",)
    assert rollup.complete is False
    assert rollup.check_runs["check_runs"] == [GREEN_RUN]
    assert "legacy commit statuses" in rollup.caveat()


@pytest.mark.asyncio
async def test_the_actions_fallback_follows_pages(monkeypatch):
    """The fallback shares the check-runs paginator, so a rollup on page two
    is read rather than merely detected — the same #2939 bound."""
    page1 = {"total_count": 150,
             "workflow_runs": [dict(GREEN_RUN, name=f"w{i}") for i in range(100)]}
    page2 = {"total_count": 150,
             "workflow_runs": [dict(GREEN_RUN, name=f"w{i}") for i in range(100, 150)]}

    async def fake_get(url, *, token, timeout, ref):
        if "/pulls/" in url:
            return {"state": "open", "merged": False, "head": {"sha": SHA}}
        if "/check-runs" in url:
            raise PRWatchAuthError("403")
        if "/actions/runs" in url:
            return page2 if "page=2" in url else page1
        return EMPTY_STATUS

    monkeypatch.setattr(prw, "_github_get", fake_get)

    rollup = await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert len(rollup.check_runs["check_runs"]) == 150
    assert prw._check_verdict(rollup.check_runs, rollup.combined_status) == "success"


@pytest.mark.asyncio
async def test_a_short_read_of_the_fallback_lowers_success_to_pending(monkeypatch):
    """``total_count`` is carried across the projection, so the unread-gate
    protection keeps applying to Actions runs."""
    async def fake_get(url, *, token, timeout, ref):
        if "/check-runs" in url:
            raise PRWatchAuthError("403")
        if "/actions/runs" in url:
            # Claims 9, sends 1, and page two comes back empty.
            return ({"total_count": 9, "workflow_runs": [GREEN_RUN]}
                    if "page=2" not in url else
                    {"total_count": 9, "workflow_runs": []})
        return EMPTY_STATUS

    monkeypatch.setattr(prw, "_github_get", fake_get)

    rollup = await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert prw._check_verdict(rollup.check_runs, rollup.combined_status) == "pending"


# ---------------------------------------------------------------------------
# poll(): the verdict a waiter actually receives, through the real _fetch
# ---------------------------------------------------------------------------


@pytest.fixture()
def _token(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.features.strategic_memory.github_integration"
        ".get_github_token",
        lambda: "github_pat_fake",
    )


async def _poll(monkeypatch, routes):
    monkeypatch.setattr(prw, "_github_get", _router(routes))
    return await CIWaitable(MagicMock()).poll(HANDLE)


@pytest.mark.asyncio
async def test_a_checks_blind_token_now_gets_a_verdict_not_a_permission_block(
    monkeypatch, _token
):
    """The whole point. On 2026-09-07 this exact shape returned PENDING with
    ``blocked="permission"`` and an agent could not merge behind it."""
    st = await _poll(monkeypatch, _live_shape())

    assert st.data.get("blocked") is None
    assert st.data["checks"] == "success"
    assert st.data["checks_source"] == CHECKS_SOURCE_WORKFLOW_RUNS


@pytest.mark.asyncio
async def test_a_pass_read_through_the_fallback_is_partial_not_done(
    monkeypatch, _token
):
    """"Everything I could see passed" is a weaker claim than "everything
    passed", and DONE cannot carry the difference — PARTIAL rides the caveat
    out to the waiter."""
    st = await _poll(monkeypatch, _live_shape())

    assert st.outcome is Outcome.PARTIAL
    assert "other than GitHub Actions" in st.data["caveat"]


@pytest.mark.asyncio
async def test_a_complete_rollup_still_reaches_done(monkeypatch, _token):
    """Control for the downgrade: it must be conditional on the blind spot,
    not a blanket demotion of every pass."""
    st = await _poll(monkeypatch, {
        "/pulls/": {"state": "open", "merged": False, "head": {"sha": SHA}},
        "/check-runs": {"total_count": 1, "check_runs": [GREEN_RUN]},
        "/status": EMPTY_STATUS,
    })

    assert st.outcome is Outcome.DONE
    assert "caveat" not in st.data
    assert "checks_source" not in st.data


@pytest.mark.asyncio
async def test_a_visible_failure_through_the_fallback_is_still_terminal_failed(
    monkeypatch, _token
):
    """An unread gate can only hide MORE failures, never turn an observed one
    into a pass, so a partial rollup does not soften a failure."""
    red = dict(GREEN_RUN, conclusion="failure")
    st = await _poll(monkeypatch, _live_shape(runs=(red,)))

    assert st.outcome is Outcome.FAILED
    # The provenance still rides along, so an audit of why it failed can see
    # the rollup behind it was partial.
    assert st.data["checks_source"] == CHECKS_SOURCE_WORKFLOW_RUNS


@pytest.mark.asyncio
async def test_a_running_actions_run_through_the_fallback_stays_pending(
    monkeypatch, _token
):
    running = {"name": "CI", "status": "in_progress", "conclusion": None}
    st = await _poll(monkeypatch, _live_shape(runs=(running,)))

    assert st.outcome is Outcome.PENDING
    assert st.data["checks"] == "pending"


@pytest.mark.asyncio
async def test_an_empty_fallback_rollup_never_claims_that_nothing_ran(
    monkeypatch, _token
):
    """The dangerous case. A checkless head SHA and an invisible third-party
    check app read identically here, and only one of them means "no CI".
    Terminal either way (#2939) — but the caveat must not make the claim."""
    st = await _poll(monkeypatch, _live_shape(runs=()))

    assert st.outcome is Outcome.PARTIAL
    assert "NOT evidence that no checks ran" in st.data["caveat"]
    assert "no checks ran on" not in st.data.get("caveat", "")


@pytest.mark.asyncio
async def test_a_complete_empty_rollup_does_still_claim_nothing_ran(
    monkeypatch, _token
):
    """Control for the pair above: when the whole rollup WAS readable and is
    empty, the strong claim is the correct one and must survive."""
    st = await _poll(monkeypatch, {
        "/pulls/": {"state": "open", "merged": False, "head": {"sha": SHA}},
        "/check-runs": {"total_count": 0, "check_runs": []},
        "/status": EMPTY_STATUS,
    })

    assert st.outcome is Outcome.PARTIAL
    assert "no checks ran on" in st.data["caveat"]
    assert "NOT evidence" not in st.data["caveat"]


@pytest.mark.asyncio
async def test_total_blindness_is_still_the_actionable_permission_block(
    monkeypatch, _token
):
    """#3248's behaviour survives, narrowed: it now takes every endpoint."""
    st = await _poll(monkeypatch, {
        "/pulls/": {"state": "open", "merged": False, "head": {"sha": SHA}},
        "/check-runs": PRWatchAuthError("403"),
        "/actions/runs": PRWatchAuthError("403"),
        "/status": PRWatchAuthError("403"),
    })

    assert st.outcome is Outcome.PENDING
    assert st.data["blocked"] == "permission"
    assert st.data["actionable"] is True
    assert "BLIND" in st.summary


@pytest.mark.asyncio
async def test_the_permission_message_states_the_remedy_that_actually_exists(
    monkeypatch, _token
):
    """The #3248 message told an operator to grant "Checks" to a fine-grained
    PAT, which cannot be done — the permission does not exist for that token
    type. An unfollowable instruction is a different way of being stuck."""
    st = await _poll(monkeypatch, {
        "/pulls/": {"state": "open", "merged": False, "head": {"sha": SHA}},
        "/check-runs": PRWatchAuthError("403"),
        "/actions/runs": PRWatchAuthError("403"),
        "/status": PRWatchAuthError("403"),
    })

    assert "Actions" in st.summary and "Commit statuses" in st.summary
    assert "GitHub App permission" in st.summary
    assert "will not resolve on its own" in st.summary


# ---------------------------------------------------------------------------
# The fingerprint half: the pr-watch signal source reads the same rollup
# ---------------------------------------------------------------------------


def test_a_degraded_rollup_does_not_fingerprint_as_a_complete_one():
    """Otherwise the watch reports "no change" across the very poll where its
    own visibility changed — a token being widened would wake nobody."""
    same_runs = {"total_count": 1, "check_runs": [GREEN_RUN]}

    complete = summarize_checks(same_runs, EMPTY_STATUS)
    degraded = summarize_checks(
        same_runs, EMPTY_STATUS,
        source=CHECKS_SOURCE_WORKFLOW_RUNS, unreadable=("check-runs",),
    )

    assert complete != degraded
    assert "source=workflow_runs" in degraded
    assert "unreadable=check-runs" in degraded


def test_an_empty_degraded_rollup_is_not_the_empty_summary():
    """``""`` is this module's "no checks exist" — a claim a blind poll cannot
    make, and the one that would let a partial read masquerade as a clean
    checkless commit."""
    assert summarize_checks(None, None) == ""
    assert summarize_checks(
        {"total_count": 0, "check_runs": []}, None,
        source=CHECKS_SOURCE_WORKFLOW_RUNS, unreadable=("check-runs",),
    ) != ""


@pytest.mark.asyncio
async def test_fetch_pr_state_degrades_rather_than_blocking_the_whole_poll(
    monkeypatch,
):
    """The signal source shares the fallback, so a checks-blind token no
    longer blocks every ``github_pr_watch`` poll either."""
    monkeypatch.setattr(prw, "_github_get", _router(_live_shape()))

    raw = await prw.fetch_pr_state("o/r", 20, token="t", timeout=1)

    assert "source=workflow_runs" in raw["checks_status"]
    assert "combined=success" in raw["checks_status"]
