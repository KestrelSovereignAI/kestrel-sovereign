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
        "/check-runs": PRWatchAuthError("GitHub returned 403 for check-runs", status_code=403),
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
async def test_both_check_endpoints_refused_is_a_hard_auth_failure(monkeypatch):
    """Nothing readable is not a caveated verdict — it is a credential to fix."""
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": PRWatchAuthError("403", status_code=403),
        "/actions/runs": PRWatchAuthError("403", status_code=403),
        "/status": PRWatchAuthError("403", status_code=403),
    }))

    with pytest.raises(PRWatchAuthError):
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")


@pytest.mark.asyncio
async def test_a_refused_status_propagates_instead_of_degrading(monkeypatch):
    """Removed capability, deliberately. Tolerating a refused status read was
    an addition beyond what the Checks fallback needs, and review found two
    separate ways for it to convert a transient into a terminal. Measured
    against the credential this exists for it buys nothing — that token holds
    ``statuses=read`` and answers 200 — so the legacy-status half is
    all-or-nothing again, exactly as it was before the fallback."""
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": {"total_count": 1, "check_runs": [GREEN_RUN]},
        "/status": PRWatchAuthError("403", status_code=403),
    }))

    with pytest.raises(PRWatchAuthError):
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")


@pytest.mark.asyncio
async def test_a_401_on_the_checks_read_does_not_degrade(monkeypatch):
    """Review round 4. A token revoked or expired MID-POLL answers 401 after
    the PR read already succeeded. That is the credential failing, not this
    endpoint refusing: nothing else is readable either, so answering from the
    half already in hand would settle a verdict for a token that can no
    longer see the repository."""
    seen: list = []
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": PRWatchAuthError("GitHub returned 401", status_code=401),
        "/actions/runs": {"total_count": 1, "workflow_runs": [GREEN_RUN]},
        "/status": EMPTY_STATUS,
    }, seen=seen))

    with pytest.raises(PRWatchAuthError) as caught:
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert caught.value.status_code == 401
    assert not [u for u in seen if "/actions/runs" in u]


@pytest.mark.asyncio
async def test_a_401_on_the_actions_fallback_does_not_degrade_either(monkeypatch):
    """The same rule on the second endpoint: a credential that dies between
    the two reads must not leave a caveated rollup behind."""
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": PRWatchAuthError("403", status_code=403),
        "/actions/runs": PRWatchAuthError("GitHub returned 401", status_code=401),
        "/status": EMPTY_STATUS,
    }))

    with pytest.raises(PRWatchAuthError) as caught:
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_an_auth_error_of_unknown_status_does_not_unlock_the_degrade(
    monkeypatch,
):
    """``status_code`` defaults to None — "not known to be a 403" — so a
    caller that constructs an auth error without saying cannot accidentally
    reach the fallback."""
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": PRWatchAuthError("no status recorded"),
        "/actions/runs": {"total_count": 1, "workflow_runs": [GREEN_RUN]},
        "/status": EMPTY_STATUS,
    }))

    with pytest.raises(PRWatchAuthError):
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")


@pytest.mark.asyncio
async def test_the_status_code_is_recorded_by_the_real_classifier(monkeypatch):
    """The wiring behind the two tests above: the code must come off the real
    HTTP response, not only from hand-built exceptions."""
    _raise_from_urlopen(monkeypatch, _http_error(401))
    with pytest.raises(PRWatchAuthError) as caught:
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")
    assert caught.value.status_code == 401

    _raise_from_urlopen(monkeypatch, _http_error(403, body=PERMISSION_BODY))
    with pytest.raises(PRWatchAuthError) as caught:
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")
    assert caught.value.status_code == 403


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
            raise PRWatchAuthError("403", status_code=403)
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
            raise PRWatchAuthError("403", status_code=403)
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
        "/check-runs": PRWatchAuthError("403", status_code=403),
        "/actions/runs": PRWatchAuthError("403", status_code=403),
        "/status": PRWatchAuthError("403", status_code=403),
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
        "/check-runs": PRWatchAuthError("403", status_code=403),
        "/actions/runs": PRWatchAuthError("403", status_code=403),
        "/status": PRWatchAuthError("403", status_code=403),
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


# ---------------------------------------------------------------------------
# GitHub reports an exhausted rate limit with the same 403 as a refusal
# ---------------------------------------------------------------------------
# Found by review, 2026-09-08. The fallback degrades a rollup on
# PRWatchAuthError, and _github_get mapped EVERY 401/403 to that — so a rate
# limit on the status request would mark statuses "unreadable", and a visible
# success would then settle terminally PARTIAL off a rollup that was merely
# throttled. Terminal-from-transient is the one conversion this module exists
# to refuse. The response headers separate the two, measured the same day
# against the live token:
#
#     permission gap   403  x-accepted-github-permissions: checks=read
#                           x-ratelimit-remaining: 4999
#     primary limit    403  x-ratelimit-remaining: 0
#     secondary limit  403  retry-after: <seconds>

import email.message
import io
import urllib.error
import urllib.request

from kestrel_sovereign.signals.sources.github_pr_watch import PRWatchRateLimitError


# The two bodies GitHub actually sends, measured 2026-09-08.
PERMISSION_BODY = b'{"message":"Resource not accessible by personal access token"}'
SECONDARY_LIMIT_BODY = (
    b'{"message":"You have exceeded a secondary rate limit. '
    b'Please wait a few minutes before you try again."}'
)


def _http_error(code, body=b"", **headers):
    hdrs = email.message.Message()
    for k, v in headers.items():
        hdrs[k.replace("_", "-")] = v
    return urllib.error.HTTPError(
        "https://api.github.com/x", code, "err", hdrs, io.BytesIO(body)
    )


def _raise_from_urlopen(monkeypatch, exc):
    """Patch the real transport, so the classification under test is the
    production one rather than a fake that has already decided the answer."""
    def fake_urlopen(req, timeout=None):
        raise exc
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


@pytest.mark.asyncio
async def test_a_primary_rate_limit_403_is_transient_not_a_permission_gap(monkeypatch):
    _raise_from_urlopen(monkeypatch, _http_error(403, x_ratelimit_remaining="0"))

    with pytest.raises(PRWatchRateLimitError) as caught:
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")

    # Transient BY INHERITANCE: every existing `except PRWatchNetworkError`
    # keeps waiting through it, and the degrade path never sees it.
    assert isinstance(caught.value, PRWatchNetworkError)
    assert not isinstance(caught.value, PRWatchAuthError)


@pytest.mark.asyncio
async def test_a_secondary_rate_limit_retry_after_is_transient(monkeypatch):
    _raise_from_urlopen(monkeypatch, _http_error(403, retry_after="60"))

    with pytest.raises(PRWatchRateLimitError):
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")


@pytest.mark.asyncio
async def test_a_429_is_transient(monkeypatch):
    _raise_from_urlopen(monkeypatch, _http_error(429))

    with pytest.raises(PRWatchRateLimitError):
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")


@pytest.mark.asyncio
async def test_the_measured_permission_403_is_an_auth_error(monkeypatch):
    """Control, with the exact headers AND body the live token returns. If
    this became a rate-limit error the fallback would never fire at all."""
    _raise_from_urlopen(monkeypatch, _http_error(
        403,
        body=PERMISSION_BODY,
        x_accepted_github_permissions="checks=read",
        x_ratelimit_limit="5000",
        x_ratelimit_remaining="4999",
    ))

    with pytest.raises(PRWatchAuthError) as caught:
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")

    assert not isinstance(caught.value, PRWatchRateLimitError)


@pytest.mark.asyncio
async def test_the_accepted_permissions_header_proves_nothing_by_itself(
    monkeypatch,
):
    """Review round 3. An earlier revision read this header's presence as
    proof of a refusal; measured, GitHub sends it on 200s too, naming what
    the ENDPOINT accepts rather than what the caller was denied::

        200  /pulls        x-accepted-github-permissions: pull_requests=read
        403  /check-runs   x-accepted-github-permissions: checks=read

    So a throttle carrying it must still be classified as a throttle."""
    _raise_from_urlopen(monkeypatch, _http_error(
        403,
        body=SECONDARY_LIMIT_BODY,
        x_accepted_github_permissions="checks=read",
        x_accepted_oauth_scopes="repo",
        x_ratelimit_remaining="4999",
    ))

    with pytest.raises(PRWatchRateLimitError):
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")


@pytest.mark.asyncio
async def test_a_secondary_limit_identified_only_in_the_body_is_transient(
    monkeypatch,
):
    """Review rounds 2 and 3, and the reason this asserts on the BODY.

    GitHub's retry guidance ends "Otherwise, wait for at least one minute
    before retrying" — a secondary limit may carry NEITHER ``retry-after``
    NOR ``x-ratelimit-remaining: 0``, and identifies itself in the message
    instead. Nothing in the headers of this response distinguishes it from a
    permission refusal; the message does."""
    _raise_from_urlopen(monkeypatch, _http_error(
        403,
        body=SECONDARY_LIMIT_BODY,
        x_ratelimit_limit="5000",
        x_ratelimit_remaining="4999",
    ))

    with pytest.raises(PRWatchRateLimitError):
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")


@pytest.mark.asyncio
async def test_a_rate_limited_status_does_not_degrade_the_rollup(monkeypatch):
    """The finding itself. A throttled status request must propagate, not
    come back as an "unreadable" gate class."""
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": {"total_count": 1, "check_runs": [GREEN_RUN]},
        "/status": PRWatchRateLimitError("GitHub rate limit hit (403)"),
    }))

    with pytest.raises(PRWatchRateLimitError):
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")


@pytest.mark.asyncio
async def test_a_rate_limited_checks_read_does_not_trigger_the_fallback(monkeypatch):
    """The same hazard on the other endpoint: falling back here would answer
    off the Actions API and caveat a rollup that was never refused."""
    seen: list = []
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": PRWatchRateLimitError("GitHub rate limit hit (403)"),
        "/actions/runs": {"total_count": 1, "workflow_runs": [GREEN_RUN]},
        "/status": EMPTY_STATUS,
    }, seen=seen))

    with pytest.raises(PRWatchRateLimitError):
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert not [u for u in seen if "/actions/runs" in u]


@pytest.mark.asyncio
async def test_a_throttled_poll_stays_pending_instead_of_settling_partial(
    monkeypatch, _token
):
    """End to end, the scenario the review described: green checks plus a
    throttled status request used to settle terminally PARTIAL. A wait must
    keep waiting through a rate limit."""
    st = await _poll(monkeypatch, {
        "/pulls/": {"state": "open", "merged": False, "head": {"sha": SHA}},
        "/check-runs": {"total_count": 1, "check_runs": [GREEN_RUN]},
        "/status": PRWatchRateLimitError("GitHub rate limit hit (403)"),
    })

    assert st.outcome is Outcome.PENDING
    assert st.data["blocked"] == "network"
    assert "caveat" not in st.data


@pytest.mark.asyncio
async def test_a_403_with_no_throttle_evidence_at_all_is_a_refusal(monkeypatch):
    """A 403 that proves nothing is read as a refusal, which is what lets the
    fallback fire. Safe because the throttle GitHub can send without headers
    still names itself in the body (the test above), so "no evidence" really
    does mean no throttle — not merely an unlabelled one."""
    err = urllib.error.HTTPError("https://api.github.com/x", 403, "err", None, None)
    _raise_from_urlopen(monkeypatch, err)

    with pytest.raises(PRWatchAuthError) as caught:
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")

    assert not isinstance(caught.value, PRWatchRateLimitError)


@pytest.mark.asyncio
async def test_a_401_is_always_a_credential_problem(monkeypatch):
    """The one status GitHub never uses for a throttle. Measured: a bad token
    answers 401 with no headers at all, so it cannot prove itself by header
    and is classified by status instead."""
    err = urllib.error.HTTPError("https://api.github.com/x", 401, "err", None, None)
    _raise_from_urlopen(monkeypatch, err)

    with pytest.raises(PRWatchAuthError):
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")


@pytest.mark.asyncio
async def test_a_primary_limit_still_wins_over_an_absent_body(monkeypatch):
    """The header markers stay authoritative on their own — a body is extra
    evidence, not a replacement for the two signals GitHub does send."""
    _raise_from_urlopen(monkeypatch, _http_error(403, x_ratelimit_remaining="0"))

    with pytest.raises(PRWatchRateLimitError):
        await prw._github_get("https://api.github.com/x", token="t", timeout=1, ref="r")


@pytest.mark.asyncio
async def test_the_measured_permission_403_still_reaches_the_fallback(monkeypatch):
    """The over-correction guard. Requiring proof of a refusal must not have
    made the refusal this whole change exists for unprovable: these are the
    exact headers the live fine-grained PAT returns for check-runs."""
    def fake_urlopen(req, timeout=None):
        if "/check-runs" in req.full_url:
            raise _http_error(
                403,
                body=PERMISSION_BODY,
                x_accepted_github_permissions="checks=read",
                x_ratelimit_limit="5000",
                x_ratelimit_remaining="4994",
            )
        import json as _json

        class _Resp:
            @staticmethod
            def read():
                if "/actions/runs" in req.full_url:
                    return _json.dumps(
                        {"total_count": 1, "workflow_runs": [GREEN_RUN]}
                    ).encode()
                return _json.dumps(EMPTY_STATUS).encode()

        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    rollup = await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert rollup.source == CHECKS_SOURCE_WORKFLOW_RUNS
    assert rollup.unreadable == ("check-runs",)


@pytest.mark.asyncio
async def test_a_body_only_throttle_does_not_degrade_the_rollup(monkeypatch):
    """End to end through the real classification rather than a pre-decided
    injected exception: the header-less secondary limit must not mark an
    endpoint unreadable."""
    def fake_urlopen(req, timeout=None):
        raise _http_error(
            403, body=SECONDARY_LIMIT_BODY, x_ratelimit_remaining="4999"
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(PRWatchNetworkError) as caught:
        await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert not isinstance(caught.value, PRWatchAuthError)


@pytest.mark.asyncio
async def test_legacy_statuses_alone_are_still_a_rollup(monkeypatch):
    """Surviving mutant, round 5: nothing distinguished "no check evidence"
    from "no CHECK-RUNS evidence". A repository whose gates report through
    legacy commit statuses has a real rollup even with both check endpoints
    refused, and hard-failing on it would blind the wait to CI it can see."""
    monkeypatch.setattr(prw, "_github_get", _router({
        "/check-runs": PRWatchAuthError("403", status_code=403),
        "/actions/runs": PRWatchAuthError("403", status_code=403),
        "/status": {"state": "success", "total_count": 1,
                    "statuses": [{"context": "buildkite", "state": "success"}]},
    }))

    rollup = await fetch_check_rollup(BASE, SHA, token="t", timeout=1, ref="o/r#20")

    assert rollup.unreadable == ("check-runs", "actions-runs")
    assert rollup.complete is False
    assert prw._check_verdict(rollup.check_runs, rollup.combined_status) == "success"
    # Still caveated: a pass read this way is not an unqualified pass.
    assert "all check runs" in rollup.caveat()
