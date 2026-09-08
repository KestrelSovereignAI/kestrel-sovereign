"""Signal source for GitHub PR/issue activity (#1618).

Provider-owned completion signals cover the job lifecycle, but not PR-only
activity that lands after a coding job exits — a reviewer leaving a comment,
CI turning red, or the PR getting merged or closed. This source removes the
need for a manual provider-status or ``gh pr view`` poll.

The polling half is the ``github_pr_watch`` ACTION cron task (wired in
``signals/sources/scheduler.py`` and handled by
``SchedulerFeature._run_github_pr_watch``). Each poll fetches the current
PR/issue state, reduces it to a small set of watched fields, hashes them
into a fingerprint, and compares against the persisted fingerprint. Only
a *relevant* change (one whose category is in the watch's ``triggers``)
emits one COGNITION ``github.pr_activity`` signal. A no-op poll — same
fingerprint — emits nothing, so the agent is not woken every 15 minutes.

Distinct from no-change, the handler reports ``blocked: auth`` /
``blocked: network`` when the fetch itself fails, so a misconfigured
token or a flaky network is never silently read as "nothing happened".

Everything in this module except :func:`fetch_pr_state` is pure and
side-effect free so the change-detection contract is unit-testable
without a network or a database.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple, List

from kestrel_sdk.signals import (
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Trust,
)

logger = logging.getLogger(__name__)


SOURCE_NAME = "github.pr_activity"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "github_pr_activity.md"
)


# Fields reduced from the raw GitHub PR/issue JSON. Changing any of these
# changes the fingerprint. Kept deliberately small — a PR's ``updated_at``
# bumps on almost any activity, so we track the semantically-meaningful
# fields explicitly rather than fingerprinting the whole payload.
#
# ``checks_status`` is NOT a field GitHub puts on a pull/issue payload — it
# is a derived summary that :func:`fetch_pr_state` computes from the real
# Checks (``/commits/{sha}/check-runs``) and Statuses
# (``/commits/{sha}/status``) APIs via :func:`summarize_checks`. The
# normalizer reads whatever the fetcher attached, so a raw payload that
# omits the key normalizes to an empty checks summary rather than raising.
WATCHED_FIELDS: Tuple[str, ...] = (
    "state",
    "merged",
    "comments",
    "review_comments",
    "updated_at",
    "head_sha",
    "checks_status",
    "mergeable_state",
)


# Map each watched field to a coarse trigger category. A watch declares
# which categories should wake it via ``triggers``; an ``updated_at``-only
# bump (category ``update``) is excluded from the defaults so routine
# timestamp churn doesn't wake the agent.
CATEGORY_FIELDS: Dict[str, Tuple[str, ...]] = {
    "state": ("state",),
    "merge": ("merged",),
    "comments": ("comments", "review_comments"),
    "checks": ("checks_status",),
    "update": ("updated_at", "head_sha", "mergeable_state"),
}

# Default trigger set: wake on state transitions, merge/close, new
# comments, and CI/check completion — but not on a bare ``updated_at``
# bump. Pass ``triggers=["any"]`` to wake on every fingerprint change.
DEFAULT_TRIGGERS: Tuple[str, ...] = ("state", "merge", "comments", "checks")


# Which endpoint a head commit's check evidence was actually read from.
# ``check_runs`` is the full rollup (``/commits/{sha}/check-runs``, every
# app's check runs). ``workflow_runs`` is the narrower Actions-only fallback
# (``/actions/runs?head_sha=``) used when the credential cannot read Checks
# at all — see :func:`fetch_check_rollup` for why that is a permanent state
# for a fine-grained PAT rather than a transient one.
CHECKS_SOURCE_CHECK_RUNS = "check_runs"
CHECKS_SOURCE_WORKFLOW_RUNS = "workflow_runs"
# Both check endpoints refused and only legacy commit statuses were readable.
# A real rollup, but not one that saw a single check run.
CHECKS_SOURCE_STATUS_ONLY = "status_only"

# "This endpoint will return up to 1,000 results for each search when using
# the following parameters: actor, branch, check_suite_id, created, event,
# head_sha, status." — GitHub, List workflow runs for a repository. The
# fallback queries by ``head_sha``, so it is subject to this cap.
ACTIONS_RESULT_CEILING = 1000


class PRWatchError(Exception):
    """Base class for github_pr_watch fetch failures."""


class PRWatchAuthError(PRWatchError):
    """Auth/permission failure (401/403) — distinct from a no-change poll.

    ``status_code`` carries which one, because the two are not
    interchangeable downstream: a 403 is *this endpoint* refusing a valid
    credential, while a 401 is the credential itself failing and says nothing
    about any endpoint. Only the former may degrade a rollup. It defaults to
    ``None`` — "not known to be a 403" — so a caller that constructs one
    without saying cannot accidentally unlock the degrade path.
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PRWatchNetworkError(PRWatchError):
    """Network/transport failure (timeout, DNS, 5xx) — distinct from no-change."""


class PRWatchRateLimitError(PRWatchNetworkError):
    """An exhausted GitHub rate limit, which is *also* reported as 403.

    A subclass of the network error rather than the auth one, and that is the
    whole point: a rate limit clears on its own, so every ``except
    PRWatchNetworkError`` already treats it correctly, while the permission
    handling that degrades a rollup (:func:`fetch_check_rollup`) does not see
    it at all. Collapsing the two is how a five-minute rate limit would settle
    a wait terminally on half a rollup.

    Measured 2026-09-08 — the response headers separate the two cleanly::

        permission gap  403  x-accepted-github-permissions: checks=read
                             x-ratelimit-remaining: 4999
        primary limit   403  x-ratelimit-remaining: 0
        secondary limit 403  retry-after: <seconds>
    """


# ---------------------------------------------------------------------------
# Pure change-detection core
# ---------------------------------------------------------------------------


def normalize_pr_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a raw GitHub PR/issue JSON object to the watched fields.

    Accepts either a PR object (has ``head``) or an issue object. Missing
    fields normalize to empty/zero so a partial payload still produces a
    stable fingerprint rather than raising.
    """
    head = raw.get("head")
    if isinstance(head, dict):
        head_sha = str(head.get("sha", "") or "")
    else:
        head_sha = str(raw.get("head_sha", "") or "")
    return {
        "state": str(raw.get("state", "") or ""),
        "merged": bool(raw.get("merged", False)),
        "comments": int(raw.get("comments", 0) or 0),
        "review_comments": int(raw.get("review_comments", 0) or 0),
        "updated_at": str(raw.get("updated_at", "") or ""),
        "head_sha": head_sha,
        # Derived by fetch_pr_state via summarize_checks; absent on a raw
        # GitHub payload, which normalizes to an empty summary.
        "checks_status": str(raw.get("checks_status", "") or ""),
        "mergeable_state": str(raw.get("mergeable_state", "") or ""),
    }


def compute_fingerprint(normalized: Dict[str, Any]) -> str:
    """Stable SHA-256 over the normalized watched fields."""
    blob = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# GitHub check-run conclusions that mean the check did NOT pass. ``success``,
# ``neutral`` and ``skipped`` are treated as non-blocking passes: they mean
# the check did not need to run. ``cancelled`` deliberately stays a failure —
# it means the check was stopped before it could tell us anything, which is an
# absence of evidence rather than a pass (#2939).
_FAIL_CONCLUSIONS = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "stale",
     "startup_failure"}
)


def _positive_count(value: Any) -> bool:
    """Whether ``value`` is a GitHub ``total_count`` greater than zero.

    Tolerates the field being absent, ``null``, or a string; anything that is
    not a positive integer reads as "no statuses reported".
    """
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _check_verdict(
    check_runs: Any = None, combined_status: Any = None
) -> str:
    """Reduce raw check-runs + combined commit status to a coarse verdict.

    Returns one of:
      * ``"unknown"`` — the rollup was not read at all (neither payload was
        fetched); an evidence gap, NOT a statement about the checks,
      * ``"none"``    — read successfully and empty: no check runs and no
        statuses exist for this commit (no CI ran),
      * ``"pending"`` — at least one check/status is not yet terminal,
      * ``"failure"`` — everything terminal and at least one failed,
      * ``"success"`` — everything terminal and all passed.

    A check run counts as terminal on ``status == "completed"`` whatever its
    conclusion, so ``skipped``/``neutral`` never hold the rollup open.
    """
    runs_read = isinstance(check_runs, (dict, list))
    runs: List[dict] = []
    if isinstance(check_runs, dict):
        raw_runs = check_runs.get("check_runs", []) or []
    elif isinstance(check_runs, list):
        raw_runs = check_runs
    else:
        raw_runs = []
    for r in raw_runs:
        if isinstance(r, dict):
            runs.append(r)

    status_read = isinstance(combined_status, dict)
    combined_state = ""
    statuses: List[dict] = []
    if isinstance(combined_status, dict):
        for s in combined_status.get("statuses", []) or []:
            if isinstance(s, dict):
                statuses.append(s)
        # GitHub reports the combined ``state`` as "pending" for a commit that
        # carries ZERO legacy statuses — the shape of every Actions-only repo,
        # where CI reports through check runs instead. Reading that as "a
        # check is still running" pins the verdict at pending forever (#2939),
        # so the combined state is evidence only when a status backs it.
        if statuses or _positive_count(combined_status.get("total_count")):
            combined_state = str(combined_status.get("state", "") or "").lower()

    if not runs_read and not status_read:
        return "unknown"

    # A ``total_count`` above what came back means gates this verdict never
    # saw (a page the fetch could not follow, or a caller that read one
    # page). An unread gate can only hide MORE failures, never turn an
    # observed one into a pass, so it acts in exactly two places: it stops
    # an empty page from reading as "no CI ran" (the rollup is not empty, it
    # is unread), and it lowers "success" to "pending". It never touches an
    # observed failure.
    unread_runs = False
    if isinstance(check_runs, dict):
        total = check_runs.get("total_count")
        unread_runs = isinstance(total, int) and total > len(runs)

    if not runs and not statuses and not combined_state:
        return "pending" if unread_runs else "none"

    # Not terminal yet if any check run is still queued/in_progress, or the
    # combined/legacy status is still pending.
    for r in runs:
        if str(r.get("status", "") or "").lower() != "completed":
            return "pending"
    if combined_state == "pending":
        return "pending"
    for s in statuses:
        if str(s.get("state", "") or "").lower() == "pending":
            return "pending"

    # Everything terminal — any failure makes the verdict a failure.
    for r in runs:
        if str(r.get("conclusion", "") or "").lower() in _FAIL_CONCLUSIONS:
            return "failure"
    if combined_state in ("failure", "error"):
        return "failure"
    for s in statuses:
        if str(s.get("state", "") or "").lower() in ("failure", "error"):
            return "failure"

    # Everything READ passed; gates never read keep it short of "success".
    if unread_runs:
        return "pending"

    return "success"


def summarize_checks(
    check_runs: Any = None,
    combined_status: Any = None,
    *,
    source: str = CHECKS_SOURCE_CHECK_RUNS,
    unreadable: Tuple[str, ...] = (),
) -> str:
    """Reduce real GitHub check-runs + commit statuses to a stable string.

    Standard pull/issue payloads carry no aggregate check field, so the
    watcher fetches the head commit's checks itself:

      - ``check_runs`` is the JSON from ``/commits/{sha}/check-runs``
        (``{"check_runs": [{"name", "status", "conclusion"}, ...]}``) or a
        bare list of those run objects.
      - ``combined_status`` is the JSON from ``/commits/{sha}/status``
        (``{"state", "statuses": [{"context", "state"}, ...]}``).

    The summary captures each check run's ``status``/``conclusion`` — EVERY
    run, one entry each: ``/commits/{sha}/check-runs`` defaults to
    ``filter=latest``, so re-run attempts are already collapsed upstream, and
    the same-named duplicates that do arrive are two concurrent check suites
    (this repo's CI fires on both ``push`` and ``pull_request``), each a real
    gate that must count (#3191) — plus each legacy status context's
    ``state``, plus a ``combined`` verdict from :func:`_check_verdict`, the
    one rollup the CI wait provider also uses, so a CI transition — queued →
    in_progress → completed/success|failure — changes the string (and
    therefore the fingerprint). It is order-independent (parts are sorted)
    so the same set of checks always summarizes identically. Returns ``""``
    when there are no checks or statuses at all, which is indistinguishable
    from "no checks key in payload".

    ``source``/``unreadable`` describe how completely the rollup was read
    (see :class:`CheckRollup`). Both are fingerprinted, for two reasons. A
    degraded read must not summarize identically to a complete one, or the
    watch would report "no change" across the moment its own visibility
    changed; and an empty *degraded* rollup must not collapse to ``""``,
    which this module's callers read as "no checks exist" — a claim a blind
    poll cannot make. So whenever either is non-default the summary is
    non-empty even with nothing else in it.
    """
    parts = []
    # Recorded separately from the check/status parts so they cannot be
    # mistaken for evidence, while still keeping the summary non-empty.
    provenance = []
    if source != CHECKS_SOURCE_CHECK_RUNS:
        provenance.append(f"source={source}")
    if unreadable:
        provenance.append("unreadable=" + ",".join(sorted(unreadable)))

    if isinstance(combined_status, dict):
        for s in combined_status.get("statuses", []) or []:
            if isinstance(s, dict):
                ctx = str(s.get("context", "") or "")
                st = str(s.get("state", "") or "")
                parts.append(f"status:{ctx}={st}")

    runs: Any
    if isinstance(check_runs, dict):
        runs = check_runs.get("check_runs", []) or []
    elif isinstance(check_runs, list):
        runs = check_runs
    else:
        runs = []
    for r in runs:
        if isinstance(r, dict):
            name = str(r.get("name", "") or "")
            status = str(r.get("status", "") or "")
            conclusion = str(r.get("conclusion", "") or "")
            parts.append(f"check:{name}={status}/{conclusion}")

    # "unknown" (nothing fetched) and "none" (fetched, empty) both mean there
    # is no verdict to fingerprint; the caller treats "" as "no checks".
    verdict = _check_verdict(check_runs, combined_status)
    combined_state = verdict if verdict in ("pending", "failure", "success") else ""
    if not parts and not combined_state and not provenance:
        return ""

    parts.sort()
    return ";".join([f"combined={combined_state}", *parts, *provenance])


def changed_categories(
    prev: Optional[Dict[str, Any]], curr: Dict[str, Any]
) -> Set[str]:
    """Return the set of trigger categories whose fields differ."""
    prev = prev or {}
    cats: Set[str] = set()
    for category, fields in CATEGORY_FIELDS.items():
        for f in fields:
            if prev.get(f) != curr.get(f):
                cats.add(category)
                break
    return cats


@dataclass
class WatchDecision:
    """Outcome of evaluating one poll against the persisted fingerprint."""

    should_signal: bool
    fingerprint: str
    normalized: Dict[str, Any]
    changed: Set[str] = field(default_factory=set)
    matched: Set[str] = field(default_factory=set)
    reason: str = ""


def evaluate_pr_watch(
    raw_state: Dict[str, Any],
    *,
    last_fingerprint: Optional[str] = None,
    last_normalized: Optional[Dict[str, Any]] = None,
    triggers: Optional[Iterable[str]] = None,
) -> WatchDecision:
    """Decide whether a poll should emit a wake signal.

    Contract:
      - First observation (``last_fingerprint is None``): persist the
        baseline, do NOT signal. Registering a watch shouldn't immediately
        wake the agent.
      - Same fingerprint: no change, no signal.
      - Changed fingerprint, but no changed category is in ``triggers``:
        no signal (e.g. a bare ``updated_at`` bump under the defaults).
      - Changed fingerprint with a matching category: signal.

    The returned ``fingerprint``/``normalized`` should always be persisted
    by the caller (even on a no-signal change) so the next poll compares
    against the latest observed state, not a stale baseline.
    """
    trigger_set = {str(t) for t in triggers} if triggers else set(DEFAULT_TRIGGERS)
    normalized = normalize_pr_state(raw_state)
    fingerprint = compute_fingerprint(normalized)

    if last_fingerprint is None:
        return WatchDecision(
            should_signal=False,
            fingerprint=fingerprint,
            normalized=normalized,
            reason="first_observation",
        )

    if fingerprint == last_fingerprint:
        return WatchDecision(
            should_signal=False,
            fingerprint=fingerprint,
            normalized=normalized,
            reason="no_change",
        )

    changed = changed_categories(last_normalized, normalized)
    if "any" in trigger_set:
        matched = set(changed)
    else:
        matched = changed & trigger_set

    if not matched:
        return WatchDecision(
            should_signal=False,
            fingerprint=fingerprint,
            normalized=normalized,
            changed=changed,
            reason="change_not_in_triggers",
        )

    return WatchDecision(
        should_signal=True,
        fingerprint=fingerprint,
        normalized=normalized,
        changed=changed,
        matched=matched,
        reason="change_matched",
    )


# ---------------------------------------------------------------------------
# Fetch (the only side-effecting function here)
# ---------------------------------------------------------------------------


async def _github_get_paged(
    url: str, key: str, *, token: str, timeout: int, ref: str
) -> Any:
    """GET a ``total_count``-plus-list GitHub collection, following pages.

    ``url`` must already carry ``per_page`` and any other query parameters;
    ``key`` is the list field to accumulate (``check_runs``,
    ``workflow_runs``). Returns the first page's object with that list
    extended, so callers see the shape the API documents.

    GitHub pages these at 30 by default; callers ask for 100 and this follows
    ``page=`` until ``total_count`` is met, so a gate on page two is read
    rather than merely detected. The crawl is bounded by GitHub's own
    ``total_count``: every non-empty page grows the list and an empty page
    ends the loop, so it issues at most ``ceil(total_count / 100)`` requests.
    There is deliberately no fixed page cap on top of that — a rollup larger
    than a cap would be read short every poll, and :func:`_check_verdict`
    would then hold it at ``"pending"`` forever, the one state class this
    rollup must never settle in (#2939).

    One paginator, not one per endpoint: the bound above is the load-bearing
    half, and a second copy of it is a second place for it to drift.
    """
    first = await _github_get(url, token=token, timeout=timeout, ref=ref)
    if not isinstance(first, dict):
        return first
    items = [r for r in (first.get(key) or []) if isinstance(r, dict)]
    total = first.get("total_count")
    page = 1
    sep = "&" if "?" in url else "?"
    while isinstance(total, int) and len(items) < total:
        page += 1
        more = await _github_get(
            f"{url}{sep}page={page}",
            token=token, timeout=timeout, ref=f"{ref} page {page}",
        )
        batch = (
            [r for r in (more.get(key) or []) if isinstance(r, dict)]
            if isinstance(more, dict) else []
        )
        if not batch:
            break
        items.extend(batch)
    return {**first, key: items}


async def _github_get_check_runs(
    base: str, head_sha: str, *, token: str, timeout: int, ref: str
) -> Any:
    """Every check run for ``head_sha`` (GitHub's ``filter=latest`` default).

    The complete rollup: every app's check runs, GitHub Actions and third
    parties alike. Needs the ``checks`` read permission — see
    :func:`fetch_check_rollup` for what happens to a credential that has no
    way to hold it.
    """
    return await _github_get_paged(
        f"{base}/commits/{head_sha}/check-runs?per_page=100",
        "check_runs", token=token, timeout=timeout, ref=ref,
    )


async def _github_get_workflow_runs(
    base: str, head_sha: str, *, token: str, timeout: int, ref: str
) -> Any:
    """Every Actions workflow run for ``head_sha``.

    The narrower half of the rollup, readable with the ``actions`` permission
    instead of ``checks``. A workflow *run* is coarser than a check *run* —
    one run covers all of its jobs, so six job-level check runs collapse to
    the single run that produced them — which is enough for a verdict (a
    failed job fails its run) but loses per-job names and per-job timing.
    """
    return await _github_get_paged(
        f"{base}/actions/runs?head_sha={head_sha}&per_page=100",
        "workflow_runs", token=token, timeout=timeout, ref=ref,
    )


def _actions_ceiling_hit(payload: Any) -> bool:
    """Whether GitHub capped this ``head_sha`` query at its 1,000-result limit.

    The shape is a ``total_count`` above what the crawl could ever collect,
    with a full ceiling's worth in hand. It matters because
    :func:`_check_verdict` lowers ``success`` to ``pending`` whenever the
    rollup was read short — the right call for a page the fetch missed, and
    the **wrong** one here, where the missing runs are unreachable by any
    number of requests. Left alone it is a wait that can never settle, which
    is the one state class this rollup must never reach (#2939).
    """
    if not isinstance(payload, dict):
        return False
    runs = [r for r in (payload.get("workflow_runs") or []) if isinstance(r, dict)]
    total = payload.get("total_count")
    return (
        isinstance(total, int)
        and total > len(runs)
        and len(runs) >= ACTIONS_RESULT_CEILING
    )


def _workflow_runs_as_check_runs(payload: Any) -> Any:
    """Project an ``/actions/runs`` payload onto the check-runs shape.

    ``status`` (``queued``/``in_progress``/``completed``) and ``conclusion``
    (``success``/``failure``/``skipped``/…) are the same vocabulary on both
    endpoints, so the projection lets :func:`_check_verdict` and
    :func:`summarize_checks` read Actions runs unchanged rather than growing
    a second, separately-drifting reducer.

    ``total_count`` is carried across verbatim when GitHub sent one, so the
    unread-gate protection in :func:`_check_verdict` — which lowers
    ``success`` to ``pending`` when the rollup was read short — keeps
    applying to the fallback. The one exception is GitHub's own 1,000-result
    ceiling (:func:`_actions_ceiling_hit`): those runs are unreachable rather
    than merely unread, so the count is clamped to what was collected and the
    shortfall is reported as a blind spot instead of as an open gate.
    """
    if not isinstance(payload, dict):
        return payload
    runs = []
    for r in payload.get("workflow_runs") or []:
        if not isinstance(r, dict):
            continue
        runs.append(
            {
                "name": str(r.get("name", "") or ""),
                "status": str(r.get("status", "") or ""),
                "conclusion": r.get("conclusion"),
            }
        )
    total = payload.get("total_count")
    if _actions_ceiling_hit(payload):
        total = len(runs)
    return {
        "total_count": total if isinstance(total, int) else len(runs),
        "check_runs": runs,
    }


def _is_rate_limited(headers: Any, body: bytes = b"") -> bool:
    """Whether a 403 shows positive evidence of a throttle rather than a refusal.

    GitHub reports an exhausted rate limit and a permission refusal with the
    same status code, and there is **no header that identifies the refusal**.
    Measured 2026-09-08, which is what settles it::

        200  /pulls        x-accepted-github-permissions: pull_requests=read
        200  /actions/runs x-accepted-github-permissions: actions=read
        403  /check-runs   x-accepted-github-permissions: checks=read

    The header is emitted on success too: it names what the *endpoint* accepts,
    not what the caller was denied, so it cannot distinguish anything. (An
    earlier revision of this module treated its presence as proof of a
    refusal. It is not.) The same goes for ``x-accepted-oauth-scopes``, which
    a classic token receives on every response, sometimes empty.

    So the discrimination has to run the other way, on the three things that
    do positively mark a throttle:

    * ``retry-after``                 — secondary limit
    * ``x-ratelimit-remaining: 0``    — primary limit
    * the phrase "rate limit" in the message body

    The body matters because GitHub documents a secondary limit that carries
    **neither** header — its retry guidance ends "Otherwise, wait for at least
    one minute before retrying" — and identifies itself in the message
    instead. The refusal's message does not contain the phrase; measured, it
    reads ``Resource not accessible by personal access token``.

    Anything with no throttle evidence at all is treated as a refusal, which
    is what lets :func:`fetch_check_rollup` degrade. That direction is chosen
    deliberately: a throttle misread as a refusal settles a caveated
    non-terminal-looking PARTIAL one poll early, while a refusal misread as a
    throttle leaves the wait pending on a gate that will never clear.
    """
    if headers is not None:
        try:
            if str(headers.get("retry-after", "") or "").strip():
                return True
            if str(headers.get("x-ratelimit-remaining", "") or "").strip() == "0":
                return True
        except Exception:  # pragma: no cover - defensive; headers are mapping-like
            pass
    try:
        text = body.decode("utf-8", "replace").lower() if body else ""
    except Exception:  # pragma: no cover - defensive
        return False
    return "rate limit" in text


async def _github_get(
    url: str, *, token: str, timeout: int, ref: str
) -> Any:
    """GET + JSON-decode one GitHub API URL.

    Raises :class:`PRWatchRateLimitError` on a 429 or on a 403 that shows
    throttle evidence, :class:`PRWatchAuthError` on any other 401/403, and
    :class:`PRWatchNetworkError` on any other transport/HTTP/parse failure, so
    the caller can report ``blocked: auth`` / ``blocked: network`` distinctly
    from a no-change poll. ``ref`` is only used to label errors.

    The split is load-bearing downstream: the auth error is what
    :func:`fetch_check_rollup` degrades a rollup on, and degrading on a
    transient would turn a five-minute throttle into a terminal verdict read
    off half the evidence. See :func:`_is_rate_limited` for why the throttle
    is the side that must be proven.
    """
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "kestrel-agent",
        },
    )

    def _do() -> bytes:
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            # Read the error body here, on the worker thread, and stash it on
            # the exception: it is readable only once, it is the only place a
            # header-less secondary rate limit identifies itself, and doing
            # the read back on the event loop would block it.
            try:
                e.kestrel_error_body = e.read()
            except Exception:  # pragma: no cover - defensive
                e.kestrel_error_body = b""
            raise

    try:
        resp = await asyncio.to_thread(_do)
    except urllib.error.HTTPError as e:
        headers = getattr(e, "headers", None)
        body = getattr(e, "kestrel_error_body", b"")
        if e.code == 401:
            # Never a throttle: GitHub answers 401 only for a credential it
            # would not accept at all (measured: "Bad credentials").
            raise PRWatchAuthError(
                f"GitHub returned {e.code} for {ref}", status_code=e.code
            ) from e
        if e.code == 429 or (e.code == 403 and _is_rate_limited(headers, body)):
            raise PRWatchRateLimitError(
                f"GitHub rate limit hit ({e.code}) for {ref}"
            ) from e
        if e.code == 403:
            raise PRWatchAuthError(
                f"GitHub returned {e.code} for {ref}", status_code=e.code
            ) from e
        raise PRWatchNetworkError(f"GitHub HTTP {e.code} for {ref}") from e
    except urllib.error.URLError as e:
        raise PRWatchNetworkError(f"network error for {ref}: {e}") from e
    except Exception as e:  # pragma: no cover - defensive
        raise PRWatchNetworkError(f"unexpected error for {ref}: {e}") from e

    try:
        return json.loads(resp)
    except (ValueError, TypeError) as e:
        raise PRWatchNetworkError(
            f"could not parse GitHub response for {ref}: {e}"
        ) from e


@dataclass(frozen=True)
class CheckRollup:
    """One head commit's check evidence, plus how completely it was read.

    ``check_runs`` is always check-runs-shaped — either the real Checks
    payload or an Actions payload projected onto it by
    :func:`_workflow_runs_as_check_runs` — so every downstream reducer takes
    it unchanged. ``source`` says which, and ``unreadable`` names the
    endpoints the credential was refused.

    The pair exists because "the rollup is empty" and "the part of the rollup
    I can see is empty" are different claims, and this module's whole
    contract turns on not confusing them (#2939): an empty rollup is terminal
    ("no CI ran"), while an empty *partial* rollup is only terminal about
    what was visible.
    """

    check_runs: Any = None
    combined_status: Any = None
    source: str = CHECKS_SOURCE_CHECK_RUNS
    unreadable: Tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether every gate class this module knows about was readable."""
        return not self.unreadable and self.source == CHECKS_SOURCE_CHECK_RUNS

    def caveat(self) -> str:
        """Operator-facing sentence naming what this rollup could not see.

        Empty when the rollup is complete. Names the blind spot in terms of
        what could be hiding in it, not just which endpoint failed — the
        point of the sentence is that a reader can decide whether it matters
        for their repository.
        """
        if self.complete:
            return ""
        holes = []
        if self.source == CHECKS_SOURCE_WORKFLOW_RUNS:
            holes.append(
                "check runs from apps other than GitHub Actions (the Checks "
                "API was refused; read via the Actions API instead)"
            )
        elif self.source == CHECKS_SOURCE_STATUS_ONLY:
            holes.append(
                "every check run (the Checks and Actions APIs were both "
                "refused; read from legacy commit statuses instead)"
            )
        elif "check-runs" in self.unreadable:
            holes.append("all check runs (the Checks API was refused)")
        if "actions-ceiling" in self.unreadable:
            holes.append(
                f"workflow runs beyond GitHub's {ACTIONS_RESULT_CEILING}-result "
                f"ceiling for one head SHA"
            )
        if not holes:
            holes.append("part of the rollup (" + ",".join(self.unreadable) + ")")
        return "this verdict cannot see " + "; ".join(holes)


async def fetch_check_rollup(
    base: str, head_sha: str, *, token: str, timeout: int, ref: str
) -> CheckRollup:
    """Read ``head_sha``'s check evidence, degrading rather than going blind.

    Order matters. The Checks API is tried first because it is the complete
    rollup. Only an *authorization* refusal falls back — a network error still
    propagates, because this module's auth/network distinction is the
    difference between "a human must act" and "wait and it will clear", and a
    fallback that swallowed transport failures would erase it. An exhausted
    rate limit is reported by GitHub with the same 403 as a refusal, and
    :func:`_github_get` separates the two by header before either reaches
    here: a rate limit arrives as :class:`PRWatchRateLimitError`, is not
    caught below, and leaves the wait pending instead of settling it on a
    rollup that was merely throttled.

    The fallback exists because for a fine-grained personal access token the
    Checks refusal is permanent, not a misconfiguration: ``checks`` is a
    GitHub *App* permission and is absent from the fine-grained repository
    permission list entirely, so ``/commits/{sha}/check-runs`` answers 403
    with ``x-accepted-github-permissions: checks=read`` — naming a permission
    that token type can never hold. ``/actions/runs?head_sha=`` answers the
    same question for GitHub Actions under ``actions=read``, which such a
    token *can* hold, and ``/commits/{sha}/status`` covers integrations that
    report through legacy commit statuses. Between them they are complete for
    a repository whose gates are Actions and statuses, and blind to exactly
    one class: a third-party app that reports only through check runs.

    That residual hole is why the result carries :meth:`CheckRollup.caveat`
    rather than being silently promoted to a full read. Callers must not turn
    an incomplete rollup into an unqualified pass.

    Two things it deliberately does NOT degrade on.

    A **401** is the credential itself failing — expired or revoked, possibly
    mid-poll, after the earlier reads succeeded. It says nothing about any
    one endpoint, nothing else will be readable either, and answering from
    the half already in hand would settle a verdict for a token that can no
    longer see the repository. Only a 403 degrades.

    A refused **status** read propagates rather than being marked unreadable.
    Tolerating it was an addition beyond what the Checks fallback needs, and
    review found two separate ways for it to convert a transient into a
    terminal; measured against the credential this exists for, it buys
    nothing, because that token holds ``statuses=read`` and answers 200. So
    the legacy-status half is all-or-nothing again, exactly as it was before
    the fallback, and the only degradation this function can produce is the
    Checks-to-Actions one it was written for.

    Raises :class:`PRWatchAuthError` when no check evidence at all is
    readable — the genuinely blind case, which no amount of waiting fixes.
    """
    unreadable: List[str] = []
    source = CHECKS_SOURCE_CHECK_RUNS
    checks_error: Optional[PRWatchAuthError] = None
    try:
        check_runs: Any = await _github_get_check_runs(
            base, head_sha, token=token, timeout=timeout, ref=f"{ref} check-runs"
        )
    except PRWatchAuthError as exc:
        if exc.status_code != 403:
            raise
        checks_error = exc
        unreadable.append("check-runs")
        check_runs = None
        try:
            workflow_runs = await _github_get_workflow_runs(
                base, head_sha, token=token, timeout=timeout,
                ref=f"{ref} actions-runs",
            )
            if _actions_ceiling_hit(workflow_runs):
                unreadable.append("actions-ceiling")
            check_runs = _workflow_runs_as_check_runs(workflow_runs)
            source = CHECKS_SOURCE_WORKFLOW_RUNS
        except PRWatchAuthError as actions_exc:
            if actions_exc.status_code != 403:
                raise
            unreadable.append("actions-runs")
            # Neither check endpoint was readable. Whatever the status
            # endpoint returns below is the whole rollup, and saying it came
            # from ``check_runs`` would contradict ``unreadable`` in the same
            # payload.
            source = CHECKS_SOURCE_STATUS_ONLY

    # Not wrapped: a refused status read is not a degradable gate class.
    combined_status: Any = await _github_get(
        f"{base}/commits/{head_sha}/status",
        token=token, timeout=timeout, ref=f"{ref} status",
    )

    if check_runs is None and combined_status is None:
        # Neither Checks nor Actions was readable and the status endpoint
        # answered with nothing. There is no verdict to caveat, only a
        # credential to fix.
        raise PRWatchAuthError(
            f"{ref}: no check evidence is readable — check-runs and "
            f"actions-runs were both refused ({checks_error})",
            status_code=403,
        )

    return CheckRollup(
        check_runs=check_runs,
        combined_status=combined_status,
        source=source,
        unreadable=tuple(unreadable),
    )

async def fetch_pr_state(
    repo: str, number: int, *, token: str, kind: str = "pr", timeout: int = 10
) -> Dict[str, Any]:
    """Fetch a PR's or issue's current state from the GitHub API.

    ``kind="pr"`` queries ``/pulls/{number}`` (the default); ``kind="issue"``
    queries ``/issues/{number}``. PRs and issues share one numbering space,
    so an issue number sent to ``/pulls`` would 404 — the endpoint must match
    the watch type. Issue payloads have no ``head``/``merged``/
    ``mergeable_state``; :func:`normalize_pr_state` already tolerates the
    missing fields.

    GitHub pull/issue payloads carry **no** aggregate check field, so for a
    PR this also fetches the head commit's real check runs
    (``/commits/{sha}/check-runs``) and combined commit status
    (``/commits/{sha}/status``), reduces them via :func:`summarize_checks`,
    and attaches the result as ``checks_status``. That makes a CI
    transition (queued → completed/failure) a real, fingerprint-affecting
    change rather than depending on a field GitHub never sends. Issues have
    no head SHA, so their ``checks_status`` stays empty.

    The checks half degrades before it blocks (:func:`fetch_check_rollup`):
    a credential that cannot read the Checks API falls back to the Actions
    API, and the resulting summary records that it did, so a partial read is
    never fingerprinted as a complete one and an empty partial read is never
    fingerprinted as "no checks exist".

    Raises :class:`PRWatchAuthError` on 401/403 and
    :class:`PRWatchNetworkError` on any other transport/HTTP failure so the
    caller can report ``blocked: auth`` / ``blocked: network`` distinctly
    from a no-change poll. A checks fetch with *no* readable evidence at all
    blocks the whole poll rather than reporting a false "checks cleared"
    change.
    """
    base = f"https://api.github.com/repos/{repo}"
    endpoint = "issues" if kind == "issue" else "pulls"
    ref = f"{repo}#{number}"
    raw = await _github_get(
        f"{base}/{endpoint}/{number}", token=token, timeout=timeout, ref=ref
    )
    if not isinstance(raw, dict):
        raise PRWatchNetworkError(
            f"GitHub returned a non-object payload for {ref}"
        )

    head = raw.get("head")
    head_sha = head.get("sha") if isinstance(head, dict) else None
    if kind != "issue" and head_sha:
        rollup = await fetch_check_rollup(
            base, head_sha, token=token, timeout=timeout, ref=ref
        )
        raw["checks_status"] = summarize_checks(
            rollup.check_runs,
            rollup.combined_status,
            source=rollup.source,
            unreadable=rollup.unreadable,
        )

    return raw


# ---------------------------------------------------------------------------
# Signal source registration + envelope builder
# ---------------------------------------------------------------------------


def _schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"github.pr_activity payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    for key in ("repo", "number"):
        if key not in payload or not str(payload[key]):
            raise ValueError(
                f"github.pr_activity payload missing required key: {key}"
            )
    # Inject defaults the prompt template indexes so a sparse payload still
    # renders cleanly through the dispatcher.
    payload.setdefault("state", "")
    payload.setdefault("merged", "false")
    payload.setdefault("comments", "0")
    payload.setdefault("review_comments", "0")
    payload.setdefault("checks_status", "")
    payload.setdefault("changed", "")
    payload.setdefault("html_url", "")
    payload.setdefault("updated_at", "")
    return payload


def _redact(payload: Dict[str, Any]) -> str:
    """Audit-log summary. Identifiers + change categories only."""
    return (
        f"github.pr_activity "
        f"repo={payload.get('repo', '?')} "
        f"number={payload.get('number', '?')} "
        f"state={payload.get('state', '?')} "
        f"changed={payload.get('changed', '?')}"
    )


def build_github_pr_activity_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        # Defense-in-depth against a misconfigured high-frequency watch;
        # well above any plausible legitimate cadence (a 15-min cron
        # cannot exceed 4/hr per watch).
        rate_limit=RateLimit(per_minute=20, per_hour=120),
        coalescing_window=timedelta(seconds=60),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        # Local-only signal sourced by the agent's own cron polling.
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )


def build_signal_for_pr_change(
    *,
    repo: str,
    number: Any,
    decision: WatchDecision,
    target_agent: str,
    html_url: str = "",
) -> Signal:
    """Build a COGNITION signal envelope for a detected PR change."""
    matched = decision.matched or decision.changed
    payload: Dict[str, Any] = {
        "repo": str(repo),
        "number": str(number),
        "state": str(decision.normalized.get("state", "")),
        "merged": "true" if decision.normalized.get("merged") else "false",
        "comments": str(decision.normalized.get("comments", 0)),
        "review_comments": str(decision.normalized.get("review_comments", 0)),
        "checks_status": str(decision.normalized.get("checks_status", "")),
        "changed": ",".join(sorted(matched)),
        "html_url": str(html_url or ""),
        "updated_at": str(decision.normalized.get("updated_at", "")),
    }
    return Signal(
        source=SOURCE_NAME,
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload=payload,
        target_agent=target_agent,
        # One wake per distinct observed fingerprint for this PR.
        dedupe_key=f"{repo}#{number}:{decision.fingerprint[:12]}",
    )
