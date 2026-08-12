"""Waitable provider for GitHub PR merge/CI-check waits (``ci:<repo>#<n>``).

The unified-wait epic (#1860) documented a ``ci:`` kind in the ``wait`` tool
but never registered a provider, so a merge/check wait could not survive a
restart (#2729). This module is that provider. A handle is a PR reference —
``owner/repo#123`` — and the provider reads the PR's current state plus its
head-commit check runs and combined status to decide a terminal verdict:

  * PR merged                       -> DONE   (the happy terminal)
  * PR closed without merge         -> FAILED
  * open + all checks passed        -> DONE
  * open + a check failed           -> FAILED
  * open + nothing ran at all       -> DONE   with ``checks: "none"``
  * open + checks still running     -> PENDING (keep watching)
  * open + evidence not conclusive  -> PENDING with ``checks: "indeterminate"``

Every state is therefore either terminal or *provably still progressing*
(#2939). A state that is neither is how a wait becomes an indefinite stall.

Five rollup rules keep the verdict honest, each learned from #2939 where
``ci:KestrelSovereignAI/kestrel-sovereign#2934`` reported ``checks: pending``
for hours — and still in its terminal payload — while all 18 check runs on
the head SHA were ``COMPLETED``:

1. **An empty legacy-status list is not "pending".** GitHub's combined
   status endpoint (``/commits/{sha}/status``) returns
   ``{"state": "pending", "statuses": [], "total_count": 0}`` for a commit
   that has no legacy commit statuses at all — which is every Actions-only
   repo. Reading that ``state`` as a real verdict pinned the rollup to
   pending forever. The combined ``state`` is only consulted when at least
   one status actually exists.
2. **Only the latest run per check name counts.** A head SHA accumulates
   check runs from superseded workflow runs; a stale entry (often
   ``cancelled`` when concurrency-cancelled) must not block or fail the
   rollup. Runs are resolved to the newest per (workflow, name).
3. **``COMPLETED`` is terminal for every conclusion.** ``skipped`` /
   ``neutral`` are non-blocking passes; ``cancelled`` stays a failure —
   it is an absence of evidence, not a pass — but rule 2 means only a
   *deliberately* cancelled latest run blocks.
4. **A terminal pass requires the COMPLETE list.** GitHub caps ``per_page``
   at 100, so page 1 of a busy head SHA can omit the one queued or failed
   run that decides the verdict. Every listing is paged to exhaustion,
   cross-checked against its ``total_count``, and a list that could not be
   completed (page cap, mid-pagination error) is ``indeterminate``, never
   ``success``. Reaching the end of the listing is what proves completeness;
   a ``total_count`` that disagrees with the rows GitHub actually served is
   logged rather than allowed to veto the verdict forever.
5. **A collapse must be provable.** Rule 2's grouping needs the workflow
   identity from ``/actions/runs``. When that listing is unavailable (a
   fine-grained token may hold Checks access without Actions read) the
   fallback groups by check *app*, which would merge two different
   workflows that share a job name — letting a newer passing ``test``
   overwrite another workflow's failed ``test``. Such a group is collapsed
   only when collapsing cannot change the verdict; otherwise the rollup is
   ``indeterminate``.

Rules 4 and 5 are one principle: the provider reports what it can *prove*.
``indeterminate`` is non-terminal and loud (a warning plus the reasons in
``data``) — it never fabricates a pass out of evidence it does not hold, and
never fabricates a failure out of a grouping it cannot justify.

A PENDING verdict on a PR GitHub calls ``clean``/mergeable is a
contradiction between two reads. The provider surfaces it (a
``contradiction`` marker in ``data`` plus a warning log) rather than
resolving it silently in either direction — it never fabricates a terminal
from ``mergeable_state``, because GitHub reports ``clean`` on a repo with no
*required* checks even while CI is queued.

The change-detection primitives (``fetch``/``summarize_checks``) are reused
from :mod:`kestrel_sovereign.signals.sources.github_pr_watch`, which is pure
core — this provider does NOT depend on the out-of-tree GitHub feature.

Transient failures (no token, auth error, network blip) return
:class:`Outcome.PENDING`, never a terminal failure: a durable
``wait("ci:...", mode="signal")`` must re-arm and complete once when the PR
truly settles, not fabricate a merge/close from a flaky poll.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Set, Tuple

from kestrel_sdk.tools import Outcome, WaitStatus

logger = logging.getLogger(__name__)

# A CI handle is a PR reference: ``owner/repo#123``. The owner/repo half may
# contain the usual GitHub name characters; the number is the PR/issue id.
_CI_HANDLE_RE = re.compile(r"^(?P<repo>[^\s#]+/[^\s#]+)#(?P<number>\d+)$")

# GitHub check-run conclusions that mean the check did NOT pass. ``success``,
# ``neutral`` and ``skipped`` are treated as non-blocking passes: they mean
# "this check did not need to run". ``cancelled`` is deliberately NOT among
# them — it means the check was stopped before it could tell us anything,
# which is an absence of evidence and must not read as success. Superseded
# auto-cancelled runs are dropped by the latest-run-per-name resolution
# before conclusions are inspected, so only a deliberately cancelled *latest*
# run blocks (#2939).
_FAIL_CONCLUSIONS = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "stale",
     "startup_failure"}
)

# How long after the PR last changed an *empty* rollup may still be read as
# "CI has not registered its checks yet". Outside this window an empty rollup
# is terminal: nothing ran, so there is nothing to wait for (#2939). Inside
# it, the wait is provably still progressing — the window is bounded by the
# clock, so the state always converges.
_EMPTY_ROLLUP_GRACE_SECONDS = 180

# GitHub caps ``per_page`` at 100 on every list endpoint, so a head SHA with
# more checks than that spills onto later pages. Listings are therefore paged
# to exhaustion, bounded by this cap so one poll can never issue an unbounded
# number of requests; hitting the cap marks the evidence incomplete rather
# than silently truncating it (#2939).
_PER_PAGE = 100
_MAX_PAGES = 10


def parse_ci_handle(handle: str) -> Tuple[str, int]:
    """Split a ``owner/repo#123`` CI handle into ``(repo, number)``.

    Raises ``ValueError`` on any other shape so a malformed handle (e.g. a
    bare A2A task id mistakenly registered as ``ci:<id>``) is rejected rather
    than silently mis-fetched.
    """
    m = _CI_HANDLE_RE.match(str(handle or "").strip())
    if not m:
        raise ValueError(
            f"ci wait handle must be 'owner/repo#<number>', got {handle!r}"
        )
    return m.group("repo"), int(m.group("number"))


def _list_and_completeness(payload: Any, key: str) -> Tuple[List[dict], bool]:
    """Extract ``payload[key]`` plus whether it is the WHOLE list.

    Two sources of truth, in order of strength:

    1. ``_truncated`` — :func:`_fetch_all_pages` records whether pagination
       actually reached the end of the listing. That is the *only* signal
       immune to ``total_count`` counting a different set than the pages
       serve: the check-runs endpoint applies ``filter=latest`` to the rows
       it returns but documents nothing about what its count includes. A
       count-only rule would pin such a repo ``indeterminate`` forever —
       re-creating the #2939 stall in a new shape — so a listing the fetcher
       paged to exhaustion is complete regardless of the count.
    2. ``total_count`` — for a payload nobody paged (a single un-paged read,
       or a fabricated one), a count above the rows in hand still proves a
       truncated page, the shape that would let a queued or failed 101st
       check read as a terminal success.

    A bare list, or a dict with neither marker nor count, is taken as
    complete: treating "no evidence of truncation" as truncation would make
    every hand-built rollup indeterminate.
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)], True
    if not isinstance(payload, dict):
        return [], True
    items = [x for x in (payload.get(key) or []) if isinstance(x, dict)]
    truncated = payload.get("_truncated")
    if truncated is not None:
        return items, not truncated
    total = payload.get("total_count")
    if isinstance(total, bool) or not isinstance(total, int):
        return items, True
    return items, len(items) >= total


def _as_run_list(check_runs: Any) -> List[dict]:
    """Normalize ``/commits/{sha}/check-runs`` JSON (or a bare list) to runs."""
    return _list_and_completeness(check_runs, "check_runs")[0]


def _suite_workflow_map(workflow_runs: Any) -> Dict[Any, Dict[str, Any]]:
    """Map ``check_suite_id -> {workflow, order}`` from ``/actions/runs``.

    A check-run payload names its check suite but not the workflow the suite
    belongs to, and GitHub Actions creates one suite per *workflow run*. The
    ``/actions/runs?head_sha=`` listing supplies the missing link, so two runs
    of the same workflow can be collapsed to the newest while two *different*
    workflows that happen to share a job name stay distinct. Optional: an
    absent/failed listing degrades to grouping by the check's app, and
    :func:`latest_check_runs` then treats the resulting collapses as
    unproven rather than trusting the guess.
    """
    if isinstance(workflow_runs, dict):
        raw = workflow_runs.get("workflow_runs", []) or []
    elif isinstance(workflow_runs, list):
        raw = workflow_runs
    else:
        raw = []
    out: Dict[Any, Dict[str, Any]] = {}
    for run in raw:
        if not isinstance(run, dict):
            continue
        suite_id = run.get("check_suite_id")
        if suite_id is None:
            continue
        out[suite_id] = {
            "workflow": run.get("workflow_id", run.get("name", "")),
            "order": (
                str(run.get("created_at", "") or ""),
                _as_int(run.get("run_number")),
                _as_int(run.get("id")),
            ),
        }
    return out


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _run_suite_id(run: dict) -> Any:
    suite = run.get("check_suite")
    return suite.get("id") if isinstance(suite, dict) else None


def _run_group_key(
    run: dict, suite_map: Dict[Any, Dict[str, Any]]
) -> Tuple[Tuple[str, str], bool]:
    """Identify the check a run is an attempt of: ``((scope, name), proven)``.

    ``proven`` is True only when the run's check suite was found in the
    workflow-run listing, i.e. the scope is the real workflow the suite
    belongs to. The ``app:`` fallback is a *guess* — two different workflows
    both defining a job called ``test`` land in one group under it — so the
    flag is carried out to :func:`latest_check_runs`, which refuses to let an
    unproven collapse decide the verdict (#2939).
    """
    suite_id = _run_suite_id(run)
    known = suite_map.get(suite_id) if suite_id is not None else None
    if known is not None:
        return (f"workflow:{known['workflow']}", str(run.get("name", "") or "")), True
    app = run.get("app")
    app = app if isinstance(app, dict) else {}
    scope = f"app:{app.get('slug') or app.get('id') or ''}"
    return (scope, str(run.get("name", "") or "")), False


def _run_order_key(run: dict, suite_map: Dict[Any, Dict[str, Any]]) -> Tuple:
    """Sort two attempts of the same check oldest-first."""
    suite_id = _run_suite_id(run)
    known = suite_map.get(suite_id) if suite_id is not None else None
    run_order = known["order"] if known is not None else ("", 0, 0)
    # Check-run ids increase over time, so they break ties on identical
    # timestamps (a matrix job's runs all start in the same second).
    return (run_order, str(run.get("started_at", "") or ""), _as_int(run.get("id")))


class ResolvedRuns(NamedTuple):
    """Outcome of resolving a head SHA's check runs to one attempt per check.

    ``runs`` is the collapsed set; ``all_runs`` is every run as fetched, kept
    so a caller can check whether an *unproven* collapse changed the verdict.
    ``ambiguous`` names the checks collapsed without workflow proof.
    """

    runs: List[dict]
    superseded: int
    ambiguous: List[str]
    all_runs: List[dict]


def latest_check_runs(
    check_runs: Any, workflow_runs: Any = None
) -> ResolvedRuns:
    """Resolve a head SHA's check runs to the latest attempt of each check.

    A head SHA accumulates the check runs of *every* workflow run that ever
    targeted it, so re-running CI (or pushing while a run is in flight)
    leaves stale entries behind — commonly ``cancelled`` by concurrency,
    sometimes never leaving ``queued``. Judging the rollup on those makes it
    non-convergent (#2939).

    Collapsing is only safe when two runs are provably attempts of the *same*
    check. With the workflow-run listing that proof is the shared workflow
    id. Without it (see :meth:`CIWaitable._fetch_workflow_runs`) the grouping
    degrades to the check's app, and a group holding runs from more than one
    check suite may be two different workflows that share a job name rather
    than two attempts of one job. Those groups are still collapsed — the
    caller needs a set to judge — but their names are reported in
    ``ambiguous`` so the verdict can refuse to depend on the guess.
    """
    suite_map = _suite_workflow_map(workflow_runs)
    all_runs = _as_run_list(check_runs)
    latest: Dict[Tuple[str, str], dict] = {}
    suites_seen: Dict[Tuple[str, str], Set[Any]] = {}
    sizes: Dict[Tuple[str, str], int] = {}
    unproven: Set[Tuple[str, str]] = set()
    superseded = 0
    for run in all_runs:
        key, proven = _run_group_key(run, suite_map)
        if not proven:
            unproven.add(key)
        suites_seen.setdefault(key, set()).add(_run_suite_id(run))
        sizes[key] = sizes.get(key, 0) + 1
        current = latest.get(key)
        if current is None:
            latest[key] = run
            continue
        superseded += 1
        if _run_order_key(run, suite_map) >= _run_order_key(current, suite_map):
            latest[key] = run

    ambiguous: Set[str] = set()
    for key in unproven:
        if sizes.get(key, 0) < 2:
            continue  # a group of one collapsed nothing
        suites = suites_seen.get(key, set())
        # Runs sharing one *known* check suite are one workflow run's attempts
        # however the group was formed — the suite id is the same proof the
        # workflow listing would have supplied. Anything else is a guess.
        if len(suites) == 1 and None not in suites:
            continue
        ambiguous.add(key[1])
    return ResolvedRuns(
        list(latest.values()), superseded, sorted(ambiguous), all_runs
    )


def _tally(
    runs: List[dict], statuses: List[dict], combined_state: str
) -> Tuple[List[str], List[str]]:
    """Name the checks that are still running and the ones that failed."""
    pending: List[str] = []
    failed: List[str] = []
    for r in runs:
        name = str(r.get("name", "") or "")
        # Anything short of ``completed`` (queued/in_progress/waiting/
        # pending/requested) is still running. ``completed`` is terminal for
        # EVERY conclusion, including skipped/neutral/cancelled.
        if str(r.get("status", "") or "").lower() != "completed":
            pending.append(name)
        elif str(r.get("conclusion", "") or "").lower() in _FAIL_CONCLUSIONS:
            failed.append(name)
    for s in statuses:
        context = str(s.get("context", "") or "")
        state = str(s.get("state", "") or "").lower()
        if state == "pending":
            pending.append(context)
        elif state in ("failure", "error"):
            failed.append(context)
    if combined_state == "pending" and not pending:
        # Legacy statuses summarize to pending without any individual status
        # saying so (e.g. a required context that has not reported yet).
        pending.append("combined-status")
    if combined_state in ("failure", "error") and not failed:
        failed.append("combined-status")
    return pending, failed


def _rollup_summary(
    check_runs: Any = None,
    combined_status: Any = None,
    workflow_runs: Any = None,
) -> Dict[str, Any]:
    """Reduce a head commit's checks to a verdict plus the names behind it.

    ``verdict`` is one of:
      * ``"none"``          — no checks or statuses exist at all,
      * ``"pending"``       — at least one check/status is not yet terminal,
      * ``"failure"``       — everything terminal and at least one failed,
      * ``"success"``       — everything terminal and all passed,
      * ``"indeterminate"`` — the evidence does not support a pass (rules 4
        and 5): a listing could not be paged to completion, or an unprovable
        grouping is the only reason nothing looks outstanding.

    ``pending``/``failed`` name the checks responsible, so a verdict can be
    audited against ``gh pr view --json statusCheckRollup`` without a second
    investigation; ``unresolved`` names the reasons behind an
    ``indeterminate``.
    """
    resolved = latest_check_runs(check_runs, workflow_runs)
    runs = resolved.runs
    _, runs_complete = _list_and_completeness(check_runs, "check_runs")

    statuses, statuses_complete = _list_and_completeness(combined_status, "statuses")
    combined_state = ""
    # GitHub reports ``state: "pending"`` on a commit that carries NO legacy
    # statuses at all, which is every Actions-only repo. Consult the combined
    # state only when it is summarizing something (#2939).
    if statuses and isinstance(combined_status, dict):
        combined_state = str(combined_status.get("state", "") or "").lower()

    pending, failed = _tally(runs, statuses, combined_state)

    if not runs and not statuses:
        verdict = "none"
    elif pending:
        verdict = "pending"
    elif failed:
        verdict = "failure"
    else:
        verdict = "success"

    # Only an optimistic verdict can be wrong about evidence it does not
    # hold. A visible failure stays a failure — more pages cannot un-fail a
    # check — and a pending verdict is already non-terminal.
    unresolved: List[str] = []
    if verdict in ("success", "none"):
        if not runs_complete:
            unresolved.append("incomplete:check_runs")
        if not statuses_complete:
            unresolved.append("incomplete:statuses")
        if resolved.ambiguous:
            # The collapsed set says nothing is outstanding, but the collapse
            # of these names was a guess. Judge the same rollup uncollapsed:
            # if that surfaces a pending or failed run, the pass existed only
            # because of the guess, so it is not reportable (#2939).
            alt_pending, alt_failed = _tally(
                resolved.all_runs, statuses, combined_state
            )
            if alt_pending or alt_failed:
                unresolved.extend(f"ambiguous:{n}" for n in resolved.ambiguous)
        if unresolved:
            verdict = "indeterminate"

    return {
        "verdict": verdict,
        "pending": sorted(set(pending)),
        "failed": sorted(set(failed)),
        "total": len(runs) + len(statuses),
        "superseded": resolved.superseded,
        "unresolved": unresolved,
    }


def _check_verdict(
    check_runs: Any = None,
    combined_status: Any = None,
    workflow_runs: Any = None,
) -> str:
    """Coarse verdict for a head commit's checks. See :func:`_rollup_summary`."""
    return str(
        _rollup_summary(check_runs, combined_status, workflow_runs)["verdict"]
    )


def _parse_github_time(value: Any) -> Optional[datetime]:
    """Parse a GitHub ISO-8601 timestamp; ``None`` when unusable."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _empty_rollup_still_settling(
    pr_raw: Dict[str, Any], now: Optional[datetime] = None
) -> bool:
    """Whether an empty rollup may still be CI that has not registered yet.

    True only inside :data:`_EMPTY_ROLLUP_GRACE_SECONDS` of the PR's last
    change (a push registers its check runs within seconds). A payload with
    no usable timestamp is treated as settled — an empty rollup that can
    never be shown to be in flight must not wait forever (#2939).
    """
    changed_at = _parse_github_time(pr_raw.get("updated_at")) or _parse_github_time(
        pr_raw.get("created_at")
    )
    if changed_at is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - changed_at).total_seconds() < _EMPTY_ROLLUP_GRACE_SECONDS


def classify_ci_state(
    pr_raw: Dict[str, Any],
    *,
    check_runs: Any = None,
    combined_status: Any = None,
    workflow_runs: Any = None,
    repo: str = "",
    number: Optional[int] = None,
    now: Optional[datetime] = None,
) -> WaitStatus:
    """Classify a PR's merge/CI state onto the generic :class:`Outcome`.

    Pure and side-effect free (apart from the contradiction/indeterminate
    warnings) so the terminal contract is unit-testable with fabricated
    payloads (no network).
    ``pr_raw`` is the GitHub pull payload; ``check_runs``/``combined_status``
    are the head commit's check-runs and combined status JSON, and
    ``workflow_runs`` the ``/actions/runs?head_sha=`` listing used to drop
    superseded runs. ``now`` overrides the clock for the empty-rollup grace.
    """
    state = str(pr_raw.get("state", "") or "").strip().lower()
    merged = bool(pr_raw.get("merged", False))
    rollup = _rollup_summary(check_runs, combined_status, workflow_runs)
    verdict = str(rollup["verdict"])
    mergeable = pr_raw.get("mergeable")
    mergeable_state = str(pr_raw.get("mergeable_state", "") or "").strip().lower()
    data: Dict[str, Any] = {
        "repo": repo,
        "number": number,
        "state": state,
        "merged": merged,
        "checks": verdict,
        "checks_total": rollup["total"],
    }
    if rollup["pending"]:
        data["checks_pending"] = rollup["pending"]
    if rollup["failed"]:
        data["checks_failed"] = rollup["failed"]
    if rollup["superseded"]:
        data["checks_superseded"] = rollup["superseded"]
    if rollup["unresolved"]:
        data["checks_unresolved"] = rollup["unresolved"]
    if mergeable is not None:
        data["mergeable"] = mergeable
    if mergeable_state:
        data["mergeable_state"] = mergeable_state
    label = f"{repo}#{number}" if repo else "PR"

    if merged:
        return WaitStatus(Outcome.DONE, f"{label} merged", data=data)
    if state == "closed":
        return WaitStatus(
            Outcome.FAILED, f"{label} closed without merge", data=data
        )
    if verdict == "failure":
        return WaitStatus(
            Outcome.FAILED,
            f"{label} CI checks failed: {', '.join(rollup['failed'])}",
            data=data,
        )
    if verdict == "indeterminate":
        # The rollup looks clean, but only because of evidence the provider
        # does not hold (a truncated listing) or a grouping it cannot justify
        # (rules 4/5). Reporting DONE here is the false-terminal polarity —
        # a merge decision made on checks nobody read (#2939).
        logger.warning(
            "ci wait %s: check rollup indeterminate (%s) — reporting pending "
            "rather than a pass the evidence does not support",
            label,
            ", ".join(rollup["unresolved"]),
        )
        return WaitStatus(
            Outcome.PENDING,
            f"{label} checks indeterminate: {', '.join(rollup['unresolved'])}",
            data=data,
        )
    if verdict == "success":
        return WaitStatus(Outcome.DONE, f"{label} CI checks passed", data=data)
    if verdict == "none":
        if _empty_rollup_still_settling(pr_raw, now):
            # Bounded: the push is recent enough that CI may not have
            # registered its check runs yet.
            data["awaiting_ci_registration"] = True
            return WaitStatus(
                Outcome.PENDING,
                f"{label} open, no checks reported yet (CI may still register)",
                data=data,
            )
        # Nothing ran and nothing is going to — terminal, but reported as
        # "none" rather than "success" so a caller can tell "everything
        # passed" from "nothing ran" (#2939).
        return WaitStatus(
            Outcome.DONE,
            f"{label} open, no checks ran (nothing to wait for)",
            data=data,
        )

    # open + checks genuinely still running — keep the wait armed.
    if mergeable_state == "clean" and mergeable is not False:
        # Two reads disagree: GitHub calls the PR mergeable/clean while the
        # rollup says a check is outstanding. Surface it instead of resolving
        # it — ``clean`` is also what GitHub reports when a repo has no
        # *required* checks and CI is merely queued, so it can never be
        # promoted to a terminal DONE (#2939).
        data["contradiction"] = "clean_but_pending"
        logger.warning(
            "ci wait %s: mergeable_state=clean but checks still pending (%s) "
            "— reporting pending; reconcile against the check rollup",
            label,
            ", ".join(rollup["pending"]) or "unknown",
        )
    return WaitStatus(
        Outcome.PENDING,
        f"{label} open, checks {verdict}"
        + (f": {', '.join(rollup['pending'])}" if rollup["pending"] else ""),
        data=data,
    )


async def _fetch_all_pages(
    url: str,
    *,
    key: str,
    token: str,
    ref: str,
    timeout: int = 10,
    tolerate_partial: bool = False,
) -> Dict[str, Any]:
    """GET every page of a GitHub list endpoint and merge them into one payload.

    GitHub caps ``per_page`` at 100, so one request is not a rollup — it is
    the first hundred rows of one. Pages are followed until a short page
    arrives (the end of the listing), ``total_count`` is satisfied, or
    :data:`_MAX_PAGES` is reached.

    The result keeps page 1's other fields (the combined status endpoint's
    ``state`` lives beside its ``statuses`` list) with ``key`` replaced by the
    concatenation of every page. ``_truncated`` is *always* set — True only
    when the pages in hand are known not to be the whole list — so
    :func:`_list_and_completeness` can rule on the evidence this function
    actually gathered rather than re-deriving it from a count.

    ``tolerate_partial`` keeps whatever pages were already fetched when a
    later page fails, for a listing that is enrichment rather than evidence;
    a first-page failure always raises so the caller can report ``blocked``.
    """
    from kestrel_sovereign.signals.sources.github_pr_watch import (
        PRWatchError,
        _github_get,
    )

    sep = "&" if "?" in url else "?"
    items: List[dict] = []
    envelope: Dict[str, Any] = {}
    total: Optional[int] = None
    truncated = False
    page = 0
    while page < _MAX_PAGES:
        page += 1
        try:
            payload = await _github_get(
                f"{url}{sep}per_page={_PER_PAGE}&page={page}",
                token=token, timeout=timeout, ref=f"{ref} page {page}",
            )
        except PRWatchError:
            if not tolerate_partial or page == 1:
                raise
            logger.debug(
                "ci wait %s: page %d unavailable; keeping %d partial rows",
                ref, page, len(items),
            )
            truncated = True
            break
        if isinstance(payload, dict):
            if page == 1:
                envelope = {k: v for k, v in payload.items() if k != key}
            batch = [x for x in (payload.get(key) or []) if isinstance(x, dict)]
            count = payload.get("total_count")
            if isinstance(count, int) and not isinstance(count, bool):
                total = count
        elif isinstance(payload, list):
            batch = [x for x in payload if isinstance(x, dict)]
        else:
            # Not a listing shape — nothing further to page.
            break
        items.extend(batch)
        if len(batch) < _PER_PAGE:
            break
        if total is not None and len(items) >= total:
            break
    else:
        # Loop ran out of pages while every page was still full.
        truncated = True
        logger.warning(
            "ci wait %s: stopped after %d pages (%d rows); rollup evidence "
            "is incomplete",
            ref, _MAX_PAGES, len(items),
        )

    envelope[key] = items
    if total is not None:
        envelope["total_count"] = total
        if not truncated and len(items) < total:
            # Pagination reached the end of the listing, yet the endpoint's
            # own count claims more rows exist. GitHub counts and serves
            # different sets here (check-runs pages apply ``filter=latest``),
            # so the pages win — refusing to converge on a count nobody can
            # page to would be the #2939 stall again. Surfaced, not silent.
            logger.warning(
                "ci wait %s: listing ended at %d rows but total_count=%d; "
                "treating the paged rows as the complete set",
                ref, len(items), total,
            )
    envelope["_truncated"] = truncated
    return envelope


class CIWaitable:
    """Polls a GitHub PR's merge/CI-check state by ``owner/repo#<number>``."""

    kind: ClassVar[str] = "ci"
    signal: ClassVar[Optional[str]] = None

    def __init__(self, feature: "object") -> None:
        # The owning feature (SchedulerFeature); only used to stay symmetric
        # with the other providers — the fetch is self-contained.
        self._feature = feature

    async def owns_handle(self, handle: str) -> Optional[bool]:
        """Whether ``handle`` is a syntactically valid CI (PR) reference.

        The cheap, network-free ownership check used at watch registration
        (#2729): ``False`` for anything that is not ``owner/repo#<number>``
        (e.g. a bare A2A task id), so ``ci:<foreign-id>`` is rejected up
        front. A well-formed reference returns ``None`` (unverifiable without
        a network round-trip → caller fails open and allows the watch).
        """
        try:
            parse_ci_handle(handle)
        except ValueError:
            return False
        return None

    async def _fetch(
        self, repo: str, number: int, token: str
    ) -> Tuple[Dict[str, Any], Any, Any, Any]:
        """Fetch the PR payload + head-commit checks. Split out for tests."""
        from kestrel_sovereign.signals.sources.github_pr_watch import _github_get

        base = f"https://api.github.com/repos/{repo}"
        ref = f"{repo}#{number}"
        pr_raw = await _github_get(
            f"{base}/pulls/{number}", token=token, timeout=10, ref=ref
        )
        if not isinstance(pr_raw, dict):
            from kestrel_sovereign.signals.sources.github_pr_watch import (
                PRWatchNetworkError,
            )
            raise PRWatchNetworkError(
                f"GitHub returned a non-object payload for {ref}"
            )
        head = pr_raw.get("head")
        head_sha = head.get("sha") if isinstance(head, dict) else None
        check_runs: Any = None
        combined_status: Any = None
        workflow_runs: Any = None
        if head_sha:
            # Both listings are paged to exhaustion: page 1 alone would hide
            # whichever checks did not fit, and the classifier would read the
            # remainder as a terminal pass (#2939).
            check_runs = await _fetch_all_pages(
                f"{base}/commits/{head_sha}/check-runs",
                key="check_runs", token=token, ref=f"{ref} check-runs",
            )
            combined_status = await _fetch_all_pages(
                f"{base}/commits/{head_sha}/status",
                key="statuses", token=token, ref=f"{ref} status",
            )
            workflow_runs = await self._fetch_workflow_runs(
                base, head_sha, token, ref
            )
        return pr_raw, check_runs, combined_status, workflow_runs

    async def _fetch_workflow_runs(
        self, base: str, head_sha: str, token: str, ref: str
    ) -> Any:
        """List the head SHA's workflow runs, used to drop superseded checks.

        Enrichment, not evidence: a failure degrades the latest-run-per-name
        resolution to grouping by check app instead of blocking the poll. The
        degraded grouping cannot be trusted to decide a pass, though, so a
        partial listing is returned marked truncated rather than discarded —
        every suite it *did* identify is one fewer ambiguous group
        (:func:`latest_check_runs`).
        """
        from kestrel_sovereign.signals.sources.github_pr_watch import PRWatchError

        try:
            return await _fetch_all_pages(
                f"{base}/actions/runs?head_sha={head_sha}",
                key="workflow_runs", token=token, ref=f"{ref} workflow-runs",
                tolerate_partial=True,
            )
        except PRWatchError as exc:
            logger.debug(
                "ci wait %s: workflow-run listing unavailable (%s); "
                "grouping superseded checks by app instead",
                ref, exc,
            )
            return None

    async def poll(self, handle: str) -> WaitStatus:
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            PRWatchAuthError,
            PRWatchNetworkError,
        )

        try:
            repo, number = parse_ci_handle(handle)
        except ValueError as exc:
            # A malformed handle only reaches poll if it slipped past
            # registration validation; report it as terminal FAILED because
            # it can never resolve.
            return WaitStatus(
                Outcome.FAILED, str(exc), data={"handle": handle}
            )

        from kestrel_sovereign.features.strategic_memory.github_integration import (
            get_github_token,
        )

        token = get_github_token()
        if not token:
            # No credential — cannot observe, but NOT terminal. Keep the
            # watch armed so it completes once a token is available.
            return WaitStatus(
                Outcome.PENDING,
                f"{repo}#{number}: blocked (no GITHUB_TOKEN)",
                data={"repo": repo, "number": number, "blocked": "auth"},
            )

        try:
            pr_raw, check_runs, combined_status, workflow_runs = await self._fetch(
                repo, number, token
            )
        except (PRWatchAuthError, PRWatchNetworkError) as exc:
            # Auth/network blip is transient — stay pending, never a false
            # merge/close terminal.
            blocked = "auth" if isinstance(exc, PRWatchAuthError) else "network"
            return WaitStatus(
                Outcome.PENDING,
                f"{repo}#{number}: blocked ({blocked}): {exc}",
                data={"repo": repo, "number": number, "blocked": blocked},
            )
        except Exception as exc:  # defensive — provider transport boundary
            return WaitStatus(
                Outcome.PENDING,
                f"{repo}#{number}: poll error: {exc}",
                data={"repo": repo, "number": number},
            )

        return classify_ci_state(
            pr_raw,
            check_runs=check_runs,
            combined_status=combined_status,
            workflow_runs=workflow_runs,
            repo=repo,
            number=number,
        )
