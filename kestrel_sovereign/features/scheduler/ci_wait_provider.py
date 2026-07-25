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
  * open + checks still running,
    or no CI configured yet         -> PENDING (keep watching)

The change-detection primitives (``fetch``/``summarize_checks``) are reused
from :mod:`kestrel_sovereign.signals.sources.github_pr_watch`, which is pure
core — this provider does NOT depend on the out-of-tree GitHub feature.

Transient failures (no token, auth error, network blip) return
:class:`Outcome.PENDING`, never a terminal failure: a durable
``wait("ci:...", mode="signal")`` must re-arm and complete once when the PR
truly settles, not fabricate a merge/close from a flaky poll.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from kestrel_sdk.tools import Outcome, WaitStatus

# A CI handle is a PR reference: ``owner/repo#123``. The owner/repo half may
# contain the usual GitHub name characters; the number is the PR/issue id.
_CI_HANDLE_RE = re.compile(r"^(?P<repo>[^\s#]+/[^\s#]+)#(?P<number>\d+)$")

# GitHub check-run conclusions that mean the check did NOT pass. ``success``,
# ``neutral`` and ``skipped`` are treated as non-blocking passes.
_FAIL_CONCLUSIONS = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "stale",
     "startup_failure"}
)


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


def _check_verdict(
    check_runs: Any = None, combined_status: Any = None
) -> str:
    """Reduce raw check-runs + combined commit status to a coarse verdict.

    Returns one of:
      * ``"none"``    — no checks or statuses exist at all (no CI configured),
      * ``"pending"`` — at least one check/status is not yet terminal,
      * ``"failure"`` — everything terminal and at least one failed,
      * ``"success"`` — everything terminal and all passed.
    """
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

    combined_state = ""
    statuses: List[dict] = []
    if isinstance(combined_status, dict):
        combined_state = str(combined_status.get("state", "") or "").lower()
        for s in combined_status.get("statuses", []) or []:
            if isinstance(s, dict):
                statuses.append(s)

    if not runs and not statuses and not combined_state:
        return "none"

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

    return "success"


def classify_ci_state(
    pr_raw: Dict[str, Any],
    *,
    check_runs: Any = None,
    combined_status: Any = None,
    repo: str = "",
    number: Optional[int] = None,
) -> WaitStatus:
    """Classify a PR's merge/CI state onto the generic :class:`Outcome`.

    Pure and side-effect free so the terminal contract is unit-testable with
    fabricated payloads (no network). ``pr_raw`` is the GitHub pull payload;
    ``check_runs``/``combined_status`` are the head commit's check-runs and
    combined status JSON (both optional — an open PR with neither stays
    PENDING).
    """
    state = str(pr_raw.get("state", "") or "").strip().lower()
    merged = bool(pr_raw.get("merged", False))
    verdict = _check_verdict(check_runs, combined_status)
    data: Dict[str, Any] = {
        "repo": repo,
        "number": number,
        "state": state,
        "merged": merged,
        "checks": verdict,
    }
    label = f"{repo}#{number}" if repo else "PR"

    if merged:
        return WaitStatus(Outcome.DONE, f"{label} merged", data=data)
    if state == "closed":
        return WaitStatus(
            Outcome.FAILED, f"{label} closed without merge", data=data
        )
    if verdict == "failure":
        return WaitStatus(Outcome.FAILED, f"{label} CI checks failed", data=data)
    if verdict == "success":
        return WaitStatus(Outcome.DONE, f"{label} CI checks passed", data=data)
    # open + checks pending / no CI yet — keep the wait armed.
    return WaitStatus(
        Outcome.PENDING,
        f"{label} open, checks {verdict}",
        data=data,
    )


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
    ) -> Tuple[Dict[str, Any], Any, Any]:
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
        if head_sha:
            check_runs = await _github_get(
                f"{base}/commits/{head_sha}/check-runs",
                token=token, timeout=10, ref=f"{ref} check-runs",
            )
            combined_status = await _github_get(
                f"{base}/commits/{head_sha}/status",
                token=token, timeout=10, ref=f"{ref} status",
            )
        return pr_raw, check_runs, combined_status

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
            pr_raw, check_runs, combined_status = await self._fetch(
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
            repo=repo,
            number=number,
        )
