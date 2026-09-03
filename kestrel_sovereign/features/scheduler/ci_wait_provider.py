"""Waitable provider for GitHub PR merge/CI-check waits (``ci:<repo>#<n>``).

The unified-wait epic (#1860) documented a ``ci:`` kind in the ``wait`` tool
but never registered a provider, so a merge/check wait could not survive a
restart (#2729). This module is that provider. A handle is a PR reference —
``owner/repo#123`` — and the provider reads the PR's current state plus its
head-commit check runs and combined status to decide a terminal verdict:

  * PR merged                       -> DONE    (the happy terminal)
  * PR closed without merge         -> FAILED
  * open + all checks passed        -> DONE
  * open + a check failed           -> FAILED
  * open + no checks ran at all     -> PARTIAL (terminal; ``checks: "none"``)
  * open + checks still running,
    or the rollup was not read      -> PENDING (keep watching)

The change-detection primitives (``fetch``/``summarize_checks``) are reused
from :mod:`kestrel_sovereign.signals.sources.github_pr_watch`, which is pure
core — this provider does NOT depend on the out-of-tree GitHub feature.

Transient failures (no token, auth error, network blip) return
:class:`Outcome.PENDING`, never a terminal failure: a durable
``wait("ci:...", mode="signal")`` must re-arm and complete once when the PR
truly settles, not fabricate a merge/close from a flaky poll.

Every observed state must be either terminal or provably still-progressing
(#2939). A state that is neither is how this provider stalled a merge on
``kestrel-sovereign#2934`` for three hours: it reported ``checks: "pending"``
against a head SHA whose 18 check runs had *all* completed, because GitHub's
combined-status endpoint reports ``state: "pending"`` for a commit carrying
**zero** legacy statuses — the shape of every Actions-only repository. Two
rules follow, and both are load-bearing:

  * an absence of evidence is never read as "still running"
    (:func:`_check_verdict` ignores a combined state with no statuses), and
  * an empty rollup is terminal-but-caveated, not pending — there is nothing
    to wait for, and a ``mode="signal"`` watch on it would otherwise never
    fire.

The empty rollup is terminal *immediately*, with no grace period for check
runs GitHub may still be creating. A settle window was tried and removed: a
single poll carries no immutable anchor to bound one from. The PR's
``updated_at`` is the only timestamp on the payload, and it is mutable — a
comment, a label, or any automation touching the PR refreshes it while the
same head SHA stays checkless, so every minute-level reconciliation would
re-enter the window and the wait would never converge. That is the #2939
stall in a different shape. Reporting "nothing ran" a few seconds early is a
caveated PARTIAL the waiter can see and act on; a wait that never fires is
not.

The converse rule holds too: a *pending* read is never promoted to a terminal
verdict. A PR that GitHub calls ``clean``/mergeable while the rollup still
reads pending is surfaced as a ``contradiction`` marker on the PENDING
payload, because a repository with no required checks is reported ``clean``
while its CI is still queued — resolving that to DONE would fabricate a merge
signal out of a race.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar, Dict, Optional, Tuple

from kestrel_sdk.tools import Outcome, WaitStatus
from kestrel_sovereign.signals.sources.github_pr_watch import _check_verdict

logger = logging.getLogger(__name__)

# A CI handle is a PR reference: ``owner/repo#123``. The owner/repo half may
# contain the usual GitHub name characters; the number is the PR/issue id.
_CI_HANDLE_RE = re.compile(r"^(?P<repo>[^\s#]+/[^\s#]+)#(?P<number>\d+)$")

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


def _mergeability(pr_raw: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """Read ``(mergeable, mergeable_state)`` off a GitHub pull payload.

    Both fields are already on the fetched pull payload, so this costs no
    extra request. ``mergeable`` is ``None`` while GitHub is still computing
    the merge commit — deliberately preserved as ``None`` rather than
    collapsed to ``False``, since "not yet known" and "not mergeable" are
    different claims.
    """
    mergeable = pr_raw.get("mergeable")
    if not isinstance(mergeable, bool):
        mergeable = None
    return mergeable, str(pr_raw.get("mergeable_state", "") or "").strip().lower()


def classify_ci_state(
    pr_raw: Dict[str, Any],
    *,
    check_runs: Any = None,
    combined_status: Any = None,
    repo: str = "",
    number: Optional[int] = None,
) -> WaitStatus:
    """Classify a PR's merge/CI state onto the generic :class:`Outcome`.

    Pure and side-effect free (bar one diagnostic log line) so the terminal
    contract is unit-testable with fabricated payloads (no network).
    ``pr_raw`` is the GitHub pull payload; ``check_runs``/``combined_status``
    are the head commit's check-runs and combined status JSON. Passing
    *neither* means the rollup was never read — an evidence gap that stays
    PENDING — which is distinct from reading it and finding it empty.
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
    if verdict == "none":
        # Read the rollup and it is empty: no CI is configured for this head
        # SHA, or no workflow matched its paths. Terminal — there is nothing
        # to wait for, and staying PENDING would stall a mode="signal" watch
        # forever (#2939). PARTIAL, not DONE: "nothing ran" must not read as
        # "everything passed", so the caveat rides all the way out to the
        # waiter via ToolResult.partial. Terminal on the FIRST such read: the
        # payload carries no immutable anchor (``updated_at`` is refreshed by
        # any unrelated PR edit) to bound a settle window against, so a grace
        # period here would re-arm indefinitely on a checkless head SHA.
        data["caveat"] = f"no checks ran on {label} head commit"
        return WaitStatus(
            Outcome.PARTIAL, f"{label} open, no checks ran", data=data
        )

    # open + checks pending, or the rollup was never read — keep the wait
    # armed. A pending read is NEVER promoted to a terminal verdict here.
    mergeable, mergeable_state = _mergeability(pr_raw)
    if mergeable_state:
        data["mergeable_state"] = mergeable_state
    if mergeable is not None:
        data["mergeable"] = mergeable
    if verdict == "pending" and mergeable_state == "clean" and mergeable is not False:
        # GitHub says the PR is clean/mergeable while our rollup still reads
        # pending. Surface the contradiction rather than resolving it: a repo
        # with no *required* checks is reported "clean" while CI is still
        # queued, so trusting it would fabricate a terminal from a race. The
        # marker exists because the #2939 stall survived three hours with no
        # way to see why the provider was saying pending.
        data["contradiction"] = "clean_but_pending"
        logger.warning(
            "ci wait %s: GitHub reports mergeable_state=clean but the check "
            "rollup reads pending — reporting pending (contradiction: "
            "clean_but_pending)",
            label,
        )
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
                f"{base}/commits/{head_sha}/check-runs?per_page=100",
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
