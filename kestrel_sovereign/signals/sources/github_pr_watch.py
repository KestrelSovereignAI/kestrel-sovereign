"""Signal source for GitHub PR/issue activity (#1618).

Talon's ``talon.job_complete`` source wakes the agent when a *Talon
background job* finishes. It does NOT cover PR-only activity that lands
after the Talon job has exited — a reviewer leaving a comment, CI turning
red, the PR getting merged or closed. Before this source the only way to
notice that was a manual ``talon_status``/``gh pr view`` poll.

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
from typing import Any, Dict, Iterable, Optional, Set, Tuple

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


class PRWatchError(Exception):
    """Base class for github_pr_watch fetch failures."""


class PRWatchAuthError(PRWatchError):
    """Auth/permission failure (401/403) — distinct from a no-change poll."""


class PRWatchNetworkError(PRWatchError):
    """Network/transport failure (timeout, DNS, 5xx) — distinct from no-change."""


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
        "checks_status": str(raw.get("checks_status", "") or ""),
        "mergeable_state": str(raw.get("mergeable_state", "") or ""),
    }


def compute_fingerprint(normalized: Dict[str, Any]) -> str:
    """Stable SHA-256 over the normalized watched fields."""
    blob = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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


async def fetch_pr_state(
    repo: str, number: int, *, token: str, kind: str = "pr", timeout: int = 10
) -> Dict[str, Any]:
    """Fetch a PR's or issue's current state from the GitHub API.

    ``kind="pr"`` queries ``/pulls/{number}`` (the default); ``kind="issue"``
    queries ``/issues/{number}``. PRs and issues share one numbering space,
    so an issue number sent to ``/pulls`` would 404 — the endpoint must match
    the watch type. Issue payloads have no ``head``/``merged``/
    ``mergeable_state``/``checks_status``; :func:`normalize_pr_state` already
    tolerates the missing fields.

    Raises :class:`PRWatchAuthError` on 401/403 and
    :class:`PRWatchNetworkError` on any other transport/HTTP failure so the
    caller can report ``blocked: auth`` / ``blocked: network`` distinctly
    from a no-change poll. Patched out in tests.
    """
    endpoint = "issues" if kind == "issue" else "pulls"
    url = f"https://api.github.com/repos/{repo}/{endpoint}/{number}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "kestrel-agent",
        },
    )

    def _do() -> bytes:
        return urllib.request.urlopen(req, timeout=timeout).read()

    try:
        resp = await asyncio.to_thread(_do)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise PRWatchAuthError(f"GitHub returned {e.code} for {repo}#{number}") from e
        raise PRWatchNetworkError(
            f"GitHub HTTP {e.code} for {repo}#{number}"
        ) from e
    except urllib.error.URLError as e:
        raise PRWatchNetworkError(f"network error for {repo}#{number}: {e}") from e
    except Exception as e:  # pragma: no cover - defensive
        raise PRWatchNetworkError(f"unexpected error for {repo}#{number}: {e}") from e

    try:
        return json.loads(resp)
    except (ValueError, TypeError) as e:
        raise PRWatchNetworkError(
            f"could not parse GitHub response for {repo}#{number}: {e}"
        ) from e


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
