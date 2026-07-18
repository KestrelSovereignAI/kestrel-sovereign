"""Pure-core GitHub issue write operations for the Talon bounded-job surface.

Orchestration (the Talon coordinator) needs a controlled, auditable path to
close / comment on / label a GitHub issue after work completes (#2581). The
write *capability* already lives in ``kestrel-feature-github``; this module is
the narrow, dependency-free seam the coordinator's ``talon_github_write``
bounded job builds on so that:

* the GitHub token stays in-process for a single authenticated REST call and is
  never handed to a shell or the read-only git/verify surface, and
* the write *target* is allowlist-scoped to the agent's own repo plus any
  configured fleet repos (mirrors the ``kestrel-feature-github`` F348 gate).

Everything here is PURE: given an operation and its parameters it produces one
or more :class:`GithubWriteRequest` (method / url / json payload / accepted
status codes) or raises :class:`GithubWriteError`. The actual HTTP call is done
by the coordinator, keeping this module trivially unit-testable — the same
pure-core split used by ``signals/sources/github_pr_watch.py``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional
from urllib.parse import quote

GITHUB_API_ROOT = "https://api.github.com"

# The operations the bounded job exposes. This is the stable public vocabulary
# orchestration calls; the issue's ``issue.close`` maps to ``close_issue`` etc.
WRITE_OPERATIONS = (
    "close_issue",
    "reopen_issue",
    "comment",
    "add_labels",
    "remove_labels",
    "update_issue",
)

# Accepted GitHub ``state_reason`` values for closing an issue.
CLOSE_STATE_REASONS = ("completed", "not_planned")

# ``owner/name`` — GitHub restricts both segments to these characters.
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GithubWriteError(ValueError):
    """Raised when a write request is malformed or the target is not allowed."""


@dataclass(frozen=True)
class GithubWriteRequest:
    """One launch-ready GitHub REST mutation.

    ``success_statuses`` lets each operation declare what "worked" means —
    e.g. a comment POST returns ``201`` and a label DELETE treats ``404``
    (label already absent) as an idempotent success.
    """

    method: str
    url: str
    payload: Optional[dict] = None
    summary: str = ""
    success_statuses: tuple[int, ...] = (200,)


def default_self_repo(env: Optional[Mapping[str, str]] = None) -> str:
    """The agent's own repo — the primary write target."""
    source = env if env is not None else os.environ
    return source.get("GITHUB_SELF_REPO", "KestrelSovereignAI/kestrel-sovereign")


def configured_fleet_repos(
    env: Optional[Mapping[str, str]] = None,
) -> tuple[str, ...]:
    """Extra repos the agent may write to, from ``GITHUB_FLEET_REPOS`` (CSV)."""
    source = env if env is not None else os.environ
    raw = source.get("GITHUB_FLEET_REPOS", "") or ""
    out: List[str] = []
    seen = set()
    for item in raw.split(","):
        name = item.strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(name)
    return tuple(out)


def write_allowlist(env: Optional[Mapping[str, str]] = None) -> tuple[str, ...]:
    """Repos this agent may WRITE to: its own repo + configured fleet."""
    allow: List[str] = [default_self_repo(env)]
    for repo in configured_fleet_repos(env):
        if repo.lower() not in {r.lower() for r in allow}:
            allow.append(repo)
    return tuple(allow)


def resolve_write_repo(
    repo: str, env: Optional[Mapping[str, str]] = None
) -> str:
    """Canonicalize and AUTHORIZE a write target against the allowlist.

    Matching is case-insensitive and returns the canonical casing from the
    allowlist. Raises :class:`GithubWriteError` when ``repo`` is neither the
    agent's own repo nor a ``GITHUB_FLEET_REPOS`` entry, so a bounded write
    never touches a repo the operator did not opt into.
    """
    candidate = (repo or "").strip()
    if not _REPO_RE.match(candidate):
        raise GithubWriteError(
            f"repo must be in 'owner/name' form, got {repo!r}"
        )
    allow = write_allowlist(env)
    for allowed in allow:
        if candidate.lower() == allowed.lower():
            return allowed
    raise GithubWriteError(
        f"Refusing to write to {candidate!r}: not the agent's own repo and "
        f"not in GITHUB_FLEET_REPOS. Allowed: {', '.join(allow)}"
    )


def parse_issue_number(issue: Any) -> int:
    """Accept ``123``, ``"123"``, or ``"#123"``; require a positive integer."""
    if isinstance(issue, bool):  # bool is an int subclass — reject explicitly
        raise GithubWriteError(f"issue must be a positive integer, got {issue!r}")
    if isinstance(issue, int):
        number = issue
    else:
        text = str(issue).strip().lstrip("#").strip()
        try:
            number = int(text)
        except (TypeError, ValueError):
            raise GithubWriteError(
                f"issue must be a positive integer, got {issue!r}"
            )
    if number < 1:
        raise GithubWriteError(f"issue must be a positive integer, got {issue!r}")
    return number


def normalize_labels(labels: Any) -> List[str]:
    """Parse a CSV string or an iterable into a deduped, ordered label list."""
    if labels is None:
        return []
    if isinstance(labels, str):
        raw: Iterable[Any] = labels.split(",")
    elif isinstance(labels, (list, tuple, set)):
        raw = labels
    else:
        raise GithubWriteError(
            "labels must be a comma-separated string or a list of names"
        )
    out: List[str] = []
    seen = set()
    for item in raw:
        name = str(item).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(name)
    return out


def build_github_write_requests(
    operation: str,
    repo: str,
    issue: Any,
    *,
    body: Optional[str] = None,
    labels: Any = None,
    title: Optional[str] = None,
    state_reason: Optional[str] = None,
) -> List[GithubWriteRequest]:
    """Translate a bounded-job write into concrete GitHub REST request(s).

    ``repo`` must already be a concrete ``owner/name`` (resolve ``"self"`` and
    authorize via :func:`resolve_write_repo` first). Raises
    :class:`GithubWriteError` on an unknown operation or invalid parameters —
    nothing is executed here.
    """
    op = (operation or "").strip().lower()
    if op not in WRITE_OPERATIONS:
        raise GithubWriteError(
            f"Unknown github write operation {operation!r}. "
            f"Valid operations: {', '.join(WRITE_OPERATIONS)}"
        )
    number = parse_issue_number(issue)
    issue_url = f"{GITHUB_API_ROOT}/repos/{repo}/issues/{number}"

    if op == "close_issue":
        reason = (state_reason or "completed").strip().lower()
        if reason not in CLOSE_STATE_REASONS:
            raise GithubWriteError(
                "close_issue state_reason must be one of "
                f"{', '.join(CLOSE_STATE_REASONS)} (got {state_reason!r})"
            )
        return [
            GithubWriteRequest(
                method="PATCH",
                url=issue_url,
                payload={"state": "closed", "state_reason": reason},
                summary=f"close {repo}#{number} ({reason})",
            )
        ]

    if op == "reopen_issue":
        return [
            GithubWriteRequest(
                method="PATCH",
                url=issue_url,
                payload={"state": "open", "state_reason": "reopened"},
                summary=f"reopen {repo}#{number}",
            )
        ]

    if op == "comment":
        text = (body or "").strip()
        if not text:
            raise GithubWriteError("comment requires a non-empty body")
        return [
            GithubWriteRequest(
                method="POST",
                url=f"{issue_url}/comments",
                payload={"body": body},
                summary=f"comment on {repo}#{number}",
                success_statuses=(201,),
            )
        ]

    if op == "add_labels":
        names = normalize_labels(labels)
        if not names:
            raise GithubWriteError("add_labels requires at least one label")
        return [
            GithubWriteRequest(
                method="POST",
                url=f"{issue_url}/labels",
                payload={"labels": names},
                summary=f"add label(s) {names} to {repo}#{number}",
            )
        ]

    if op == "remove_labels":
        names = normalize_labels(labels)
        if not names:
            raise GithubWriteError("remove_labels requires at least one label")
        # One DELETE per label; 404 (label already absent) is treated as an
        # idempotent success so removing a label twice never hard-fails.
        return [
            GithubWriteRequest(
                method="DELETE",
                url=f"{issue_url}/labels/{quote(name, safe='')}",
                summary=f"remove label {name!r} from {repo}#{number}",
                success_statuses=(200, 404),
            )
            for name in names
        ]

    # op == "update_issue"
    payload: dict = {}
    if title is not None and title.strip():
        payload["title"] = title
    if body is not None:
        payload["body"] = body
    if not payload:
        raise GithubWriteError(
            "update_issue requires a non-empty title and/or a body"
        )
    return [
        GithubWriteRequest(
            method="PATCH",
            url=issue_url,
            payload=payload,
            summary=f"update {repo}#{number} ({'/'.join(payload)})",
        )
    ]


def extract_error_message(body: Optional[str]) -> str:
    """Pull GitHub's ``{"message": ...}`` out of an error response body."""
    if not body:
        return ""
    try:
        doc = json.loads(body)
    except (ValueError, TypeError):
        return str(body)[:300]
    if isinstance(doc, dict):
        message = doc.get("message")
        errors = doc.get("errors")
        if message and errors:
            return f"{message}: {errors}"
        if message:
            return str(message)
    return str(body)[:300]
