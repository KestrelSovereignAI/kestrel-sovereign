"""Reconcile ledger blockers against live GitHub issue state.

57 of Emma's 110 blocker rows referenced issues GitHub had already closed
(#2954). Nothing was wrong with the rows when they were written — there was
simply no path from "the issue closed" back to "the blocker is stale", so the
list only ever grew and the agent reasoned over a backlog that had partly
ceased to exist.

This module supplies that path. It reads GitHub and reports; applying the
result is a separate, explicit decision by the caller, because closing a
GitHub issue is not by itself proof that the strategic blocker it stood for is
gone.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .github_integration import get_github_token, github_api_get
from .ledger import active_blockers

logger = logging.getLogger(__name__)

#: Returned as ``reason`` when a row simply cannot be checked. Kept distinct
#: from "open" so a lookup failure is never reported as a live blocker
#: confirmed still blocking.
UNRESOLVABLE = "unresolvable"

#: Returned as ``reason`` when the row names an issue number but not which
#: repository it lives in, and more than one is configured. Distinct from
#: ``UNRESOLVABLE`` because the row is perfectly well-formed -- what is missing
#: is the identity of the project, and no amount of retrying supplies it.
AMBIGUOUS_REPO = "ambiguous_repository"


def _issue_number(row: Dict[str, Any]) -> Optional[int]:
    raw = str(row.get("issue") or "").strip().lstrip("#")
    # Rows written by hand sometimes carry "owner/repo#123".
    if "#" in raw:
        raw = raw.rsplit("#", 1)[-1]
    if not raw.isdigit():
        return None
    return int(raw)


def configured_repos(strategy_data: Dict[str, Any]) -> List[str]:
    """The repositories STRATEGY.yaml says this agent scans."""
    if not isinstance(strategy_data, dict):
        return []
    config = strategy_data.get("morning_signal_config", {})
    if not isinstance(config, dict):
        return []
    repos = config.get("scan_repos", [])
    if not isinstance(repos, list):
        return []
    return [str(r).strip() for r in repos if str(r).strip()]


def resolve_row_repo(
    row: Dict[str, Any], configured: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """The one repository this row's issue belongs to, or why it is unknown.

    Returns ``(repo, problem)`` with exactly one of the two set.

    There is no "try them all" branch, and that is the point. ``#42`` is not
    an issue identifier; ``owner/repo#42`` is. Searching every configured repo
    for a number and binding to the first hit resolved a blocker whose issue 42
    was open in one project because a *different* project had closed its own
    issue 42. A row that cannot name its repository is reported unchecked.
    """
    declared = str(row.get("repo") or "").strip()
    if declared:
        return declared, None
    raw = str(row.get("issue") or "").strip()
    if "#" in raw and "/" in raw.split("#", 1)[0]:
        return raw.split("#", 1)[0], None
    if len(configured) == 1:
        # Exactly one configured repository: unqualified is unambiguous.
        return configured[0], None
    if not configured:
        return None, UNRESOLVABLE
    return None, AMBIGUOUS_REPO


async def check_blockers(
    ledger_data: Dict[str, Any],
    strategy_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Look up each active blocker's issue and classify it.

    Returns a report with ``closed``/``open``/``unresolvable`` row lists and a
    ``reason`` when the check could not run at all. A missing token or an empty
    ``scan_repos`` is a *skipped* check, never an empty result set — reporting
    "0 stale blockers" because nothing was queried would be the same lie the
    ticket was filed about.
    """
    rows = active_blockers(
        ledger_data.get("blockers", []) if isinstance(ledger_data, dict) else []
    )
    report: Dict[str, Any] = {
        "checked": 0,
        "closed": [],
        "open": [],
        "unresolvable": [],
        "ran": False,
    }
    if not rows:
        report["ran"] = True
        return report

    configured = configured_repos(strategy_data)
    resolved = [(row, *resolve_row_repo(row, configured)) for row in rows]
    if not any(repo for _, repo, _ in resolved):
        report["reason"] = (
            "No scan_repos configured in morning_signal_config and no blocker "
            "carries an explicit repo -- nothing could be looked up."
        )
        return report

    token = get_github_token()
    if not token:
        report["reason"] = (
            "No GITHUB_TOKEN found -- live blocker state could not be checked."
        )
        return report

    report["ran"] = True
    for row, repo, problem in resolved:
        number = _issue_number(row)
        entry = {
            "id": row.get("id"),
            "issue": row.get("issue"),
            "title": row.get("title"),
            "repo": repo,
        }
        if number is None or repo is None:
            entry["reason"] = problem or UNRESOLVABLE
            if entry["reason"] == AMBIGUOUS_REPO:
                entry["candidate_repos"] = list(configured)
            report["unresolvable"].append(entry)
            continue

        issue = await github_api_get(f"/repos/{repo}/issues/{number}", token)
        state = (
            str(issue["state"]).lower()
            if isinstance(issue, dict) and issue.get("state")
            else None
        )
        report["checked"] += 1
        entry["state"] = state
        if state == "closed":
            report["closed"].append(entry)
        elif state == "open":
            report["open"].append(entry)
        else:
            entry["reason"] = UNRESOLVABLE
            report["unresolvable"].append(entry)
    return report
