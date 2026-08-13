"""Signal source for scheduled stale-work / red-CI discovery wakes (#2281).

The polling half is the ``ecosystem_discovery_watch`` ACTION cron task wired
by ``SchedulerFeature``. It delegates to an explicitly named feature-owned
discovery tool, normalizes the tool result into compact actionable
findings, fingerprints that compact state, and emits one COGNITION signal only
when actionable findings are new, changed, or just resolved.

Discovery is deliberately not a repair primitive. The signal payload gives the
agent enough context to route by lane and choose the next evidence gate; it
does not pre-approve dispatch or closure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from kestrel_sdk.signals import (
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Trust,
)


SOURCE_NAME = "ecosystem.discovery_findings"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "signals" / "ecosystem_discovery_findings.md"
)

DEFAULT_MAX_FINDINGS = 20

_CLEAN_STATUSES = {
    "clean",
    "closed",
    "done",
    "fixed",
    "green",
    "ok",
    "passed",
    "resolved",
    "success",
}


@dataclass(frozen=True)
class DiscoveryState:
    """Compact state used for dedupe and the COGNITION payload."""

    fingerprint: str
    findings: tuple[dict[str, Any], ...]
    summary: str


@dataclass(frozen=True)
class DiscoveryDecision:
    """Outcome of comparing one discovery scan with persisted state."""

    should_signal: bool
    reason: str
    state: DiscoveryState
    previous_findings: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _coerce_result(raw: Any) -> Any:
    """Unwrap common tool-return containers and JSON strings."""
    data = getattr(raw, "data", raw)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {"summary": data, "findings": []}
    if isinstance(data, dict) and "data" in data and "status" in data:
        nested = data.get("data")
        if isinstance(nested, (dict, list)):
            return nested
    return data


def _iter_candidate_lists(data: Any) -> Iterable[Any]:
    if isinstance(data, list):
        yield data
        return
    if not isinstance(data, dict):
        return

    for key in (
        "findings",
        "actionable_findings",
        "stale_items",
        "items",
        "results",
        "issues",
        "prs",
        "pull_requests",
        "red_ci",
        "red_ci_findings",
        "failed_checks",
    ):
        value = data.get(key)
        if isinstance(value, list):
            yield value


def _first_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def _infer_kind(item: dict[str, Any]) -> str:
    explicit = _first_text(item, ("kind", "type", "category"))
    if explicit:
        return explicit
    if item.get("check") or item.get("job") or "ci" in str(item).lower():
        return "red_ci"
    if item.get("pr") or item.get("pull_request"):
        return "pr"
    if item.get("issue") or item.get("number"):
        return "issue"
    return "stale_work"


def _suggest_gate(kind: str, item: dict[str, Any]) -> str:
    explicit = _first_text(item, ("suggested_gate", "next_gate", "gate"))
    if explicit:
        return explicit
    text = f"{kind} {item}".lower()
    if "ci" in text or "check" in text or "failure" in text or "red" in text:
        return "verify_ci_then_dispatch_fix"
    if "pr" in text or item.get("pr") or item.get("pull_request"):
        return "review_pr_state"
    if "issue" in text or item.get("issue"):
        return "triage_issue"
    return "triage_lane"


def _finding_key(finding: dict[str, Any]) -> str:
    parts = [
        finding.get("repo", ""),
        finding.get("kind", ""),
        finding.get("number", ""),
        finding.get("branch", ""),
        finding.get("check", ""),
        finding.get("job", ""),
        finding.get("id", ""),
        finding.get("title", ""),
    ]
    return "|".join(str(p) for p in parts if str(p))


def _is_actionable(item: dict[str, Any]) -> bool:
    if item.get("actionable") is False:
        return False
    status = _first_text(item, ("status", "state", "conclusion", "result")).lower()
    if status in _CLEAN_STATUSES:
        return False
    severity = _first_text(item, ("severity", "priority")).lower()
    if severity in {"none", "clean", "info", "resolved"}:
        return False
    return True


def normalize_discovery_result(
    raw: Any, *, default_repo: str = "", max_findings: int = DEFAULT_MAX_FINDINGS
) -> DiscoveryState:
    """Normalize a discovery tool result to compact actionable findings."""
    data = _coerce_result(raw)
    findings: list[dict[str, Any]] = []

    for candidates in _iter_candidate_lists(data):
        for raw_item in candidates:
            if not isinstance(raw_item, dict) or not _is_actionable(raw_item):
                continue
            kind = _infer_kind(raw_item)
            finding = {
                "repo": _first_text(raw_item, ("repo", "repository")) or default_repo,
                "kind": kind,
                "number": _first_text(
                    raw_item, ("number", "issue", "pr", "pull_request")
                ),
                "branch": _first_text(raw_item, ("branch", "head_branch", "ref")),
                "check": _first_text(raw_item, ("check", "check_name", "context")),
                "job": _first_text(raw_item, ("job", "job_name", "workflow")),
                "severity": _first_text(raw_item, ("severity", "priority")) or "medium",
                "status": _first_text(
                    raw_item, ("status", "state", "conclusion", "result")
                ),
                "title": _first_text(raw_item, ("title", "summary", "name")),
                "url": _first_text(raw_item, ("url", "html_url", "link")),
                "suggested_gate": _suggest_gate(kind, raw_item),
            }
            finding["key"] = _finding_key(finding)
            findings.append(finding)

    findings.sort(key=lambda f: (str(f.get("repo", "")), str(f.get("key", ""))))
    compact = tuple(findings[: max(0, int(max_findings))])
    summary = _summarize(data, compact)
    fingerprint = hashlib.sha256(
        json.dumps(compact, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return DiscoveryState(fingerprint=fingerprint, findings=compact, summary=summary)


def _summarize(data: Any, findings: tuple[dict[str, Any], ...]) -> str:
    if isinstance(data, dict):
        for key in ("summary", "message", "text"):
            value = data.get(key)
            if value is not None and str(value):
                return str(value)
    if findings:
        return f"{len(findings)} actionable finding(s)"
    return "No actionable stale-work or red-CI findings."


def evaluate_discovery_watch(
    raw_result: Any,
    *,
    last_fingerprint: str | None = None,
    last_state: dict[str, Any] | None = None,
    default_repo: str = "",
    max_findings: int = DEFAULT_MAX_FINDINGS,
) -> DiscoveryDecision:
    """Decide whether a discovery scan should wake cognition."""
    state = normalize_discovery_result(
        raw_result, default_repo=default_repo, max_findings=max_findings
    )
    previous = tuple((last_state or {}).get("findings") or ())

    if last_fingerprint == state.fingerprint:
        return DiscoveryDecision(False, "no_change", state, previous)
    if state.findings:
        reason = "new_findings" if not previous else "changed_findings"
        return DiscoveryDecision(True, reason, state, previous)
    if previous:
        return DiscoveryDecision(True, "resolved_findings", state, previous)
    return DiscoveryDecision(False, "clean", state, previous)


def state_to_json(state: DiscoveryState) -> str:
    return json.dumps(
        {"fingerprint": state.fingerprint, "findings": list(state.findings)},
        default=str,
        sort_keys=True,
    )


def state_from_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _schema(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "ecosystem.discovery_findings payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    payload.setdefault("watch_key", "")
    tool_name = payload.get("tool")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("ecosystem.discovery_findings payload requires non-empty tool")
    payload.setdefault("reason", "")
    payload.setdefault("summary", "")
    payload.setdefault("findings_count", "0")
    payload.setdefault("findings", "[]")
    payload.setdefault("previous_findings", "[]")
    return payload


def _redact(payload: dict[str, Any]) -> str:
    return (
        "ecosystem.discovery_findings "
        f"watch={payload.get('watch_key', '?')} "
        f"reason={payload.get('reason', '?')} "
        f"count={payload.get('findings_count', '?')}"
    )


def build_ecosystem_discovery_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        trust=Trust.TRUSTED,
        rate_limit=RateLimit(per_minute=10, per_hour=60),
        coalescing_window=timedelta(seconds=60),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_redact,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=14,
    )


def build_signal_for_discovery_findings(
    *,
    watch_key: str,
    tool_name: str,
    decision: DiscoveryDecision,
    target_agent: str,
) -> Signal:
    payload = {
        "watch_key": str(watch_key),
        "tool": str(tool_name),
        "reason": decision.reason,
        "summary": decision.state.summary,
        "findings_count": str(len(decision.state.findings)),
        "findings": json.dumps(list(decision.state.findings), default=str),
        "previous_findings": json.dumps(list(decision.previous_findings), default=str),
    }
    return Signal(
        source=SOURCE_NAME,
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload=payload,
        target_agent=target_agent,
        dedupe_key=f"{watch_key}:{decision.state.fingerprint[:12]}:{decision.reason}",
    )
