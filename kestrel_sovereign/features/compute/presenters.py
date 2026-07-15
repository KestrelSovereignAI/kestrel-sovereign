"""Pure presentation helpers for compute read tools.

The compute feature owns orchestration and persistence queries.  This module
owns only the deterministic conversion of compute models into ``ToolResult``
responses; it deliberately has no dependency on the feature, stores,
security policy, or executors.
"""

from collections.abc import Sequence
from typing import Final

from kestrel_sdk.tools.result import ToolResult

from .models import ComputeScript, ExecutionRecord, ScriptState

_SCRIPT_STATE_ICONS: Final = {
    ScriptState.DRAFT: "📝",
    ScriptState.SIGNED: "✍️",
    ScriptState.PENDING_REVIEW: "⏳",
    ScriptState.APPROVED: "✅",
    ScriptState.REJECTED: "⛔",
    ScriptState.QUEUED: "📋",
    ScriptState.RUNNING: "⚡",
    ScriptState.COMPLETED: "✅",
    ScriptState.FAILED: "❌",
}

_FINDING_SEVERITY_ICONS: Final = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
    "info": "ℹ️",
}


def present_script_list(scripts: Sequence[ComputeScript]) -> ToolResult:
    """Format the result of a script-list query."""
    if not scripts:
        return ToolResult.ok(
            confirmation="No scripts found.",
            data={"scripts": [], "count": 0},
        )

    lines = ["📜 Scripts:\n"]
    for script in scripts:
        status_icon = _SCRIPT_STATE_ICONS.get(script.state, "❓")
        lines.append(
            f"  {status_icon} {script.id[:8]} | {script.name[:20]:<20} | "
            f"{script.language:<6} | {script.state.value:<14} | "
            f"risk:{script.risk_score:>3}"
        )

    return ToolResult.ok(
        confirmation="\n".join(lines),
        data={
            "count": len(scripts),
            "scripts": [
                {
                    "id": script.id,
                    "name": script.name,
                    "language": script.language,
                    "state": script.state.value,
                    "risk_score": script.risk_score,
                }
                for script in scripts
            ],
        },
    )


def present_script_detail(script: ComputeScript) -> ToolResult:
    """Format one script and its bounded security/content previews."""
    lines = [
        f"📜 Script: {script.name}",
        f"   ID: {script.id}",
        f"   Language: {script.language}",
        f"   State: {script.state.value}",
        f"   Purpose: {script.purpose}",
        f"   Risk Score: {script.risk_score}/100",
        f"   Created: {script.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if script.signed_by:
        lines.append(f"   Signed by: {script.signed_by[:32]}...")

    if script.requirements:
        lines.append(f"   Requirements: {', '.join(script.requirements)}")

    if script.review_notes:
        lines.append(f"   Review Notes: {script.review_notes}")

    if script.security_findings:
        lines.append("\n🔒 Security Findings:")
        for finding in script.security_findings[:5]:
            icon = _FINDING_SEVERITY_ICONS.get(finding.severity, "❓")
            lines.append(
                f"   {icon} [{finding.severity.upper()}] {finding.description}"
            )
            if finding.line_number:
                lines.append(
                    f"      Line {finding.line_number}: {finding.pattern_matched[:50]}"
                )

    lines.append("\n📝 Content:")
    content_lines = script.content.split("\n")
    for line_number, line in enumerate(content_lines[:20], 1):
        lines.append(f"   {line_number:3}| {line}")
    if len(content_lines) > 20:
        lines.append(f"   ... ({len(content_lines) - 20} more lines)")

    return ToolResult.ok(
        confirmation="\n".join(lines),
        data={
            "script_id": script.id,
            "name": script.name,
            "language": script.language,
            "state": script.state.value,
            "risk_score": script.risk_score,
            "purpose": script.purpose,
            "requirements": script.requirements,
            "review_notes": script.review_notes,
            "findings_count": len(script.security_findings or []),
        },
    )


def present_execution_history(
    executions: Sequence[ExecutionRecord],
) -> ToolResult:
    """Format the result of an execution-history query."""
    if not executions:
        return ToolResult.ok(
            confirmation="No executions found.",
            data={"executions": [], "count": 0},
        )

    lines = ["📊 Execution History:\n"]
    for execution in executions:
        status = "✅" if execution.succeeded else "❌"
        duration = (
            f"{execution.duration_seconds:.2f}s"
            if execution.duration_seconds
            else "N/A"
        )
        lines.append(
            f"  {status} {execution.id[:8]} | script:{execution.script_id[:8]} | "
            f"exit:{execution.exit_code} | {duration} | {execution.executor}"
        )

    return ToolResult.ok(
        confirmation="\n".join(lines),
        data={
            "count": len(executions),
            "executions": [
                {
                    "id": execution.id,
                    "script_id": execution.script_id,
                    "exit_code": execution.exit_code,
                    "duration_seconds": execution.duration_seconds,
                    "executor": execution.executor,
                    "succeeded": execution.succeeded,
                }
                for execution in executions
            ],
        },
    )


__all__ = [
    "present_execution_history",
    "present_script_detail",
    "present_script_list",
]
