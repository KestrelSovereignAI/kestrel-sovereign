"""Security-audit writes for tool calls that never reach the permission layer.

The tool-name allowlist check in
:func:`kestrel_sovereign.security.input_guardrails.validate_tool_arguments`
runs *before* PRE_TOOL_USE, so a rejection there used to leave no
``security_audit_log`` row at all: the call simply vanished, and the only
evidence was the model's own confusion (#2929). Amendment IX's premise is that
every dangerous-capability invocation is recorded with the chain of layers that
allowed or refused it — a pre-permission bounce is exactly the hole that
premise cannot tolerate.

Writes are best-effort, mirroring
:func:`kestrel_sovereign.security.demo_isolation._record_audit`: an agent
running without a SecurityFeature (early startup, constrained tests) still
surfaces the refusal through the logger instead of raising back into the tool
loop.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: ``feature_name`` recorded for a refusal. Deliberately not a real feature —
#: the whole point of these rows is that the rejected name resolved to none.
REJECTION_FEATURE_NAME = "tool_guardrails"

#: The guardrail refused the call while validating its name/arguments.
ACTION_TOOL_VALIDATION = "tool_validation"

#: The name passed validation but no loaded feature could dispatch it.
ACTION_TOOL_RESOLUTION = "tool_resolution"

#: Audit ``decision`` for a refusal. ``blocked`` is already one of the
#: wellness FrictionCalculator's friction decisions, so these rows show up in
#: the friction metric rather than needing a new vocabulary.
REJECTION_DECISION = "blocked"


def _permission_store(agent: Any) -> Optional[Any]:
    """Return the agent's PermissionStore, or ``None`` when unavailable."""
    try:
        features = getattr(agent, "features", {}) or {}
        feature = features.get("SecurityFeature") or features.get("Security")
    except Exception:  # noqa: BLE001 — defensive: never break the tool loop
        return None
    if feature is None:
        return None
    store = getattr(feature, "permission_store", None)
    if store is None or not callable(getattr(store, "log_decision", None)):
        return None
    return store


async def record_tool_rejection(
    agent: Any,
    *,
    tool_name: str,
    reason: str,
    action: str = ACTION_TOOL_VALIDATION,
    args: Optional[dict] = None,
) -> bool:
    """Record one refused tool call in ``security_audit_log``.

    Args:
        agent: The agent whose SecurityFeature owns the audit store.
        tool_name: The rejected tool name, recorded verbatim so an operator
            can see exactly what the model tried to call.
        reason: Why it was refused (the guardrail's own error text).
        action: Which guardrail refused — :data:`ACTION_TOOL_VALIDATION` or
            :data:`ACTION_TOOL_RESOLUTION`.
        args: The call's arguments; masked and truncated before they are
            written (the audit table is plaintext SQLite).

    Returns:
        True when a row was written, False when the audit store was
        unavailable or the write failed. Callers must not branch on the
        rejection itself — that decision has already been made.
    """
    from kestrel_sovereign.features.security.args_summary import summarize_args

    summary = reason
    args_summary = summarize_args(args)
    if args_summary:
        summary = f"{reason} | args={args_summary}"

    store = _permission_store(agent)
    if store is None:
        logger.warning(
            "[TOOL-GUARDRAIL] %s refused tool %r — audit store unavailable, "
            "reason=%s", action, tool_name, reason,
        )
        return False

    try:
        await store.log_decision(
            feature_name=REJECTION_FEATURE_NAME,
            tool_name=tool_name,
            action=action,
            decision=REJECTION_DECISION,
            args_summary=summary,
        )
        return True
    except Exception as e:  # noqa: BLE001 — a failed audit must not wedge a turn
        logger.warning(
            "[TOOL-GUARDRAIL] failed to record refusal of %r (%s); reason=%s",
            tool_name, e, reason,
        )
        return False
