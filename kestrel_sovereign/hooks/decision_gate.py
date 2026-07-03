"""Shared PRE-hook decision gate.

Every dispatch surface that fires a blocking hook (``PRE_TOOL_USE``,
``PRE_SUBAGENT_CALL``, ``USER_PROMPT_SUBMIT``) must honor the SAME
contract: a ``DENY`` decision blocks, and — just as importantly — an
``ASK`` decision *also* blocks (it routes the call to an approval queue;
the tool MUST NOT run until that approval lands). ``continue_execution=
False`` is treated as blocking too, even when no explicit
``permission_decision`` was set.

Historically this logic was open-coded at each call site, and several
sites honored ``DENY`` but silently fell through on ``ASK`` (fail-OPEN —
kestrel-sovereign #2107 / F038, F245). This module extracts the one
``decision -> blocking envelope`` helper so the contract cannot drift
apart again: chat, subagent, and scheduler dispatch all route through
:func:`evaluate_blocking_decision`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from kestrel_sdk.hooks.base import PermissionDecision

__all__ = ["BlockedDecision", "evaluate_blocking_decision"]


@dataclass(frozen=True)
class BlockedDecision:
    """A hook decision that blocks execution (DENY or ASK).

    Carries the normalized reason plus the canonical tool-shaped error
    string / envelope every dispatch surface returns, so the wording
    ("Permission denied: …" / "Approval required: …") stays identical
    across the chat, subagent, and scheduler paths.
    """

    decision: PermissionDecision
    reason: str

    @property
    def is_ask(self) -> bool:
        return self.decision == PermissionDecision.ASK

    @property
    def error(self) -> str:
        prefix = (
            "Approval required"
            if self.decision == PermissionDecision.ASK
            else "Permission denied"
        )
        return f"{prefix}: {self.reason}"

    @property
    def envelope(self) -> dict:
        """The ``{"success": False, "error": ...}`` blocking envelope."""
        return {"success": False, "error": self.error}


def evaluate_blocking_decision(
    hook_output: Any,
    *,
    deny_reason_default: str = "Blocked by security policy",
    ask_reason_default: str = "Requires approval",
) -> Optional[BlockedDecision]:
    """Return a :class:`BlockedDecision` when the hook output blocks
    execution, or ``None`` when the tool/prompt may proceed.

    Blocking states (all fail-CLOSED):

    - ``permission_decision == DENY`` → blocked, "Permission denied".
    - ``permission_decision == ASK`` → blocked, "Approval required"
      (the call is queued for approval; it must NOT run now).
    - ``continue_execution is False`` with no explicit DENY/ASK → treated
      as a DENY-equivalent stop.
    """
    if hook_output is None:
        return None

    decision = getattr(hook_output, "permission_decision", None)
    reason_attr = getattr(hook_output, "permission_reason", None)

    if decision == PermissionDecision.DENY:
        return BlockedDecision(
            PermissionDecision.DENY, reason_attr or deny_reason_default
        )
    if decision == PermissionDecision.ASK:
        return BlockedDecision(
            PermissionDecision.ASK, reason_attr or ask_reason_default
        )

    if getattr(hook_output, "continue_execution", True) is False:
        reason = (
            reason_attr
            or getattr(hook_output, "stop_reason", None)
            or deny_reason_default
        )
        return BlockedDecision(PermissionDecision.DENY, reason)

    return None
