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
from copy import deepcopy
from typing import Any, Optional

from kestrel_sdk.hooks.base import PermissionDecision

from kestrel_sovereign.hooks.manager import FailedHookDenial, deciding_hook

__all__ = ["BlockedDecision", "decided_tool_input", "evaluate_blocking_decision"]


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
    #: The enforcing hook whose failure (a raise or a timeout) produced this
    #: DENY, or ``None`` when a hook decided it. A failed hook decided
    #: nothing: the manager failed closed on its behalf (#3749).
    failed_hook: Optional[str] = None
    #: The hook whose DENY or ASK this is, as the hook manager recorded it,
    #: or ``None`` when that is not known (a failed hook, an output no
    #: manager chain produced). An identifier, never the hook's reason
    #: (#3873).
    deciding_hook: Optional[str] = None

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

    A DENY the hook manager produced because an enforcing hook failed is
    still blocking, and carries the hook's name in ``failed_hook``. A decided
    DENY or ASK carries the deciding hook's name in ``deciding_hook``.
    """
    if hook_output is None:
        return None

    decision = getattr(hook_output, "permission_decision", None)
    reason_attr = getattr(hook_output, "permission_reason", None)
    decided_by = deciding_hook(hook_output)

    if decision == PermissionDecision.DENY:
        return BlockedDecision(
            PermissionDecision.DENY,
            reason_attr or deny_reason_default,
            failed_hook=(
                hook_output.failed_hook
                if isinstance(hook_output, FailedHookDenial)
                else None
            ),
            deciding_hook=decided_by,
        )
    if decision == PermissionDecision.ASK:
        return BlockedDecision(
            PermissionDecision.ASK,
            reason_attr or ask_reason_default,
            deciding_hook=decided_by,
        )

    if getattr(hook_output, "continue_execution", True) is False:
        reason = (
            reason_attr
            or getattr(hook_output, "stop_reason", None)
            or deny_reason_default
        )
        return BlockedDecision(
            PermissionDecision.DENY, reason, deciding_hook=decided_by
        )

    return None


def decided_tool_input(hook_input: Any, hook_output: Any) -> Any:
    """The arguments a PRE-hook chain decided on, whatever it decided.

    The manager threads a MODIFY rewrite into ``hook_input.tool_input`` for
    the next hook, and a hook may also mutate that payload in place, so it
    holds the arguments as they stood when the deciding hook ran. A deciding
    output's own ``updated_input`` overrides it (#3953). An executed call runs
    with these and a refused one is recorded with them: a redaction a hook made
    before a later hook denied is never undone in the record.

    Detach the decided binding from retained hook inputs and outputs before
    task or claim persistence awaits. The same owned values then supply the
    request record, effect digest, execution and audit.
    """
    updated = getattr(hook_output, "updated_input", None)
    return deepcopy(updated if updated is not None else hook_input.tool_input)
