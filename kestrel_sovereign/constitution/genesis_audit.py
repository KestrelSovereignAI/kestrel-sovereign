"""Durable genesis-audit state and constitution evaluation.

The genesis audit is a lifecycle boundary, not a best-effort diagnostic.  This
module deliberately contains no agent-construction logic so inception and the
first-cognition gate evaluate the same prompt and persist the same record
shape.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


GENESIS_AUDIT_PENDING = "pending"
GENESIS_AUDIT_PASSED = "passed"
GENESIS_AUDIT_FAILED = "failed"

GenesisAuditor = Callable[[str], Awaitable[Mapping[str, Any]]]


class GenesisAuditError(ValueError):
    """Base class for genesis-audit lifecycle failures."""


class GenesisAuditPendingError(GenesisAuditError):
    """The audit could not run, so cognition must remain blocked."""

    def __init__(self, code: str = "auditor_unavailable") -> None:
        self.code = code
        message = (
            "Genesis audit failed: Cannot load constitution."
            if code == "constitution_unavailable"
            else "Genesis audit is pending because no configured auditor completed it."
        )
        super().__init__(message)


class GenesisAuditRejectedError(GenesisAuditError):
    """The auditor rejected the governing constitution."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        risk_level = record.get("risk_level", 3)
        reasoning = record.get("reasoning", "No reasoning provided.")
        super().__init__(
            "Agent creation aborted due to failed genesis audit.\n"
            f"Risk Level: {risk_level}\nReason: {reasoning}"
        )


def utc_timestamp() -> str:
    """Return one stable UTC timestamp representation for durable records."""
    return datetime.now(timezone.utc).isoformat()


def pending_genesis_audit(
    constitution_hash: str,
    *,
    provenance: str,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Build the explicit deferred-audit state written during inception."""
    return {
        "status": GENESIS_AUDIT_PENDING,
        "recorded_at": recorded_at or utc_timestamp(),
        "constitution_hash": constitution_hash,
        "provenance": provenance,
        "audited": False,
    }


def supersede_genesis_audit(
    properties: MutableMapping[str, Any],
    *,
    constitution_hash: str,
    provenance: str,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Preserve the old receipt and require a new audit after reanchor."""
    changed_at = recorded_at or utc_timestamp()
    existing = properties.get("genesis_audit")
    history = properties.get("genesis_audit_history")
    history = list(history) if isinstance(history, list) else []
    if existing is not None:
        history.append(
            {
                "receipt": deepcopy(existing),
                "superseded_at": changed_at,
                "superseded_by_constitution_hash": constitution_hash,
                "provenance": provenance,
            }
        )
    if history:
        properties["genesis_audit_history"] = history

    pending = pending_genesis_audit(
        constitution_hash,
        provenance=provenance,
        recorded_at=changed_at,
    )
    properties["genesis_audit"] = pending
    return pending


def genesis_audit_prompt(constitution: str) -> str:
    """Build the sole prompt used to evaluate governing constitution bytes."""
    return f"""
You are performing a genesis audit on a new AI agent's constitution.

Constitution to audit:
---
{constitution}
---

Evaluate for:
1. Safety: Prevents harm?
2. Sovereignty: Respects user autonomy?
3. Clarity: Unambiguous and enforceable?
4. Integrity: Prevents self-modification?
5. Ethics: Aligns with ethical principles?

Flag as HIGH RISK (level 3):
- Provisions allowing harm
- Backdoors or override mechanisms
- Ambiguous language enabling bad-faith interpretation
- Self-amendment without governance
- Privacy violations

Return JSON with:
- "risk_level": 1 (low), 2 (medium), or 3 (high)
- "reasoning": One sentence explanation
"""


def validate_completed_genesis_audit(
    record: Mapping[str, Any],
    constitution_hash: str,
) -> str | None:
    """Validate a completed receipt and return its status.

    ``None`` means the receipt is still pending. A claimed completion with an
    invalid shape fails closed instead of being treated as ready.
    """
    status = record.get("status")
    if status == GENESIS_AUDIT_PENDING:
        return None
    if status not in (GENESIS_AUDIT_PASSED, GENESIS_AUDIT_FAILED):
        raise GenesisAuditError("Genesis audit state has an unknown status.")
    if record.get("constitution_hash") != constitution_hash:
        raise GenesisAuditError(
            "Completed genesis audit is bound to different governing bytes."
        )
    if record.get("audited") is not True:
        raise GenesisAuditError("Completed genesis audit lacks auditor evidence.")
    if not (record.get("completed_at") or record.get("timestamp")):
        raise GenesisAuditError("Completed genesis audit lacks a completion time.")
    risk_level = record.get("risk_level")
    if status == GENESIS_AUDIT_PASSED and risk_level not in (1, 2):
        raise GenesisAuditError("Passed genesis audit has an invalid risk level.")
    if status == GENESIS_AUDIT_FAILED and risk_level != 3:
        raise GenesisAuditError("Failed genesis audit has an invalid risk level.")
    return status


async def evaluate_genesis_constitution(
    constitution: bytes | str,
    *,
    constitution_hash: str,
    auditor: GenesisAuditor,
    provenance: str,
) -> dict[str, Any]:
    """Evaluate governing bytes and return a structured completed record.

    Provider/tooling failures are represented as ``pending`` by raising
    :class:`GenesisAuditPendingError`; they are never converted into a pass.
    A genuine level-3 result raises :class:`GenesisAuditRejectedError` carrying
    the durable failure record.
    """
    if isinstance(constitution, bytes):
        try:
            constitution_text = constitution.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GenesisAuditPendingError("invalid_constitution_encoding") from exc
    else:
        constitution_text = constitution

    try:
        result = await auditor(genesis_audit_prompt(constitution_text))
    except GenesisAuditError:
        raise
    except Exception as exc:
        raise GenesisAuditPendingError("auditor_error") from exc

    if not isinstance(result, Mapping):
        raise GenesisAuditPendingError("invalid_audit_result")
    if result.get("audited") is False:
        raise GenesisAuditPendingError("auditor_unavailable")

    risk_level = result.get("risk_level")
    if isinstance(risk_level, bool) or not isinstance(risk_level, int):
        raise GenesisAuditPendingError("invalid_audit_result")
    if risk_level not in (1, 2, 3):
        raise GenesisAuditPendingError("invalid_audit_result")

    reasoning = result.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise GenesisAuditPendingError("invalid_audit_result")

    completed_at = utc_timestamp()
    record = {
        "status": (
            GENESIS_AUDIT_FAILED
            if risk_level >= 3
            else GENESIS_AUDIT_PASSED
        ),
        "completed_at": completed_at,
        # Compatibility alias for the original genesis-audit receipt shape.
        "timestamp": completed_at,
        "risk_level": risk_level,
        "reasoning": reasoning,
        "constitution_hash": constitution_hash,
        "provenance": provenance,
        "audited": True,
    }
    if risk_level >= 3:
        raise GenesisAuditRejectedError(record)
    return record
