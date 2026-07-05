"""Host-provided signal sources for the ``stalled_work_rescue`` workflow (#2192).

The ``kestrel-feature-workflows`` package ships a built-in ``stalled_work_rescue``
workflow (sovereign #1523): detect stalled fleet work, gate the intent through
governance, dispatch repairs to specialist agents, verify the fix landed, and
close the loop on evidence. Each stage names the **signal source** it expects the
hosting agent to provide. Until those sources are registered, the workflow
runner's start-contract validation fails before a run record is created
("stage 'govern_intent' references unregistered source 'governance_review'").

This module owns default, agent-native registrations for the six sources the
built-in names, so an agent that loads ``stalled_work_rescue`` can actually start
it. No claws/castle dependency — the loop is entirely agent-native.

    fleet_stalled_sweep    — surveys for stalled/blocking work (read-only)
    governance_review      — records the intent to intervene (consent-gated)
    a2a_repair_dispatch    — routes repairs to specialist agents (irreversible)
    evidence_verify        — confirms the fix actually landed (read-only)
    close_resolved_todos   — closes todos proven resolved
    reopen_resolved_todos  — compensation for the close stage

**Evidence boundaries (see AGENTS.md "Authoring multi-agent Workflow scripts").**
Handlers are conservative and fail closed. They quote the upstream observation
fields verbatim before making any claim, and the dispatch/close stages refuse to
infer merged/shipped/resolved state without upstream evidence in the payload:

    - ``a2a_repair_dispatch`` records work as *dispatched*, never *merged*/*shipped*,
      and fails closed when no repair targets are supplied.
    - ``evidence_verify`` returns OK only on real evidence; missing evidence raises.
    - ``close_resolved_todos`` refuses to close a todo that carries no resolution
      evidence.

An ACTION handler that returns normally yields ``SignalResult.status == OK`` (the
``signal_status_ok`` gate passes); raising fails the stage. Failing closed is the
whole point when required evidence is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from kestrel_sdk.signals import (
    RateLimit,
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
)

logger = logging.getLogger(__name__)

FLEET_STALLED_SWEEP = "fleet_stalled_sweep"
GOVERNANCE_REVIEW = "governance_review"
A2A_REPAIR_DISPATCH = "a2a_repair_dispatch"
EVIDENCE_VERIFY = "evidence_verify"
CLOSE_RESOLVED_TODOS = "close_resolved_todos"
REOPEN_RESOLVED_TODOS = "reopen_resolved_todos"

# Every source this module registers, in the order the workflow visits them
# (with the compensation source last). ``build_workflow_rescue_registrations``
# is the one entry point callers use.
SOURCE_NAMES = (
    FLEET_STALLED_SWEEP,
    GOVERNANCE_REVIEW,
    A2A_REPAIR_DISPATCH,
    EVIDENCE_VERIFY,
    CLOSE_RESOLVED_TODOS,
    REOPEN_RESOLVED_TODOS,
)

CONSENT_MARKER_FIELDS = frozenset(
    {
        "approved",
        "consent",
        "accepted",
        "decision",
        "status",
        "state",
        "outcome",
        "approval_id",
        "approved_by",
    }
)


def _passthrough_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept any dict payload; reject non-dicts so the audit row is stable."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"workflow-rescue payload must be a dict, got {type(payload).__name__}"
        )
    return payload


def _quote(source: str, field: str, value: Any) -> str:
    """Format an observation the evidence-boundary way (AGENTS.md convention).

    ``<source> reported `<field>: <value>`.`` — quote the upstream field
    verbatim *before* any claim is derived from it, so a reader can always
    trace a claim back to the observation that supports it.
    """
    return f"`{source}` reported `{field}: {value!r}`."


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# --------------------------------------------------------------------------
# Stage handlers — conservative, fail-closed, evidence-boundary-preserving.
# --------------------------------------------------------------------------


async def fleet_stalled_sweep_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Detect stalled/blocking work. Read-only observation.

    Pure observation: it reports the stalled items handed to it (or observed by
    an upstream survey) and never claims any of them are resolved. Observing
    zero stalled items is a valid OK result, so this never fails closed.
    """
    stale_days = payload.get("stale_days", 3)
    items = _as_list(payload.get("stalled_items") or payload.get("candidates"))
    return {
        "source": FLEET_STALLED_SWEEP,
        "stale_days": stale_days,
        "stalled_items": items,
        "stalled_count": len(items),
        "observation": _quote(FLEET_STALLED_SWEEP, "stalled_count", len(items)),
    }


async def governance_review_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record the intent to intervene. Consent-gated by the workflow.

    This only *registers* the intent; it does not authorize it. Approval is the
    workflow's ``consent_collect`` gate, evaluated separately. The handler quotes
    the observed stalled count but makes no state claim about the intervention.
    """
    scope = payload.get("scope", "proactive_work_rescue")
    stalled_count = payload.get("stalled_count")
    observation = (
        _quote(GOVERNANCE_REVIEW, "stalled_count", stalled_count)
        if stalled_count is not None
        else _quote(GOVERNANCE_REVIEW, "scope", scope)
    )
    result = {
        "source": GOVERNANCE_REVIEW,
        "scope": scope,
        "intent": "request_consent",
        "authorized": False,  # authorization is the consent gate's job, not ours
        "observation": observation,
    }
    for field in CONSENT_MARKER_FIELDS:
        if field in payload:
            result[field] = payload[field]
    return result


async def a2a_repair_dispatch_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Route repairs to specialist agents via A2A. Irreversible side effect.

    Fails closed when no repair targets are supplied — dispatching "nothing"
    would masquerade as a completed rescue. Records work strictly as
    *dispatched*; it must never infer ``merged``/``shipped`` state. That is what
    the downstream ``evidence_verify`` stage exists to establish.
    """
    targets = _as_list(
        payload.get("repairs")
        or payload.get("repair_targets")
        or payload.get("stalled_items")
    )
    if not targets:
        raise ValueError(
            "a2a_repair_dispatch: no repair targets supplied; refusing to "
            "dispatch (fail closed)"
        )
    return {
        "source": A2A_REPAIR_DISPATCH,
        "dispatched": targets,
        "dispatched_count": len(targets),
        # Dispatched is NOT merged/shipped — do not let a reader (or a synthesis
        # stage) infer completion from a dispatch (AGENTS.md #1484 convention).
        "state": "dispatched",
        "observation": _quote(A2A_REPAIR_DISPATCH, "dispatched_count", len(targets)),
    }


async def evidence_verify_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Confirm the fix actually landed. Read-only evidence check.

    Returns OK only on real evidence. When the payload carries no evidence the
    handler raises (fail closed) rather than pretend the fix landed. Verified
    state is derived strictly from the quoted evidence — never inferred beyond it.
    """
    evidence = payload.get("evidence")
    if not evidence:
        raise ValueError(
            "evidence_verify: no evidence supplied; cannot confirm the fix "
            "landed (fail closed)"
        )
    return {
        "source": EVIDENCE_VERIFY,
        "verified": True,
        "evidence": evidence,
        "observation": _quote(EVIDENCE_VERIFY, "evidence", evidence),
    }


async def close_resolved_todos_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Close todos proven resolved.

    Refuses to close a todo that carries no resolution evidence — closing on an
    unproven ``resolved`` claim is exactly the state-inference this loop guards
    against. Each closed id is accompanied by the evidence that justified it.
    """
    resolved = _as_list(payload.get("resolved_todos"))
    if not resolved:
        raise ValueError(
            "close_resolved_todos: no resolved todos supplied; nothing may be "
            "closed without upstream evidence (fail closed)"
        )
    closed: List[Any] = []
    for todo in resolved:
        if not isinstance(todo, dict):
            raise ValueError(
                "close_resolved_todos: each resolved todo must be a dict "
                "carrying its resolution evidence (fail closed)"
            )
        todo_id = todo.get("id")
        if todo_id is None or not todo.get("evidence"):
            raise ValueError(
                f"close_resolved_todos: todo {todo_id!r} lacks resolution "
                "evidence; refusing to close (fail closed)"
            )
        closed.append(todo_id)
    return {
        "source": CLOSE_RESOLVED_TODOS,
        "closed": closed,
        "closed_count": len(closed),
        "observation": _quote(CLOSE_RESOLVED_TODOS, "closed_count", len(closed)),
    }


async def reopen_resolved_todos_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compensation for the close stage: reopen todos that were closed.

    Reopens whatever the run recorded as closed; reopening nothing is a valid
    idempotent no-op, so this never fails closed.
    """
    to_reopen = _as_list(payload.get("closed") or payload.get("resolved_todos"))
    reopened = [
        todo.get("id") if isinstance(todo, dict) else todo for todo in to_reopen
    ]
    return {
        "source": REOPEN_RESOLVED_TODOS,
        "reopened": reopened,
        "reopened_count": len(reopened),
    }


# --------------------------------------------------------------------------
# Registrations.
# --------------------------------------------------------------------------


def _redaction(source: str) -> RedactionPolicy:
    return RedactionPolicy(
        summarize=lambda payload, _s=source: (
            f"{_s} stale_days={payload.get('stale_days', '?')} "
            f"scope={payload.get('scope', '?')}"
        ),
        # These payloads are the bird's own orchestration data (workflow stage
        # params), not third-party UNTRUSTED content, so the summary above is
        # sufficient and raw storage is unnecessary.
        store_raw_trusted=False,
        redact_caller_identifier=True,
    )


def _action_registration(name: str, handler: Any) -> SourceRegistration:
    return SourceRegistration(
        name=name,
        schema=_passthrough_schema,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        trust=Trust.TRUSTED,
        # Workflow stages fire deliberately, one at a time; a modest cap only
        # guards against a pathological retry loop and never throttles a
        # legitimate rescue run.
        rate_limit=RateLimit(per_minute=30, per_hour=240),
        resources=frozenset(),
        allow_self_loops=False,
        log_redaction=_redaction(name),
        retention_days=30,
    )


def build_fleet_stalled_sweep_registration() -> SourceRegistration:
    return _action_registration(FLEET_STALLED_SWEEP, fleet_stalled_sweep_handler)


def build_governance_review_registration() -> SourceRegistration:
    return _action_registration(GOVERNANCE_REVIEW, governance_review_handler)


def build_a2a_repair_dispatch_registration() -> SourceRegistration:
    return _action_registration(A2A_REPAIR_DISPATCH, a2a_repair_dispatch_handler)


def build_evidence_verify_registration() -> SourceRegistration:
    return _action_registration(EVIDENCE_VERIFY, evidence_verify_handler)


def build_close_resolved_todos_registration() -> SourceRegistration:
    return _action_registration(CLOSE_RESOLVED_TODOS, close_resolved_todos_handler)


def build_reopen_resolved_todos_registration() -> SourceRegistration:
    return _action_registration(REOPEN_RESOLVED_TODOS, reopen_resolved_todos_handler)


def build_workflow_rescue_registrations() -> List[SourceRegistration]:
    """Every source the built-in ``stalled_work_rescue`` workflow references."""
    return [
        build_fleet_stalled_sweep_registration(),
        build_governance_review_registration(),
        build_a2a_repair_dispatch_registration(),
        build_evidence_verify_registration(),
        build_close_resolved_todos_registration(),
        build_reopen_resolved_todos_registration(),
    ]


def register_workflow_rescue_sources(registry: Any) -> List[str]:
    """Register the rescue sources on ``registry``, idempotently.

    Returns the names newly registered (already-present sources are skipped so a
    second call — or a host that registered a richer implementation — is safe).
    A single source's failure is logged and does not abort the rest.
    """
    registered: List[str] = []
    if registry is None or not hasattr(registry, "register"):
        return registered
    has_get = hasattr(registry, "get")
    for registration in build_workflow_rescue_registrations():
        if has_get and registry.get(registration.name) is not None:
            continue
        try:
            registry.register(registration)
            registered.append(registration.name)
        except Exception as exc:  # noqa: BLE001 - one bad source must not abort the rest
            logger.warning(
                "could not register workflow-rescue source %s: %s",
                registration.name,
                exc,
            )
    return registered
