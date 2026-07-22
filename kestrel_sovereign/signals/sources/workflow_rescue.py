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
      and fails closed unless **explicit** repair targets (``repairs`` /
      ``repair_targets``) are supplied. It deliberately does NOT dispatch the
      raw ``stalled_items`` a survey observed — turning "detected" straight into
      "dispatched" would let a recurring loop auto-dispatch without per-run
      approval (#2200).
    - ``evidence_verify`` returns OK only on real evidence; missing evidence raises.
    - ``close_resolved_todos`` refuses to close a todo that carries no resolution
      evidence.

**Recurring observation-only ticks (#2249).** A recurring schedule runs with
``recurring: True`` in its params (see :func:`build_recurring_schedule_request`),
and the workflow runner merges the run params into every stage payload. When a
recurring tick reaches an irreversible/evidence-gated stage with **nothing
approved to act on** — no repair targets, no evidence, no resolved todos — the
stage completes cleanly as a *no-op* (``skipped: True``, zero count) instead of
failing the whole unattended run. This does not relax the fail-closed contract:
a **direct** (non-recurring) call to those stages still raises when its required
targets/evidence are absent, and even a recurring tick never auto-dispatches the
survey's detected ``stalled_items``. Dispatch/close only ever fire once a per-run
approval selects an explicit target and supplies its evidence.

Crucially, the no-op branch is gated on *nothing having been selected this run*.
If a recurring run **did** select explicit repair targets (``repairs`` /
``repair_targets`` merged into every stage payload from the run params) and
dispatched real work, the downstream ``evidence_verify`` / ``close_resolved``
stages still fail closed on missing evidence — a genuine irreversible dispatch
must be proven, never skipped past as a no-op (#2249 P1).

An ACTION handler that returns normally yields ``SignalResult.status == OK`` (the
``signal_status_ok`` gate passes); raising fails the stage. Failing closed is the
whole point when required evidence is absent.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from kestrel_sdk.signals import (
    RateLimit,
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
)

from kestrel_sovereign.signals.registry import (
    RegistrationPolicy,
    RegistrationState,
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


def _is_recurring_tick(payload: Dict[str, Any]) -> bool:
    """True when this stage payload belongs to a recurring observation-only loop.

    A recurring ``stalled_work_rescue`` schedule runs with ``recurring: True`` in
    its params (see :func:`build_recurring_schedule_request`); the workflow runner
    merges the run params into every stage payload, so the flag reaches each
    irreversible/evidence-gated stage. When a recurring tick reaches such a stage
    with **nothing approved to act on**, the stage completes cleanly as a no-op
    instead of failing the whole unattended run (#2249). A **direct**
    (non-recurring) call is unaffected and still fails closed when its required
    targets/evidence are absent — so "detected" never becomes "dispatched" and
    nothing is closed without evidence.
    """
    return bool(payload.get("recurring"))


# Markers that mean a real, per-run-approved action was selected/performed this
# run: explicit repair targets (merged into every stage payload from the run
# params) and the dispatch stage's own output (forwarded downstream). When any
# of these is present, later evidence-gated stages must NOT take the recurring
# no-op branch — a genuine irreversible dispatch has to be proven, not skipped
# (#2249 P1).
_ACTION_TARGET_FIELDS = ("repairs", "repair_targets")
_DISPATCHED_MARKER_FIELDS = ("dispatched", "dispatched_count")


def _run_selected_action(payload: Dict[str, Any]) -> bool:
    """True when this run selected/performed an explicit irreversible action.

    Guards the recurring no-op branch of the evidence-gated stages: a recurring
    tick is only allowed to skip when *nothing* was approved to act on. If the
    run carries explicit repair targets or a dispatched-work marker (from
    ``a2a_repair_dispatch``), the fix must be verified with real evidence — the
    no-op branch must not turn a real dispatch into a completed run without proof.
    A skipped (no-op) dispatch sets ``dispatched_count: 0`` and is not counted.
    """
    for field in _ACTION_TARGET_FIELDS:
        if payload.get(field):
            return True
    if payload.get("skipped"):
        # A no-op dispatch forwarded its own ``dispatched: []`` / count 0 — that
        # is not a selected action.
        return False
    for field in _DISPATCHED_MARKER_FIELDS:
        if payload.get(field):
            return True
    return False


def _recurring_skip(source: str, count_field: str, reason: str) -> Dict[str, Any]:
    """A clean no-op result for a recurring tick with nothing to act on.

    Records ``skipped: True`` and a zero count so no reader (or downstream
    synthesis stage) can mistake it for real work performed. Returning normally
    yields ``SignalResult.status == OK`` so the recurring run completes cleanly.
    """
    return {
        "source": source,
        "skipped": True,
        "recurring": True,
        "state": "skipped",
        count_field: 0,
        "reason": reason,
        "observation": _quote(source, count_field, 0),
    }


# --------------------------------------------------------------------------
# Stage handlers — conservative, fail-closed, evidence-boundary-preserving.
# --------------------------------------------------------------------------


async def _fleet_stalled_sweep(
    payload: Dict[str, Any], discover: Optional[Any]
) -> Dict[str, Any]:
    """Shared body for the fleet-stalled-sweep handler.

    Pure observation: it reports the stalled items handed to it, or — when none
    are pre-seeded and a live ``discover`` callable is bound — surveys the real
    fleet for stalled work. It never claims any item is resolved. Observing zero
    stalled items is a valid OK result, so this never fails closed; a discovery
    error degrades to "observed nothing" rather than aborting the loop.
    """
    stale_days = payload.get("stale_days", 3)
    items = _as_list(payload.get("stalled_items") or payload.get("candidates"))
    discovered = False
    if not items and discover is not None:
        try:
            surveyed = await discover(stale_days)
        except Exception as exc:  # noqa: BLE001 - observation degrades, never aborts
            logger.warning("fleet_stalled_sweep live discovery failed: %s", exc)
            surveyed = None
        items = _as_list(surveyed)
        discovered = bool(items)
    return {
        "source": FLEET_STALLED_SWEEP,
        "stale_days": stale_days,
        "stalled_items": items,
        "stalled_count": len(items),
        # True when these candidates came from a live survey (not pre-seeded),
        # so a reader can tell a recurring tick actually observed real work.
        "discovered": discovered,
        "observation": _quote(FLEET_STALLED_SWEEP, "stalled_count", len(items)),
    }


async def fleet_stalled_sweep_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Detect stalled/blocking work. Read-only observation (echo-only).

    The default, discovery-less handler: it reports the stalled items handed to
    it and never claims any of them are resolved. A host that can enumerate live
    stalled work binds a ``discover`` callable via
    :func:`build_fleet_stalled_sweep_registration` so a recurring tick observes
    real candidates instead of relying on pre-seeded ones (#2200).
    """
    return await _fleet_stalled_sweep(payload, discover=None)


def _make_fleet_stalled_sweep_handler(discover: Optional[Any]) -> Any:
    """Return a fleet-stalled-sweep handler bound to an optional ``discover``.

    ``discover`` is an ``async def discover(stale_days) -> list`` that surveys
    the live fleet for stalled work. When ``None``, the returned handler is the
    plain echo-only :func:`fleet_stalled_sweep_handler`.
    """
    if discover is None:
        return fleet_stalled_sweep_handler

    async def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        return await _fleet_stalled_sweep(payload, discover=discover)

    return handler


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

    Requires **explicit** repair targets (``repairs`` / ``repair_targets``) and
    fails closed otherwise — dispatching "nothing" would masquerade as a
    completed rescue. It deliberately does NOT fall back to the survey's
    ``stalled_items``: a recurring loop that forwards the observation stage's
    output must not let *detected* work become a *dispatched* target without
    fresh per-run approval selecting it (#2200). Records work strictly as
    *dispatched*; it must never infer ``merged``/``shipped`` state. That is what
    the downstream ``evidence_verify`` stage exists to establish.
    """
    targets = _as_list(
        payload.get("repairs")
        or payload.get("repair_targets")
    )
    if not targets:
        # A recurring observation-only tick reached dispatch with no per-run
        # approval having selected any target: complete cleanly as a no-op
        # rather than failing the unattended run (#2249). This does NOT relax
        # the fail-closed contract — a direct call still raises below, and even
        # a recurring tick never auto-dispatches detected ``stalled_items``
        # (they are not forwarded here; only explicit repair targets dispatch).
        if _is_recurring_tick(payload):
            return _recurring_skip(
                A2A_REPAIR_DISPATCH,
                "dispatched_count",
                "recurring observation-only tick: no approved repair targets "
                "selected this run",
            )
        raise ValueError(
            "a2a_repair_dispatch: no explicit repair targets supplied "
            "(repairs/repair_targets); refusing to dispatch. Detected "
            "stalled_items are not auto-dispatched — repair targets are "
            "selected per-run with fresh approval (fail closed)"
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
        # A recurring observation-only tick that dispatched nothing has nothing
        # to verify: complete cleanly as a no-op (#2249). But a recurring run
        # that DID select explicit repair targets and dispatched real work must
        # still be proven — otherwise the no-op branch would turn a genuine
        # irreversible dispatch into a completed run without evidence (#2249 P1).
        # A direct call is likewise unaffected and still fails closed.
        if _is_recurring_tick(payload) and not _run_selected_action(payload):
            return _recurring_skip(
                EVIDENCE_VERIFY,
                "verified_count",
                "recurring observation-only tick: no dispatched work to verify",
            )
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
        # A recurring observation-only tick that resolved nothing has nothing to
        # close: complete cleanly as a no-op (#2249). But a recurring run that
        # selected explicit repair targets / dispatched real work must not slip
        # past the close gate as a no-op — a real intervention has to reconcile
        # its own resolution evidence (#2249 P1). A direct call still fails
        # closed — a todo is never closed without upstream resolution evidence.
        if _is_recurring_tick(payload) and not _run_selected_action(payload):
            return _recurring_skip(
                CLOSE_RESOLVED_TODOS,
                "closed_count",
                "recurring observation-only tick: no resolved todos to close",
            )
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


def build_fleet_stalled_sweep_registration(
    discover: Optional[Any] = None,
) -> SourceRegistration:
    """Build the ``fleet_stalled_sweep`` source.

    ``discover`` is an optional ``async def discover(stale_days) -> list`` the
    host binds so a recurring tick surveys real stalled work instead of relying
    on pre-seeded ``stalled_items`` (#2200). Omit it for echo-only behavior.
    """
    return _action_registration(
        FLEET_STALLED_SWEEP, _make_fleet_stalled_sweep_handler(discover)
    )


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


def build_workflow_rescue_registrations(
    *, fleet_stalled_discover: Optional[Any] = None
) -> List[SourceRegistration]:
    """Every source the built-in ``stalled_work_rescue`` workflow references.

    ``fleet_stalled_discover`` is threaded to the ``fleet_stalled_sweep`` source
    so a host can bind live stalled-work discovery (#2200); the other sources
    are pure functions of their payload.
    """
    return [
        build_fleet_stalled_sweep_registration(fleet_stalled_discover),
        build_governance_review_registration(),
        build_a2a_repair_dispatch_registration(),
        build_evidence_verify_registration(),
        build_close_resolved_todos_registration(),
        build_reopen_resolved_todos_registration(),
    ]


def register_workflow_rescue_sources(
    registry: Any, *, fleet_stalled_discover: Optional[Any] = None
) -> List[str]:
    """Register the rescue sources on ``registry`` under the OPTIONAL policy.

    ``fleet_stalled_discover`` (optional ``async def(stale_days) -> list``) binds
    live stalled-work discovery onto ``fleet_stalled_sweep`` so a recurring tick
    observes real candidates rather than pre-seeded ones (#2200).

    These are feature sources whose absence is a degraded-but-tolerable state, so
    registration uses :attr:`RegistrationPolicy.OPTIONAL` (#2522): an equivalent
    re-registration is a no-op, a validation failure or a clash with a
    *non-equivalent* existing contract is reported (logged loudly) rather than
    silently equated, and nothing ever raises — one bad source cannot abort the
    rest. Returns the names *newly* registered by this call.
    """
    if registry is None or not hasattr(registry, "register_batch"):
        return []
    outcomes = registry.register_batch(
        build_workflow_rescue_registrations(
            fleet_stalled_discover=fleet_stalled_discover
        ),
        RegistrationPolicy.OPTIONAL,
    )
    return [
        outcome.name
        for outcome in outcomes
        if outcome.state is RegistrationState.REGISTERED
    ]


# --------------------------------------------------------------------------
# Safe recurring scheduling (#2200).
#
# ``stalled_work_rescue`` can complete a fully-evidenced control run, but
# scheduling the *all-stage* workflow as a recurring loop is only safe if the
# irreversible/evidence-gated stages (``dispatch_repairs``, ``close_resolved``)
# never fire off a blanket schedule. They must run per-run with fresh approval
# and fresh evidence.
#
# The recurring configuration below schedules the workflows feature's
# ``workflow_run`` tool against the built-in ``stalled_work_rescue`` definition
# with *observation-only* params: each tick detects stalled work and requests
# fresh consent at the ``govern_intent`` gate; dispatch/close only proceed once
# that per-run approval is granted and the evidence-gated handlers pass. The
# builder fails closed if a caller tries to pre-seed repair targets, resolution
# evidence, or a blanket approval marker into the recurring params.
# --------------------------------------------------------------------------

# The built-in workflow name and the schedulable feature-tool task that runs it.
RECURRING_WORKFLOW_NAME = "stalled_work_rescue"
RECURRING_SCHEDULE_TASK_NAME = "workflow_run"
# The built-in's own CRON trigger cadence (feature ``library.py``): every 6h.
RECURRING_DEFAULT_CRON = "0 */6 * * *"
# Consent scope the ``govern_intent`` stage gates on (must match the built-in).
RECURRING_CONSENT_SCOPE = "proactive_work_rescue"

# Params that must NEVER be baked into a recurring schedule: pre-seeding any of
# these pre-targets the irreversible/evidence-gated stages, turning an
# unattended recurring run into an auto-dispatch / auto-close. Repair targets
# and resolution evidence are supplied per-run, alongside fresh approval.
_PRESEEDED_ACTION_PARAM_KEYS = frozenset(
    {
        "repairs",  # a2a_repair_dispatch targets
        "repair_targets",  # a2a_repair_dispatch targets (alias)
        "resolved_todos",  # close_resolved_todos targets
        "evidence",  # evidence_verify / close_resolved evidence
    }
)
# Blanket approval markers a recurring schedule must never carry — consent is
# collected fresh per run by the workflow's ``consent_collect`` gate (#2198).
_BLANKET_APPROVAL_PARAM_KEYS = CONSENT_MARKER_FIELDS


class UnsafeRecurringScheduleError(ValueError):
    """Recurring-schedule params would enable an irreversible stage without
    fresh, per-run approval and evidence (fail closed)."""


def assert_safe_recurring_params(params: Any) -> Dict[str, Any]:
    """Fail closed unless ``params`` are safe to schedule as a recurring loop.

    Rejects params that pre-seed the irreversible/evidence-gated stages
    (repair targets, resolution evidence) or that carry a blanket approval
    marker — both must be supplied per-run with fresh consent, never baked into
    a recurring schedule. Returns the params unchanged when safe.
    """
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise UnsafeRecurringScheduleError(
            f"recurring params must be a dict, got {type(params).__name__}"
        )
    preseeded = sorted(k for k in _PRESEEDED_ACTION_PARAM_KEYS if params.get(k))
    if preseeded:
        raise UnsafeRecurringScheduleError(
            "recurring stalled_work_rescue schedule must not pre-seed "
            f"irreversible-stage targets {preseeded}; repair targets and "
            "resolution evidence are supplied per-run with fresh approval "
            "(fail closed)"
        )
    approvals = sorted(k for k in _BLANKET_APPROVAL_PARAM_KEYS if params.get(k))
    if approvals:
        raise UnsafeRecurringScheduleError(
            "recurring stalled_work_rescue schedule must not carry a blanket "
            f"approval marker {approvals}; consent is collected fresh per run "
            "by the govern_intent gate (fail closed)"
        )
    return params


def is_safe_recurring_params(params: Any) -> bool:
    """``True`` when :func:`assert_safe_recurring_params` accepts ``params``."""
    try:
        assert_safe_recurring_params(params)
    except UnsafeRecurringScheduleError:
        return False
    return True


def build_recurring_schedule_request(
    *,
    cron: Optional[str] = None,
    stale_days: int = 3,
    notify: Any = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the exact ``schedule_add`` invocation for a *safe* recurring loop.

    The returned dict maps 1:1 onto ``SchedulerFeature.schedule_add`` kwargs
    (``cron_expression``, ``task_name``, ``args_json``). The scheduled task is
    the workflows feature's ``workflow_run`` tool, started against the built-in
    ``stalled_work_rescue`` definition with observation-only params: it detects
    stalled work and requests fresh consent, so the irreversible dispatch/close
    stages only proceed once that per-run approval (and its evidence) is granted.

    Fails closed via :func:`assert_safe_recurring_params` if ``extra_params``
    tries to pre-seed repair targets, resolution evidence, or a blanket
    approval marker.
    """
    params: Dict[str, Any] = {"stale_days": int(stale_days), "recurring": True}
    if extra_params:
        params.update(extra_params)
    assert_safe_recurring_params(params)
    args: Dict[str, Any] = {"name": RECURRING_WORKFLOW_NAME, "params": params}
    if notify is not None:
        args["notify"] = notify
    return {
        "cron_expression": cron or RECURRING_DEFAULT_CRON,
        "task_name": RECURRING_SCHEDULE_TASK_NAME,
        "args_json": json.dumps(args, sort_keys=True),
    }
