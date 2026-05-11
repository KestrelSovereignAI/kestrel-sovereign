"""Workflow-level Prometheus counters.

SignalDispatcher already owns per-signal observability. These counters
roll up the workflow-only outcomes Phase 1 adds: gate decisions and
compensation state transitions. Labels are bounded by registered
workflow definitions and the closed outcome vocabularies.

Alert-tiering counters intentionally split compensation failure from
irreversible residue. A failed compensator needs intervention and pages;
``cancelled_with_irreversible_residue`` is expected for irreversible
stages and stays dashboard/log-only.
"""

from __future__ import annotations

import logging

from kestrel_sdk.metrics import PROMETHEUS_AVAILABLE

logger = logging.getLogger(__name__)


if PROMETHEUS_AVAILABLE:
    from prometheus_client import Counter

    from kestrel_sdk.metrics import REGISTRY

    WORKFLOW_GATE_OUTCOMES_TOTAL = Counter(
        "kestrel_workflow_gate_outcomes_total",
        "Workflow stage gate outcomes",
        ["workflow_name", "stage_name", "outcome"],
        registry=REGISTRY,
    )
    WORKFLOW_COMPENSATIONS_TOTAL = Counter(
        "kestrel_workflow_compensations_total",
        "Workflow stage compensation states",
        ["workflow_name", "stage_name", "state"],
        registry=REGISTRY,
    )
    WORKFLOW_COMPENSATE_FAILED_TOTAL = Counter(
        "kestrel_workflow_compensate_failed_total",
        "Workflow compensation failures that require operator intervention",
        ["workflow_name", "stage_name"],
        registry=REGISTRY,
    )
    WORKFLOW_CANCELLED_WITH_IRREVERSIBLE_RESIDUE_TOTAL = Counter(
        "kestrel_workflow_cancelled_with_irreversible_residue_total",
        "Workflow runs cancelled with irreversible residue; dashboard-only",
        ["workflow_name"],
        registry=REGISTRY,
    )
else:
    WORKFLOW_GATE_OUTCOMES_TOTAL = None  # type: ignore[assignment]
    WORKFLOW_COMPENSATIONS_TOTAL = None  # type: ignore[assignment]
    WORKFLOW_COMPENSATE_FAILED_TOTAL = None  # type: ignore[assignment]
    WORKFLOW_CANCELLED_WITH_IRREVERSIBLE_RESIDUE_TOTAL = None  # type: ignore[assignment]


def record_gate_outcome(
    workflow_name: str, stage_name: str, outcome: str
) -> None:
    if WORKFLOW_GATE_OUTCOMES_TOTAL is None:
        return
    try:
        WORKFLOW_GATE_OUTCOMES_TOTAL.labels(
            workflow_name=workflow_name,
            stage_name=stage_name,
            outcome=outcome,
        ).inc()
    except Exception:
        logger.exception(
            "Failed to record workflow gate metric "
            "workflow=%s stage=%s outcome=%s",
            workflow_name,
            stage_name,
            outcome,
        )


def record_compensation_state(
    workflow_name: str, stage_name: str, state: str
) -> None:
    if WORKFLOW_COMPENSATIONS_TOTAL is None:
        return
    try:
        WORKFLOW_COMPENSATIONS_TOTAL.labels(
            workflow_name=workflow_name,
            stage_name=stage_name,
            state=state,
        ).inc()
    except Exception:
        logger.exception(
            "Failed to record workflow compensation metric "
            "workflow=%s stage=%s state=%s",
            workflow_name,
            stage_name,
            state,
        )


def record_compensation_failed(workflow_name: str, stage_name: str) -> None:
    if WORKFLOW_COMPENSATE_FAILED_TOTAL is None:
        return
    try:
        WORKFLOW_COMPENSATE_FAILED_TOTAL.labels(
            workflow_name=workflow_name,
            stage_name=stage_name,
        ).inc()
    except Exception:
        logger.exception(
            "Failed to record workflow compensation-failed alert metric "
            "workflow=%s stage=%s",
            workflow_name,
            stage_name,
        )


def record_irreversible_residue(workflow_name: str) -> None:
    if WORKFLOW_CANCELLED_WITH_IRREVERSIBLE_RESIDUE_TOTAL is None:
        return
    try:
        WORKFLOW_CANCELLED_WITH_IRREVERSIBLE_RESIDUE_TOTAL.labels(
            workflow_name=workflow_name,
        ).inc()
    except Exception:
        logger.exception(
            "Failed to record workflow irreversible-residue metric "
            "workflow=%s",
            workflow_name,
        )


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "WORKFLOW_CANCELLED_WITH_IRREVERSIBLE_RESIDUE_TOTAL",
    "WORKFLOW_COMPENSATE_FAILED_TOTAL",
    "WORKFLOW_COMPENSATIONS_TOTAL",
    "WORKFLOW_GATE_OUTCOMES_TOTAL",
    "record_compensation_failed",
    "record_compensation_state",
    "record_gate_outcome",
    "record_irreversible_residue",
]
