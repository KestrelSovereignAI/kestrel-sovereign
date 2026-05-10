"""Workflow-level Prometheus counters.

SignalDispatcher already owns per-signal observability. These counters
roll up the workflow-only outcomes Phase 1 adds: gate decisions and
compensation state transitions. Labels are bounded by registered
workflow definitions and the closed outcome vocabularies.
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
else:
    WORKFLOW_GATE_OUTCOMES_TOTAL = None  # type: ignore[assignment]
    WORKFLOW_COMPENSATIONS_TOTAL = None  # type: ignore[assignment]


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


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "WORKFLOW_COMPENSATIONS_TOTAL",
    "WORKFLOW_GATE_OUTCOMES_TOTAL",
    "record_compensation_state",
    "record_gate_outcome",
]
