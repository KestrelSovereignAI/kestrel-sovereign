"""Workflow Prometheus metric wrappers."""

from __future__ import annotations

import pytest

from kestrel_sovereign.features.workflows import metrics


def test_workflow_metric_recorders_noop_when_handles_missing(monkeypatch):
    monkeypatch.setattr(metrics, "WORKFLOW_GATE_OUTCOMES_TOTAL", None)
    monkeypatch.setattr(metrics, "WORKFLOW_COMPENSATIONS_TOTAL", None)

    metrics.record_gate_outcome("release", "lint", "pass")
    metrics.record_compensation_state("release", "lint", "complete")


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE,
    reason="prometheus-client not installed",
)
def test_workflow_gate_metric_increments_when_prometheus_available():
    from kestrel_sdk.metrics import REGISTRY

    before = REGISTRY.get_sample_value(
        "kestrel_workflow_gate_outcomes_total",
        {
            "workflow_name": "release_metrics",
            "stage_name": "lint",
            "outcome": "pass",
        },
    ) or 0.0

    metrics.record_gate_outcome("release_metrics", "lint", "pass")

    after = REGISTRY.get_sample_value(
        "kestrel_workflow_gate_outcomes_total",
        {
            "workflow_name": "release_metrics",
            "stage_name": "lint",
            "outcome": "pass",
        },
    )
    assert after == before + 1


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE,
    reason="prometheus-client not installed",
)
def test_workflow_compensation_metric_increments_when_prometheus_available():
    from kestrel_sdk.metrics import REGISTRY

    before = REGISTRY.get_sample_value(
        "kestrel_workflow_compensations_total",
        {
            "workflow_name": "release_metrics",
            "stage_name": "lint",
            "state": "complete",
        },
    ) or 0.0

    metrics.record_compensation_state("release_metrics", "lint", "complete")

    after = REGISTRY.get_sample_value(
        "kestrel_workflow_compensations_total",
        {
            "workflow_name": "release_metrics",
            "stage_name": "lint",
            "state": "complete",
        },
    )
    assert after == before + 1
