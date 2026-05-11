"""Workflow Prometheus metric wrappers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kestrel_sovereign.features.workflows import metrics


REPO_ROOT = Path(__file__).resolve().parents[2]


def _promql(expr: str) -> str:
    return " ".join(expr.split())


def test_workflow_metric_recorders_noop_when_handles_missing(monkeypatch):
    monkeypatch.setattr(metrics, "WORKFLOW_GATE_OUTCOMES_TOTAL", None)
    monkeypatch.setattr(metrics, "WORKFLOW_COMPENSATIONS_TOTAL", None)
    monkeypatch.setattr(metrics, "WORKFLOW_COMPENSATE_FAILED_TOTAL", None)
    monkeypatch.setattr(
        metrics,
        "WORKFLOW_CANCELLED_WITH_IRREVERSIBLE_RESIDUE_TOTAL",
        None,
    )

    metrics.record_gate_outcome("release", "lint", "pass")
    metrics.record_compensation_state("release", "lint", "complete")
    metrics.record_compensation_failed("release", "lint")
    metrics.record_irreversible_residue("release")


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


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE,
    reason="prometheus-client not installed",
)
def test_workflow_compensate_failed_metric_increments_when_prometheus_available():
    from kestrel_sdk.metrics import REGISTRY

    before = REGISTRY.get_sample_value(
        "kestrel_workflow_compensate_failed_total",
        {
            "workflow_name": "release_metrics",
            "stage_name": "rollback",
        },
    ) or 0.0

    metrics.record_compensation_failed("release_metrics", "rollback")

    after = REGISTRY.get_sample_value(
        "kestrel_workflow_compensate_failed_total",
        {
            "workflow_name": "release_metrics",
            "stage_name": "rollback",
        },
    )
    assert after == before + 1


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE,
    reason="prometheus-client not installed",
)
def test_workflow_irreversible_residue_metric_increments_when_prometheus_available():
    from kestrel_sdk.metrics import REGISTRY

    before = REGISTRY.get_sample_value(
        "kestrel_workflow_cancelled_with_irreversible_residue_total",
        {"workflow_name": "release_metrics"},
    ) or 0.0

    metrics.record_irreversible_residue("release_metrics")

    after = REGISTRY.get_sample_value(
        "kestrel_workflow_cancelled_with_irreversible_residue_total",
        {"workflow_name": "release_metrics"},
    )
    assert after == before + 1


def test_workflow_alert_metric_names_match_issue_1143_contract():
    if metrics.PROMETHEUS_AVAILABLE:
        assert (
            metrics.WORKFLOW_COMPENSATE_FAILED_TOTAL._name
            == "kestrel_workflow_compensate_failed"
        )
        assert (
            metrics.WORKFLOW_CANCELLED_WITH_IRREVERSIBLE_RESIDUE_TOTAL._name
            == "kestrel_workflow_cancelled_with_irreversible_residue"
        )


def test_workflow_prometheus_alert_rules_separate_page_from_dashboard_only():
    rules_path = REPO_ROOT / "docs/deployment/prometheus-workflows-alerts.yml"
    data = yaml.safe_load(rules_path.read_text())
    rules = data["groups"][0]["rules"]
    by_alert = {rule["alert"]: rule for rule in rules if "alert" in rule}
    by_record = {rule["record"]: rule for rule in rules if "record" in rule}

    assert by_record["workflow_compensate_failed_total"]["expr"] == (
        "sum(kestrel_workflow_compensate_failed_total) or vector(0)"
    )
    assert by_record["workflow_cancelled_with_irreversible_residue_total"][
        "expr"
    ] == (
        "sum(kestrel_workflow_cancelled_with_irreversible_residue_total) "
        "or vector(0)"
    )

    page = by_alert["KestrelWorkflowCompensateFailedPage"]
    assert _promql(page["expr"]) == _promql(
        """
        (
          (sum(increase(kestrel_workflow_compensate_failed_total[5m])) or vector(0))
          +
          (sum(kestrel_workflow_compensate_failed_total unless kestrel_workflow_compensate_failed_total offset 5m) or vector(0))
        ) > 0
        """
    )
    assert page["labels"]["severity"] == "page"
    assert page["labels"]["pages"] == "true"

    residue = by_alert["KestrelWorkflowIrreversibleResidueDashboardOnly"]
    assert _promql(residue["expr"]) == _promql(
        """
        (
          (sum(increase(kestrel_workflow_cancelled_with_irreversible_residue_total[15m])) or vector(0))
          +
          (sum(kestrel_workflow_cancelled_with_irreversible_residue_total unless kestrel_workflow_cancelled_with_irreversible_residue_total offset 15m) or vector(0))
        ) > 0
        """
    )
    assert residue["labels"]["severity"] == "info"
    assert residue["labels"]["pages"] == "false"
