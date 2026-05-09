"""Tests for the constitutional-injection Prometheus counters.

Pin the counter contract documented in
`kestrel_sovereign/signals/constitution_metrics.py`. The counters
are no-ops when `prometheus-client` is not installed; with it
installed (the test extra includes it), the counters are real
Prometheus handles bucketed by `source` label.

kestrel-sovereign#1137 chunk 1H.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.signals import constitution_metrics


def _counter_value(metric, **labels) -> float:
    """Read a single labeled Counter sample from the registry. Returns
    0.0 for the no-op case (prometheus-client absent)."""
    if metric is None:
        return 0.0
    # Counters expose `_value` on each labeled child; reach in via the
    # public collect API to avoid private-attr coupling.
    for sample in metric.collect()[0].samples:
        if sample.name.endswith("_total") and sample.labels == labels:
            return sample.value
    return 0.0


@pytest.mark.skipif(
    not constitution_metrics.PROMETHEUS_AVAILABLE,
    reason="prometheus-client not installed",
)
def test_record_echo_verified_increments_per_source():
    metric = constitution_metrics.CONSTITUTION_ECHO_VERIFIED_TOTAL
    before = _counter_value(metric, source="src_a")
    constitution_metrics.record_echo_verified("src_a")
    constitution_metrics.record_echo_verified("src_a")
    after = _counter_value(metric, source="src_a")
    assert after - before == 2


@pytest.mark.skipif(
    not constitution_metrics.PROMETHEUS_AVAILABLE,
    reason="prometheus-client not installed",
)
def test_record_echo_missing_independent_per_source():
    metric = constitution_metrics.CONSTITUTION_ECHO_MISSING_TOTAL
    before_a = _counter_value(metric, source="src_a")
    before_b = _counter_value(metric, source="src_b")
    constitution_metrics.record_echo_missing("src_a")
    constitution_metrics.record_echo_missing("src_b")
    constitution_metrics.record_echo_missing("src_b")
    after_a = _counter_value(metric, source="src_a")
    after_b = _counter_value(metric, source="src_b")
    assert after_a - before_a == 1
    assert after_b - before_b == 2


@pytest.mark.skipif(
    not constitution_metrics.PROMETHEUS_AVAILABLE,
    reason="prometheus-client not installed",
)
def test_record_doctrine_bundle_drift():
    metric = constitution_metrics.DOCTRINE_BUNDLE_DRIFT_TOTAL
    before = _counter_value(metric, source="src_drift")
    constitution_metrics.record_doctrine_bundle_drift("src_drift")
    after = _counter_value(metric, source="src_drift")
    assert after - before == 1


def test_record_calls_no_op_when_prometheus_unavailable(monkeypatch):
    """With the metric handles set to None (simulating no
    prometheus-client), record_* must NOT raise."""
    monkeypatch.setattr(
        constitution_metrics, "CONSTITUTION_ECHO_VERIFIED_TOTAL", None
    )
    monkeypatch.setattr(
        constitution_metrics, "CONSTITUTION_ECHO_MISSING_TOTAL", None
    )
    monkeypatch.setattr(
        constitution_metrics, "DOCTRINE_BUNDLE_DRIFT_TOTAL", None
    )
    # All three calls succeed silently.
    constitution_metrics.record_echo_verified("any")
    constitution_metrics.record_echo_missing("any")
    constitution_metrics.record_doctrine_bundle_drift("any")


def test_metric_names_match_design_spec():
    """The design specifies exact metric names; an accidental rename
    would break operator dashboards. Pin them."""
    if constitution_metrics.PROMETHEUS_AVAILABLE:
        metrics = [
            constitution_metrics.CONSTITUTION_ECHO_VERIFIED_TOTAL,
            constitution_metrics.CONSTITUTION_ECHO_MISSING_TOTAL,
            constitution_metrics.DOCTRINE_BUNDLE_DRIFT_TOTAL,
        ]
        names = sorted(m._name for m in metrics if m is not None)
        assert names == sorted(
            [
                "kestrel_constitution_echo_verified",
                "kestrel_constitution_echo_missing",
                "kestrel_doctrine_bundle_drift",
            ]
        )
