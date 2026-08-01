"""Executable, explicitly provisioned Kite HTTP release workloads.

These workloads are intentionally not added to ``default_catalog_workloads``:
the caller must provide a factory for a fresh, loopback-only ``KiteHttpHarness``
with its explicit SQLite or disposable-PostgreSQL storage configuration.  This
keeps public CLI execution from selecting a listener, home, port, or database.
"""

from __future__ import annotations

from collections.abc import Callable
import math

from .kite_release_evidence import KiteGate, KiteHttpHarness
from .release_evidence import release_gate_specs
from .release_evidence_execution import CatalogWorkload, CatalogWorkloadResult
from .release_evidence_models import GateSpec, PerformanceMetric, ReleaseEvidenceError


KiteHarnessFactory = Callable[[KiteGate, str], KiteHttpHarness]

_LIVE_GATE_BY_ID = {
    "kite_http_stable_only_release_drill": KiteGate.STABLE_ONLY,
    "kite_http_experimental_enabled_release_drill": KiteGate.EXPERIMENTAL_ENABLED,
    "stable_persisted_data_no_canonical_migration_drill": KiteGate.PERSISTED_STABLE,
}


def _run_with_owned_harness(harness: KiteHttpHarness, callback):
    """Own the isolated lifecycle even when an evidence operation fails."""
    harness.prepare()
    harness.start()
    try:
        return callback(harness)
    finally:
        harness.stop()


def _live_workload(spec: GateSpec, factory: KiteHarnessFactory) -> CatalogWorkloadResult:
    gate = _LIVE_GATE_BY_ID.get(spec.gate_id)
    if gate is None:
        raise ReleaseEvidenceError("Kite live workload is not in the immutable catalog")
    observation = _run_with_owned_harness(factory(gate, "sqlite"), lambda harness: harness.run_release_gate())
    return CatalogWorkloadResult(observation=observation.to_mapping())


def _sleep_workload(spec: GateSpec, factory: KiteHarnessFactory) -> CatalogWorkloadResult:
    target = spec.performance_target
    if (
        target is None
        or target.metric not in {PerformanceMetric.CHANGED_WORK_SLEEP, PerformanceMetric.UNCHANGED_SLEEP}
        or target.mode != "kite_http"
    ):
        raise ReleaseEvidenceError("Kite sleep workload must match its immutable performance target")
    samples = _run_with_owned_harness(
        factory(KiteGate.STABLE_ONLY, target.backend),
        lambda harness: harness.measure_sleep(
            changed=target.metric is PerformanceMetric.CHANGED_WORK_SLEEP,
        ),
    )
    p95 = sorted(samples)[math.ceil(len(samples) * 0.95) - 1]
    return CatalogWorkloadResult(
        observation={"sample_count": len(samples), "p95_ms": p95},
        samples=samples,
    )


def kite_http_workloads(factory: KiteHarnessFactory) -> dict[tuple[str, str], CatalogWorkload]:
    """Return only real Kite HTTP operations for an explicitly provisioned run.

    Correlated erasure and external-adapter gates deliberately stay absent: a
    concrete core-surface probe plus independently operated external adapter is
    required before those claims can execute.  Their absence emits the normal
    content-free ``catalog_workload_unavailable`` block rather than a mock pass.
    """
    if not callable(factory):
        raise ReleaseEvidenceError("Kite workloads require an explicit harness factory")
    workloads: dict[tuple[str, str], CatalogWorkload] = {}
    for spec in release_gate_specs():
        if spec.gate_id in _LIVE_GATE_BY_ID:
            workloads[(spec.runner.runner_id, spec.runner.command_id)] = (
                lambda candidate, bound_factory=factory: _live_workload(candidate, bound_factory)
            )
        elif (
            spec.performance_target is not None
            and spec.performance_target.metric
            in {PerformanceMetric.CHANGED_WORK_SLEEP, PerformanceMetric.UNCHANGED_SLEEP}
        ):
            workloads[(spec.runner.runner_id, spec.runner.command_id)] = (
                lambda candidate, bound_factory=factory: _sleep_workload(candidate, bound_factory)
            )
    return workloads


__all__ = ["KiteHarnessFactory", "kite_http_workloads"]
