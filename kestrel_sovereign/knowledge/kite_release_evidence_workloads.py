"""Executable, explicitly provisioned Kite HTTP release workloads.

The core erasure runner is registered only because this module owns *both*
isolated storage backends.  It creates PostgreSQL through the acknowledged
disposable-database authority, runs a distinct loopback Kite process against
SQLite and that exact generated database, and emits an aggregate only after
both observations pass.  The public CLI still cannot select a listener, home,
port, DSN, or observation.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
import socket
import tempfile

from .kite_release_evidence import (
    KiteEvidenceError,
    KiteGate,
    KiteHttpHarness,
    KiteIsolationConfig,
    KiteStorageConfig,
)
from .release_evidence_execution import (
    CatalogWorkload,
    CatalogWorkloadResult,
    CatalogWorkloadUnavailable,
)
from .release_evidence_postgres import DisposablePostgresDatabase
from .release_evidence import release_gate_specs
from .release_evidence_models import GateSpec, PerformanceMetric, ReleaseEvidenceError
from .release_evidence_models import ErasureStage


KiteHarnessFactory = Callable[[KiteGate, KiteStorageConfig], KiteHttpHarness]

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


async def _run_with_owned_postgres_harness(harness: KiteHttpHarness, callback):
    """Seed the fresh test identity before starting an owned PostgreSQL host."""
    harness.prepare()
    try:
        seed_identity = getattr(harness, "seed_disposable_postgres_test_identity", None)
        if not callable(seed_identity):
            raise ReleaseEvidenceError("Kite PostgreSQL harness lacks the isolated identity seed")
        await seed_identity()
        harness.start()
        return callback(harness)
    finally:
        harness.stop()


def _live_workload(spec: GateSpec, factory: KiteHarnessFactory) -> CatalogWorkloadResult:
    gate = _LIVE_GATE_BY_ID.get(spec.gate_id)
    if gate is None:
        raise ReleaseEvidenceError("Kite live workload is not in the immutable catalog")
    observation = _run_with_owned_harness(
        factory(gate, KiteStorageConfig()), lambda harness: harness.run_release_gate()
    )
    return CatalogWorkloadResult(observation=observation.to_mapping())


def _sleep_workload(spec: GateSpec, factory: KiteHarnessFactory) -> CatalogWorkloadResult:
    target = spec.performance_target
    if (
        target is None
        or target.metric not in {PerformanceMetric.CHANGED_WORK_SLEEP, PerformanceMetric.UNCHANGED_SLEEP}
        or target.mode != "kite_http"
    ):
        raise ReleaseEvidenceError("Kite sleep workload must match its immutable performance target")
    if target.backend != "sqlite":
        # A performance runner has no database authority.  Do not let a
        # caller smuggle a PostgreSQL DSN through the old backend label.
        raise CatalogWorkloadUnavailable("kite_postgres_disposable_authority_required")
    samples = _run_with_owned_harness(
        factory(KiteGate.STABLE_ONLY, KiteStorageConfig()),
        lambda harness: harness.measure_sleep(
            changed=target.metric is PerformanceMetric.CHANGED_WORK_SLEEP,
        ),
    )
    p95 = sorted(samples)[math.ceil(len(samples) * 0.95) - 1]
    return CatalogWorkloadResult(
        observation={"sample_count": len(samples), "p95_ms": p95},
        samples=samples,
    )


async def _core_erasure_workload(
    spec: GateSpec, factory: KiteHarnessFactory
) -> CatalogWorkloadResult:
    """Drive a fixed core stage through separately owned SQLite and PostgreSQL.

    There is intentionally no SQLite-only path for this catalog gate: its
    immutable environment promises ``dual_backend``.  Database acquisition is
    first, so a missing acknowledgement, driver, admin endpoint, create, or
    drop failure produces a blocked/failed record rather than an SQLite pass.
    """
    prefix = "erasure_"
    if not spec.gate_id.startswith(prefix):
        raise ReleaseEvidenceError("Kite erasure workload requires an erasure gate")
    try:
        stage = ErasureStage(spec.gate_id[len(prefix):])
    except ValueError as error:
        raise ReleaseEvidenceError("Kite erasure workload has an unknown stage") from error
    if stage is ErasureStage.SERVED_ADAPTER_ELIGIBILITY:
        raise ReleaseEvidenceError("served-adapter evidence belongs to parametric-self")
    if spec.environment.backend != "dual_backend" or spec.environment.mode != "kite_http":
        raise ReleaseEvidenceError(
            "core Kite erasure workload requires the immutable dual-backend HTTP spec"
        )

    # ``create`` is the sole PostgreSQL entry point.  It rejects ambient DSNs
    # and an unacknowledged admin channel before a harness receives any config.
    database = await DisposablePostgresDatabase.create()
    async with database:
        sqlite_observation = _run_with_owned_harness(
            factory(KiteGate.STABLE_ONLY, KiteStorageConfig()),
            lambda harness: harness.core_erasure_stage(stage),
        )
        postgres_observation = await _run_with_owned_postgres_harness(
            factory(
                KiteGate.STABLE_ONLY,
                KiteStorageConfig(backend="postgres", disposable_postgres=database),
            ),
            lambda harness: harness.core_erasure_stage(stage),
        )

    # SurfaceErasureObservation validates each backend independently.  Keep
    # the public record content-free: only the aggregate schema fields escape.
    erased_count = sqlite_observation.erased_count + postgres_observation.erased_count
    remaining_count = sqlite_observation.remaining_count + postgres_observation.remaining_count
    if erased_count <= 0 or remaining_count != 0:
        raise CatalogWorkloadUnavailable("kite_dual_backend_erasure_invalid")
    return CatalogWorkloadResult(
        observation={"erased_count": erased_count, "remaining_count": remaining_count}
    )


def kite_http_workloads(factory: KiteHarnessFactory) -> dict[tuple[str, str], CatalogWorkload]:
    """Return only real Kite HTTP operations for an explicitly provisioned run.

    Core erasure stages execute only through the server-owned typed drill.  The
    serving-adapter stage remains absent because core may not import or imitate
    the optional parametric-self feature's independent evidence.
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
        elif (
            spec.category == "erasure"
            and spec.owner == "kestrel_core"
            and spec.runner.runner_id == "kite_http"
        ):
            workloads[(spec.runner.runner_id, spec.runner.command_id)] = (
                lambda candidate, bound_factory=factory: _core_erasure_workload(candidate, bound_factory)
            )
    return workloads


def _owned_catalog_harness(gate: KiteGate, storage: KiteStorageConfig) -> KiteHttpHarness:
    """Create the one local-only harness accepted by the immutable catalog.

    This deliberately derives its worktree, fresh home, and loopback port in
    package code.  The public CLI receives no listener, storage, observation,
    or arbitrary command configuration.  PostgreSQL must arrive as the live
    authority-created object, never a caller-provided DSN.
    """
    if not isinstance(storage, KiteStorageConfig):
        raise KiteEvidenceError("catalog Kite harness requires a typed storage configuration")
    worktree = Path(__file__).resolve().parents[2]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = int(reservation.getsockname()[1])
    # ``prepare`` requires the actual home to be absent so it can reject a
    # reused agent state.  Keep a separate private parent for later forensic
    # inspection of the owned process's content-free log and marker.
    home = Path(tempfile.mkdtemp(prefix="kestrel-kite-release-")) / "home"
    return KiteHttpHarness(
        KiteIsolationConfig(worktree=worktree, home=home, port=port, gate=gate, storage=storage)
    )


def owned_kite_http_workloads() -> dict[tuple[str, str], CatalogWorkload]:
    """Return only the core erasure work owned by the no-config catalog.

    The earlier live-recall/sleep runners remain deliberately unregistered
    pending their separate HTTP readiness review.  This prevents a catalog
    expansion from silently promoting an unrelated old scaffold to release
    evidence merely because the erasure authority is now available.
    """
    return {
        key: workload
        for key, workload in kite_http_workloads(_owned_catalog_harness).items()
        if key[1].startswith("erasure_")
    }


__all__ = ["KiteHarnessFactory", "kite_http_workloads", "owned_kite_http_workloads"]
