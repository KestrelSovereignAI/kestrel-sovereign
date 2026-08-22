"""Executable, explicitly provisioned Kite HTTP release workloads.

The core live and erasure runners are registered only because this module owns
*both* isolated storage backends. It creates PostgreSQL through the
acknowledged disposable-database authority, runs a distinct loopback Kite
process against SQLite and that exact generated database, and emits an
aggregate only after both observations pass. Sleep runs the exact backend
named by its immutable target. The public CLI still cannot select a listener,
home, port, DSN, or observation.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
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
logger = logging.getLogger(__name__)

_LIVE_GATE_BY_ID = {
    "kite_http_stable_only_release_drill": KiteGate.STABLE_ONLY,
    "kite_http_experimental_enabled_release_drill": KiteGate.EXPERIMENTAL_ENABLED,
    "stable_persisted_data_no_canonical_migration_drill": KiteGate.PERSISTED_STABLE,
}


def _dispose_owned_harness(harness: KiteHttpHarness) -> None:
    """Remove a catalog-created home without requiring fake harnesses to know it."""
    cleanup = getattr(harness, "cleanup_owned_home", None)
    if callable(cleanup):
        cleanup()


def _log_ephemeral_harness_cleanup(harness: KiteHttpHarness) -> None:
    """Emit only catalog metadata when an ephemeral run fails."""
    config = getattr(harness, "config", None)
    gate = getattr(getattr(config, "gate", None), "value", "unknown")
    backend = getattr(getattr(config, "storage", None), "backend", "unknown")
    logger.info(
        "Kite catalog harness failed; removing ephemeral home",
        extra={"kite_gate": gate, "kite_backend": backend},
    )


def _run_with_owned_harness(harness: KiteHttpHarness, callback):
    """Own the isolated lifecycle even when an evidence operation fails."""
    try:
        harness.prepare()
        harness.start()
        return callback(harness)
    except BaseException:
        _log_ephemeral_harness_cleanup(harness)
        raise
    finally:
        try:
            harness.stop()
        finally:
            _dispose_owned_harness(harness)


async def _run_with_owned_postgres_harness(harness: KiteHttpHarness, callback):
    """Seed the fresh test identity before starting an owned PostgreSQL host."""
    try:
        harness.prepare()
        seed_identity = getattr(harness, "seed_disposable_postgres_test_identity", None)
        if not callable(seed_identity):
            raise ReleaseEvidenceError("Kite PostgreSQL harness lacks the isolated identity seed")
        await seed_identity()
        harness.start()
        return callback(harness)
    except BaseException:
        _log_ephemeral_harness_cleanup(harness)
        raise
    finally:
        try:
            harness.stop()
        finally:
            _dispose_owned_harness(harness)


async def _dual_backend_observations(
    factory: KiteHarnessFactory,
    gate: KiteGate,
    callback,
):
    """Run distinct SQLite and authority-created PostgreSQL Kite processes."""
    database = await DisposablePostgresDatabase.create()
    async with database:
        sqlite_observation = _run_with_owned_harness(
            factory(gate, KiteStorageConfig()), callback
        )
        postgres_observation = await _run_with_owned_postgres_harness(
            factory(
                gate,
                KiteStorageConfig(backend="postgres", disposable_postgres=database),
            ),
            callback,
        )
    return sqlite_observation, postgres_observation


def _aggregate_backend_mappings(
    sqlite_observation: dict[str, int], postgres_observation: dict[str, int]
) -> dict[str, int]:
    """Emit only schema fields after both backend observations agree exactly."""
    if set(sqlite_observation) != set(postgres_observation):
        raise ReleaseEvidenceError("Kite backend observations exposed different aggregate fields")
    aggregate: dict[str, int] = {}
    for key in sqlite_observation:
        left, right = sqlite_observation[key], postgres_observation[key]
        if type(left) is not int or type(right) is not int or left < 0 or right < 0:
            raise ReleaseEvidenceError(
                "Kite backend observation must contain non-negative integer counts"
            )
        aggregate[key] = left + right
    return aggregate


async def _live_workload(
    spec: GateSpec, factory: KiteHarnessFactory
) -> CatalogWorkloadResult:
    gate = _LIVE_GATE_BY_ID.get(spec.gate_id)
    if gate is None:
        raise ReleaseEvidenceError("Kite live workload is not in the immutable catalog")
    expected_mode = {
        KiteGate.STABLE_ONLY: "kite_http_stable",
        KiteGate.EXPERIMENTAL_ENABLED: "kite_http_experimental",
        KiteGate.PERSISTED_STABLE: "kite_http_persisted",
    }[gate]
    if spec.environment.backend != "dual_backend" or spec.environment.mode != expected_mode:
        raise ReleaseEvidenceError(
            "Kite live workload requires its immutable dual-backend catalog contract"
        )
    sqlite_observation, postgres_observation = await _dual_backend_observations(
        factory,
        gate,
        lambda harness: harness.run_release_gate(),
    )
    return CatalogWorkloadResult(
        observation=_aggregate_backend_mappings(
            sqlite_observation.to_mapping(), postgres_observation.to_mapping()
        )
    )


async def _sleep_workload(
    spec: GateSpec, factory: KiteHarnessFactory
) -> CatalogWorkloadResult:
    target = spec.performance_target
    if (
        target is None
        or target.metric not in {PerformanceMetric.CHANGED_WORK_SLEEP, PerformanceMetric.UNCHANGED_SLEEP}
        or target.mode != "kite_http"
    ):
        raise ReleaseEvidenceError("Kite sleep workload must match its immutable performance target")
    callback = lambda harness: harness.measure_sleep(
        changed=target.metric is PerformanceMetric.CHANGED_WORK_SLEEP,
    )
    if target.backend == "sqlite":
        samples = _run_with_owned_harness(
            factory(KiteGate.STABLE_ONLY, KiteStorageConfig()), callback
        )
    elif target.backend == "postgres":
        database = await DisposablePostgresDatabase.create()
        async with database:
            samples = await _run_with_owned_postgres_harness(
                factory(
                    KiteGate.STABLE_ONLY,
                    KiteStorageConfig(backend="postgres", disposable_postgres=database),
                ),
                callback,
            )
    else:
        raise ReleaseEvidenceError("Kite sleep workload has an unsupported catalog backend")
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
    sqlite_observation, postgres_observation = await _dual_backend_observations(
        factory,
        KiteGate.STABLE_ONLY,
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


class _OwnedCatalogKiteHttpHarness(KiteHttpHarness):
    """A harness whose fresh parent is removed after every catalog attempt."""

    def __init__(self, config: KiteIsolationConfig, temporary_root) -> None:
        super().__init__(config)
        self.__temporary_root = temporary_root

    def cleanup_owned_home(self) -> None:
        """Erase the isolated home, including its DB, keys, nonce ledger, and log."""
        self.__temporary_root.cleanup()


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
    # reused agent state. The catalog owns its parent and removes it in a
    # finally block, including the transient SQLite DB, key, nonce ledger, and
    # child log. Failures retain only the sanitized lifecycle log above.
    temporary_root = tempfile.TemporaryDirectory(prefix="kestrel-kite-release-")
    home = Path(temporary_root.name) / "home"
    try:
        config = KiteIsolationConfig(
            worktree=worktree, home=home, port=port, gate=gate, storage=storage,
        )
        return _OwnedCatalogKiteHttpHarness(config, temporary_root)
    except BaseException:
        temporary_root.cleanup()
        raise


def owned_kite_http_workloads() -> dict[tuple[str, str], CatalogWorkload]:
    """Register every core-owned, no-config Kite catalog workload.

    The immutable catalog chooses the gate/backend.  This factory supplies
    only fresh loopback process state and a core-created disposable database;
    no CLI argument can turn a live/sleep/erasure record into a synthetic pass.
    """
    return kite_http_workloads(_owned_catalog_harness)


__all__ = ["KiteHarnessFactory", "kite_http_workloads", "owned_kite_http_workloads"]
