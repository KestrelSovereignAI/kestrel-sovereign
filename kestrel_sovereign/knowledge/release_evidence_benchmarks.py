"""Real, isolated semantic benchmark workloads for release evidence.

The public release-evidence CLI cannot select a database, pass a DSN, or
provide samples.  This module owns the immutable workload implementation.  A
PostgreSQL benchmark is deliberately opt-in through an explicitly named,
operator-owned *isolated* database configuration; the DSN is used only to
construct the test storage and is never retained in a result or artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from tempfile import TemporaryDirectory
from typing import Final
from uuid import uuid4

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_assertion_store import _issue_assertion_tenant_capability
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

from .release_evidence_execution import (
    CatalogWorkload,
    CatalogWorkloadResult,
    CatalogWorkloadUnavailable,
)
from .release_evidence_models import (
    BenchmarkRun,
    GateSpec,
    PerformanceMetric,
    ReleaseEvidenceError,
    SemanticBenchmarkHarness,
)
from .release_evidence_postgres import DisposablePostgresDatabase


_BENCHMARK_ITERATIONS: Final = 3
_BENCHMARK_VALUE_BYTES: Final = 1_048_576


@dataclass(slots=True)
class _StorageSample:
    storage: AsyncStorage
    sqlite_path: Path | None


class _IsolatedStorageFactory:
    """Create a fresh agent-scoped storage sample for each measurement.

    SQLite samples use a distinct temporary database.  PostgreSQL samples use
    distinct tenant scopes within the caller-created disposable database.
    """

    def __init__(self, backend: str, *, postgres_dsn: str | None = None) -> None:
        if backend not in {"sqlite", "postgres"}:
            raise ReleaseEvidenceError("semantic benchmark backend must be sqlite or postgres")
        self._backend = backend
        self._temporary_directory = TemporaryDirectory(prefix="kestrel-semantic-benchmark-")
        self._root = Path(self._temporary_directory.name)
        if backend == "postgres" and not postgres_dsn:
            raise ReleaseEvidenceError("postgres benchmark requires a generated disposable database")
        self._postgres_dsn = postgres_dsn

    async def open(self) -> _StorageSample:
        tenant = f"did:kestrel:semantic-evidence:{uuid4().hex}"
        capability = _issue_assertion_tenant_capability(tenant)
        if self._backend == "sqlite":
            database_path = self._root / f"{uuid4().hex}.db"
            storage = AsyncStorage(
                str(database_path),
                agent_id=tenant,
                _assertion_tenant_capability=capability,
            )
            await storage.initialize()
            return _StorageSample(storage, database_path)
        assert self._postgres_dsn is not None
        storage = AsyncStorage(
            backend="postgres",
            dsn=self._postgres_dsn,
            agent_id=tenant,
            _assertion_tenant_capability=capability,
        )
        await storage.initialize()
        return _StorageSample(storage, None)

    async def close(self, sample: _StorageSample | None) -> None:
        if sample is not None:
            await sample.storage.close()

    def dispose(self) -> None:
        self._temporary_directory.cleanup()


def _inference_profile():
    from kestrel_sovereign.knowledge import InferenceProfile, OntologyRef

    return InferenceProfile(
        OntologyRef(
            "http://www.w3.org/2000/01/rdf-schema#",
            "1.0.0",
            "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
            "semantic-kb-v1",
        ),
        "1.0.0",
    )


async def _save_governed_fact(sample: _StorageSample, sequence: int) -> None:
    storage = PrivacyEnforcingStorage(sample.storage, PrivacyMode.NORMAL)
    result = await storage.save_explicit_fact(
        subject="user",
        predicate="preferred_deploy_region",
        value=f"benchmark-region-{sequence}",
        confidence=0.9,
        invocation_id=f"semantic-release-benchmark-save-{sequence}",
    )
    if not result.saved:
        raise ReleaseEvidenceError("semantic benchmark governed assertion write was not accepted")


async def _run_maintenance_to_terminal(storage: AsyncStorage) -> None:
    profile = _inference_profile()
    for _ in range(4):
        result = await storage.run_semantic_maintenance(profile)
        if result.status.value in {"complete", "no_op"}:
            return
    raise ReleaseEvidenceError("semantic benchmark maintenance did not reach a terminal checkpoint")


async def _benchmark_operation(
    metric: PerformanceMetric,
    sample: _StorageSample,
    sequence: int,
) -> None:
    if metric is PerformanceMetric.STARTUP:
        return
    if metric is PerformanceMetric.ASSERTION_WRITE_VALIDATION:
        await _save_governed_fact(sample, sequence)
        return
    if metric is PerformanceMetric.BOUNDED_INFERENCE:
        await _save_governed_fact(sample, sequence)
        await sample.storage.materialize_semantic_inference(_inference_profile())
        return
    if metric is PerformanceMetric.HYBRID_RECALL:
        await _save_governed_fact(sample, sequence)
        await _run_maintenance_to_terminal(sample.storage)
        await sample.storage.semantic_recall_candidates(
            query="benchmark-region",
            candidate_scan_limit=10,
            inference_profile=_inference_profile(),
        )
        return
    if metric is PerformanceMetric.REPRESENTATIVE_MIGRATION:
        assert sample.storage.db is not None
        node_id = f"semantic-benchmark-legacy-{sequence}"
        await sample.storage.db.execute(
            "INSERT INTO graph_nodes (node_id, node_type, label, properties) VALUES (?, 'fact', ?, ?)",
            (
                node_id,
                "legacy fact",
                '{"subject":"user","predicate":"preferred_deploy_region",'
                '"value":"benchmark-region","created_at":"2026-01-01T00:00:00+00:00"}',
            ),
        )
        await sample.storage.db.execute(
            "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
            (node_id, sample.storage.agent_id),
        )
        result = await sample.storage.legacy_graph_fact_migration().run(batch_size=1)
        if result.migrated != 1:
            raise ReleaseEvidenceError("semantic benchmark migration did not migrate its isolated legacy fact")
        return
    if metric is PerformanceMetric.STORAGE_GROWTH:
        assert sample.storage.graph is not None
        await sample.storage.graph.add_node(
            GraphNode(
                node_id=f"semantic-benchmark-storage-{sequence}",
                node_type="semantic_benchmark_payload",
                label="semantic benchmark storage payload",
                properties={
                    "agent_id": sample.storage.agent_id,
                    "payload": "x" * _BENCHMARK_VALUE_BYTES,
                },
            )
        )
        return
    raise CatalogWorkloadUnavailable("kite_http_benchmark_runner_required")


async def _storage_bytes(sample: _StorageSample) -> int:
    if sample.sqlite_path is not None:
        try:
            # SQLite can keep a committed write in its WAL while the isolated
            # sample connection remains open.  The actual backend footprint is
            # therefore the database plus its active journal sidecars, never a
            # wall-clock proxy or only the stale main database file.
            paths = (
                sample.sqlite_path,
                Path(f"{sample.sqlite_path}-wal"),
                Path(f"{sample.sqlite_path}-shm"),
                Path(f"{sample.sqlite_path}-journal"),
            )
            return sum(path.stat().st_size for path in paths if path.exists())
        except OSError as error:
            raise ReleaseEvidenceError("isolated sqlite benchmark database is unavailable") from error
    assert sample.storage.db is not None
    value = await sample.storage.db.fetchval("SELECT pg_database_size(current_database())")
    if type(value) is not int or value < 0:
        raise ReleaseEvidenceError("isolated postgres benchmark did not return a database byte size")
    return value


async def _run_benchmark(spec: GateSpec) -> CatalogWorkloadResult:
    target = spec.performance_target
    if target is None:
        raise ReleaseEvidenceError("semantic benchmark workload requires a performance catalog gate")
    if target.metric in {PerformanceMetric.CHANGED_WORK_SLEEP, PerformanceMetric.UNCHANGED_SLEEP}:
        # Catalog mode is kite_http.  The real HTTP drill owns these metrics;
        # invoking maintenance directly would misrepresent the mode.
        raise CatalogWorkloadUnavailable("kite_http_benchmark_runner_required")
    disposable_database: DisposablePostgresDatabase | None = None
    if target.backend == "postgres":
        disposable_database = await DisposablePostgresDatabase.create()
    factory = _IsolatedStorageFactory(
        target.backend,
        postgres_dsn=disposable_database.dsn if disposable_database else None,
    )
    current: _StorageSample | None = None
    sequence = 0

    async def open_sample() -> None:
        nonlocal current, sequence
        if current is not None:
            raise ReleaseEvidenceError("semantic benchmark sample lifecycle is not isolated")
        sequence += 1
        current = await factory.open()

    async def operation() -> None:
        nonlocal current
        # Startup owns creation/initialization, so it opens only within the
        # timed operation.  Every other metric is prepared outside the timer.
        if target.metric is PerformanceMetric.STARTUP:
            await open_sample()
        assert current is not None
        try:
            await _benchmark_operation(target.metric, current, sequence)
        finally:
            if target.metric is not PerformanceMetric.STORAGE_GROWTH:
                await factory.close(current)

    async def prepare_storage_growth_sample() -> None:
        if target.metric is PerformanceMetric.STORAGE_GROWTH:
            await open_sample()

    async def read_storage_bytes() -> int:
        if current is None:
            raise ReleaseEvidenceError("semantic benchmark storage sample is not prepared")
        return await _storage_bytes(current)

    async def close_storage_growth_sample() -> None:
        nonlocal current
        await factory.close(current)
        current = None

    async def timed_operation() -> None:
        nonlocal current
        if target.metric is not PerformanceMetric.STARTUP:
            await open_sample()
        await operation()
        # The next iteration must not observe an old sample object.  A closed
        # SQLite path remains readable for byte observation; Postgres reads
        # happen before this assignment in the harness.
        if target.metric is not PerformanceMetric.STORAGE_GROWTH:
            current = None

    try:
        harness = SemanticBenchmarkHarness(iterations=_BENCHMARK_ITERATIONS)
        if target.metric is PerformanceMetric.STORAGE_GROWTH:
            # ``operation`` closes after the write.  The SQLite file remains
            # available to measure its true on-disk delta and the Postgres
            # database-size query is executed through the same isolated DB.
            run = await harness.run(
                spec,
                operation,
                storage_bytes=read_storage_bytes,
                before_sample=prepare_storage_growth_sample,
                after_sample=close_storage_growth_sample,
            )
        else:
            run = await harness.run(spec, timed_operation)
        return _benchmark_result(spec, run)
    finally:
        await factory.close(current)
        factory.dispose()
        if disposable_database is not None:
            await disposable_database.close()


def _benchmark_result(spec: GateSpec, run: BenchmarkRun) -> CatalogWorkloadResult:
    target = spec.performance_target
    assert target is not None
    p95 = sorted(run.samples)[math.ceil(len(run.samples) * 0.95) - 1]
    return CatalogWorkloadResult(
        observation={
            "sample_count": len(run.samples),
            "p95_bytes" if target.unit == "bytes" else "p95_ms": p95,
        },
        samples=run.samples,
    )


def semantic_benchmark_workloads() -> dict[tuple[str, str], CatalogWorkload]:
    """Resolve every immutable benchmark command to the real benchmark runner.

    The runner itself blocks the two Kite-mode metrics until their dedicated
    HTTP runner is available; this preserves their exact catalog mode rather
    than timing an in-process approximation.
    """
    from .release_evidence import performance_targets

    workloads: dict[tuple[str, str], CatalogWorkload] = {}
    for target in performance_targets():
        command_id = f"benchmark_{target.gate_suffix}_v1"

        async def workload(spec: GateSpec) -> CatalogWorkloadResult:
            return await _run_benchmark(spec)

        workloads[("semantic_benchmark", command_id)] = workload
    return workloads
