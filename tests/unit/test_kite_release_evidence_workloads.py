"""Dual-backend ownership contracts for the registered Kite erasure runner."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kestrel_sovereign.knowledge import kite_release_evidence_workloads as workloads
from kestrel_sovereign.knowledge.kite_release_evidence import (
    KiteEvidenceError,
    KiteAggregateObservation,
    KiteGate,
    KiteIsolationConfig,
    KiteStorageConfig,
    SurfaceErasureObservation,
)
from kestrel_sovereign.knowledge.release_evidence import erasure_drill_binding, release_gate_specs
from kestrel_sovereign.knowledge.release_evidence_execution import (
    CatalogExecutionAuthority,
    CatalogSigningIdentity,
    CatalogWorkloadUnavailable,
)
from kestrel_sovereign.knowledge.release_evidence_models import (
    ErasureStage,
    EvidenceState,
    ExecutionEnvironment,
    ReleaseEvidenceError,
)
from kestrel_sovereign.knowledge.release_evidence_postgres import DisposablePostgresDatabase


def _gate(gate_id: str):
    return next(spec for spec in release_gate_specs() if spec.gate_id == gate_id)


class _AdminConnection:
    """Small asyncpg-shaped connection that records verified disposal."""

    def __init__(self) -> None:
        self.closed = False
        self.commands: list[str] = []

    async def execute(self, query: str, *_args: object) -> None:
        self.commands.append(query)

    async def fetchval(self, _query: str, *_args: object) -> bool:
        return False

    async def close(self) -> None:
        self.closed = True


def _database() -> tuple[DisposablePostgresDatabase, _AdminConnection]:
    connection = _AdminConnection()
    database = DisposablePostgresDatabase(
        dsn="postgresql://isolated/kestrel_semantic_release_0123456789abcdef0123456789abcdef",
        database_name="kestrel_semantic_release_0123456789abcdef0123456789abcdef",
        _admin_dsn="postgresql://isolated/postgres",
        _connection=connection,
    )
    return database, connection


class _Harness:
    def __init__(
        self,
        stage: ErasureStage,
        gate: KiteGate,
        calls: list[str],
        backend: str,
        *,
        fail: bool = False,
    ) -> None:
        self._stage = stage
        self._gate = gate
        self._calls = calls
        self._backend = backend
        self._fail = fail

    def prepare(self) -> None:
        self._calls.append(f"prepare:{self._backend}")

    def start(self) -> None:
        self._calls.append(f"start:{self._backend}")

    def stop(self) -> None:
        self._calls.append(f"stop:{self._backend}")

    async def seed_disposable_postgres_test_identity(self) -> None:
        assert self._backend == "postgres"
        self._calls.append("seed:postgres")

    def core_erasure_stage(self, stage: ErasureStage) -> SurfaceErasureObservation:
        assert stage is self._stage
        self._calls.append(f"stage:{self._backend}")
        if self._fail:
            raise KiteEvidenceError("postgres harness failed")
        return SurfaceErasureObservation(stage, 1, 0, erasure_drill_binding())

    def run_release_gate(self) -> KiteAggregateObservation:
        self._calls.append(f"live:{self._backend}")
        if self._gate is KiteGate.PERSISTED_STABLE:
            return KiteAggregateObservation(
                gate_id=self._gate.value,
                invoke_count=1,
                scenario_count=1,
                persisted_assertion_count=1,
            )
        if self._gate is KiteGate.EXPERIMENTAL_ENABLED:
            return KiteAggregateObservation(
                gate_id=self._gate.value,
                invoke_count=1,
                scenario_count=1,
                experimental_selection_count=1,
            )
        return KiteAggregateObservation(
            gate_id=self._gate.value,
            invoke_count=1,
            scenario_count=1,
            provenance_check_count=1,
        )

    def measure_sleep(self, *, changed: bool) -> tuple[float, ...]:
        self._calls.append(f"sleep:{self._backend}:{changed}")
        return (1.0, 2.0, 3.0)


def test_core_erasure_runs_both_owned_backends_and_closes_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, connection = _database()

    async def create() -> DisposablePostgresDatabase:
        return database

    monkeypatch.setattr(DisposablePostgresDatabase, "create", staticmethod(create))
    received: list[KiteStorageConfig] = []
    calls: list[str] = []
    spec = _gate("erasure_active_assertions")

    def factory(gate: KiteGate, storage: KiteStorageConfig) -> _Harness:
        received.append(storage)
        return _Harness(ErasureStage.ACTIVE_ASSERTIONS, gate, calls, storage.backend)

    result = asyncio.run(workloads._core_erasure_workload(spec, factory))

    assert result.observation == {"erased_count": 2, "remaining_count": 0}
    assert [storage.backend for storage in received] == ["sqlite", "postgres"]
    assert received[1].disposable_postgres is database
    assert calls == [
        "prepare:sqlite", "start:sqlite", "stage:sqlite", "stop:sqlite",
        "prepare:postgres", "seed:postgres", "start:postgres", "stage:postgres", "stop:postgres",
    ]
    assert database._closed is True
    assert connection.closed is True
    assert any(command.startswith("DROP DATABASE") for command in connection.commands)


def test_core_erasure_closes_disposable_database_when_postgres_harness_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, connection = _database()

    async def create() -> DisposablePostgresDatabase:
        return database

    monkeypatch.setattr(DisposablePostgresDatabase, "create", staticmethod(create))
    calls: list[str] = []

    def factory(gate: KiteGate, storage: KiteStorageConfig) -> _Harness:
        return _Harness(
            ErasureStage.ACTIVE_ASSERTIONS,
            gate,
            calls,
            storage.backend,
            fail=storage.backend == "postgres",
        )

    with pytest.raises(KiteEvidenceError, match="postgres harness failed"):
        asyncio.run(workloads._core_erasure_workload(_gate("erasure_active_assertions"), factory))

    assert calls[-1] == "stop:postgres"
    assert database._closed is True
    assert connection.closed is True


def test_core_erasure_requires_postgres_before_starting_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable() -> DisposablePostgresDatabase:
        raise CatalogWorkloadUnavailable("isolated_postgres_admin_unavailable")

    monkeypatch.setattr(DisposablePostgresDatabase, "create", staticmethod(unavailable))
    factory_called = False

    def factory(_gate: KiteGate, _storage: KiteStorageConfig) -> _Harness:
        nonlocal factory_called
        factory_called = True
        raise AssertionError("a harness must not start without PostgreSQL")

    identity = CatalogSigningIdentity(
        issuer_id="test_ci",
        key_id="kite_dual_backend",
        private_key=Ed25519PrivateKey.from_private_bytes(bytes([2]) * 32),
    )
    execution = asyncio.run(
        CatalogExecutionAuthority(
            identity,
            {
                ("kite_http", "erasure_active_assertions_v1"):
                lambda spec: workloads._core_erasure_workload(spec, factory)
            },
        ).execute(_gate("erasure_active_assertions"))
    )

    assert execution.record.state is EvidenceState.BLOCKED
    assert execution.record.reason_code == "isolated_postgres_admin_unavailable"
    assert factory_called is False


def test_live_workload_requires_and_aggregates_both_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, connection = _database()

    async def create() -> DisposablePostgresDatabase:
        return database

    monkeypatch.setattr(DisposablePostgresDatabase, "create", staticmethod(create))
    received: list[KiteStorageConfig] = []
    calls: list[str] = []

    def factory(gate: KiteGate, storage: KiteStorageConfig) -> _Harness:
        received.append(storage)
        return _Harness(ErasureStage.ACTIVE_ASSERTIONS, gate, calls, storage.backend)

    result = asyncio.run(
        workloads._live_workload(_gate("kite_http_stable_only_release_drill"), factory)
    )

    assert result.observation == {
        "invoke_count": 2,
        "scenario_count": 2,
        "provenance_check_count": 2,
    }
    assert [storage.backend for storage in received] == ["sqlite", "postgres"]
    assert calls == [
        "prepare:sqlite", "start:sqlite", "live:sqlite", "stop:sqlite",
        "prepare:postgres", "seed:postgres", "start:postgres", "live:postgres", "stop:postgres",
    ]
    assert database._closed is True
    assert connection.closed is True


def test_postgres_sleep_uses_only_the_disposable_authority_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, connection = _database()

    async def create() -> DisposablePostgresDatabase:
        return database

    monkeypatch.setattr(DisposablePostgresDatabase, "create", staticmethod(create))
    received: list[KiteStorageConfig] = []
    calls: list[str] = []

    def factory(gate: KiteGate, storage: KiteStorageConfig) -> _Harness:
        received.append(storage)
        return _Harness(ErasureStage.ACTIVE_ASSERTIONS, gate, calls, storage.backend)

    result = asyncio.run(
        workloads._sleep_workload(_gate("performance_changed_work_sleep_postgres_kite_http"), factory)
    )

    assert result.observation == {"sample_count": 3, "p95_ms": 3.0}
    assert [storage.backend for storage in received] == ["postgres"]
    assert received[0].disposable_postgres is database
    assert calls == [
        "prepare:postgres", "seed:postgres", "start:postgres", "sleep:postgres:True", "stop:postgres",
    ]
    assert database._closed is True
    assert connection.closed is True


def test_sqlite_sleep_never_acquires_postgres_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_create() -> DisposablePostgresDatabase:
        raise AssertionError("sqlite sleep must not acquire PostgreSQL")

    monkeypatch.setattr(DisposablePostgresDatabase, "create", staticmethod(unexpected_create))
    received: list[KiteStorageConfig] = []
    calls: list[str] = []

    def factory(gate: KiteGate, storage: KiteStorageConfig) -> _Harness:
        received.append(storage)
        return _Harness(ErasureStage.ACTIVE_ASSERTIONS, gate, calls, storage.backend)

    result = asyncio.run(
        workloads._sleep_workload(_gate("performance_unchanged_sleep_sqlite_kite_http"), factory)
    )

    assert result.observation == {"sample_count": 3, "p95_ms": 3.0}
    assert [storage.backend for storage in received] == ["sqlite"]
    assert calls == ["prepare:sqlite", "start:sqlite", "sleep:sqlite:False", "stop:sqlite"]


def test_core_erasure_rejects_a_non_dual_backend_catalog_contract() -> None:
    spec = _gate("erasure_active_assertions")
    sqlite_only = replace(
        spec,
        environment=ExecutionEnvironment("sqlite", "kite_http", spec.environment.profile),
    )

    with pytest.raises(ReleaseEvidenceError, match="dual-backend HTTP spec"):
        asyncio.run(workloads._core_erasure_workload(sqlite_only, lambda _gate, _storage: None))


@pytest.mark.parametrize("fails", (False, True))
def test_owned_catalog_harness_removes_ephemeral_home_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, fails: bool,
) -> None:
    """No catalog-owned SQLite DB, key, nonce ledger, or log survives a run."""
    harness = workloads._owned_catalog_harness(KiteGate.STABLE_ONLY, KiteStorageConfig())
    temporary_root = harness.config.home.parent

    def prepare() -> None:
        harness.config.home.mkdir()
        for name in (
            "kite-evidence.sqlite3",
            ".kite-evidence-ed25519.key",
            ".kite-evidence-nonces.sqlite3",
            "kite-evidence.log",
        ):
            (harness.config.home / name).write_text("transient", encoding="utf-8")

    monkeypatch.setattr(harness, "prepare", prepare)
    monkeypatch.setattr(harness, "start", lambda: None)
    if fails:
        def fail(_harness) -> None:
            raise KiteEvidenceError("forced owned harness failure")

        with pytest.raises(KiteEvidenceError, match="forced owned harness failure"):
            workloads._run_with_owned_harness(harness, fail)
    else:
        assert workloads._run_with_owned_harness(harness, lambda _harness: "complete") == "complete"

    assert not temporary_root.exists()


def test_kite_postgres_child_receives_only_the_disposable_authority_dsn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database, _connection = _database()
    worktree = tmp_path / "source"
    (worktree / ".git").mkdir(parents=True)
    monkeypatch.setenv("TEST_POSTGRES_URL", "postgresql://ambient/shared")
    monkeypatch.setenv("DATABASE_URL", "postgresql://ambient/shared")
    config = KiteIsolationConfig(
        worktree=worktree,
        home=tmp_path / "home",
        port=48123,
        gate=KiteGate.STABLE_ONLY,
        storage=KiteStorageConfig(backend="postgres", disposable_postgres=database),
    )

    environment = config.environment()

    assert "TEST_POSTGRES_URL" not in environment
    assert environment["DATABASE_URL"] == database.dsn
    assert environment["KESTREL_DATABASE_URL"] == database.dsn
    assert environment["DATABASE_URL"] != "postgresql://ambient/shared"
