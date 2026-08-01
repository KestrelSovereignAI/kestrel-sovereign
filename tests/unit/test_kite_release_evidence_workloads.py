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
    def __init__(self, stage: ErasureStage, calls: list[str], backend: str, *, fail: bool = False) -> None:
        self._stage = stage
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

    def factory(_gate: KiteGate, storage: KiteStorageConfig) -> _Harness:
        received.append(storage)
        return _Harness(ErasureStage.ACTIVE_ASSERTIONS, calls, storage.backend)

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

    def factory(_gate: KiteGate, storage: KiteStorageConfig) -> _Harness:
        return _Harness(
            ErasureStage.ACTIVE_ASSERTIONS,
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


def test_core_erasure_rejects_a_non_dual_backend_catalog_contract() -> None:
    spec = _gate("erasure_active_assertions")
    sqlite_only = replace(
        spec,
        environment=ExecutionEnvironment("sqlite", "kite_http", spec.environment.profile),
    )

    with pytest.raises(ReleaseEvidenceError, match="dual-backend HTTP spec"):
        asyncio.run(workloads._core_erasure_workload(sqlite_only, lambda _gate, _storage: None))


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
