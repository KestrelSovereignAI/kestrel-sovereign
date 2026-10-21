"""Native startup effect and physical rollback retain the strongest receipt."""
import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody,
    execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review30_pg import lose_ack
from tests.unit.test_execution_custody_review29 import native_owner

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("surface", ["pool", "protocol", "scheduler", "manager"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_native_startup_rollback_cannot_demote_cleanup_control(
    native_pg, monkeypatch, tmp_path, surface, kind,
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    storage = AsyncStorage(backend=backend)
    storage.db = db
    # Produce the receipt through actual PostgreSQL COMMIT and an actual
    # observer Task withdrawal, rather than constructing its classification.
    entered, release = asyncio.Event(), asyncio.Event()
    async def committed_cleanup_evidence():
        with bind_execution_custody(GenerationFence()):
            await backend.execute("INSERT INTO effects VALUES (350, 'cleanup COMMIT')")
    with monkeypatch.context() as ack:
        lose_ack(ack, cancelled=False, entered=entered, release=release)
        source = create_owned_task(committed_cleanup_evidence(), retain_originals=True)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if kind == "cancel_unknown":
                source.cancel("original native cleanup observer withdrawal")
            release.set()
            failure = (await await_owned_task(source)).error
        finally:
            release.set()
            await await_owned_task(source)
    assert failure is not None and execution_commit_outcome(failure) == "unknown"
    if kind == "group":
        connection = sqlite3.connect(tmp_path / "closed-cleanup.db")
        connection.close()
        try:
            connection.execute("SELECT 1")
        except sqlite3.ProgrammingError as independent:
            failure = BaseExceptionGroup("independent original cleanup errors", [failure, independent])
    original_graph = tuple(_explicit_control_chain(failure))
    startup_errors = []
    closed = asyncio.Event()
    original_close = backend.close
    app = FastAPI()

    async def native_startup_failure(*_args):
        await db.execute("INSERT INTO effects VALUES (35, 'startup effect')")
        original = ValueError("ordinary startup failure")
        startup_errors.append(original)
        raise original

    async def native_close_receipt():
        await original_close()
        closed.set()
        raise failure

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://owned-fixture-only")
    monkeypatch.setattr(backend, "close", native_close_receipt)
    manager = SimpleNamespace(
        local_agent_configs_by_did=lambda _config: ready_configs(),
        cold_scheduler_identity_failures=[],
    )

    async def ready_configs():
        return {"did:test:native": ("Native", object())}

    if surface == "pool":
        monkeypatch.setattr(backend, "connect", native_startup_failure)
        monkeypatch.setattr("kestrel_sovereign.storage.db.postgres.PostgresBackend",
                            lambda **_kwargs: backend)
        operation = server._start_shared_agent_postgres_backend(app)
    elif surface == "manager":
        agent, old = await native_owner(tmp_path)
        await old.close()
        agent.storage = agent._raw_storage = storage

        async def agent_shutdown():
            await db.execute("INSERT INTO effects VALUES (35, 'manager rollback effect')")
            await storage.close()

        monkeypatch.setattr(agent, "shutdown", agent_shutdown)
        manager = AgentManager(base_data_dir=tmp_path)
        manager._agents["Native"] = agent
        manager._agent_names[agent.agent_id] = "Native"
        operation = server._rollback_startup_agent_manager(manager)
    else:
        monkeypatch.setattr("kestrel_sovereign.storage.async_storage.AsyncStorage",
                            lambda **_kwargs: storage)
        from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
        if surface == "protocol":
            monkeypatch.setattr(SchedulerRunner, "_ensure_tables", native_startup_failure)
            operation = server._prepare_shared_postgres_scheduler_protocol(app, manager, object())
        else:
            monkeypatch.setattr(SchedulerRunner, "start", native_startup_failure)
            operation = server._start_host_scheduler(app, manager, object())
    task = asyncio.create_task(operation)
    try:
        outcome = await await_owned_task(task)
        assert closed.is_set(), "startup owner skipped physical PostgreSQL retirement"
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=35') == 1
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=350') == 1
        assert is_execution_control_error(outcome.error), "startup rollback demoted original native cleanup control"
        assert execution_commit_outcome(outcome.error) == "unknown", "startup rollback erased original COMMIT uncertainty"
        chain = tuple(_explicit_control_chain(outcome.error))
        assert failure in chain, "startup rollback discarded its original native cleanup carrier"
        assert all(original in chain for original in startup_errors), "startup rollback discarded its original body failure"
        assert tuple(_explicit_control_chain(failure)) == original_graph, "startup rollback mutated its original native cleanup receipt"
    finally:
        await await_owned_task(task)
        monkeypatch.setattr(backend, "close", original_close)
        await original_close()
