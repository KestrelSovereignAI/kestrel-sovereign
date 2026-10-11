"""Native startup effect and physical rollback retain the strongest receipt."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_execution_custody_review29 import native_owner
from tests.unit.test_execution_custody_review32 import receipt

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
    failure = receipt(kind)
    closed = asyncio.Event()
    original_close = backend.close
    app = FastAPI()

    async def native_startup_failure(*_args):
        await db.execute("INSERT INTO effects VALUES (35, 'startup effect')")
        raise ValueError("ordinary startup failure")

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
        assert outcome.error is failure, "startup rollback demoted original native cleanup receipt"
    finally:
        await await_owned_task(task)
        monkeypatch.setattr(backend, "close", original_close)
        await original_close()
