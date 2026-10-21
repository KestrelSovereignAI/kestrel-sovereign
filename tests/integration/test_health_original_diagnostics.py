"""Actual PostgreSQL commit uncertainty cannot become an ordinary health report."""

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from kestrel_sovereign.bootstrap.service import BootstrapService
from kestrel_sovereign.endpoints.agent import agent_health_trigger
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
)
from kestrel_sovereign.features.health import checks
from kestrel_sovereign.features.health.feature import HealthFeature
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.server import (
    _agent_health_result,
    _fleet_agent_health_results,
    health_detailed,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_cleanup_task_admission import _partial_native_agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        "database",
        "scheduler",
        "failures",
        "bootstrap",
        "server",
        "trigger",
        "server_outage",
        "fleet",
    ],
)
async def test_health_diagnostics_preserve_actual_native_unknown(
    native_pg, tmp_path, monkeypatch, route
):
    backend, observer, schema = native_pg
    agent, original_db = await _partial_native_agent(tmp_path, monkeypatch)
    db = AsyncDatabase(backend)
    storage = AsyncStorage(backend=backend)
    storage.db = db
    agent.storage = agent._raw_storage = storage
    agent.features = {"SchedulerFeature": SchedulerFeature(agent)}
    agent.bootstrap_service = BootstrapService(
        db,
        agent.did,
        "partial health agent",
        agent.llm_service,
        tmp_path,
        storage=storage,
    )
    backend._execution_custody = ExecutionCustody(GenerationFence())
    try:
        lose_native_ack(
            monkeypatch,
            boundary="commit",
            original=OSError("actual health diagnostic COMMIT ACK lost"),
        )
        with pytest.raises(BaseException) as write:
            await db.execute(
                "INSERT INTO effects VALUES (40, 'original diagnostic effect')"
            )
        assert execution_commit_outcome(write.value) == "unknown"
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=40'
            )
            == 1
        )
        if route == "database":
            operation = checks.check_database(db)
        elif route == "scheduler":
            operation = checks.check_scheduler_liveness(agent, db)
        elif route == "failures":
            operation = checks.check_scheduled_task_failures(agent, db)
        elif route == "bootstrap":
            operation = checks.check_bootstrap_state(agent)
        elif route == "server":
            operation = _agent_health_result(agent)
        elif route == "fleet":
            # Actual shared fleet reader, with an unpublished real Agent.
            # No fabricated manager roster, genesis or READY publication.
            operation = _fleet_agent_health_results([agent])
        elif route == "server_outage":
            app = FastAPI()
            app.state.agent = agent
            app.state.scheduler_readiness_failures = [
                {"phase": "actual diagnostic outage"}
            ]
            operation = health_detailed(Request({"type": "http", "app": app}))
        else:
            health = HealthFeature(agent)
            health._db, health._agent_id, health._in_memory_history = db, agent.did, []
            health._health_persist_tasks = set()
            agent.features["HealthFeature"] = health
            app = FastAPI()
            app.state.agent = agent
            operation = agent_health_trigger(Request({"type": "http", "app": app}))
        with pytest.raises(BaseException) as caught:
            await operation
        assert execution_commit_outcome(caught.value) == "unknown", (
            "original native control became diagnostics/HTTP failure"
        )
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=40'
            )
            == 1
        )
        assert not agent._runtime_publication_ready.is_set()
    finally:
        backend._execution_custody = None
        agent.features = {}
        await agent.shutdown()
        await original_db.close()
