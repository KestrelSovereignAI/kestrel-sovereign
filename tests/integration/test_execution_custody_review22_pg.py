"""Production single-agent construction and actual native teardown denial."""

import asyncio

import pytest

from kestrel_sovereign.execution_custody import ExecutionAuthorityError, ProcessRuntimeExecutionFence
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.unit.test_execution_custody_review22 import stub_single_startup

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_actual_single_server_birth_fences_native_child_during_shutdown(native_pg, monkeypatch, tmp_path):
    from kestrel_sovereign import kestrel_agent as agent_module
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    backend, control, schema = native_pg
    await control.execute(f'SET search_path TO "{schema}"')
    server, app = stub_single_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(LLMService, "_load_from_disk_cache", lambda _self: False)
    release, entered = asyncio.Event(), asyncio.Event()
    children, candidates = [], []

    async def initialize(agent):
        # Keep the actual production constructor/backend factory. Only the
        # unrelated provider/feature/genesis boot phases are omitted; the
        # disposable pool already owns this test's unique schema.
        native = agent._build_shared_pool_postgres_backend()
        raw = AsyncStorage(backend=native, agent_id=agent.did)
        raw.db = AsyncDatabase(native)
        agent._raw_storage = agent.storage = raw

        async def surviving_child():
            entered.set()
            await release.wait()
            await native.execute("INSERT INTO effects VALUES (93, 'single startup teardown escape')")

        children.append(asyncio.create_task(surviving_child()))
        await entered.wait()

    def construct(**kwargs):
        # Do not supply any execution_custody argument here: server startup
        # alone must choose and carry its original process runtime.
        agent = KestrelAgent(pg_pool=backend.operational_pool, **kwargs)
        candidates.append(agent)
        return agent

    monkeypatch.setattr(server, "KestrelAgent", construct)
    monkeypatch.setattr(KestrelAgent, "initialize", initialize)

    class RetirementObserved(Exception):
        pass

    def first_teardown_budget(_storage):
        release.set()
        raise RetirementObserved("observe teardown before native pool closure")

    monkeypatch.setattr(agent_module, "_minimum_storage_potential_preclose_timeout", first_teardown_budget)
    try:
        async with server._lifespan_startup(app):
            agent = app.state.agent
            assert agent is candidates[0]
            with pytest.raises(RetirementObserved):
                await agent.shutdown()
            results = await asyncio.gather(*children, return_exceptions=True)
            assert isinstance(results[0], ExecutionAuthorityError), "real single startup child escaped shutdown retirement"
            assert await control.fetchval("SELECT COUNT(*) FROM effects") == 0
            scope = agent._execution_custody
            assert isinstance(scope.fence, ProcessRuntimeExecutionFence)
            assert agent._raw_storage._backend._execution_custody is scope
            assert agent.llm_service._execution_custody is scope
    finally:
        release.set()
        await asyncio.gather(*children, return_exceptions=True)
        for agent in candidates:
            await agent.llm_service.close()
            if agent._raw_storage is not None:
                await agent._raw_storage.close()
