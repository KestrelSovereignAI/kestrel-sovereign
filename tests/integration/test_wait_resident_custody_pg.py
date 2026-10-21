"""Real wait-worker lifetime after hosted bootstrap admission retires (#3385).

Uses the actual initialize/READY boundary with only the wait publication phase,
actual WaitFeature/reconciler, and native PostgreSQL. This is deliberately not
a claim that the complete application boot or a hosted deployment was tested.
"""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.agent.boot import BootPhase, BootPhaseState
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    bind_execution_custody,
    require_execution_work,
)
from kestrel_sovereign.features.wait.feature import WaitFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase, core_schema_sql
from kestrel_sovereign.storage.async_wait_signal_store import WaitSignalStore
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import run_wait_reconcile
from tests.integration.test_execution_authority_postgres import (
    GenerationFence,
    native_pg as _native_pg,
)
from tests.unit.test_agent_boot_phases import _make_agent

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("retirement", ["generation", "feature-disable"])
async def test_wait_resident_outlives_boot_admission_but_not_original_generation(
    native_pg, tmp_path, monkeypatch, retirement,
):
    backend, _, _ = native_pg
    async def initialize_schema(database):
        # The actual canonical base schema contains the wait ledger. Optional
        # embedding-vector backfills are not part of this lifecycle regression.
        await database.execute_script(core_schema_sql(database.backend_type))

    db = await AsyncDatabase.from_connected_backend(
        backend, schema_initializer=initialize_schema,
    )
    agent = _make_agent(tmp_path)
    agent._execution_custody = ExecutionCustody(GenerationFence())
    agent._raw_storage = SimpleNamespace(db=db)
    agent.wait_registry = WaitRegistry()
    feature = WaitFeature(agent)
    agent.features = {"WaitFeature": feature}
    # Finite event-driven regression; no stress/busy loops or real-time wait.
    feature._FALLBACK_RECONCILE_POLL_SECONDS = 0.01
    feature._FALLBACK_RECONCILE_STALE_SECONDS = 0
    queried = asyncio.Event()
    native_pending = WaitSignalStore.list_pending

    async def observe_native_query(store):
        result = await native_pending(store)
        queried.set()
        return result

    monkeypatch.setattr(WaitSignalStore, "list_pending", observe_native_query)

    async def publish_wait(ctx):
        await feature.initialize()
        await feature.post_all_features_loaded(agent)
        ctx.on_rollback("wait-worker", feature.shutdown)
        assert not queried.is_set(), "resident work ran before READY publication"

    agent._boot_phases = lambda: [BootPhase("wait-publication", publish_wait)]
    worker = observed = ordinary_child = None
    release_child = asyncio.Event()

    async def ordinary_effect_child():
        await release_child.wait()
        require_execution_work(agent)
        await backend.execute("INSERT INTO effects VALUES (1, 'escaped boot')")

    try:
        with bind_execution_custody(GenerationFence()) as boot_admission:
            await agent.initialize()
            assert agent._boot_state is BootPhaseState.READY
            assert agent._runtime_publication_ready.is_set()
            worker, = feature._owned_background_tasks
            ordinary_child = agent._track_background_task(
                ordinary_effect_child(), name="ordinary-bootstrap-child",
            )
        with pytest.raises(ExecutionAuthorityError, match="admission retired"):
            boot_admission.require_work()
        # The ordinary child MUST keep the retired original admission. Only
        # the explicit resident wait service may use the runtime handoff.
        release_child.set()
        ordinary = await await_owned_task(ordinary_child)
        assert isinstance(ordinary.error, ExecutionAuthorityError)
        assert await backend.fetch_val("SELECT count(*) FROM effects") == 0

        observed = asyncio.create_task(queried.wait())
        done, _ = await asyncio.wait(
            {observed, worker}, timeout=5, return_when=asyncio.FIRST_COMPLETED,
        )
        assert observed in done and not worker.done(), (
            "wait fallback inherited retired bootstrap admission: "
            f"{(await await_owned_task(worker)).error if worker.done() else 'no native tick'}"
        )
        # The same singleton must remain usable by the scheduler cron. A
        # failed fallback tick must not poison it with retired boot custody.
        assert (await run_wait_reconcile(agent)).status == "ok"
        assert agent._wait_reconciler._terminal_control is None

        if retirement == "generation":
            agent._execution_custody.revoke("original hosted generation retired")
            terminal = await asyncio.wait_for(await_owned_task(worker), timeout=5)
            assert isinstance(terminal.error, ExecutionAuthorityError)
            assert "original hosted generation retired" in str(terminal.error)
            with pytest.raises(ExecutionAuthorityError, match="generation retired"):
                await run_wait_reconcile(agent)
        else:
            await feature.shutdown()
            assert worker.done() and worker not in agent._background_tasks
            assert not feature._owned_background_tasks
            agent._execution_custody.require_work()
    finally:
        release_child.set()
        for task in (observed, ordinary_child, worker):
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                await await_owned_task(task)
        # The expected original authority failure remains sticky through
        # actual feature cleanup; never erase it to make shutdown look clean.
        try:
            await feature.shutdown()
        except BaseException as error:
            assert isinstance(error, ExecutionAuthorityError)
        await agent.llm_service.close()
