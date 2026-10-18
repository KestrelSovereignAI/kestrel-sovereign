"""Real host rollback retains admission debt and joins its one native attempt."""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm import service as service_module
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.security.assertion_tenant_resolver import _resolve_authenticated_agent_assertion_capability
from kestrel_sovereign.storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_storage_backend_parity import _incepted_assertion_identity
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.fixture
async def onboarding_owner(tmp_path, monkeypatch):
    did, identity = await _incepted_assertion_identity(tmp_path, "onboarding")
    capability = _resolve_authenticated_agent_assertion_capability(did, identity)
    storage = AsyncStorage(":memory:", agent_id=did, _assertion_tenant_capability=capability)
    await storage.initialize()
    monkeypatch.setattr(service_module, "load_section", lambda _: {})
    service = service_module.LLMService(agent_data_dir=tmp_path / "onboarding-llm")
    agent = KestrelAgent(
        did=storage.agent_id, storage_path=str(tmp_path / "onboarding-agent.db"),
        llm_service=service,
    )
    agent.storage = agent._raw_storage = storage
    database = storage._assertion_store()._database
    await database.execute("CREATE TABLE host_rollback_effects (id INTEGER PRIMARY KEY)")
    manager = AgentManager(base_data_dir=tmp_path)
    try:
        yield manager, agent, database
    finally:
        await service.close()
        await service.wait_for_close_completion()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["rollback", "unregistered"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
async def test_host_rollback_refusal_keeps_original_registration_and_closes_coroutine(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel, mode,
):
    manager, agent, database = onboarding_owner
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    effects, refused, accepted = [], [], []
    async def rollback():
        await database.execute("INSERT INTO host_rollback_effects VALUES (1)")
        effects.append(True)
    async def hook(_name, _agent):
        return rollback
    manager.set_agent_registration_hook(hook)
    await manager._on_agent_registered("Original", agent)
    registration = agent._host_onboarding_rollback
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.failed and getattr(callback, "__name__", "") == "close_unstarted":
                self.failed = True
                if mode == "publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise original
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(getattr(operation, "cr_code", None), "co_qualname", "").endswith(".<locals>.rollback"):
            if mode == "refuse":
                refused.append((coroutine, operation))
                raise original
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    async def invoke():
        if surface == "rollback":
            _cancelled, failure = await manager._rollback_host_onboarding("Original", agent)
            if failure is not None:
                raise failure
        else:
            await manager._shutdown_unregistered_agent("Original", agent)
    owner = create_owned_task(invoke(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        assert original in tuple(_explicit_control_chain(outcome.error))
        assert getattr(agent, "_host_onboarding_rollback", None) is registration, "admission refusal lost original provisional registration"
        assert len(refused if mode == "refuse" else accepted) == 1 and not effects
        assert all(task.done() for task in accepted), "rollback abandoned accepted publication-failed original Task"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "rollback admission leaked its refused original coroutine"
        expected = None if mode == "refuse" else accepted[0]
        assert any(resource is registration and task is expected for resource, task in execution_cleanup_debts(outcome.error)), "refused rollback falsely acknowledged native retirement"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts
        with pytest.raises(BaseException) as committed:
            manager._commit_host_onboarding(agent)
        assert original in tuple(_explicit_control_chain(committed.value)), "failed rollback was acknowledged by a new host commit"
        second = await await_owned_task(create_owned_task(invoke(), retain_originals=True))
        assert original in tuple(_explicit_control_chain(second.error)) and len(refused if mode == "refuse" else accepted) == 1, "failed original rollback acquired a replacement native attempt"
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()


@pytest.mark.asyncio
async def test_normal_batch_wait_interrupts_each_original_initializer_only_once(native_pg, onboarding_owner, monkeypatch):
    manager, _agent, database = onboarding_owner
    source = await _original_control(native_pg[0], monkeypatch)
    graph, debts = tuple(_explicit_control_chain(source)), tuple(execution_cleanup_debts(source))
    entered, unwinding, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    originals, withdrawals = {}, []
    async def initialize(name, *_args, **_kwargs):
        from kestrel_sovereign.multi_agent.agent_manager import _BATCH_STAGE_CALLBACK
        originals[name] = asyncio.current_task()
        if len(originals) == 2:
            entered.set()
        try:
            await asyncio.get_running_loop().create_future()
        except asyncio.CancelledError as cancellation:
            withdrawals.append(cancellation)
            if len(withdrawals) == 2:
                unwinding.set()
            try:
                await release.wait()
                await database.execute("INSERT INTO host_rollback_effects VALUES (?)", (21 if name == "Alpha" else 22,))
                raise source
            finally:
                await _BATCH_STAGE_CALLBACK.get()(name)
    monkeypatch.setattr(manager, "_initialize_agent", initialize)
    config = MultiAgentConfig(agents={
        "Alpha": LocalAgentConfig(data_dir="alpha", port=8801),
        "Beta": LocalAgentConfig(data_dir="beta", port=8802),
    })
    owner = create_owned_task(manager.load_from_config(config), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual first batch observer withdrawal")
        await asyncio.wait_for(unwinding.wait(), 5)
        owner.cancel("actual repeated batch observer withdrawal")
        await asyncio.sleep(0)
        release.set()
        outcome = await await_owned_task(owner)
        assert all(task.done() and task.cancelling() == 1 for task in originals.values()), "normal batch aggregate repeatedly interrupted accepted native initializer unwind"
        assert source in tuple(_explicit_control_chain(outcome.error)), "repeated batch withdrawal erased the original native control"
        assert tuple(_explicit_control_chain(source)) == graph and tuple(execution_cleanup_debts(source)) == debts
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id IN (21,22)") == 2
        assert not manager._agent_operations
    finally:
        release.set()
        await await_owned_task(owner)
        for task in originals.values():
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
async def test_host_rollback_joins_original_native_attempt_and_retains_all_control(
    native_pg, onboarding_owner, monkeypatch, source_cancel, withdraw,
):
    manager, agent, database = onboarding_owner
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    entered, release = asyncio.Event(), asyncio.Event()
    attempts, withdrawals = [], []
    async def rollback():
        attempts.append(asyncio.current_task())
        await database.execute("INSERT INTO host_rollback_effects VALUES (2)")
        entered.set()
        await release.wait()
        raise original
    async def hook(_name, _agent):
        return rollback
    manager.set_agent_registration_hook(hook)
    await manager._on_agent_registered("Original", agent)
    registration = agent._host_onboarding_rollback
    from kestrel_sovereign.multi_agent import agent_manager as manager_module
    actual_join = manager_module.await_owned_task
    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    monkeypatch.setattr(manager_module, "await_owned_task", join)
    async def invoke():
        _cancelled, failure = await manager._rollback_host_onboarding("Original", agent)
        if failure is not None:
            raise failure
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if withdraw:
            owner.cancel("actual onboarding rollback observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done(), "rollback returned before native write observer settled"
        release.set()
        outcome = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original in chain
        assert getattr(agent, "_host_onboarding_rollback", None) is registration, "failed physical rollback lost its only owner"
        assert len(attempts) == 1 and attempts[0].done()
        assert any(resource is registration and task is attempts[0] for resource, task in execution_cleanup_debts(outcome.error)), "failed native rollback lost exact original Task debt"
        if withdraw:
            assert len(withdrawals) == 1 and withdrawals[0] in chain, "rollback replaced actual observer withdrawal"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts
        second = await await_owned_task(create_owned_task(invoke(), retain_originals=True))
        assert original in tuple(_explicit_control_chain(second.error)) and len(attempts) == 1
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        await await_owned_task(owner)
        for task in attempts:
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("already_cancelled", [False, True])
async def test_batch_admission_joins_every_accepted_original_before_releasing_names(
    native_pg, onboarding_owner, monkeypatch, mode, source_cancel, eager, already_cancelled,
):
    manager, agent, database = onboarding_owner
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    accepted, refused, produced, staging = [], [], [], []
    async def initialize(name, *_args, **_kwargs):
        from kestrel_sovereign.multi_agent.agent_manager import _BATCH_STAGE_CALLBACK
        if name == "Alpha":
            staging.append(_BATCH_STAGE_CALLBACK.get())
        assert name == "Alpha", "unstarted rejected initializer performed native work"
        await database.execute("INSERT INTO host_rollback_effects VALUES (3)")
        produced.append(agent)
        return agent
    monkeypatch.setattr(manager, "_initialize_agent", initialize)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    base_factory = asyncio.eager_task_factory if eager else previous
    admissions = 0
    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.failed and getattr(callback, "__name__", "") in {"close_unstarted", "owned_task_error"}:
                self.failed = True
                if mode == "publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise original
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        nonlocal admissions
        operation = _actual_operation(coroutine)
        is_initializer = getattr(getattr(operation, "cr_code", None), "co_qualname", "").endswith(".<locals>._bounded_initialize")
        if is_initializer:
            admissions += 1
            if admissions == 2:
                if already_cancelled:
                    accepted[0].cancel("original initializer cancellation already requested")
                if mode == "refuse":
                    refused.append((coroutine, operation))
                    raise original
                task = PublicationTask(coroutine, loop=loop, **kwargs, **({"eager_start": True} if eager else {}))
                accepted.append(task)
                return task
        task = base_factory(loop, coroutine, **kwargs) if base_factory else asyncio.Task(coroutine, loop=loop, **kwargs)
        if is_initializer:
            accepted.append(task)
        return task
    config = MultiAgentConfig(agents={
        "Alpha": LocalAgentConfig(data_dir="alpha", port=8801),
        "Beta": LocalAgentConfig(data_dir="beta", port=8802),
    })
    owner = create_owned_task(manager.load_from_config(config), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        assert original in tuple(_explicit_control_chain(outcome.error))
        assert admissions == 2
        assert all(task.done() for task in accepted), "batch released names while an accepted original initializer was still live"
        if already_cancelled:
            assert accepted[0].cancelling() == 1, "batch repeated interruption of an already-cancelling original initializer"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "batch leaked a refused original initializer coroutine"
        assert not manager._agent_operations and not manager.list_agents()
        if produced:
            assert database.backend._connection is None, "initialized private agent escaped native storage retirement"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        # The old broken admission path never starts Beta. Unblock the real
        # batch stage barrier so fixture retirement itself cannot deadlock.
        stage_release = asyncio.create_task(staging[0]("Beta")) if staging else None
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        if stage_release is not None:
            await await_owned_task(stage_release)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
