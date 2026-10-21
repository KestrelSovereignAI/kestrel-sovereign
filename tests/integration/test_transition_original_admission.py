"""Actual transition Task admission cannot abandon its accepted lease heartbeat.

This drives genuine SQLite config staging and a real PG original receipt.
It is not isolated-child launch, provider response or Agent READY acceptance.
"""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.isolated_runtime import ProxyFeature
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm import service as service_module
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_semantic_inference import _GOVERNED_STORAGES, assertion_store as assertion_store
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.fixture
async def transition_owner(assertion_store, tmp_path, monkeypatch):
    storage = _GOVERNED_STORAGES[id(assertion_store)]
    monkeypatch.setattr(service_module, "load_section", lambda _: {})
    service = service_module.LLMService(agent_data_dir=tmp_path / "transition-llm")
    agent = KestrelAgent(
        did=storage.agent_id, storage_path=str(tmp_path / "transition-agent.db"),
        llm_service=service,
    )
    agent.storage = agent._raw_storage = storage
    runtime = InstalledFeatureRuntime(
        class_name="OriginalTransitionFeature",
        entry_point="local.feature:OriginalTransitionFeature", distribution="local-fixture",
        runtime="isolated-venv", service="original-transition",
    )
    feature = ProxyFeature(agent, runtime)
    transition = await feature._stage_pending_config({"enabled": True})
    assert transition.persistent and transition.staged_properties
    try:
        yield feature, transition
    finally:
        await service.close()
        await service.wait_for_close_completion()


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["heartbeat", "hook"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_transition_refused_task_owns_original_coroutine_and_accepted_peer(
    native_pg, transition_owner, monkeypatch, which, source_cancel,
):
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    graph = tuple(_explicit_control_chain(original))
    feature, transition = transition_owner
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused, accepted = [], []
    names = {
        "heartbeat": "ProxyFeature._run_transition_lease_heartbeat",
        "hook": "ProxyFeature._prepare_config_transition",
    }
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        name = getattr(getattr(operation, "cr_code", None), "co_qualname", "")
        if name == names[which]:
            refused.append((coroutine, operation))
            raise original
        task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if name in names.values():
            accepted.append(task)
        return task
    owner = create_owned_task(feature._prepare_config_transition_with_lease(transition), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        assert len(refused) == 1 and original in tuple(_explicit_control_chain(outcome.error))
        assert all(task.done() for task in accepted), "transition admission abandoned an accepted original heartbeat"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "transition admission leaked original refused coroutine"
        assert tuple(_explicit_control_chain(original)) == graph
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        for pair in refused:
            for coro in pair:
                coro.close()



@pytest.mark.asyncio
@pytest.mark.parametrize("withdraw", [False, True])
async def test_transition_heartbeat_cas_physically_settles_before_result(
    native_pg, transition_owner, transition_wire, monkeypatch, withdraw,
):
    from datetime import timedelta
    from kestrel_sdk.isolated_feature import ConfigTransitionResult
    from kestrel_sovereign.features import isolated_runtime as runtime_module
    original = await _original_control(native_pg[0], monkeypatch)
    graph = tuple(_explicit_control_chain(original))
    feature, transition = transition_owner
    renewed, release, hook_done = (asyncio.Event() for _ in range(3))
    async def served_transition(config):
        await renewed.wait()
        return ConfigTransitionResult.restart_required()
    await transition_wire(served_transition)
    actual_renew, actual_join = feature._renew_transition_lease, runtime_module.await_owned_task
    withdrawals, accepted = [], []
    async def renew(current):
        await actual_renew(current)
        renewed.set()
        await release.wait()
        raise original
    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    monkeypatch.setattr(runtime_module, "_PENDING_CONFIG_LEASE_TTL", timedelta(seconds=0.03))
    monkeypatch.setattr(feature, "_renew_transition_lease", renew)
    monkeypatch.setattr(runtime_module, "await_owned_task", join)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        name = getattr(getattr(operation, "cr_code", None), "co_qualname", "")
        task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if name == "ProxyFeature._prepare_config_transition":
            task.add_done_callback(lambda _: hook_done.set())
        if name in ("ProxyFeature._prepare_config_transition", "ProxyFeature._run_transition_lease_heartbeat"):
            accepted.append(task)
        return task
    owner = create_owned_task(feature._prepare_config_transition_with_lease(transition), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        await asyncio.wait_for(hook_done.wait(), 5)
        assert renewed.is_set()
        await asyncio.wait({owner}, timeout=0.05)
        assert not owner.done(), "transition returned before accepted original heartbeat CAS settled"
        if withdraw:
            owner.cancel("actual heartbeat CAS retirement observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done()
        release.set()
        outcome = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original in chain
        if withdraw:
            assert withdrawals and all(error in chain for error in withdrawals)
        assert all(task.done() for task in accepted)
        assert tuple(_explicit_control_chain(original)) == graph
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)


@pytest.fixture
async def transition_wire(transition_owner):
    from kestrel_sdk.isolated_feature import IsolatedFeatureClient, IsolatedFeatureService
    from tests.integration.test_isolated_feature_config_transition import _QueueReader, _QueueWriter, _SDKWireClient
    feature, _ = transition_owner
    originals = []
    async def start(transition_hook):
        class ActualTransitionService(IsolatedFeatureService):
            def __init__(self):
                super().__init__(name="original-transition", version="1.0.0")
                self.advertise_config_transition()
            async def on_config_transition(self, config):
                return await transition_hook(config)
        host_reader, service_reader = _QueueReader(), _QueueReader()
        server = create_owned_task(ActualTransitionService().serve(service_reader, _QueueWriter(host_reader)), retain_originals=True)
        client = _SDKWireClient(IsolatedFeatureClient(host_reader, _QueueWriter(service_reader)), {})
        originals.append((client, server))
        await client.start()
        feature._client = client
        return client
    try:
        yield start
    finally:
        for client, server in originals:
            await client.stop()
            await await_owned_task(server)
        feature._client = None



@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
async def test_transition_actual_rpc_unwind_preserves_lease_hook_and_observer_control(
    native_pg, transition_owner, transition_wire, monkeypatch, source_cancel, withdraw,
):
    from datetime import timedelta
    from kestrel_sdk.isolated_feature import ConfigTransitionResult
    from kestrel_sovereign.features import isolated_runtime as runtime_module

    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    graph = tuple(_explicit_control_chain(original))
    feature, transition = transition_owner
    entered, service_release, closing, close_release = (asyncio.Event() for _ in range(4))
    async def served_transition(config):
        entered.set()
        await service_release.wait()
        return ConfigTransitionResult.restart_required()
    await transition_wire(served_transition)
    actual_renew, actual_prepare = feature._renew_transition_lease, feature._prepare_config_transition
    renewed, hook_errors, native_tasks, withdrawals = [], [], [], []
    actual_join = runtime_module.await_owned_task
    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    async def renew(current):
        await entered.wait()
        await actual_renew(current)
        renewed.append(True)
        raise original
    async def prepare(config):
        native_tasks.append(asyncio.current_task())
        try:
            return await actual_prepare(config)
        except asyncio.CancelledError as cancellation:
            closing.set()
            await close_release.wait()
            try:
                await transition.storage.db.execute("INSERT INTO missing_original_hook_cleanup VALUES (1)")
            except Exception as error:
                hook_errors.append((error, tuple(_explicit_control_chain(error))))
                raise BaseExceptionGroup("actual hook cancellation and native cleanup", [cancellation, error])
    monkeypatch.setattr(runtime_module, "_PENDING_CONFIG_LEASE_TTL", timedelta(seconds=0.03))
    monkeypatch.setattr(feature, "_renew_transition_lease", renew)
    monkeypatch.setattr(feature, "_prepare_config_transition", prepare)
    monkeypatch.setattr(runtime_module, "await_owned_task", join)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    accepted = []
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        name = getattr(getattr(operation, "cr_code", None), "co_qualname", "")
        task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if name == "ProxyFeature._run_transition_lease_heartbeat":
            def prior_observer(done):
                try:
                    done.exception()
                except asyncio.CancelledError:
                    pass
            task.add_done_callback(prior_observer)
            accepted.append(task)
        return task
    owner = create_owned_task(feature._prepare_config_transition_with_lease(transition), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        await asyncio.wait_for(closing.wait(), 5)
        assert renewed and not owner.done()
        if withdraw:
            owner.cancel("actual config hook retirement observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done()
        close_release.set()
        outcome = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original in chain, "transition hook retirement replaced original lease control"
        assert len(hook_errors) == 1 and hook_errors[0][0] in chain, "transition lost independent actual hook cleanup failure"
        if withdraw:
            assert withdrawals and all(error in chain for error in withdrawals), "transition dropped actual independent retirement observer cancellation"
        assert tuple(_explicit_control_chain(hook_errors[0][0])) == hook_errors[0][1], "transition rewrote independent original hook failure graph"
        assert all(task.done() for task in accepted + native_tasks), "transition returned before original hook/heartbeat settlement"
        assert tuple(_explicit_control_chain(original)) == graph
    finally:
        loop.set_task_factory(previous)
        close_release.set()
        service_release.set()
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)



@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["heartbeat", "hook"])
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_transition_publication_keeps_accepted_start_gated_original(
    native_pg, transition_owner, monkeypatch, which, after, eager, source_cancel,
):
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    graph = tuple(_explicit_control_chain(original))
    feature, transition = transition_owner
    names = {"heartbeat": "ProxyFeature._run_transition_lease_heartbeat", "hook": "ProxyFeature._prepare_config_transition"}
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    accepted, selected = [], []
    class OriginalTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *, context=None):
            if self.failed or getattr(callback, "__name__", None) != "close_unstarted":
                return super().add_done_callback(callback, context=context)
            self.failed = True
            if after:
                super().add_done_callback(callback, context=context)
            raise original
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        name = getattr(getattr(operation, "cr_code", None), "co_qualname", "")
        if name == names[which]:
            task = OriginalTask(coroutine, loop=loop, **kwargs, **({"eager_start": True} if eager else {}))
            selected.append(task)
        else:
            task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if name in names.values():
            accepted.append(task)
        return task
    owner = create_owned_task(feature._prepare_config_transition_with_lease(transition), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        assert original in tuple(_explicit_control_chain(outcome.error))
        assert len(selected) == 1 and all(task.done() for task in accepted), "transition publication abandoned an accepted original"
        failed = await await_owned_task(selected[0])
        assert original in tuple(_explicit_control_chain(failed.error)), "failed transition publication spent the unadmitted native hook"
        assert tuple(_explicit_control_chain(original)) == graph
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
