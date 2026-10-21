"""A refused peer cleanup cannot abandon originals or reuse unsafe names."""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.security.assertion_tenant_resolver import _resolve_authenticated_agent_assertion_capability
from kestrel_sovereign.storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_storage_backend_parity import _incepted_assertion_identity
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["refuse", "publish-before", "publish-after"])
async def test_unpublished_peer_refusal_joins_accepted_cleanup_and_reserves_unsafe_name(
    native_pg, onboarding_owner, monkeypatch, tmp_path, fault,
):
    manager, alpha, alpha_db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    did, identity = await _incepted_assertion_identity(tmp_path, "unpublished-beta")
    capability = _resolve_authenticated_agent_assertion_capability(did, identity)
    storage = AsyncStorage(":memory:", agent_id=did, _assertion_tenant_capability=capability)
    await storage.initialize()
    service = LLMService(agent_data_dir=tmp_path / "beta-service")
    beta = KestrelAgent(did=did, storage_path=str(tmp_path / "beta.db"), llm_service=service)
    beta.storage = beta._raw_storage = storage
    alpha_admission, _ = await manager._admit_agent_operation("Alpha", kind="load")
    beta_admission, _ = await manager._admit_agent_operation("Beta", kind="load")
    entered, release, accepted, refused = asyncio.Event(), asyncio.Event(), [], []
    actual_shutdown = KestrelAgent.shutdown
    async def shutdown(agent):
        if agent is alpha:
            entered.set()
            await release.wait()
        await actual_shutdown(agent)
    monkeypatch.setattr(KestrelAgent, "shutdown", shutdown)
    task_type = None
    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.failed and getattr(callback, "__name__", "") == "close_unstarted":
                self.failed = True
                if fault == "publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    task_type = PublicationTask
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code, frame = getattr(operation, "cr_code", None), getattr(operation, "cr_frame", None)
        if code is not None and code.co_name == "_discard_unpublished_initialized_agent":
            if frame.f_locals["name"] == "Beta":
                if fault == "refuse":
                    refused.append((coroutine, operation))
                    raise source.error
                task = task_type(coroutine, loop=loop, **kwargs)
                accepted.append(("Beta", task))
                return task
            task = asyncio.Task(coroutine, loop=loop, **kwargs)
            accepted.append(("Alpha", task))
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = create_owned_task(manager._discard_unpublished_initialized_agents([("Alpha", alpha), ("Beta", beta)]), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        done, _ = await asyncio.wait({owner}, timeout=0.05)
        assert not done, "peer allocation refusal abandoned an accepted original unpublished cleanup"
        release.set()
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        assert accepted and all(task.done() for _name, task in accepted)
        assert alpha_db.backend._connection is None
        assert storage._assertion_store()._database.backend._connection is not None
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        assert any(resource is beta for resource, _task in execution_cleanup_debts(outcome.error)), "refused unpublished cleanup lost actual agent custody"
        attempts = len(accepted) + len(refused)
        repeated = await await_owned_task(create_owned_task(manager._discard_unpublished_initialized_agents([("Alpha", alpha), ("Beta", beta)]), retain_originals=True))
        source.assert_retained(repeated.error)
        assert len(accepted)+len(refused) == attempts, "failed unpublished cleanup acquired a replacement attempt"
        releases = await await_owned_task(create_owned_task(manager._release_agent_operations([alpha_admission, beta_admission]), retain_originals=True))
        source.assert_retained(releases.error)
        assert "alpha" not in manager._agent_operations and manager._agent_operations.get("beta") is beta_admission, "unsafe private cleanup released its name admission"
        assert beta_admission.child is beta
        assert source.error in tuple(_explicit_control_chain(releases.error))
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(owner)
        for _name, task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await service.close()
        await service.wait_for_close_completion()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["refuse", "publish-before", "publish-after"])
async def test_declared_spawn_slot_retirement_keeps_original_admission(native_pg, onboarding_owner, monkeypatch, fault):
    # The cap-slot snapshot is declared input, not a signed spawn, budget or
    # provider acceptance. The name admission and PG original are genuine.
    manager, _agent, _database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    admission, created = await manager._admit_agent_operation("OriginalSlot", kind="spawn")
    assert created
    manager._pending_spawns = 1
    admission.spawn_slot_active = True
    admission.spawn_slot_terminal = asyncio.get_running_loop().create_future()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    accepted, refused, attempts = [], [], []
    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.failed and getattr(callback, "__name__", "") == "close_unstarted":
                self.failed = True
                if fault == "publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_name == "retire_spawn_slot":
            attempts.append(operation)
            if fault == "refuse":
                refused.append((coroutine, operation))
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if code is not None and code.co_name in {"_release_agent_operation", "release"}:
            accepted.append(task)
        return task
    async def retire():
        await manager._retire_spawn_slot_and_admission(admission, spawn_slot_admitted=True)
    owner = create_owned_task(retire(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        first = await await_owned_task(owner)
        await asyncio.sleep(0)
        source.assert_retained(first.error)
        assert all(task.done() for task in accepted)
        assert manager._agent_operations.get("originalslot") is admission, "failed cap-slot retirement released its original name admission"
        assert admission.spawn_slot_active and manager._pending_spawns == 1
        assert any(resource is admission for resource, _task in execution_cleanup_debts(first.error)), "unaccepted cap retirement lost original admission debt"
        repeated = await await_owned_task(create_owned_task(retire(), retain_originals=True))
        source.assert_retained(repeated.error)
        assert len(attempts) == 1, "failed cap retirement admitted a replacement state mutation"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
