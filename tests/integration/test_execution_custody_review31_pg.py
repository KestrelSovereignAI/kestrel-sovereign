"""Late physical COMMIT receipts survive bounded facade and LLM handoffs."""
import asyncio
from types import SimpleNamespace

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import bind_execution_custody, execution_commit_outcome
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.unit.test_execution_custody_proxy_ingress import owner
from tests.unit.test_execution_custody_review29 import retire
from tests.unit.test_execution_custody_review30 import service_owner

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def hold_native_ack(monkeypatch, entered, cancelled, release, *, cancel_receipt):
    original = asyncpg.transaction.Transaction.__aexit__

    async def exit_after_commit(transaction, error_type, error, traceback):
        result = await original(transaction, error_type, error, traceback)
        if error_type is None:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            if cancel_receipt:
                raise asyncio.CancelledError("late native COMMIT acknowledgment cancelled")
            raise OSError("late native COMMIT acknowledgment lost")
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit_after_commit)


@pytest.mark.parametrize("completion", ["ack", "retry"])
@pytest.mark.parametrize("cancel_receipt", [False, True])
async def test_proxy_late_native_commit_survives_actual_deadline_handoff(
    native_pg, monkeypatch, tmp_path, completion, cancel_receipt,
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    agent, _, feature, client, event, _, release = await owner(monkeypatch, tmp_path, None)
    entered, cancelled, completed, settled = (asyncio.Event() for _ in range(4))
    hold_native_ack(monkeypatch, entered, cancelled, release, cancel_receipt=cancel_receipt)
    task = operation = None

    async def call(name, payload):
        with bind_execution_custody(GenerationFence()):
            await db.execute("INSERT INTO effects (id) VALUES (1)")

    client.call_host_ingress = call
    try:
        data = event["payload"]
        request = (feature._split_inbound_event_acknowledgement(data)[1]
                   if completion == "ack" else feature._inbound_event_retry_completion(data))
        feature._schedule_event_ingress_completion(client, request, kind=completion,
            expected_state="acknowledged" if completion == "ack" else "retrying")
        (task,) = tuple(feature._event_ack_tasks)
        task.add_done_callback(lambda done: completed.set())
        await asyncio.wait_for(entered.wait(), 5)
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 1
        await asyncio.wait_for(completed.wait(), 5)
        assert not cancelled.is_set(), "observer deadline interrupted the original native COMMIT ACK"
        (ownership,) = tuple(feature._terminal_lifecycle_tasks)
        operation = ownership.task
        assert not operation.done()
        operation.add_done_callback(lambda done: settled.set())
        release.set()
        await asyncio.wait_for(settled.wait(), 5)
        assert not cancelled.is_set(), "native ACK received observer cancellation during retained retirement"
        assert not feature._terminal_lifecycle_tasks
        with pytest.raises(BaseException) as original:
            operation.result()
        assert execution_commit_outcome(original.value) == "unknown"
        with pytest.raises(BaseException) as caught:
            await feature.shutdown()
        assert caught.value is original.value
        assert client.stopped
        assert feature._client is None
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        if operation is not None:
            await await_owned_task(operation._host_task)
        await retire(agent)


@pytest.mark.parametrize("stage", ["preference", "adapter", "client"])
@pytest.mark.parametrize("cancel_receipt", [False, True])
async def test_llm_bounded_native_commit_owner_reports_late_uncertainty(
    native_pg, monkeypatch, stage, cancel_receipt,
):
    from kestrel_sovereign.llm import service as service_module

    backend, observer, schema = native_pg
    db, service = AsyncDatabase(backend), service_owner()
    entered, cancelled, release = (asyncio.Event() for _ in range(3))
    calls, writer, closer = [], None, None
    hold_native_ack(monkeypatch, entered, cancelled, release, cancel_receipt=cancel_receipt)

    async def native_effect():
        calls.append(stage)
        with bind_execution_custody(GenerationFence()):
            await db.execute("INSERT INTO effects (id) VALUES (1)")

    async def later():
        calls.append("later")

    if stage == "preference":
        writer = asyncio.create_task(native_effect())
        writer.add_done_callback(service._handle_preference_persistence_done)
        service._preference_persistence_tasks.add(writer)
        service.providers = [{"client": SimpleNamespace(close=later)}]
    else:
        resource = SimpleNamespace(close=native_effect, aclose=native_effect)
        service.providers = [{stage: resource}, {"client": SimpleNamespace(close=later)}]
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.02)
    original_wait = asyncio.wait

    async def wait_after_commit(tasks, *, timeout):
        if timeout == 0.02 and any(
            task.get_name() == f"llm-close:{'preference-writers' if stage == 'preference' else stage}"
            for task in tasks
        ):
            await asyncio.wait_for(entered.wait(), 5)
        return await original_wait(tasks, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait", wait_after_commit)
    try:
        closer = asyncio.create_task(service.close())
        await asyncio.wait_for(entered.wait(), 5)
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 1
        done, _ = await original_wait({closer}, timeout=1.5)
        assert done, "native COMMIT owner defeated bounded cleanup"
        with pytest.raises(RuntimeError, match="retained cleanup owners"):
            await closer
        assert "later" in calls
        pending = tuple(task for task in service._llm_close_tasks.values() if task is not None)
        release.set()
        for task in pending:
            await await_owned_task(task)
        with pytest.raises(BaseException) as caught:
            await service.close()
        assert execution_commit_outcome(caught.value) == "unknown"
        assert caught.value is service._llm_close_control
        assert calls.count(stage) == 1
        assert calls.count("later") == 1
        if stage == "preference":
            # Finished writers release borrowed storage; their receipt remains
            # separately latched. They are not physical transport close debt.
            assert not any(task is not None for task in service._llm_close_tasks.values())
        else:
            retained = tuple(task for task in service._llm_close_tasks.values() if task is not None)
            assert retained == pending
            assert all(task.done() for task in retained)
            assert service.has_pending_close_owners()
            # Rejoining MUST NOT invoke an uncertain native close a second time.
            with pytest.raises(BaseException) as repeated:
                await service.close()
            assert repeated.value is caught.value
            assert calls.count(stage) == 1 and calls.count("later") == 1
            assert tuple(task for task in service._llm_close_tasks.values() if task is not None) == retained
    finally:
        release.set()
        for task in (closer, writer, *tuple(vars(service).get("_llm_close_tasks", {}).values())):
            if task is not None:
                await await_owned_task(task)
