"""Resident recovery preserves original owners and distinguishes close custody."""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio

from kestrel_sovereign._async_ownership import create_owned_task, owned_task_error
from kestrel_sovereign.a2a.push_policy import NativePushPolicy
from kestrel_sovereign.a2a.push_service import (
    NativePushService,
    start_native_push_service,
)
from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.storage.db.interface import DatabaseError
from tests.unit.test_a2a_native_push_outbox import outbox as outbox  # noqa: PLC0414
from tests.unit.test_a2a_native_push_outbox import (
    push_store as push_store,  # noqa: PLC0414 - outbox fixture dependency
)
from tests.unit.test_a2a_native_push_worker import worker
from tests.unit.test_a2a_native_question_supervisor import no_peer_read
from tests.unit.test_a2a_native_question_supervisor import (
    questions as questions,  # noqa: PLC0414
)
from tests.unit.test_a2a_native_task_manager import request
from tests.utils.database_query_faults import prepare_transient_database_query_fault
from tests.utils.native_a2a import manager_for
from tests.utils.native_push import configure_push_test_key, intercept_push_commit_ack


@pytest_asyncio.fixture
async def push_runtime(native_a2a_backend_factory, monkeypatch):
    configure_push_test_key(monkeypatch)
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:resident", "tenant"))
    service = None
    try:
        await manager.initialize()
        signals = DurableSignalStore(backend)
        await signals.initialize()
        service = await start_native_push_service(
            manager,
            signals,
            policy=NativePushPolicy(retry_seconds=1),
            task_factory=create_owned_task,
        )
        yield SimpleNamespace(manager=manager, service=service, backend=backend)
    finally:
        # Tests own the real connection and original producer. Failed assertions
        # still join that producer before retiring the disposable database.
        if service is not None:
            await asyncio.gather(service.close(), return_exceptions=True)
        await backend.close()


async def _wait_for_original_recovery(owner, recovered):
    observation = asyncio.create_task(recovered.wait())
    try:
        await asyncio.wait((owner, observation), timeout=5)
        assert recovered.is_set() and not owner.done(), (
            "one transient native query permanently stopped the original resident"
        )
    finally:
        observation.cancel()
        await asyncio.gather(observation, return_exceptions=True)


async def test_push_query_contention_recovers_in_the_same_resident(
    push_runtime, monkeypatch
):
    p = push_runtime
    fail_query = await prepare_transient_database_query_fault(p.backend)
    owner = p.service.task
    runner = p.service.worker
    runner.retry_delay = timedelta(milliseconds=10)
    original = runner.run_once
    failures = []
    recovered = asyncio.Event()

    async def database_pass():
        if len(failures) < 4:
            try:
                await fail_query()
            except DatabaseError as error:
                failures.append(error)
                raise
        result = await original()
        recovered.set()
        return result

    monkeypatch.setattr(runner, "run_once", database_pass)
    await _wait_for_original_recovery(owner, recovered)
    assert len(failures) == 4 and all(
        not is_execution_control_error(error) for error in failures
    )
    assert p.service.task is owner and p.service.require_ready() is p.service.worker
    await p.manager.close()
    assert owner.done() and not p.backend.is_connected


async def test_push_original_session_loss_recovers_without_replacing_its_owner(
    push_runtime, native_a2a_backend_factory, monkeypatch
):
    p = push_runtime
    if p.backend.backend_type != "postgres":
        pytest.skip("PostgreSQL session termination requires a real PostgreSQL server")
    control = native_a2a_backend_factory()
    await control.connect()
    pending = p.service.worker.outbox.changes.pending
    original_owner = p.service.task
    terminated, recovered = [], asyncio.Event()

    async def terminate_original_session(*, limit):
        if not terminated:
            native = p.backend._current_txn_conn()._con
            closed = asyncio.Event()
            native.add_termination_listener(lambda connection: closed.set())
            pid = await p.backend.fetch_val("SELECT pg_backend_pid()")
            assert await control.fetch_val("SELECT pg_terminate_backend(?)", (pid,))
            await asyncio.wait_for(closed.wait(), 5)
            terminated.append(native)
        result = await pending(limit=limit)
        recovered.set()
        return result

    monkeypatch.setattr(p.service.worker.outbox.changes, "pending", terminate_original_session)
    try:
        await _wait_for_original_recovery(original_owner, recovered)
        assert len(terminated) == 1 and terminated[0].is_closed()
        assert p.service.task is original_owner
        assert p.service.require_ready() is p.service.worker
        created = await p.manager.create_task(request(), creator_agent_id="did:creator")
        assert await p.backend.fetch_val(
            "SELECT COUNT(*) FROM a2a_tasks WHERE id = ?", (created.id,)
        ) == 1
    finally:
        await control.close()


async def test_historical_query_failure_does_not_hold_retired_primary_storage(
    push_runtime, monkeypatch
):
    p = push_runtime
    existing = await p.manager.create_task(request(), creator_agent_id="did:creator")

    async def invalid_query(*, limit):
        return await p.backend.fetch_all(
            "SELECT missing_column FROM native_resident_missing_table"
        )

    monkeypatch.setattr(p.service.worker.outbox.changes, "pending", invalid_query)
    done, _ = await asyncio.wait((p.service.task,), timeout=5)
    assert done, "a permanent SQL defect must fail clearly instead of retrying"
    original = owned_task_error(p.service.task)
    assert isinstance(original, DatabaseError)
    assert not is_execution_control_error(original)
    with pytest.raises(DatabaseError) as unavailable:
        p.service.require_ready()
    assert unavailable.value is original
    admitted = await p.manager.create_task(
        request("another-context"), creator_agent_id="did:creator"
    )
    followup = request(existing.context_id)
    followup.message.task_id = existing.id
    continued = await p.manager.continue_task(
        followup, creator_agent_id="did:creator"
    )
    assert admitted.id != existing.id and continued.id == existing.id
    assert await p.backend.fetch_one("SELECT COUNT(*) FROM a2a_task_inputs") == (3,)
    callback = request("callback-context")
    callback.configuration.task_push_notification_config.url = (
        "https://1.1.1.1/callback"
    )
    with pytest.raises(DatabaseError) as push_unavailable:
        await p.manager.create_task(callback, creator_agent_id="did:creator")
    assert push_unavailable.value is original
    assert await p.backend.fetch_one("SELECT COUNT(*) FROM a2a_tasks") == (2,)
    await p.manager.close()
    assert not p.backend.is_connected
    await p.manager.close()
    assert owned_task_error(p.service.task) is original


async def test_uncertain_original_commit_still_holds_primary_storage(
    push_runtime, monkeypatch
):
    p = push_runtime
    original = OSError("original resident journal COMMIT acknowledgement lost")
    calls = []

    async def failed_ack():
        calls.append(True)
        raise original

    intercept_push_commit_ack(p.service.worker.outbox, monkeypatch, failed_ack)
    done, _ = await asyncio.wait((p.service.task,), timeout=5)
    assert done and len(calls) == 1
    failure = owned_task_error(p.service.task)
    assert is_execution_control_error(failure)
    assert execution_commit_outcome(failure) == "unknown"
    with pytest.raises((ExecutionCommitOutcomeError, BaseExceptionGroup)) as admission:
        await p.manager.create_task(request(), creator_agent_id="did:creator")
    assert any(node is failure for node in _explicit_control_chain(admission.value))
    for _ in range(2):
        with pytest.raises((ExecutionCommitOutcomeError, BaseExceptionGroup)) as refused:
            await p.manager.close()
        assert is_execution_control_error(refused.value)
        assert execution_commit_outcome(refused.value) == "unknown"
        assert any(node is original for node in _explicit_control_chain(refused.value))
        assert p.backend.is_connected, "uncertain original commit released primary storage"


async def test_transient_startup_query_is_not_retried_or_advertised(outbox, monkeypatch):
    value, _ = outbox
    fail_query = await prepare_transient_database_query_fault(value.backend)
    with pytest.raises(DatabaseError) as failed:
        await fail_query()
    original_error = failed.value
    original_forward = value.forward_pending
    calls = []

    async def forward_pending(*, limit):
        calls.append(True)
        if len(calls) == 1:
            raise original_error
        return await original_forward(limit=limit)

    monkeypatch.setattr(value, "forward_pending", forward_pending)
    service = NativePushService(worker(value), task_factory=create_owned_task)
    try:
        with pytest.raises(DatabaseError) as caught:
            await service.start()
        assert caught.value is original_error
        assert len(calls) == 1 and service.task.done() and not service.worker.ready
    finally:
        await service.close()


@pytest.mark.parametrize("continuation", [False, True])
async def test_commit_control_racing_admission_still_refuses_callback_free_input(
    push_runtime, monkeypatch, continuation
):
    p = push_runtime
    existing = await p.manager.create_task(request(), creator_agent_id="did:creator")
    original = OSError("original push COMMIT acknowledgement lost during admission")

    async def failed_ack():
        raise original

    @asynccontextmanager
    async def original_admission():
        intercept_push_commit_ack(p.service.worker.outbox, monkeypatch, failed_ack)
        done, _ = await asyncio.wait((p.service.task,), timeout=5)
        assert done, "the original resident did not expose its COMMIT failure"
        yield

    params = request("next-context")
    operation = p.manager.create_task
    if continuation:
        params.message.task_id = existing.id
        params.message.context_id = existing.context_id
        operation = p.manager.continue_task
    with pytest.raises((ExecutionCommitOutcomeError, BaseExceptionGroup)) as refused:
        await operation(
            params, creator_agent_id="did:creator", admission_fence=original_admission
        )
    assert execution_commit_outcome(refused.value) == "unknown"
    assert any(node is original for node in _explicit_control_chain(refused.value))
    assert await p.backend.fetch_one("SELECT COUNT(*) FROM a2a_task_inputs") == (1,)


@pytest.mark.parametrize("lane", ["_read_pass", "_deadline_pass", "handoff_wakes"])
async def test_question_database_query_recovers_without_replacing_its_loop(
    questions, monkeypatch, lane
):
    p = questions
    fail_query = await prepare_transient_database_query_fault(p.backend)
    supervisor = NativeQuestionSupervisor(
        store=p.store,
        open_dispatch=no_peer_read,
        dispatcher=None,  # no question or wake exists in this real empty ledger
        interval=0.01,
        is_current=lambda: True,
    )
    original = getattr(supervisor, lane)
    failed_owner = None
    failures = []
    recovered = asyncio.Event()

    async def database_pass():
        nonlocal failed_owner
        current = asyncio.current_task()
        if len(failures) < 4:
            failed_owner = current
            try:
                await fail_query()
            except DatabaseError as error:
                failures.append(error)
                raise
        result = await original()
        if current is failed_owner:
            recovered.set()
        return result

    monkeypatch.setattr(supervisor, lane, database_pass)
    owner = asyncio.create_task(supervisor.run())
    try:
        await _wait_for_original_recovery(owner, recovered)
        assert failed_owner is not None and not failed_owner.done()
        assert len(failures) == 4 and all(
            not is_execution_control_error(error) for error in failures
        )
        assert not await p.store.unsettled() and not await p.store.pending_wakes()
    finally:
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)


async def test_question_resident_requires_a_typed_database_retry_policy(questions):
    with pytest.raises(TypeError, match="database retry policy"):
        NativeQuestionSupervisor(
            store=questions.store,
            open_dispatch=no_peer_read,
            dispatcher=None,
            interval=0.01,
            is_current=lambda: True,
            database_retry_policy=True,
        )
