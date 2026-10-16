"""Actual native lookup control must not grant durable watch admission.

The recipient view and structural native provider are admitted helper inputs,
not a full Agent, verified peer or genesis/READY acceptance test.
"""

from dataclasses import dataclass
from uuid import uuid4

import asyncpg
import pytest
from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign.a2a import outbound_store
from kestrel_sovereign.a2a.stores.unified import (
    ObservabilityStore,
    SessionService,
    TaskStore,
)
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.peers.wait_provider import A2AWaitable
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import (
    _provider_origin_session,
    register_wait_watch,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        "task-own",
        "task-register",
        "generic-register",
        "origin",
        "a2a-own",
        "a2a-register",
        "a2a-poll",
        "a2a-peer",
    ],
)
async def test_native_recipient_control_cannot_become_unverifiable_watch(
    native_pg, tmp_path, monkeypatch, route
):
    backend, observer, schema = native_pg
    manager = TaskManager(
        TaskStore(backend), SessionService(backend), ObservabilityStore(backend)
    )
    await manager.initialize()
    did = "did:web:original-wait-" + uuid4().hex
    task = await manager.create_task(
        TaskSendParams(
            id=uuid4().hex,
            sessionId=uuid4().hex,
            message=Message(role="user", parts=[TextPart(text="native ownership")]),
        ),
        agent_name=did,
    )
    local_backend = SQLiteBackend(str(tmp_path / "local-wait.db"))
    await local_backend.connect()
    storage = AsyncStorage.from_backend(local_backend, agent_id=did)
    await storage.initialize()

    @dataclass
    class RecipientView:
        did: str
        wait_registry: WaitRegistry
        _raw_storage: AsyncStorage

    recipient = RecipientView(did, WaitRegistry(), storage)
    feature = TaskFeature(recipient)
    feature.set_task_manager(manager)
    await feature.initialize()
    await feature.post_all_features_loaded(recipient)
    task_provider = recipient.wait_registry.get("task")

    class NativeWaitProvider:
        # An actual structural Waitable plugin whose reads use the production
        # TaskManager/store. It has no control-consuming exception handler.
        kind = "native"
        signal = None

        async def owns_handle(self, handle):
            return await manager.get_task_for_recipient(handle, did) is not None

        async def origin_session_id(self, handle):
            row = await manager.get_task_for_recipient(handle, did)
            return row.sessionId if row is not None else None

        async def poll(self, handle):
            row = await manager.get_task_for_recipient(handle, did)
            return WaitStatus(
                Outcome.PENDING, row.status.state.value if row else "absent"
            )

    provider = NativeWaitProvider()
    recipient.wait_registry.register(provider)
    outbound_db = AsyncDatabase(backend)
    await outbound_store.ensure_a2a_outbound_tasks_table(outbound_db)
    await outbound_store.record_outbound_dispatch(
        outbound_db,
        agent_id=did,
        task_id=task.id,
        recipient="native-peer",
        recipient_agent_id=did,
        verb="task",
        session_id=task.sessionId,
        dispatch_tool="native_wait_fixture",
    )

    @dataclass
    class NativeOutboundView:
        # Actual initialized outbound store and real native callback are
        # provider inputs, NOT peer HTTP/signature or live Agent proof.
        agent: RecipientView
        _db: AsyncDatabase
        _outbound_route_store_ready: bool = True

        async def get_peer_task_result(self, recipient, handle):
            return await manager.get_task_for_recipient(handle, did)

    a2a_provider = A2AWaitable(NativeOutboundView(recipient, outbound_db))
    recipient.wait_registry.register(a2a_provider)
    original_outbound_get = outbound_store.get_outbound_task
    original_get = manager.get_task_for_recipient
    original_release = asyncpg.pool.PoolAcquireContext.__aexit__
    original = OSError("original native recipient read release ACK lost")
    errors, releases = [], []

    async def observed_get(*args, **kwargs):
        try:
            return await original_get(*args, **kwargs)
        except BaseException as error:
            errors.append(error)
            raise

    async def lose_ack(checkout, error_type, error, tb):
        result = await original_release(checkout, error_type, error, tb)
        releases.append(checkout)
        if len(releases) == (2 if route == "a2a-peer" else 1) and error_type is None:
            raise original
        return result

    async def observed_outbound_get(*args, **kwargs):
        try:
            return await original_outbound_get(*args, **kwargs)
        except BaseException as error:
            errors.append(error)
            raise

    failure = None
    try:
        backend._execution_custody = ExecutionCustody(GenerationFence())
        monkeypatch.setattr(manager, "get_task_for_recipient", observed_get)
        monkeypatch.setattr(outbound_store, "get_outbound_task", observed_outbound_get)
        monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
        try:
            if route == "task-own":
                await task_provider.owns_handle(task.id)
            elif route == "a2a-own":
                await a2a_provider.owns_handle(task.id)
            elif route in {"a2a-poll", "a2a-peer"}:
                await a2a_provider.poll(task.id)
            elif route == "origin":
                await _provider_origin_session(provider, task.id)
            else:
                kind = (
                    "task"
                    if route == "task-register"
                    else "a2a"
                    if route == "a2a-register"
                    else "native"
                )
                await register_wait_watch(recipient, kind + ":" + task.id)
        except BaseException as error:
            failure = error
        watch_rows = await storage.db.fetchone(
            "SELECT count(*) AS n FROM wait_signal_state"
        )
        assert releases and errors and is_execution_control_error(errors[0])
        assert execution_commit_outcome(errors[0]) == "committed"
        assert carries(errors[0], original)
        assert failure is not None and carries(failure, errors[0]), (
            f"actual native recipient control became unverifiable ownership/session; durable watches={watch_rows}"
        )
        assert execution_commit_outcome(failure) == "committed"
        assert watch_rows == (0,), "a native ownership failure admitted a durable watch"
    finally:
        backend._execution_custody = None
        monkeypatch.setattr(
            asyncpg.pool.PoolAcquireContext, "__aexit__", original_release
        )
        await feature.shutdown()
        await manager.close()
        await storage.close()
