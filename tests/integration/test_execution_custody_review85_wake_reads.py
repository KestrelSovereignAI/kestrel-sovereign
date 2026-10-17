"""New wake-reply reads preserve their actual PostgreSQL retirement owner."""

from contextvars import ContextVar
from types import SimpleNamespace
from uuid import uuid4

import asyncpg
import pytest
from fastapi import FastAPI, Request

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_runtime,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg

native_pg = _native_pg
pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("door", ["wake_readback", "message_endpoint"])
@pytest.mark.parametrize("failure_kind", ["native_reset", "retired_scope"])
async def test_wake_reply_reads_preserve_original_native_control(
    native_pg, monkeypatch, door, failure_kind,
):
    from kestrel_sovereign.endpoints.conversations import get_message

    backend, _control, _schema = native_pg
    agent_id, signal_id = "did:test:wake-" + uuid4().hex, uuid4().hex
    storage = AsyncStorage(backend=backend, agent_id=agent_id)
    await storage.initialize()
    await storage.add_conversation("assistant", "entire stored reply",
        metadata={"signal_wake_reply": signal_id, "session_id": "same-original-session"})
    message_id = await storage.find_signal_wake_reply_id(signal_id)
    assert isinstance(message_id, int)
    agent = SimpleNamespace(storage=storage, privacy_agent=SimpleNamespace(can_store=lambda _kind: True))
    app = FastAPI()
    app.state.agent = agent
    request = Request({"type": "http", "app": app, "method": "GET", "path": "/"})
    scope = ExecutionCustody(ProcessRuntimeExecutionFence(agent_id))
    owner = SimpleNamespace(_execution_custody=scope)
    marked = ContextVar("original-wake-read", default=False)
    failure = RuntimeError("original native wake-read RESET failed")
    original_reset = asyncpg.connection.Connection.reset
    failed_connections = []

    async def fail_original_reset(connection, *args, **kwargs):
        if marked.get() and not failed_connections:
            failed_connections.append(connection)
            raise failure
        return await original_reset(connection, *args, **kwargs)

    monkeypatch.setattr(asyncpg.connection.Connection, "reset", fail_original_reset)
    try:
        token = marked.set(True)
        try:
            with bind_execution_runtime(owner):
                if failure_kind == "retired_scope":
                    scope.revoke("original message read admission retired")
                with pytest.raises(Exception) as caught:
                    if door == "wake_readback":
                        await KestrelAgent._stored_wake_reply_id(agent, signal_id)
                    else:
                        await get_message(request, message_id)
        finally:
            marked.reset(token)
        error = caught.value
        assert is_execution_control_error(error), "native control was replaced by an optional/UI error"
        if failure_kind == "native_reset":
            debts = execution_cleanup_debts(error)
            assert len(failed_connections) == 1 and debts
            assert all(closer is not None and closer.done() for _resource, closer in debts)
            assert failed_connections[0].is_closed()
        else:
            assert isinstance(error, ExecutionAuthorityError)
    finally:
        await storage.close()
