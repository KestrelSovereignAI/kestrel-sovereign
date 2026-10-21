"""Actual native reads cannot become HTTP/tool or optional-context success.

The storage, context planner, salvage reader and TaskManager are real. Legacy
constitution/reflection and request/agent projections are component inputs, not
authenticated HTTP, canonical constitutional admission or Agent READY proof.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from uuid import uuid4

import pytest
from a2a.types import Message, Part, Role, SendMessageRequest
from fastapi import HTTPException, Request
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.agent import salvage
from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.endpoints.agent import compute_context_status
from kestrel_sovereign.endpoints.task_inbox import read_inbox
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.context.feature import ContextFeature
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.utils.native_a2a import manager_for


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door",
    [
        "history-http",
        "history-tool",
        "history-tool-typed",
        "constitution",
        "reflection",
        "salvage",
        "task",
    ],
)
@pytest.mark.parametrize("kind", ["native-control", "ordinary", "healthy"])
async def test_native_context_and_task_reads_preserve_control_before_shaping(
    native_memory_owners, monkeypatch, door, kind
):
    n = native_memory_owners
    session = "native-context-component"
    await n.pg_storage.db.execute(
        "INSERT INTO agent_metadata (agent_id,key,value) VALUES (?,?,?)",
        (
            n.did,
            "component-context",
            "Component governing text, not an authority grant.",
        ),
    )
    tasks = manager_for(n.backend, NativeRecipientScope(n.did), owns_backend=False)
    await tasks.initialize()
    task = await tasks.create_task(SendMessageRequest(message=Message(role=Role.ROLE_USER, parts=[Part(text='native stored task')], message_id=uuid4().hex, context_id=session)), creator_agent_id=n.did)
    builder = ContextBuilder(n.pg_storage, llm_service=n.service, model="gpt-4o")
    context = ContextManager(
        n.pg_storage,
        model="gpt-4o",
        agent_id=n.did,
        memory_retriever=n.pg_memory.retriever,
        llm_service=n.service,
        context_builder=builder,
    )
    pool = n.backend._pool
    original = OSError("original context native release acknowledgement lost")
    ordinary = OSError(
        "ordinary context callback unavailable after acknowledged native read"
    )
    armed, fired, sources, reached = False, [], [], []

    @asynccontextmanager
    async def fail_release(native):
        async with native as connection:
            yield connection
        if armed and not fired:
            fired.append(True)
            raise original

    async def at_read(operation, *args, **kwargs):
        nonlocal armed
        reached.append(True)
        if kind == "native-control":
            armed = True
        try:
            result = await operation(*args, **kwargs)
        except BaseException as error:
            sources.append(error)
            if door == "history-tool-typed":
                raise ValueError("declared typed context read carrier") from error
            raise
        if kind == "ordinary":
            if door == "history-tool-typed":
                raise ValueError("ordinary typed context read failure") from ordinary
            raise ordinary
        return result

    async def context_row():
        return await n.pg_storage.db.fetchval(
            "SELECT value FROM agent_metadata WHERE agent_id=? AND key=?",
            (n.did, "component-context"),
        )

    @dataclass
    class ContextProjection:
        did: str = n.did
        agent_id: str = n.did
        storage: object = field(default_factory=lambda: n.pg_storage)
        _raw_storage: object = field(default_factory=lambda: n.pg_storage)
        context_manager: object = field(default_factory=lambda: context)
        task_manager: object = field(default_factory=lambda: tasks)
        llm_service: object = field(default_factory=lambda: n.service)
        _active_session_id: str = session
        _privacy_mode: str = "NORMAL"
        _session_briefed: bool = False
        features: dict = field(default_factory=dict)

        def get_current_model(self):
            return "gpt-4o"

        def _build_all_tools(self):
            return []

        async def get_constitution(self):
            if door == "constitution":
                return await at_read(context_row)
            return await context_row()

    agent = ContextProjection()
    # Actual store supplied to the real salvage-count function, not fabricated
    # counts or a replacement plan/render result.
    agent.conversation_manager = SimpleNamespace(
        _get_conversation_store=lambda: n.pg_storage.conversation
    )
    if door.startswith("history-"):
        actual = n.pg_storage.get_conversation_history

        async def history(*args, **kwargs):
            return await at_read(actual, *args, **kwargs)

        monkeypatch.setattr(n.pg_storage, "get_conversation_history", history)
    elif door == "reflection":

        class Guidance:
            async def get_active_guidance(self):
                return [await at_read(context_row)]

        agent.features["ReflectionFeature"] = Guidance()
    elif door == "salvage":
        actual = salvage.get_salvage_state_counts

        async def counts(*args, **kwargs):
            return await at_read(actual, *args, **kwargs)

        monkeypatch.setattr(salvage, "get_salvage_state_counts", counts)
    elif door == "task":
        actual = tasks.get_task_snapshot

        async def read_task(*args, **kwargs):
            return await at_read(actual, *args, **kwargs)

        monkeypatch.setattr(tasks, "get_task_snapshot", read_task)

    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    if kind == "native-control":
        n.backend._pool = PostgresPoolAdapter(pool, fail_release)
    failure = result = None
    try:
        try:
            if door == "task":
                result = await read_inbox(
                    Request({"type": "http", "state": {"agent": agent}}), task.id
                )
            elif door.startswith("history-tool"):
                result = await ContextFeature(agent).context_status()
            else:
                result = await compute_context_status(agent, session, full=False)
        except BaseException as error:
            failure = error

        assert reached, (
            f"actual selected context/task read was not reached: {failure!r}"
        )
        if kind == "native-control":
            assert (
                fired
                and sources
                and execution_commit_outcome(sources[0]) == "committed"
            )
            assert original in tuple(_explicit_control_chain(sources[0]))
            assert failure is not None and is_execution_control_error(failure), (
                f"native context/task control became ordinary HTTP/tool/fallback output: "
                f"door={door} failure={failure!r} result={result!r}"
            )
            assert sources[0] in tuple(_explicit_control_chain(failure))
            assert execution_commit_outcome(failure) == "committed"
        elif kind == "ordinary" and door.startswith("history-tool"):
            assert failure is None and result.status is ToolResultStatus.ERROR
        elif kind == "ordinary" and door == "task":
            assert failure is ordinary
        elif kind == "ordinary" and door == "history-http":
            assert isinstance(failure, HTTPException) and failure.status_code == 500
            assert not is_execution_control_error(failure)
        else:
            assert failure is None, (
                f"ordinary availability/healthy path broke: {failure!r}"
            )
            if door.startswith("history-tool"):
                assert result.status is ToolResultStatus.OK
            else:
                assert isinstance(result, dict)
        assert n.requests == [], "cheap context/status readers must not call a provider"
        assert (
            await n.observer.fetchval(f'SELECT count(*) FROM "{n.schema}".a2a_tasks')
            == 1
        )
    finally:
        n.backend._pool = pool
        n.backend._execution_custody = None
        await tasks.close()
