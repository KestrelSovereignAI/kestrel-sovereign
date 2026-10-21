"""Actual pre-turn compaction delegates through native context storage.

Codex occupancy is a declared component input to the real adapter cache; no
Codex session, provider subscription request, Agent genesis or READY is claimed.
"""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["read", "marker", "metadata"])
@pytest.mark.parametrize("kind", ["native-control", "ordinary", "healthy"])
async def test_actual_preturn_compaction_keeps_original_control(
    native_memory_owners, monkeypatch, door, kind
):
    n = native_memory_owners
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    n.service.providers[0]["model"] = "gpt-4o"
    n.service.set_model_preference(
        "gpt-4o", vendor="openai", route="local-fixture", validate=False
    )
    session = "native-preturn"
    for index in range(10):
        await n.pg_storage.conversation.add_conversation(
            "user", f"Native preturn context {index}. " * 6, session_id=session
        )
    context = ContextManager(
        n.pg_storage, model="gpt-4o", agent_id=n.did, llm_service=n.service
    )
    codex = CodexAdapter()
    codex._session_threads[session] = ("declared-component-thread", "component")
    codex._record_thread_occupancy(
        session,
        {
            "last": {"inputTokens": 900},
            "modelContextWindow": 1000,
        },
    )
    n.service.providers.append(
        {"name": "openai:plan", "vendor": "openai", "route": "plan", "adapter": codex}
    )
    agent = SimpleNamespace(llm_service=n.service, context_manager=context)
    agent._codex_adapter = lambda: KestrelAgent._codex_adapter(agent)
    agent._codex_compact_threshold_pct = lambda: (
        KestrelAgent._codex_compact_threshold_pct(agent)
    )

    db, pool, armed = n.pg_storage.db, n.backend._pool, False
    reached, sources, fired = [], [], []
    original = OSError("actual preturn compaction native release acknowledgement lost")

    @asynccontextmanager
    async def lose_ack(native):
        async with native as connection:
            yield connection
        if armed and not fired:
            fired.append(True)
            raise original

    target = {
        "read": "FROM CONVERSATION_HISTORY",
        "marker": "INSERT INTO CONVERSATION_HISTORY",
        "metadata": "UPDATE CONVERSATION_HISTORY",
    }[door]
    for name in ("fetchall", "fetchone", "execute", "execute_commit"):
        actual = getattr(db, name)

        async def intercept(sql, *args, _actual=actual, **kwargs):
            nonlocal armed
            selected = target in " ".join(sql.upper().split()) and not reached
            if selected:
                reached.append(sql)
                armed = kind == "native-control"
            try:
                result = await _actual(sql, *args, **kwargs)
            except BaseException as error:
                if selected:
                    sources.append(error)
                raise
            if selected and kind == "ordinary":
                raise OSError("ordinary preturn callback unavailable after ACK")
            return result

        monkeypatch.setattr(db, name, intercept)
    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    if kind == "native-control":
        n.backend._pool = PostgresPoolAdapter(pool, lose_ack)
    failure = None
    try:
        try:
            await KestrelAgent._maybe_compact_codex_thread(agent, session)
        except BaseException as error:
            failure = error
        assert reached, (
            f"actual compaction producer was not reached: {door} {failure!r}"
        )
        rows = await n.observer.fetch(
            f'SELECT metadata FROM "{n.schema}".conversation_history WHERE agent_id=$1 ORDER BY id',
            n.did,
        )
        if kind == "native-control":
            assert (
                fired
                and sources
                and execution_commit_outcome(sources[0]) == "committed"
            )
            assert original in tuple(_explicit_control_chain(sources[0]))
            assert failure is not None and is_execution_control_error(failure), (
                f"actual preturn compaction control was discarded: {door} {failure!r}"
            )
            assert sources[0] in tuple(_explicit_control_chain(failure))
            assert session in codex._session_threads, (
                "uncertain compaction cannot reset native thread"
            )
            assert len(rows) == (10 if door == "read" else 11)
            excluded = sum(
                bool(json.loads(r[0]).get("excluded_from_context")) for r in rows
            )
            assert excluded == (1 if door == "metadata" else 0)
        else:
            assert failure is None
            assert (session in codex._session_threads) is (kind == "ordinary")
            if kind == "healthy":
                assert len(rows) == 11
                assert (
                    sum(
                        bool(json.loads(r[0]).get("excluded_from_context"))
                        for r in rows
                    )
                    == 4
                )
        assert len(n.requests) == (0 if door == "read" and kind != "healthy" else 1)
        assert codex._client is None, (
            "component cache test must not start a Codex process"
        )
    finally:
        n.backend._pool = pool
        n.backend._execution_custody = None
