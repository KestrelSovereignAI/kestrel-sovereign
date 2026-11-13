"""#4001: a turn parked before its first LLM call, and Stop reaching it.

On 2026-11-09 and 11-10 Emma (twice) and Claw each had a turn begin and then
hold CONVERSATION, for up to 17 hours, without an LLM call, a tool call or a
log line. Later turns, and a scheduled tool holding the memory lock,
queued behind it. ``kestrel stop`` reported the agents ``unreachable``.

The reproduced cycle: a turn's effect-journal retirement holds the
privacy-transition lock after the turn releases CONVERSATION, and joins every
Task the turn created. A dispatched signal's COGNITION route is one of those
and waits for CONVERSATION; a turn that took CONVERSATION in between waits for
the privacy lock. A dispatched signal is now created outside the dispatching
turn's privacy lifetime; those regressions are in
``test_wake_detached_from_enclosing_turn.py`` (#4023). This file covers Stop
reaching a parked turn or a parked scheduled tool, and the locks a failed
required audit leaves behind.

Every agent here is real, on real storage; only the model is scripted. Each
scenario runs on SQLite and on PostgreSQL. The PostgreSQL leg owns a schema
of its own and is a ``native_pg`` consumer, so a run with
``KESTREL_REQUIRE_POSTGRES_TESTS=1`` fails it rather than skipping it
(tests/shared/postgres_requirement.py).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from contextlib import asynccontextmanager

import asyncpg
import pytest
from kestrel_sdk.signals import (
    ResourceLock,
)
from kestrel_sdk.storage.database.interface import QueryError

from kestrel_sovereign.agent.invocation import InvocationCancelledError
from kestrel_sovereign.await_chain import format_await_chain
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.stop import StopCleanupRegistry
from kestrel_sovereign.stop.authority import CancellationAuthority
from kestrel_sovereign.stop.receipt import StopReceiptStore
from kestrel_sovereign.stop.runtime_target import build_runtime_stop_target
from kestrel_sovereign.stop.types import (
    StopDisposition,
    StopDoor,
    StopRequest,
    StopScope,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.shared.booted_agent import booted_agent
from tests.utils.postgres_schema import (
    disposable_postgres_schema,
    pgvector_schema,
    quoted_search_path,
    with_search_path,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(60)]

REPLY = "Noted."
#: Each phase of a scenario must settle well inside this. A deadlock does not.
SETTLE_SECONDS = 10.0


async def _no_schema_ddl(database):
    """The admin connection boots no schema of its own on the shared database."""


@pytest.fixture(params=["sqlite", "postgres"])
def agent_backend(request):
    """The engine this case's agent keeps all of its state on."""
    return request.param


@pytest.fixture
async def native_pg(agent_backend):
    """The URL of this case's own PostgreSQL schema, or ``None`` on SQLite.

    Every scenario requests this fixture by name. That name marks the case
    ``native_postgres``, the selection of the required native PostgreSQL CI
    lane, and makes a skipped PostgreSQL leg a failure under
    ``KESTREL_REQUIRE_POSTGRES_TESTS=1`` (tests/shared/postgres_requirement.py).
    In CI these cases run in the unit tier, which provides PostgreSQL and sets
    that variable; the native lane collects only ``tests/integration/``.
    pytest resolves fixtures per function, so the SQLite leg carries the name
    too; it never skips, so the mark cannot turn it into a failure.

    The schema is dropped at teardown, so the case changes no row it did not
    create. pgvector's schema is on the path, as the agent's stores need it
    (#3401).
    """
    if agent_backend == "sqlite":
        yield None
        return
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL required for the PostgreSQL leg")
    admin = await AsyncDatabase.postgres(
        url, min_pool_size=1, max_pool_size=2, schema_initializer=_no_schema_ddl
    )
    try:
        vector = await pgvector_schema(admin)
        async with disposable_postgres_schema(admin, "parked_turn") as schema:
            yield with_search_path(url, quoted_search_path(schema, vector))
    finally:
        await admin.close()


def _script_replies(agent, monkeypatch, generate=None):
    async def generate_with_messages(**kwargs):
        if generate is not None:
            return await generate(kwargs)
        return LLMResponse(content=REPLY)

    async def stream_with_tool_detection(**kwargs):
        yield REPLY

    monkeypatch.setattr(agent.llm_service, "generate_with_messages", generate_with_messages)
    monkeypatch.setattr(
        agent.llm_service, "stream_with_tool_detection", stream_with_tool_detection
    )


def _await_chains(agent) -> str:
    """Where every live turn and lock holder is parked, for a failure message."""
    chains = []
    for task in asyncio.all_tasks():
        if task.get_name().startswith(("invocation-turn", "turn-", "signal_dispatch")):
            chains.append(format_await_chain(task))
    return (
        f"holds: {agent._get_lock_manager().active_hold_diagnostics()}\n"
        f"privacy lock held: {agent._get_privacy_transition_lock().locked()}\n"
        + "\n".join(chains)
    )


async def _settle(agent, tasks, label):
    done, pending = await asyncio.wait(set(tasks), timeout=SETTLE_SECONDS)
    if pending:
        chains = _await_chains(agent)
        for task in pending:
            task.cancel()
        # A deadlocked turn can absorb its cancellation; report, do not wait.
        await asyncio.wait(pending, timeout=1.0)
        pytest.fail(f"{label}: {len(pending)} task(s) never finished\n{chains}")
    return done


async def _park_after_begin(agent, monkeypatch):
    """Park the next turn on a future nothing resolves, just after ``begin``.

    The genesis gate is the turn's first await inside the turn lifecycle,
    before the privacy span, the context build or any model call.
    """
    park = asyncio.get_running_loop().create_future()
    parked = asyncio.Event()
    armed = {"armed": True}
    real_gate = agent._genesis_audit_cognition_block

    async def gate(user_input):
        if agent._live_turn_id is not None and armed["armed"]:
            armed["armed"] = False
            parked.set()
            await park
        return await real_gate(user_input)

    monkeypatch.setattr(agent, "_genesis_audit_cognition_block", gate)
    return parked


def _authority(agent, receipts) -> CancellationAuthority:
    return CancellationAuthority(
        lambda: (build_runtime_stop_target(agent, agent_id=agent.agent_id),),
        cleanup_registry=StopCleanupRegistry(),
        receipt_store=receipts,
        door=StopDoor.AGENT,
    )


@asynccontextmanager
async def _receipt_store(tmp_path, database_url):
    """The Stop receipt store, on the engine the agent uses.

    On PostgreSQL it is the agent's database, as the host keeps it there.
    """
    if database_url is None:
        database = await AsyncDatabase.sqlite(str(tmp_path / "stop-receipts.db"))
    else:
        database = await AsyncDatabase.postgres(
            database_url, min_pool_size=1, max_pool_size=2
        )
    try:
        await _assert_on_engine(database.backend, database_url)
        store = StopReceiptStore(database)
        await store.ensure_schema()
        yield store
    finally:
        await database.close()


async def _assert_on_engine(backend, database_url):
    """*backend* is on the engine this leg names, and on PostgreSQL in the
    case's own schema: a leg that silently fell back to SQLite, or to the
    shared schema, would otherwise pass as the PostgreSQL case."""
    expected = "sqlite" if database_url is None else "postgres"
    assert backend.backend_type == expected, (
        f"{backend!r} is on {backend.backend_type}, not {expected}"
    )
    if database_url is not None:
        schema = await backend.fetch_val("SELECT current_schema()")
        assert schema.startswith("parked_turn_"), schema


async def _assert_agent_on_engine(agent, database_url):
    """The agent's storage, effect journal and dispatch audit are all on it."""
    for backend in (
        agent._raw_storage.db.backend,
        agent.tool_effect_store.database.backend,
        agent.observability_store.backend,
    ):
        await _assert_on_engine(backend, database_url)


def _dispatch_audit_write_failure(database_url) -> QueryError:
    """What the backend raises when the dispatch row's column is missing.

    The 11-09 shape (#4002): the row names a column the table does not have.
    Each backend raises ``QueryError`` caused by its own driver's error.
    """
    if database_url is None:
        cause = sqlite3.OperationalError(
            "table a2a_tool_dispatches has no column named dispatch_id"
        )
    else:
        cause = asyncpg.exceptions.UndefinedColumnError(
            'column "dispatch_id" of relation "a2a_tool_dispatches" does not exist'
        )
    error = QueryError(f"Query failed: {cause}")
    error.__cause__ = cause
    return error


async def test_a_turn_parked_after_begin_is_stopped_and_the_queued_turn_runs(
    native_pg, tmp_path, monkeypatch
):
    async with _receipt_store(tmp_path, native_pg) as receipts:
        async with booted_agent(
            tmp_path / "agent",
            name="Parked",
            provenance="test:4001-turn",
            features=set(),
            database_url=native_pg,
        ) as agent:
            await _assert_agent_on_engine(agent, native_pg)
            _script_replies(agent, monkeypatch)
            parked = await _park_after_begin(agent, monkeypatch)
            first = asyncio.create_task(agent.process_input("one", session_id="s"))
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
            parked_turn = agent._live_turn_id
            second = asyncio.create_task(agent.process_input("two", session_id="s"))
            while agent._get_lock_manager().active_hold_diagnostics()[0][
                "blocked_acquirers"
            ] == 0:
                await asyncio.sleep(0.01)

            request = StopRequest(
                scope=StopScope.TURN,
                actor_id="operator",
                target=parked_turn,
                target_agent_id=agent.agent_id,
                target_is_turn_id=True,
            )
            outcomes = await asyncio.wait_for(
                _authority(agent, receipts).stop(request), SETTLE_SECONDS
            )

            assert [outcome.disposition for outcome in outcomes] == [
                StopDisposition.STOPPED
            ], outcomes
            await _settle(agent, (first, second), "the stopped and queued turns")
            assert isinstance(first.exception(), InvocationCancelledError)
            assert second.result() == REPLY
            assert not agent._get_lock_manager().is_held(ResourceLock.CONVERSATION)
            receipt = await receipts.load(request)
            assert receipt is not None
            assert [outcome.disposition for outcome in receipt.outcomes] == [
                StopDisposition.STOPPED
            ]


async def test_agent_stop_of_a_turn_parked_after_begin_is_receipted_stopped(
    native_pg, tmp_path, monkeypatch
):
    """``kestrel stop <agent>``: every current unit stops, and says so."""
    async with _receipt_store(tmp_path, native_pg) as receipts:
        async with booted_agent(
            tmp_path / "agent",
            name="Parked",
            provenance="test:4001-agent",
            features=set(),
            database_url=native_pg,
        ) as agent:
            await _assert_agent_on_engine(agent, native_pg)
            _script_replies(agent, monkeypatch)
            parked = await _park_after_begin(agent, monkeypatch)
            first = asyncio.create_task(agent.process_input("one", session_id="s"))
            await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
            queued = asyncio.create_task(agent.process_input("two", session_id="s"))
            while agent._get_lock_manager().active_hold_diagnostics()[0][
                "blocked_acquirers"
            ] == 0:
                await asyncio.sleep(0.01)

            request = StopRequest(
                scope=StopScope.AGENT,
                actor_id="operator",
                target=agent.agent_id,
                target_agent_id=agent.agent_id,
            )
            outcomes = await asyncio.wait_for(
                _authority(agent, receipts).stop(request), SETTLE_SECONDS
            )

            assert [outcome.disposition for outcome in outcomes] == [
                StopDisposition.STOPPED
            ], outcomes
            await _settle(agent, (first, queued), "the stopped turns")
            assert isinstance(first.exception(), InvocationCancelledError)
            assert isinstance(queued.exception(), InvocationCancelledError)
            assert not agent._get_lock_manager().is_held(ResourceLock.CONVERSATION)
            assert await asyncio.wait_for(
                agent.process_input("three", session_id="s"), SETTLE_SECONDS
            ) == REPLY


async def test_a_failed_required_audit_leaves_the_next_turn_runnable(
    native_pg, tmp_path, monkeypatch
):
    """The 11-09 trigger shape (#4002): a required dispatch audit write raises
    during a tool batch. The next turn still takes both locks and finishes."""
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError

    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    async with booted_agent(
        tmp_path / "agent",
        name="Audited",
        provenance="test:4001-audit",
        features=set(),
        database_url=native_pg,
    ) as agent:

        async def generate(kwargs):
            messages = kwargs.get("messages") or [{}]
            if messages[-1].get("role") != "tool":
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="status-1", name="wait_status", arguments={"limit": 5}),
                        ToolCall(id="status-2", name="wait_status", arguments={"limit": 1}),
                    ],
                )
            return LLMResponse(content=REPLY)

        _script_replies(agent, monkeypatch, generate)
        await _assert_agent_on_engine(agent, native_pg)
        store = agent.observability_store
        real_log = store.log_tool_dispatch
        failure = _dispatch_audit_write_failure(native_pg)
        failures = {"left": 1}

        async def log_tool_dispatch(entry, **kwargs):
            if failures["left"]:
                failures["left"] -= 1
                raise failure
            return await real_log(entry, **kwargs)

        monkeypatch.setattr(store, "log_tool_dispatch", log_tool_dispatch)
        with pytest.raises(ToolAuditUnconfirmedError) as unconfirmed:
            await asyncio.wait_for(
                agent.process_input("check waits", session_id="s"), SETTLE_SECONDS
            )
        assert failures["left"] == 0
        assert unconfirmed.value.__cause__ is failure
        assert not agent._get_lock_manager().is_held(ResourceLock.CONVERSATION)
        assert not agent._get_privacy_transition_lock().locked()

        following = asyncio.create_task(agent.process_input("again", session_id="s"))
        await _settle(agent, (following,), "the turn after the failed audit")
        assert REPLY in following.result()


async def test_a_scheduled_tool_holding_the_memory_lock_before_its_body_is_stopped(
    native_pg, native_tool_effect_agent, monkeypatch
):
    """The 03:00Z ``training_cycle`` shape: a cron dispatch holds the memory
    lock and its tool parks before the body runs. Stop cancels it, the memory
    lock is released, and the body never runs."""
    import json
    from datetime import UTC, datetime, timedelta

    from kestrel_sovereign.features.base import Feature, tool
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    from kestrel_sovereign.features.scheduler.runner import (
        SCHEDULER_PROTOCOL_VERSION,
        SchedulerRunner,
    )
    from tests.utils.scheduler_ticks import tick_and_settle

    owner = await native_tool_effect_agent(
        did="did:test:4001-scheduled",
        connected_backend=(
            None
            if native_pg is None
            else PostgresBackend(native_pg, min_pool_size=1, max_pool_size=4)
        ),
    )
    database = owner.tool_effect_store.database
    await _assert_on_engine(database.backend, native_pg)
    bodies = []

    class TrainingFeature(Feature):
        @property
        def tool_description(self):
            return "A scheduled tool that needs the memory lock"

        async def initialize(self):
            pass

        @tool(name="training_cycle_4001", description="Run one training cycle")
        async def training_cycle_4001(self) -> dict:
            bodies.append(True)
            return {"success": True}

    feature = TrainingFeature(owner)
    owner.features[feature.name] = feature
    scheduler = SchedulerFeature(owner)
    scheduler._db, scheduler._agent_id = database, owner.did

    park = asyncio.get_running_loop().create_future()
    parked = asyncio.Event()
    real_body = scheduler._run_tool_hook_gated_body

    async def body_after_park(*args, **kwargs):
        parked.set()
        await park
        return await real_body(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_run_tool_hook_gated_body", body_after_park)
    locks = owner._get_lock_manager()

    async def execute(name, arguments):
        # The dispatcher's cron route takes the source's declared resources
        # before it runs the tool, as it does for ``cron.training_cycle``.
        async with locks.acquire({ResourceLock.MEMORY}, label=f"cron.{name}"):
            return await scheduler._lookup_and_run_tool(name, arguments)

    runner = SchedulerRunner(database, owner.did, execute, owner_id="scheduler-4001")
    try:
        await runner._ensure_tables()
        due = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
        await database.execute(
            """INSERT INTO scheduled_tasks
            (id,agent_id,task_name,cron_expression,args_json,enabled,next_run_at,
             created_at,schedule_kind,timezone_name,misfire_policy,idempotency_key,
             scheduler_protocol_version)
            VALUES (?,?,?,?,?,1,?,?,'cron','UTC','skip',?,?)""",
            ("schedule-4001", owner.did, "training_cycle_4001", "0 3 * * *",
             json.dumps({}), due, due, "schedule-4001-source", SCHEDULER_PROTOCOL_VERSION),
        )
        tick = asyncio.create_task(tick_and_settle(runner))
        await asyncio.wait_for(parked.wait(), SETTLE_SECONDS)
        assert locks.is_held(ResourceLock.MEMORY)

        agent_id = getattr(owner, "agent_id", None) or owner.did
        target = build_runtime_stop_target(owner, agent_id=agent_id)
        disposition = await asyncio.wait_for(
            target.cancel(
                StopRequest(
                    scope=StopScope.AGENT,
                    actor_id="operator",
                    target=agent_id,
                    target_agent_id=agent_id,
                )
            ),
            SETTLE_SECONDS,
        )

        assert disposition is StopDisposition.STOPPED
        await asyncio.wait_for(tick, SETTLE_SECONDS)
        assert not locks.is_held(ResourceLock.MEMORY)
        assert bodies == []
        (status,) = await database.fetchone(
            "SELECT status FROM task_execution_log WHERE task_id=?", ("schedule-4001",)
        )
        assert status != "success"
    finally:
        await runner.stop()
