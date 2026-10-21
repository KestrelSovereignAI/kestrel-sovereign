"""Original native receipts at the five review-53 consumer boundaries."""
import asyncio
from types import SimpleNamespace

import aiosqlite
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent import operator_signals
from kestrel_sovereign.agent.invocation import (
    invocation_scope,
    mark_current_invocation_effect_completed,
)
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, execution_cleanup_debts
from tests.utils.original_failure import OriginalFailure
from kestrel_sovereign.llm.model_discovery import ModelDiscoveryMixin
from kestrel_sovereign.storage.db import sqlite as sqlite_module
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review48_native import terminal
from tests.unit.test_execution_custody_review49_native import KINDS, join


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_repeated_cancel_retains_close_owner_after_actual_worker_exit(tmp_path, monkeypatch, kind):
    connection = await aiosqlite.connect(str(tmp_path / "close.db"))
    await connection.execute("CREATE TABLE effects(id INTEGER)")
    await connection.execute("INSERT INTO effects VALUES (1)")
    await connection.commit()
    native_close = connection.close
    closed, release = asyncio.Event(), asyncio.Event()
    original, retained = terminal(kind), {}

    async def native_close_then_late_receipt():
        await native_close()
        await sqlite_module._wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)
        closed.set()
        await await_owned_task(create_owned_task(release.wait()))
        raise original

    monkeypatch.setattr(connection, "close", native_close_then_late_receipt)
    owner = create_owned_task(sqlite_module._close_aiosqlite_connection(
        connection, retained_closes=retained,
        deadline=asyncio.get_running_loop().time() + 2,
    ))
    try:
        await asyncio.wait_for(closed.wait(), 2)
        assert not sqlite_module._aiosqlite_worker_is_alive(connection)
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        assert (await join(owner)).error is not None
        assert connection in retained, "native worker exit shed its unfinished original close owner"
        close_owner = retained[connection].close_task
        assert not close_owner.done()
        release.set()
        assert (await join(close_owner)).error is original
    finally:
        release.set()
        await join(owner)
        for record in list(retained.values()):
            await join(record.close_task)
            if record.retirement_task is not None:
                await join(record.retirement_task)
        monkeypatch.undo()
        await native_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_benign_interrupt_text_does_not_erase_explicit_native_receipt(tmp_path, monkeypatch, kind):
    connection = await aiosqlite.connect(str(tmp_path / "interrupt.db"))
    await connection.execute("CREATE TABLE effects(id INTEGER)")
    await connection.execute("INSERT INTO effects VALUES (1)")
    await connection.commit()
    native_close, native_interrupt = connection.close, connection.interrupt
    release, retained = asyncio.Event(), {}
    original = ValueError("no active connection")
    original.__cause__ = terminal(kind)
    source = OriginalFailure.capture(original)

    async def delayed_close():
        await await_owned_task(create_owned_task(release.wait()))
        await native_close()

    async def native_interrupt_then_receipt():
        await native_interrupt()
        raise original

    monkeypatch.setattr(connection, "close", delayed_close)
    monkeypatch.setattr(connection, "interrupt", native_interrupt_then_receipt)
    try:
        outcome = await join(create_owned_task(sqlite_module._close_aiosqlite_connection(
            connection, retained_closes=retained,
            deadline=asyncio.get_running_loop().time() + 0.02,
        )))
        source.assert_retained(outcome.error)
        record = retained[connection]
        # Admission may be cancelled before its first turn on a busy runner;
        # a done Python Task still cannot certify retirement of a live worker.
        assert record.close_task is not None
        assert sqlite_module._aiosqlite_worker_is_alive(connection)
        assert (connection, record.close_task) in execution_cleanup_debts(outcome.error)
    finally:
        release.set()
        for record in list(retained.values()):
            await join(record.close_task)
            if record.retirement_task is not None:
                await join(record.retirement_task)
        monkeypatch.undo()
        await native_close()
        await sqlite_module._wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("entry", ["generation", "cancelled_tool_loop"])
async def test_original_generation_receipt_wins_over_native_checkpoint_failure(tmp_path, monkeypatch, kind, entry):
    agent, db = await native_owner(tmp_path)
    original = terminal(kind)
    weak = ExecutionAuthorityError("checkpoint acknowledgement denied later")

    async def checkpoint(*args, **kwargs):
        await db.execute("INSERT INTO effects VALUES (2)")
        raise weak

    async def cancelled_loop(*args, **kwargs):
        await db.execute("INSERT INTO effects VALUES (1)")
        raise original

    monkeypatch.setattr(agent, "_persist_assistant_turn_safely", checkpoint)
    if entry == "cancelled_tool_loop" and kind != "cancel_unknown":
        original = asyncio.CancelledError("original tool-loop cancelled")
        original.__cause__ = terminal(kind)
    monkeypatch.setattr(OrchestratorEngineMixin, "_handle_orchestrator_response_impl", cancelled_loop)

    async def invoke():
        with invocation_scope():
            if entry == "generation":
                await db.execute("INSERT INTO effects VALUES (1)")
                mark_current_invocation_effect_completed("native")
                return await agent._settle_failed_generation(original, session_id="native")
            return await agent._handle_orchestrator_response(
                response="partial", feature_tools=[], system_prompt="", force_local_only=True,
                effective_model="offline", session_id="native", tool_results=[{"native": True}],
            )

    try:
        assert (await join(create_owned_task(invoke()))).error is original
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 2
    finally:
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("entry", ["audit", "fallback"])
async def test_operator_native_commit_receipt_is_not_best_effort_or_requeued(tmp_path, monkeypatch, kind, entry):
    agent, db = await native_owner(tmp_path)
    original = terminal(kind)
    producer = operator_signals.OperatorSignalProducer(agent)
    producer.enqueue_auto_mode("session")
    agent.operator_signal_producer = producer
    monkeypatch.setattr(operator_signals, "supports_inline_system_for_next_route", lambda *a, **k: (False, "offline"))

    async def native_acknowledgement(*args, **kwargs):
        await db.execute("INSERT INTO effects VALUES (1)")
        raise original

    if entry == "audit":
        monkeypatch.setattr(operator_signals.OperatorNoticeAuditStore, "record_collected", native_acknowledgement)
    else:
        # The actual producer/lifecycle owns its audit rows; only the native
        # fallback acknowledgement boundary injects the original receipt.
        agent.privacy_agent = SimpleNamespace(add_conversation=native_acknowledgement)
    try:
        outcome = await join(create_owned_task(operator_signals.inject_operator_turn(
            agent, [], SimpleNamespace(budget_summary=None, state_of_mind=None), "native", None, True,
        )))
        assert outcome.error is original
        assert producer._pending_auto_mode == [], "uncertain committed notice was requeued"
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_two_embedding_readers_receive_same_original_native_receipt(tmp_path, kind):
    agent, db = await native_owner(tmp_path)
    original, entered, release = terminal(kind), asyncio.Event(), asyncio.Event()

    class NativeDiscovery(ModelDiscoveryMixin):
        async def _discover_embedding_models_uncached(self):
            await db.execute("INSERT INTO effects VALUES (1)")
            entered.set()
            await release.wait()
            raise original

    service = NativeDiscovery()
    first = create_owned_task(service._discover_embedding_models_coalesced())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = create_owned_task(service._discover_embedding_models_coalesced())
        await asyncio.sleep(0)
        release.set()
        assert (await join(first)).error is original
        assert (await join(second)).error is original
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        release.set()
        await join(first)
        if second is not None:
            await join(second)
        await retire(agent)
        await db.close()
