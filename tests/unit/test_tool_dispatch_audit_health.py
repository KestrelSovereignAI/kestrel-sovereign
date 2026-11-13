"""#4002: health reports when a dispatch-audit row last landed and how many failed.

From 2026-11-06 every ``a2a_tool_dispatches`` write on every existing agent
database failed, and the best-effort writer only printed to stderr. Nothing
an operator reads counted it until #3999 made the audit required and every
tool call failed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
    ToolDispatchEntry,
)
from kestrel_sovereign.features.health import checks as health_checks
from kestrel_sovereign.features.health.checks import (
    check_tool_dispatch_audit,
    run_standard_checks,
)
from kestrel_sovereign.storage.db import TransactionError
from kestrel_sovereign.storage.db.interface import QueryError
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


def _entry(dispatch_id: str = "dispatch-1") -> ToolDispatchEntry:
    return ToolDispatchEntry(
        agent_did="did:test:audit",
        session_id=None,
        turn_id="turn-1",
        tool_name="wait_status",
        adapter="direct",
        args_redacted={"limit": 5},
        result_status="success",
        error_class=None,
        error_message=None,
        latency_ms=1,
        result_size_bytes=None,
        dispatch_id=dispatch_id,
    )


@pytest.fixture
async def store(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "audit.db"))
    await backend.connect()
    value = ObservabilityStore(backend)
    try:
        await value.initialize()
        yield value
    finally:
        await value.close()


async def _break_audit_table(store: ObservabilityStore) -> None:
    """The pre-#3995 table, which the current writer cannot insert into."""
    for index in ("idx_tool_dispatches_dispatch", "idx_tool_dispatches_parent"):
        await store.backend.execute(f"DROP INDEX {index}")
    await store.backend.execute("ALTER TABLE a2a_tool_dispatches DROP COLUMN dispatch_id")


class _RollBack(Exception):
    """Raised inside a transaction so that it rolls back."""


async def _dispatch_ids(store: ObservabilityStore) -> list:
    rows = await store.backend.fetch_all(
        "SELECT dispatch_id FROM a2a_tool_dispatches ORDER BY id"
    )
    return [row[0] for row in rows]


async def test_a_fresh_store_has_audited_nothing(store):
    health = store.tool_dispatch_audit_health
    assert (health.last_success_at, health.failures) == (None, 0)
    result = await check_tool_dispatch_audit(SimpleNamespace(observability_store=store))
    assert result["status"] == "pass"
    assert result["details"] == {"last_success_at": None, "failures": 0}
    assert "No tool dispatch audited" in result["message"]


async def test_a_landed_row_records_its_time(store):
    before = datetime.now(timezone.utc)
    await store.log_tool_dispatch(_entry(), required=True)
    health = store.tool_dispatch_audit_health
    assert health.failures == 0
    assert before <= health.last_success_at <= datetime.now(timezone.utc)
    result = await check_tool_dispatch_audit(SimpleNamespace(observability_store=store))
    assert result["status"] == "pass"
    assert result["details"]["last_success_at"] == health.last_success_at.isoformat()


async def test_a_row_written_in_a_caller_transaction_lands_at_its_commit(store):
    async with store.backend.transaction():
        await store.log_tool_dispatch(_entry("joined"), required=True)
        # Written, not landed: it commits or rolls back with the caller.
        assert store.tool_dispatch_audit_health.last_success_at is None
        committing_at = datetime.now(timezone.utc)
    assert await _dispatch_ids(store) == ["joined"]
    health = store.tool_dispatch_audit_health
    assert health.failures == 0
    assert committing_at <= health.last_success_at <= datetime.now(timezone.utc)


async def test_a_rolled_back_caller_transaction_reports_no_landed_row(store):
    with pytest.raises(TransactionError):
        async with store.backend.transaction():
            await store.log_tool_dispatch(_entry("rolled-back"), required=True)
            raise _RollBack()

    assert await _dispatch_ids(store) == []
    health = store.tool_dispatch_audit_health
    assert (health.last_success_at, health.failures) == (None, 0)
    result = await check_tool_dispatch_audit(SimpleNamespace(observability_store=store))
    assert result["details"] == {"last_success_at": None, "failures": 0}
    assert "No tool dispatch audited" in result["message"]


async def test_a_rolled_back_savepoint_reports_no_landed_row(store):
    async with store.backend.transaction():
        with pytest.raises(_RollBack):
            async with store.backend.transaction(savepoint=True):
                await store.log_tool_dispatch(_entry("undone"), required=True)
                raise _RollBack()
    assert await _dispatch_ids(store) == []
    assert store.tool_dispatch_audit_health.last_success_at is None

    async with store.backend.transaction():
        async with store.backend.transaction(savepoint=True):
            await store.log_tool_dispatch(_entry("released"), required=True)
        assert store.tool_dispatch_audit_health.last_success_at is None
    assert await _dispatch_ids(store) == ["released"]
    assert store.tool_dispatch_audit_health.last_success_at is not None


class _CommitBlindBackend:
    """A backend that writes but cannot report commits or an open transaction."""

    backend_type = "sqlite"
    owns_open_transaction = None

    def __init__(self):
        self.statements = []

    async def execute(self, sql, params=()):
        self.statements.append(sql)


async def test_a_row_whose_commit_cannot_be_observed_is_not_reported_landed():
    backend = _CommitBlindBackend()
    store = ObservabilityStore(backend, owns_backend=False)
    await store.log_tool_dispatch(_entry(), required=True)
    assert any("INSERT INTO a2a_tool_dispatches" in sql for sql in backend.statements)
    health = store.tool_dispatch_audit_health
    assert (health.last_success_at, health.failures) == (None, 0)


async def test_a_required_write_that_fails_is_counted_and_raised(store):
    await store.log_tool_dispatch(_entry("landed"), required=True)
    landed_at = store.tool_dispatch_audit_health.last_success_at
    await _break_audit_table(store)

    with pytest.raises(QueryError, match="dispatch_id"):
        await store.log_tool_dispatch(_entry("lost"), required=True)

    health = store.tool_dispatch_audit_health
    assert health.failures == 1
    assert health.last_success_at == landed_at
    assert health.last_failure_at >= landed_at
    # The class and a content-free summary, never the driver's message text.
    assert health.last_error.startswith("QueryError ")
    assert "dispatch_id" not in health.last_error

    result = await check_tool_dispatch_audit(SimpleNamespace(observability_store=store))
    assert result["status"] == "warn"
    assert result["details"]["failures"] == 1
    assert result["details"]["last_success_at"] == landed_at.isoformat()
    assert "the most recent write failed" in result["message"]


async def test_a_best_effort_write_that_fails_is_counted_and_logged(store, caplog):
    await _break_audit_table(store)
    with caplog.at_level(logging.WARNING):
        await store.log_tool_dispatch(_entry())
    assert store.tool_dispatch_audit_health.failures == 1
    assert any(
        "a2a_tool_dispatches write failed for wait_status" in record.getMessage()
        for record in caplog.records
    )


async def test_failures_stay_counted_after_a_later_row_lands(store):
    async def failing_execute(*_args, **_kwargs):
        raise RuntimeError("disk I/O error")

    original = store.backend.execute
    store.backend.execute = failing_execute
    try:
        await store.log_tool_dispatch(_entry("lost"))
    finally:
        store.backend.execute = original
    await store.log_tool_dispatch(_entry("landed"), required=True)

    health = store.tool_dispatch_audit_health
    assert health.failures == 1
    result = await check_tool_dispatch_audit(SimpleNamespace(observability_store=store))
    assert result["status"] == "warn"
    assert f"the last row landed at {health.last_success_at.isoformat()}" in result["message"]


async def test_an_agent_without_the_store_passes():
    result = await check_tool_dispatch_audit(SimpleNamespace(observability_store=None))
    assert result["status"] == "pass"
    assert "No tool dispatch audit store" in result["message"]


async def test_every_health_surface_runs_the_check(store):
    await _break_audit_table(store)
    await store.log_tool_dispatch(_entry())
    agent = SimpleNamespace(
        bootstrap_service=None,
        context_manager=None,
        dispatcher=None,
        llm_service=None,
        memory_retriever=None,
        memory_consolidator=None,
        storage=None,
        observability_store=store,
    )
    db = SimpleNamespace(fetchone=AsyncMock(return_value=(1,)))
    checks = await run_standard_checks(agent, db)
    audit = next(check for check in checks if check["name"] == "tool_dispatch_audit")
    assert audit["status"] == "warn"
    assert health_checks.derive_overall_status([audit]) == "degraded"
