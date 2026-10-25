"""Operational telemetry retention and database size reporting (#3785).

Meridian's database reached 1.89 GB for 708 messages. Almost all of it was
rows the agent writes about itself on a cadence and never removed:
``health_log`` (~860 MB), ``task_execution_log``, ``signal_log`` and
``restart_status_events``. These tests pin the retention policy, each owner's
purge predicate (on SQLite and PostgreSQL), the ``trash_retention`` rail that
runs them, and the size the liveness probe reports so an agent can see its
database grow.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.features.health.feature import purge_health_log
from kestrel_sovereign.features.restart_coordinator.event_store import (
    ensure_restart_status_events_table,
    purge_events_before,
)
from kestrel_sovereign.features.scheduler.execution_log_retention import (
    purge_execution_log,
)
from kestrel_sovereign.storage import footprint
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.retention import (
    DEFAULT_TELEMETRY_MAX_ROWS_PER_SWEEP,
    delete_rows_in_batches,
    resolve_telemetry_retention,
)
from tests.utils.postgres_schema import (
    disposable_postgres_schema,
    postgres_test_url,
    with_search_path,
)

AGENT = "did:test:telemetry-owner"
OTHER = "did:test:telemetry-neighbour"
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
CUTOFF = (NOW - timedelta(days=7)).isoformat()


def _at(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


async def _no_schema(_db):
    """Schema initializer for a connection that needs no core tables."""


# Parametrized with "postgres" so the #3381 guard fails the PostgreSQL case,
# rather than letting it skip, in a job that provides PostgreSQL.
@pytest.fixture(params=["sqlite", "postgres"])
async def db(request, tmp_path):
    if request.param == "sqlite":
        database = await AsyncDatabase.sqlite(
            str(tmp_path / "telemetry.db"), schema_initializer=_no_schema
        )
        try:
            yield database
        finally:
            await database.close()
        return

    url = postgres_test_url()
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    admin = PostgresBackend(url)
    await admin.connect()
    try:
        async with disposable_postgres_schema(admin, "telemetry") as schema:
            backend = PostgresBackend(with_search_path(url, schema))
            await backend.connect()
            try:
                yield AsyncDatabase(backend)
            finally:
                await backend.close()
    finally:
        await admin.close()


# --- the policy --------------------------------------------------------------


def test_policy_defaults_bound_every_table():
    policy = resolve_telemetry_retention({})
    assert policy.days == {
        "health_log": 7,
        "task_execution_log": 30,
        "restart_status_events": 30,
    }
    assert policy.max_rows_per_sweep == DEFAULT_TELEMETRY_MAX_ROWS_PER_SWEEP
    assert policy.cutoff("health_log", NOW) == _at(7)


def test_policy_honours_overrides_and_keeps_a_zero_window_forever():
    policy = resolve_telemetry_retention({
        "health_log_days": 2,
        "task_execution_log_days": 0,
        "restart_status_events_days": -1,
        "max_rows_per_sweep": 10,
    })
    assert policy.days["health_log"] == 2
    assert policy.cutoff("task_execution_log", NOW) is None
    assert policy.cutoff("restart_status_events", NOW) is None
    assert policy.max_rows_per_sweep == 10


@pytest.mark.parametrize("bad", ["7", 7.5, True, None])
def test_policy_falls_back_on_malformed_values(bad):
    """A typo must not disable the sweep that bounds the database."""
    policy = resolve_telemetry_retention({
        "health_log_days": bad, "max_rows_per_sweep": bad,
    })
    assert policy.days["health_log"] == 7
    assert policy.max_rows_per_sweep == DEFAULT_TELEMETRY_MAX_ROWS_PER_SWEEP


@pytest.mark.parametrize("cap", [0, -5])
def test_policy_refuses_a_non_positive_cap(cap):
    policy = resolve_telemetry_retention({"max_rows_per_sweep": cap})
    assert policy.max_rows_per_sweep == DEFAULT_TELEMETRY_MAX_ROWS_PER_SWEEP


# --- the batched delete ------------------------------------------------------


async def _create_health_log(db):
    await db.execute(
        """
        CREATE TABLE health_log (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            status TEXT NOT NULL,
            checks_json TEXT NOT NULL,
            overall_healthy INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


async def _health_row(db, row_id, agent_id, created_at):
    await db.execute(
        "INSERT INTO health_log VALUES (?, ?, 'healthy', '[]', 1, ?)",
        (row_id, agent_id, created_at),
    )


async def _ids(db, table):
    return sorted(row[0] for row in await db.fetchall(f"SELECT id FROM {table}"))


async def test_batched_delete_commits_in_batches_up_to_the_cap(db):
    await _create_health_log(db)
    for index in range(5):
        await _health_row(db, f"h{index}", AGENT, _at(30))
    statements = []
    real_execute = db.execute

    async def counting_execute(sql, params=()):
        statements.append(sql)
        return await real_execute(sql, params)

    db.execute = counting_execute
    try:
        deleted = await delete_rows_in_batches(
            db, table="health_log", predicate="candidate.agent_id = ?",
            params=(AGENT,), max_rows=4, batch_size=2,
        )
        assert deleted == 4
        assert len(statements) == 2
        assert len(await _ids(db, "health_log")) == 1

        statements.clear()
        assert await delete_rows_in_batches(
            db, table="health_log", predicate="candidate.agent_id = ?",
            params=(AGENT,), max_rows=10, batch_size=2,
        ) == 1
        assert len(statements) == 1, "a short batch ends the sweep"
    finally:
        db.execute = real_execute


# --- health_log --------------------------------------------------------------


async def test_health_log_purge_keeps_the_window_and_other_agents(db):
    await _create_health_log(db)
    await _health_row(db, "old", AGENT, _at(8))
    await _health_row(db, "edge", AGENT, _at(6.9))
    await _health_row(db, "new", AGENT, _at(0))
    await _health_row(db, "neighbour-old", OTHER, _at(30))

    assert await purge_health_log(
        db, agent_id=AGENT, older_than=CUTOFF, max_rows=100,
    ) == 1
    assert await _ids(db, "health_log") == ["edge", "neighbour-old", "new"]


@pytest.mark.parametrize("agent_id", ["", None])
async def test_health_log_purge_refuses_an_empty_scope(db, agent_id):
    await _create_health_log(db)
    await _health_row(db, "old", AGENT, _at(8))
    with pytest.raises(ValueError):
        await purge_health_log(
            db, agent_id=agent_id, older_than=CUTOFF, max_rows=10,
        )
    assert await _ids(db, "health_log") == ["old"]


# --- restart_status_events ---------------------------------------------------


async def _restart_event(db, row_id, request_id, agent_id, created_at):
    await db.execute(
        """
        INSERT INTO restart_status_events
            (id, request_id, state, agent_id, dedupe_signature,
             payload_json, created_at)
        VALUES (?, ?, 'deferred', ?, ?, '{}', ?)
        """,
        (row_id, request_id, agent_id, f"{request_id}:deferred", created_at),
    )


async def test_restart_event_purge_keeps_each_requests_newest_event(db):
    await ensure_restart_status_events_table(db)
    # A request polled for days: only its old polls go.
    await _restart_event(db, "r1-a", "req-1", AGENT, _at(40))
    await _restart_event(db, "r1-b", "req-1", AGENT, _at(35))
    await _restart_event(db, "r1-c", "req-1", AGENT, _at(1))
    # A request finished long ago: its last event is its final state.
    await _restart_event(db, "r2-a", "req-2", AGENT, _at(50))
    await _restart_event(db, "r2-final", "req-2", AGENT, _at(45))
    await _restart_event(db, "n-a", "req-3", OTHER, _at(50))
    await _restart_event(db, "n-b", "req-3", OTHER, _at(45))

    assert await purge_events_before(
        db, agent_id=AGENT, older_than=CUTOFF, max_rows=100,
    ) == 3
    assert await _ids(db, "restart_status_events") == [
        "n-a", "n-b", "r1-c", "r2-final",
    ]


@pytest.mark.parametrize("agent_id", ["", "  "])
async def test_restart_event_purge_refuses_an_empty_scope(db, agent_id):
    await ensure_restart_status_events_table(db)
    with pytest.raises(ValueError):
        await purge_events_before(
            db, agent_id=agent_id, older_than=CUTOFF, max_rows=10,
        )


# --- task_execution_log ------------------------------------------------------


async def _create_scheduler_tables(db):
    """The columns the purge predicate reads, with the runner's types."""
    await db.execute(
        """
        CREATE TABLE scheduled_tasks (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            claim_execution_id TEXT,
            failure_streak_last_execution_id TEXT
        )
        """
    )
    await db.execute(
        """
        CREATE TABLE task_execution_log (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            status TEXT NOT NULL,
            result_text TEXT,
            duration_ms INTEGER NOT NULL,
            executed_at TEXT NOT NULL,
            outcome_signal REAL
        )
        """
    )


async def _execution(db, row_id, task_id, executed_at, *, status="success",
                     agent_id=AGENT, outcome=None):
    await db.execute(
        "INSERT INTO task_execution_log VALUES (?, ?, ?, ?, NULL, 1, ?, ?)",
        (row_id, task_id, agent_id, status, executed_at, outcome),
    )


async def test_execution_log_purge_keeps_every_row_something_still_reads(db):
    await _create_scheduler_tables(db)
    await db.execute(
        "INSERT INTO scheduled_tasks VALUES ('cron', ?, 'claimed-old', 'streak-old')",
        (AGENT,),
    )
    # The expired no-op ticks of a cron: the only rows that go.
    await _execution(db, "tick-1", "cron", _at(40))
    await _execution(db, "tick-2", "cron", _at(39), status="failed")
    await _execution(db, "tick-3", "cron", _at(38), status="skipped")
    # Old, but still read.
    await _execution(db, "claimed-old", "cron", _at(37), status="claimed")
    await _execution(db, "streak-old", "cron", _at(36), status="failed")
    await _execution(db, "executing-old", "cron", _at(35), status="executing")
    await _execution(db, "rated-old", "cron", _at(34), outcome=0.8)
    await _execution(db, "cron-newest", "cron", _at(0))
    # A one-shot follow-up long done: its only run is its outcome.
    await _execution(db, "followup-only", "followup", _at(90))
    # Another agent's expired ticks are that agent's to sweep.
    await _execution(db, "n-1", "theirs", _at(40), agent_id=OTHER)
    await _execution(db, "n-2", "theirs", _at(1), agent_id=OTHER)

    assert await purge_execution_log(
        db, agent_id=AGENT, older_than=CUTOFF, max_rows=100,
    ) == 3
    assert await _ids(db, "task_execution_log") == [
        "claimed-old", "cron-newest", "executing-old", "followup-only",
        "n-1", "n-2", "rated-old", "streak-old",
    ]


async def test_execution_log_purge_keeps_runs_inside_the_window(db):
    await _create_scheduler_tables(db)
    await _execution(db, "recent-1", "cron", _at(3))
    await _execution(db, "recent-2", "cron", _at(2))
    assert await purge_execution_log(
        db, agent_id=AGENT, older_than=CUTOFF, max_rows=100,
    ) == 0


@pytest.mark.parametrize("agent_id", ["", None])
async def test_execution_log_purge_refuses_an_empty_scope(db, agent_id):
    await _create_scheduler_tables(db)
    with pytest.raises(ValueError):
        await purge_execution_log(
            db, agent_id=agent_id, older_than=CUTOFF, max_rows=10,
        )


# --- footprint ---------------------------------------------------------------


@pytest.mark.parametrize("count,text", [
    (0, "0 B"),
    (999, "999 B"),
    (1_500, "1.50 kB"),
    (33_000_000, "33.0 MB"),
    (1_890_131_968, "1.89 GB"),
])
def test_format_bytes(count, text):
    assert footprint.format_bytes(count) == text


async def test_database_size_reports_the_whole_database(db):
    await _create_health_log(db)
    for index in range(200):
        await _health_row(db, f"h{index}", AGENT, "x" * 400 + str(index))
    size = await footprint.measure_database_size(db)
    assert size is not None
    assert size.total_bytes > 0
    assert size.backend == db.backend_type
    if db.backend_type == "postgres":
        assert size.shared is True
        assert size.free_bytes is None
        return
    page_count = await db.fetchval("SELECT page_count FROM pragma_page_count()")
    page_size = await db.fetchval("SELECT page_size FROM pragma_page_size()")
    assert size.total_bytes == page_count * page_size
    assert size.free_bytes == 0
    assert isinstance(size.wal_bytes, int)

    await db.execute("DELETE FROM health_log")
    after = await footprint.measure_database_size(db)
    assert after.free_bytes > 0, "deleted pages are free, not returned"
    assert after.total_bytes == size.total_bytes
    assert "free pages" in after.describe()


async def test_database_size_refuses_a_replaced_database_file(tmp_path):
    from kestrel_sovereign.storage.db.interface import (
        ConnectionError as DatabaseConnectionError,
    )

    path = tmp_path / "replaced.db"
    database = await AsyncDatabase.sqlite(str(path), schema_initializer=_no_schema)
    try:
        path.rename(tmp_path / "moved.db")
        sqlite3.connect(path).close()
        with pytest.raises(DatabaseConnectionError):
            await footprint.measure_database_size(database)
    finally:
        await database.close()


async def test_database_size_never_pairs_one_file_with_anothers_log(tmp_path):
    """The page counts come through the agent's connection, the log by its
    pathname. A file replaced between the two must not be reported as one
    database: that pairing is what the read-by-pathname table scan got wrong.
    """
    from kestrel_sovereign.storage.db.interface import (
        ConnectionError as DatabaseConnectionError,
    )

    path = tmp_path / "raced.db"
    database = await AsyncDatabase.sqlite(str(path), schema_initializer=_no_schema)
    query_size = database.fetchone

    async def size_then_replace(sql, *args, **kwargs):
        row = await query_size(sql, *args, **kwargs)
        path.rename(tmp_path / "moved.db")
        with closing(sqlite3.connect(path)) as replacement:
            replacement.execute("CREATE TABLE unrelated_replacement_table (v)")
        return row

    try:
        database.fetchone = size_then_replace
        with pytest.raises(DatabaseConnectionError):
            await footprint.measure_database_size(database)
    finally:
        del database.fetchone
        await database.close()


async def test_footprint_does_not_guess_for_an_unknown_adapter():
    unknown = SimpleNamespace(backend_type="duckdb")
    assert await footprint.measure_database_size(unknown) is None


async def test_footprint_of_an_in_memory_database():
    database = await AsyncDatabase.sqlite(":memory:", schema_initializer=_no_schema)
    try:
        size = await footprint.measure_database_size(database)
    finally:
        await database.close()
    assert size.wal_bytes is None


# --- the trash_retention rail ------------------------------------------------


async def _seed_rail(agent, db, store):
    """Expired and current rows in every swept table, for the agent and a
    neighbour sharing the database."""
    await _create_health_log(db)
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    new = datetime.now(timezone.utc).isoformat()
    for index in range(3):
        await _health_row(db, f"h-old-{index}", agent.did, old)
    await _health_row(db, "h-new", agent.did, new)
    await _health_row(db, "h-neighbour", OTHER, old)

    await db.execute(
        """
        INSERT INTO scheduled_tasks (id, agent_id, task_name, cron_expression, created_at)
        VALUES ('cron-1', ?, 'wait_reconcile', '* * * * *', ?)
        """,
        (agent.did, new),
    )
    for index, executed_at in enumerate((old, old, new)):
        await db.execute(
            """
            INSERT INTO task_execution_log
                (id, task_id, agent_id, status, result_text, duration_ms, executed_at)
            VALUES (?, 'cron-1', ?, 'success', '{}', 3, ?)
            """,
            (f"run-{index}", agent.did, executed_at),
        )

    from kestrel_sdk.signals import (
        RedactionPolicy, Signal, SignalMode, SignalResult, SourceRegistration, Status,
    )

    async def _handler(_payload):
        return None

    registration = SourceRegistration(
        name="test.retention",
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=_handler,
        log_redaction=RedactionPolicy(summarize=lambda p: "", store_raw_trusted=False),
        retention_days=1,
    )
    for target in (agent.did, agent.did, OTHER):
        signal = Signal(
            source="test.retention", kind="tick", mode=SignalMode.ACTION,
            payload={}, target_agent=target,
        )
        await store.append(signal, registration, SignalResult(
            signal_id=signal.id, status=Status.OK, mode=signal.mode, duration_ms=1,
        ))
    await db.execute(
        "UPDATE signal_log SET retention_until = ?",
        (store.to_timestamp_param(datetime.now(timezone.utc) - timedelta(days=1)),),
    )


def _rail_config(section):
    """``kestrel.toml`` with only this ``[telemetry_retention]`` section."""
    return patch(
        "kestrel_sovereign.config.load_section",
        side_effect=lambda name: section if name == "telemetry_retention" else {},
    )


async def test_trash_retention_rail_ages_out_telemetry(followup_env):
    agent, feature, _runner, db, backend = followup_env
    from kestrel_sovereign.signals import SignalLogStore

    store = SignalLogStore(backend)
    await _seed_rail(agent, db, store)

    with _rail_config({}):
        result = json.loads(await feature._run_trash_retention({}))

    report = result["telemetry_retention"]
    assert report["health_log"] == {"retention_days": 7, "rows_purged": 3}
    assert report["task_execution_log"] == {"retention_days": 30, "rows_purged": 2}
    assert report["restart_status_events"] == {"absent": True}
    assert report["signal_log"] == {"rows_purged": 2}
    assert await _ids(db, "health_log") == ["h-neighbour", "h-new"]
    assert await _ids(db, "task_execution_log") == ["run-2"]
    remaining = await db.fetchall("SELECT target_agent FROM signal_log")
    assert [row[0] for row in remaining] == [OTHER]


async def test_trash_retention_rail_reports_kept_and_capped_tables(followup_env):
    agent, feature, _runner, db, backend = followup_env
    from kestrel_sovereign.signals import SignalLogStore

    await _seed_rail(agent, db, SignalLogStore(backend))

    with _rail_config({"task_execution_log_days": 0, "max_rows_per_sweep": 2}):
        result = json.loads(await feature._run_trash_retention({}))

    report = result["telemetry_retention"]
    assert report["task_execution_log"] == {"kept": True}
    assert report["health_log"] == {
        "retention_days": 7, "rows_purged": 2, "capped": True,
    }
    assert report["signal_log"] == {"rows_purged": 2, "capped": True}
    assert len(await _ids(db, "task_execution_log")) == 3


async def test_one_failing_table_does_not_stop_the_others(followup_env, monkeypatch):
    agent, feature, _runner, db, backend = followup_env
    from kestrel_sovereign.features.health import feature as health_feature
    from kestrel_sovereign.signals import SignalLogStore

    await _seed_rail(agent, db, SignalLogStore(backend))
    monkeypatch.setattr(
        health_feature, "purge_health_log",
        AsyncMock(side_effect=RuntimeError("health_log is locked")),
    )

    with _rail_config({}):
        result = json.loads(await feature._run_trash_retention({}))

    report = result["telemetry_retention"]
    assert report["health_log"] == {
        "retention_days": 7, "error": "health_log is locked",
    }
    assert report["task_execution_log"]["rows_purged"] == 2
    assert report["signal_log"] == {"rows_purged": 2}


async def test_execution_control_is_not_reported_as_a_table_error(
    followup_env, monkeypatch,
):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.features.health import feature as health_feature

    _agent, feature, _runner, db, _backend = followup_env
    await _create_health_log(db)
    monkeypatch.setattr(
        health_feature, "purge_health_log",
        AsyncMock(side_effect=ExecutionAuthorityError("lease lost")),
    )
    with _rail_config({}), pytest.raises(ExecutionAuthorityError):
        await feature._run_trash_retention({})


async def test_rail_refuses_to_sweep_without_an_agent_identity(followup_env, caplog):
    agent, feature, _runner, db, _backend = followup_env
    await _create_health_log(db)
    await _health_row(db, "h-old", OTHER, _at(400))
    feature.agent = SimpleNamespace(did=None, agent_id=agent.did)

    with _rail_config({}):
        assert await feature._purge_operational_telemetry() is None
    assert await _ids(db, "health_log") == ["h-old"]
    assert any("identity unavailable" in r.getMessage() for r in caplog.records)


# --- the health probe --------------------------------------------------------


async def test_health_probe_reports_the_database_size(tmp_path):
    from kestrel_sovereign.features.health.checks import check_database_size

    database = await AsyncDatabase.sqlite(
        str(tmp_path / "probe.db"), schema_initializer=_no_schema
    )
    try:
        await _create_health_log(database)
        result = await check_database_size(database)
    finally:
        await database.close()
    assert result["name"] == "database_size"
    assert result["status"] == "pass"
    assert result["details"]["backend"] == "sqlite"
    assert result["details"]["total_bytes"] > 0
    assert result["message"] == f"Database {result['details']['summary']}"


async def test_health_probe_size_is_report_only_for_a_missing_or_unknown_db():
    from kestrel_sovereign.features.health.checks import check_database_size

    missing = await check_database_size(None)
    unknown = await check_database_size(SimpleNamespace(backend_type="duckdb"))
    assert (missing["status"], unknown["status"]) == ("pass", "pass")
    assert "not measured" in unknown["message"]


async def test_health_probe_warns_when_the_size_query_fails():
    from kestrel_sovereign.features.health.checks import check_database_size

    broken = SimpleNamespace(
        backend_type="sqlite",
        fetchone=AsyncMock(side_effect=RuntimeError("disk I/O error")),
    )
    result = await check_database_size(broken)
    assert result["status"] == "warn"
    assert "disk I/O error" in result["message"]


async def test_health_probe_propagates_execution_control():
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.features.health.checks import check_database_size

    revoked = SimpleNamespace(
        backend_type="sqlite",
        fetchone=AsyncMock(side_effect=ExecutionAuthorityError("lease lost")),
    )
    with pytest.raises(ExecutionAuthorityError):
        await check_database_size(revoked)


# --- signal_log --------------------------------------------------------------


async def test_signal_log_purge_on_both_backends(db):
    """``retention_until`` is TIMESTAMPTZ on PostgreSQL; the batched subquery
    must bind it the way the old single statement did."""
    from kestrel_sdk.signals import (
        RedactionPolicy, Signal, SignalMode, SignalResult, SourceRegistration, Status,
    )
    from kestrel_sovereign.signals import SignalLogStore

    async def _handler(_payload):
        return None

    store = SignalLogStore(db.backend)
    await store.initialize()
    for target, days in ((AGENT, 1), (AGENT, 1), (AGENT, 30), (OTHER, 1)):
        registration = SourceRegistration(
            name="test.retention",
            schema=dict,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=_handler,
            log_redaction=RedactionPolicy(summarize=lambda p: "", store_raw_trusted=False),
            retention_days=days,
        )
        signal = Signal(
            source="test.retention", kind="tick", mode=SignalMode.ACTION,
            payload={}, target_agent=target,
        )
        await store.append(signal, registration, SignalResult(
            signal_id=signal.id, status=Status.OK, mode=signal.mode, duration_ms=1,
        ))

    later = datetime.now(timezone.utc) + timedelta(days=7)
    assert await store.purge_expired(target_agent=AGENT, now=later, max_rows=1) == 1
    assert await store.purge_expired(target_agent=AGENT, now=later) == 1
    remaining = await db.fetchall(
        "SELECT target_agent FROM signal_log ORDER BY target_agent"
    )
    assert sorted(row[0] for row in remaining) == sorted([AGENT, OTHER])
