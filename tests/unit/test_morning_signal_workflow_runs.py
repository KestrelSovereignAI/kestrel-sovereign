"""The Morning Signal's "Workflow runs (24h)" section (#3519).

``stalled_work_rescue`` failed four times a day from 09-29 to 10-08 and
nothing reported it; it was found by reading ``workflow_runs`` by hand. These
cases pin the section that now reads those tables: the counts, the failing
stage and gate reason of each failure, and the call-out for a definition
whose last runs all failed.

The store tables are built here with the columns kestrel-feature-workflows
creates (core cannot import that package). When it is installed,
``test_reader_matches_the_real_workflows_schema`` checks the reader against
the real ``WorkflowStore`` schema too.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature
from kestrel_sovereign.features.strategic_memory.morning_signal import (
    generate_morning_signal,
)
from kestrel_sovereign.features.strategic_memory.workflow_runs import (
    SECTION_TITLE,
    WorkflowRunsNotAssessed,
    assess_workflow_runs,
    read_workflow_run_report,
    render_workflow_runs_section,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.storage.db.timestamp import TimestamptzParameter
from tests.utils.postgres_schema import disposable_postgres_schema, with_search_path

OWNER = "did:pkh:eip155:1:0x00000000000000000000000000000000000000a1"
OTHER_AGENT = "did:web:agents.example.test:someone-else"
NOW = datetime(2026, 10, 8, 12, 30, tzinfo=timezone.utc)

RESCUE = "stalled_work_rescue"
PIPELINE = "fleet_coding_pipeline"
NO_TARGETS = (
    "ValueError: a2a_repair_dispatch: no explicit repair targets supplied "
    "(repairs/repair_targets); refusing to dispatch. (fail closed)"
)
CI_RED = "ValueError: fleet_ci_probe: talon PR CI not verified green (fail closed)"


def _schema(timestamp_type: str) -> list[str]:
    # The columns kestrel-feature-workflows' WorkflowStore creates that the
    # reader touches, plus the NOT NULL ones a real row always carries.
    return [
        f"""
        CREATE TABLE workflow_runs (
            run_id         TEXT PRIMARY KEY,
            workflow_name  TEXT NOT NULL,
            workflow_ver   INTEGER NOT NULL,
            params_json    TEXT NOT NULL,
            status         TEXT NOT NULL,
            engine_nonce   TEXT NOT NULL,
            started_by_did TEXT NOT NULL,
            started_at     {timestamp_type} NOT NULL,
            finished_at    {timestamp_type},
            deleted_at     {timestamp_type}
        )
        """,
        f"""
        CREATE TABLE workflow_stage_links (
            link_id         TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL REFERENCES workflow_runs(run_id),
            stage_name      TEXT NOT NULL,
            attempt_number  INTEGER NOT NULL,
            idempotency_key TEXT NOT NULL,
            gate_outcome    TEXT,
            gate_reason     TEXT,
            execution_state TEXT NOT NULL DEFAULT 'prepared',
            actor_did       TEXT NOT NULL,
            actor_sig       TEXT NOT NULL,
            occurred_at     {timestamp_type} NOT NULL,
            UNIQUE (run_id, stage_name, attempt_number)
        )
        """,
    ]


class _Store:
    """Writes Workflows rows the way the store does on either backend."""

    def __init__(self, db: AsyncDatabase) -> None:
        self.db = db

    def _ts(self, moment: datetime | None):
        if moment is None:
            return None
        if self.db.backend_type == "postgres":
            return TimestamptzParameter(moment)
        return moment.isoformat()

    async def create(self) -> None:
        timestamp_type = "TIMESTAMPTZ" if self.db.backend_type == "postgres" else "TEXT"
        for statement in _schema(timestamp_type):
            await self.db.execute(statement)

    async def run(
        self,
        name: str,
        status: str,
        ended_at: datetime | None,
        *,
        owner: str = OWNER,
        deleted: bool = False,
        failed_at: tuple[tuple[str, str | None], ...] = (),
        passed: tuple[str, ...] = (),
    ) -> str:
        """Record one run, its passing stages, and its failing gate(s).

        ``failed_at`` holds ``(stage, reason)`` pairs, in the order the gates
        failed.
        """
        run_id = str(uuid.uuid4())
        started_at = (ended_at or NOW) - timedelta(seconds=30)
        await self.db.execute(
            "INSERT INTO workflow_runs (run_id, workflow_name, workflow_ver, "
            "params_json, status, engine_nonce, started_by_did, started_at, "
            "finished_at, deleted_at) VALUES (?, ?, 1, '{}', ?, 'n', ?, ?, ?, ?)",
            (
                run_id,
                name,
                status,
                owner,
                self._ts(started_at),
                self._ts(ended_at),
                self._ts(ended_at if deleted else None),
            ),
        )
        moment = started_at
        for stage in passed:
            moment += timedelta(seconds=1)
            await self._link(run_id, stage, 1, "pass", None, moment)
        for attempt, (stage, reason) in enumerate(failed_at, start=1):
            moment += timedelta(seconds=1)
            await self._link(run_id, stage, attempt, "fail", reason, moment)
        return run_id

    async def _link(self, run_id, stage, attempt, outcome, reason, occurred_at):
        await self.db.execute(
            "INSERT INTO workflow_stage_links (link_id, run_id, stage_name, "
            "attempt_number, idempotency_key, gate_outcome, gate_reason, "
            "actor_did, actor_sig, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'sig', ?)",
            (
                str(uuid.uuid4()),
                run_id,
                stage,
                attempt,
                str(uuid.uuid4()),
                outcome,
                reason,
                OWNER,
                self._ts(occurred_at),
            ),
        )


async def _no_schema_ddl(db):
    """Schema initializer for a connection that must not boot core's schema."""


async def _sqlite_db(tmp_path: Path) -> AsyncDatabase:
    raw = SQLiteBackend(str(tmp_path / "workflow-runs.db"))
    await raw.connect()
    return AsyncDatabase(raw)


# Parametrized with "postgres" so the #3381 guard fails the PostgreSQL case,
# rather than letting it skip, in a job that provides PostgreSQL.
@pytest.fixture(params=["sqlite", "postgres"])
async def database(request, tmp_path):
    if request.param == "sqlite":
        db = await _sqlite_db(tmp_path)
        try:
            yield db
        finally:
            await db.close()
        return
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    # The tables are created in a schema of the case's own, so the reader
    # never sees another run's workflow rows.
    admin = await AsyncDatabase.postgres(url, schema_initializer=_no_schema_ddl)
    try:
        async with disposable_postgres_schema(admin, "morning_signal_workflow_runs") as schema:
            db = await AsyncDatabase.postgres(
                with_search_path(url, schema), schema_initializer=_no_schema_ddl
            )
            try:
                yield db
            finally:
                await db.close()
    finally:
        await admin.close()


@pytest.fixture
async def store(database):
    workflows = _Store(database)
    await workflows.create()
    return workflows


def _agent(db, did: str = OWNER):
    return SimpleNamespace(did=did, _raw_storage=SimpleNamespace(db=db))


def _by_name(report):
    return {outcomes.workflow_name: outcomes for outcomes in report.workflows}


def _hours_ago(hours: float) -> datetime:
    return NOW - timedelta(hours=hours)


# ---------------------------------------------------------------------------
# No runs
# ---------------------------------------------------------------------------


async def test_no_runs_says_so(store):
    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    assert report.workflows == ()
    assert report.store_present is True
    section = render_workflow_runs_section(report)
    assert section == [
        f"## {SECTION_TITLE}",
        "No workflow runs finished in the last 24 hours.",
    ]


async def test_no_runs_in_the_window_says_so(store):
    # Runs that ended before the window, are still going, were cancelled,
    # were deleted, or belong to another agent are not this agent's outcomes
    # in the last 24 hours.
    await store.run(RESCUE, "failed", _hours_ago(25), failed_at=(("dispatch_repairs", NO_TARGETS),))
    await store.run(RESCUE, "running", None)
    await store.run(RESCUE, "cancelled", _hours_ago(1))
    await store.run(RESCUE, "failed", _hours_ago(1), deleted=True)
    await store.run(RESCUE, "failed", _hours_ago(1), owner=OTHER_AGENT)

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    assert report.workflows == ()
    assert "No workflow runs finished in the last 24 hours." in render_workflow_runs_section(report)


async def test_an_agent_without_a_workflows_store_says_so(database):
    report = await read_workflow_run_report(_agent(database), now=NOW)

    assert report.store_present is False
    assert report.workflows == ()
    assert render_workflow_runs_section(report)[1] == (
        "No workflow runs finished in the last 24 hours: this agent has no "
        "Workflows run store."
    )


# ---------------------------------------------------------------------------
# Mixed results
# ---------------------------------------------------------------------------


async def test_mixed_results_are_counted_per_workflow(store):
    for hours in (2, 8):
        await store.run(PIPELINE, "completed", _hours_ago(hours), passed=("talon_run", "verify_ci"))
    ci_failure = await store.run(
        PIPELINE, "failed", _hours_ago(1), passed=("talon_run",), failed_at=(("verify_ci", CI_RED),)
    )
    rescue_failures = [
        await store.run(
            RESCUE,
            "failed",
            _hours_ago(hours),
            passed=("detect_stalled", "govern_intent"),
            failed_at=(("dispatch_repairs", NO_TARGETS),),
        )
        for hours in (0.5, 6.5, 12.5)
    ]
    await store.run(RESCUE, "completed", _hours_ago(18.5))
    # Not counted: outside the window, another agent's, still running.
    await store.run(RESCUE, "completed", _hours_ago(30))
    await store.run(PIPELINE, "failed", _hours_ago(3), owner=OTHER_AGENT)
    await store.run(PIPELINE, "waiting", None)

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    outcomes = _by_name(report)
    assert set(outcomes) == {PIPELINE, RESCUE}
    assert (outcomes[PIPELINE].completed, outcomes[PIPELINE].failed) == (2, 1)
    assert (outcomes[RESCUE].completed, outcomes[RESCUE].failed) == (1, 3)
    assert [run.run_id for run in outcomes[RESCUE].failed_runs] == rescue_failures
    assert outcomes[PIPELINE].failed_runs[0].run_id == ci_failure
    assert outcomes[PIPELINE].failed_runs[0].stage_name == "verify_ci"
    assert outcomes[PIPELINE].failed_runs[0].gate_reason == CI_RED
    # Three failures after a success is a run of three: persistent.
    assert report.persistently_failing() == (outcomes[RESCUE],)

    section = "\n".join(render_workflow_runs_section(report))
    assert f"- `{PIPELINE}`: 2 completed, 1 failed" in section
    assert f"- `{RESCUE}`: 1 completed, 3 failed" in section
    assert (
        f"  - 1 failed at stage `verify_ci`: {CI_RED} "
        f"(latest run `{ci_failure}` at 2026-10-08 11:30 UTC)"
    ) in section
    assert (
        f"  - 3 failed at stage `dispatch_repairs`: {NO_TARGETS} "
        f"(latest run `{rescue_failures[0]}` at 2026-10-08 12:00 UTC)"
    ) in section


async def test_each_failure_names_its_own_stage_and_reason(store):
    await store.run(PIPELINE, "failed", _hours_ago(1), failed_at=(("verify_ci", CI_RED),))
    workspace = "ValueError: talon_pipeline_dispatch: No talon workspace exists (fail closed)"
    await store.run(PIPELINE, "failed", _hours_ago(2), failed_at=(("talon_run", workspace),))
    await store.run(PIPELINE, "completed", _hours_ago(3))

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    section = "\n".join(render_workflow_runs_section(report))
    assert f"1 failed at stage `verify_ci`: {CI_RED}" in section
    assert f"1 failed at stage `talon_run`: {workspace}" in section
    assert "PERSISTENTLY FAILING" not in section


async def test_a_retried_stage_reports_the_gate_the_run_ended_on(store):
    await store.run(
        PIPELINE,
        "failed",
        _hours_ago(1),
        failed_at=(("verify_ci", "first attempt: checks pending"), ("verify_ci", CI_RED)),
    )

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    (failure,) = _by_name(report)[PIPELINE].failed_runs
    assert (failure.stage_name, failure.gate_reason) == ("verify_ci", CI_RED)


async def test_a_failure_with_no_failed_gate_is_reported_as_such(store):
    await store.run(RESCUE, "failed", _hours_ago(1), passed=("detect_stalled",))

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    section = "\n".join(render_workflow_runs_section(report))
    assert f"- `{RESCUE}`: 0 completed, 1 failed" in section
    assert "1 failed with no failing stage recorded: no gate reason recorded" in section


async def test_a_long_gate_reason_is_shortened_visibly(store):
    snapshot = "ValueError: CI snapshot:\n" + "\n".join(f"- check {i}: completed/cancelled" for i in range(40))
    await store.run(PIPELINE, "failed", _hours_ago(1), failed_at=(("verify_ci", snapshot),))

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    (line,) = [line for line in render_workflow_runs_section(report) if "verify_ci" in line]
    assert "ValueError: CI snapshot: - check 0: completed/cancelled - check 1" in line
    assert "…" in line
    assert "\n" not in line


async def test_distinct_failures_beyond_the_limit_are_counted_not_dropped(store):
    for i in range(7):
        await store.run(PIPELINE, "failed", _hours_ago(i + 1), failed_at=((f"stage_{i}", f"reason {i}"),))

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    section = render_workflow_runs_section(report)
    assert sum(1 for line in section if line.startswith("  - 1 failed at stage")) == 5
    assert "  - 2 more distinct failure(s) across 2 run(s) not listed" in section


# ---------------------------------------------------------------------------
# Persistent failure
# ---------------------------------------------------------------------------


async def test_three_consecutive_failures_are_called_out(store):
    last_success = datetime(2026, 9, 26, 18, 0, 20, tzinfo=timezone.utc)
    await store.run(RESCUE, "completed", last_success - timedelta(hours=6))
    await store.run(RESCUE, "completed", last_success)
    for hours in (60, 30, 12, 6, 0.5):
        latest = await store.run(
            RESCUE,
            "failed",
            _hours_ago(hours),
            passed=("detect_stalled", "govern_intent"),
            failed_at=(("dispatch_repairs", NO_TARGETS),),
        )
    # A cancelled run neither breaks the run of failures nor adds to it.
    await store.run(RESCUE, "cancelled", _hours_ago(3))

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    (rescue,) = report.persistently_failing()
    assert rescue.workflow_name == RESCUE
    assert rescue.consecutive_failures == 5
    assert rescue.last_success_at == last_success
    section = render_workflow_runs_section(report)
    assert section[1] == (
        f"- **PERSISTENTLY FAILING** `{RESCUE}`: its last 5 runs failed. "
        f"Latest run `{latest}` failed at stage `dispatch_repairs`: "
        f"{NO_TARGETS}. Last success: 2026-09-26."
    )


async def test_two_consecutive_failures_are_not_persistent(store):
    await store.run(RESCUE, "completed", _hours_ago(20))
    for hours in (10, 2):
        await store.run(RESCUE, "failed", _hours_ago(hours), failed_at=(("dispatch_repairs", NO_TARGETS),))

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    assert _by_name(report)[RESCUE].consecutive_failures == 2
    assert report.persistently_failing() == ()
    assert not any("PERSISTENTLY FAILING" in line for line in render_workflow_runs_section(report))


async def test_a_success_after_failures_ends_the_run(store):
    for hours in (20, 15, 10):
        await store.run(RESCUE, "failed", _hours_ago(hours), failed_at=(("dispatch_repairs", NO_TARGETS),))
    await store.run(RESCUE, "completed", _hours_ago(1))

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    assert _by_name(report)[RESCUE].consecutive_failures == 0
    assert report.persistently_failing() == ()


async def test_a_definition_that_never_succeeded_says_so(store):
    for hours in (20, 10, 1):
        await store.run(RESCUE, "failed", _hours_ago(hours), failed_at=(("dispatch_repairs", NO_TARGETS),))

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    (rescue,) = report.persistently_failing()
    assert rescue.last_success_at is None
    assert render_workflow_runs_section(report)[1].endswith("Last success: none on record.")


async def test_another_agents_success_does_not_end_the_run(store):
    for hours in (20, 10, 1):
        await store.run(RESCUE, "failed", _hours_ago(hours), failed_at=(("dispatch_repairs", NO_TARGETS),))
    await store.run(RESCUE, "completed", _hours_ago(0.5), owner=OTHER_AGENT)

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    assert [o.workflow_name for o in report.persistently_failing()] == [RESCUE]


async def test_runs_that_ended_after_the_window_are_not_counted(store):
    for hours in (20, 10, 1):
        await store.run(RESCUE, "failed", _hours_ago(hours), failed_at=(("dispatch_repairs", NO_TARGETS),))
    # Later than the moment the report is for: neither counted nor allowed to
    # end the run of failures it reports.
    await store.run(RESCUE, "completed", NOW + timedelta(hours=1))
    await store.run(RESCUE, "failed", NOW + timedelta(hours=2), failed_at=(("dispatch_repairs", NO_TARGETS),))

    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    rescue = _by_name(report)[RESCUE]
    assert (rescue.completed, rescue.failed) == (0, 3)
    assert rescue.consecutive_failures == 3
    assert rescue.last_success_at is None
    assert report.persistently_failing() == (rescue,)


async def test_the_threshold_is_configurable(store):
    for hours in (20, 10, 1):
        await store.run(RESCUE, "failed", _hours_ago(hours), failed_at=(("dispatch_repairs", NO_TARGETS),))
    agent = _agent(store.db)

    stricter = await assess_workflow_runs(
        agent, {"morning_signal_config": {"persistent_failure_runs": 4}}, now=NOW
    )
    looser = await assess_workflow_runs(
        agent, {"morning_signal_config": {"persistent_failure_runs": 1}}, now=NOW
    )
    default = await assess_workflow_runs(agent, {}, now=NOW)

    assert stricter.persistently_failing() == ()
    assert [o.workflow_name for o in looser.persistently_failing()] == [RESCUE]
    assert default.persistent_failure_runs == 3
    assert [o.workflow_name for o in default.persistently_failing()] == [RESCUE]


@pytest.mark.parametrize("value", [0, -2, "3", 2.5, True])
async def test_an_invalid_threshold_is_refused_not_guessed(store, value):
    assessment = await assess_workflow_runs(
        _agent(store.db), {"morning_signal_config": {"persistent_failure_runs": value}}, now=NOW
    )

    assert isinstance(assessment, WorkflowRunsNotAssessed)
    assert render_workflow_runs_section(assessment)[1] == (
        "Workflow runs could not be assessed: morning_signal_config."
        f"persistent_failure_runs must be a positive integer, not {value!r}"
    )


# ---------------------------------------------------------------------------
# Unreadable is not empty
# ---------------------------------------------------------------------------


async def test_an_agent_without_a_did_is_not_assessed(store):
    with pytest.raises(WorkflowRunsNotAssessed, match="no DID"):
        await read_workflow_run_report(_agent(store.db, did=""), now=NOW)


async def test_an_agent_without_a_database_is_not_assessed():
    agent = SimpleNamespace(did=OWNER)

    assessment = await assess_workflow_runs(agent, {}, now=NOW)

    assert isinstance(assessment, WorkflowRunsNotAssessed)
    assert "no database" in str(assessment)


async def test_a_half_present_store_is_not_assessed(database):
    await database.execute(_schema("TEXT")[0])

    with pytest.raises(WorkflowRunsNotAssessed, match="workflow_stage_links is missing"):
        await read_workflow_run_report(_agent(database), now=NOW)


async def test_a_store_read_failure_is_reported(database):
    # A store whose shape the reader does not know: the query fails.
    await database.execute("CREATE TABLE workflow_runs (run_id TEXT PRIMARY KEY)")
    await database.execute("CREATE TABLE workflow_stage_links (link_id TEXT PRIMARY KEY)")

    assessment = await assess_workflow_runs(_agent(database), {}, now=NOW)

    assert isinstance(assessment, WorkflowRunsNotAssessed)
    assert str(assessment).startswith("reading the Workflows store failed: ")
    assert "No workflow runs" not in "\n".join(render_workflow_runs_section(assessment))


# ---------------------------------------------------------------------------
# The briefing
# ---------------------------------------------------------------------------


async def test_the_briefing_carries_the_section_and_suggests_the_repair(store):
    await store.run(RESCUE, "completed", datetime(2026, 9, 26, 18, 0, tzinfo=timezone.utc))
    for hours in (20, 10, 1):
        await store.run(RESCUE, "failed", _hours_ago(hours), failed_at=(("dispatch_repairs", NO_TARGETS),))
    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    briefing = await generate_morning_signal(
        {"morning_signal_config": {"scan_repos": []}}, report
    )

    assert f"## {SECTION_TITLE}" in briefing
    assert f"**PERSISTENTLY FAILING** `{RESCUE}`" in briefing
    assert (
        f"Repair workflow `{RESCUE}`: failing at stage `dispatch_repairs`, "
        "last success 2026-09-26"
    ) in briefing
    assert "No urgent items detected" not in briefing


async def test_the_briefing_reports_runs_without_a_strategy_file(store):
    for hours in (20, 10, 1):
        await store.run(RESCUE, "failed", _hours_ago(hours), failed_at=(("dispatch_repairs", NO_TARGETS),))
    report = await read_workflow_run_report(_agent(store.db), now=NOW)

    briefing = await generate_morning_signal({}, report)

    assert briefing.startswith("No strategic memory loaded.")
    assert f"**PERSISTENTLY FAILING** `{RESCUE}`" in briefing


async def test_the_morning_signal_tool_reads_the_agents_runs(store):
    now = datetime.now(timezone.utc)
    for hours in (3, 2, 1):
        await store.run(RESCUE, "failed", now - timedelta(hours=hours), failed_at=(("dispatch_repairs", NO_TARGETS),))
    feature = StrategicMemoryFeature(agent=_agent(store.db))
    feature._data = {"morning_signal_config": {"scan_repos": []}}

    result = await feature.morning_signal()

    assert result.status is ToolResultStatus.OK
    assert f"- `{RESCUE}`: 0 completed, 3 failed" in result.confirmation
    assert f"**PERSISTENTLY FAILING** `{RESCUE}`" in result.confirmation


async def test_the_morning_signal_tool_says_when_runs_cannot_be_read():
    feature = StrategicMemoryFeature(agent=MagicMock())
    feature._data = {"morning_signal_config": {"scan_repos": []}}

    result = await feature.morning_signal()

    assert result.status is ToolResultStatus.OK
    assert f"## {SECTION_TITLE}\nWorkflow runs could not be assessed: " in result.confirmation


# ---------------------------------------------------------------------------
# The real Workflows schema
# ---------------------------------------------------------------------------


async def test_reader_matches_the_real_workflows_schema(tmp_path):
    """Every column the reader names exists in the store Workflows creates."""
    store_module = pytest.importorskip("kestrel_feature_workflows.store")
    db = await _sqlite_db(tmp_path)
    try:
        await store_module.WorkflowStore(db.backend).initialize()

        report = await read_workflow_run_report(_agent(db), now=NOW)
    finally:
        await db.close()

    assert report.store_present is True
    assert report.workflows == ()
