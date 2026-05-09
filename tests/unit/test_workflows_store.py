"""Phase 0 chunk B — workflow storage migration tests.

Pins the three-table DDL from §5 of the design doc against both
SQLite (the default backend per ``feedback_sqlite_non_negotiable.md``)
and that the dialect-helper-driven schema is round-trip safe for the
canonical signed JSON form.

Postgres parity is exercised by the existing dialect-helper test
infrastructure that ``signal_log`` already validates; this file does
not duplicate that coverage.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sdk.signals import SignalMode

from kestrel_sovereign.features.workflows import (
    Edge,
    EdgeKind,
    Stage,
    WorkflowSpec,
)
from kestrel_sovereign.features.workflows.store import WorkflowStore
from kestrel_sovereign.storage.db import SQLiteBackend


@pytest.fixture
async def store(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "workflows.db"))
    await backend.connect()
    s = WorkflowStore(backend)
    await s.initialize()
    yield s
    await backend.close()


def _action_stage(**overrides):
    base = dict(
        name="lint",
        signal_source="ci.lint",
        signal_mode=SignalMode.ACTION,
        read_only=True,
    )
    base.update(overrides)
    return Stage(**base)


def _signed_spec(**overrides):
    """Construct a minimal spec with a populated ``spec_hash`` /
    ``author_sig`` (Phase 0's signing helpers are chunk C; until then,
    use a deterministic placeholder so the insert path can be tested)."""
    base = dict(
        name="release",
        version=1,
        stages=[_action_stage(name="lint")],
        author_did="did:web:k.example",
    )
    base.update(overrides)
    spec = WorkflowSpec(**base)
    # Phase 1's signing helper computes spec_hash + author_sig; for
    # store-tests we attach a placeholder hash and signature so the
    # insert succeeds without depending on chunk C.
    spec_hash = spec.compute_spec_hash()
    return WorkflowSpec(
        **{
            **base,
            "spec_hash": spec_hash,
            "author_sig": "deadbeef" * 8,  # placeholder
        }
    )


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


async def test_initialize_creates_three_tables(store: WorkflowStore):
    rows = await store.backend.fetch_all(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'workflow_%'"
    )
    names = sorted(r[0] for r in rows)
    assert names == [
        "workflow_definitions",
        "workflow_runs",
        "workflow_stage_links",
    ]


async def test_initialize_idempotent(store: WorkflowStore):
    """Re-running initialize must not raise (CREATE TABLE IF NOT EXISTS
    + CREATE INDEX IF NOT EXISTS pattern)."""
    await store.initialize()
    await store.initialize()


async def test_definition_unique_per_name_version(store: WorkflowStore):
    spec = _signed_spec()
    await store.insert_definition_for_test(spec)
    with pytest.raises(Exception):
        await store.insert_definition_for_test(spec)


async def test_definition_round_trip(store: WorkflowStore):
    spec = _signed_spec(retention_days=30)
    await store.insert_definition_for_test(spec)
    row = await store.get_definition_row(spec.name, spec.version)
    assert row is not None
    assert row["name"] == "release"
    assert row["version"] == 1
    assert row["spec_hash"] == spec.spec_hash
    assert row["author_did"] == "did:web:k.example"
    assert row["author_sig"] == "deadbeef" * 8
    assert row["retention_days"] == 30
    assert isinstance(row["created_at"], datetime)
    assert row["deleted_at"] is None
    parsed = json.loads(row["spec_json"])
    # Canonical payload survives storage exactly.
    assert WorkflowSpec.from_dict(parsed).spec_hash == spec.spec_hash


async def test_unsigned_spec_rejected(store: WorkflowStore):
    """Defense-in-depth: the store refuses to persist an unsigned
    draft. Phase 1's tool surface enforces signing earlier; this is
    the second line."""
    spec = WorkflowSpec(
        name="r",
        version=1,
        stages=[_action_stage(name="lint")],
        author_did="did:web:k.example",
        # spec_hash, author_sig deliberately empty
    )
    with pytest.raises(ValueError):
        await store.insert_definition_for_test(spec)


# ---------------------------------------------------------------------------
# Run + stage_link basics (Phase 1 will use the higher-level helpers; for
# Phase 0 we exercise the tables via direct SQL to prove the schema works).
# ---------------------------------------------------------------------------


async def test_runs_table_accepts_minimal_row(store: WorkflowStore):
    spec = _signed_spec()
    await store.insert_definition_for_test(spec)
    await store.backend.execute(
        f"""
        INSERT INTO {store.RUNS_TABLE}
            (run_id, workflow_name, workflow_ver, params_json,
             status, started_by_did)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("run-1", "release", 1, "{}", "running", "did:web:k.example"),
    )
    row = await store.backend.fetch_one(
        f"SELECT run_id, status FROM {store.RUNS_TABLE} WHERE run_id = ?",
        ("run-1",),
    )
    assert row == ("run-1", "running")


async def test_stage_links_unique_constraint(store: WorkflowStore):
    spec = _signed_spec()
    await store.insert_definition_for_test(spec)
    await store.backend.execute(
        f"INSERT INTO {store.RUNS_TABLE} (run_id, workflow_name, workflow_ver, "
        f"params_json, status, started_by_did) VALUES (?, ?, ?, ?, ?, ?)",
        ("run-1", "release", 1, "{}", "running", "did:web:k.example"),
    )
    insert_link = (
        f"INSERT INTO {store.STAGE_LINKS_TABLE} "
        f"(link_id, run_id, stage_name, attempt_number, idempotency_key, "
        f"actor_did, actor_sig) VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    await store.backend.execute(
        insert_link,
        ("l-1", "run-1", "lint", 1, "0" * 64, "did:web:k.example", "sig"),
    )
    # Same (run_id, stage_name, attempt_number) is rejected.
    with pytest.raises(Exception):
        await store.backend.execute(
            insert_link,
            ("l-2", "run-1", "lint", 1, "0" * 64, "did:web:k.example", "sig"),
        )
    # Fresh attempt_number is accepted.
    await store.backend.execute(
        insert_link,
        ("l-3", "run-1", "lint", 2, "1" * 64, "did:web:k.example", "sig"),
    )


# ---------------------------------------------------------------------------
# Retention sweep
# ---------------------------------------------------------------------------


async def test_purge_expired_runs_respects_retention_days(store: WorkflowStore):
    """Codex round-1 P1: the predicate must be ``finished_at + retention_days
    < now``, not ``finished_at < now``. retention_days=1 means a run
    finished 12 hours ago must SURVIVE; only 30-days-old gets purged."""
    spec = _signed_spec(retention_days=1)
    await store.insert_definition_for_test(spec)

    now = datetime.now(timezone.utc)
    long_ago = (now - timedelta(days=30)).isoformat()
    half_a_day = (now - timedelta(hours=12)).isoformat()
    just_finished = (now - timedelta(minutes=5)).isoformat()

    for run_id, finished_at_iso, status in (
        ("run-old", long_ago, "completed"),
        ("run-12h", half_a_day, "completed"),
        ("run-fresh", just_finished, "completed"),
    ):
        await store.backend.execute(
            f"INSERT INTO {store.RUNS_TABLE} (run_id, workflow_name, "
            f"workflow_ver, params_json, status, started_by_did, "
            f"finished_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, "release", 1, "{}",
                status, "did:web:k.example", finished_at_iso,
            ),
        )
    # Unfinished run must NEVER be purged regardless of age.
    await store.backend.execute(
        f"INSERT INTO {store.RUNS_TABLE} (run_id, workflow_name, workflow_ver, "
        f"params_json, status, started_by_did) VALUES (?, ?, ?, ?, ?, ?)",
        ("run-running", "release", 1, "{}", "running", "did:web:k.example"),
    )

    purged = await store.purge_expired_runs(now=now)
    assert purged == 1  # only run-old; run-12h and run-fresh survive

    rows = await store.backend.fetch_all(
        f"SELECT run_id FROM {store.RUNS_TABLE} ORDER BY run_id"
    )
    surviving = sorted(r[0] for r in rows)
    assert surviving == ["run-12h", "run-fresh", "run-running"]


async def test_purge_handles_batch_size_overflow(store: WorkflowStore):
    """Codex round-3 P2: > _PURGE_BATCH_SIZE expired runs must still
    purge in bounded batches; a single unbounded IN(?,?,...) would
    hit SQLite/Postgres bind-parameter caps."""
    from kestrel_sovereign.features.workflows.store import _PURGE_BATCH_SIZE

    spec = _signed_spec(retention_days=1)
    await store.insert_definition_for_test(spec)

    long_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    # Insert just over one batch worth so we exercise both batches
    # without making the test absurdly slow.
    n = _PURGE_BATCH_SIZE + 5
    for i in range(n):
        await store.backend.execute(
            f"INSERT INTO {store.RUNS_TABLE} (run_id, workflow_name, "
            f"workflow_ver, params_json, status, started_by_did, "
            f"finished_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"run-{i:04d}", "release", 1, "{}",
                "completed", "did:web:k.example", long_ago,
            ),
        )

    purged = await store.purge_expired_runs()
    assert purged == n
    remaining = await store.backend.fetch_all(
        f"SELECT run_id FROM {store.RUNS_TABLE}"
    )
    assert remaining == []


async def test_purge_cascades_stage_links(store: WorkflowStore):
    """Codex round-1 P2: ``ON DELETE CASCADE`` on the stage_links FK
    means purging an expired run also drops its stage history rows.
    Without cascade the FK would block the run delete."""
    spec = _signed_spec(retention_days=1)
    await store.insert_definition_for_test(spec)

    now = datetime.now(timezone.utc)
    long_ago = (now - timedelta(days=30)).isoformat()
    await store.backend.execute(
        f"INSERT INTO {store.RUNS_TABLE} (run_id, workflow_name, workflow_ver, "
        f"params_json, status, started_by_did, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("run-old", "release", 1, "{}", "completed", "did:web:k.example", long_ago),
    )
    await store.backend.execute(
        f"INSERT INTO {store.STAGE_LINKS_TABLE} "
        f"(link_id, run_id, stage_name, attempt_number, idempotency_key, "
        f"actor_did, actor_sig) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("l-1", "run-old", "lint", 1, "0" * 64, "did:web:k.example", "sig"),
    )

    purged = await store.purge_expired_runs(now=now)
    assert purged == 1
    remaining_links = await store.backend.fetch_all(
        f"SELECT link_id FROM {store.STAGE_LINKS_TABLE}"
    )
    assert remaining_links == []


async def test_purge_skips_definitions_with_null_retention(store: WorkflowStore):
    """``retention_days IS NULL`` means retain forever (design §5)."""
    spec = _signed_spec(retention_days=None)
    await store.insert_definition_for_test(spec)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    await store.backend.execute(
        f"INSERT INTO {store.RUNS_TABLE} (run_id, workflow_name, workflow_ver, "
        f"params_json, status, started_by_did, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("run-old", "release", 1, "{}", "completed", "did:web:k.example", long_ago),
    )
    purged = await store.purge_expired_runs(now=datetime.now(timezone.utc))
    assert purged == 0
