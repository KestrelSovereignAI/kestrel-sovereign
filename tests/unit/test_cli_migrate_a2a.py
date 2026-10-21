"""Exercise the shipped CLI in separate processes against owned legacy files."""

import asyncio
import json
import os
import subprocess
import sys

import pytest
import pytest_asyncio

from kestrel_sovereign.multi_agent.liveness import ServingRecord
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_a2a_legacy_cutover import legacy, ownership


@pytest_asyncio.fixture
async def draft(tmp_path):
    path = tmp_path / "agent.db"
    backend = SQLiteBackend(str(path))
    await backend.connect()
    try:
        await legacy(backend)
    finally:
        await backend.close()
    owners = tmp_path / "owners.json"
    from dataclasses import asdict

    owners.write_text(
        json.dumps({key: asdict(value) for key, value in ownership().items()})
    )
    return path, owners, tmp_path / "reviewed-plan.json"


async def cli(*args):
    return await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "kestrel_sovereign.cli", "migrate-a2a", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
    )


async def preview(draft):
    database, owners, plan = draft
    result = await cli(
        "preview", "--sqlite", database, "--owners", owners, "--output", plan
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["action"] == "preview"
    assert "Historical answer" not in result.stdout
    if os.name != "nt":
        assert plan.stat().st_mode & 0o777 == 0o600
    return result


async def test_actual_cli_previews_and_applies_only_the_reviewed_database(draft):
    database, _, plan = draft
    await preview(draft)
    result = await cli("apply", "--sqlite", database, "--plan", plan)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["action"] == "applied"
    assert receipt["migration_id"] == json.loads(plan.read_text())["migration_id"]
    backend = SQLiteBackend(str(database))
    await backend.connect()
    try:
        assert await backend.fetch_one("SELECT revision FROM a2a_protocol_schema") == (
            1,
        )
        assert await backend.fetch_one(
            "SELECT cognition_wake_eligible FROM a2a_tasks"
        ) == (0,)
        assert json.loads(
            (await backend.fetch_one("SELECT history FROM a2a_tasks"))[0]
        )[0]["messageId"]
    finally:
        await backend.close()
    repeated = await cli("apply", "--sqlite", database, "--plan", plan)
    assert repeated.returncode == 1 and "already versioned" in repeated.stderr


async def test_actual_cli_refuses_stale_or_modified_review_without_conversion(draft):
    database, _, plan = draft
    await preview(draft)
    review = json.loads(plan.read_text())
    review["tasks"][0]["native_sha256"] = "unreviewed-conversion"
    plan.write_text(json.dumps(review))
    result = await cli("apply", "--sqlite", database, "--plan", plan)
    assert result.returncode == 1 and "no longer matches" in result.stderr
    backend = SQLiteBackend(str(database))
    await backend.connect()
    try:
        assert not await backend.table_exists("a2a_protocol_schema")
        assert not await backend.table_exists("a2a_legacy_migrations")
    finally:
        await backend.close()


async def test_actual_cli_refuses_a_running_host_before_opening_migration_state(draft):
    database, owners, plan = draft
    record = ServingRecord.acquire(database.parent)
    try:
        result = await cli(
            "preview", "--sqlite", database, "--owners", owners, "--output", plan
        )
        assert (
            result.returncode == 1
            and "Stop the selected database's host" in result.stderr
        )
        assert not plan.exists()
    finally:
        record.release()


@pytest.mark.parametrize(
    "document",
    [
        "{",
        '{"a":1,"a":2}',
        '{"historical":NaN}',
        '{"historical":{"creator_agent_id":"did:forged"}}',
    ],
)
async def test_actual_cli_refuses_malformed_or_incomplete_provenance(draft, document):
    database, owners, plan = draft
    owners.write_text(document)
    result = await cli(
        "preview", "--sqlite", database, "--owners", owners, "--output", plan
    )
    assert result.returncode == 1 and "migration refused" in result.stderr
    assert not plan.exists()


async def test_postgres_apply_refuses_other_live_database_connections(
    native_a2a_backend_factory, tmp_path
):
    from types import SimpleNamespace

    from kestrel_sovereign.a2a.legacy_cutover import prepare_legacy_migration
    from kestrel_sovereign.cli_migrate_a2a import _run

    backend = native_a2a_backend_factory()
    if backend.backend_type != "postgres":
        pytest.skip("PostgreSQL connection quiescence contract")
    await backend.connect()
    try:
        await legacy(backend)
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan.review()))
        args = SimpleNamespace(
            sqlite=None,
            postgres_dsn_env="TEST_POSTGRES_URL",
            migration_action="apply",
            plan=path,
        )
        with pytest.raises(ValueError, match="Stop all other connections"):
            await _run(args)
        assert not await backend.table_exists("a2a_protocol_schema")
    finally:
        await backend.close()


@pytest.mark.parametrize("action", ["preview", "apply"])
@pytest.mark.parametrize("serving_path", ["alias", "target"])
async def test_symlink_database_cannot_hide_a_serving_host(
    draft, tmp_path, action, serving_path
):
    import hashlib

    database, owners, plan = draft
    profile = tmp_path / "profile"
    profile.mkdir()
    alias = profile / "agent.db"
    alias.symlink_to(database)
    if action == "apply":
        await preview((alias, owners, plan))
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    record = ServingRecord.acquire(
        profile if serving_path == "alias" else database.parent
    )
    try:
        arguments = [action, "--sqlite", alias]
        arguments.extend(
            ["--owners", owners, "--output", plan]
            if action == "preview"
            else ["--plan", plan]
        )
        result = await cli(*arguments)
        assert (
            result.returncode == 1
            and "Stop the selected database's host" in result.stderr
        )
        assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    finally:
        record.release()


async def test_actual_cli_reviews_and_retires_old_signal_history_without_preview_writes(
    draft,
):
    import hashlib

    from kestrel_sovereign.signals.durable import DurableSignalStore
    from tests.unit.test_a2a_legacy_signal_cutover import consumer, old_signal

    database, _, plan = draft
    backend = SQLiteBackend(str(database))
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        accepted = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
    finally:
        await backend.close()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    result = await preview(draft)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    review = json.loads(plan.read_text())
    assert review["revision"] == 2 and len(review["retired_signal_histories"]) == 1
    assert review["retired_signal_histories"][0]["sequence"] == accepted.source_sequence
    assert "Historical input" not in result.stdout
    result = await cli("apply", "--sqlite", database, "--plan", plan)
    assert result.returncode == 0, result.stderr
    backend = SQLiteBackend(str(database))
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.register_consumer(consumer())
        assert await store.list_deliveries(agent_id="did:recipient") == []
        archives = json.loads(result.stdout)["archives"]
        assert await backend.fetch_one(
            f'SELECT event_id FROM "{archives[store.EVENTS]}"'
        ) == (accepted.event_id,)
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "invalid", ["old-revision", "non-array", "malformed-boundary", "duplicate-scope"]
)
async def test_actual_cli_refuses_unreviewed_or_invalid_signal_history_plan(
    draft, invalid
):
    from kestrel_sovereign.signals.durable import DurableSourceBoundary
    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    database, _, plan = draft
    await preview(draft)
    review = json.loads(plan.read_text())
    if invalid == "old-revision":
        review["revision"] = 1
    elif invalid == "non-array":
        review["retired_signal_histories"] = {}
    elif invalid == "malformed-boundary":
        review["retired_signal_histories"] = [{"sequence": -1}]
    else:
        boundary = DurableSourceBoundary("did:recipient", SOURCE_NAME, 1).to_dict()
        review["retired_signal_histories"] = [boundary, boundary]
    plan.write_text(json.dumps(review))
    result = await cli("apply", "--sqlite", database, "--plan", plan)
    assert result.returncode == 1
    if invalid == "duplicate-scope":
        assert "repeats a durable source history" in result.stderr
    backend = SQLiteBackend(str(database))
    await backend.connect()
    try:
        assert not await backend.table_exists("a2a_protocol_schema")
        assert not await backend.table_exists("a2a_legacy_migrations")
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "invalid", ["old-revision", "non-array", "boundary-version", "duplicate-scope"]
)
async def test_review_parser_rejects_invalid_history_contract_before_returning_owners(
    draft, invalid
):
    from kestrel_sovereign.cli_migrate_a2a import _plan_ownership
    from kestrel_sovereign.signals.durable import DurableSourceBoundary
    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    await preview(draft)
    review = json.loads(draft[2].read_text())
    boundary = DurableSourceBoundary("did:recipient", SOURCE_NAME, 1).to_dict()
    if invalid == "old-revision":
        review["revision"] = 1
    elif invalid == "non-array":
        review["retired_signal_histories"] = {}
    elif invalid == "boundary-version":
        boundary["version"] = 99
        review["retired_signal_histories"] = [boundary]
    else:
        review["retired_signal_histories"] = [boundary, boundary]
    with pytest.raises(ValueError):
        _plan_ownership(review)
