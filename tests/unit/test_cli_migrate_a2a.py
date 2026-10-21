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
