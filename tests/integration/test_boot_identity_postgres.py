"""#2894 — a freshly incepted agent must be able to boot on PostgreSQL.

Cannot be reproduced on SQLite: there the anchor *is* the runtime database, so
the birth record inception writes is the record startup reads. With
``KESTREL_DB_BACKEND=postgres`` they are two databases, and at the moment
``kestrel create`` finishes the second one does not exist yet — its schema is
created at agent boot. Startup resolved identity from it anyway, found nothing,
and refused; the boot that would have replicated the birth record into it
(#2871) runs inside ``KestrelAgent.initialize()``, downstream of that refusal.

Measured on ``171355ea``: ``/health`` 503 for the full 120s window,
``agent.log`` carrying ``ValueError: No agent found in the database. Please run
inception service first.``

Run against any throwaway PostgreSQL:

    TEST_POSTGRES_URL=postgresql://u:p@127.0.0.1:5432/db pytest \
        tests/integration/test_boot_identity_postgres.py

Each test gets its OWN database — that is the custody rule this resolver
enforces, so borrowing a shared one would test a configuration Kestrel refuses.
Skipped when the URL is not set, so CI stays green.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.main import get_agent_did_async

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_URL = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("KESTREL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)

if not POSTGRES_URL:  # pragma: no cover - environment gate
    pytest.skip(
        "TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL required",
        allow_module_level=True,
    )

CONSTITUTION = b"""# Kestrel Constitution (boot identity test)

## Book I: Universal Values

Honesty. Sovereignty. Transparency.
""" * 6


@pytest.fixture(autouse=True)
def _no_ambient_backend(monkeypatch):
    """The operator machine must not decide this test's answers — the point is
    that the *explicit arguments* choose PostgreSQL."""
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    monkeypatch.delenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH", raising=False)


def _url_for(database: str) -> str:
    parts = urlsplit(POSTGRES_URL)
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment)
    )


@pytest.fixture
async def empty_database():
    """A database that has never had an agent boot against it.

    This is the state ``kestrel create`` leaves behind, and it is why the
    defect exists: there is nothing in here, not even the schema.
    """
    import asyncpg

    name = f"kestrel_2894_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(POSTGRES_URL)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    try:
        yield _url_for(name)
    finally:
        admin = await asyncpg.connect(POSTGRES_URL)
        try:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()",
                name,
            )
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        finally:
            await admin.close()


async def _incept(tmp_path, monkeypatch, name="Kite2894"):
    """Incept for real, against a test-local governing source.

    ``config.CONSTITUTION_PATH`` has to move with it: inception refuses a
    source the periodic integrity audit would not recompute the same hash from,
    because an agent anchored elsewhere is guaranteed to safe-mode later
    (#2463). Passing the path alone is exactly that refused shape.
    """
    import kestrel_sovereign.config as ks_config

    tmp_path.mkdir(parents=True, exist_ok=True)
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION)
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(constitution_path))
    agent_dir = tmp_path / "agent_data" / name
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name=name,
        is_test_instance=True,
    )
    return agent_dir, creds.agent_did


async def test_startup_resolves_a_freshly_incepted_agent(
    tmp_path, monkeypatch, empty_database
):
    """The whole ticket, end to end through the real inception service."""
    agent_dir, agent_did = await _incept(tmp_path, monkeypatch)

    did = await get_agent_did_async(
        str(agent_dir), db_backend="postgres", database_url=empty_database
    )

    assert did == agent_did


async def test_the_runtime_database_really_is_empty_at_that_point(
    tmp_path, monkeypatch, empty_database
):
    """The premise, asserted rather than assumed.

    If inception ever starts writing the birth record straight to PostgreSQL,
    this fails and the test above stops being about anything.
    """
    import asyncpg

    await _incept(tmp_path, monkeypatch)

    connection = await asyncpg.connect(empty_database)
    try:
        graph_nodes = await connection.fetchval(
            "SELECT to_regclass('public.graph_nodes')"
        )
    finally:
        await connection.close()

    assert graph_nodes is None, (
        "PostgreSQL already has a graph_nodes table after inception; the "
        "#2894 premise no longer holds and this suite needs revisiting."
    )


async def test_a_neighbours_database_is_refused_rather_than_adopted(
    tmp_path, monkeypatch, empty_database
):
    """Two agents, one database each — then point the second agent's startup at
    the first agent's database. Booting this directory's identity against
    another agent's governance is the failure this whole cluster is about."""
    from kestrel_sovereign.storage import AsyncStorage, GraphNode

    _, neighbour_did = await _incept(
        tmp_path / "first", monkeypatch, name="Neighbour2894"
    )
    agent_dir, _ = await _incept(
        tmp_path / "second", monkeypatch, name="Kite2894"
    )

    async with AsyncStorage(backend="postgres", dsn=empty_database) as storage:
        storage.graph.bind_agent(neighbour_did)
        await storage.graph.add_node(
            GraphNode(
                node_id=neighbour_did,
                node_type="agent",
                label="Neighbour2894",
                properties={"name": "Neighbour2894"},
            )
        )

    with pytest.raises(ValueError) as excinfo:
        await get_agent_did_async(
            str(agent_dir), db_backend="postgres", database_url=empty_database
        )

    assert neighbour_did in str(excinfo.value)
