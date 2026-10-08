"""xdist workers do not race each other through PostgreSQL's catalogs (#3383).

The unit tier runs ``-n auto`` against one PostgreSQL database, and
``AsyncDatabase._init_schema()`` is idempotent in sequence but not in
parallel: two workers booting the same fresh schema fail on a catalog unique
index. Each worker therefore gets a schema of its own. The race test boots
the core schema from several simulated workers at once, all handed the same
fresh URL; without the isolation they share that schema and it fails.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.shared import postgres_worker_isolation
from tests.shared.postgres_requirement import REQUIRE_ENV, URL_ENV, postgres_required
from tests.shared.postgres_worker_isolation import (
    CONNECT_TIMEOUT_SECONDS,
    WORKER_ENV,
    WorkerSchema,
    isolate_xdist_worker,
    release_worker_schema,
    worker_schema_name,
)
from tests.utils.postgres_schema import (
    disposable_postgres_schema,
    postgres_test_url,
    quoted_search_path,
    with_search_path,
)

WORKERS = 4
UNREACHABLE = "postgresql://u:p@127.0.0.1:1/unreachable"


# ---------------------------------------------------------------------------
# Which processes are isolated
# ---------------------------------------------------------------------------


def test_a_process_that_is_not_an_xdist_worker_keeps_its_url():
    env = {URL_ENV: UNREACHABLE}
    assert isolate_xdist_worker(env) is None
    assert env == {URL_ENV: UNREACHABLE}


def test_a_worker_without_a_postgres_url_has_nothing_to_isolate():
    env = {WORKER_ENV: "gw0"}
    assert isolate_xdist_worker(env) is None
    assert env == {WORKER_ENV: "gw0"}


def search_path_of(url: str) -> str:
    return dict(parse_qsl(urlsplit(url).query)).get("search_path", "")


def test_this_worker_runs_in_a_schema_of_its_own():
    """The conftest wiring, not just the function: CI's workers are isolated."""
    worker = os.environ.get(WORKER_ENV)
    if not worker or not postgres_required():
        pytest.skip("only a required xdist run is certain to isolate its workers")
    assert search_path_of(os.environ[URL_ENV]).startswith(f'"pytest_{worker}_')


def test_worker_schema_names_are_unique_per_run_and_safe_identifiers():
    first, second = worker_schema_name("gw0"), worker_schema_name("gw0")
    assert first != second
    assert first.startswith("pytest_gw0_")
    assert worker_schema_name('gw"1; x').startswith("pytest_gw_1__x_")


# ---------------------------------------------------------------------------
# An unreachable database: #3381 still decides
# ---------------------------------------------------------------------------


def test_a_required_worker_that_cannot_create_its_schema_stops_the_session():
    env = {WORKER_ENV: "gw0", URL_ENV: UNREACHABLE, REQUIRE_ENV: "1"}
    with pytest.raises(pytest.UsageError, match="gw0"):
        isolate_xdist_worker(env)
    assert env[URL_ENV] == UNREACHABLE


def test_an_unrequired_worker_keeps_its_url_so_its_cases_skip_as_before():
    env = {WORKER_ENV: "gw0", URL_ENV: UNREACHABLE}
    assert isolate_xdist_worker(env) is None
    assert env[URL_ENV] == UNREACHABLE


def test_a_schema_that_cannot_be_dropped_warns_instead_of_failing_the_run():
    with pytest.warns(UserWarning, match="pytest_gw0_x"):
        release_worker_schema(
            WorkerSchema(admin_url=UNREACHABLE, schema="pytest_gw0_x")
        )


def test_releasing_nothing_is_a_no_op():
    release_worker_schema(None)


# ---------------------------------------------------------------------------
# The race itself, on real PostgreSQL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
# Parametrized with "postgres" so the #3381 guard fails this case, rather
# than letting it skip, in a job that provides PostgreSQL.
@pytest.mark.parametrize("backend", ["postgres"])
async def test_concurrent_workers_boot_the_core_schema_from_one_fresh_url(backend):
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    url = postgres_test_url()
    if not url:
        pytest.skip(f"{URL_ENV} required for the PostgreSQL race")
    admin = PostgresBackend(url)
    try:
        await admin.connect()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"PostgreSQL not available: {exc}")
    try:
        async with disposable_postgres_schema(admin, "worker_race") as fresh:
            shared = with_search_path(url, fresh)
            environs = [
                {WORKER_ENV: f"gw{i}", URL_ENV: shared} for i in range(WORKERS)
            ]
            owned = [isolate_xdist_worker(env) for env in environs]
            backends = [PostgresBackend(env[URL_ENV]) for env in environs]
            try:
                for worker_backend in backends:
                    await worker_backend.connect()
                results = await asyncio.gather(
                    *(AsyncDatabase(b)._init_schema() for b in backends),
                    return_exceptions=True,
                )
                failures = [repr(r) for r in results if r is not None]
                assert not failures, failures

                schemas = [o.schema for o in owned if o is not None]
                assert len(set(schemas)) == WORKERS
                booted = await admin.fetch_all(
                    "SELECT schemaname FROM pg_tables "
                    "WHERE tablename = 'graph_nodes' AND schemaname = ANY(?)",
                    (schemas,),
                )
                assert sorted(row[0] for row in booted) == sorted(schemas)
            finally:
                for worker_backend in backends:
                    await worker_backend.close()
                for schema in owned:
                    release_worker_schema(schema)

            left = await admin.fetch_all(
                "SELECT nspname FROM pg_namespace WHERE nspname = ANY(?)",
                ([o.schema for o in owned if o is not None],),
            )
            assert left == []
    finally:
        await admin.close()


# ---------------------------------------------------------------------------
# pgvector: one extension per database, so every worker must see it (#3401)
# ---------------------------------------------------------------------------


def database_url(url: str, database: str) -> str:
    """*url* naming *database*, with the job's own ``search_path`` default."""

    parts = urlsplit(url)
    query = [
        (key, value) for key, value in parse_qsl(parts.query) if key != "search_path"
    ]
    return urlunsplit(parts._replace(path=f"/{database}", query=urlencode(query)))


@asynccontextmanager
async def fresh_database() -> AsyncIterator[str]:
    """A new database on the test server, with no extension installed.

    pgvector is installed once per database, and the shared test database
    has it from the moment this run's first worker configured, so only a
    database of its own shows what the first workers of a run meet.
    """

    url = postgres_test_url()
    if not url:
        pytest.skip(f"{URL_ENV} required for a fresh PostgreSQL database")
    try:
        admin = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    except (OSError, TimeoutError, asyncpg.PostgresError) as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")
    name = f"pytest_fresh_{uuid4().hex[:12]}"
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
        try:
            yield database_url(url, name)
        finally:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await admin.close()


@asynccontextmanager
async def login_role() -> AsyncIterator[tuple[str, str]]:
    """A role that may log in and nothing more; its name and password."""

    url = postgres_test_url()
    if not url:
        pytest.skip(f"{URL_ENV} required for a PostgreSQL role")
    try:
        admin = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    except (OSError, TimeoutError, asyncpg.PostgresError) as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")
    name, password = f"pytest_role_{uuid4().hex[:12]}", uuid4().hex
    try:
        await admin.execute(f"CREATE ROLE \"{name}\" LOGIN PASSWORD '{password}'")
        try:
            yield name, password
        finally:
            await admin.execute(f'DROP ROLE IF EXISTS "{name}"')
    finally:
        await admin.close()


def url_as(url: str, role: str, password: str) -> str:
    parts = urlsplit(url)
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit(parts._replace(netloc=f"{role}:{password}@{host}"))


async def extension_schema(db: AsyncDatabase) -> str | None:
    row = await db.fetchone(
        "SELECT n.nspname FROM pg_extension e "
        "JOIN pg_namespace n ON n.oid = e.extnamespace "
        "WHERE e.extname = 'vector'"
    )
    return None if row is None else row[0]


async def installed_extensions(url: str) -> list[str]:
    conn = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        rows = await conn.fetch("SELECT extname FROM pg_extension ORDER BY extname")
    finally:
        await conn.close()
    return [row["extname"] for row in rows]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
async def test_workers_on_a_fresh_database_all_resolve_one_pgvector(backend):
    """Two workers boot a fresh database; each can still name ``vector``.

    The core schema's startup migrations run ``CREATE EXTENSION IF NOT
    EXISTS vector``. With only its own schema on the ``search_path``, the
    first worker to run it installs pgvector into that private schema, and
    the other worker's ``vector(N)`` then does not resolve. The first
    worker's release would also drop the extension, and with it every
    other worker's vector column.
    """
    from kestrel_sovereign.storage.db.postgres import PostgresBackend
    from kestrel_sovereign.storage.embedding_column import (
        ensure_embedding_vec_column,
    )

    async with fresh_database() as fresh:
        environs = [
            {WORKER_ENV: f"gw{i}", URL_ENV: fresh, REQUIRE_ENV: "1"}
            for i in range(2)
        ]
        owned = [isolate_xdist_worker(env) for env in environs]
        backends = [PostgresBackend(env[URL_ENV]) for env in environs]
        try:
            for worker_backend in backends:
                await worker_backend.connect()
            databases = [AsyncDatabase(b) for b in backends]
            results = await asyncio.gather(
                *(db._init_schema() for db in databases), return_exceptions=True
            )
            failures = [repr(r) for r in results if r is not None]
            assert not failures, failures

            schemas = [o.schema for o in owned]
            installed_in = await extension_schema(databases[0])
            assert installed_in is not None
            assert installed_in not in schemas, (
                f"pgvector was installed into worker schema {installed_in}"
            )
            # The first embedded write's DDL: ``vector(2)`` in the worker's
            # own ``document_chunks``.
            resolved = [
                await ensure_embedding_vec_column(db, "document_chunks", 2)
                for db in databases
            ]
            assert resolved == [True, True]

            await backends[0].close()
            release_worker_schema(owned[0])
            owned[0] = None
            assert await extension_schema(databases[1]) == installed_in
            assert await databases[1].column_exists("document_chunks", "embedding_vec")
        finally:
            for worker_backend in backends:
                await worker_backend.close()
            for schema in owned:
                release_worker_schema(schema)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
async def test_workers_configuring_at_once_install_pgvector_once(backend):
    """``CREATE EXTENSION IF NOT EXISTS`` races, so configuring is serialized.

    Real workers configure together. Without the lock two of them create
    pgvector at once on a fresh database, and the loser stops its session
    on a catalog unique index.
    """
    async with fresh_database() as fresh:
        environs = [
            {WORKER_ENV: f"gw{i}", URL_ENV: fresh, REQUIRE_ENV: "1"}
            for i in range(WORKERS)
        ]
        results = await asyncio.gather(
            *(asyncio.to_thread(isolate_xdist_worker, env) for env in environs),
            return_exceptions=True,
        )
        owned = [r for r in results if isinstance(r, WorkerSchema)]
        try:
            failures = [repr(r) for r in results if not isinstance(r, WorkerSchema)]
            assert not failures, failures
            assert [search_path_of(env[URL_ENV]) for env in environs] == [
                quoted_search_path(o.schema, "public") for o in owned
            ]
            assert "vector" in await installed_extensions(fresh)
        finally:
            for schema in owned:
                release_worker_schema(schema)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
async def test_a_server_without_the_extension_gives_the_worker_its_schema_alone(
    backend, monkeypatch
):
    monkeypatch.setattr(
        postgres_worker_isolation, "SHARED_EXTENSION", "kestrel_no_such_extension"
    )
    async with fresh_database() as fresh:
        env = {WORKER_ENV: "gw0", URL_ENV: fresh, REQUIRE_ENV: "1"}
        owned = isolate_xdist_worker(env)
        try:
            assert owned is not None
            assert search_path_of(env[URL_ENV]) == quoted_search_path(owned.schema)
            assert await installed_extensions(fresh) == ["plpgsql"]
        finally:
            release_worker_schema(owned)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
async def test_a_role_that_may_not_install_the_extension_is_still_isolated(backend):
    """pgvector is untrusted: a role that is not superuser cannot install it.

    Nor can the migrations running as that role, so the worker keeps its own
    schema alone rather than losing its isolation over the extension.
    """
    async with login_role() as (role, password), fresh_database() as fresh:
        conn = await asyncpg.connect(fresh, timeout=CONNECT_TIMEOUT_SECONDS)
        try:
            database = await conn.fetchval("SELECT current_database()")
            await conn.execute(f'GRANT CREATE ON DATABASE "{database}" TO "{role}"')
        finally:
            await conn.close()
        env = {WORKER_ENV: "gw0", URL_ENV: url_as(fresh, role, password)}
        owned = isolate_xdist_worker(env)
        try:
            assert owned is not None
            assert search_path_of(env[URL_ENV]) == quoted_search_path(owned.schema)
            assert await installed_extensions(fresh) == ["plpgsql"]
        finally:
            release_worker_schema(owned)
