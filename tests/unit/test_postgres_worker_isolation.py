"""xdist workers do not race or reach each other through PostgreSQL (#3383).

The unit tier runs ``-n auto`` against one PostgreSQL server, and
``AsyncDatabase._init_schema()`` is idempotent in sequence but not in
parallel: two workers booting the same fresh schema fail on a catalog unique
index. Each worker therefore gets a database of its own. The race test boots
the core schema from several simulated workers at once, all handed the same
fresh URL; without the isolation they share that schema and it fails.

A schema of their own was not enough (#3401, #3515). The database around it
stays shared: pgvector, installed once per database, and every table in the
schema pgvector lives in, which a worker had to put on its ``search_path`` to
resolve ``vector``. The last section shows a worker reaches neither.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.shared.postgres_requirement import REQUIRE_ENV, URL_ENV, postgres_required
from tests.shared.postgres_worker_isolation import (
    CONNECT_TIMEOUT_SECONDS,
    WORKER_ENV,
    WorkerDatabase,
    isolate_xdist_worker,
    release_worker_database,
    worker_database_name,
)
from tests.utils.postgres_schema import (
    database_url,
    disposable_postgres_schema,
    postgres_test_url,
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


def database_of(url: str) -> str:
    return unquote(urlsplit(url).path.lstrip("/"))


def search_path_of(url: str) -> str:
    return dict(parse_qsl(urlsplit(url).query)).get("search_path", "")


def test_this_worker_runs_in_a_database_of_its_own():
    """The conftest wiring, not just the function: CI's workers are isolated."""
    worker = os.environ.get(WORKER_ENV)
    if not worker or not postgres_required():
        pytest.skip("only a required xdist run is certain to isolate its workers")
    assert database_of(os.environ[URL_ENV]).startswith(f"pytest_{worker}_")
    assert search_path_of(os.environ[URL_ENV]) == ""


def test_worker_database_names_are_unique_per_run_and_safe_identifiers():
    first, second = worker_database_name("gw0"), worker_database_name("gw0")
    assert first != second
    assert first.startswith("pytest_gw0_")
    assert worker_database_name('gw"1; x').startswith("pytest_gw_1__x_")


def test_a_worker_url_names_its_database_and_keeps_every_other_option():
    """The job's ``search_path`` names its schemas, not the worker's."""
    url = "postgresql://u:p@h:5433/kestrel?sslmode=disable&search_path=tenant"
    assert database_url(url, "pytest_gw0_x") == (
        "postgresql://u:p@h:5433/pytest_gw0_x?sslmode=disable"
    )


# ---------------------------------------------------------------------------
# An unreachable database: #3381 still decides
# ---------------------------------------------------------------------------


def test_a_required_worker_that_cannot_create_its_database_stops_the_session():
    env = {WORKER_ENV: "gw0", URL_ENV: UNREACHABLE, REQUIRE_ENV: "1"}
    with pytest.raises(pytest.UsageError, match="gw0"):
        isolate_xdist_worker(env)
    assert env[URL_ENV] == UNREACHABLE


def test_an_unrequired_worker_keeps_its_url_so_its_cases_skip_as_before():
    env = {WORKER_ENV: "gw0", URL_ENV: UNREACHABLE}
    assert isolate_xdist_worker(env) is None
    assert env[URL_ENV] == UNREACHABLE


def test_a_database_that_cannot_be_dropped_warns_instead_of_failing_the_run():
    with pytest.warns(UserWarning, match="pytest_gw0_x"):
        release_worker_database(
            WorkerDatabase(admin_url=UNREACHABLE, database="pytest_gw0_x")
        )


def test_releasing_nothing_is_a_no_op():
    release_worker_database(None)


# ---------------------------------------------------------------------------
# The race itself, on real PostgreSQL
# ---------------------------------------------------------------------------


async def databases_named(url: str, names: list[str]) -> list[str]:
    conn = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        rows = await conn.fetch(
            "SELECT datname FROM pg_database WHERE datname = ANY($1) ORDER BY 1",
            names,
        )
    finally:
        await conn.close()
    return [row["datname"] for row in rows]


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
            owned: list[WorkerDatabase | None] = []
            backends = []
            try:
                for env in environs:
                    owned.append(isolate_xdist_worker(env))
                    backends.append(PostgresBackend(env[URL_ENV]))
                databases = [o.database for o in owned if o is not None]
                for worker_backend in backends:
                    await worker_backend.connect()
                results = await asyncio.gather(
                    *(AsyncDatabase(b)._init_schema() for b in backends),
                    return_exceptions=True,
                )
                failures = [repr(r) for r in results if r is not None]
                assert not failures, failures

                assert len(set(databases)) == WORKERS
                booted = [await b.table_exists("graph_nodes") for b in backends]
                assert booted == [True] * WORKERS
                assert await admin.fetch_all(
                    "SELECT tablename FROM pg_tables WHERE schemaname = ?", (fresh,)
                ) == []
            finally:
                for worker_backend in backends:
                    await worker_backend.close()
                for database in owned:
                    release_worker_database(database)

            assert await databases_named(url, databases) == []
    finally:
        await admin.close()


# ---------------------------------------------------------------------------
# What a schema of its own still shared (#3401, #3515)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def fresh_database() -> AsyncIterator[str]:
    """A new database on the test server, with no extension installed.

    It stands for a job's database as a run first meets it, so what a
    serial run leaves in it, and whether a worker reaches that, is visible.
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
async def login_role(*attributes: str) -> AsyncIterator[tuple[str, str]]:
    """A role that may log in, plus *attributes*; its name and password."""

    url = postgres_test_url()
    if not url:
        pytest.skip(f"{URL_ENV} required for a PostgreSQL role")
    try:
        admin = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    except (OSError, TimeoutError, asyncpg.PostgresError) as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")
    name, password = f"pytest_role_{uuid4().hex[:12]}", uuid4().hex
    options = " ".join(("LOGIN", *attributes))
    try:
        await admin.execute(f"CREATE ROLE \"{name}\" {options} PASSWORD '{password}'")
        try:
            yield name, password
        finally:
            # A role cannot be dropped while it owns a database, and a case
            # whose own release failed has already failed on that.
            for row in await admin.fetch(
                "SELECT datname FROM pg_database "
                "WHERE datdba = (SELECT oid FROM pg_roles WHERE rolname = $1)",
                name,
            ):
                await admin.execute(
                    f'DROP DATABASE IF EXISTS "{row["datname"]}" WITH (FORCE)'
                )
            await admin.execute(f'DROP ROLE IF EXISTS "{name}"')
    finally:
        await admin.close()


def url_as(url: str, role: str, password: str) -> str:
    parts = urlsplit(url)
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit(parts._replace(netloc=f"{role}:{password}@{host}"))


async def public_tables(url: str) -> set[str]:
    conn = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
    finally:
        await conn.close()
    return {row["tablename"] for row in rows}


async def installed_extensions(url: str) -> list[str]:
    conn = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        rows = await conn.fetch("SELECT extname FROM pg_extension ORDER BY extname")
    finally:
        await conn.close()
    return [row["extname"] for row in rows]


async def extension_schema(db: AsyncDatabase) -> str | None:
    row = await db.fetchone(
        "SELECT n.nspname FROM pg_extension e "
        "JOIN pg_namespace n ON n.oid = e.extnamespace "
        "WHERE e.extname = 'vector'"
    )
    return None if row is None else row[0]


async def require_installable_pgvector(url: str) -> None:
    """Skip unless a database on this server may install pgvector.

    A worker's migrations install it into the worker's database, as a serial
    run's do, so the server must provide it and this role must be allowed to
    create it: pgvector is not a trusted extension. Nothing stays installed.

    A required run (#3381) turns this skip into a failure. That is intended:
    the unit tier's other pgvector cases already need both, so a job that
    requires PostgreSQL must provide them, as CI's ``pgvector/pgvector``
    service and its superuser do.
    """

    conn = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        available = await conn.fetchval(
            "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
        )
        if not available:
            pytest.skip("this PostgreSQL server does not provide pgvector")
        probe = conn.transaction()
        await probe.start()
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except asyncpg.InsufficientPrivilegeError:
            pytest.skip("this role may not install pgvector, an untrusted extension")
        finally:
            await probe.rollback()
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
async def test_a_worker_cannot_reach_the_tables_a_serial_run_left(backend):
    """A serial run's tables stay out of a worker's reach (#3515).

    A serial run boots the core schema into ``public`` and installs pgvector
    there with it, where the server provides it. A worker in a schema of its
    own had to put ``public`` on its ``search_path`` to resolve ``vector``,
    and every name its own schema lacked then fell through to the serial
    run's tables: ``to_regclass`` reported ``saved_items`` as the worker's,
    and the worker's unqualified ``DROP TABLE IF EXISTS`` removed it.
    """
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    async with fresh_database() as fresh:
        serial = PostgresBackend(fresh)
        await serial.connect()
        try:
            await AsyncDatabase(serial)._init_schema()
        finally:
            await serial.close()
        assert "saved_items" in await public_tables(fresh)

        env = {WORKER_ENV: "gw0", URL_ENV: fresh, REQUIRE_ENV: "1"}
        owned = isolate_xdist_worker(env)
        worker = PostgresBackend(env[URL_ENV])
        try:
            await worker.connect()
            seen = await worker.table_exists("saved_items")
            await worker.execute("DROP TABLE IF EXISTS saved_items")
        finally:
            await worker.close()
            release_worker_database(owned)
        left = "saved_items" in await public_tables(fresh)
        assert (seen, left) == (False, True), "the worker reached the serial run's table"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
async def test_workers_each_resolve_pgvector_and_keep_it_past_a_peers_release(
    backend,
):
    """Every worker can name ``vector``, and no peer's release takes it (#3401).

    With one shared extension, the first worker to boot installed pgvector
    into its own schema, out of the other worker's reach, and that worker's
    release then dropped it, with every other worker's vector column. Each
    worker's migrations now install it into the worker's own database.
    """
    from kestrel_sovereign.storage.db.postgres import PostgresBackend
    from kestrel_sovereign.storage.embedding_column import (
        ensure_embedding_vec_column,
    )

    async with fresh_database() as fresh:
        await require_installable_pgvector(fresh)
        environs = [
            {WORKER_ENV: f"gw{i}", URL_ENV: fresh, REQUIRE_ENV: "1"}
            for i in range(2)
        ]
        owned: list[WorkerDatabase | None] = []
        backends = []
        try:
            for env in environs:
                owned.append(isolate_xdist_worker(env))
                backends.append(PostgresBackend(env[URL_ENV]))
            for worker_backend in backends:
                await worker_backend.connect()
            databases = [AsyncDatabase(b) for b in backends]
            results = await asyncio.gather(
                *(db._init_schema() for db in databases), return_exceptions=True
            )
            failures = [repr(r) for r in results if r is not None]
            assert not failures, failures

            # The first embedded write's DDL: ``vector(2)`` in the worker's
            # own ``document_chunks``.
            resolved = [
                await ensure_embedding_vec_column(db, "document_chunks", 2)
                for db in databases
            ]
            assert resolved == [True, True]
            installed_in = [await extension_schema(db) for db in databases]
            assert None not in installed_in

            await backends[0].close()
            release_worker_database(owned[0])
            owned[0] = None
            assert await extension_schema(databases[1]) == installed_in[1]
            assert await databases[1].column_exists("document_chunks", "embedding_vec")
            assert await installed_extensions(fresh) == ["plpgsql"]
        finally:
            for worker_backend in backends:
                await worker_backend.close()
            for database in owned:
                release_worker_database(database)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
async def test_workers_configuring_at_once_each_create_a_database(backend):
    """Real workers configure together, and each gets a database."""
    async with fresh_database() as fresh:
        environs = [
            {WORKER_ENV: f"gw{i}", URL_ENV: fresh, REQUIRE_ENV: "1"}
            for i in range(WORKERS)
        ]
        results = await asyncio.gather(
            *(asyncio.to_thread(isolate_xdist_worker, env) for env in environs),
            return_exceptions=True,
        )
        owned = [r for r in results if isinstance(r, WorkerDatabase)]
        try:
            failures = [repr(r) for r in results if not isinstance(r, WorkerDatabase)]
            assert not failures, failures
            names = [o.database for o in owned]
            assert [database_of(env[URL_ENV]) for env in environs] == names
            assert await databases_named(fresh, names) == sorted(names)
        finally:
            for database in owned:
                release_worker_database(database)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
async def test_a_connection_a_test_leaked_does_not_keep_the_worker_database(backend):
    """The worker's database is dropped even while a connection still uses it."""
    async with fresh_database() as fresh:
        env = {WORKER_ENV: "gw0", URL_ENV: fresh, REQUIRE_ENV: "1"}
        owned = isolate_xdist_worker(env)
        try:
            leaked = await asyncpg.connect(
                env[URL_ENV], timeout=CONNECT_TIMEOUT_SECONDS
            )
            try:
                release_worker_database(owned)
                assert await databases_named(fresh, [owned.database]) == []
            finally:
                leaked.terminate()
        finally:
            release_worker_database(owned)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
async def test_a_role_that_may_create_databases_is_isolated_without_superuser(
    backend,
):
    """Isolation needs ``CREATEDB``, not the superuser pgvector needs."""
    async with login_role("CREATEDB") as (role, password), fresh_database() as fresh:
        env = {
            WORKER_ENV: "gw0",
            URL_ENV: url_as(fresh, role, password),
            REQUIRE_ENV: "1",
        }
        owned = isolate_xdist_worker(env)
        try:
            assert owned is not None
            conn = await asyncpg.connect(env[URL_ENV], timeout=CONNECT_TIMEOUT_SECONDS)
            try:
                assert await conn.fetchval("SELECT current_database()") == owned.database
            finally:
                await conn.close()
        finally:
            release_worker_database(owned)
        assert await databases_named(fresh, [owned.database]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["postgres"])
@pytest.mark.parametrize("required", [True, False], ids=["required", "unrequired"])
async def test_a_role_that_may_not_create_databases_is_refused_or_left_alone(
    backend, required
):
    """A required run stops; an unrequired one keeps the job's URL (#3381)."""
    async with login_role() as (role, password), fresh_database() as fresh:
        role_url = url_as(fresh, role, password)
        env = {WORKER_ENV: "gw0", URL_ENV: role_url}
        if required:
            env[REQUIRE_ENV] = "1"
            with pytest.raises(pytest.UsageError, match="gw0"):
                isolate_xdist_worker(env)
        else:
            assert isolate_xdist_worker(env) is None
        assert env[URL_ENV] == role_url
