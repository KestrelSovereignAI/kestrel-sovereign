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
from urllib.parse import parse_qsl, urlsplit

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.shared.postgres_requirement import REQUIRE_ENV, URL_ENV, postgres_required
from tests.shared.postgres_worker_isolation import (
    WORKER_ENV,
    WorkerSchema,
    isolate_xdist_worker,
    release_worker_schema,
    worker_schema_name,
)
from tests.utils.postgres_schema import (
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


def test_this_worker_runs_in_a_schema_of_its_own():
    """The conftest wiring, not just the function: CI's workers are isolated."""
    worker = os.environ.get(WORKER_ENV)
    if not worker or not postgres_required():
        pytest.skip("only a required xdist run is certain to isolate its workers")
    query = dict(parse_qsl(urlsplit(os.environ[URL_ENV]).query))
    assert query.get("search_path", "").startswith(f"pytest_{worker}_")


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
