"""Give each xdist worker its own PostgreSQL database (#3383).

The unit tier runs ``pytest -n auto`` against one PostgreSQL server. Many
PostgreSQL cases boot the core schema (``AsyncDatabase._init_schema()``),
whose ``CREATE TABLE/INDEX IF NOT EXISTS`` loop is idempotent in sequence but
not in parallel: PostgreSQL tests for existence before it takes the lock that
would exclude a peer, so two workers booting the same fresh schema can both
proceed and one dies on a catalog unique index (``pg_type_typname_nsp_index``).
Tests that drop and rebuild core tables reopen the same window mid-run, so
initializing the schema once up front would not close it.

Instead, each worker creates a database of its own when it configures and
rewrites ``TEST_POSTGRES_URL`` to name it. Every reader of that variable — the
``db_backend`` fixture, a test opening a second connection to "the same
database", a child process — then agrees on the worker's database, and within
a worker tests run one at a time. The database is dropped when the worker
unconfigures.

A database, not a schema (#3401, #3515). A schema isolates only the names it
holds; a worker in one still shares the rest of the database. pgvector, which
the core schema's migrations install, exists once per database, in one
schema, and a worker resolves ``vector`` only with that schema on its
``search_path``, where it also resolves everything else the schema holds. A
serial run installs pgvector into ``public`` together with the core tables it
boots there, so a worker's unqualified ``DROP TABLE IF EXISTS`` removed a
table every run shares, and ``to_regclass`` reported it as the worker's own.
Moving pgvector to a schema of its own would break the next serial run, whose
migrations name ``vector`` unqualified, and the migrations' catalog probes
that no schema qualifies saw every other worker's tables regardless. A worker
in a database of its own shares none of this: it meets what a serial run meets
on a fresh database, and its migrations install pgvector into its own
``public`` exactly as a serial run's do, where the server provides it and the
role may install it.

The worker's URL keeps every option of the job's URL except ``search_path``,
which names schemas of the job's database that the worker's does not have:
the worker runs on the server's default ``search_path``, as a serial run on a
fresh database does.

Only xdist workers are isolated: a serial run has no peer to race, and its URL
is left as it was.

This does not relax #3381. A required run whose worker cannot create its
database (the server is unreachable, or the role may not create databases)
stops with a usage error. An unrequired one leaves the URL alone: its
PostgreSQL cases skip on a connection failure as they always did, and on a
server they can reach but whose role may not create a database they run
unisolated, in the job's own database.
"""

from __future__ import annotations

import asyncio
import os
import re
import warnings
from collections.abc import Coroutine, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import uuid4

import asyncpg
import pytest

from tests.shared.postgres_requirement import REQUIRE_ENV, URL_ENV, postgres_required
from tests.utils.postgres_schema import database_url, postgres_test_url

WORKER_ENV = "PYTEST_XDIST_WORKER"
CONNECT_TIMEOUT_SECONDS = 10.0
# ``FORCE`` ends a connection a test leaked into the worker's database, and
# the lock timeout bounds a wait on any other lock held on it, so neither
# hangs the session's end.
DROP_LOCK_TIMEOUT = "10s"

_T = TypeVar("_T")

_CONNECTION_ERRORS = (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError)


@dataclass(frozen=True)
class WorkerDatabase:
    """A database one worker owns, and the URL it was created through."""

    admin_url: str
    database: str


def worker_database_name(worker_id: str) -> str:
    """A fresh database name for *worker_id*, unique to this run."""

    worker = re.sub(r"[^a-z0-9_]", "_", worker_id.lower())
    return f"pytest_{worker}_{uuid4().hex[:12]}"


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    # A thread of its own: the caller may be pytest's configure hook or a
    # running event loop, and neither may be handed to ``asyncio.run``.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _execute(url: str, *statements: str) -> None:
    conn = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        for statement in statements:
            await conn.execute(statement)
    finally:
        await conn.close()


def isolate_xdist_worker(
    environ: MutableMapping[str, str] | None = None,
) -> WorkerDatabase | None:
    """Point this worker's ``TEST_POSTGRES_URL`` at a database of its own.

    Returns the created database, or ``None`` when this is not an xdist
    worker, no PostgreSQL URL is configured, or an unrequired run cannot
    create the database.
    """

    env = os.environ if environ is None else environ
    worker_id = env.get(WORKER_ENV)
    base_url = postgres_test_url(env)
    if not worker_id or not base_url:
        return None
    database = worker_database_name(worker_id)
    try:
        _run(_execute(base_url, f'CREATE DATABASE "{database}"'))
    except _CONNECTION_ERRORS as exc:
        if postgres_required(env):
            raise pytest.UsageError(
                f"{REQUIRE_ENV}=1 but xdist worker {worker_id} could not create "
                f"its PostgreSQL database: {exc!r}"
            ) from exc
        return None
    env[URL_ENV] = database_url(base_url, database)
    return WorkerDatabase(admin_url=base_url, database=database)


def release_worker_database(owned: WorkerDatabase | None) -> None:
    """Drop the database :func:`isolate_xdist_worker` created, if any."""

    if owned is None:
        return
    try:
        _run(
            _execute(
                owned.admin_url,
                f"SET lock_timeout = '{DROP_LOCK_TIMEOUT}'",
                f'DROP DATABASE IF EXISTS "{owned.database}" WITH (FORCE)',
            )
        )
    except _CONNECTION_ERRORS as exc:
        # The tests have already reported; a leftover database is litter on
        # a test server, not a verdict on the run.
        warnings.warn(
            f"could not drop xdist worker database {owned.database}: {exc!r}",
            stacklevel=2,
        )
