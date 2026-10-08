"""Give each xdist worker its own PostgreSQL schema (#3383).

The unit tier runs ``pytest -n auto`` against one PostgreSQL database. Many
PostgreSQL cases boot the core schema (``AsyncDatabase._init_schema()``),
whose ``CREATE TABLE/INDEX IF NOT EXISTS`` loop is idempotent in sequence but
not in parallel: PostgreSQL tests for existence before it takes the lock that
would exclude a peer, so two workers booting the same fresh schema can both
proceed and one dies on a catalog unique index (``pg_type_typname_nsp_index``).
Tests that drop and rebuild core tables reopen the same window mid-run, so
initializing the schema once up front would not close it.

Instead, each worker creates a schema of its own when it configures and
rewrites ``TEST_POSTGRES_URL`` to select it through ``search_path``. Every
reader of that variable — the ``db_backend`` fixture, a test opening a second
connection to "the same database", a child process — then agrees on the
worker's schema, and within a worker tests run one at a time. The schema is
dropped when the worker unconfigures.

pgvector is the exception (#3401). An extension is installed once per
database, into the first schema on the installing connection's
``search_path``, and the core schema's migrations install it with ``CREATE
EXTENSION IF NOT EXISTS vector``. Left to them, the first worker would put it
in its own schema: no other worker could resolve ``vector(N)``, and that
worker's drop would take every other worker's vector columns with it. So
each worker first installs pgvector where a serial run would (``public``,
normally), holding an advisory lock because ``CREATE EXTENSION IF NOT
EXISTS`` races like ``CREATE TABLE IF NOT EXISTS``, and puts that schema
after its own on the ``search_path``. Tables are created in, and resolve
first to, the worker's schema; only a name the worker's schema lacks falls
through to that shared one, so a test whose own table is gone (dropped and
not yet rebuilt) reaches whatever a serial run left there. Where pgvector is
absent and cannot be installed (the server lacks it, or the role is not a
superuser: it is not a trusted extension), the worker gets its own schema
alone.

Only xdist workers are isolated: a serial run has no peer to race, and its
``search_path`` is left as it was.

This does not relax #3381. A required run whose worker cannot set up its
schema stops with a usage error; an unrequired one leaves the URL alone, so
its PostgreSQL cases skip on the same connection failure they always did.
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
from tests.utils.postgres_schema import (
    postgres_test_url,
    quoted_search_path,
    with_search_path,
)

WORKER_ENV = "PYTEST_XDIST_WORKER"
CONNECT_TIMEOUT_SECONDS = 10.0
# A worker that leaked a connection holding a lock must not hang the
# session's end on the drop.
DROP_LOCK_TIMEOUT = "10s"
# The extension the core schema's migrations create, which every worker of
# the database must resolve from one shared schema.
SHARED_EXTENSION = "vector"
EXTENSION_LOCK = "kestrel.tests.postgres_worker_isolation.shared_extension"

_T = TypeVar("_T")

_CONNECTION_ERRORS = (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError)


@dataclass(frozen=True)
class WorkerSchema:
    """A schema one worker owns, and the URL it was created through."""

    admin_url: str
    schema: str


def worker_schema_name(worker_id: str) -> str:
    """A fresh schema name for *worker_id*, unique to this run."""

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


async def _shared_extension_schema(conn: asyncpg.Connection) -> str | None:
    """Install :data:`SHARED_EXTENSION` if absent; the schema it lives in.

    ``None`` when the server does not provide the extension, or this role may
    not install it: the migrations, running as the same role, cannot either.
    """

    available = await conn.fetchval(
        "SELECT 1 FROM pg_available_extensions WHERE name = $1", SHARED_EXTENSION
    )
    if not available:
        return None
    try:
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1))", EXTENSION_LOCK
            )
            await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{SHARED_EXTENSION}"')
    except asyncpg.InsufficientPrivilegeError:
        # pgvector is not a trusted extension, so only a superuser installs it.
        return None
    return await conn.fetchval(
        "SELECT n.nspname FROM pg_extension e "
        "JOIN pg_namespace n ON n.oid = e.extnamespace "
        "WHERE e.extname = $1",
        SHARED_EXTENSION,
    )


async def _create_worker_schema(url: str, schema: str) -> str | None:
    """Create *schema*; return the shared extension's schema, if any."""

    conn = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        # Before the worker's schema exists, so a failure leaves nothing.
        extension_schema = await _shared_extension_schema(conn)
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    finally:
        await conn.close()
    return extension_schema


def isolate_xdist_worker(
    environ: MutableMapping[str, str] | None = None,
) -> WorkerSchema | None:
    """Point this worker's ``TEST_POSTGRES_URL`` at a schema of its own.

    Returns the created schema, or ``None`` when this is not an xdist worker,
    no PostgreSQL URL is configured, or an unrequired run cannot reach it.
    """

    env = os.environ if environ is None else environ
    worker_id = env.get(WORKER_ENV)
    base_url = postgres_test_url(env)
    if not worker_id or not base_url:
        return None
    schema = worker_schema_name(worker_id)
    try:
        extension_schema = _run(_create_worker_schema(base_url, schema))
    except _CONNECTION_ERRORS as exc:
        if postgres_required(env):
            raise pytest.UsageError(
                f"{REQUIRE_ENV}=1 but xdist worker {worker_id} could not set up "
                f"its PostgreSQL schema: {exc!r}"
            ) from exc
        return None
    search_path = (schema,) if extension_schema is None else (schema, extension_schema)
    env[URL_ENV] = with_search_path(base_url, quoted_search_path(*search_path))
    return WorkerSchema(admin_url=base_url, schema=schema)


def release_worker_schema(owned: WorkerSchema | None) -> None:
    """Drop the schema :func:`isolate_xdist_worker` created, if any."""

    if owned is None:
        return
    try:
        _run(
            _execute(
                owned.admin_url,
                f"SET lock_timeout = '{DROP_LOCK_TIMEOUT}'",
                f'DROP SCHEMA IF EXISTS "{owned.schema}" CASCADE',
            )
        )
    except _CONNECTION_ERRORS as exc:
        # The tests have already reported; a leftover schema is litter in a
        # test database, not a verdict on the run.
        warnings.warn(
            f"could not drop xdist worker schema {owned.schema}: {exc!r}",
            stacklevel=2,
        )
