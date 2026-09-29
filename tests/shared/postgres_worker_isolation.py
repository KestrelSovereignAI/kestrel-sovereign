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

Only xdist workers are isolated: a serial run has no peer to race, and its
``search_path`` (including any extension installed in ``public``, such as
pgvector for the integration tier) is left as it was.

This does not relax #3381. A required run whose worker cannot create its
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
from typing import Any
from uuid import uuid4

import asyncpg
import pytest

from tests.shared.postgres_requirement import REQUIRE_ENV, URL_ENV, postgres_required
from tests.utils.postgres_schema import postgres_test_url, with_search_path

WORKER_ENV = "PYTEST_XDIST_WORKER"
CONNECT_TIMEOUT_SECONDS = 10.0
# A worker that leaked a connection holding a lock must not hang the
# session's end on the drop.
DROP_LOCK_TIMEOUT = "10s"

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


def _run(coro: Coroutine[Any, Any, None]) -> None:
    # A thread of its own: the caller may be pytest's configure hook or a
    # running event loop, and neither may be handed to ``asyncio.run``.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, coro).result()


async def _execute(url: str, *statements: str) -> None:
    conn = await asyncpg.connect(url, timeout=CONNECT_TIMEOUT_SECONDS)
    try:
        for statement in statements:
            await conn.execute(statement)
    finally:
        await conn.close()


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
        _run(_execute(base_url, f'CREATE SCHEMA "{schema}"'))
    except _CONNECTION_ERRORS as exc:
        if postgres_required(env):
            raise pytest.UsageError(
                f"{REQUIRE_ENV}=1 but xdist worker {worker_id} could not create "
                f"its PostgreSQL schema: {exc!r}"
            ) from exc
        return None
    env[URL_ENV] = with_search_path(base_url, schema)
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
