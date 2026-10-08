"""Disposable PostgreSQL schemas for tests that need a database of their own.

A PostgreSQL test database outlives the run: locally it is reused between
runs, and a serial run's cases all share it (an xdist worker gets a database
of its own, see ``tests/shared/postgres_worker_isolation.py``). A test whose
store refuses to adopt state it did not write (Hold's initialization witness
is the example that prompted this) passes against a fresh database and fails
on the next run. Giving such a test its own schema, dropped at teardown, makes it
independent of whatever ran before it.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from uuid import uuid4


def postgres_test_url(environ: Mapping[str, str] | None = None) -> str | None:
    """The PostgreSQL URL dual-backend tests run against, if any is set."""

    env = os.environ if environ is None else environ
    return (
        env.get("TEST_POSTGRES_URL")
        or env.get("KESTREL_DATABASE_URL")
        or env.get("DATABASE_URL")
    )


def with_search_path(url: str, schema: str) -> str:
    """*url* with its ``search_path`` replaced by *schema*."""

    parts = urlsplit(url)
    query = [
        (key, value) for key, value in parse_qsl(parts.query) if key != "search_path"
    ]
    query.append(("search_path", schema))
    return urlunsplit(parts._replace(query=urlencode(query)))


def database_url(url: str, database: str) -> str:
    """*url* naming *database*, on the server's default ``search_path``.

    A ``search_path`` in *url* names schemas of *url*'s database, which
    *database* need not have, so it is dropped; every other option is kept.
    """

    parts = urlsplit(url)
    query = [
        (key, value) for key, value in parse_qsl(parts.query) if key != "search_path"
    ]
    return urlunsplit(
        parts._replace(path="/" + quote(database, safe=""), query=urlencode(query))
    )


def quoted_search_path(*schemas: str) -> str:
    """A ``search_path`` value naming *schemas* in order, each quoted once."""

    unique = dict.fromkeys(schemas)
    return ",".join('"' + schema.replace('"', '""') + '"' for schema in unique)


async def pgvector_schema(db) -> str:
    """The schema holding pgvector's ``vector`` type, installing it if absent.

    The extension is installed once per database, into whichever schema first
    created it, and a connection whose ``search_path`` names only a test's
    own schema (``with_search_path(url, schema)``) does not see it. So
    ``CREATE EXTENSION IF NOT EXISTS vector`` can succeed while an
    unqualified ``vector`` still does not resolve (#3401). Qualify the type
    with this schema, or name it on the connection's ``search_path``.
    """

    await db.execute("CREATE EXTENSION IF NOT EXISTS vector")
    row = await db.fetchone(
        "SELECT n.nspname FROM pg_extension e "
        "JOIN pg_namespace n ON n.oid = e.extnamespace "
        "WHERE e.extname = 'vector'"
    )
    return row[0]


@asynccontextmanager
async def disposable_postgres_schema(admin, prefix: str) -> AsyncIterator[str]:
    """Create a uniquely named schema through *admin*; drop it on exit.

    *admin* is any connected backend on the target database. The schema is
    dropped with ``CASCADE`` whatever the body raised, so a failing test does
    not leave its tables for the next run to trip over.
    """

    schema = f"{prefix}_{uuid4().hex}"
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield schema
    finally:
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
