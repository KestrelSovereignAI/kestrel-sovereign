"""Disposable PostgreSQL schemas for tests that need a database of their own.

A PostgreSQL test database outlives the run: locally it is reused between
runs, and in CI every xdist worker shares one. A test whose store refuses to
adopt state it did not write (Hold's initialization witness is the example
that prompted this) passes against a fresh database and fails on the next
run. Giving such a test its own schema, dropped at teardown, makes it
independent of whatever ran before it.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4


def postgres_test_url() -> str | None:
    """The PostgreSQL URL dual-backend tests run against, if any is set."""

    return (
        os.environ.get("TEST_POSTGRES_URL")
        or os.environ.get("KESTREL_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
    )


def with_search_path(url: str, schema: str) -> str:
    """*url* with its ``search_path`` replaced by *schema*."""

    parts = urlsplit(url)
    query = [
        (key, value) for key, value in parse_qsl(parts.query) if key != "search_path"
    ]
    query.append(("search_path", schema))
    return urlunsplit(parts._replace(query=urlencode(query)))


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
