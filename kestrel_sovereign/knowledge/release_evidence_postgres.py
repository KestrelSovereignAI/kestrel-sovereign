"""Disposable PostgreSQL databases for semantic release evidence.

Release evidence must never point a parity test or benchmark at an ambient
``TEST_POSTGRES_URL``/application database.  This module creates one uniquely
named database beneath an explicitly acknowledged isolated admin endpoint,
passes its generated DSN only to the workload, then terminates connections,
drops it, and verifies that it no longer exists.

The admin DSN is process-local configuration.  It is deliberately never
returned in observations, artifacts, errors, or release records.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from .release_evidence_execution import CatalogWorkloadUnavailable
from .release_evidence_models import ReleaseEvidenceError


_ISOLATED_ACK_ENV: Final = "KESTREL_SEMANTIC_RELEASE_ISOLATED"
_ADMIN_DSN_ENV: Final = "KESTREL_SEMANTIC_RELEASE_ISOLATED_POSTGRES_ADMIN_DSN"
_DATABASE_PREFIX: Final = "kestrel_semantic_release_"
_DATABASE_NAME_RE: Final = re.compile(r"^kestrel_semantic_release_[0-9a-f]{32}$")
def _quoted_identifier(value: str) -> str:
    if not _DATABASE_NAME_RE.fullmatch(value):
        raise ReleaseEvidenceError("release evidence database name is not a generated disposable name")
    return f'"{value}"'


def _target_dsn(admin_dsn: str, database_name: str) -> str:
    """Replace only the database component of a PostgreSQL DSN."""
    _quoted_identifier(database_name)
    parsed = urlsplit(admin_dsn)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.netloc:
        raise CatalogWorkloadUnavailable("isolated_postgres_admin_dsn_invalid")
    current_name = parsed.path.lstrip("/")
    if not current_name or "/" in current_name:
        raise CatalogWorkloadUnavailable("isolated_postgres_admin_dsn_invalid")
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, ""))


@dataclass(slots=True)
class DisposablePostgresDatabase:
    """A created-and-verified disposable database, with no persisted DSN."""

    dsn: str
    database_name: str
    _admin_dsn: str
    _connection: object
    _closed: bool = False

    @classmethod
    async def create(cls) -> "DisposablePostgresDatabase":
        if os.environ.get(_ISOLATED_ACK_ENV) != "1":
            raise CatalogWorkloadUnavailable("isolated_postgres_ack_required")
        admin_dsn = os.environ.get(_ADMIN_DSN_ENV)
        if not admin_dsn:
            raise CatalogWorkloadUnavailable("isolated_postgres_admin_unavailable")
        database_name = f"{_DATABASE_PREFIX}{uuid4().hex}"
        target_dsn = _target_dsn(admin_dsn, database_name)
        try:
            import asyncpg
        except ImportError as error:
            raise CatalogWorkloadUnavailable("isolated_postgres_driver_unavailable") from error
        connection = None
        created = False
        try:
            connection = await asyncpg.connect(admin_dsn)
            actual_admin_database = await connection.fetchval("SELECT current_database()")
            if not isinstance(actual_admin_database, str) or not actual_admin_database:
                raise CatalogWorkloadUnavailable("isolated_postgres_admin_database_invalid")
            await connection.execute(f"CREATE DATABASE {_quoted_identifier(database_name)}")
            created = True
        except CatalogWorkloadUnavailable:
            raise
        except Exception as error:
            raise CatalogWorkloadUnavailable("isolated_postgres_database_create_failed") from error
        finally:
            if connection is not None and not created:
                try:
                    await connection.close()
                except Exception:
                    # The original failure is already content-free and more
                    # useful than a shutdown failure from the admin channel.
                    pass
        assert connection is not None
        return cls(target_dsn, database_name, admin_dsn, connection)

    async def close(self) -> None:
        """Terminate any workload sessions, drop the DB, and verify its removal."""
        if self._closed:
            return
        self._closed = True
        connection = self._connection
        try:
            await connection.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()",
                self.database_name,
            )
            await connection.execute(f"DROP DATABASE IF EXISTS {_quoted_identifier(self.database_name)}")
            exists = await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)",
                self.database_name,
            )
            if exists is not False:
                raise ReleaseEvidenceError("disposable postgres database removal could not be verified")
        finally:
            await connection.close()

    async def __aenter__(self) -> "DisposablePostgresDatabase":
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> bool:
        await self.close()
        return False


__all__ = [
    "DisposablePostgresDatabase",
    "_DATABASE_PREFIX",
    "_target_dsn",
]
