"""How much space an agent database takes, and which tables take it (#3785).

Meridian's database reached 1.89 GB for 708 messages, and nothing an agent
could call said where the bytes were: ``memory_status`` counted messages and
the liveness probe never looked at the file. :func:`measure_database_size` is
cheap enough for the probe to run every tick; :func:`measure_largest_tables`
reads the whole file and runs only on request.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from kestrel_sovereign._async_ownership import run_blocking_operation
from kestrel_sovereign.storage.db.interface import (
    ConnectionError as DatabaseConnectionError,
)

logger = logging.getLogger(__name__)

#: How many tables :func:`measure_largest_tables` names by default.
DEFAULT_LARGEST_TABLES = 10

#: Wall-clock budget for walking a SQLite file with ``dbstat``, which reads
#: every page of the database: seconds for a 2 GB file on a local disk.
TABLE_SCAN_BUDGET_S = 30.0

#: SQLite VM instructions between deadline checks during the walk.
_PROGRESS_INTERVAL = 100_000

_SQLITE_SIZE_SQL = (
    "SELECT page_count * page_size, freelist_count * page_size "
    "FROM pragma_page_count(), pragma_page_size(), pragma_freelist_count()"
)

# One row per table: its own b-tree plus its indexes. ``sqlite_schema`` has
# no ``sqlite_master`` row and is reported under its own name.
_SQLITE_TABLES_SQL = (
    "SELECT COALESCE(owner.tbl_name, page.name) AS table_name, "
    "SUM(page.pgsize) AS bytes "
    "FROM dbstat AS page "
    "LEFT JOIN sqlite_master AS owner ON owner.name = page.name "
    "WHERE page.aggregate = TRUE "
    "GROUP BY table_name ORDER BY bytes DESC LIMIT ?"
)

_POSTGRES_SIZE_SQL = "SELECT pg_database_size(current_database())"

# ``pg_total_relation_size`` counts a table with its indexes and TOAST.
_POSTGRES_TABLES_SQL = (
    "SELECT c.relname, pg_total_relation_size(c.oid) AS bytes "
    "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
    "WHERE c.relkind IN ('r', 'm') AND n.nspname = current_schema() "
    "ORDER BY bytes DESC LIMIT ?"
)


def format_bytes(count: int) -> str:
    """*count* in decimal units: ``1.89 GB``, ``33.0 MB``, ``512 B``."""
    if count < 1000:
        return f"{count} B"
    value = float(count)
    for unit in ("kB", "MB", "GB", "TB"):
        value /= 1000
        if value < 1000 or unit == "TB":
            return f"{value:.2f} {unit}" if value < 10 else f"{value:.1f} {unit}"
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class DatabaseSize:
    """The size of one database, as its backend reports it.

    ``free_bytes`` is SQLite free pages: space the file holds but no row
    uses, which only ``VACUUM`` returns to the filesystem. ``wal_bytes`` is
    the SQLite write-ahead log beside the file. A PostgreSQL database is
    ``shared`` by every agent the host keeps in it, so its size is theirs
    together.
    """

    backend: str
    total_bytes: int
    free_bytes: Optional[int] = None
    wal_bytes: Optional[int] = None
    shared: bool = False

    def describe(self) -> str:
        parts = []
        if self.free_bytes is not None:
            parts.append(f"{format_bytes(self.free_bytes)} free pages")
        if self.wal_bytes is not None:
            parts.append(f"WAL {format_bytes(self.wal_bytes)}")
        if self.shared:
            parts.append("shared by every agent on this database")
        detail = f" ({', '.join(parts)})" if parts else ""
        return f"{format_bytes(self.total_bytes)}{detail}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "wal_bytes": self.wal_bytes,
            "shared": self.shared,
            "summary": self.describe(),
        }


@dataclass(frozen=True)
class TableSize:
    """One table with its indexes (and TOAST, on PostgreSQL)."""

    name: str
    bytes: int


@dataclass(frozen=True)
class LargestTables:
    """The largest tables, or why they could not be measured."""

    tables: Tuple[TableSize, ...] = field(default_factory=tuple)
    unavailable: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.unavailable is not None:
            return {"unavailable": self.unavailable}
        return {
            "tables": [
                {
                    "name": table.name,
                    "bytes": table.bytes,
                    "size": format_bytes(table.bytes),
                }
                for table in self.tables
            ]
        }


def _backend_type(db: Any) -> Optional[str]:
    backend_type = getattr(db, "backend_type", None)
    return backend_type if backend_type in ("sqlite", "postgres") else None


def _sqlite_file(db: Any) -> Optional[Path]:
    """The file a SQLite database lives in; ``None`` for an in-memory one."""
    path = getattr(getattr(db, "backend", None), "db_path", None)
    if not isinstance(path, str) or path == ":memory:":
        return None
    return Path(path)


async def measure_database_size(db: Any) -> Optional[DatabaseSize]:
    """The size of *db*: header reads only, cheap enough for every probe.

    Returns ``None`` for a database adapter that is neither SQLite nor
    PostgreSQL, which this cannot measure. Query failures propagate.
    """
    backend_type = _backend_type(db)
    if backend_type == "postgres":
        row = await db.fetchone(_POSTGRES_SIZE_SQL)
        return DatabaseSize(backend="postgres", total_bytes=int(row[0]), shared=True)
    if backend_type != "sqlite":
        return None
    row = await db.fetchone(_SQLITE_SIZE_SQL)
    wal_bytes = None
    path = _sqlite_file(db)
    if path is not None:
        try:
            wal_bytes = os.stat(f"{path}-wal").st_size
        except FileNotFoundError:
            wal_bytes = 0
    return DatabaseSize(
        backend="sqlite",
        total_bytes=int(row[0]),
        free_bytes=int(row[1]),
        wal_bytes=wal_bytes,
    )


async def measure_largest_tables(
    db: Any, *, limit: int = DEFAULT_LARGEST_TABLES,
) -> LargestTables:
    """The *limit* largest tables of *db*, each with its indexes.

    On SQLite this walks every page with ``dbstat`` on a separate read-only
    connection, so the agent's own connection stays free for the seconds the
    walk takes; it gives up after :data:`TABLE_SCAN_BUDGET_S`. A SQLite build
    without ``dbstat``, an in-memory database, or a walk that runs out of
    time is reported as ``unavailable`` with the reason.
    """
    backend_type = _backend_type(db)
    if backend_type == "postgres":
        rows = await db.fetchall(_POSTGRES_TABLES_SQL, (limit,))
        return LargestTables(
            tables=tuple(TableSize(str(row[0]), int(row[1])) for row in rows)
        )
    if backend_type != "sqlite":
        return LargestTables(unavailable="unsupported database adapter")
    path = _sqlite_file(db)
    if path is None:
        return LargestTables(unavailable="in-memory database")
    # Refuse to report on a file that is no longer the one the agent holds.
    assert_valid = getattr(db.backend, "assert_connected_file_still_valid", None)
    if callable(assert_valid):
        try:
            assert_valid()
        except DatabaseConnectionError as exc:
            return LargestTables(unavailable=str(exc))
    try:
        rows = await run_blocking_operation(_scan_sqlite_tables, path, limit)
    except sqlite3.Error as exc:
        logger.warning("Table size scan of %s failed: %s", path, exc)
        return LargestTables(unavailable=f"table size scan failed: {exc}")
    return LargestTables(
        tables=tuple(TableSize(str(name), int(size)) for name, size in rows)
    )


def _scan_sqlite_tables(path: Path, limit: int) -> list:
    deadline = time.monotonic() + TABLE_SCAN_BUDGET_S

    def past_deadline() -> int:
        return 1 if time.monotonic() > deadline else 0

    with closing(
        sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    ) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.set_progress_handler(past_deadline, _PROGRESS_INTERVAL)
        try:
            return connection.execute(_SQLITE_TABLES_SQL, (limit,)).fetchall()
        except sqlite3.OperationalError as exc:
            if time.monotonic() > deadline:
                raise sqlite3.OperationalError(
                    f"gave up after {TABLE_SCAN_BUDGET_S:g} seconds"
                ) from exc
            raise
