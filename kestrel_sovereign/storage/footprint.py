"""How much space an agent database takes (#3785).

Meridian's database reached 1.89 GB for 708 messages, and nothing an agent
could call reported it: the liveness probe never looked at the file.
:func:`measure_database_size` reads only page counts and the size of the
write-ahead log, so the probe runs it every tick.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_SQLITE_SIZE_SQL = (
    "SELECT page_count * page_size, freelist_count * page_size "
    "FROM pragma_page_count(), pragma_page_size(), pragma_freelist_count()"
)

_POSTGRES_SIZE_SQL = "SELECT pg_database_size(current_database())"


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

    The page counts come through *db*'s own connection. The write-ahead log
    is read by its pathname, so the connected file is checked to still be the
    one at that path afterwards; a replaced file raises rather than pairing
    one database's size with another's log. Returns ``None`` for a database
    adapter that is neither SQLite nor PostgreSQL, which this cannot measure.
    Query failures propagate.
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
        assert_valid = getattr(db.backend, "assert_connected_file_still_valid", None)
        if callable(assert_valid):
            assert_valid()
    return DatabaseSize(
        backend="sqlite",
        total_bytes=int(row[0]),
        free_bytes=int(row[1]),
        wal_bytes=wal_bytes,
    )
