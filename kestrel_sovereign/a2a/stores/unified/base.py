"""
Base class for unified stores.

Provides common functionality for all stores using DatabaseBackend.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)


class UnifiedStoreBase:
    """
    Base class for backend-agnostic stores.

    Provides:
    - Backend reference and type checking
    - SQL dialect helpers (timestamps, JSON, etc.)
    - Common patterns for row conversion
    """

    def __init__(self, backend: DatabaseBackend):
        """
        Initialize store with database backend.

        Args:
            backend: DatabaseBackend instance (SQLite or PostgreSQL)
        """
        self._backend = backend

    @property
    def backend(self) -> DatabaseBackend:
        """Get the database backend."""
        return self._backend

    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL backend."""
        return self._backend.backend_type == "postgres"

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite backend."""
        return self._backend.backend_type == "sqlite"

    # ==========================================================================
    # SQL Dialect Helpers
    # ==========================================================================

    def now_sql(self) -> str:
        """
        Get SQL expression for current timestamp.

        Returns:
            "NOW()" for PostgreSQL, "datetime('now')" for SQLite
        """
        if self.is_postgres:
            return "NOW()"
        return "datetime('now')"

    def now_default(self) -> str:
        """
        Get SQL DEFAULT clause for timestamp columns.

        Returns:
            "DEFAULT NOW()" for PostgreSQL, "DEFAULT (datetime('now'))" for SQLite
        """
        if self.is_postgres:
            return "DEFAULT NOW()"
        return "DEFAULT (datetime('now'))"

    def timestamp_type(self) -> str:
        """
        Get SQL type for timestamp columns.

        Returns:
            "TIMESTAMPTZ" for PostgreSQL, "TEXT" for SQLite
        """
        if self.is_postgres:
            return "TIMESTAMPTZ"
        return "TEXT"

    def json_type(self) -> str:
        """
        Get SQL type for JSON columns.

        Returns:
            "JSONB" for PostgreSQL, "TEXT" for SQLite
        """
        if self.is_postgres:
            return "JSONB"
        return "TEXT"

    def boolean_type(self) -> str:
        """
        Get SQL type for boolean columns.

        Returns:
            "BOOLEAN" for PostgreSQL, "INTEGER" for SQLite
        """
        if self.is_postgres:
            return "BOOLEAN"
        return "INTEGER"

    def interval_days(self, days: int) -> str:
        """
        Get SQL expression for interval subtraction.

        Args:
            days: Number of days to subtract

        Returns:
            SQL expression for (now - days)
        """
        if self.is_postgres:
            return f"NOW() - INTERVAL '{days} days'"
        return f"datetime('now', '-{days} days')"

    # ==========================================================================
    # Value Conversion Helpers
    # ==========================================================================

    def to_bool_param(self, value: bool) -> Any:
        """
        Convert Python bool to database parameter.

        Args:
            value: Boolean value

        Returns:
            value for PostgreSQL, 1/0 for SQLite
        """
        if self.is_postgres:
            return value
        return 1 if value else 0

    def from_bool_field(self, value: Any) -> bool:
        """
        Convert database value to Python bool.

        Args:
            value: Database value (bool or int)

        Returns:
            Boolean value
        """
        return bool(value)

    def to_timestamp_param(self, dt: Optional[datetime]) -> Any:
        """
        Convert Python datetime to database parameter.

        Args:
            dt: Datetime value (or None)

        Returns:
            datetime for PostgreSQL, ISO string for SQLite
        """
        if dt is None:
            return None
        if self.is_postgres:
            return dt
        return dt.isoformat()

    def from_timestamp_field(self, value: Any) -> Optional[datetime]:
        """
        Convert database timestamp to Python datetime.

        Args:
            value: Database value (datetime or string)

        Returns:
            Datetime value (or None)
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        # SQLite stores as ISO string
        return datetime.fromisoformat(value)

    def now_utc(self) -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)

    def now_utc_param(self) -> Any:
        """
        Get current UTC timestamp as database parameter.

        Returns:
            datetime for PostgreSQL, ISO string for SQLite
        """
        return self.to_timestamp_param(self.now_utc())

    # ==========================================================================
    # Lifecycle Methods
    # ==========================================================================

    async def close(self) -> None:
        """
        Close the underlying database backend connection.

        Should be called when the store is no longer needed to prevent
        resource leaks from open database connections.
        """
        if self._backend.is_connected:
            await self._backend.close()
