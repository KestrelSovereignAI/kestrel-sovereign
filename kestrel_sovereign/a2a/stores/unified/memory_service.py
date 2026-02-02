"""
Unified MemoryService - Backend-Agnostic Long-term Memory.

Provides persistent, searchable long-term memory for agents:
- Full-text search via FTS5 (SQLite) or GIN (PostgreSQL)
- Tag-based filtering
- Session history linkage

Works with both SQLite and PostgreSQL backends.
"""

import logging
from typing import Any, Optional

from pydantic import BaseModel

from kestrel_sovereign.a2a.stores.base import generate_id, json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """A searchable memory entry."""

    memory_id: str
    session_id: Optional[str]
    content: str  # Searchable text
    tags: list[str]
    created_at: Any  # datetime
    metadata: dict[str, Any]


class MemoryService(UnifiedStoreBase):
    """
    Backend-agnostic memory service.

    Replaces both SQLiteMemoryService and PostgresMemoryService with a single
    implementation that works with any DatabaseBackend.

    Note: Full-text search is implemented differently per backend:
    - SQLite: Uses FTS5 virtual table
    - PostgreSQL: Uses tsvector with GIN index
    """

    def __init__(self, backend: DatabaseBackend):
        """
        Initialize memory service with database backend.

        Args:
            backend: DatabaseBackend instance (SQLite or PostgreSQL)
        """
        super().__init__(backend)

    async def initialize(self) -> None:
        """Create tables if not exists."""
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()

        if self.is_postgres:
            await self._initialize_postgres(ts_type, ts_default, json_type)
        else:
            await self._initialize_sqlite(ts_type, ts_default, json_type)

        logger.info(f"MemoryService initialized ({self._backend.backend_type})")

    async def _initialize_postgres(self, ts_type: str, ts_default: str, json_type: str) -> None:
        """Initialize PostgreSQL-specific schema with tsvector."""
        # Enable pg_trgm extension for fuzzy search
        await self._backend.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_memory (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                user_id TEXT,
                content TEXT,
                content_tsv TSVECTOR,
                tags {json_type} DEFAULT '[]',
                metadata {json_type} DEFAULT '{{}}',
                created_at {ts_type} {ts_default}
            )
        """)

        # GIN index for full-text search
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_fts ON a2a_memory USING GIN(content_tsv)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_session ON a2a_memory(session_id)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_user ON a2a_memory(user_id)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_tags ON a2a_memory USING GIN(tags)"
        )

        # Create trigger to auto-update tsvector
        await self._backend.execute("""
            CREATE OR REPLACE FUNCTION a2a_memory_tsv_trigger() RETURNS trigger AS $$
            BEGIN
                NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)

        # Check if trigger exists before creating
        trigger_exists = await self._backend.fetch_val("""
            SELECT EXISTS (
                SELECT 1 FROM pg_trigger
                WHERE tgname = 'a2a_memory_tsv_update'
            )
        """)

        if not trigger_exists:
            await self._backend.execute("""
                CREATE TRIGGER a2a_memory_tsv_update
                BEFORE INSERT OR UPDATE ON a2a_memory
                FOR EACH ROW EXECUTE FUNCTION a2a_memory_tsv_trigger()
            """)

    async def _initialize_sqlite(self, ts_type: str, ts_default: str, json_type: str) -> None:
        """Initialize SQLite-specific schema with FTS5."""
        # Main memory table
        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_memory (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                content TEXT,
                tags {json_type} DEFAULT '[]',
                metadata {json_type} DEFAULT '{{}}',
                created_at {ts_type} {ts_default}
            )
        """)

        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_session ON a2a_memory(session_id)"
        )

        # FTS5 virtual table for full-text search
        # Note: SQLite requires separate executescript for virtual table
        try:
            await self._backend.execute_script("""
                CREATE VIRTUAL TABLE IF NOT EXISTS a2a_memory_fts
                USING fts5(
                    content,
                    content='a2a_memory',
                    content_rowid='rowid'
                )
            """)

            # Triggers to keep FTS in sync
            await self._backend.execute_script("""
                CREATE TRIGGER IF NOT EXISTS a2a_memory_ai AFTER INSERT ON a2a_memory BEGIN
                    INSERT INTO a2a_memory_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)
            await self._backend.execute_script("""
                CREATE TRIGGER IF NOT EXISTS a2a_memory_ad AFTER DELETE ON a2a_memory BEGIN
                    INSERT INTO a2a_memory_fts(a2a_memory_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                END
            """)
            await self._backend.execute_script("""
                CREATE TRIGGER IF NOT EXISTS a2a_memory_au AFTER UPDATE ON a2a_memory BEGIN
                    INSERT INTO a2a_memory_fts(a2a_memory_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                    INSERT INTO a2a_memory_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)
        except Exception as e:
            logger.warning(f"FTS5 setup failed (may already exist): {e}")

    async def add_memory(
        self,
        session_id: str,
        content: str,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a memory entry. Returns memory_id."""
        memory_id = generate_id()
        user_id = metadata.get("user_id") if metadata else None
        now = self.now_utc_param()

        if self.is_postgres:
            await self._backend.execute(
                """
                INSERT INTO a2a_memory
                (id, session_id, user_id, content, tags, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    session_id,
                    user_id,
                    content,
                    json_dumps(tags or []),
                    json_dumps(metadata or {}),
                    now,
                ),
            )
        else:
            await self._backend.execute(
                """
                INSERT INTO a2a_memory
                (id, session_id, content, tags, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    session_id,
                    content,
                    json_dumps(tags or []),
                    json_dumps(metadata or {}),
                    now,
                ),
            )

        return memory_id

    async def add_session_to_memory(
        self,
        session_id: str,
        content: str,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Convert session history to searchable memory. Returns memory_id."""
        return await self.add_memory(
            session_id=session_id,
            content=content,
            tags=tags,
            metadata=metadata,
        )

    async def search_memory(
        self,
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search memory by keyword (FTS) and/or tags."""
        if self.is_postgres:
            return await self._search_postgres(query, tags, session_id, limit)
        else:
            return await self._search_sqlite(query, tags, session_id, limit)

    async def _search_postgres(
        self,
        query: Optional[str],
        tags: Optional[list[str]],
        session_id: Optional[str],
        limit: int,
    ) -> list[MemoryEntry]:
        """PostgreSQL full-text search using tsvector."""
        conditions = []
        params: list[Any] = []

        if query:
            conditions.append("content_tsv @@ plainto_tsquery('english', ?)")
            params.append(query)

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        if tags:
            conditions.append("tags @> ?::jsonb")
            params.append(json_dumps(tags))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        # Order by rank if FTS query, otherwise by date
        order_by = (
            "ts_rank(content_tsv, plainto_tsquery('english', ?)) DESC"
            if query
            else "created_at DESC"
        )
        if query:
            params.insert(-1, query)  # Add query param for ORDER BY

        rows = await self._backend.fetch_all(
            f"""
            SELECT id, session_id, content, tags, metadata, created_at
            FROM a2a_memory
            {where}
            ORDER BY {order_by}
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_memory(row) for row in rows]

    async def _search_sqlite(
        self,
        query: Optional[str],
        tags: Optional[list[str]],
        session_id: Optional[str],
        limit: int,
    ) -> list[MemoryEntry]:
        """SQLite full-text search using FTS5."""
        if query:
            # Use FTS5 MATCH for full-text search
            conditions = ["a2a_memory_fts MATCH ?"]
            params: list[Any] = [query]

            if session_id:
                conditions.append("m.session_id = ?")
                params.append(session_id)

            params.append(limit)

            rows = await self._backend.fetch_all(
                f"""
                SELECT m.id, m.session_id, m.content, m.tags, m.metadata, m.created_at
                FROM a2a_memory m
                JOIN a2a_memory_fts fts ON m.rowid = fts.rowid
                WHERE {' AND '.join(conditions)}
                ORDER BY rank
                LIMIT ?
                """,
                tuple(params),
            )
        else:
            # No FTS query, just filter by session
            conditions = []
            params = []

            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)

            rows = await self._backend.fetch_all(
                f"""
                SELECT id, session_id, content, tags, metadata, created_at
                FROM a2a_memory
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                tuple(params),
            )

        results = [self._row_to_memory(row) for row in rows]

        # Filter by tags in Python (SQLite JSON support is limited)
        if tags:
            results = [m for m in results if any(tag in m.tags for tag in tags)]

        return results

    async def get_session_history(self, session_id: str) -> list[MemoryEntry]:
        """Get all memory entries for a specific session."""
        rows = await self._backend.fetch_all(
            """
            SELECT id, session_id, content, tags, metadata, created_at
            FROM a2a_memory
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        return [self._row_to_memory(row) for row in rows]

    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        row = await self._backend.fetch_one(
            "SELECT id, session_id, content, tags, metadata, created_at FROM a2a_memory WHERE id = ?",
            (memory_id,),
        )
        if not row:
            return None
        return self._row_to_memory(row)

    async def list_memories(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List memories with optional session filter."""
        if session_id:
            rows = await self._backend.fetch_all(
                """
                SELECT id, session_id, content, tags, metadata, created_at
                FROM a2a_memory
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
        else:
            rows = await self._backend.fetch_all(
                """
                SELECT id, session_id, content, tags, metadata, created_at
                FROM a2a_memory
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [self._row_to_memory(row) for row in rows]

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if deleted."""
        rows_affected = await self._backend.execute(
            "DELETE FROM a2a_memory WHERE id = ?",
            (memory_id,),
        )
        return rows_affected > 0

    def _row_to_memory(self, row: tuple) -> MemoryEntry:
        """
        Convert database row to MemoryEntry object.

        Row columns (in order):
        0: id, 1: session_id, 2: content, 3: tags, 4: metadata, 5: created_at
        """
        return MemoryEntry(
            memory_id=row[0],
            session_id=row[1],
            content=row[2] or "",
            tags=json_loads(row[3]) if row[3] else [],
            created_at=self.from_timestamp_field(row[5]),
            metadata=json_loads(row[4]) if row[4] else {},
        )
