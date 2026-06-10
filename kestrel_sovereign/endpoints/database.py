"""Database explorer endpoints."""
from fastapi import APIRouter, HTTPException, Query, Request
from pathlib import Path
import logging

from kestrel_sovereign.sql_utils import safe_table_name, safe_column_name
from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/db", tags=["database"])

ALLOWED_TABLES = {
    "conversation_history",
    "graph_nodes",
    "graph_edges",
    "documents",
    "document_chunks",
    "fts_documents",
}


def _agent_scope_condition(column_names, agent_id):
    """Return ``(condition, params)`` scoping a query to one agent, else ``(None, [])``.

    Tables that carry an ``agent_id`` column (e.g. ``conversation_history``)
    are shared across agents in a single database, so the explorer must never
    surface another agent's rows. Tables without the column get no scope.
    """
    if agent_id is not None and "agent_id" in set(column_names):
        return "agent_id = ?", [agent_id]
    return None, []


async def _list_table_names(db):
    """Return table names for the active backend."""
    if db.backend_type == "postgres":
        rows = await db.fetchall(
            """SELECT table_name
               FROM information_schema.tables
               WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
               ORDER BY table_name"""
        )
    else:
        rows = await db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    return [row[0] for row in rows] if rows else []


async def _get_table_columns(db, table_name: str):
    """Return normalized column metadata for the active backend."""
    safe_name = safe_table_name(table_name)
    if db.backend_type == "postgres":
        rows = await db.fetchall(
            """
            SELECT
                c.column_name,
                c.data_type,
                c.is_nullable,
                EXISTS (
                    SELECT 1
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema = c.table_schema
                      AND tc.table_name = c.table_name
                      AND tc.constraint_type = 'PRIMARY KEY'
                      AND kcu.column_name = c.column_name
                ) AS is_primary_key
            FROM information_schema.columns c
            WHERE c.table_schema = 'public' AND c.table_name = ?
            ORDER BY c.ordinal_position
            """,
            (safe_name,),
        )
        return [
            {"name": col[0], "type": col[1], "nullable": col[2] == "YES", "pk": bool(col[3])}
            for col in (rows or [])
        ]

    rows = await db.fetchall(f"PRAGMA table_info({safe_name})")
    return [
        {"name": col[1], "type": col[2], "nullable": not col[3], "pk": bool(col[5])}
        for col in (rows or [])
    ]


@router.get("/tables")
async def list_database_tables(request: Request):
    """List SQLite tables with row counts and schema info."""
    try:
        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(agent, "agent_id", None)

        # Use async database query
        all_tables = await _list_table_names(storage.db)

        tables = []
        for table_name in all_tables:
            if table_name.startswith('sqlite_'):
                continue

            try:
                safe_name = safe_table_name(table_name)
            except ValueError:
                logger.warning(f"Skipping table with invalid name: {table_name!r}")
                continue

            try:
                columns = await _get_table_columns(storage.db, table_name)
            except Exception as e:
                logger.warning(f"Failed to get columns for table {table_name}: {e}")
                columns = []

            # Scope the row count to this agent when the table carries an
            # agent_id column, so a shared multi-agent DB doesn't report
            # another agent's row totals through the explorer (#1651).
            scope_cond, scope_params = _agent_scope_condition(
                [c["name"] for c in columns], agent_id
            )
            where_clause = f"WHERE {scope_cond}" if scope_cond else ""
            try:
                count_row = await storage.db.fetchone(
                    f"SELECT COUNT(*) FROM {safe_name} {where_clause}".strip(),
                    scope_params,
                )
                row_count = count_row[0] if count_row else 0
            except Exception as e:
                logger.warning(f"Failed to count rows in table {table_name}: {e}")
                row_count = 0

            tables.append({
                "name": table_name,
                "row_count": row_count,
                "columns": columns,
                "queryable": table_name in ALLOWED_TABLES,
            })

        # Get db_path from storage (not storage.db which is AsyncDatabase)
        db_path = getattr(storage, 'db_path', None)
        if db_path is None:
            # Try to get from wrapped storage (PrivacyEnforcingStorage wraps AsyncStorage)
            inner_storage = getattr(storage, '_storage', None)
            if inner_storage:
                db_path = getattr(inner_storage, 'db_path', None)

        db_size = 0
        if db_path and Path(db_path).exists():
            db_size = Path(db_path).stat().st_size

        return {
            "tables": tables,
            "table_count": len(tables),
            "db_size": db_size,
            "db_path": str(db_path) if db_path else "unknown",
        }
    except Exception as e:
        logger.error(f"Error listing database tables: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing tables.")


@router.get("/tables/{table_name}")
async def query_database_table(
    request: Request,
    table_name: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: str = None
):
    """Read-only query of a specific table with pagination.

    Scoped to the requesting agent: tables with an ``agent_id`` column only
    return that agent's rows, and for those tables EPHEMERAL/ISOLATED privacy
    modes return nothing (the persisted rows aren't part of the agent's
    visible state). See #1651.
    """
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(
            status_code=403,
            detail=f"Table '{table_name}' is not queryable. Allowed: {', '.join(ALLOWED_TABLES)}"
        )

    try:
        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(agent, "agent_id", None)

        # Validate table name for safe SQL interpolation (defense-in-depth;
        # ALLOWED_TABLES check above is the primary gate)
        safe_name = safe_table_name(table_name)

        # Get column info using backend-aware introspection
        column_info = await _get_table_columns(storage.db, table_name)
        columns = [col["name"] for col in column_info]

        scope_cond, scope_params = _agent_scope_condition(columns, agent_id)

        # Privacy gate: the explorer reads the raw persistent DB directly,
        # bypassing the privacy wrapper. For agent-scoped tables, EPHEMERAL
        # and ISOLATED modes promise the persisted rows are not part of the
        # agent's visible state, so don't surface them here either (#1651).
        if scope_cond is not None:
            pconf = getattr(storage, "privacy_config", None)
            if pconf is not None and (pconf.is_ephemeral() or pconf.uses_temp_storage()):
                return {
                    "table": table_name,
                    "columns": columns,
                    "rows": [],
                    "total_rows": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "note": "Hidden in EPHEMERAL/ISOLATED privacy mode.",
                }

        # Compose the WHERE clause: agent scope (when applicable) AND the
        # optional free-text search, sharing one ordered params list.
        clauses = []
        params = []
        if scope_cond is not None:
            clauses.append(scope_cond)
            params.extend(scope_params)
        if search and len(search) >= 2:
            search_conditions = [
                f"CAST({safe_column_name(col)} AS TEXT) LIKE ?" for col in columns
            ]
            clauses.append("(" + " OR ".join(search_conditions) + ")")
            params.extend([f"%{search}%"] * len(columns))

        where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        count_row = await storage.db.fetchone(
            f"SELECT COUNT(*) FROM {safe_name} {where_clause}".strip(),
            params,
        )
        total_rows = count_row[0] if count_row else 0

        rows = await storage.db.fetchall(
            f"SELECT * FROM {safe_name} {where_clause} LIMIT ? OFFSET ?".strip(),
            params + [limit, offset],
        )

        rows = rows or []

        data = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                if isinstance(value, str) and len(value) > 500:
                    value = value[:500] + "..."
                row_dict[col] = value
            data.append(row_dict)

        return {
            "table": table_name,
            "columns": columns,
            "rows": data,
            "total_rows": total_rows,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_rows,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying table {table_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error querying table.")
