"""Database explorer endpoints."""
from fastapi import APIRouter, HTTPException, Query, Request
from pathlib import Path
import logging

from kestrel_sovereign.sql_utils import safe_table_name, safe_column_name
from endpoints.agent_helpers import get_agent

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


@router.get("/tables")
async def list_database_tables(request: Request):
    """List SQLite tables with row counts and schema info."""
    try:
        agent = get_agent(request)
        storage = agent.storage

        # Use async database query
        table_rows = await storage.db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        all_tables = [row[0] for row in table_rows] if table_rows else []

        tables = []
        for table_name in all_tables:
            if table_name.startswith('sqlite_'):
                continue

            try:
                safe_name = safe_table_name(table_name)
                count_row = await storage.db.fetchone(f"SELECT COUNT(*) FROM {safe_name}")
                row_count = count_row[0] if count_row else 0
            except ValueError:
                logger.warning(f"Skipping table with invalid name: {table_name!r}")
                continue
            except Exception as e:
                logger.warning(f"Failed to count rows in table {table_name}: {e}")
                row_count = 0

            try:
                col_rows = await storage.db.fetchall(f"PRAGMA table_info({safe_name})")
                columns = [
                    {"name": col[1], "type": col[2], "nullable": not col[3], "pk": bool(col[5])}
                    for col in col_rows
                ] if col_rows else []
            except Exception as e:
                logger.warning(f"Failed to get columns for table {table_name}: {e}")
                columns = []

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
    """Read-only query of a specific table with pagination."""
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(
            status_code=403,
            detail=f"Table '{table_name}' is not queryable. Allowed: {', '.join(ALLOWED_TABLES)}"
        )

    try:
        agent = get_agent(request)
        storage = agent.storage

        # Validate table name for safe SQL interpolation (defense-in-depth;
        # ALLOWED_TABLES check above is the primary gate)
        safe_name = safe_table_name(table_name)

        # Get column info using async query
        col_rows = await storage.db.fetchall(f"PRAGMA table_info({safe_name})")
        columns = [col[1] for col in col_rows] if col_rows else []

        if search and len(search) >= 2:
            search_conditions = []
            for col in columns:
                search_conditions.append(f"CAST({safe_column_name(col)} AS TEXT) LIKE ?")
            where_clause = f"WHERE {' OR '.join(search_conditions)}"
            search_params = [f"%{search}%"] * len(columns)

            count_row = await storage.db.fetchone(
                f"SELECT COUNT(*) FROM {safe_name} {where_clause}",
                search_params
            )
            total_rows = count_row[0] if count_row else 0

            rows = await storage.db.fetchall(
                f"SELECT * FROM {safe_name} {where_clause} LIMIT ? OFFSET ?",
                search_params + [limit, offset]
            )
        else:
            count_row = await storage.db.fetchone(f"SELECT COUNT(*) FROM {safe_name}")
            total_rows = count_row[0] if count_row else 0

            rows = await storage.db.fetchall(
                f"SELECT * FROM {safe_name} LIMIT ? OFFSET ?",
                (limit, offset)
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
