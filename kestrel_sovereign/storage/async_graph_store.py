"""
Async Graph Store for Kestrel Storage.

Provides async knowledge graph storage with nodes and edges.
"""
import json
import logging
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass

from .async_database import AsyncDatabase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Backend-specific JSON property indexes on graph_nodes.
#
# These accelerate property-level queries (agent_id, status, created_at)
# without requiring a separate table.  Each backend needs its own syntax
# for extracting values from the JSON ``properties`` column.
#
# **PostgreSQL parity**: all three property indexes (agent_id, status,
# created_at) are present for *both* backends.  The Postgres indexes use
# expression indexes on ``(properties::jsonb ->> 'key')``.
# ─────────────────────────────────────────────────────────────────────────────

_SQLITE_JSON_INDEXES: List[str] = [
    # agent isolation — every query_nodes_by_type_and_property call filters
    # on agent_id first.
    """CREATE INDEX IF NOT EXISTS idx_graph_nodes_agent
       ON graph_nodes(node_type, json_extract(properties, '$.agent_id'))""",
    # action_item status filter (partial — only action_item rows)
    """CREATE INDEX IF NOT EXISTS idx_graph_nodes_action_status
       ON graph_nodes(json_extract(properties, '$.status'))
       WHERE node_type = 'action_item'""",
    # created_at range / ORDER BY (partial — only action_item rows)
    """CREATE INDEX IF NOT EXISTS idx_graph_nodes_action_created
       ON graph_nodes(json_extract(properties, '$.created_at'))
       WHERE node_type = 'action_item'""",
]

_POSTGRES_JSON_INDEXES: List[str] = [
    # agent isolation
    """CREATE INDEX IF NOT EXISTS idx_graph_nodes_agent
       ON graph_nodes(node_type, ((properties::jsonb)->>'agent_id'))""",
    # action_item status filter (partial)
    """CREATE INDEX IF NOT EXISTS idx_graph_nodes_action_status
       ON graph_nodes(((properties::jsonb)->>'status'))
       WHERE node_type = 'action_item'""",
    # created_at range / ORDER BY (partial)
    """CREATE INDEX IF NOT EXISTS idx_graph_nodes_action_created
       ON graph_nodes(((properties::jsonb)->>'created_at'))
       WHERE node_type = 'action_item'""",
    # GIN index for ad-hoc JSONB containment queries
    """CREATE INDEX IF NOT EXISTS idx_graph_nodes_properties_gin
       ON graph_nodes USING gin ((properties::jsonb))""",
]


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    node_id: str
    node_type: str
    label: str
    properties: Dict[str, Any]


@dataclass
class Edge:
    """Represents an edge between nodes."""
    source_id: str
    target_id: str
    label: str
    properties: Optional[Dict[str, Any]] = None


class AsyncGraphStore:
    """Async knowledge graph storage."""

    def __init__(self, db: AsyncDatabase):
        self.db = db

    # ─────────────────────────────────────────────────────────────────
    # Backend-agnostic helpers
    # ─────────────────────────────────────────────────────────────────

    def _json_extract(self, column: str, path: str) -> str:
        """Return backend-appropriate SQL for extracting a JSON property.

        SQLite:     ``json_extract(column, '$.path')``
        PostgreSQL: ``(column::jsonb)->>'path'``
        """
        if self.db.backend_type == "postgres":
            return f"({column}::jsonb)->>'{path}'"
        return f"json_extract({column}, '$.{path}')"

    async def ensure_property_indexes(self) -> None:
        """Create backend-specific JSON property indexes on graph_nodes.

        Safe to call multiple times — every statement uses
        ``CREATE INDEX IF NOT EXISTS``.
        """
        stmts = (
            _POSTGRES_JSON_INDEXES
            if self.db.backend_type == "postgres"
            else _SQLITE_JSON_INDEXES
        )
        for stmt in stmts:
            try:
                await self.db.execute(stmt)
            except Exception:
                logger.debug("Index creation skipped (may already exist): %s", stmt[:80])

    # ─────────────────────────────────────────────────────────────────
    # Property-level queries
    # ─────────────────────────────────────────────────────────────────

    async def query_nodes_by_type_and_property(
        self,
        node_type: str,
        *,
        filters: Optional[Dict[str, str]] = None,
        created_since: Optional[str] = None,
        order_by_created: str = "DESC",
        limit: int = 100,
    ) -> List[GraphNode]:
        """Query graph nodes by type with optional JSON property filters.

        Args:
            node_type: Required node_type filter (uses the B-tree index).
            filters: Equality filters on JSON properties, e.g.
                ``{"agent_id": "a1", "status": "pending"}``.
            created_since: ISO-8601 lower bound for ``created_at``
                (lexicographic comparison — see *sortable-timestamp
                invariant* in ``schema_router.py``).
            order_by_created: ``"DESC"`` (default) or ``"ASC"``.
            limit: Max rows to return.

        Returns:
            List of matching GraphNode objects.
        """
        clauses: List[str] = ["node_type = ?"]
        params: List[Any] = [node_type]

        for key, value in (filters or {}).items():
            clauses.append(f"{self._json_extract('properties', key)} = ?")
            params.append(value)

        if created_since:
            clauses.append(f"{self._json_extract('properties', 'created_at')} >= ?")
            params.append(created_since)

        direction = "ASC" if order_by_created.upper() == "ASC" else "DESC"
        sql = (
            "SELECT node_id, node_type, label, properties FROM graph_nodes"
            f" WHERE {' AND '.join(clauses)}"
            f" ORDER BY {self._json_extract('properties', 'created_at')} {direction}"
            f" LIMIT ?"
        )
        params.append(limit)

        rows = await self.db.fetchall(sql, tuple(params))
        return [
            GraphNode(
                node_id=row[0],
                node_type=row[1],
                label=row[2],
                properties=json.loads(row[3]) if row[3] else {},
            )
            for row in rows
        ]

    def _upsert_node_sql(self) -> str:
        """Get upsert SQL for nodes based on database backend."""
        if self.db.backend_type == "postgres":
            return """
                INSERT INTO graph_nodes (node_id, node_type, label, properties)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (node_id) DO UPDATE SET
                    node_type = EXCLUDED.node_type,
                    label = EXCLUDED.label,
                    properties = EXCLUDED.properties
            """
        return "INSERT OR REPLACE INTO graph_nodes (node_id, node_type, label, properties) VALUES (?, ?, ?, ?)"

    def _upsert_edge_sql(self) -> str:
        """Get upsert SQL for edges based on database backend."""
        if self.db.backend_type == "postgres":
            return """
                INSERT INTO graph_edges (source_id, target_id, label, properties)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (source_id, target_id, label) DO UPDATE SET
                    properties = EXCLUDED.properties
            """
        return "INSERT OR REPLACE INTO graph_edges (source_id, target_id, label, properties) VALUES (?, ?, ?, ?)"

    async def add_node(self, node: GraphNode) -> None:
        """Add or update a node."""
        await self.db.execute_commit(
            self._upsert_node_sql(),
            (node.node_id, node.node_type, node.label, json.dumps(node.properties))
        )
    
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        row = await self.db.fetchone(
            "SELECT node_id, node_type, label, properties FROM graph_nodes WHERE node_id = ?",
            (node_id,)
        )
        if not row:
            return None
        return GraphNode(
            node_id=row[0],
            node_type=row[1],
            label=row[2],
            properties=json.loads(row[3]) if row[3] else {}
        )
    
    async def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        rows = await self.db.fetchall(
            "SELECT node_id, node_type, label, properties FROM graph_nodes WHERE node_type = ?",
            (node_type,)
        )
        return [
            GraphNode(
                node_id=row[0],
                node_type=row[1],
                label=row[2],
                properties=json.loads(row[3]) if row[3] else {}
            )
            for row in rows
        ]
    
    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges."""
        async with self.db.transaction():
            await self.db.execute(
                "DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id)
            )
            await self.db.execute(
                "DELETE FROM graph_nodes WHERE node_id = ?",
                (node_id,)
            )
    
    async def add_edge(self, source_id: str, target_id: str, label: str,
                       properties: Optional[Dict] = None) -> None:
        """Add an edge between nodes.

        Upserts by (source_id, target_id, label) — calling add_edge twice
        with the same triple updates the properties, not duplicates the edge.
        """
        await self.db.execute_commit(
            self._upsert_edge_sql(),
            (source_id, target_id, label, json.dumps(properties) if properties else None)
        )

    async def delete_edge(self, source_id: str, target_id: str, label: str) -> None:
        """Remove a specific edge by its (source, target, label) triple."""
        await self.db.execute_commit(
            "DELETE FROM graph_edges WHERE source_id = ? AND target_id = ? AND label = ?",
            (source_id, target_id, label),
        )
    
    async def get_edges(self, node_id: str, direction: str = "both") -> List[Edge]:
        """Get edges connected to a node."""
        edges = []
        
        if direction in ("out", "both"):
            rows = await self.db.fetchall(
                "SELECT source_id, target_id, label, properties FROM graph_edges WHERE source_id = ?",
                (node_id,)
            )
            edges.extend([
                Edge(
                    source_id=row[0],
                    target_id=row[1],
                    label=row[2],
                    properties=json.loads(row[3]) if row[3] else None
                )
                for row in rows
            ])
        
        if direction in ("in", "both"):
            rows = await self.db.fetchall(
                "SELECT source_id, target_id, label, properties FROM graph_edges WHERE target_id = ?",
                (node_id,)
            )
            edges.extend([
                Edge(
                    source_id=row[0],
                    target_id=row[1],
                    label=row[2],
                    properties=json.loads(row[3]) if row[3] else None
                )
                for row in rows
            ])
        
        return edges
