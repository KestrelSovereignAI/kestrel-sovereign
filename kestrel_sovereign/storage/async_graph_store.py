"""
Async Graph Store for Kestrel Storage.

Provides async knowledge graph storage with nodes and edges.

Sortable-timestamp invariant
----------------------------
The ``query_nodes_by_type_and_property`` method supports ``created_since``
range filtering and ``ORDER BY created_at DESC`` sorting.  Both rely on
lexicographic SQL comparison, which produces correct results **only** when
every ``created_at`` value stored in ``graph_nodes.properties`` is a
UTC ISO-8601 string whose text sort order matches chronological order —
i.e. ``YYYY-MM-DDTHH:MM:SS+00:00``.

All code paths that persist ``created_at`` MUST use
``datetime.now(timezone.utc).isoformat()`` (or an equivalent that
produces a fixed-offset ``+00:00`` suffix, never a bare naive string).
"""
import json
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from .async_database import AsyncDatabase

logger = logging.getLogger(__name__)


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

    # -----------------------------------------------------------------
    # Property-level query helpers (use JSON-path indexes)
    # -----------------------------------------------------------------

    def _json_extract(self, column: str, path: str) -> str:
        """Return backend-appropriate JSON extraction SQL.

        SQLite:     json_extract(column, '$.path')
        PostgreSQL: (column::jsonb)->>'path'
        """
        if self.db.backend_type == "postgres":
            return f"({column}::jsonb)->>'{path}'"
        return f"json_extract({column}, '$.{path}')"

    async def query_nodes_by_type_and_property(
        self,
        node_type: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        created_since: Optional[str] = None,
        order_by_created: bool = True,
        limit: int = 200,
    ) -> List[GraphNode]:
        """Query graph nodes by type with property-level SQL filters.

        Pushes equality and range filters into SQL so the database can
        use the JSON-path partial indexes (``idx_graph_nodes_agent``,
        ``idx_graph_nodes_action_status``, ``idx_graph_nodes_action_created``).

        Args:
            node_type: Required ``node_type`` value (e.g. ``"action_item"``).
            filters: Dict of ``{property_name: value}`` for equality checks
                pushed into ``WHERE json_extract(properties, '$.key') = ?``.
            created_since: ISO-8601 timestamp lower bound on
                ``properties->>'created_at'``.  Uses ``>=`` comparison.
            order_by_created: If True (default), results are ordered by
                ``properties->>'created_at' DESC``.
            limit: Maximum rows returned (clamped to 1-10000).

        Returns:
            List of matching :class:`GraphNode` instances.
        """
        limit = max(1, min(limit, 10000))
        clauses: List[str] = ["node_type = ?"]
        params: List[Any] = [node_type]

        for key, value in (filters or {}).items():
            clauses.append(f"{self._json_extract('properties', key)} = ?")
            params.append(value)

        if created_since is not None:
            clauses.append(f"{self._json_extract('properties', 'created_at')} >= ?")
            params.append(created_since)

        where = " AND ".join(clauses)
        order = ""
        if order_by_created:
            order = f" ORDER BY {self._json_extract('properties', 'created_at')} DESC"

        sql = (
            f"SELECT node_id, node_type, label, properties "
            f"FROM graph_nodes WHERE {where}{order} LIMIT ?"
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

    async def purge_agent_nodes(self, agent_id: str) -> int:
        """Hard-delete every graph_node tagged with this ``agent_id`` (#767).

        EPHEMERAL agents are not supposed to write to ``graph_nodes`` —
        the privacy wrapper rejects persistent writes in that mode. This
        method exists as the safety net for the case where a write
        slipped through anyway: when the agent leaves EPHEMERAL or its
        session closes, the ephemeral hard-purge calls in here to clean
        up any leak.

        Scoping uses the same JSON-path predicate as the
        ``idx_graph_nodes_agent`` partial index so the DELETE matches a
        live index. Edges are scrubbed too — any edge touching a node
        owned by this agent goes with it.

        Returns:
            Number of node rows destroyed. Zero is the happy path; any
            non-zero value means the privacy layer leaked.
        """
        if not agent_id:
            return 0

        if self.db.backend_type == "postgres":
            agent_path = "(properties::jsonb->>'agent_id')"
        else:
            agent_path = "json_extract(properties, '$.agent_id')"

        async with self.db.transaction():
            # Wipe edges that touch any node owned by this agent first
            # (foreign-key-like consistency, even though we don't have
            # FK constraints on these tables).
            await self.db.execute(
                f"DELETE FROM graph_edges "
                f"WHERE source_id IN (SELECT node_id FROM graph_nodes WHERE {agent_path} = ?) "
                f"   OR target_id IN (SELECT node_id FROM graph_nodes WHERE {agent_path} = ?)",
                (agent_id, agent_id),
            )
            affected = await self.db.execute(
                f"DELETE FROM graph_nodes WHERE {agent_path} = ?",
                (agent_id,),
            )
        return affected if isinstance(affected, int) else 0
    
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
