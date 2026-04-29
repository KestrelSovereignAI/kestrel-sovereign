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

    async def purge_agent_nodes(
        self, agent_id: str, *, since_iso: Optional[str] = None
    ) -> int:
        """Hard-delete graph_nodes tagged with this ``agent_id`` (#767/#867).

        EPHEMERAL agents are not supposed to write to ``graph_nodes`` —
        the privacy wrapper rejects persistent writes in that mode. This
        method exists as the safety net for the case where a write
        slipped through anyway.

        Scoping uses the same JSON-path predicate as the
        ``idx_graph_nodes_agent`` partial index so the DELETE matches a
        live index. Edges are scrubbed too — any edge touching a node
        owned by this agent goes with it.

        Args:
            agent_id: agent's DID.
            since_iso: Optional ISO-8601 timestamp.  When provided, only
                nodes whose ``properties.created_at >= since_iso`` are
                destroyed — this scopes the EPHEMERAL leak-purge to the
                rows authored *during* the EPHEMERAL stint and leaves
                preexisting NORMAL data alone (#867).  When omitted,
                every node owned by this agent is destroyed (legacy
                behaviour preserved for restore-from-CAR and explicit
                administrative wipes).

        Returns:
            Number of node rows destroyed. Zero is the happy path; any
            non-zero value during a leak-purge means the privacy layer
            leaked.
        """
        if not agent_id:
            return 0

        if self.db.backend_type == "postgres":
            agent_path = "(properties::jsonb->>'agent_id')"
            # graph_nodes.properties.created_at is documented as
            # ``YYYY-MM-DDTHH:MM:SS+00:00`` (ISO with T separator, fixed
            # offset).  Normalise it to ``YYYY-MM-DD HH:MM:SS`` so it can
            # be lex-compared against the SQLite-format watermark the
            # privacy wrapper records.  Without this normalisation the
            # ``T`` (0x54) sorts AFTER space (0x20) and every same-day
            # graph row appears strictly greater than the watermark — so
            # pre-stint nodes get purged.
            created_normalized = (
                "to_char(("
                "  CASE WHEN (properties::jsonb->>'created_at') IS NULL THEN NULL "
                "       ELSE ((properties::jsonb->>'created_at')::timestamptz) "
                "  END "
                ") AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')"
            )
        else:
            agent_path = "json_extract(properties, '$.agent_id')"
            # SQLite normalisation: ``T`` → space, then truncate to
            # ``YYYY-MM-DD HH:MM:SS`` (length 19).  Handles both ISO
            # (``2026-04-26T16:31:06+00:00``) and SQLite-format (``2026-04-26
            # 16:31:06``) inputs uniformly.
            created_normalized = (
                "substr("
                "  replace(json_extract(properties, '$.created_at'), 'T', ' '), "
                "  1, 19"
                ")"
            )

        if since_iso:
            # Nodes without a ``created_at`` are excluded from the scoped
            # purge — we can't prove they're in-window leaks, so we
            # preserve them rather than risk destroying real preexisting
            # data.  Operators get a WARNING below if any such nodes
            # exist for this agent (visible-but-skipped surface).
            agent_clause = (
                f"({agent_path} = ? AND {created_normalized} IS NOT NULL "
                f"AND {created_normalized} >= ?)"
            )
            agent_args: tuple = (agent_id, since_iso)
        else:
            agent_clause = f"{agent_path} = ?"
            agent_args = (agent_id,)

        # When scoping by since_iso, count nodes for this agent that have
        # NO created_at — they're skipped by the predicate and we want
        # operators to see them so they can investigate the missing
        # provenance.  Cheap row count, scoped to the agent.
        if since_iso:
            try:
                untimed_row = await self.db.fetchone(
                    f"SELECT COUNT(*) FROM graph_nodes "
                    f"WHERE {agent_path} = ? "
                    f"  AND {created_normalized} IS NULL",
                    (agent_id,),
                )
                untimed = int(untimed_row[0]) if untimed_row else 0
                if untimed > 0:
                    logger.warning(
                        "purge_agent_nodes (scoped): %d node(s) for agent=%s "
                        "have no properties.created_at and were skipped — "
                        "leak coverage is incomplete for them.  Investigate "
                        "the writer and stamp created_at going forward.",
                        untimed, agent_id,
                    )
            except Exception:
                # Pre-flight count is informational only.
                pass

        async with self.db.transaction():
            # Wipe edges that touch any node we're about to remove first
            # (foreign-key-like consistency, even though we don't have
            # FK constraints on these tables).
            await self.db.execute(
                f"DELETE FROM graph_edges "
                f"WHERE source_id IN (SELECT node_id FROM graph_nodes WHERE {agent_clause}) "
                f"   OR target_id IN (SELECT node_id FROM graph_nodes WHERE {agent_clause})",
                agent_args + agent_args,
            )
            affected = await self.db.execute(
                f"DELETE FROM graph_nodes WHERE {agent_clause}",
                agent_args,
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
