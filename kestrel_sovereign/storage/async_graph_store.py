"""
Async Graph Store for Kestrel Storage.

Provides async knowledge graph storage with nodes and edges.
"""
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from .async_database import AsyncDatabase


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
