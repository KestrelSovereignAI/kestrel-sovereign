"""Memory and knowledge graph endpoints."""
from fastapi import APIRouter, HTTPException, Query, Request
import logging

from kestrel_sovereign.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["memories"])


@router.get("/memories")
async def list_memories(request: Request, node_type: str = None, limit: int = Query(100, ge=1, le=500)):
    """List knowledge graph nodes (memories)."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage

        if node_type:
            nodes = await storage.get_nodes_by_type(node_type)
        else:
            all_nodes = []
            for ntype in ["agent", "document", "memory", "backup_artifact", "sovereignty_receipt"]:
                type_nodes = await storage.get_nodes_by_type(ntype)
                all_nodes.extend(type_nodes)
            nodes = all_nodes[:limit]

        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "label": n.label,
                    "properties": n.properties,
                }
                for n in nodes[:limit]
            ],
            "total": len(nodes),
        }
    except Exception as e:
        logger.error(f"Error listing memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving memories.")


@router.get("/memories/{node_id}")
async def get_memory_detail(request: Request, node_id: str):
    """Get detailed information about a memory node."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage

        node = await storage.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Node not found.")

        edges_out = await storage.get_edges_from(node_id)
        edges_in = await storage.get_edges_to(node_id)

        content = None
        content_preview = None
        if node.node_type == "document":
            try:
                content_bytes = await storage.retrieve_file(node_id)
                if content_bytes:
                    content = content_bytes.decode('utf-8')
                    content_preview = content[:500] + "..." if len(content) > 500 else content
            except Exception as e:
                logger.warning(f"Could not retrieve file content for node {node_id}: {e}")
                content_preview = "[Error retrieving content]"

        return {
            "node": {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "label": node.label,
                "properties": node.properties,
            },
            "relationships": {
                "outgoing": [
                    {"target": e.target_id, "type": e.label, "properties": e.properties}
                    for e in edges_out
                ],
                "incoming": [
                    {"source": e.source_id, "type": e.label, "properties": e.properties}
                    for e in edges_in
                ],
            },
            "content": content,
            "content_preview": content_preview,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving memory details.")


@router.get("/identity-chain")
async def get_identity_chain(request: Request):
    """Get the complete identity chain for the agent."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage

        agent_node = await storage.get_node(agent.agent_id)
        if not agent_node:
            raise HTTPException(status_code=404, detail="Agent node not found.")

        constitution_hash = agent_node.properties.get("constitution_hash")
        constitution_node = await storage.get_node(constitution_hash) if constitution_hash else None

        edges = await storage.get_edges_from(agent.agent_id)

        governance_edges = []
        for e in edges:
            target_node = await storage.get_node(e.target_id)
            governance_edges.append({
                "target": e.target_id,
                "type": e.label,
                "target_label": target_node.label if target_node else None,
            })

        chain = {
            "agent": {
                "did": agent.agent_id,
                "created_at": agent_node.properties.get("created_at"),
                "balance": agent_node.properties.get("initialBalance"),
            },
            "constitution": {
                "hash": constitution_hash,
                "label": constitution_node.label if constitution_node else None,
                "created_at": constitution_node.properties.get("created_at") if constitution_node else None,
                "relationship": "governed_by",
            } if constitution_hash else None,
            "governance_edges": governance_edges,
        }

        return chain
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting identity chain: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving identity chain.")


@router.delete("/memories/{node_id}")
@limiter.limit("30/minute")
async def delete_memory(request: Request, node_id: str):
    """Delete a memory node from the knowledge graph."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage

        node = await storage.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Node not found.")

        protected_types = ["agent", "document"]
        if node.node_type in protected_types:
            raise HTTPException(status_code=403, detail=f"Cannot delete {node.node_type} nodes.")

        await storage.delete_node(node_id)

        return {"deleted": True, "node_id": node_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting memory.")
