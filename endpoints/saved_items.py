"""Saved Items API endpoints.

REST API for managing saved items (stashes, files, excerpts, structured items)
with semantic search and IPFS support.
"""
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import logging

from kestrel_sovereign.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/saved-items", tags=["saved-items"])


# =============================================================================
# Pydantic Models
# =============================================================================

class SaveItemRequest(BaseModel):
    """Request body for saving an item."""
    item_type: str = Field(..., description="Type: stash, file, excerpt, structured")
    name: str = Field(..., description="Human-readable name")
    content: str = Field(..., description="Content (JSON or text)")
    summary: Optional[str] = Field(None, description="Summary for search")
    source_type: Optional[str] = Field(None, description="Source: conversation, file, url, manual")
    source_ref: Optional[str] = Field(None, description="Reference to original source")
    schema_id: Optional[str] = Field(None, description="Schema type for structured items")
    tags: Optional[List[str]] = Field(None, description="Tags for filtering")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    pin_to_ipfs: bool = Field(False, description="Pin content to IPFS")


class SaveStructuredItemRequest(BaseModel):
    """Request body for saving a structured item."""
    schema_id: str = Field(..., description="Schema type (recipe, contact, story, etc.)")
    content: Dict[str, Any] = Field(..., description="Structured content matching schema")
    name: Optional[str] = Field(None, description="Optional name override")
    summary: Optional[str] = Field(None, description="Optional summary")
    tags: Optional[List[str]] = Field(None, description="Tags for filtering")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    pin_to_ipfs: bool = Field(False, description="Pin content to IPFS")


class UpdateItemRequest(BaseModel):
    """Request body for updating an item."""
    name: Optional[str] = Field(None, description="New name")
    summary: Optional[str] = Field(None, description="New summary")
    tags: Optional[List[str]] = Field(None, description="New tags (replaces existing)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to merge")


class SearchRequest(BaseModel):
    """Request body for semantic search."""
    query: str = Field(..., description="Search query")
    item_type: Optional[str] = Field(None, description="Filter by item type")
    limit: int = Field(10, ge=1, le=100, description="Max results")


# =============================================================================
# Helper Functions
# =============================================================================

def _get_saved_items_store(request: Request):
    """Get SavedItemsStore from agent storage."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    agent = request.app.state.agent
    storage = agent.storage

    # Get the database
    db = getattr(storage, 'db', None)
    if not db:
        raise HTTPException(status_code=503, detail="Database not available.")

    # Create SavedItemsStore
    from kestrel_sovereign.storage.saved_items_store import SavedItemsStore
    agent_id = getattr(storage, 'agent_id', '') or getattr(agent, 'agent_id', '')
    return SavedItemsStore(db, agent_id)


# =============================================================================
# Endpoints
# =============================================================================

@router.get("")
async def list_items(
    request: Request,
    item_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=500, description="Max items to return")
):
    """List saved items, optionally filtered by type."""
    try:
        store = _get_saved_items_store(request)
        items = await store.list_items(item_type=item_type, limit=limit)

        return {
            "items": [item.to_dict() for item in items],
            "total": len(items),
            "item_type_filter": item_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing saved items: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing saved items.")


@router.get("/stats")
async def get_stats(request: Request):
    """Get statistics about saved items."""
    try:
        store = _get_saved_items_store(request)
        stats = await store.get_stats()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error getting statistics.")


@router.get("/schemas")
async def list_schemas(request: Request):
    """List available schemas for structured items."""
    try:
        from kestrel_sovereign.storage.saved_items_store import list_schemas, ITEM_SCHEMAS
        return {
            "schemas": list_schemas(),
            "schema_details": ITEM_SCHEMAS
        }
    except Exception as e:
        logger.error(f"Error listing schemas: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing schemas.")


@router.get("/tags")
async def get_all_tags(request: Request):
    """Get all unique tags across saved items."""
    try:
        store = _get_saved_items_store(request)
        tags = await store.get_all_tags()
        return {"tags": tags, "total": len(tags)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error getting tags.")


@router.get("/by-tag/{tag}")
async def list_by_tag(
    request: Request,
    tag: str,
    item_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=500)
):
    """List items with a specific tag."""
    try:
        store = _get_saved_items_store(request)
        items = await store.list_by_tag(tag=tag, item_type=item_type, limit=limit)

        return {
            "items": [item.to_dict() for item in items],
            "total": len(items),
            "tag": tag
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing by tag: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing by tag.")


@router.get("/by-schema/{schema_id}")
async def list_by_schema(
    request: Request,
    schema_id: str,
    limit: int = Query(50, ge=1, le=500)
):
    """List structured items by schema type."""
    try:
        store = _get_saved_items_store(request)
        items = await store.list_by_schema(schema_id=schema_id, limit=limit)

        return {
            "items": [item.to_dict() for item in items],
            "total": len(items),
            "schema_id": schema_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing by schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing by schema.")


@router.get("/{item_id}")
async def get_item(request: Request, item_id: str):
    """Get a specific saved item by ID."""
    try:
        store = _get_saved_items_store(request)
        item = await store.get_by_id(item_id)

        if not item:
            raise HTTPException(status_code=404, detail="Item not found.")

        return {"item": item.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error getting item.")


@router.post("")
@limiter.limit("30/minute")
async def save_item(request: Request, body: SaveItemRequest):
    """Save a new item."""
    try:
        store = _get_saved_items_store(request)

        item = await store.save_item(
            item_type=body.item_type,
            name=body.name,
            content=body.content,
            summary=body.summary,
            source_type=body.source_type,
            source_ref=body.source_ref,
            schema_id=body.schema_id,
            tags=body.tags,
            metadata=body.metadata,
            pin_to_ipfs=body.pin_to_ipfs
        )

        return {
            "success": True,
            "item": item.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request parameters")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error saving item.")


@router.post("/structured")
@limiter.limit("30/minute")
async def save_structured_item(request: Request, body: SaveStructuredItemRequest):
    """Save a structured item with schema validation."""
    try:
        store = _get_saved_items_store(request)

        item = await store.save_structured_item(
            schema_id=body.schema_id,
            content=body.content,
            name=body.name,
            summary=body.summary,
            tags=body.tags,
            metadata=body.metadata,
            pin_to_ipfs=body.pin_to_ipfs
        )

        return {
            "success": True,
            "item": item.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request parameters")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving structured item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error saving structured item.")


@router.post("/search")
@limiter.limit("30/minute")
async def search_items(request: Request, body: SearchRequest):
    """Semantic search across saved items."""
    try:
        store = _get_saved_items_store(request)

        results = await store.search(
            query=body.query,
            item_type=body.item_type,
            limit=body.limit
        )

        return {
            "results": results,
            "total": len(results),
            "query": body.query
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching items: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error searching items.")


@router.patch("/{item_id}")
@limiter.limit("30/minute")
async def update_item(request: Request, item_id: str, body: UpdateItemRequest):
    """Update an existing item."""
    try:
        store = _get_saved_items_store(request)

        item = await store.update_item(
            item_id=item_id,
            name=body.name,
            summary=body.summary,
            tags=body.tags,
            metadata=body.metadata
        )

        if not item:
            raise HTTPException(status_code=404, detail="Item not found.")

        return {
            "success": True,
            "item": item.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating item.")


@router.post("/{item_id}/pin")
@limiter.limit("30/minute")
async def pin_to_ipfs(request: Request, item_id: str):
    """Pin an existing item to IPFS."""
    try:
        store = _get_saved_items_store(request)

        item = await store.get_by_id(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found.")

        cid = await store.pin_item_to_ipfs(item_id)

        return {
            "success": cid is not None,
            "item_id": item_id,
            "ipfs_cid": cid,
            "message": "Already pinned" if item.ipfs_cid else ("Pinned successfully" if cid else "IPFS not available")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pinning to IPFS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error pinning to IPFS.")


@router.delete("/{item_id}")
@limiter.limit("30/minute")
async def delete_item(request: Request, item_id: str):
    """Delete a saved item."""
    try:
        store = _get_saved_items_store(request)

        item = await store.get_by_id(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found.")

        await store.delete_item(item_id)

        return {
            "success": True,
            "deleted_id": item_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting item.")
