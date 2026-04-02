"""
Save Feature for Kestrel Agent.

Provides persistent storage for content with embeddings for semantic search.
Supports saving:
- Stashes (persisted context)
- Files (documents)
- Conversation excerpts
- Structured items (recipes, stories, etc.)

All saved items get embeddings for semantic retrieval via !recall.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory
from kestrel_sovereign.storage.saved_items_store import (
    SavedItemsStore, SavedItemType, SourceType
)

logger = logging.getLogger(__name__)


class SaveFeature(Feature):
    """
    Persistent storage with semantic search.

    Provides tools for:
    - Saving stashes to long-term storage
    - Saving files and documents
    - Saving conversation excerpts
    - Recalling saved items via semantic search
    """

    @property
    def tool_description(self) -> str:
        return "Save content for later retrieval - stashes, files, excerpts, structured items"

    async def initialize(self):
        """Initialize the save feature with required references."""
        self.storage = getattr(self.agent, 'storage', None)
        self.context_manager = getattr(self.agent, 'context_manager', None)
        self.agent_id = self.agent.did
        self._saved_items_store = None

        if not self.storage:
            logger.warning("SaveFeature initialized without storage - tools may not work")

        logger.info("SaveFeature initialized")

    def _get_store(self) -> Optional[SavedItemsStore]:
        """Get or create the saved items store."""
        if self._saved_items_store is None and self.storage:
            db = getattr(self.storage, 'db', None)
            if db:
                self._saved_items_store = SavedItemsStore(db, self.agent_id)
        return self._saved_items_store

    @tool(
        name="save_stash",
        description="Save a stash to long-term storage for later retrieval. The stash content gets an embedding so you can find it later with semantic search.",
        category=ToolCategory.MEMORY,
        command_prefix="!save stash"
    )
    async def save_stash(
        self,
        stash_id: str = "",
        name: str = "",
        summary: str = "",
        tags: str = ""
    ) -> Dict[str, Any]:
        """
        Persist a stash to long-term storage.

        Args:
            stash_id: Specific stash to save (default: most recent)
            name: Name for the saved item (default: stash name)
            summary: Optional summary for search
            tags: Comma-separated tags for filtering
        """
        store = self._get_store()
        if not store:
            return {"success": False, "error": "Storage not available"}

        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            # Get the conversation store to access stash
            conv_store = self.context_manager._get_conversation_store()
            if not conv_store:
                return {"success": False, "error": "Conversation store not available"}

            # Get the stash
            if stash_id:
                stashed = await conv_store.get_stashed_messages(stash_id=stash_id)
                stash_name = stash_id
            else:
                # Get most recent stash
                stashes = await conv_store.list_stashes()
                if not stashes:
                    return {"success": False, "error": "No stashes found"}
                stash_id = stashes[0]["stash_id"]
                stash_name = stashes[0].get("name", stash_id)
                stashed = await conv_store.get_stashed_messages(stash_id=stash_id)

            if not stashed:
                return {"success": False, "error": f"Stash {stash_id} is empty or not found"}

            # Build content from stashed messages
            messages_content = []
            for msg in stashed:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                messages_content.append(f"{role}: {content}")

            content_json = json.dumps({
                "stash_id": stash_id,
                "stash_name": stash_name,
                "message_count": len(stashed),
                "messages": stashed
            })

            # Build source reference
            source_ref = json.dumps({
                "stash_id": stash_id,
                "message_ids": [m.get("id") for m in stashed]
            })

            # Parse tags
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

            # Generate summary if not provided
            if not summary:
                # Use first 500 chars of messages as summary
                preview = "\n".join(messages_content)[:500]
                summary = f"Stash '{stash_name}' with {len(stashed)} messages: {preview}..."

            # Save the item
            item = await store.save_item(
                item_type=SavedItemType.STASH.value,
                name=name or stash_name,
                content=content_json,
                summary=summary,
                source_type=SourceType.CONVERSATION.value,
                source_ref=source_ref,
                tags=tag_list,
                metadata={"original_stash_id": stash_id}
            )

            return {
                "success": True,
                "saved_item_id": item.id,
                "name": item.name,
                "message_count": len(stashed),
                "has_embedding": item.embedding is not None
            }

        except Exception as e:
            logger.error(f"save_stash failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="save_excerpt",
        description="Save conversation messages for later retrieval. Use this to preserve important discussions, decisions, or information.",
        category=ToolCategory.MEMORY,
        command_prefix="!save excerpt"
    )
    async def save_excerpt(
        self,
        target: str,
        name: str,
        summary: str = "",
        tags: str = ""
    ) -> Dict[str, Any]:
        """
        Save conversation excerpt to long-term storage.

        Args:
            target: Messages to save - "last_N" or "ids:1,2,3"
            name: Name for the saved excerpt
            summary: Optional summary for search
            tags: Comma-separated tags
        """
        store = self._get_store()
        if not store:
            return {"success": False, "error": "Storage not available"}

        if not self.context_manager:
            return {"success": False, "error": "Context manager not available"}

        try:
            conv_store = self.context_manager._get_conversation_store()
            if not conv_store:
                return {"success": False, "error": "Conversation store not available"}

            # Get messages based on target
            if target.startswith("last_"):
                try:
                    n = int(target.split("_")[1])
                    all_messages = await conv_store.get_full_history_with_ids()
                    messages = all_messages[-n:] if len(all_messages) >= n else all_messages
                except ValueError:
                    return {"success": False, "error": f"Invalid last_N format: {target}"}
            elif target.startswith("ids:"):
                try:
                    ids_str = target[4:]
                    message_ids = [int(x.strip()) for x in ids_str.split(",")]
                    messages = await conv_store.get_messages_by_ids(message_ids)
                except ValueError:
                    return {"success": False, "error": f"Invalid message IDs: {target}"}
            else:
                return {
                    "success": False,
                    "error": f"Invalid target: {target}. Use 'last_N' or 'ids:1,2,3'"
                }

            if not messages:
                return {"success": False, "error": "No messages found"}

            # Build content
            content_json = json.dumps({
                "message_count": len(messages),
                "messages": messages
            })

            source_ref = json.dumps({
                "message_ids": [m.get("id") for m in messages]
            })

            # Parse tags
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

            # Generate summary if not provided
            if not summary:
                messages_text = []
                for msg in messages[:3]:  # First 3 messages
                    role = msg.get("role", "")
                    content = msg.get("content", "")[:200]
                    messages_text.append(f"{role}: {content}")
                summary = f"{len(messages)} messages: " + " | ".join(messages_text)

            # Save
            item = await store.save_item(
                item_type=SavedItemType.EXCERPT.value,
                name=name,
                content=content_json,
                summary=summary,
                source_type=SourceType.CONVERSATION.value,
                source_ref=source_ref,
                tags=tag_list
            )

            return {
                "success": True,
                "saved_item_id": item.id,
                "name": item.name,
                "message_count": len(messages),
                "has_embedding": item.embedding is not None
            }

        except Exception as e:
            logger.error(f"save_excerpt failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="save_item",
        description="Save arbitrary content (text, JSON) for later retrieval. Use for recipes, notes, decisions, or any structured content.",
        category=ToolCategory.MEMORY,
        command_prefix="!save item"
    )
    async def save_item(
        self,
        name: str,
        content: str,
        item_type: str = "structured",
        summary: str = "",
        tags: str = "",
        schema_id: str = ""
    ) -> Dict[str, Any]:
        """
        Save arbitrary content to long-term storage.

        Args:
            name: Name for the item
            content: The content to save (text or JSON)
            item_type: Type of item (default: structured)
            summary: Optional summary for search
            tags: Comma-separated tags
            schema_id: Optional schema identifier (e.g., "recipe", "user_story")
        """
        store = self._get_store()
        if not store:
            return {"success": False, "error": "Storage not available"}

        try:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

            item = await store.save_item(
                item_type=item_type,
                name=name,
                content=content,
                summary=summary or content[:500],
                source_type=SourceType.MANUAL.value,
                schema_id=schema_id if schema_id else None,
                tags=tag_list
            )

            return {
                "success": True,
                "saved_item_id": item.id,
                "name": item.name,
                "item_type": item.item_type,
                "has_embedding": item.embedding is not None
            }

        except Exception as e:
            logger.error(f"save_item failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="recall",
        description="Search saved items using semantic search. Find previously saved stashes, excerpts, files, and items by meaning.",
        category=ToolCategory.MEMORY,
        command_prefix="!recall"
    )
    async def recall(
        self,
        query: str,
        item_type: str = "",
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Semantic search across saved items.

        Args:
            query: Search query (searches by meaning, not just keywords)
            item_type: Filter by type (stash, file, excerpt, structured)
            limit: Maximum results to return
        """
        store = self._get_store()
        if not store:
            return {"success": False, "error": "Storage not available"}

        try:
            results = await store.search(
                query=query,
                item_type=item_type if item_type else None,
                limit=limit
            )

            # Format results for display
            formatted = []
            for r in results:
                item = r["item"]
                formatted.append({
                    "id": item["id"],
                    "name": item["name"],
                    "type": item["item_type"],
                    "summary": item.get("summary", "")[:200],
                    "score": round(r["score"], 3),
                    "tags": item.get("tags", []),
                    "created_at": item.get("created_at")
                })

            return {
                "success": True,
                "query": query,
                "result_count": len(formatted),
                "results": formatted
            }

        except Exception as e:
            logger.error(f"recall failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="recall_list",
        description="List all saved items, optionally filtered by type.",
        category=ToolCategory.MEMORY,
        command_prefix="!recall list"
    )
    async def recall_list(
        self,
        item_type: str = "",
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        List saved items.

        Args:
            item_type: Filter by type (stash, file, excerpt, structured)
            limit: Maximum items to return
        """
        store = self._get_store()
        if not store:
            return {"success": False, "error": "Storage not available"}

        try:
            items = await store.list_items(
                item_type=item_type if item_type else None,
                limit=limit
            )

            formatted = []
            for item in items:
                formatted.append({
                    "id": item.id,
                    "name": item.name,
                    "type": item.item_type,
                    "summary": item.summary[:100] if item.summary else "",
                    "tags": item.tags,
                    "created_at": item.created_at.isoformat() if item.created_at else None
                })

            return {
                "success": True,
                "count": len(formatted),
                "items": formatted
            }

        except Exception as e:
            logger.error(f"recall_list failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="recall_get",
        description="Get the full content of a saved item by ID.",
        category=ToolCategory.MEMORY,
        command_prefix="!recall get"
    )
    async def recall_get(self, item_id: str) -> Dict[str, Any]:
        """
        Get full content of a saved item.

        Args:
            item_id: The ID of the item to retrieve
        """
        store = self._get_store()
        if not store:
            return {"success": False, "error": "Storage not available"}

        try:
            item = await store.get_by_id(item_id)
            if not item:
                return {"success": False, "error": f"Item not found: {item_id}"}

            return {
                "success": True,
                "item": item.to_dict()
            }

        except Exception as e:
            logger.error(f"recall_get failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="recall_delete",
        description="Delete a saved item by ID.",
        category=ToolCategory.MEMORY,
        command_prefix="!recall delete"
    )
    async def recall_delete(self, item_id: str) -> Dict[str, Any]:
        """
        Delete a saved item.

        Args:
            item_id: The ID of the item to delete
        """
        store = self._get_store()
        if not store:
            return {"success": False, "error": "Storage not available"}

        try:
            # Check if exists first
            item = await store.get_by_id(item_id)
            if not item:
                return {"success": False, "error": f"Item not found: {item_id}"}

            await store.delete_item(item_id)

            return {
                "success": True,
                "deleted_id": item_id,
                "deleted_name": item.name
            }

        except Exception as e:
            logger.error(f"recall_delete failed: {e}")
            return {"success": False, "error": str(e)}
