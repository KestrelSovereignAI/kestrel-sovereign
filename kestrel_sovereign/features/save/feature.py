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

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
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
        self._db = resolve_feature_database(self.agent)
        self.context_manager = getattr(self.agent, 'context_manager', None)
        self.agent_id = self.agent.did
        self._saved_items_store = None

        if not self.storage:
            logger.warning("SaveFeature initialized without storage - tools may not work")

        logger.info("SaveFeature initialized")

    def _get_store(self) -> Optional[SavedItemsStore]:
        """Get or create the saved items store."""
        if self._saved_items_store is None and self._db:
            self._saved_items_store = SavedItemsStore(self._db, self.agent_id)
        return self._saved_items_store

    @staticmethod
    def _parse_tags(raw: str) -> List[str]:
        return [t.strip() for t in raw.split(",") if t.strip()] if raw else []

    @staticmethod
    def _saved_item_partial_or_ok(
        *,
        item: Any,
        confirmation: str,
        data: Dict[str, Any],
    ) -> ToolResult:
        """Surface no-embedding saves as PARTIAL.

        Saving succeeded, but if the item has no embedding the user
        will never find it via semantic !recall — only via exact-name
        list/get. The agent must speak that limitation rather than
        narrate an unconditional "saved" success.
        """
        if not getattr(item, "embedding", None):
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    "saved without an embedding — semantic recall will not "
                    "find this item; only !recall list / !recall get by id "
                    "can retrieve it. Re-save once embeddings are available "
                    "if semantic search is needed."
                ),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

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
    ) -> ToolResult:
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
            return ToolResult.failed("Storage not available")

        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        try:
            conv_store = self.context_manager._get_conversation_store()
            if not conv_store:
                return ToolResult.failed("Conversation store not available")

            if stash_id:
                stashed = await conv_store.get_stashed_messages(stash_id=stash_id)
                stash_name = stash_id
            else:
                stashes = await conv_store.list_stashes()
                if not stashes:
                    return ToolResult.failed("No stashes found")
                stash_id = stashes[0]["stash_id"]
                stash_name = stashes[0].get("name", stash_id)
                stashed = await conv_store.get_stashed_messages(stash_id=stash_id)

            if not stashed:
                return ToolResult.failed(
                    f"Stash {stash_id} is empty or not found",
                    data={"stash_id": stash_id},
                )

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

            source_ref = json.dumps({
                "stash_id": stash_id,
                "message_ids": [m.get("id") for m in stashed]
            })

            tag_list = self._parse_tags(tags)

            if not summary:
                preview = "\n".join(messages_content)[:500]
                summary = f"Stash '{stash_name}' with {len(stashed)} messages: {preview}..."

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

        except Exception as e:
            logger.error(f"save_stash failed: {e}")
            return ToolResult.failed(str(e))

        data = {
            "success": True,
            "saved_item_id": item.id,
            "name": item.name,
            "message_count": len(stashed),
            "has_embedding": item.embedding is not None,
        }
        return self._saved_item_partial_or_ok(
            item=item,
            confirmation=f"Saved stash '{item.name}' as item {item.id} ({len(stashed)} messages)",
            data=data,
        )

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
    ) -> ToolResult:
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
            return ToolResult.failed("Storage not available")

        if not self.context_manager:
            return ToolResult.failed("Context manager not available")

        requested_n: Optional[int] = None

        try:
            conv_store = self.context_manager._get_conversation_store()
            if not conv_store:
                return ToolResult.failed("Conversation store not available")

            if target.startswith("last_"):
                try:
                    requested_n = int(target.split("_")[1])
                    all_messages = await conv_store.get_full_history_with_ids()
                    messages = (
                        all_messages[-requested_n:]
                        if len(all_messages) >= requested_n
                        else all_messages
                    )
                except ValueError:
                    return ToolResult.failed(
                        f"Invalid last_N format: {target}",
                        data={"target": target},
                    )
            elif target.startswith("ids:"):
                try:
                    ids_str = target[4:]
                    message_ids = [int(x.strip()) for x in ids_str.split(",")]
                    messages = await conv_store.get_messages_by_ids(message_ids)
                except ValueError:
                    return ToolResult.failed(
                        f"Invalid message IDs: {target}",
                        data={"target": target},
                    )
            else:
                return ToolResult.failed(
                    f"Invalid target: {target}. Use 'last_N' or 'ids:1,2,3'",
                    data={"target": target},
                )

            if not messages:
                return ToolResult.failed("No messages found", data={"target": target})

            content_json = json.dumps({
                "message_count": len(messages),
                "messages": messages
            })

            source_ref = json.dumps({
                "message_ids": [m.get("id") for m in messages]
            })

            tag_list = self._parse_tags(tags)

            if not summary:
                messages_text = []
                for msg in messages[:3]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")[:200]
                    messages_text.append(f"{role}: {content}")
                summary = f"{len(messages)} messages: " + " | ".join(messages_text)

            item = await store.save_item(
                item_type=SavedItemType.EXCERPT.value,
                name=name,
                content=content_json,
                summary=summary,
                source_type=SourceType.CONVERSATION.value,
                source_ref=source_ref,
                tags=tag_list
            )

        except Exception as e:
            logger.error(f"save_excerpt failed: {e}")
            return ToolResult.failed(str(e))

        data: Dict[str, Any] = {
            "success": True,
            "saved_item_id": item.id,
            "name": item.name,
            "message_count": len(messages),
            "has_embedding": item.embedding is not None,
        }

        # Composite PARTIAL: shortfall on last_N AND/OR no embedding.
        partial_errs: List[str] = []
        if requested_n is not None and len(messages) < requested_n:
            data["requested_count"] = requested_n
            data["shortfall"] = requested_n - len(messages)
            partial_errs.append(
                f"requested last_{requested_n} but only {len(messages)} "
                "messages were available; the saved excerpt is shorter "
                "than requested"
            )
        if not item.embedding:
            partial_errs.append(
                "saved without an embedding — semantic recall will not "
                "find this excerpt; only !recall list / !recall get by id "
                "can retrieve it"
            )

        confirmation = f"Saved excerpt '{item.name}' as item {item.id} ({len(messages)} messages)"
        if partial_errs:
            return ToolResult.partial(
                confirmation=confirmation,
                error=" | ".join(partial_errs),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

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
    ) -> ToolResult:
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
            return ToolResult.failed("Storage not available")

        try:
            tag_list = self._parse_tags(tags)

            item = await store.save_item(
                item_type=item_type,
                name=name,
                content=content,
                summary=summary or content[:500],
                source_type=SourceType.MANUAL.value,
                schema_id=schema_id if schema_id else None,
                tags=tag_list
            )
        except Exception as e:
            logger.error(f"save_item failed: {e}")
            return ToolResult.failed(str(e))

        data = {
            "success": True,
            "saved_item_id": item.id,
            "name": item.name,
            "item_type": item.item_type,
            "has_embedding": item.embedding is not None,
        }
        return self._saved_item_partial_or_ok(
            item=item,
            confirmation=f"Saved item '{item.name}' (id={item.id}, type={item.item_type})",
            data=data,
        )

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
    ) -> ToolResult:
        """
        Semantic search across saved items.

        Args:
            query: Search query (searches by meaning, not just keywords)
            item_type: Filter by type (stash, file, excerpt, structured)
            limit: Maximum results to return
        """
        store = self._get_store()
        if not store:
            return ToolResult.failed("Storage not available")

        try:
            results = await store.search(
                query=query,
                item_type=item_type if item_type else None,
                limit=limit
            )
        except Exception as e:
            logger.error(f"recall failed: {e}")
            return ToolResult.failed(str(e))

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

        data = {
            "success": True,
            "query": query,
            "result_count": len(formatted),
            "results": formatted,
        }
        if not formatted:
            return ToolResult.ok(
                confirmation=f"No matches for query: {query!r}",
                data=data,
            )
        return ToolResult.ok(
            confirmation=f"Found {len(formatted)} match(es) for {query!r}",
            data=data,
        )

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
    ) -> ToolResult:
        """
        List saved items.

        Args:
            item_type: Filter by type (stash, file, excerpt, structured)
            limit: Maximum items to return
        """
        store = self._get_store()
        if not store:
            return ToolResult.failed("Storage not available")

        try:
            items = await store.list_items(
                item_type=item_type if item_type else None,
                limit=limit
            )
        except Exception as e:
            logger.error(f"recall_list failed: {e}")
            return ToolResult.failed(str(e))

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

        data = {
            "success": True,
            "count": len(formatted),
            "items": formatted,
        }
        return ToolResult.ok(
            confirmation=f"Listed {len(formatted)} saved item(s)"
            + (f" (type={item_type})" if item_type else ""),
            data=data,
        )

    @tool(
        name="recall_get",
        description="Get the full content of a saved item by ID.",
        category=ToolCategory.MEMORY,
        command_prefix="!recall get"
    )
    async def recall_get(self, item_id: str) -> ToolResult:
        """
        Get full content of a saved item.

        Args:
            item_id: The ID of the item to retrieve
        """
        store = self._get_store()
        if not store:
            return ToolResult.failed("Storage not available")

        try:
            item = await store.get_by_id(item_id)
        except Exception as e:
            logger.error(f"recall_get failed: {e}")
            return ToolResult.failed(str(e))

        if not item:
            return ToolResult.failed(
                f"Item not found: {item_id}",
                data={"item_id": item_id},
            )

        return ToolResult.ok(
            confirmation=f"Retrieved item {item_id}",
            data={"success": True, "item": item.to_dict()},
        )

    @tool(
        name="recall_delete",
        description="Delete a saved item by ID.",
        category=ToolCategory.MEMORY,
        command_prefix="!recall delete"
    )
    async def recall_delete(self, item_id: str) -> ToolResult:
        """
        Delete a saved item.

        Args:
            item_id: The ID of the item to delete
        """
        store = self._get_store()
        if not store:
            return ToolResult.failed("Storage not available")

        try:
            item = await store.get_by_id(item_id)
            if not item:
                return ToolResult.failed(
                    f"Item not found: {item_id}",
                    data={"item_id": item_id},
                )
            await store.delete_item(item_id)
        except Exception as e:
            logger.error(f"recall_delete failed: {e}")
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=f"Deleted item {item_id} ({item.name!r})",
            data={"success": True, "deleted_id": item_id, "deleted_name": item.name},
        )
