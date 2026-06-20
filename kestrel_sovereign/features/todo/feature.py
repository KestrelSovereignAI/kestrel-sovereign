"""Durable active todo queue for operational work tracking.

Todos are intentionally separate from memory ``action_item`` nodes:
action items are passive extracted memories, while ``todo_item`` nodes
are an agent-operated queue for active loops across turns and wakes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.storage.async_graph_store import GraphNode

logger = logging.getLogger(__name__)


TODO_NODE_TYPE = "todo_item"

OPEN_STATUSES = {"open", "in_progress", "waiting", "blocked"}
TERMINAL_STATUSES = {"done", "cancelled"}
TODO_STATUSES = OPEN_STATUSES | TERMINAL_STATUSES
TODO_SCOPES = {"session", "global", "repo", "issue", "agent"}
TODO_PRIORITIES = {"low", "normal", "high", "urgent"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(
    value: Optional[Any], *, field: str
) -> tuple[Optional[List[Any]], Optional[str]]:
    if value is None:
        return None, None
    if not isinstance(value, list):
        return None, f"{field} must be a list"
    return list(value), None


def _as_dict(
    value: Optional[Any], *, field: str
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if value is None:
        return None, None
    if not isinstance(value, dict):
        return None, f"{field} must be an object"
    return dict(value), None


def _coerce_limit(
    value: Any, *, default: int = 50, maximum: int = 500
) -> tuple[Optional[int], Optional[str]]:
    if value is None:
        return default, None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None, f"limit must be an integer, got {value!r}"
    if limit < 1 or limit > maximum:
        return None, f"limit must be in [1, {maximum}], got {limit}"
    return limit, None


class TodoFeature(Feature):
    """Active operational todo queue exposed as agent tools."""

    @property
    def tool_description(self) -> str:
        return (
            "Manage durable active todos across sessions, external jobs, "
            "evidence gates, and restart/workflow loops"
        )

    @property
    def promote_tools_on_startup(self) -> bool:
        return True

    async def initialize(self):
        self.agent_id = (
            getattr(self.agent, "did", None)
            or getattr(self.agent, "agent_id", None)
            or "default"
        )
        self.enabled = True

    def _graph(self):
        storage = getattr(self.agent, "storage", None)
        return getattr(storage, "graph", None)

    def _current_turn_id(self) -> Optional[str]:
        getter = getattr(self.agent, "_get_current_turn_id", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return None
        return None

    def _current_session_id(self) -> Optional[str]:
        return getattr(self.agent, "_active_session_id", None) or getattr(
            self.agent, "session_id", None
        )

    def _shape(self, node: GraphNode) -> Dict[str, Any]:
        props = dict(node.properties or {})
        props.setdefault("id", node.node_id)
        props.setdefault("title", node.label)
        return props

    def _validate_status(self, status: Optional[str]) -> Optional[str]:
        if status is not None and status not in TODO_STATUSES:
            return f"status must be one of {', '.join(sorted(TODO_STATUSES))}, got {status!r}"
        return None

    def _validate_scope(self, scope: Optional[str]) -> Optional[str]:
        if scope is not None and scope not in TODO_SCOPES:
            return f"scope must be one of {', '.join(sorted(TODO_SCOPES))}, got {scope!r}"
        return None

    def _validate_priority(self, priority: Optional[str]) -> Optional[str]:
        if priority is not None and priority not in TODO_PRIORITIES:
            return f"priority must be one of {', '.join(sorted(TODO_PRIORITIES))}, got {priority!r}"
        return None

    async def _get_owned_node(self, todo_id: str) -> tuple[Optional[GraphNode], Optional[str]]:
        graph = self._graph()
        if graph is None:
            return None, "Graph store not available"
        try:
            node = await graph.get_node(todo_id)
        except Exception as e:
            logger.error("todo lookup failed: %s", e, exc_info=True)
            return None, str(e)
        if node is None or node.node_type != TODO_NODE_TYPE:
            return None, f"Todo {todo_id} not found"
        if (node.properties or {}).get("agent_id") != self.agent_id:
            return None, f"Todo {todo_id} not found"
        return node, None

    async def _persist(self, props: Dict[str, Any]) -> None:
        await self._graph().add_node(
            GraphNode(
                node_id=props["id"],
                node_type=TODO_NODE_TYPE,
                label=props["title"][:120],
                properties=props,
            )
        )

    @tool(
        name="todo_add",
        description=(
            "Create a durable active todo with terminal criteria and optional "
            "external links."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!todo add",
    )
    async def todo_add(
        self,
        title: str,
        description: str = "",
        scope: str = "session",
        status: str = "open",
        priority: str = "normal",
        owner: Optional[str] = None,
        links: Optional[List[Dict[str, Any]]] = None,
        terminal_condition: Optional[str] = None,
        next_check_at: Optional[str] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Create a todo item."""
        if self._graph() is None:
            return ToolResult.failed("Graph store not available")
        if not title or not str(title).strip():
            return ToolResult.failed("title is required")
        for validator, value in (
            (self._validate_scope, scope),
            (self._validate_status, status),
            (self._validate_priority, priority),
        ):
            err = validator(value)
            if err:
                return ToolResult.failed(err)
        link_list, err = _as_list(links, field="links")
        if err:
            return ToolResult.failed(err)
        metadata, err = _as_dict(source_metadata, field="source_metadata")
        if err:
            return ToolResult.failed(err)

        now = _utc_now_iso()
        if status == "done":
            return ToolResult.failed(
                "todo_add cannot create a todo already marked done; create it "
                "active, then use todo_complete with evidence when the "
                "terminal condition is satisfied"
            )
        todo_id = f"todo:{uuid4().hex}"
        metadata = metadata or {}
        metadata.setdefault("turn_id", self._current_turn_id())
        metadata.setdefault("session_id", self._current_session_id())
        props = {
            "id": todo_id,
            "agent_id": self.agent_id,
            "title": str(title).strip(),
            "description": description or "",
            "scope": scope,
            "status": status,
            "priority": priority,
            "owner": owner or self.agent_id,
            "links": link_list or [],
            "terminal_condition": terminal_condition,
            "next_check_at": next_check_at,
            "source_turn": metadata,
            "superseded_by": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": now if status in TERMINAL_STATUSES else None,
        }
        try:
            await self._persist(props)
        except Exception as e:
            logger.error("todo_add write failed: %s", e, exc_info=True)
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=f"Created todo {todo_id}: {props['title']}",
            data={"todo": props},
        )

    @tool(
        name="todo_update",
        description=(
            "Update an active todo without marking it complete unless status "
            "is explicitly terminal."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!todo update",
    )
    async def todo_update(
        self,
        todo_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        scope: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        owner: Optional[str] = None,
        links: Optional[List[Dict[str, Any]]] = None,
        terminal_condition: Optional[str] = None,
        next_check_at: Optional[str] = None,
        superseded_by: Optional[str] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Update a todo. Null fields are preserved."""
        node, err = await self._get_owned_node(todo_id)
        if err:
            return ToolResult.failed(err)
        for validator, value in (
            (self._validate_scope, scope),
            (self._validate_status, status),
            (self._validate_priority, priority),
        ):
            err = validator(value)
            if err:
                return ToolResult.failed(err)
        link_list, err = _as_list(links, field="links")
        if err:
            return ToolResult.failed(err)
        metadata, err = _as_dict(source_metadata, field="source_metadata")
        if err:
            return ToolResult.failed(err)

        props = dict(node.properties or {})
        updates: List[str] = []
        for key, value in (
            ("title", title),
            ("description", description),
            ("scope", scope),
            ("status", status),
            ("priority", priority),
            ("owner", owner),
            ("terminal_condition", terminal_condition),
            ("next_check_at", next_check_at),
            ("superseded_by", superseded_by),
        ):
            if value is not None:
                props[key] = value
                updates.append(key)
        if link_list is not None:
            props["links"] = link_list
            updates.append("links")
        if metadata is not None:
            props.setdefault("source_turn", {})
            props["source_turn"].update(metadata)
            updates.append("source_metadata")
        if not updates:
            return ToolResult.failed("no fields to update")

        if status == "done":
            return ToolResult.failed(
                "use todo_complete with terminal_condition_satisfied=True "
                "and evidence to mark this todo done"
            )
        if status in TERMINAL_STATUSES:
            props["completed_at"] = props.get("completed_at") or _utc_now_iso()
        elif status is not None:
            props["completed_at"] = None
        props["updated_at"] = _utc_now_iso()

        try:
            await self._persist(props)
            if superseded_by:
                await self._graph().add_edge(
                    superseded_by,
                    todo_id,
                    "supersedes",
                    {"created_at": props["updated_at"], "agent_id": self.agent_id},
                )
        except Exception as e:
            logger.error("todo_update write failed: %s", e, exc_info=True)
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=f"Updated todo {todo_id}: {', '.join(updates)}",
            data={"todo": props, "updates": updates},
        )

    @tool(
        name="todo_link_task",
        description=(
            "Attach an external reference to a todo, such as a GitHub issue, "
            "Talon job, A2A task, scheduled job, restart request, action item, "
            "or evidence URL."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!todo link",
    )
    async def todo_link_task(
        self,
        todo_id: str,
        link_type: str,
        target: str,
        title: Optional[str] = None,
        status: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Append one external link to a todo."""
        node, err = await self._get_owned_node(todo_id)
        if err:
            return ToolResult.failed(err)
        meta, err = _as_dict(metadata, field="metadata")
        if err:
            return ToolResult.failed(err)
        if not link_type or not target:
            return ToolResult.failed("link_type and target are required")

        props = dict(node.properties or {})
        links = list(props.get("links") or [])
        link = {
            "type": link_type,
            "target": target,
            "title": title,
            "status": status,
            "url": url,
            "metadata": meta or {},
            "created_at": _utc_now_iso(),
        }
        links.append(link)
        props["links"] = links
        props["updated_at"] = _utc_now_iso()

        try:
            await self._persist(props)
            await self._graph().add_edge(
                todo_id,
                f"{link_type}:{target}",
                "linked_to",
                {"link": link, "agent_id": self.agent_id},
            )
        except Exception as e:
            logger.error("todo_link_task write failed: %s", e, exc_info=True)
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=f"Linked {link_type}:{target} to todo {todo_id}",
            data={"todo": props, "link": link},
        )

    @tool(
        name="todo_list",
        description=(
            "List durable todos by scope/status/owner, excluding superseded "
            "items by default."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!todo list",
    )
    async def todo_list(
        self,
        scope: Optional[str] = None,
        status: Optional[str] = None,
        owner: Optional[str] = None,
        include_done: bool = False,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> ToolResult:
        """List todo items with optional filters."""
        graph = self._graph()
        if graph is None:
            return ToolResult.failed("Graph store not available")
        for validator, value in (
            (self._validate_scope, scope),
            (self._validate_status, status),
        ):
            err = validator(value)
            if err:
                return ToolResult.failed(err)
        limit_val, err = _coerce_limit(limit)
        if err:
            return ToolResult.failed(err)

        filters: Dict[str, Any] = {"agent_id": self.agent_id}
        if scope:
            filters["scope"] = scope
        if status:
            filters["status"] = status
        if owner:
            filters["owner"] = owner
        try:
            nodes = await graph.query_nodes_by_type_and_property(
                TODO_NODE_TYPE,
                filters=filters,
                order_by_created=True,
                limit=limit_val,
            )
        except Exception as e:
            logger.error("todo_list query failed: %s", e, exc_info=True)
            return ToolResult.failed(str(e))

        todos = []
        for node in nodes:
            item = self._shape(node)
            if not include_done and item.get("status") in TERMINAL_STATUSES:
                continue
            if not include_superseded and item.get("superseded_by"):
                continue
            todos.append(item)

        return ToolResult.ok(
            confirmation=f"Retrieved {len(todos)} todo(s)",
            data={"todos": todos, "count": len(todos), "limit_requested": limit_val},
        )

    @tool(
        name="todo_complete",
        description=(
            "Mark a todo done only when its terminal condition is explicitly "
            "satisfied, or cancel/supersede it with a reason."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!todo complete",
    )
    async def todo_complete(
        self,
        todo_id: str,
        outcome: str = "done",
        evidence: Optional[str] = None,
        terminal_condition_satisfied: bool = False,
        superseded_by: Optional[str] = None,
    ) -> ToolResult:
        """Complete, cancel, or supersede a todo."""
        if outcome not in {"done", "cancelled", "superseded"}:
            return ToolResult.failed("outcome must be done, cancelled, or superseded")
        node, err = await self._get_owned_node(todo_id)
        if err:
            return ToolResult.failed(err)

        props = dict(node.properties or {})
        terminal_condition = props.get("terminal_condition")
        if outcome == "done" and not terminal_condition_satisfied:
            return ToolResult.failed(
                "terminal_condition is not satisfied; leave the todo active until "
                "the condition is met or cancel/supersede it",
                data={"terminal_condition": terminal_condition, "todo": props},
            )

        now = _utc_now_iso()
        props["status"] = "cancelled" if outcome in {"cancelled", "superseded"} else "done"
        props["completion_evidence"] = evidence
        props["terminal_condition_satisfied"] = bool(terminal_condition_satisfied)
        props["updated_at"] = now
        props["completed_at"] = now
        if superseded_by:
            props["superseded_by"] = superseded_by

        try:
            await self._persist(props)
            if superseded_by:
                await self._graph().add_edge(
                    superseded_by,
                    todo_id,
                    "supersedes",
                    {"created_at": now, "agent_id": self.agent_id},
                )
        except Exception as e:
            logger.error("todo_complete write failed: %s", e, exc_info=True)
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=f"Completed todo {todo_id} as {props['status']}",
            data={"todo": props},
        )

    @tool(
        name="todo_rollup",
        description=(
            "Summarize pending/waiting/in-progress todos across sessions "
            "and linked systems."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!todo rollup",
    )
    async def todo_rollup(
        self,
        include_done: bool = False,
        limit: int = 100,
    ) -> ToolResult:
        """Return global rollup grouped by status, scope, and link type."""
        graph = self._graph()
        if graph is None:
            return ToolResult.failed("Graph store not available")
        limit_val, err = _coerce_limit(limit, maximum=1000)
        if err:
            return ToolResult.failed(err)

        try:
            nodes = await graph.query_nodes_by_type_and_property(
                TODO_NODE_TYPE,
                filters={"agent_id": self.agent_id},
                order_by_created=True,
                limit=limit_val,
            )
        except Exception as e:
            logger.error("todo_rollup query failed: %s", e, exc_info=True)
            return ToolResult.failed(str(e))

        by_status: Dict[str, int] = {}
        by_scope: Dict[str, int] = {}
        linked_systems: Dict[str, int] = {}
        active: List[Dict[str, Any]] = []
        waiting_or_in_progress: List[Dict[str, Any]] = []

        for node in nodes:
            item = self._shape(node)
            if item.get("superseded_by"):
                continue
            status = item.get("status", "open")
            if not include_done and status in TERMINAL_STATUSES:
                continue
            scope = item.get("scope", "global")
            by_status[status] = by_status.get(status, 0) + 1
            by_scope[scope] = by_scope.get(scope, 0) + 1
            for link in item.get("links") or []:
                if isinstance(link, dict):
                    link_type = str(link.get("type") or "unknown")
                    linked_systems[link_type] = linked_systems.get(link_type, 0) + 1
            if status in OPEN_STATUSES:
                active.append(item)
            if status in {"in_progress", "waiting"}:
                waiting_or_in_progress.append(item)

        return ToolResult.ok(
            confirmation=(
                f"Todo rollup: {len(active)} active item(s), "
                f"{len(waiting_or_in_progress)} in_progress/waiting"
            ),
            data={
                "active": active,
                "waiting_or_in_progress": waiting_or_in_progress,
                "counts": {
                    "by_status": by_status,
                    "by_scope": by_scope,
                    "linked_systems": linked_systems,
                },
                "limit_requested": limit_val,
            },
        )
