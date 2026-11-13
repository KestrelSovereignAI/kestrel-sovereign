"""Native MCP memory callbacks use Core's canonical governed dispatcher."""

import hashlib
import json
from typing import Annotated, Any

from kestrel_sdk.resources import ResourceError
from kestrel_sdk.tools.base import ToolSchema
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from kestrel_sovereign.agent.invocation import validate_invocation_id
from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.features.base import _serialize_tool_result

from .tool_caller import bind_native_mcp_tool_caller


class NativeMCPMemoryTools:
    """A fixed protocol surface, without another execution or memory registry."""

    def __init__(self, publication):
        self.publication = publication

    async def _validate_arguments(self, arguments, schema: ToolSchema | None = None):
        policy = self.publication.policy
        if "limit" in arguments:
            limit = arguments["limit"]
        else:
            if not isinstance(schema, ToolSchema) or not isinstance(schema.parameters, list):
                raise ToolError("Canonical memory tool count default is unavailable")
            parameters = [parameter for parameter in schema.parameters if parameter.name == "limit"]
            if len(parameters) != 1 or parameters[0].required:
                raise ToolError("Canonical memory tool count default is unavailable")
            limit = parameters[0].default
        if type(limit) is not int or not 1 <= limit <= policy.max_memory_records:
            raise ToolError("Requested memory count exceeds the configured server limit")
        if "query" in arguments and (not arguments["query"].strip()
                or len(arguments["query"].encode("utf-8")) > policy.max_query_bytes):
            raise ToolError("Memory query is empty or exceeds the configured server limit")

    async def _invoke(self, context, name, arguments, operation_id, conversation_id):
        policy = self.publication.policy
        try:
            operation_id = validate_invocation_id(operation_id)
            conversation_id = validate_invocation_id(conversation_id)
        except ValueError as error:
            raise ToolError("A bounded operation_id and conversation_id are required") from error
        await self._validate_arguments(arguments)
        try:
            async with bind_native_mcp_tool_caller(self.publication, context, "memory:read",
                    disclose=True) as caller:
                session = "mcp:" + hashlib.sha256(json.dumps([caller.identity, conversation_id],
                    separators=(",", ":")).encode()).hexdigest()
                result = await self.publication.agent.execute_named_tool(name, arguments,
                    session_id=session, source="mcp", operation_id=operation_id,
                    argument_guard=self._validate_arguments)
                payload = _serialize_tool_result(result)
                if not isinstance(payload, dict):
                    raise ToolError("Canonical memory tool returned an invalid result envelope")
                encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                response = CallToolResult(content=[TextContent(type="text", text=encoded)],
                    structured_content=payload,
                    is_error=payload.get("status") == "error" or payload.get("success") is False)
                if len(response.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")) > policy.max_result_bytes:
                    raise ToolError("Memory result exceeds the configured server byte limit")
                return response
        except (ResourceError, ExecutionAuthorityError, ToolNotRegisteredError, TimeoutError) as error:
            raise ToolError("Native memory operation was refused or its outcome is unconfirmed; retain operation_id") from error

    async def search_memory(self, context: Context, query: str,
            operation_id: str, conversation_id: str,
            limit: Annotated[int, Field(strict=True, ge=1)] = 20,
            session_id: str | None = None) -> Annotated[CallToolResult, dict[str, Any]]:
        """Search Kestrel memory. Retain operation_id on retries; use a new ID for a new intention."""
        return await self._invoke(context, "search_memory",
            {"query": query, "limit": limit, "session_id": session_id}, operation_id, conversation_id)

    async def recall_recent(self, context: Context, operation_id: str, conversation_id: str,
            limit: Annotated[int, Field(strict=True, ge=1)] = 20) -> Annotated[CallToolResult, dict[str, Any]]:
        """Recall recent Kestrel messages. Retain operation_id on retries; use a new ID for a new intention."""
        return await self._invoke(context, "recall_recent", {"limit": limit}, operation_id, conversation_id)

    def register(self):
        annotations = ToolAnnotations(read_only_hint=False, destructive_hint=False,
            idempotent_hint=False, open_world_hint=False)
        for callback in (self.search_memory, self.recall_recent):
            self.publication.runtime.server.add_tool(callback, annotations=annotations)
