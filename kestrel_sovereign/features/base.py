import inspect
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Type, Union, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sdk.hooks.base import Hook
from abc import ABC, abstractmethod
from kestrel_sdk.tools.base import ToolSchema, ToolParameter, ToolCategory, AgentTool
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
# Import the A2A types from the SDK directly rather than the
# ``kestrel_sovereign.a2a`` re-export package. Importing the sovereign a2a
# package runs its ``__init__`` which eagerly pulls in task_manager/task_worker
# → the A2A stores → the storage SQLA models, and those models size a vector
# column at import time via ``get_provider_embedding_service()``. When
# ``features.base`` is still being initialized (it is imported very early), that
# chain re-enters ``from kestrel_sovereign.features.base import Feature`` against
# the half-built module and raises a circular ImportError that silently disables
# provider embeddings (#1792). The sovereign modules are pure re-exports of
# these same SDK symbols, so importing them from the SDK is equivalent.
from kestrel_sdk.a2a.agent_card import AgentCard, AgentSkill, AgentCapabilities
from kestrel_sdk.a2a.types import Task, TaskState, TaskStatus, Artifact, DataPart, Message, TextPart

# The SDK Feature is the canonical base class for feature packages.
# Sovereign's richer Feature inherits from it so extracted packages that
# subclass kestrel_sdk.features.base.Feature are also recognized as
# kestrel_sovereign.features.base.Feature at runtime (issubclass passes).
from kestrel_sdk.features.base import Feature as _SdkFeature

logger = logging.getLogger(__name__)

# Maximum tool call iterations (configurable via environment variable)
# Increased to 50 for long-running tasks like code analysis and multi-step operations
MAX_TOOL_ITERATIONS = int(os.environ.get("KESTREL_MAX_TOOL_ITERATIONS", "50"))

CONTINUATION_INTENT_RE = re.compile(
    r"\b("
    r"let me|i(?:'ll| will| am going to)|"
    r"one moment|hang on|checking|calling|running|searching|looking up"
    r")\b.{0,80}\b("
    r"check|look|search|call|run|fetch|inspect|open|query|verify|use|try|"
    r"github|tool|cli|browser|file|repo|issue|database|db|talon"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

TURN_COMPLETION_REPAIR_PROMPT = """You just wrote text that indicates this task is still in progress, but you did not emit a tool call.

Continue the same task now:
- If the work requires an available tool, emit the tool call now.
- If no tool is needed or available, provide the final answer now.
- Do not describe a future tool call without making it."""


def _serialize_tool_result(result: Any) -> Any:
    """Convert a tool result to a JSON-serializable format.

    Handles dataclasses with to_dict(), lists, enums, and nested structures.
    """
    if result is None or isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, dict):
        return {k: _serialize_tool_result(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [_serialize_tool_result(item) for item in result]
    if hasattr(result, 'to_dict'):
        return result.to_dict()
    if hasattr(result, 'value'):  # Enum
        return result.value
    return str(result)


@runtime_checkable
class TaskHandler(Protocol):
    """Protocol for A2A task handling. Features implement this."""
    async def handle_task(self, task: Task) -> Task:
        """Handle an A2A task and return the updated task."""
        ...


def parse_docstring_params(docstring: Optional[str]) -> Dict[str, str]:
    """
    Parse parameter descriptions from a docstring.
    
    Supports common docstring formats:
    - Google style: `param_name: Description here`
    - Sphinx style: `:param param_name: Description here`
    - NumPy style: `param_name : type\n    Description here`
    
    Args:
        docstring: The docstring to parse
        
    Returns:
        Dict mapping parameter names to their descriptions
    """
    if not docstring:
        return {}
    
    param_descriptions = {}
    
    # Patterns for different docstring styles
    patterns = [
        # Google style: "    param_name: Description"
        r'^\s*(\w+)\s*:\s*(.+?)(?=\n\s*\w+\s*:|$)',
        # Sphinx style: ":param param_name: Description"
        r':param\s+(\w+)\s*:\s*(.+?)(?=\n\s*:|$)',
        # NumPy style: "param_name : type" followed by indented description
        r'^\s*(\w+)\s*:\s*\w+.*?\n\s+(.+?)(?=\n\s*\w+\s*:|$)',
    ]
    
    # Try Google/reStructuredText Args section first
    args_section_match = re.search(
        r'(?:Args|Arguments|Parameters):\s*\n((?:\s+.+\n?)+)',
        docstring,
        re.IGNORECASE | re.MULTILINE
    )
    
    if args_section_match:
        args_section = args_section_match.group(1)
        # Parse individual parameters from Args section
        # Match: "    param_name: description" or "    param_name (type): description"
        param_pattern = r'^\s+(\w+)\s*(?:\([^)]+\))?\s*:\s*(.+?)(?=\n\s+\w+|\Z)'
        for match in re.finditer(param_pattern, args_section, re.MULTILINE | re.DOTALL):
            param_name = match.group(1).strip()
            description = match.group(2).strip()
            # Clean up multi-line descriptions
            description = re.sub(r'\s+', ' ', description)
            param_descriptions[param_name] = description
    
    # Try Sphinx style :param: tags
    if not param_descriptions:
        for match in re.finditer(r':param\s+(\w+)\s*:\s*(.+?)(?=\n\s*:|$)', docstring, re.DOTALL):
            param_name = match.group(1).strip()
            description = match.group(2).strip()
            description = re.sub(r'\s+', ' ', description)
            param_descriptions[param_name] = description
    
    return param_descriptions

class Feature(_SdkFeature):
    """
    Base class for Kestrel Features - each Feature IS a subagent.

    Extends the SDK's minimal Feature interface with sovereign-specific runtime
    behavior (LLM calls, subagent execution, hook enforcement). Packages that
    subclass kestrel_sdk.features.base.Feature are ALSO recognized as sovereign
    Features at discovery time because of this inheritance chain.

    A Feature encapsulates a specific domain of functionality (e.g., Sovereignty, MCP, Models).
    It can expose methods as Tools to the agent, and can be called AS a tool by the orchestrator
    with its own LLM context (A2A pattern).
    """

    # Node type used for persisting feature config in the knowledge graph.
    _CONFIG_NODE_TYPE = "feature_config"

    def __init__(self, agent):
        self.agent = agent
        self.name = self.__class__.__name__
        self.disabled_skills: set = set()

    @staticmethod
    def _signals_unfinished_tool_work(content: Optional[str]) -> bool:
        """Return True when assistant text promises more tool-backed work."""
        if not content:
            return False
        return bool(CONTINUATION_INTENT_RE.search(content))

    @staticmethod
    def _append_missing_tool_call_repair(messages: list, content: str) -> list:
        """Return a repaired message list for one no-tool continuation retry."""
        repaired = list(messages)
        repaired.append({"role": "assistant", "content": content or ""})
        repaired.append({"role": "user", "content": TURN_COMPLETION_REPAIR_PROMPT})
        return repaired

    @staticmethod
    def _extract_response_reasoning_content(response: Any) -> Optional[str]:
        """Return provider reasoning that must be replayed with tool history."""
        raw = getattr(response, "raw", None)
        if isinstance(raw, dict):
            reasoning = raw.get("reasoning_content")
            return reasoning if isinstance(reasoning, str) and reasoning else None

        try:
            message = raw.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return None

        reasoning = getattr(message, "reasoning_content", None)
        return reasoning if isinstance(reasoning, str) and reasoning else None

    def _build_subagent_assistant_tool_history_msg(self, response: Any) -> dict:
        """Build assistant tool-call history for feature subagent loops."""
        message = {
            "role": "assistant",
            "content": getattr(response, "content", None) or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": (
                            tc.arguments if isinstance(tc.arguments, dict)
                            else json.loads(tc.arguments) if tc.arguments else {}
                        ),
                    },
                }
                for tc in response.tool_calls
            ],
        }

        reasoning_content = self._extract_response_reasoning_content(response)
        if reasoning_content and getattr(response, "tool_calls", None):
            message["reasoning_content"] = reasoning_content

        return message

    async def _repair_subagent_premature_yield(
        self,
        response: Any,
        messages: list,
        tools: List[Dict[str, Any]],
        tool_executor: Optional[Any] = None,
    ) -> Any:
        """Give a feature subagent one more step when it narrates but emits no tool.

        ``tool_executor`` is threaded through to ``generate_with_messages``
        so codex-routed repair turns don't hit the same "requires a
        tool_executor callback" error the initial subagent call was
        wired around (codex round 1 P2 on #1461 follow-up).
        """
        content = getattr(response, "content", "") or ""
        if not tools or not self._signals_unfinished_tool_work(content):
            return response

        logger.warning(
            "[SUBAGENT %s] Model signaled continuation without tool_calls; issuing one repair turn",
            self.name,
        )
        return await self.agent.llm_service.generate_with_messages(
            messages=self._append_missing_tool_call_repair(messages, content),
            tools=tools if tools else None,
            tool_executor=tool_executor,
        )

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    @abstractmethod
    async def initialize(self):
        """Initialize the feature."""
        pass

    async def shutdown(self):
        """Cleanup resources."""
        pass

    async def on_enable(self):
        """Called when feature is enabled.

        Register hooks, start background tasks. Hooks returned by
        ``get_hooks()`` are auto-registered before this method is called,
        so only use this for additional setup beyond hook registration.
        """
        pass

    async def on_disable(self):
        """Called when feature is disabled.

        Unregister hooks, stop background tasks. Hooks returned by
        ``get_hooks()`` are auto-unregistered after this method is called,
        so only use this for additional teardown beyond hook unregistration.
        """
        pass

    async def on_remove(self):
        """Called before feature package is uninstalled. Clean up stored data."""
        pass

    def get_hooks(self) -> List["Hook"]:
        """Return hooks this feature wants registered.

        Hooks are auto-registered with the agent's HooksManager when the
        feature is enabled, and auto-unregistered when disabled. Features
        that need hooks should override this instead of manually calling
        ``hooks_manager.register()``.

        Returns:
            List of Hook instances to register.
        """
        return []

    def get_router(self):
        """Return a FastAPI APIRouter to mount, or None.

        Features that expose HTTP endpoints can override this to return
        an APIRouter instance. The agent will include the router in the
        FastAPI app after all features are loaded.

        Returns:
            Optional APIRouter instance, or None.
        """
        return None

    @property
    def promote_tools_on_startup(self) -> bool:
        """Whether this feature's individual tools should be direct at startup.

        Most features start as a single dispatcher tool and promote their
        individual tools after first use. Features with meta-orchestration or
        agent-management tools can opt in here so startup remains generic.
        """
        return False

    async def post_all_features_loaded(self, agent):
        """Called after ALL features are discovered and initialized.

        Use this for cross-feature wiring that depends on other features
        being available. The ``agent.features`` dict is fully populated
        when this method is called.

        Args:
            agent: The KestrelAgent instance with all features loaded.
        """
        pass

    @property
    def config_schema(self) -> Optional[Dict]:
        """JSON Schema for feature configuration.

        UI can render a form from this schema. Return None if the feature
        has no user-configurable settings.
        """
        return None

    async def get_config(self) -> Dict:
        """Return the feature's current configuration."""
        return {}

    async def set_config(self, config: Dict) -> None:
        """Update the feature's configuration.

        Args:
            config: New configuration values (validated against config_schema).
        """
        pass

    # ------------------------------------------------------------------
    # Config persistence helpers
    # ------------------------------------------------------------------

    def _config_node_id(self) -> str:
        """Return the graph node ID used to persist this feature's config."""
        return f"feature_config:{self.name}"

    async def load_persisted_config(self) -> Optional[Dict]:
        """Load persisted config from agent storage (graph store).

        Returns the stored config dict, or None if nothing is persisted.
        """
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            return None
        try:
            node = await storage.get_node(self._config_node_id())
            if node is not None:
                config = node.properties.get("config")
                if isinstance(config, str):
                    config = json.loads(config)
                # Restore disabled_skills from persisted config
                disabled = config.get("disabled_skills") if config else None
                if isinstance(disabled, list):
                    self.disabled_skills = set(disabled)
                return config
        except Exception as e:
            logger.warning(f"Failed to load persisted config for {self.name}: {e}")
        return None

    async def persist_config(self, config: Dict) -> None:
        """Save config to agent storage (graph store).

        Stores the config as a graph node so it survives restarts.
        """
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            logger.debug(f"No storage available to persist config for {self.name}")
            return
        try:
            from kestrel_sovereign.storage.async_graph_store import GraphNode
            node = GraphNode(
                node_id=self._config_node_id(),
                node_type=self._CONFIG_NODE_TYPE,
                label=f"{self.name} config",
                properties={"config": config},
            )
            await storage.add_node(node)
        except Exception as e:
            logger.warning(f"Failed to persist config for {self.name}: {e}")

    # =========================================================================
    # Feature-as-Subagent Interface (A2A Pattern)
    # =========================================================================

    @property
    def tool_name(self) -> str:
        """
        Name used when this feature is called as a tool by the orchestrator.
        Converts class name to snake_case (e.g., ModelAgent -> model_agent).
        """
        # Convert CamelCase to snake_case
        name = self.name
        # Insert underscore before uppercase letters and lowercase everything
        import re
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return snake

    @property
    @abstractmethod
    def tool_description(self) -> str:
        """
        Description of what this feature/subagent can do.
        This is shown to the orchestrator LLM when selecting which feature to call.

        Example:
            "Manage LLM models - list available models, change active model, pull new models"
        """
        pass

    # =========================================================================
    # A2A Protocol Implementation
    # =========================================================================

    def get_agent_card(self) -> AgentCard:
        """
        Generate an AgentCard for this Feature.

        This allows the Feature to be discovered and called as an A2A agent.
        The AgentCard describes the Feature's capabilities (skills) to other agents.

        Uses the canonical AgentSkill attached by the @tool decorator — same
        metadata source as get_tools(), no parallel construction.
        Skills in ``self.disabled_skills`` are excluded (get_tools() already
        filters them, so the card stays in sync).
        """
        skills = []
        for tool in self.get_tools():
            if hasattr(tool, 'agent_skill') and tool.agent_skill is not None:
                skills.append(tool.agent_skill)
            else:
                # Fallback for tools without a decorator-attached AgentSkill
                schema = tool.schema
                skills.append(AgentSkill(
                    id=schema.name,
                    name=schema.name,
                    description=schema.description,
                    tags=[schema.category.value] if schema.category else None,
                    inputModes=["application/json"],
                    outputModes=["application/json"],
                    category=schema.category.value if schema.category else None,
                ))

        return AgentCard(
            name=self.tool_name,
            description=self.tool_description,
            url=f"/agents/{self.tool_name}",
            version="1.0.0",
            capabilities=AgentCapabilities(
                streaming=False,
                pushNotifications=False,
                stateTransitionHistory=False,
            ),
            skills=skills,
        )

    async def handle_task(self, task: Task) -> Task:
        """
        Handle an A2A task by routing to the appropriate skill/tool.

        This is the A2A TaskHandler implementation. When TaskManager routes
        a task to this Feature, this method:
        1. Extracts the skill name from task metadata
        2. Finds the corresponding @tool method
        3. Executes it with the provided arguments
        4. Returns the updated task with results

        Args:
            task: The A2A Task to handle

        Returns:
            Updated Task with status and artifacts
        """
        try:
            # Update task to WORKING state
            task.status = TaskStatus(state=TaskState.WORKING)

            # Extract skill and args from task metadata
            metadata = task.metadata or {}
            skill_name = metadata.get("skill")
            args = metadata.get("args", {})

            if not skill_name:
                # If no skill specified, try to infer from message
                if task.history and task.history[-1].parts:
                    for part in task.history[-1].parts:
                        if hasattr(part, 'text'):
                            # Could parse command from text here
                            pass

                # Default to first skill if only one exists
                tools = self.get_tools()
                if len(tools) == 1:
                    skill_name = tools[0].name
                else:
                    raise ValueError(f"No skill specified. Available: {[t.name for t in tools]}")

            # Find and execute the tool
            tool = self._get_tool_by_name(skill_name)
            if not tool:
                raise ValueError(f"Unknown skill: {skill_name}")

            result = await tool.execute(**args)

            # Create artifact with result
            artifact = Artifact(
                name=f"{skill_name}_result",
                parts=[DataPart(data=result)],
            )
            task.artifacts = [artifact]

            # Mark completed
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text=f"Completed {skill_name}")]
                )
            )

            return task

        except Exception as e:
            logger.error(f"Feature {self.name} task handling failed: {e}")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text=f"Error: {str(e)}")]
                )
            )
            return task

    def _get_tool_by_name(self, name: str) -> Optional[AgentTool]:
        """Get a tool by its name."""
        for tool in self.get_tools():
            if tool.name == name:
                return tool
        return None

    def get_skill_for_command(self, command: str) -> Optional[str]:
        """
        Find the skill that handles a given command prefix.

        Args:
            command: Command string like "!list-models"

        Returns:
            Skill name if found, None otherwise
        """
        for tool in self.get_tools():
            if tool.schema.command_prefix and command.startswith(tool.schema.command_prefix):
                return tool.name
        return None

    def to_orchestrator_tool(self) -> Dict[str, Any]:
        """
        Convert this feature to an orchestrator-level tool definition.

        The orchestrator sees features as high-level tools, not individual
        tool methods. Each feature gets a 'task' parameter describing what
        the orchestrator wants it to do.

        Returns:
            OpenAI function calling format tool definition
        """
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "What you want this agent to do"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional additional context from the conversation"
                        }
                    },
                    "required": ["task"]
                }
            }
        }

    async def execute_as_subagent(
        self,
        task: str,
        context: Optional[str] = None,
        max_iterations: Optional[int] = None,
        denied_tools: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Execute this feature as a subagent with its own LLM context.

        This is the A2A (Agent-to-Agent) pattern. When the orchestrator calls
        a feature as a tool, the feature:
        1. Gets its own system prompt (feature-specific)
        2. Receives the task from the orchestrator
        3. Has access to ITS OWN tools only (minus any denied by security)
        4. Makes its own LLM call(s) to decide what to do
        5. Returns results to the orchestrator

        Args:
            task: What the orchestrator wants this feature to do
            context: Optional conversation context from the orchestrator
            denied_tools: Tool names denied by security policy (stripped from palette)

        Returns:
            Dict with success status and result
        """
        try:
            # Get feature's own tools, excluding any denied by security policy
            available_tools = self.get_tools()
            if denied_tools:
                available_tools = [t for t in available_tools if t.name not in denied_tools]
                logger.info(f"Feature {self.name}: stripped {len(denied_tools)} denied tools, {len(available_tools)} remaining")

            # If ALL tools are denied, return immediately with denial
            if not available_tools and denied_tools:
                denied_list = ", ".join(sorted(denied_tools))
                return {
                    "success": False,
                    "error": f"All tools in {self.name} are blocked by security policy (denied: {denied_list}). "
                             f"The requested operation cannot be performed.",
                }

            feature_tools = [
                tool.schema.to_openai_format()
                for tool in available_tools
            ]
            logger.debug(f"Feature {self.name} has {len(feature_tools)} tools available")

            # Feature-specific system prompt
            system_prompt = self._get_subagent_prompt()

            # Build user prompt with task and context
            user_prompt = f"Task: {task}"
            if context:
                user_prompt += f"\n\nConversation context: {context}"

            logger.info(f"Feature {self.name} executing subagent task: {task[:100]}...")

            # ``llm_service.generate`` accepts ``tool_executor`` and
            # delegates to ``get_response`` which calls
            # ``messages_for(adapter)`` per-provider so the message
            # shape gets translated to each route's native format
            # (Gemini's ``parts`` + ``_system`` vs OpenAI's
            # ``role``/``content``). Using ``generate_with_messages``
            # with a hand-built OpenAI-style list would bypass that
            # translation and break Gemini/Vertex routes — codex
            # round 3 P2 on #1461 follow-up.
            tool_executor = (
                self._make_feature_inline_tool_executor()
                if feature_tools else None
            )
            response = await self.agent.llm_service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=feature_tools if feature_tools else None,
                tool_executor=tool_executor,
            )

            # Log what we got back
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_names = [tc.name for tc in response.tool_calls]
                logger.info(f"Feature {self.name} LLM called tools: {tool_names}")
            else:
                content_preview = str(response)[:100] if response else "None"
                logger.debug(f"Feature {self.name} LLM returned text (no tool calls): {content_preview}...")

            # Handle tool calls within this feature's context. Thread
            # ``tool_executor`` through so every nested generate_with_
            # messages call (continuation, repair) on the codex-routed
            # path has the executor it needs to satisfy inline tool
            # calls (codex round 1 P2 on #1461 follow-up).
            result = await self._handle_feature_tool_calls(
                response,
                feature_tools,
                system_prompt,
                max_iterations=max_iterations,
                user_prompt=user_prompt,
                tool_executor=tool_executor,
            )

            # Debug: Log what we're returning to the orchestrator
            result_preview = str(result)[:500] if result else "None"
            logger.info(f"[SUBAGENT-RESULT] Feature {self.name} returning to orchestrator: {result_preview}")

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Feature {self.name} subagent execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _make_feature_inline_tool_executor(self):
        """Build an inline ``(name, args) -> result_dict`` async callable
        bound to this feature's OWN tool palette, gated by the same
        ``PRE_TOOL_USE`` hooks the non-inline ``_handle_feature_tool_calls``
        path enforces.

        Required by adapters that execute tool calls INSIDE the LLM
        turn and block until the result arrives — the codex app-server
        (openai:plan, gpt-5.5) is the live case today. Without an
        executor, codex-routed subagent LLM calls fail at the provider
        layer with "requires a tool_executor callback when tools are
        provided", which is what hid Emma's memory_feature failures
        until the observability fix (#1461) made the error visible.

        Scoped to this feature's tools rather than the agent's global
        palette — a subagent shouldn't be able to reach for tools
        outside its own feature mid-turn. A name not in this feature's
        palette returns a structured error envelope; a PRE_TOOL_USE
        deny returns the same PERMISSION DENIED envelope the
        non-inline path produces (codex round 1 P1 on #1461 follow-up
        — without this, hook-gated policies were bypassed by the
        inline-execution path)."""
        async def _exec(name: str, args: Dict[str, Any]):
            return await self._execute_subagent_tool(
                tool_name=name,
                args=args or {},
                tools_by_name={t.name: t for t in self.get_tools()},
                return_with_effective_args=True,
            )
        return _exec

    async def _execute_subagent_tool(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
        tools_by_name: Dict[str, Any],
        return_with_effective_args: bool = False,
    ) -> Any:
        """Execute one of this feature's tools with PRE_TOOL_USE hook
        enforcement. Shared between the inline-executor path (used by
        codex app-server) and the post-LLM tool loop in
        ``_handle_feature_tool_calls`` so security policies, approval
        prompts, and argument-redaction hooks apply uniformly
        regardless of which transport ran the tool call.

        ``return_with_effective_args`` controls the return shape:

          - ``False`` (default, used by the post-LLM loop): returns
            just the ``result`` dict. The loop already knows the args.
          - ``True`` (used by the inline executor): returns the
            ``(effective_args, result)`` tuple the codex adapter
            expects. Codex round 4 P1 on #1461 follow-up — without
            this, audit / observability paths (``executed_tool_calls``,
            ``a2a_tool_dispatches.args_redacted``, persisted
            ``tool_results``) record the PRE-redaction args even
            when a hook rewrote them, leaking PII into log storage."""
        def _shape(effective_args_value: Dict[str, Any], result_value: Any) -> Any:
            """Return either the raw result or the
            ``(effective_args, result)`` tuple per the
            ``return_with_effective_args`` flag — this is what tells
            the codex adapter which args to log into
            ``executed_tool_calls`` / ``a2a_tool_dispatches``."""
            if return_with_effective_args:
                return (effective_args_value, result_value)
            return result_value

        tool = tools_by_name.get(tool_name)
        if tool is None:
            return _shape(args, {
                "success": False,
                "error": (
                    f"Tool {tool_name!r} is not in subagent "
                    f"{self.name!r}'s palette; available: "
                    f"{sorted(tools_by_name)}"
                ),
            })

        hooks_manager = getattr(self.agent, "hooks_manager", None)
        effective_args = args
        if hooks_manager is not None:
            from kestrel_sdk.hooks.base import (
                HookEvent, HookInput, PermissionDecision,
            )
            hook_input = HookInput(
                session_id="subagent",
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=tool_name,
                tool_input=args,
                feature_name=type(self).__name__,
            )
            hook_output = await hooks_manager.execute_hooks(
                HookEvent.PRE_TOOL_USE, hook_input,
            )
            # Compute the effective args from the post-hook state FIRST,
            # before the block check. A hook chain may redact via an
            # early MODIFY hook (in-place mutation of
            # ``hook_input.tool_input``) and then DENY via a later
            # PermissionHook; the blocking branch must surface the
            # REDACTED args to the codex audit path or PII the
            # redaction hook removed will leak straight into
            # ``a2a_tool_dispatches.args_redacted`` /
            # ``executed_tool_calls``. Codex round 5 P1 on #1461
            # follow-up.
            mutated_input = getattr(hook_input, "tool_input", None)
            if isinstance(mutated_input, dict):
                effective_args = mutated_input
            updated = getattr(hook_output, "updated_input", None)
            if isinstance(updated, dict):
                effective_args = updated

            # Both DENY and ASK must short-circuit. ASK means "human
            # approval required" — the orchestrator-driven path's
            # ``execute_named_tool`` blocks both, and the codex inline
            # subagent path must match that contract or approval-gated
            # tools silently run without approval (codex round 2 P1
            # on #1461 follow-up).
            if hook_output.permission_decision in (
                PermissionDecision.DENY,
                PermissionDecision.ASK,
            ):
                reason = (
                    hook_output.permission_reason
                    or "Blocked by security policy"
                )
                decision_label = (
                    "PERMISSION DENIED"
                    if hook_output.permission_decision == PermissionDecision.DENY
                    else "APPROVAL REQUIRED"
                )
                logger.info(
                    "[SUBAGENT-TOOL] %s blocked (%s): %s",
                    tool_name, decision_label, reason,
                )
                # Surface the POST-hook args even on the block path —
                # an upstream redaction hook may have run before the
                # downstream permission hook denied, and the codex
                # audit row should record the redacted form, not the
                # raw PII (codex round 5 P1 on #1461 follow-up).
                return _shape(effective_args, {
                    "success": False,
                    "error": (
                        f"{decision_label}: {reason}. The tool was "
                        f"NOT executed. Do NOT tell the user this "
                        f"action succeeded — inform them it was "
                        f"blocked by security policy."
                    ),
                })
            # ``effective_args`` was already resolved above to the
            # post-hook state (mutated ``tool_input`` first, then any
            # ``updated_input`` override) — see the comment block
            # before the DENY/ASK branch.

        try:
            result = await tool.execute(**effective_args)
            return _shape(effective_args, _serialize_tool_result(result))
        except Exception as e:
            logger.warning(
                "[SUBAGENT-TOOL] %s raised %s",
                tool_name, e,
            )
            return _shape(
                effective_args,
                {"success": False, "error": f"{type(e).__name__}: {e}"},
            )

    def _get_subagent_prompt(self) -> str:
        """
        Get the system prompt for this feature's subagent context.

        Override this in subclasses for more specialized prompts.
        """
        tool_names = [t.name for t in self.get_tools()]
        tools_list = ", ".join(tool_names) if tool_names else "None"

        return f"""EXECUTION MODE: You are now executing as the {self.name} subagent.

You have been invoked to perform a specific task. DO NOT engage in conversation.
DO NOT ask clarifying questions. DO NOT respond with greetings or pleasantries.
DO NOT say you are "awaiting task input" - you already have a task.
EXECUTE THE TASK IMMEDIATELY using your tools.

Your capabilities: {self.tool_description}
Available tools: {tools_list}

EXECUTION PROTOCOL:
1. The task is in the next message - execute it immediately
2. Call the appropriate tool(s) to complete the task
3. If a tool fails or returns no results, report what you tried and what happened
4. Do NOT ask for more input - summarize results and complete your response
5. Use function calling to invoke tools - do not describe actions, DO THEM
6. If multiple tools are needed, call them in sequence
7. After getting tool results (success or failure), provide a brief summary

CRITICAL: You have ONE task. Execute it now. Do not wait for more input.

ABSOLUTE PROHIBITION - NEVER FABRICATE:
- NEVER invent a CID, hash, transaction ID, wallet address, or any cryptographic value
- NEVER generate a plausible-looking result without actually calling a tool
- If a tool call fails or is not available, say so explicitly - do not fill in fake values
- A fabricated cryptographic value is a lie and a constitutional violation"""

    async def _handle_feature_tool_calls(
        self,
        response: Union[str, Any],
        tools: List[Dict[str, Any]],
        system_prompt: str,
        max_iterations: int = None,
        user_prompt: Optional[str] = None,
        tool_executor: Optional[Any] = None,
    ) -> str:
        """
        Handle tool calls within this feature's context.

        This is similar to the orchestrator's tool handling but scoped to
        this feature's tools only.

        Args:
            response: Initial LLM response (string or message with tool_calls)
            tools: This feature's tools in OpenAI format
            system_prompt: The feature's system prompt for continuation
            max_iterations: Maximum tool call iterations to prevent infinite loops.
                           Defaults to KESTREL_MAX_TOOL_ITERATIONS env var (default: 5)

        Returns:
            Final text response after all tool calls are processed
        """
        # Use module constant if not explicitly specified
        if max_iterations is None:
            max_iterations = MAX_TOOL_ITERATIONS

        # If response is just a string, return it directly
        if isinstance(response, str):
            return response

        # Build message history for multi-turn tool calling
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        # Check if response has tool_calls attribute
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            response = await self._repair_subagent_premature_yield(
                response,
                messages,
                tools,
                tool_executor=tool_executor,
            )
            if isinstance(response, str):
                return response
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                return response.content or ""

        messages.append(self._build_subagent_assistant_tool_history_msg(response))

        # Get tools by name for execution
        tools_by_name = {tool.name: tool for tool in self.get_tools()}

        for iteration in range(max_iterations):
            # Warn when approaching iteration limit
            if iteration >= max_iterations * 0.8:  # 80% threshold
                logger.warning(f"[SUBAGENT {self.name}] Approaching max iterations: {iteration + 1}/{max_iterations}")
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call.name

                # arguments is already a dict from our ToolCall dataclass
                if isinstance(tool_call.arguments, dict):
                    args = tool_call.arguments
                else:
                    try:
                        args = json.loads(tool_call.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                # Execute through the shared hook-enforced helper so
                # this loop AND the inline-executor path apply
                # identical PRE_TOOL_USE policy. Without this the
                # codex app-server inline path would bypass security
                # hooks the orchestrator-driven path enforces (codex
                # round 1 P1 on #1461 follow-up).
                result = await self._execute_subagent_tool(
                    tool_name=tool_name,
                    args=args,
                    tools_by_name=tools_by_name,
                )
                result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                logger.info(
                    f"[SUBAGENT-TOOL] {tool_name} result ({len(result_str)} chars): {result_str[:300]}..."
                )

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            # Continue conversation with tool results — thread the
            # ``tool_executor`` through so codex-routed continuation
            # turns don't hit the same "requires a tool_executor"
            # provider error the initial subagent call avoided
            # (codex round 1 P2 on #1461 follow-up).
            response = await self.agent.llm_service.generate_with_messages(
                messages=messages,
                tools=tools if tools else None,
                tool_executor=tool_executor,
            )

            # If response is string or has no more tool calls, we're done
            if isinstance(response, str):
                return response

            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                response = await self._repair_subagent_premature_yield(
                    response,
                    messages,
                    tools,
                    tool_executor=tool_executor,
                )
                if isinstance(response, str):
                    return response
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    messages.append(self._build_subagent_assistant_tool_history_msg(response))
                    continue
                return response.content or ""

            # Add assistant response with new tool calls to messages
            messages.append(self._build_subagent_assistant_tool_history_msg(response))

        return "Error: Maximum tool call iterations exceeded"

    # =========================================================================
    # Tool Discovery
    # =========================================================================

    def get_tools(self) -> List[AgentTool]:
        """
        Auto-discover methods decorated with @tool and return them as AgentTool instances.

        Each returned tool carries the canonical AgentSkill created by the @tool
        decorator, so get_agent_card() can reuse it without rebuilding metadata.

        Tools whose names appear in ``self.disabled_skills`` are excluded.
        """
        tools = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_tool_schema"):
                schema_data = method._tool_schema

                # Skip disabled skills
                if schema_data["name"] in self.disabled_skills:
                    continue

                agent_skill = getattr(method, "_agent_skill", None)

                # Create a dynamic AgentTool wrapper
                class DynamicTool(AgentTool):
                    def __init__(self, func, schema_data, agent_skill):
                        self.func = func
                        self._schema_data = schema_data
                        self.agent_skill = agent_skill

                    @property
                    def name(self) -> str:
                        return self._schema_data["name"]

                    @property
                    def schema(self) -> ToolSchema:
                        return ToolSchema(
                            name=self._schema_data["name"],
                            description=self._schema_data["description"],
                            category=self._schema_data["category"],
                            parameters=self._schema_data["parameters"],
                            command_prefix=self._schema_data.get("command_prefix")
                        )

                    async def execute(self, **kwargs) -> Dict[str, Any]:
                        try:
                            result = await self.func(**kwargs)
                        except Exception as e:
                            logger.error(f"Error executing tool {self.name}: {e}")
                            return {
                                "success": False,
                                "error": str(e),
                                "tool": self.name,
                            }

                        # ToolResult-returning @tool methods (#1042
                        # layer 4 contract) get serialized at the wrap
                        # site so downstream readers never see the raw
                        # frozen-dataclass instance. Without this,
                        # in-process callers (run_workflow,
                        # check_task_status) end up with a non-JSON-
                        # serializable object embedded in the wire
                        # payload — the workaround was per-callsite
                        # ``_serialize_step_payload`` helpers in
                        # PR-E pilot (#1066) which this fix
                        # supersedes. See #1070.
                        #
                        # Honesty: the wrapper's ``success`` flag
                        # historically meant "the call did not raise."
                        # That conflated transport with semantic
                        # outcome — a migrated tool returning
                        # ``ToolResult.failed`` would still surface
                        # ``success: True`` to callers like
                        # ``command_handler`` that branch on it.
                        # We now derive ``success`` from the
                        # ToolResult status:
                        #   - OK → success=True
                        #   - PARTIAL → success=True (it succeeded
                        #     enough to produce a confirmation; the
                        #     ``error`` field is also populated so
                        #     callers that surface both still get the
                        #     full picture)
                        #   - ERROR → success=False, error copied
                        #     into the wrapper's top-level error
                        if isinstance(result, ToolResult):
                            wire = result.to_dict()
                            response: Dict[str, Any] = {
                                "result": wire,
                                "tool": self.name,
                            }
                            status = result.status
                            if status is ToolResultStatus.ERROR:
                                response["success"] = False
                                response["error"] = result.error
                            else:
                                # OK and PARTIAL both ran the action.
                                response["success"] = True
                                if status is ToolResultStatus.PARTIAL:
                                    # Surface the partial caveat at
                                    # the wrapper level so legacy
                                    # callers that only read ``error``
                                    # don't miss it.
                                    response["error"] = result.error
                            return response

                        # Pre-migration return shape (Dict[str, Any]
                        # or other) — keep the original wrapper. The
                        # ``success: True`` here remains transport-
                        # level for un-migrated tools; the #1061
                        # bulk waves migrate them away one by one.
                        return {
                            "success": True,
                            "result": result,
                            "tool": self.name,
                        }

                tools.append(DynamicTool(method, schema_data, agent_skill))
        return tools

def tool(name: str, description: str, category: ToolCategory = ToolCategory.SYSTEM, command_prefix: str = None):
    """
    Decorator to mark a method as an agent tool.
    The method's signature is inspected to generate parameters.
    Parameter descriptions are extracted from the function's docstring.
    
    Supports docstring formats:
    - Google style: `param_name: Description here`
    - Sphinx style: `:param param_name: Description here`
    
    Example:
        @tool("my_tool", "Does something useful")
        async def my_tool(self, file_path: str, count: int = 10):
            '''
            Do something with a file.
            
            Args:
                file_path: The path to the file to process
                count: Number of items to process (default: 10)
            '''
            pass
    """
    def decorator(func):
        # Parse docstring for parameter descriptions
        docstring = func.__doc__
        param_descriptions = parse_docstring_params(docstring)
        
        # Inspect signature to build parameters
        sig = inspect.signature(func)
        parameters = []
        
        type_map = {
            str: "string",
            int: "integer",
            bool: "boolean",
            float: "number",
            list: "array",
            dict: "object"
        }
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # Handle typing generics
            from typing import get_origin, get_args
            origin = get_origin(param.annotation)
            items_schema = None
            if origin is not None:
                param_type = type_map.get(origin, "string")
                # For List[X], derive items schema from the type argument
                if param_type == "array":
                    type_args = get_args(param.annotation)
                    if type_args:
                        inner = type_args[0]
                        inner_type = type_map.get(inner, None)
                        if inner_type:
                            items_schema = {"type": inner_type}
            else:
                param_type = type_map.get(param.annotation, "string")
            required = param.default == inspect.Parameter.empty

            # Get description from parsed docstring, fallback to placeholder
            param_desc = param_descriptions.get(
                param_name,
                f"The {param_name.replace('_', ' ')} parameter"
            )

            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=param_desc,
                required=required,
                default=None if required else param.default,
                items=items_schema,
            ))
            
        func._tool_schema = {
            "name": name,
            "description": description,
            "category": category,
            "parameters": parameters,
            "command_prefix": command_prefix
        }

        # Also create AgentSkill metadata for A2A protocol — single source of truth
        func._agent_skill = AgentSkill(
            id=name,
            name=name,
            description=description,
            tags=[category.value] if category else None,
            inputModes=["application/json"],
            outputModes=["application/json"],
            category=category.value if category else None,
        )

        return func
    return decorator
