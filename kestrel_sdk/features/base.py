"""
Base class for Kestrel Features — SDK interface.

This module contains the Feature ABC, tool decorator, and supporting types
that feature packages need. Implementation details (subagent execution,
LLM calls, hook enforcement) live in kestrel_sovereign.features.base.

Feature packages should import from here:
    from kestrel_sdk.features.base import Feature, tool, parse_docstring_params
"""

import inspect
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable, TYPE_CHECKING

from kestrel_sdk.tools.base import ToolSchema, ToolParameter, ToolCategory, AgentTool
from kestrel_sdk.a2a.agent_card import AgentCard, AgentSkill, AgentCapabilities
from kestrel_sdk.a2a.types import Task, TaskState, TaskStatus, Artifact, DataPart, Message, TextPart

logger = logging.getLogger(__name__)


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
    - NumPy style: `param_name : type\\n    Description here`

    Args:
        docstring: The docstring to parse

    Returns:
        Dict mapping parameter names to their descriptions
    """
    if not docstring:
        return {}

    param_descriptions = {}

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


class Feature(ABC):
    """
    Base class for Kestrel Features - each Feature IS a subagent.

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

    def get_hooks(self) -> List:
        """Return hooks this feature wants registered.

        Returns:
            List of Hook instances to register.
        """
        return []

    def get_router(self):
        """Return a FastAPI APIRouter to mount, or None."""
        return None

    async def post_all_features_loaded(self, agent):
        """Called after ALL features are discovered and initialized."""
        pass

    @property
    def config_schema(self) -> Optional[Dict]:
        """JSON Schema for feature configuration."""
        return None

    async def get_config(self) -> Dict:
        """Return the feature's current configuration."""
        return {}

    async def set_config(self, config: Dict) -> None:
        """Update the feature's configuration."""
        pass

    # =========================================================================
    # Feature-as-Subagent Interface (A2A Pattern)
    # =========================================================================

    @property
    def tool_name(self) -> str:
        """Name used when this feature is called as a tool by the orchestrator."""
        name = self.name
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return snake

    @property
    @abstractmethod
    def tool_description(self) -> str:
        """Description of what this feature/subagent can do."""
        pass

    # =========================================================================
    # A2A Protocol Implementation
    # =========================================================================

    def get_agent_card(self) -> AgentCard:
        """Generate an AgentCard for this Feature."""
        skills = []
        for tool in self.get_tools():
            if hasattr(tool, 'agent_skill') and tool.agent_skill is not None:
                skills.append(tool.agent_skill)
            else:
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

    def get_skill_for_command(self, command: str) -> Optional[str]:
        """Find the skill that handles a given command prefix."""
        for tool in self.get_tools():
            if tool.schema.command_prefix and command.startswith(tool.schema.command_prefix):
                return tool.name
        return None

    def to_orchestrator_tool(self) -> Dict[str, Any]:
        """Convert this feature to an orchestrator-level tool definition."""
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

    # =========================================================================
    # Tool Discovery
    # =========================================================================

    def get_tools(self) -> List[AgentTool]:
        """
        Auto-discover methods decorated with @tool and return them as AgentTool instances.

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
                            return {
                                "success": True,
                                "result": result,
                                "tool": self.name
                            }
                        except Exception as e:
                            logger.error(f"Error executing tool {self.name}: {e}")
                            return {
                                "success": False,
                                "error": str(e),
                                "tool": self.name
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
