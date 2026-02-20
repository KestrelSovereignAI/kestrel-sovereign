"""
Base classes and interfaces for Kestrel agent tools.

This module defines the core abstractions for the tool system, enabling agents
to autonomously select and execute capabilities based on context and user needs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolCategory(Enum):
    """Categories of agent tools for organization and filtering."""
    MODEL_MANAGEMENT = "model_management"
    FILE_OPERATIONS = "file_operations"
    WEB_SEARCH = "web_search"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    DATA_ACCESS = "data_access"
    COMPUTE = "compute"
    UTILITY = "utility"  # Task monitoring, status checks, background operations


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[str]] = None
    items: Optional[Dict[str, Any]] = None  # JSON Schema for array element type


@dataclass
class ToolSchema:
    """
    Schema defining a tool's capabilities and interface.

    This schema is used by the agent to understand when and how to use a tool,
    enabling autonomous tool selection based on context.
    """
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    command_prefix: Optional[str] = None  # e.g., "!list-models"

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum
                }
                for p in self.parameters
            ],
            "examples": self.examples,
            "command_prefix": self.command_prefix
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert schema to OpenAI function calling format.

        Returns a tool definition compatible with OpenAI's chat completions API
        tools parameter format.
        """
        # Build properties dict for parameters
        properties = {}
        required = []

        for param in self.parameters:
            prop_def: Dict[str, Any] = {
                "type": param.type,
                "description": param.description
            }

            # Add items schema for array types
            if param.type == "array":
                if param.items:
                    prop_def["items"] = param.items
                else:
                    prop_def["items"] = {"type": "object"}

            # Add enum if present
            if param.enum:
                prop_def["enum"] = param.enum

            # Add default if present and not required
            if param.default is not None and not param.required:
                prop_def["default"] = param.default

            properties[param.name] = prop_def

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class AgentTool(ABC):
    """
    Base class for all agent tools.

    Tools are capabilities that agents can autonomously select and execute
    based on context, user requests, or their own decision-making.

    Example:
        class MyTool(AgentTool):
            @property
            def name(self) -> str:
                return "my_tool"

            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="my_tool",
                    description="Does something useful",
                    category=ToolCategory.SYSTEM,
                    parameters=[
                        ToolParameter(name="arg1", type="string", description="First arg", required=True)
                    ]
                )

            async def execute(self, **kwargs) -> Dict[str, Any]:
                result = await self._do_work(kwargs["arg1"])
                return {"success": True, "result": result}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        pass

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Schema defining this tool's capabilities and interface."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters as defined in schema

        Returns:
            Dict with at minimum:
                - success: bool indicating if execution succeeded
                - result: Any data returned by the tool
                - error: Optional error message if success=False
        """
        pass

    def can_handle_command(self, user_input: str) -> bool:
        """
        Check if this tool can handle a user command.

        Args:
            user_input: Raw user input string

        Returns:
            True if this tool should handle the command
        """
        if not self.schema.command_prefix:
            return False

        return user_input.strip().startswith(self.schema.command_prefix)

    def parse_command_args(self, user_input: str) -> Dict[str, Any]:
        """
        Parse command-line style arguments from user input.
        Handles JSON objects/arrays and quoted strings for complex parameters.
        Performs type coercion based on parameter schema.
        """
        import json

        parts = user_input.strip().split()

        # First part is the command itself
        if not parts or not parts[0].startswith("!"):
            return {}

        # Determine how many words the command prefix has (e.g., "!memory search" = 2)
        prefix = self.schema.command_prefix or parts[0]
        prefix_word_count = len(prefix.split())

        args = {}
        current_part_idx = prefix_word_count

        for param in self.schema.parameters:
            if current_part_idx >= len(parts):
                if param.default is not None:
                    args[param.name] = param.default
                continue

            # If this is an object/array, consume the rest
            if param.type in ["object", "array"]:
                # Join the rest of the parts
                value_str = " ".join(parts[current_part_idx:])

                try:
                    args[param.name] = json.loads(value_str)
                except json.JSONDecodeError:
                    # Fallback to string if parsing fails (might be intended as string)
                    args[param.name] = value_str

                # We consumed everything for this complex type
                # (Assuming complex types are usually last or we don't support multiple complex types well yet)
                break
            elif param == self.schema.parameters[-1]:
                 # Last parameter consumes the rest if it's a string
                value = " ".join(parts[current_part_idx:])
                args[param.name] = self._coerce_type(value, param.type)
                break
            else:
                # Simple positional argument - coerce to expected type
                value = parts[current_part_idx]
                args[param.name] = self._coerce_type(value, param.type)
                current_part_idx += 1

        return args

    def _coerce_type(self, value: str, param_type: str) -> Any:
        """
        Coerce a string value to the expected type based on schema.

        Args:
            value: The string value from command parsing
            param_type: The expected type from parameter schema

        Returns:
            The coerced value, or original string if coercion fails
        """
        if param_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type == "number":
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        elif param_type == "boolean":
            if value.lower() in ("true", "1", "yes", "on"):
                return True
            elif value.lower() in ("false", "0", "no", "off"):
                return False
            return value
        # Default: return as string
        return value

    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against schema.

        Args:
            **kwargs: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        for param in self.schema.parameters:
            if param.required and param.name not in kwargs:
                return False, f"Required parameter '{param.name}' is missing"

            if param.name in kwargs:
                value = kwargs[param.name]

                # Type validation
                expected_type = param.type
                if expected_type == "string" and not isinstance(value, str):
                    return False, f"Parameter '{param.name}' must be a string"
                elif expected_type == "integer" and not isinstance(value, int):
                    return False, f"Parameter '{param.name}' must be an integer"
                elif expected_type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter '{param.name}' must be a boolean"

                # Enum validation
                if param.enum and value not in param.enum:
                    return False, f"Parameter '{param.name}' must be one of: {', '.join(param.enum)}"

        return True, None


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str, original_error: Optional[Exception] = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {message}")
