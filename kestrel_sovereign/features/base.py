import inspect
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Type, Union, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from kestrel_sovereign.tools.base import ToolSchema, ToolParameter, ToolCategory, AgentTool
from kestrel_sovereign.a2a.agent_card import AgentCard, AgentSkill, AgentCapabilities
from kestrel_sovereign.a2a.types import Task, TaskState, TaskStatus, Artifact, DataPart, Message, TextPart

logger = logging.getLogger(__name__)

# Maximum tool call iterations (configurable via environment variable)
# Increased to 50 for long-running tasks like code analysis and multi-step operations
MAX_TOOL_ITERATIONS = int(os.environ.get("KESTREL_MAX_TOOL_ITERATIONS", "50"))


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

class Feature(ABC):
    """
    Base class for Kestrel Features - each Feature IS a subagent.

    A Feature encapsulates a specific domain of functionality (e.g., Sovereignty, MCP, Models).
    It can expose methods as Tools to the agent, and can be called AS a tool by the orchestrator
    with its own LLM context (A2A pattern).
    """

    def __init__(self, agent):
        self.agent = agent
        self.name = self.__class__.__name__

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
        """
        skills = []
        for tool in self.get_tools():
            schema = tool.schema
            skill = AgentSkill(
                id=schema.name,
                name=schema.name,
                description=schema.description,
                tags=[schema.category.value] if schema.category else None,
                inputModes=["application/json"],
                outputModes=["application/json"],
            )
            skills.append(skill)

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

            # Make LLM call with feature's tools
            response = await self.agent.llm_service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=feature_tools if feature_tools else None
            )

            # Log what we got back
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_names = [tc.name for tc in response.tool_calls]
                logger.info(f"Feature {self.name} LLM called tools: {tool_names}")
            else:
                content_preview = str(response)[:100] if response else "None"
                logger.debug(f"Feature {self.name} LLM returned text (no tool calls): {content_preview}...")

            # Handle tool calls within this feature's context
            result = await self._handle_feature_tool_calls(
                response,
                feature_tools,
                system_prompt,
                max_iterations=max_iterations
            )

            # Debug: Log what we're returning to the orchestrator
            result_preview = str(result)[:500] if result else "None"
            logger.info(f"[SUBAGENT-RESULT] Feature {self.name} returning to orchestrator: {result_preview}")

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Feature {self.name} subagent execution failed: {e}")
            return {"success": False, "error": str(e)}

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
        max_iterations: int = None
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

        # Check if response has tool_calls attribute
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            return response.content or ""

        # Build message history for multi-turn tool calling
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": response.content, "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments
                    }
                }
                for tc in response.tool_calls
            ]}
        ]

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

                # Find and execute the tool (with security hook enforcement)
                tool = tools_by_name.get(tool_name)
                if tool:
                    # Check security hooks before executing
                    hooks_manager = getattr(self.agent, 'hooks_manager', None)
                    if hooks_manager:
                        from kestrel_sovereign.hooks import HookInput, HookEvent
                        from kestrel_sovereign.hooks.base import PermissionDecision
                        hook_input = HookInput(
                            session_id="subagent",
                            hook_event_name=HookEvent.PRE_TOOL_USE.value,
                            tool_name=tool_name,
                            tool_input=args,
                            feature_name=type(self).__name__,
                        )
                        hook_output = await hooks_manager.execute_hooks(
                            HookEvent.PRE_TOOL_USE, hook_input
                        )
                        if hook_output.permission_decision == PermissionDecision.DENY:
                            reason = hook_output.permission_reason or "Blocked by security policy"
                            result = {
                                "success": False,
                                "error": f"PERMISSION DENIED: {reason}. The tool was NOT executed. Do NOT tell the user this action succeeded — inform them it was blocked by security policy."
                            }
                            logger.info(f"[SUBAGENT-TOOL] {tool_name} blocked by security: {reason}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                            continue

                    try:
                        result = await tool.execute(**args)
                        result = _serialize_tool_result(result)
                        # Debug: Log tool result
                        result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                        logger.info(f"[SUBAGENT-TOOL] {tool_name} result ({len(result_str)} chars): {result_str[:300]}...")
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
                        logger.error(f"[SUBAGENT-TOOL] {tool_name} failed: {e}")
                else:
                    result = {"success": False, "error": f"Unknown tool: {tool_name}"}
                    logger.warning(f"[SUBAGENT-TOOL] Unknown tool: {tool_name}")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            # Continue conversation with tool results
            response = await self.agent.llm_service.generate_with_messages(
                messages=messages,
                tools=tools if tools else None
            )

            # If response is string or has no more tool calls, we're done
            if isinstance(response, str):
                return response

            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                return response.content or ""

            # Add assistant response with new tool calls to messages
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments
                        }
                    }
                    for tc in response.tool_calls
                ]
            })

        return "Error: Maximum tool call iterations exceeded"

    # =========================================================================
    # Tool Discovery
    # =========================================================================

    def get_tools(self) -> List[AgentTool]:
        """
        Auto-discover methods decorated with @tool and return them as AgentTool instances.
        """
        tools = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_tool_schema"):
                schema_data = method._tool_schema
                
                # Create a dynamic AgentTool wrapper
                class DynamicTool(AgentTool):
                    def __init__(self, func, schema_data):
                        self.func = func
                        self._schema_data = schema_data
                        
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
                            
                tools.append(DynamicTool(method, schema_data))
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

        # Also create AgentSkill metadata for A2A protocol
        func._agent_skill = AgentSkill(
            id=name,
            name=name,
            description=description,
            tags=[category.value] if category else None,
            inputModes=["application/json"],
            outputModes=["application/json"],
        )

        return func
    return decorator
