"""Commands discovery endpoint - exposes available commands to UI."""
from fastapi import APIRouter, Request
from typing import List, Dict, Any
import logging

from endpoints.agent_helpers import get_agent
from kestrel_sovereign.command_handler import BUILTIN_COMMAND_SPECS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["commands"])


BUILTIN_COMMANDS = [
    {
        key: value
        for key, value in spec.items()
        if key != "handler"
    }
    for spec in BUILTIN_COMMAND_SPECS
]


@router.get("/api/commands")
async def get_commands(request: Request) -> Dict[str, Any]:
    """
    Return all available commands from built-ins and loaded features.

    This endpoint dynamically discovers commands from:
    1. Built-in commands in CommandHandler
    2. Feature commands via @tool decorator with command_prefix
    """
    commands = []

    # Add built-in commands
    commands.extend(BUILTIN_COMMANDS)

    # Add feature commands from loaded features
    try:
        agent = get_agent(request)
    except Exception:
        return {
            "commands": commands,
            "count": len(commands),
        }
    if hasattr(agent, 'features'):
        for feature_name, feature in agent.features.items():
            try:
                tools = feature.get_tools() if hasattr(feature, 'get_tools') else []
                for tool in tools:
                    schema = getattr(tool, 'schema', None)
                    if schema and getattr(schema, 'command_prefix', None):
                        # Build args string from parameters
                        args = getattr(schema, 'parameters_summary', None)

                        commands.append({
                            "cmd": schema.command_prefix,
                            "description": schema.description or f"{tool.name} tool",
                            "args": args,
                            "category": feature_name,
                            "feature": feature_name,
                        })
            except Exception as e:
                logger.warning(f"Error getting tools from feature {feature_name}: {e}")

    # Sort by category then command name
    commands.sort(key=lambda c: (c.get("category", ""), c.get("cmd", "")))

    return {
        "commands": commands,
        "count": len(commands),
    }
