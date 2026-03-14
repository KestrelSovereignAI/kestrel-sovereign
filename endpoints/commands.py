"""Commands discovery endpoint - exposes available commands to UI."""
from fastapi import APIRouter, Request
from typing import List, Dict, Any
import logging

from endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["commands"])


# Built-in commands defined in CommandHandler
# These are agent-level commands not in features
BUILTIN_COMMANDS = [
    # System
    {"cmd": "!help", "description": "Show available commands", "category": "System"},
    {"cmd": "!status", "description": "Show agent status", "category": "System"},
    {"cmd": "!audit", "description": "Toggle or check audit status", "args": "[on|off]", "category": "System"},

    # Constitution
    {"cmd": "!verify-constitution", "description": "Verify constitution integrity", "category": "Constitution"},
    {"cmd": "!safe-mode", "description": "Check or exit safe mode", "args": "[exit]", "category": "Constitution"},

    # Privacy
    {"cmd": "!privacy", "description": "Get or set privacy mode", "args": "[mode]", "category": "Privacy"},
    {"cmd": "!set-privacy-mode", "description": "Set privacy mode", "args": "<mode>", "category": "Privacy"},
    {"cmd": "!get-privacy-mode", "description": "Get current privacy mode", "category": "Privacy"},
    {"cmd": "!privacy-status", "description": "Detailed privacy status", "category": "Privacy"},
    {"cmd": "!privacy-save", "description": "Save isolated session", "category": "Privacy"},
    {"cmd": "!privacy-discard", "description": "Discard isolated session", "category": "Privacy"},

    # Backup
    {"cmd": "!backup", "description": "Create a backup", "args": "[--tier local|ipfs|filecoin]", "category": "Backup"},
    {"cmd": "!promote-backup", "description": "Save isolated session and backup", "args": "[--tier ...]", "category": "Backup"},

    # Model (agent-level preference, not ModelFeature)
    {"cmd": "!model", "description": "Show current model", "category": "Model"},
    {"cmd": "!model-set", "description": "Set active model", "args": "<provider/model>", "category": "Model"},

    # Agent
    {"cmd": "!create-agent", "description": "Create trusted agent", "args": "<name>", "category": "Agent"},
    {"cmd": "!anchor", "description": "Anchor memory state to blockchain", "category": "Agent"},

    # Tasks
    {"cmd": "!tasks", "description": "List background tasks", "args": "[all|completed|working|failed]", "category": "Tasks"},
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
