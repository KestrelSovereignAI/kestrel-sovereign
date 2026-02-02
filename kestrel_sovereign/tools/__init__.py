"""
Kestrel Agent Tools

This package provides the tool system that enables agents to autonomously
select and execute capabilities based on context and user needs.

Note: ToolRegistry has been deprecated in favor of A2A TaskManager.
Features register directly with TaskManager via get_agent_card() and handle_task().
"""

# Tool base classes for Features
from .base import (
    AgentTool,
    ToolSchema,
    ToolParameter,
    ToolCategory,
    ToolExecutionError
)

__all__ = [
    "AgentTool",
    "ToolSchema",
    "ToolParameter",
    "ToolCategory",
    "ToolExecutionError",
]
