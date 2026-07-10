"""Fleet coordination: sovereign-side agent definitions (#2321).

The :mod:`kestrel_sovereign.fleet.orchestrator` module defines the **Fleet
Orchestrator** agent — a governed child of the Sovereign that dispatches coding
work exclusively by starting ``fleet_coding_pipeline`` workflow runs (#2303) and
never edits code, approves its own consent gates, or invokes talon dispatch
tools directly.
"""

from kestrel_sovereign.fleet.orchestrator import (
    FLEET_ORCHESTRATOR_NAME,
    FLEET_ORCHESTRATOR_SLUG,
    FEATURE_ALLOWLIST,
    TOOL_ALLOWLIST,
    RESTRICTED_TOOLS,
    RESTRICTED_TOOL_ARGS,
    REFLECTION_SCHEDULE,
    additional_constraints,
    build_local_agent_config,
    build_restriction_hook,
    build_scoped_constitution,
    build_spawn_mandate,
    constitution_text,
    is_tool_allowed,
    is_tool_call_allowed,
    is_tool_denied,
)

__all__ = [
    "FLEET_ORCHESTRATOR_NAME",
    "FLEET_ORCHESTRATOR_SLUG",
    "FEATURE_ALLOWLIST",
    "TOOL_ALLOWLIST",
    "RESTRICTED_TOOLS",
    "RESTRICTED_TOOL_ARGS",
    "REFLECTION_SCHEDULE",
    "additional_constraints",
    "build_local_agent_config",
    "build_restriction_hook",
    "build_scoped_constitution",
    "build_spawn_mandate",
    "constitution_text",
    "is_tool_allowed",
    "is_tool_call_allowed",
    "is_tool_denied",
]
