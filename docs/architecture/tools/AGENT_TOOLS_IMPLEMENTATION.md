# Agent Tools Implementation

**Status:** Active implementation map
**Last updated:** 2026-05-14
**Related:** [Agent Tools Architecture](AGENT_TOOLS_ARCHITECTURE.md), [Building Features](../../guides/BUILDING_FEATURES.md), [SDK README](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk#readme)

This document maps the modern tool system to the files on `main`. It is not an
installation checklist for a separate tool layer. Tools are implemented inside
features and discovered by the runtime.

## Core Files

| Area | File | Responsibility |
|---|---|---|
| Feature base and tool wrapping | `kestrel_sovereign/features/base.py` | `Feature`, lifecycle methods, `get_tools()`, `DynamicTool`, `tool()` compatibility export |
| Feature discovery | `kestrel_sovereign/features/__init__.py` | local feature discovery plus `kestrel_sovereign.features` entry points |
| Runtime tool catalog | `kestrel_sovereign/agent/tool_registry.py` | feature dispatcher tools, direct-tool promotion, hiding, eviction |
| Result contract validation | `kestrel_sovereign/tools/result_contract.py` | scoped validator for migrated `@tool -> ToolResult` modules |
| Command bridge | `kestrel_sovereign/command_handler.py` | command-prefix execution and `ToolResult` envelope rendering |
| Honesty prompt | `kestrel_sovereign/security/input_guardrails.py` | `TOOL_HONESTY_SYSTEM_PROMPT` and prompt assembly |
| Narration check | `kestrel_sovereign/security/narration_check.py` | deterministic pre-tool prose vs tool-result analysis |
| Response audit hook | `kestrel_sovereign/features/response_audit/` | hook that applies audit and narration-risk checks |

The SDK owns the shared data structures imported by these files:

- `kestrel_sdk.features.base.Feature`
- `kestrel_sdk.features.base.tool`
- `kestrel_sdk.tools.base.AgentTool`
- `kestrel_sdk.tools.base.ToolCategory`
- `kestrel_sdk.tools.base.ToolParameter`
- `kestrel_sdk.tools.base.ToolSchema`
- `kestrel_sdk.tools.result.ToolResult`
- `kestrel_sdk.tools.result.ToolResultStatus`

## Writing a Tool

A tool is an async feature method decorated with `@tool`. The method signature
and docstring become the JSON schema shown to the model.

```python
from kestrel_sdk.features.base import tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature


class NotesFeature(Feature):
    @property
    def tool_description(self) -> str:
        return "Create and inspect short notes"

    async def initialize(self):
        self.notes = []

    @tool(
        name="note_add",
        description="Save a short note",
        category=ToolCategory.MEMORY,
        command_prefix="!note-add",
    )
    async def note_add(self, text: str) -> ToolResult:
        """Save a short note.

        Args:
            text: Note text to save.
        """
        self.notes.append(text)
        return ToolResult.ok(
            confirmation="Saved note.",
            data={"count": len(self.notes)},
        )
```

Use `ToolResult.ok()` when the action succeeded, `ToolResult.partial()` when
there is a usable result with a caveat, and `ToolResult.failed()` when the tool
could not complete the requested action.

## Registration and Execution

`Feature.get_tools()` discovers decorated methods on a feature instance. For
each method it creates a `DynamicTool` wrapper with:

- `name` from the decorator
- `schema` from SDK `ToolSchema`
- `execute(**kwargs)` that calls the feature method
- optional A2A `AgentSkill` metadata

When a method returns `ToolResult`, `DynamicTool.execute()` serializes it with
`to_dict()` and returns a JSON-safe wrapper:

```python
{
    "success": true,
    "result": {
        "status": "ok",
        "confirmation": "Saved note.",
        "error": null,
        "data": {"count": 1}
    },
    "tool": "note_add"
}
```

For `ToolResultStatus.ERROR`, the wrapper sets top-level `success` to `False`
and copies the error to the top level for older callers. For
`ToolResultStatus.PARTIAL`, the wrapper keeps `success` true but also surfaces
the caveat in `error`.

## Discovery Sources

Core features are discovered from `kestrel_sovereign/features/`.

External feature packages register here:

```toml
[project.entry-points."kestrel_sovereign.features"]
notes = "kestrel_feature_notes.feature:NotesFeature"
```

Provider packages register in provider-specific groups when they extend a
registry rather than shipping a whole feature:

```toml
[project.entry-points."kestrel_sovereign.cloud_providers"]
[project.entry-points."kestrel_sovereign.voice_providers"]
[project.entry-points."kestrel_sovereign.storage_providers"]
```

The framework's own `pyproject.toml` declares these groups so package managers
and feature authors have stable targets.

## ToolResult Migration Guard

`kestrel_sovereign/tools/result_contract.py` contains
`MIGRATED_FEATURE_MODULES`, a scoped allowlist of modules whose `@tool` methods
must annotate `-> ToolResult`. The validator runs when direct tools are
registered.

This keeps the migration incremental:

- unmigrated modules can continue returning legacy data while they are being
  converted
- migrated modules fail fast if a new tool forgets the `ToolResult` contract
- extracted feature packages can be guarded by dotted module name just like
  in-tree modules

The escape hatch `KESTREL_TOOL_RESULT_CONTRACT_WARN_ONLY=1` downgrades failures
to warnings for emergency rollback only.

## Commands

Command prefixes are metadata on individual tools, not a separate command-tool
class. A decorated method can set `command_prefix="!something"`. The command
handler resolves matching tools and renders a `ToolResult` envelope into CLI
text for command-style use.

Prefer command prefixes for stable operator affordances. Do not build a parallel
command framework for a feature package.

## Testing

Recommended tests for a feature-owned tool:

- instantiate the feature with a minimal mocked agent
- call the decorated method directly and assert the `ToolResult` status
- call `feature.get_tools()` and execute the wrapper to verify schema and
  serialization
- test command-prefix rendering when the tool is command-addressable
- add the feature module to `MIGRATED_FEATURE_MODULES` only after all tools in
  that module return `ToolResult`

Useful existing tests:

- `tests/unit/test_dynamic_tool_execute_toolresult.py`
- `tests/unit/test_feature_inventory_contracts.py`
- `tests/unit/test_feature_startup_promotion.py`
- `tests/unit/test_tool_result_contract.py`
- `tests/unit/test_response_audit.py`

## Packaging Checklist

For an out-of-tree feature package:

1. Depend on `kestrel-sovereign-sdk`.
2. Expose one or more `Feature` subclasses.
3. Register the feature in `kestrel_sovereign.features`.
4. Return `ToolResult` from migrated tool methods.
5. Include tests that execute both the feature method and the dynamic wrapper.
6. Publish the package independently; do not add it as a framework dependency.

For a provider package:

1. Depend on the SDK or owning provider contract.
2. Register under the provider-specific entry-point group.
3. Keep the feature that consumes the provider responsible for user-facing
   tools and policy.
