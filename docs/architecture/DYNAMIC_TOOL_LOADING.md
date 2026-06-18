---
type: Architecture Spec
title: Dynamic Tool Loading
description: '**Status:** Implemented **Author:** Design review **Date:** 2026-02-16
  **Affected files:** `kestrel_sovereign/kestrel_agent.py`, `kestrel_sovereign/features/base.py`,
  `kestrel_s...'
resource: /docs/architecture/DYNAMIC_TOOL_LOADING.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Dynamic Tool Loading

**Status:** Implemented
**Author:** Design review
**Date:** 2026-02-16
**Affected files:** `kestrel_sovereign/kestrel_agent.py`, `kestrel_sovereign/features/base.py`, `kestrel_sovereign/prompts/system_prompt.md`

---

## Problem Statement

The Kestrel agent uses an orchestrator/subagent architecture where 25 features expose 147 individual tools. Every feature call today requires **two LLM roundtrips**:

1. **Orchestrator LLM** decides which feature to call, generates `task` and `context` strings.
2. **Subagent LLM** receives the task, decides which of the feature's tools to call, executes them, and returns a text summary.

This means even a trivial read operation like "what model am I using?" costs two LLM calls: the orchestrator dispatches to `model_agent(task="get current model")`, the ModelAgent subagent LLM decides to call `get_current_model()`, executes it, and summarizes the result back.

### Consequences

- **Latency**: Every feature interaction takes 2-4 seconds minimum (two LLM calls).
- **Cost**: Double token consumption for simple queries.
- **Behavioral impact**: The orchestrator naturally stops after one batch of subagent dispatches and asks the user what to do next. Because each dispatch is expensive, the LLM is reluctant to chain multiple operations autonomously.
- **Wasted reasoning**: The subagent LLM call for `list_models` is pure overhead -- there is no decision to make, just a function to call.

## Proposed Solution: Dynamic Tool Loading

After the orchestrator dispatches to a feature via the subagent path, that feature's individual tools become available for **direct calling** in subsequent orchestrator turns. No subagent LLM hop required.

### Lifecycle

```
Session start:
  Orchestrator sees 25 feature-level dispatch tools (unchanged from today)

First contact with a feature (subagent dispatch):
  orchestrator -> model_agent(task="list models") -> subagent LLM -> list_models() -> result
  Side effect: ModelAgent's 7 individual tools are registered with the orchestrator

Subsequent calls (direct execution):
  orchestrator -> list_models(provider="openai") -> tool.execute() -> JSON result
  No subagent LLM, no second roundtrip. Just function execution.
```

### Tool List Growth

The orchestrator starts with 25 tools (one per feature). As features are explored through subagent dispatch, their individual tools are added. After exploring 5 features, the orchestrator might have 55-60 total tools. This remains well within the context limits of any modern LLM.

## Architecture Details

### Current Flow (unchanged for subagent dispatch)

This is the existing code path in `_handle_orchestrator_response()` at line 1046 of `kestrel_agent.py`:

```python
# Orchestrator calls model_agent(task="list models", context="...")
feature = features_by_tool_name.get(tool_name)  # finds ModelAgent
result = await feature.execute_as_subagent(task=task, context=context)
# execute_as_subagent() makes its own LLM call, runs tools, returns text
```

### New Flow (direct execution for explored features)

When the orchestrator calls a tool name that matches an individual tool from an explored feature, bypass the subagent entirely:

```python
# Orchestrator calls list_models(provider="openai") directly
tool = direct_tools_by_name.get(tool_name)  # finds the AgentTool instance
result = await tool.execute(**args)
result = _serialize_tool_result(result)
# No LLM call. Just function execution -> JSON result.
```

### State Tracking

Add a set to track which features have been explored, and a dict mapping individual tool names to their `AgentTool` instances:

```python
# On KestrelAgent (or scoped to the orchestrator loop)
self._explored_features: Set[str] = set()
self._direct_tools: Dict[str, AgentTool] = {}        # tool_name -> AgentTool
self._direct_tool_defs: List[Dict[str, Any]] = []    # OpenAI format definitions
self._tool_to_feature: Dict[str, str] = {}           # tool_name -> feature_name
```

### Registration After Subagent Dispatch

After a successful `execute_as_subagent()` call, register the feature's individual tools:

```python
# In _handle_orchestrator_response(), after line 1062
result = await feature.execute_as_subagent(task=task, context=context)

# --- NEW: Register individual tools for direct calling ---
if feature.tool_name not in self._explored_features:
    self._explored_features.add(feature.tool_name)
    for tool in feature.get_tools():
        self._direct_tools[tool.name] = tool
        self._direct_tool_defs.append(tool.schema.to_openai_format())
        self._tool_to_feature[tool.name] = feature.tool_name
    logging.info(
        f"[DYNAMIC-TOOLS] Explored {feature.tool_name}, "
        f"registered {len(feature.get_tools())} direct tools. "
        f"Total direct tools: {len(self._direct_tools)}"
    )
```

### Tool Dispatch Logic

Modify the dispatch logic in `_handle_orchestrator_response()` to check direct tools before falling through to the unknown-tool error:

```python
# Current code at line 1046:
feature = features_by_tool_name.get(tool_name)
if feature:
    # ... existing subagent dispatch ...

# NEW: Check if this is a direct tool from an explored feature
elif tool_name in self._direct_tools:
    try:
        tool = self._direct_tools[tool_name]
        raw_result = await tool.execute(**args)
        result = _serialize_tool_result(raw_result)

        dispatch_duration = int((time.time() - dispatch_start) * 1000)
        await self.observability_store.log_tool_response(
            event_id=dispatch_event_id,
            success=True,
            duration_ms=dispatch_duration,
        )
        logging.info(
            f"[DIRECT-TOOL] {tool_name} executed directly "
            f"({dispatch_duration}ms)"
        )
    except Exception as e:
        logging.error(f"[DIRECT-TOOL] {tool_name} failed: {e}")
        result = {"success": False, "error": str(e)}
        # ... log failure ...

else:
    result = {"success": False, "error": f"Unknown feature tool: {tool_name}"}
```

### Dynamic Tool List for LLM Calls

The `feature_tools` list passed to `generate_with_messages()` must include both the original feature dispatch tools and any direct tools from explored features:

```python
# In _build_feature_tools() or at the call site
def _build_all_tools(self) -> List[Dict[str, Any]]:
    """Build combined tool list: feature dispatchers + explored individual tools."""
    tools = self._build_feature_tools()     # 25 feature-level tools
    tools.extend(self._direct_tool_defs)    # Individual tools from explored features
    return tools
```

This combined list is what gets passed to the LLM on each subsequent call within the orchestrator loop. The LLM can then choose between dispatching to a feature (for complex multi-step tasks) or calling an individual tool directly (for simple operations).

### Streaming Handler

The same changes apply to `_handle_orchestrator_response_streaming()` (line 1162). The direct tool dispatch path is identical except for the streaming activity indicators:

```python
elif tool_name in self._direct_tools:
    tool = self._direct_tools[tool_name]
    raw_result = await tool.execute(**args)
    result = _serialize_tool_result(raw_result)
    dispatch_duration = int((time.time() - dispatch_start) * 1000)
    yield f"-> {tool_name} ({dispatch_duration}ms)\n"
```

## What Changes

### `kestrel_sovereign/kestrel_agent.py`

| Location | Change |
|----------|--------|
| `__init__()` | Add `_explored_features`, `_direct_tools`, `_direct_tool_defs`, `_tool_to_feature` |
| `_build_feature_tools()` | Rename or add `_build_all_tools()` that appends `_direct_tool_defs` |
| `_handle_orchestrator_response()` | Add `elif tool_name in self._direct_tools` branch; after subagent dispatch, register tools |
| `_handle_orchestrator_response_streaming()` | Same direct-tool branch as non-streaming version |
| Tool list passed to `generate_with_messages()` | Use combined list instead of feature-only list |

### `kestrel_sovereign/features/base.py`

| Location | Change |
|----------|--------|
| `Feature` class | Add `get_tools_openai_format()` convenience method (optional -- callers can already do `[t.schema.to_openai_format() for t in feature.get_tools()]`) |

### `kestrel_sovereign/prompts/system_prompt.md`

| Location | Change |
|----------|--------|
| "HOW TO USE TOOLS" section | Add note explaining that after exploring a feature, its individual tools become available for direct calling |

## What Does NOT Change

- **Feature code**: No new decorators, annotations, or interfaces. Every existing `@tool` method works as-is.
- **Subagent dispatch**: The `execute_as_subagent()` path remains fully functional. The orchestrator can still delegate complex multi-step tasks to subagents.
- **`!command` routing**: Commands typed by the user still go through `CommandHandler`.
- **Feature isolation**: Subagent calls still get their own system prompt, LLM context, and tool scope.
- **Tool schemas**: `ToolSchema.to_openai_format()` already produces the correct format for individual tools.
- **Result serialization**: `_serialize_tool_result()` already handles all tool return types.
- **All 147 tools**: Every tool continues to work through the subagent path. Direct calling is purely additive.

## Example Flow

```
User: "Run a full diagnostic"

--- Round 1: Subagent dispatches (orchestrator -> 3 features) ---

orchestrator LLM call #1:
  -> model_agent(task="list models and check status")
     subagent LLM call -> list_models() -> get_current_model() -> text result
     Side effect: 7 ModelAgent tools now available directly

  -> memory_feature(task="check memory status")
     subagent LLM call -> memory_status() -> text result
     Side effect: 9 MemoryFeature tools now available directly

  -> task_feature(task="list pending tasks")
     subagent LLM call -> list_my_tasks() -> text result
     Side effect: 5 TaskFeature tools now available directly

[3 subagent LLM calls. 21 individual tools now registered for direct use.]

--- Round 2: Direct calls (NO subagent hops) ---

orchestrator LLM call #2 (tool list now includes 25 + 21 = 46 tools):
  -> get_current_model()         -> direct execute -> JSON
  -> get_episodes(limit=5)       -> direct execute -> JSON
  -> check_task_status(id="x")   -> direct execute -> JSON
  -> memory_status()             -> direct execute -> JSON
  -> wallet_balance()            -> Unknown (wallet not yet explored)

orchestrator LLM call #3 (same 46 tools, results from call #2):
  -> wallet_feature(task="check balance")  -> subagent dispatch
     Side effect: 13 WalletFeature tools now registered

[Total tools now: 25 + 21 + 13 = 59]

--- Round 3: Final synthesis ---

orchestrator LLM call #4: No tool calls, generates diagnostic report.
```

**Before dynamic tool loading:** This would require 4+ subagent dispatches with 4+ subagent LLM calls, plus orchestrator calls = ~8 LLM roundtrips.

**After:** 4 orchestrator LLM calls + 3 subagent LLM calls = 7 total, with Round 2 executing 4 tool calls at near-zero cost (no LLM, just function execution).

The real win is behavioral: after Round 1, the orchestrator sees cheap direct tools and is far more willing to chain calls autonomously instead of stopping to ask the user.

## Risks and Mitigations

### Tool count growth

**Risk**: If the user explores many features in one session, the tool list could grow large enough to degrade LLM performance or exceed context limits.

**Mitigation**: Cap the direct tool count at ~60-70. If the limit is reached, evict tools from the least-recently-used feature. Feature dispatch tools (the original 25) are never evicted. At 70 tools with ~100 tokens per tool definition, the tool section is ~7K tokens -- well within budget.

```python
MAX_DIRECT_TOOLS = 60

def _maybe_evict_direct_tools(self):
    """Evict least-recently-used feature's tools if over limit."""
    while len(self._direct_tools) > MAX_DIRECT_TOOLS:
        # Find the feature explored longest ago
        oldest = next(iter(self._explored_features))  # Set preserves insertion order in 3.7+
        self._evict_feature_tools(oldest)
```

### Loss of subagent context isolation

**Risk**: Direct calls skip the subagent's system prompt and multi-tool reasoning. A complex task that requires the subagent to chain multiple tools with intermediate reasoning would fail if called directly.

**Mitigation**: This is acceptable because:
1. The orchestrator still has the feature dispatch tool available. For complex tasks, it will naturally choose the higher-level dispatch.
2. Direct calls are best suited for simple read/query operations (`list_models`, `memory_status`, `get_episodes`).
3. The orchestrator LLM provides its own reasoning between direct tool calls -- it just cannot access the feature-specific system prompt.

### Name collisions between features

**Risk**: Two features could have tools with the same name (e.g., both have a `status` tool).

**Mitigation**: Prefix individual tool names with the feature name during registration if a collision is detected:

```python
if tool.name in self._direct_tools:
    # Collision - prefix with feature name
    prefixed_name = f"{feature.tool_name}__{tool.name}"
    self._direct_tools[prefixed_name] = tool
    # Update the OpenAI format definition too
```

In practice, current tools already use descriptive names (`list_models`, `memory_status`, `check_task_status`) so collisions are unlikely. A scan of all 147 current tool names should be done before implementation to verify.

### Direct call results differ from subagent results

**Risk**: Direct calls return raw JSON from `tool.execute()`, while subagent calls return a text summary from the subagent LLM. The orchestrator needs to handle both formats.

**Mitigation**: This is already handled. The orchestrator receives tool results as JSON in the `tool` role message regardless of whether the result came from a subagent or direct execution. The `_serialize_tool_result()` function ensures consistent JSON serialization. The subagent path wraps results in `{"success": True, "result": ...}`, and direct calls will use the same wrapper via `tool.execute()` which already returns that format (see `DynamicTool.execute()` in `base.py` line 593).

### Session persistence

**Risk**: If explored tools persist across sessions, the tool list could be stale after feature code changes or restarts.

**Mitigation**: Explored tools are session-scoped. The `_explored_features` set and `_direct_tools` dict are initialized empty on each agent construction. They are populated organically during the conversation and discarded when the session ends. No persistence, no staleness.

## Estimated Effort

| Area | Scope |
|------|-------|
| Core implementation | ~100-150 lines across `kestrel_agent.py` and `features/base.py` |
| Streaming handler | ~30 lines (mirror of non-streaming changes) |
| System prompt update | ~5 lines added to `system_prompt.md` |
| Unit tests | Test direct dispatch, test explore-then-direct flow, test eviction |
| Integration tests | Test full conversation with mixed subagent/direct calls |
| Migration | None -- purely additive, backwards compatible |

## Testing Plan

### Unit Tests

```python
async def test_direct_tool_registration():
    """After subagent dispatch, feature tools are registered."""
    agent = await create_test_agent()
    assert len(agent._direct_tools) == 0

    # Dispatch to model_agent via subagent
    await agent._handle_orchestrator_response(...)

    # ModelAgent's tools should now be registered
    assert "list_models" in agent._direct_tools
    assert "get_current_model" in agent._direct_tools
    assert "model_agent" in agent._explored_features


async def test_direct_tool_execution():
    """Direct tools execute without subagent LLM hop."""
    agent = await create_test_agent()
    # Pre-register a tool
    agent._direct_tools["list_models"] = mock_tool

    # When orchestrator calls list_models, it should execute directly
    # (not go through execute_as_subagent)
    result = await agent._direct_tools["list_models"].execute()
    assert result["success"] is True


async def test_tool_eviction():
    """Least-recently-used feature tools are evicted at capacity."""
    agent = await create_test_agent()
    # Register tools from many features to exceed MAX_DIRECT_TOOLS
    # Verify oldest feature's tools are evicted first


async def test_feature_dispatch_still_works():
    """Subagent dispatch continues to work for explored features."""
    agent = await create_test_agent()
    # Explore model_agent
    # Then call model_agent(task="complex multi-step task")
    # Should still go through execute_as_subagent, not direct
```

### Integration Tests

```python
async def test_explore_then_direct_flow():
    """Full conversation: subagent dispatch -> direct tool calls."""
    # Send message that triggers model_agent subagent dispatch
    # Verify model_agent tools are now in the tool list
    # Send follow-up that should trigger direct list_models call
    # Verify no subagent LLM call was made for the direct call
```

## Open Questions

1. **Should the system prompt describe individual tools from explored features?** Currently the LOADED FEATURES section lists features and their commands. After exploration, the orchestrator sees individual tools in its function calling schema, but the system prompt does not describe them. The LLM should handle this fine since tool definitions include descriptions, but explicit prompt guidance could help.

2. **Should certain tools be excluded from direct registration?** Some tools may have side effects that benefit from subagent reasoning (e.g., `delete_episode`, `set_model`). A `direct_callable: bool` flag on `ToolSchema` could control this, but adds complexity. Start without this and add if needed.

3. **Should direct tools be available across orchestrator turns or only within a single `_handle_orchestrator_response` call?** The proposal scopes them to the session (agent lifetime). This means a second user message in the same session benefits from prior exploration. This is the intended behavior -- tools explored in turn 1 are available in turn 2.
