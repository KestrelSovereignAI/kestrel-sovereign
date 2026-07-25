---
type: Architecture Spec
title: Kestrel Context Management Contract
description: Canonical current-state contract for prompt assembly, persistence, budgeting, pruning, retrieval, provider rendering, and context diagnostics.
resource: /docs/architecture/CONTEXT_SYSTEM_DESIGN.md
tags:
- docs
- architecture
- architecture-spec
- context
timestamp: '2026-07-25T00:00:00Z'
status: active
owner: context-runtime
canonical: true
generated: false
privacy: public
---

# Kestrel Context Management Contract

> **Status: active and canonical.** This page describes the current context
> runtime. It is the source of truth for what Kestrel stores, selects, budgets,
> renders, and reports. Provider routing remains canonical in
> [LLM Service Architecture](LLM_SERVICE_ARCHITECTURE.md), and the proposed
> end-state for automatic lossless pruning remains separately marked as design
> in [Context C Durable Salvage](CONTEXT_C_DURABLE_SALVAGE.md).

## Status vocabulary

This page uses four labels deliberately:

| Label | Meaning |
|---|---|
| **Shipped** | Runs on the normal production turn path. |
| **Conditional** | Implemented, but only runs for a named route, privacy mode, configuration, feature flag, or data shape. |
| **Diagnostic** | Observes or estimates runtime state; it is not proof of the exact provider payload or production decision path. |
| **Aspirational** | Design intent that is not a current runtime guarantee. |

## Contract at a glance

The production path is a selection-and-rendering pipeline, not a second copy of
conversation storage:

```text
canonical session history
    + stable system/constitution/tool-schema inputs
    + eligible episodes, memories, and query-dependent RAG
    -> route-qualified elastic token budget
    -> cache-stable, lumpy history selection
    -> rendered current user turn
    -> persist canonical input + exact rendered sent form
    -> provider adapter transport
```

The current owner of the live coordinator is
[`ContextManager.build_context()`](../../kestrel_sovereign/agent/context_manager.py).
The HTTP and tool diagnostics share section builders with that path, but do not
run the same coordinator and therefore must not be treated as an exact trace.

## Ownership and source anchors

| Responsibility | Current owner |
|---|---|
| Production section ordering, retrieval gates, elastic allocation, pruning, and degraded mode | [`kestrel_sovereign/agent/context_manager.py`](../../kestrel_sovereign/agent/context_manager.py) |
| System construction, history rendering, token measurement, and diagnostic breakdown | [`kestrel_sovereign/agent/context_builder.py`](../../kestrel_sovereign/agent/context_builder.py) |
| Shared section vocabulary, rendered-message emission, wrappers, and lumpy-anchor primitives | [`kestrel_sovereign/agent/context_stages.py`](../../kestrel_sovereign/agent/context_stages.py) |
| Fixed, adaptive, and production elastic budgets; response reserve | [`kestrel_sovereign/agent/token_budget.py`](../../kestrel_sovereign/agent/token_budget.py) |
| Route/model context limits and token counting | [`kestrel_sovereign/agent/token_counter.py`](../../kestrel_sovereign/agent/token_counter.py) and [`kestrel_sovereign/llm/model_catalog.py`](../../kestrel_sovereign/llm/model_catalog.py) |
| Canonical and rendered conversation persistence | [`kestrel_sovereign/storage/async_conversation_store.py`](../../kestrel_sovereign/storage/async_conversation_store.py) |
| Turn orchestration, feature-prompt gate, and sent-form write | [`kestrel_sovereign/kestrel_agent.py`](../../kestrel_sovereign/kestrel_agent.py) and [`kestrel_sovereign/agent/streaming.py`](../../kestrel_sovereign/agent/streaming.py) |
| Provider payload transformation and provider-specific constraints | [`kestrel_sovereign/llm/`](../../kestrel_sovereign/llm/) |
| `/api/agent/context-status` and shared status computation | [`kestrel_sovereign/endpoints/agent.py`](../../kestrel_sovereign/endpoints/agent.py) |
| Context tools and manual context-management operations | [`kestrel_sovereign/features/context/feature.py`](../../kestrel_sovereign/features/context/feature.py) |

`context_stages` is the shared vocabulary, not a claim that production and
measurement are identical. `context_manager` coordinates production;
`ContextBuilder.measure_context_breakdown()` coordinates the projection.

## The shipped production turn

### 1. Resolve route identity and the usable window

The selected LLM identity is route-qualified (`vendor:route/model`) before
budget construction. `TokenCounter` resolves the context limit with this
precedence:

1. a cap for the selected route;
2. a runtime-discovered limit for the model;
3. a cached limit from an earlier discovery;
4. a catalogued model limit;
5. the conservative fallback of 32,768 tokens.

Route caps have their own precedence: environment override, runtime-discovered
cap, then configuration/catalog cap. This matters when a model advertises a
larger window than the transport route permits.

Kestrel reserves **1,024 tokens** for the response. The production input budget
is therefore:

```text
total input budget = resolved context limit - 1,024 response reserve
```

Token counting uses the model tokenizer when available and a character-based
estimate otherwise. Message accounting also includes per-message and
conversation framing overhead. Counts remain estimates until a provider reports
actual usage.

### 2. Establish the mandatory system floor

The stable system section contains the core system prompt and constitution.
Tool schemas are not part of that mandatory floor. Context diagnostics
estimate them as a separate section because providers receive them through
their tool-definition channel rather than as ordinary chat text; the
provider's final schema encoding is not budget-exact.

Production creates an **elastic** budget. The mandatory system measurement may
raise the system allocation above the adaptive share. The increase is funded,
in order, by reducing RAG, memories, episodes, and history. If the mandatory
floor itself cannot fit, `ContextManager` returns degraded mode and the agent
refuses to call the LLM with a known-oversized prompt.

The adaptive starting shares are:

| Conversation size | System | History | Episodes | Memories | RAG |
|---|---:|---:|---:|---:|---:|
| Short, fewer than 10 messages | 15% | 60% | 5% | 5% | 15% |
| Medium, 10–29 messages | 15% | 40% | 20% | 10% | 15% |
| Long, 30 or more messages | 15% | 25% | 35% | 10% | 15% |

These are starting allocations, not hard partitions. When a section is
finalized below its allocation, its slack becomes available to later sections.
In the current production order, history is finalized after retrieval and can
use the slack left by all earlier sections.

The separately configured system-prompt byte budget applies while bootstrap,
state/doctrine, per-turn addendum, reflection, and episode material is
assembled. Priority-aware truncation and append guards keep lower-priority
optional material from silently displacing the constitutional floor. The
loaded-feature prompt is handled later by its own total-token gate.

### 3. Read canonical, session-scoped history

Outside `EPHEMERAL` privacy mode, production reads the active session's stored
history. Rows marked `excluded_from_context` are not selected. In `EPHEMERAL`
mode Kestrel intentionally supplies the system/constitution plus an ephemeral
notice, without persisted history, memory, or RAG.

Conversation rows have two different content contracts:

| Field | Contract |
|---|---|
| `content` | Canonical conversation content used by search, recall, summarization, and user-facing history. For a user turn this is the clean input, normally in the standard user-input wrapper, without retrieved context injected into it. |
| `rendered_content` with `metadata.sent_form=true` | Write-once transport form that was actually sent for that user/system turn, including any retrieval block. It is replayed byte-for-byte on later turns. |

The split prevents transient retrieval text from becoming the user's canonical
utterance while preserving byte-stable replay for prompt caches. Both columns
use the conversation store's configured encryption path. Legacy `sent_form`
rows that stored rendered bytes in `content` are split during read and
opportunistically migrated.

`ContextBuilder.format_conversation_history()` replays rendered sent forms. A
legacy user row without that marker is wrapped with the current anti-injection
user wrapper. Canonical content, not injected transport text, is what manual
compaction and conversation search consume.

Before history rendering, microcompaction replaces older, unprotected tool
result bodies in the in-memory history copy with compact JSON markers while
preserving tool-call pairing. The five most recent tool results are kept by
default; `KESTREL_MICROCOMPACT_KEEP_RECENT` configures the count. Protected,
decay-protected, and already-excluded rows are skipped. This stage does not
overwrite the canonical database rows.

### 4. Build stable and dynamic sections

`context_stages` provides typed sections and canonical wrappers. The stable
system prefix is finalized before turn-dependent retrieval. Reflection and
episode blocks are optional and budget-gated; episode loading starts only for
longer histories and is bounded by both count and byte/token budgets.

Retrieved memory and RAG are deliberately outside the stable system prefix.
They are joined into the current user transport under
`<retrieved_context>...</retrieved_context>`. Moving this material out of the
system prefix keeps old cacheable bytes stationary from one turn to the next.

### 5. Apply retrieval insertion and exclusion gates

Retrieval is **query dependent**. The current user query is passed to both
memory retrieval and RAG; Kestrel does not preload a fixed RAG block for every
turn.

Production applies these gates:

- Trivial turns, such as acknowledgements without a substantive query, skip
  both memory retrieval and RAG.
- The privacy mode must permit the relevant storage/retrieval operation.
- The corresponding retriever must exist and return content.
- Memory results must satisfy the configured score and relevance floors
  (`0.3` and `0.2` by default).
- RAG chunks must satisfy the configured RAG score floor (`0.5` by default).
- The elastic section must still have capacity. A section is omitted when its
  formatted insertion cannot fit.
- Retrieval failures are recorded as warnings and do not replace canonical
  history.

The memory retriever is invoked read-only from context assembly. Retrieval
results are transport material for this turn; they do not mutate the user's
canonical utterance.

### 6. Select cache-stable history with lumpy pruning

History is not continuously squeezed by a few messages on every turn.
`compute_lumpy_anchor()` selects a suffix that fits a stable target. The anchor
uses the section's **static history allocation**, rather than the turn's
temporary elastic slack, so a small change in retrieved material does not move
the history boundary.

When the window must shrink, Kestrel drops an older chunk and targets 75% of
the available budget by default. `KESTREL_PRUNE_TARGET_FRAC` can configure that
fraction from `0.05` through `1.0`. This hysteresis makes the selected suffix
remain stable for several turns, which gives prefix caches a useful run of
identical history.

There are two additional guards:

- history is pre-trimmed for message-wrapper overhead; and
- a safety-net lumpy prune runs if final assembly still exceeds the total
  elastic budget.

A single oversized message is emitted as an in-memory head/tail excerpt with a
pointer to the stored row. The canonical database row is not overwritten.

**Shipped default:** dropped history is omitted from the provider window but
remains stored. This is not the same as an automatic durable summary.
Feature-flagged salvage is described under
[Durable salvage status](#durable-salvage-status).

### 7. Gate the loaded-feature prompt

The loaded-feature prompt is assembled after `ContextManager` returns. The
same helper is used by streaming and non-streaming chat.

When budget accounting is available, Kestrel measures:

- the context already assembled;
- the current user prompt increment;
- the security addendum; and
- the cached `LOADED FEATURES` section.

If the projection would exceed `budget_summary.total_budget`, the loaded-feature
text is omitted for that turn. This does **not** unregister tools: enabled tool
schemas still travel through the tool-definition channel, and commands remain
callable through the tool registry. The gate protects the provider input
window; it is not a feature-disable operation. It does not have a
provider-exact tool-schema/framing count, so it can still understate the final
wire payload.

### 8. Persist the sent form and invoke the adapter

For a normal stored user turn, Kestrel writes the canonical wrapped input to
`content` and the fully rendered user transport to `rendered_content`, with
`sent_form=true`. The write occurs as part of turn orchestration in both
streaming and non-streaming paths.

The selected LLM adapter then transforms the generic messages and tools into
the provider's transport. Provider payloads are therefore downstream products
of context assembly, not canonical conversation history.

## Cache-stability contract

Kestrel can make the provider prefix cache-friendly, but it cannot guarantee a
provider cache hit.

**Shipped invariants:**

- stable system/constitution bytes precede turn-dependent retrieval;
- historical sent-form messages replay their original rendered bytes;
- dynamic retrieval is attached to the current user turn;
- the history suffix moves in chunks rather than one message at a time; and
- provider adapters apply only the cache controls their route supports.

Cache stability can still be lost when the route/model/tool fingerprint
changes, the system/constitution changes, manual compaction rewrites the
selected history, a provider evicts its cache, or a route does not expose a
prompt-cache facility.

## Provider transport constraints and route caps

The generic context contract ends at the adapter boundary. Current adapters
apply these material transformations:

| Route family | Current transport behavior |
|---|---|
| Anthropic / Claude | Extracts the leading system prompt into Anthropic's top-level system field, converts tool results to Anthropic blocks, and attaches supported ephemeral cache-control breakpoints to stable system/tools/history positions. Inline system messages are retained only on routes that advertise support; otherwise the adapter repairs/demotes them. |
| Native OpenAI | Sends OpenAI message/tool forms and may supply a stable `prompt_cache_key`. Compatible OpenAI-shaped routes are not assumed to support native-only cache or message features. |
| Gemini / Vertex | Converts messages to native role/parts structures. A system instruction may be represented as an initial user turn plus a model acknowledgement where required by the transport. |
| Ollama | Sends Ollama chat messages and separate image fields. `num_ctx` is a route option and can cap the usable window independently of a model-family headline limit. |
| `openai:plan` Codex | The leading system message becomes thread instructions. An existing Codex thread receives only the latest user input because Codex retains prior turns server-side; a fresh thread is seeded from Kestrel's selected transcript. Changes to thread-scoped model/instruction/tool settings invalidate the cached thread. |

For `openai:plan`, the per-turn Kestrel projection and Codex's server-side
thread occupancy are different measurements. Codex token-usage notifications
drive the latter, and Kestrel can compact durable session history and reset the
thread at the configured occupancy threshold. Do not infer server thread
occupancy from the size of one outbound turn.

See the adapter modules in [`kestrel_sovereign/llm/`](../../kestrel_sovereign/llm/)
and the routing contract in
[LLM Service Architecture](LLM_SERVICE_ARCHITECTURE.md).

## Context diagnostics

### HTTP surface

`GET /api/agent/context-status` is the shared diagnostic entry point.

- Without an active session, it returns an idle status and does not query
  storage.
- The default (`full=false`) is a cheap projection. It measures stored history,
  stable sections, and best-effort tool schemas, but intentionally skips live
  memory and RAG retrieval.
- `full=true` uses the latest stored canonical user turn as the retrieval query
  and performs read-only memory and RAG lookup. If no stored user query exists,
  the RAG row is explicitly labeled as an estimate without a representative
  query.

The response includes the resolved model/route identity, context limit,
response reserve, total budget, measured section breakdown, utilization and
warning status, route-cap details, salvage counters, and—when available—the
separate Codex thread occupancy. Tool-schema and provider-framing counts remain
best-effort estimates.

### Honesty boundary: issue #2534

> **Diagnostic limitation:** neither mode is an exact dry-run of
> `ContextManager.build_context()`. Open
> [#2534](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2534)
> tracks the remaining drift. Measurement uses its own non-elastic adaptive
> coordinator and does not execute the production lumpy-anchor/safety-prune
> path. Its full retrieval path also does not reproduce every production
> trivial-turn, relevance, and configured RAG-floor gate. Therefore
> `context-status` is a planning signal, not a receipt for the exact bytes sent
> to a provider.

Shared wrappers and counters reduce drift; they do not erase this coordinator
difference. This documentation issue intentionally does not change or close
#2534.

## Context feature and tool surface

The bundled Context feature exposes `context_status` using the same shared
status computation as the HTTP endpoint, in cheap mode for the active session.
It also exposes explicit operations for:

- summarizing or hierarchically compacting selected history;
- marking, excluding, and restoring messages;
- creating, listing, inspecting, applying, popping, dropping, and saving
  context stashes; and
- recursively querying a bounded context source.

These tools are operator/agent-invoked persistence operations. In particular,
`compact_context` creates a durable summary marker and marks original rows
excluded; the originals remain restorable. They must not be described as proof
that automatic Context C salvage runs on every production prune.

The generated tool and command inventory remains
[`KESTREL_FEATURES.md`](../../KESTREL_FEATURES.md).

## Durable salvage status

Three distinct behaviors coexist:

| Behavior | Status |
|---|---|
| Default lumpy history omission; source rows remain stored | **Shipped** |
| Manual durable compaction/exclusion through Context tools and Codex occupancy handling | **Shipped**, when invoked or route-triggered |
| Synchronous salvage marker/write plus background `SalvageWorker` processing during automatic pruning | **Conditional**, behind `KESTREL_CONTEXT_C_DURABLE_SALVAGE` |
| Complete automatic Context C lifecycle and all guarantees in the original design | **Aspirational** |

When the experimental flag is enabled, automatic pruning fails closed if the
durable marker/write cannot be established. The asynchronous worker and janitor
live in [`kestrel_sovereign/agent/salvage.py`](../../kestrel_sovereign/agent/salvage.py).
That partial implementation does not make every state machine, dispatcher, UX,
or recovery guarantee in the design record current. See
[Context C Durable Salvage](CONTEXT_C_DURABLE_SALVAGE.md) for the intended
end state.

## Evidence and regression anchors

Focused tests for the runtime contracts include:

- canonical versus rendered storage and replay:
  [`test_canonical_transport_split.py`](../../tests/unit/test_canonical_transport_split.py)
  and
  [`test_conversation_sent_form.py`](../../tests/unit/test_conversation_sent_form.py);
- elastic budgets and route caps:
  [`test_elastic_token_budget.py`](../../tests/unit/test_elastic_token_budget.py)
  and
  [`test_model_catalog.py`](../../tests/unit/test_model_catalog.py);
- lumpy pruning and cache stability:
  [`test_lumpy_prune.py`](../../tests/unit/test_lumpy_prune.py);
- retrieval and whole-window measurement:
  [`test_context_stages_equivalence.py`](../../tests/unit/test_context_stages_equivalence.py)
  and
  [`test_context_breakdown_measurement.py`](../../tests/unit/test_context_breakdown_measurement.py);
- context-status behavior:
  [`test_context_management.py`](../../tests/unit/test_context_management.py).

## Related contracts

- [LLM Service Architecture](LLM_SERVICE_ARCHITECTURE.md) — vendor, route,
  model, adapter, and mandate ownership.
- [Memory System](MEMORY_SYSTEM.md) — memory scoring and consolidation.
- [Memory Ownership](MEMORY_OWNERSHIP.md) — ownership boundaries among
  storage, retrieval, context, and features.
- [Storage Architecture](storage/STORAGE_ARCHITECTURE.md) — persistence
  backends and data-layer contracts.
- [Context C Durable Salvage](CONTEXT_C_DURABLE_SALVAGE.md) — explicitly
  aspirational automatic-salvage design.
