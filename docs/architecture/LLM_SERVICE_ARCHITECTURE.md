---
type: Architecture Spec
title: Kestrel LLM Service Architecture
description: '**Canonical spec for every change touching the LLM service, provider
  registry, discovery, or routing.** If this doc contradicts code, the code wins —
  and this doc is a bug. Upda...'
resource: /docs/architecture/LLM_SERVICE_ARCHITECTURE.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-07-25T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Kestrel LLM Service Architecture

> **Canonical spec for every change touching the LLM service, provider registry, discovery, or routing.** If this doc contradicts code, the code wins — and this doc is a bug. Update it in the same change, not later.
>
> **Last major rewrite:** 2026-04 — replaced the single-string "provider" model with vendor / route / model. See GitHub epic #688 and the CLAUDE.md case study "Model Selection System — What NOT to Do".

## Principles

1. **Three orthogonal dimensions.** An LLM call is a (vendor, route, model) triple. Collapsing any two into one string is the antipattern that motivated this rewrite.
2. **Model independence.** Users and agents switch vendors, routes, and models at runtime without restart.
3. **No hardcoded model IDs in code.** Discovery is the source of truth. Config holds patterns and routes, never specific IDs.
4. **Discovery-driven visibility.** The UI's model list reflects what the vendor's API actually serves, not a maintained allowlist.

---

## The three dimensions

### 1. Vendor

Who makes the weights. Examples: `openai`, `anthropic`, `google`, `ollama`, `openrouter`.

Each vendor owns exactly one model catalog. Discovery runs per vendor; all routes for that vendor share the catalog. When grouping models for the UI, we group by vendor.

```python
@dataclass
class ModelInfo:
    id: str           # "<some model id>" — vendor decides
    provider: str     # the vendor name (field name retained for file-format compatibility;
                      # the *semantic* is vendor, not "execution provider")
    display_name: str
    category: ModelCategory   # chat | embedding | image | audio
    is_featured: bool
    is_hidden: bool
    created_at: Optional[str]
    supports_tools: bool
    supports_vision: bool
    ...
```

### 2. Route

How to reach a vendor. A route bundles: `base_url` + auth method + adapter class + per-route defaults.

One vendor can have **several** routes. For example:

| Route | Auth | Adapter | Purpose |
|---|---|---|---|
| `anthropic:api` | API key (`ANTHROPIC_API_KEY`) | `AnthropicAdapter` | Metered API billing |
| `anthropic:plan` | OAuth (`ANTHROPIC_AUTH_TOKEN`) | `ClaudeMaxAdapter` | Claude Max subscription |
| `openai:api` | API key (`OPENAI_API_KEY`) | `OpenAIAdapter` | Metered API billing |
| `openai:plan` | OAuth (`CODEX_AUTH_TOKEN` / `~/.codex/auth.json`) | `CodexAdapter` | ChatGPT subscription |
| `ollama:local` | none | `OllamaAdapter` | localhost (or `OLLAMA_HOST`) |

A route's composite key is `"<vendor>:<route>"`. The route key is the routing identity; the vendor is the grouping identity.

#### Codex reasoning-effort compatibility

`openai:plan` reads the running Codex app-server's model catalog before a turn
when `model_reasoning_effort` is configured. Legacy `max` remains a native
value for models that advertise it; for a model with a lower advertised
ceiling (currently GPT-5.5 at `xhigh`), Kestrel treats `max` as a compatibility
alias for that ceiling. Unknown capability metadata fails before
`thread/start` instead of forwarding an unproved value. The adapter freezes
the resulting model/effort pair for both `thread/start` and `turn/start`.

The static compatibility rank is scheduled for removal in Kestrel 1.0 once
the Codex protocol exposes stable semantic ranks. Live capability negotiation
remains authoritative; the disk model cache is only a bounded fallback when
the live catalog RPC is unavailable.

#### Native provider retirement and turn isolation

Withdrawal stops new admission; it does not retire an already acquired native
resource or an already accepted tool invocation. If authority changes after SDK
stream acquisition but before publication, `await_execution_work` delegates the
exact acquired value and its physical exit callback to
`close_native_execution_stream`. That canonical boundary admits the callback
lazily under cleanup-only custody, joins its original Task, and retains explicit
resource/Task debt if admission or physical retirement fails. A denied publication
must retain the original denial together with actual close errors and observer
cancellation, without changing their original control graphs.

The physical-close boundary retains and isolates those finite originals by
default. Plain iterator `GeneratorExit` remains close protocol, while a
control-bearing one is preserved as evidence. Non-stream OpenAI, Anthropic and
Ollama producers share the canonical response owner across SDK normalization
and retries, including late per-model catalog metadata requests. Availability
probes cannot suppress that owner's original native control, and fresh caller
scopes may still reuse a healthy shared client. Temporary client lifetime
retirement remains distinct from per-operation response ownership. Private
OpenRouter response/stream clients and temporary Google catalog SDKs retire
before their request lifetime exits; Vertex's cached private client belongs to
the adapter's `aclose()`. Supplied route clients remain borrowed. All GenAI
shutdown observers join the same original SDK/session/pool Task, including
when callback publication fails. Refused or unstarted retirement retains the
actual resource and original Task rather than reporting physical success.
Private close debt is carried separately from shared source control errors.
Asynchronous abandoned-result cleanup and inline Codex Stop checkpoints use
the same receipt-and-join boundary before releasing their original stream,
turn sink or mutex. A failed canonical callback cannot hide an accepted Task
behind a replacement Future or allow its unstarted body to run eagerly.

Offline embedding resolution establishes the same unpublished-service owner
immediately after constructing a private `LLMService`, before awaiting persisted
configuration or discovery. A resolution that cannot return retires that owner
itself. `AgentEmbeddingResolution` then hands off the lifetime explicitly: use
`async with resolution` or `await resolution.aclose()`. Reindex owns it across
dry-run, refusal, write and failure exits; reanchor uses that same lifetime with
its known committed-body distinction. Injected services remain borrowed.
Synchronous target lookup never constructs a hidden service. Before-Task closer
refusal retains both the original body/cancellation and actual resource debt.

Semantic maintenance classifies original execution control before business
status projection. Native uncertainty is not an ordinary failed-run diagnostic
or stale-lease result. Failed diagnostics retain their original worker error,
and exact holder/token lease release uses the canonical physical-close owner
under cleanup-only custody. Its body, native close and actual observer withdrawal
are isolated finite evidence, never mutations to a shared source receipt.

Codex turn shutdown first closes callback admission, directly joins the original
accepted reader Tasks, and holds the original thread mutex and event sink through
any accepted completed-effect checkpoint. It must not allocate an intermediate
collector to own Tasks already accepted. Checkpoint admission uses the canonical
owned-Task boundary with the original executor as the retirement resource.
Physical Task settlement also retires its active-set membership immediately;
deferred completion bookkeeping must not make that finite drain repeat a join.

Temporary image paths enter the owned cleanup inventory immediately after file
construction, before context entry or writing. One unlink collector serves both
partial materialization and turn finalization, attempts every owned path without
following leaf symlinks, and retains cleanup errors rather than masking the
opening failure. Within a turn, a failed unlink retains its actual path and the
original accepted turn Task as native retirement debt, so provider fallback cannot
consume it as ordinary availability. Image cleanup errors cannot bypass callback
admission closure, accepted-handler/checkpoint retirement or sink/mutex release.

Provider-key service calls retain original runtime admission across client
acquisition and result disclosure. The common native HTTP hooks revalidate at
transport dispatch, after arbitrary request observers. Public SDK/convenience
wrappers keep both the operation and their existing awaited client shutdown
inside that same admission; publication is checked again after shutdown.
Cleanup is not replacement-work authority. Independent body and epilogue
failures retain their original carriers without rewriting a shared source graph.
Bootstrap route initialization and payer child-key provisioning use that same
complete lifecycle, including post-close credential/route disclosure. Bootstrap
cache hits retain the enclosing caller's admission too; a cached credential does
not authorize a retired runtime to construct a new route.

A registered callback's pure own-scope denial can remain local to that turn only
when it carries neither canonical commit outcome nor cleanup debt. An acknowledged
commit followed by an own-scope denial is still durable execution evidence: the
shared connection must fail closed rather than treating it as a harmless refusal.
These rules preserve the original source evidence rather than inferring safety
from a Task name, exception class, or physical close alone.

### 3. Model

Which weights. Lives inside a vendor. Always an opaque ID string — **never** a literal in Python.

---

## Configuration shape

Canonical `[llm]` section of `kestrel.toml`:

```toml
[llm]
# Fallback order at the route level. Each entry is "<vendor>:<route>".
route_priority = [
    "anthropic:plan",
    "openai:plan",
    "openrouter:api",
    "anthropic:api",
    "openai:api",
    "ollama:local",
]

[llm.vendors.anthropic]
is_cloud = true

[llm.vendors.anthropic.routes.api]
adapter        = "AnthropicAdapter"
api_key_env    = "ANTHROPIC_API_KEY"
model          = "auto"
selection_hints = ["sonnet", "haiku", "opus"]

[llm.vendors.anthropic.routes.plan]
adapter        = "ClaudeMaxAdapter"
auth_token_env = "ANTHROPIC_AUTH_TOKEN"
model          = "auto"

[llm.vendors.openai]
is_cloud = true

[llm.vendors.openai.routes.api]
adapter        = "OpenAIAdapter"
api_key_env    = "OPENAI_API_KEY"
model          = "auto"

[llm.vendors.openai.routes.plan]
adapter        = "CodexAdapter"
auth_token_env = "CODEX_AUTH_TOKEN"
model          = "auto"

[llm.vendors.ollama]
is_cloud = false

[llm.vendors.ollama.routes.local]
adapter        = "OllamaAdapter"
host           = "http://localhost:11434"
model          = "auto"

[llm.vendors.openrouter]
is_cloud = true

[llm.vendors.openrouter.routes.api]
adapter        = "OpenRouterAdapter"
base_url       = "https://openrouter.ai/api/v1"
api_key_env    = "OPENROUTER_API_KEY"
model          = "auto"
selection_hints = ["chat"]
```

> Pre-2026-05 setups put this same content in a standalone `llm_config.toml` at the repo root. That path was removed in epic #938; run `kestrel migrate-llm-config` to fold a legacy file into `kestrel.toml [llm]`.

### Rules

- `model = "auto"` means "resolve via discovery using `selection_hints`." Never hardcode a specific ID.
- `selection_hints` are substring patterns (e.g. `"sonnet"`, `"mini"`), not full IDs.
- `is_cloud` on a vendor defaults to `true`; set to `false` for local-only vendors (Ollama, llama.cpp). Used by streaming gating and privacy routing.
- `is_local` on a route flags local endpoints (currently only llama.cpp's `local = true`).

### Adding a vendor or route

No code changes required unless the vendor needs a new adapter class. Add a `[vendors.<name>.routes.<route>]` block with the adapter name; it appears in the dropdown automatically.

### Adding a new adapter class

`LLMAdapter` lives in `kestrel-sovereign-sdk` (since SDK 0.5 / epic [#1048](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1048) Wave 1A). The base class and the response/marker types (`LLMResponse`, `ToolCall`, `ToolCallStarted`, `ModelInfo`) are importable from `kestrel_sdk.llm`; the conformance helpers (`drain_streaming_with_tools`, `assert_response_contract`, etc.) live in `kestrel_sdk.testing`.

There are **two paths** depending on whether the adapter ships in this repo or as an external package.

**In-tree adapter** (lives in `kestrel_sovereign/llm/`):

1. Subclass [`kestrel_sdk.llm.LLMAdapter`](https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk/blob/main/kestrel_sdk/llm/adapter.py) — the framework's `kestrel_sovereign/llm/adapter.py` is a thin subclass that adds image-handling helpers; in-tree adapters typically extend that.
2. Register in `_ADAPTER_REGISTRY` in [provider_registry.py](../../kestrel_sovereign/llm/provider_registry.py).
3. Implement `list_models()` to return models tagged with the correct vendor — or raise `NotImplementedError` and let the adapter share a canonical route's discovery via the per-vendor rule below.
4. (Optional) Implement `get_streaming_response_with_tools` and emit `ToolCallStarted` markers per the timing rules in [llm/PROVIDER_PLUGINS.md](llm/PROVIDER_PLUGINS.md). Required if you want the constitutional honesty layer's bubble retraction to fire on this adapter — see [llm/HONESTY_LAYER.md](llm/HONESTY_LAYER.md).
5. (Optional) Override the adapter metadata methods (`substrate_type`, `deliberation_style`, `cost_per_1m_tokens`, `display_name`, `key_env_var`, `contribute_system_prompt`) so the council, costing, and substrate-routing surfaces work without adding `if vendor == "x"` branches in framework code. See [llm/PROVIDER_PLUGINS.md §"Adapter metadata"](llm/PROVIDER_PLUGINS.md#adapter-metadata) for the full method list and their default behavior.

**External plugin** (separate `pip`-installable package): no edits to this repo. The plugin author publishes a package that depends on `kestrel-sovereign-sdk` only and registers under the `kestrel_sovereign.llm_providers` entry-point group. Framework discovers the adapter at startup. Full author guide in [llm/PROVIDER_PLUGINS.md](llm/PROVIDER_PLUGINS.md), including the conformance suite (`pytest` against `kestrel_sdk.testing` helpers).

### Per-vendor discovery rule

Multiple routes can target the same vendor. Discovery runs **once per vendor**: we pick the first route whose adapter implements `list_models()`. Subscription adapters (`ClaudeMaxAdapter`, `CodexAdapter`) raise `NotImplementedError` — their routes share the api-route's catalog by virtue of being under the same vendor.

This means `openai:plan` shows the same model list as `openai:api`, without alias tables.

### Discovery deadline

Each route's discovery steps run under one per-route deadline,
`[llm] discovery_route_deadline_seconds` (default 5; a non-positive or
non-numeric value fails at startup). The steps are the vendor's chat catalog,
the route's embedding listing, and its decision listing. A step that overruns
is cancelled, logged at WARNING with the vendor, route and step, and recorded
as that vendor's discovery failure, which the `model_discovery` health check
shows. The rest of discovery returns without it, and within one discovery call
the route's later steps are skipped, so a route that accepts connections and
never answers delays `!model-list`, `/api/models` or a turn's `auto` warm-up by
one deadline instead of stalling it (#3646). A route with a pinned
`decision_model` that did not answer is not sent its pin canary either; the pin
keeps its last verification, marked stale. A turn whose route stays `auto`
because its discovery failed is refused by the route chain, with that failure
named, rather than sent the literal `auto`.

A cache hit still reconciles every route's embedding and decision listings. An
overrun there amends the snapshot the hit served: the same catalog and age,
with the vendor's failure added, so the health check shows it whichever
service hit the cache. An overrun outside any discovery pass (local-only `auto`
resolution, or an embedding or decision reconcile called on its own) amends the
catalog that was current when the step began, the same way. Amendments against
one catalog merge: services that each saw a different vendor fail both add
theirs, and a vendor reported failing twice reports the later failure. Only a
new catalog (a refresh, a restore from disk, an invalidation) discards an
amendment made against the one it replaced. The health check's
`catalog_version` names the catalog, which an amendment keeps.

An embedding sweep with an overrun is not cached, so the next discovery call
asks that route again. Within one call (a listing, a reconcile with every
route it resolves, an embedding settings read) the incomplete sweep is reused,
and a caller that joined another caller's sweep records its overruns too. The
Ollama adapter bounds its own listing calls by the same value. Its chat client
carries no timeout: generation is bounded by the cancellable generation path
and the orchestrator's call watchdog, not by this deadline.

Owned stream producers carry the original call's observation scope even when
the stream was constructed before the watchdog's first pull. This transfers
only progress/attempt timing, not authority, privacy reentry or turn ownership.
The watchdog recognizes the explicit iterator forwarding lineage; an arbitrary
child copied from a producer cannot claim it. An already-pinned observation
scope is never combined with another consumer's listeners or attempt bound.
Timeouts carrying native commit uncertainty, authority denial or retained
cleanup debt are control outcomes, not ordinary discovery outages; their
original carrier propagates instead of becoming an empty catalog.

Implicit Ollama chat/embedding catalogs own their temporary HTTPX pool through
the same canonical LLM client-close registry as other private clients. The
adapter inventory references that original Task; no raw wrapper must be admitted
before cleanup ownership exists. Before-Task refusal retains client/None debt;
failed receipt publication retains the accepted start-gated Task. Failed closes
are never repeated. Body, physical-close and actual joiner withdrawal are finite
isolated evidence, not private cleanup appended to a shared source error.
Successful physical pool retirement removes obsolete client ownership, but
first retains independent canonical publication control at the adapter. That
control fences new private discovery and survives repeated adapter/service
shutdown. Its finite history is bounded by borrowers admitted before fencing;
actual joiner withdrawal stays local to its caller and invents no pool debt.

The process-wide background catalog refresh has one original Task and one
bounded strongest nonlocal-control receipt. Completion callbacks and concurrent
or later readers share that original receipt, including cancellation carrying
native commit uncertainty. Fresh/stale reads, refresh joins and new refresh
admissions observe it even before deferred callback bookkeeping runs. Clearing
catalog data does not settle native uncertainty or authorize replacement work.
Only a physically settled refresh's pure denial of its exact original typed
scope snapshot, with no commit or cleanup debt, stays local to that source;
its original waiters retain the refusal, but it cannot Hold independent agents.
This is the same canonical locality predicate used for registered Codex turns,
never permission for the stopped owner to resume. Source admission, deferred
factory entry and catalog publication all require original live runtime custody.
Ordinary provider failures and internal cancellation without native control stay
cache misses; a withdrawing reader does not acquire or cancel the shared owner.
Refused Task admission closes the unstarted refresh coroutine without inventing
a native cleanup acknowledgment.
Forced discovery (`use_cache=False`) still observes the shared receipt before
entering discovery. It controls data reuse, not control provenance. Every catalog
publication, disk restore and outcome amendment rechecks the same receipt under
the cache lock, so an independent live caller cannot overwrite unresolved source
uncertainty that appeared while its own discovery was running.
Cache-hit completion observes that same receipt again after awaited embedding
and decision reconciliation, even when there are no availability amendments.
This check preserves the selected snapshot; it does not substitute a newer
catalog or infer permission from the independent reader's live scope.

The deadline can only cancel a step that awaits. A listing that blocks the
event loop cannot be cut off and holds every other route's step and deadline
with it, so adapter listings use asynchronous clients: the Google and Vertex
adapters list through the route's `google-genai` client's `aio` surface.

---

## Streaming, tools, and the honesty layer

The streaming response surface has three modes, all routed through `LLMService`:

| Method | Yields | Used when |
|---|---|---|
| `get_streaming_response(...)` | `str` chunks | Plain streaming, no tools. |
| `stream_with_messages(messages=...)` | `str` chunks | Post-tool synthesis (the orchestrator already injected the `tool` messages). |
| `stream_with_tool_detection(messages=, tools=)` | `Union[str, ToolCallStarted, LLMResponse]` (tagged union) | Single-call streaming + tool detection. The default chat path. |

The third one is the load-bearing path. Adapters yielding `ToolCallStarted` (since SDK 0.7) at the moment a tool call first appears in the provider stream — *before* arguments finish accumulating — drive the constitutional honesty layer:

- The framework emits an in-band sentinel (`\x1eKESTREL:REVISE:<json>\x1e`) on `/api/agent/stream` so the chat client can retract the in-flight bubble.
- A parallel SSE `revising` event fires on `/api/agent/notifications/sse` as a reliability backup.
- The audit hook (when enabled) reads the marker boundary as the deterministic divider between pre-tool prose and post-tool synthesis, and the resulting `HookInput.pre_tool_prose` / `tool_calls` / `tool_results` (SDK 0.9 fields) feed `analyze_narration` to catch confident-lie patterns even when the audit LLM is unavailable.

**End-to-end pipeline + guarantees:** see [llm/HONESTY_LAYER.md](llm/HONESTY_LAYER.md).
**Adapter author's view + per-provider marker timing:** see [llm/PROVIDER_PLUGINS.md](llm/PROVIDER_PLUGINS.md).

The mandate-restricted no-silent-fallback rule applies here: when the user pinned a route (`mandate_restricted = providers_to_use length == 1`), the streaming loop fails loudly with `LLMStreamingError` rather than falling through to a different vendor. Multi-route default chains still retry through the list, but the fallback happens in server logs — never by injecting a `[Route X unavailable, trying next...]` note into the stream where it would corrupt the agent's response.

---

## Mandate preference schema

The user's "which model" selection is stored per agent as a `{vendor, model, route?}` dict. `vendor` and `route` may be null; `model` is the model ID.

Persistence lives in `agent_metadata.model_preference` as JSON:

```json
{"vendor": "anthropic", "model": "<model-id>", "route": "plan"}
```

- **Vendor unset:** routing tries all vendors for this model in `route_priority` order.
- **Route unset:** first configured route for that vendor is used.
- **Both set:** exactly that `<vendor>:<route>` route is used, no fallback unless `_mandate_fallbacks` is configured.

Stale rows using the deprecated `{model, provider}` shape are dropped silently on load. The agent starts with no mandate; user re-selects via UI once.

---

## Routing

All routing funnels through `LLMService.resolve_provider_routing`:

1. **Explicit `model_override`**:
   - `"<vendor>/<model>"` — vendor filter, specific model.
   - `"<vendor>:<route>/<model>"` — exact route, specific model.
   - `"<model>"` — model only, all routes try it.
2. **Mandate preference** — same semantics as model_override, read from `_mandate_preference`.
3. **Default** — all routes, in `route_priority` order.

`force_local_only=True` filters to routes where `is_local=True`.

Nothing in routing code hardcodes vendor or model names. The `_filter_providers_by_selector` helper matches on the `vendor` attribute (vendor-only selector) or the composite `name` attribute (full route key).

---

## Context payload boundary

The LLM service owns route resolution and provider transformation; it does not
own conversation selection, retrieval policy, or pruning. The canonical
current contract for those stages is
[Kestrel Context Management Contract](CONTEXT_SYSTEM_DESIGN.md).

Before an LLM call, the selected route-qualified model identity is passed into
context budgeting. A route cap can therefore constrain the input window even
when the underlying model advertises a larger limit. Context assembly produces
generic messages and tool schemas; the selected adapter then maps those inputs
to provider-native system fields, roles, content blocks, cache controls, and
stateful-thread semantics.

This boundary is especially visible on `openai:plan`: Kestrel budgets and
selects a turn payload, while Codex retains a separate server-side thread.
Per-turn context projection and server thread occupancy are distinct
measurements. `/api/agent/context-status` can report both when data is
available. Its full dry-run is byte/token-equivalent to Kestrel's production
context plan, but it is not a provider-payload receipt: provider framing and
the stateful server-side thread remain outside that plan.

---

## API surface

| Endpoint | Returns / accepts |
|---|---|
| `GET /api/models` | `{by_vendor: {openai: [...], anthropic: [...], ...}, routes: [{vendor, route, model, is_local}, ...], default, featured, all, count}` |
| `GET /api/model/current` | `{vendor, route, model, model_name}` |
| `POST /api/model/set` | body: `{vendor?, route?, model}`, or combined `{"model": "<vendor>[:<route>]/<model>"}` |

The frontend groups the dropdown by vendor. When a vendor has more than one route, the UI exposes a route selector; otherwise it's hidden.

---

## Visibility

No maintained "always show" or "always hide" allowlists. Curation is **computed from provider-supplied metadata** — chiefly the `created_at` timestamp every adapter already reads (OpenAI `created`, Anthropic `created_at`). Recency is read from that timestamp, **not** by parsing a version number out of the model id. Name-parsing is brittle: it can't order codename tiers (Anthropic `opus`/`sonnet`/`haiku`, OpenAI's `gpt-5.6` Sol/Terra/Luna) and ranks a number-less id below `gpt-3.5`. Ranking on `created_at` is naming-agnostic — the newest release wins however the vendor names it. (#2015)

### Category (computed)

`category != "chat"` is filtered out of chat dropdowns. Resolution order: explicit `[categories.*]` in `model_catalog.toml` → **id-pattern inference** (catches the `tts`/`whisper`/`transcribe`/`realtime`/`image`/`video` models that OpenAI's API stamps as plain `chat`) → adapter-reported category.

### Deprecation (computed)

- Vendor-reported `description` contains "deprecated" | "legacy" | "will be retired" → `is_deprecated`.
- **Age-based:** a chat model older than `deprecate_after_days` *when a newer chat model exists for the vendor* → `is_deprecated`. Computable, zero-maintenance — this is what demotes `gpt-3.5-turbo`/`gpt-4` even though OpenAI's API returns no description. Deprecated models stay selectable under "Show all"; they're only kept out of the featured set.

### Featured (computed, recency-gated)

Per vendor, the featured set is the **top `featured_per_vendor` chat models by `created_at`**, excluding hidden/deprecated/preview, and preferring an undated alias over its dated snapshots. `is_canonical_alias` is still detected (to break alias-vs-snapshot ties) but no longer auto-features on its own — "undated lineage root" is forever-true and used to float stale roots (`gpt-3.5-turbo`, `gpt-4`) into the dropdown indefinitely. Legacy `[featured]` membership and `frecency_score > 0` remain additive promotions.

The selector defaults to this featured set (a clean handful) with an explicit **"Show all N models"** expander; each vendor switch resets to the featured view.

### Emergency overrides

`model_catalog.toml` carries a `[hidden]` section keyed by vendor. Empty by default. Use only when a vendor mislabels something. There is **no** `[always_show]` or `[pinned]` counterpart — pinning a specific ID is user-state, not catalog-state.

### Two dials, not lists — `[visibility]` in `model_catalog.toml`

- `featured_per_vendor` (default `8`) — cap on how many models per vendor are featured. Hard bound against the "everything featured" regression.
- `deprecate_after_days` (default `365`) — age past which a superseded chat model is marked deprecated.

---

## Embeddings

Model discovery already classifies embedding models with
`ModelCategory.EMBEDDING`, and the UI/filtering logic keeps those models out of
chat-model dropdowns. Runtime embedding execution is separate from chat model
discovery: storage asks the active provider route for embeddings only when that
route advertises embedding support.

Current execution truth as of 2026-05-31:

- The SDK LLM contract includes `aembed()`, `aembed_batch()`,
  `ProviderCapabilities.supports_embeddings`, `embedding_model`, and
  `embedding_dim`.
- `kestrel_sovereign/llm/embedding_service.py` still keeps the legacy
  Ollama-backed `EmbeddingService`, but saved-items and RAG now use
  `ProviderEmbeddingService` through `LLMService.get_embedding_service()`.
- OpenAI, Google/Gemini, Vertex, and Ollama advertise embedding capability in
  tree. Anthropic does not. OpenRouter is disabled by default and advertises
  embeddings only when its route explicitly configures `embedding_model` and
  `embedding_dim`; otherwise storage falls back to keyword/BM25/LIKE search
  rather than silently using an unrelated embedding service.
- RAG and saved-item vector search consume embeddings through storage/vector
  backends once embeddings are written.
- Provider embedding lookup borrows an explicitly supplied `LLMService`;
  without that lifecycle owner it returns `None`, never constructs a private
  service from ambient credentials. Standalone storage callers needing
  embeddings must construct, inject and close their service themselves.
- `conversation_history.embedding_vec` exists as SQLAlchemy/vector storage
  groundwork, but the current `MemoryRetriever` semantic score still uses
  keyword/concept overlap.

The adapter boundary shape is common: one embedding is `Optional[list[float]]`,
and batch embeddings return one result slot per input text. The vectors are not
semantically interchangeable across providers or embedding models. Operators
who switch embedding models need to re-embed existing rows or ensure vector
columns match the new dimensions.

---

## Decisions

Typed `choice` / `score` / `noul` questions evaluated against a state
(`/v1/systemone`) are a third modality beside chat and embeddings. Unlike
embeddings, which bind to one route to keep a single vector space, decisions
are stateless and are routed per call through `LLMService.decide`. The
canonical spec, covering contract, discovery, routing, privacy, fit and
calibration, is [llm/DECISIONS.md](llm/DECISIONS.md) (draft, epic #3424).

---

## No hardcoded model IDs in code

Model identifiers must never appear as literals inside `kestrel_sovereign/**/*.py`, `endpoints/**/*.py`, or frontend JS.

### Allowed locations

- `model_catalog.toml` — `featured_models`, `display_overrides`, `[hidden]`. Config.
- `model_mandate.toml` — `defaults.preferred`, `defaults.cheap_model`, role mandates. Config.
- `kestrel.toml` `[llm]` (or `[llm.vendors.*.routes.*]`) — `selection_hints` (substring patterns, not IDs), `model = "auto"`. Config.
- Parameterized test fixtures — documented as historical examples.
- This spec, when giving an example — marked as "example as of YYYY-MM; consult discovery" so the doc never becomes a source of truth for specific IDs.

### Not allowed

- `if model == "<some-id>": ...` in adapter or service code.
- Enum members or module constants naming specific models.
- Default values that substitute a literal ID when config is missing (log a warning and let discovery pick instead).
- Per-vendor ID lists in code.

Verification:

```bash
rg -n '"(gpt-|claude-|gemini-|llama[0-9]|mistral-|deepseek-|qwen|kimi-)' \
  kestrel_sovereign/ endpoints/ \
  --glob '!*.toml' --glob '!*.md'
```

New hits in service / adapter / endpoint code fail review.

---

## Related files

- [provider_registry.py](../../kestrel_sovereign/llm/provider_registry.py) — vendor/route initialization.
- [service.py](../../kestrel_sovereign/llm/service.py) — `LLMService`, `_mandate_preference`, `resolve_provider_routing`.
- [model_discovery.py](../../kestrel_sovereign/llm/model_discovery.py) — per-vendor discovery, dispatcher.
- [model_catalog.py](../../kestrel_sovereign/llm/model_catalog.py) — enrichment, overrides.
- [mandate.py](../../kestrel_sovereign/llm/mandate.py) — selector resolution for role mandates.
- [endpoints/models.py](../../kestrel_sovereign/endpoints/models.py) — `/api/models`, `/api/model/current`, `/api/model/set`.
- [agent/model_preference.py](../../kestrel_sovereign/agent/model_preference.py) — persistence.

## Related architecture docs

- **[CONTEXT_SYSTEM_DESIGN.md](CONTEXT_SYSTEM_DESIGN.md)** — canonical
  current-state context contract from conversation persistence through
  budgeting, retrieval, pruning, provider rendering, and diagnostics.
- **[llm/PROVIDER_PLUGINS.md](llm/PROVIDER_PLUGINS.md)** — adapter contract surface for third-party plugin authors. The `kestrel-sovereign-sdk` boundary, marker emission rules, conformance helpers.
- **[llm/DECISIONS.md](llm/DECISIONS.md)** — the decisions modality (typed choice / score / noul over `/v1/systemone`): contract, discovery, per-call routing, privacy, calibration. Draft, epic [#3424](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3424).
- **[llm/HONESTY_LAYER.md](llm/HONESTY_LAYER.md)** — streaming honesty enforcement: `ToolCallStarted` markers, in-band revise sentinel on `/api/agent/stream`, parallel SSE backup, deterministic narration check in the audit hook. Closes [#1042](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1042).

## Related decisions

- CLAUDE.md "Model Selection System — What NOT to Do" — the case study this refactor closes out.
- GitHub epic [#688](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/688) — the umbrella ticket for this architecture change.
