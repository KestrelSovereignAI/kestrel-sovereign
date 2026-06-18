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
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
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

## API surface

| Endpoint | Returns / accepts |
|---|---|
| `GET /api/models` | `{by_vendor: {openai: [...], anthropic: [...], ...}, routes: [{vendor, route, model, is_local}, ...], default, featured, all, count}` |
| `GET /api/model/current` | `{vendor, route, model, model_name}` |
| `POST /api/model/set` | body: `{vendor?, route?, model}`, or combined `{"model": "<vendor>[:<route>]/<model>"}` |

The frontend groups the dropdown by vendor. When a vendor has more than one route, the UI exposes a route selector; otherwise it's hidden.

---

## Visibility

No maintained "always show" or "always hide" allowlists. Capability and usage signals drive curation.

### Auto-hide (computed from discovered metadata)

- `category != "chat"` — filtered out of chat dropdowns. Covered by `[categories.embedding|image|audio|completion]` in `model_catalog.toml`.
- Vendor-reported `description` contains "deprecated" | "legacy" | "will be retired" — marked hidden at enrichment.
- Present in the previous discovery cache but absent from today's run — marked deprecated (structural signal; vendor retired it).

### Auto-feature (computed)

- `frecency_score > 0` — you've used it, it floats up.
- **Canonical alias:** the ID has no date suffix in a lineage where dated siblings exist. Pure string analysis, no config.
- Newest `created_at` in a lineage + `supports_tools` + not preview.

### Emergency overrides

`model_catalog.toml` carries a `[hidden]` section keyed by vendor. Empty by default. Use only when a vendor mislabels something. There is **no** `[always_show]` or `[pinned]` counterpart — pinning a specific ID is user-state, not catalog-state.

### Two dials, not lists

- `visibility_auto_hide_deprecated_months` — grace period before truly removing deprecated models from the cache.
- `visibility_preview_demotion_terms` — substrings that demote a model from featured (default: `["preview", "beta", "experimental", "exp"]`).

---

## Embeddings

Model discovery already classifies embedding models with
`ModelCategory.EMBEDDING`, and the UI/filtering logic keeps those models out of
chat-model dropdowns. That is model metadata, not yet the embedding execution
contract.

Current execution truth as of 2026-05-31:

- `kestrel_sovereign/llm/embedding_service.py` is still the in-tree embedding
  generator.
- That service uses Ollama's embedding API, defaulting to `nomic-embed-text`
  at 768 dimensions.
- RAG and saved-item vector search consume embeddings through storage/vector
  backends once embeddings are written.
- `conversation_history.embedding_vec` exists as SQLAlchemy/vector storage
  groundwork, but the current `MemoryRetriever` semantic score still uses
  keyword/concept overlap.

The architecture direction is to make embeddings a standard provider capability
on LLM adapters, so the storage and retrieval layers can request embeddings
through the configured LLM provider instead of a hardcoded Ollama side service.
Until that lands in code, do not document provider embedding functions as
available runtime behavior.

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
- [endpoints/models.py](../../endpoints/models.py) — `/api/models`, `/api/model/current`, `/api/model/set`.
- [agent/model_preference.py](../../kestrel_sovereign/agent/model_preference.py) — persistence.

## Related architecture docs

- **[llm/PROVIDER_PLUGINS.md](llm/PROVIDER_PLUGINS.md)** — adapter contract surface for third-party plugin authors. The `kestrel-sovereign-sdk` boundary, marker emission rules, conformance helpers.
- **[llm/HONESTY_LAYER.md](llm/HONESTY_LAYER.md)** — streaming honesty enforcement: `ToolCallStarted` markers, in-band revise sentinel on `/api/agent/stream`, parallel SSE backup, deterministic narration check in the audit hook. Closes [#1042](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1042).

## Related decisions

- CLAUDE.md "Model Selection System — What NOT to Do" — the case study this refactor closes out.
- GitHub epic [#688](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/688) — the umbrella ticket for this architecture change.
