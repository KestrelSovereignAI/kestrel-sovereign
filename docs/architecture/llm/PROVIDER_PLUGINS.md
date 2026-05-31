# Building a Kestrel LLM Provider Plugin

> **Audience:** developers extending Kestrel with a new LLM backend (Kimi, DeepSeek, a private API, a local server) without touching the framework codebase.
>
> **Substrate:** `kestrel-sovereign-sdk >= 0.8.0`. Plugins depend on the SDK, not the framework.

A "provider plugin" is a `pip`-installable package that:

1. Subclasses [`kestrel_sdk.llm.LLMAdapter`][LLMAdapter]
2. Registers under the `kestrel_sovereign.llm_providers` entry-point group
3. Optionally tests itself against the conformance helpers in `kestrel_sdk.testing`

That's the entire surface. No edits to `kestrel-sovereign` are required for the new vendor to show up in route initialization, model discovery, council deliberation, or service-key UI — the framework reads everything it needs off the SDK contract.

This doc walks through the contract, then the minimum-viable plugin, then the parts adapters typically want to override beyond the abstract minimum, then the conformance suite. By the end you should be able to build, test, and ship a plugin in a single afternoon.

[LLMAdapter]: https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk/blob/main/kestrel_sdk/llm/adapter.py

---

## The contract surface

Everything a plugin needs is exported from `kestrel_sdk.llm`:

```python
from kestrel_sdk.llm import (
    LLMAdapter,             # the base class you subclass
    LLMResponse, ToolCall,  # what get_response returns
    ToolCallStarted,        # streaming-with-tools event marker
    ModelInfo, ModelCategory, # what list_models returns
    ProviderInfo,           # registration record (rarely constructed by plugins)
    BackendType,            # cloud / local / remote_gpu enum
    SDK_LLM_CONTRACT_VERSION,  # currently 4; pin against this if you need to detect drift
)
```

`LLMAdapter` has **one** abstract method — `get_response`. Every other method is optional, with a sensible default (mostly `NotImplementedError` or `None`). Plugins implement only what their backend supports.

### The mandatory method

```python
async def get_response(
    self,
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    format: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Type[BaseModel]] = None,
    **kwargs: Any,
) -> LLMResponse: ...
```

The framework hands you a provider-native client (constructed during route init from your config), a concrete model id (the framework resolves `"auto"` upstream — you never see it), OpenAI-format messages, and optional tools / structured-output schema. Return an `LLMResponse`.

### The optional surface

| Method | Default | What it does |
|---|---|---|
| `get_streaming_response(...)` | raises `NotImplementedError` | yields text chunks for plain streaming |
| `get_streaming_response_with_tools(...)` | raises `NotImplementedError` | yields `Union[str, ToolCallStarted, LLMResponse]` for streaming with tool calls |
| `list_models(client)` | raises `NotImplementedError` | returns `List[ModelInfo]` for the discovery dropdown |
| `aembed(client, text, model=None)` | returns `None` | returns one embedding as `Optional[list[float]]` |
| `aembed_batch(client, texts, model=None)` | calls `aembed` per text | returns `List[Optional[list[float]]]` |
| `contribute_system_prompt(model_id, base)` | returns `base` unchanged | injects model-family-specific overlays (rare) |
| `cost_per_1m_tokens()` | `None` | `{"input": float, "output": float}` for council cost accounting |
| `substrate_type()` | `None` | `"claude" \| "gpt" \| "gemini" \| "llama" \| ...` for substrate-aware routing |
| `display_name()` | `None` | human-readable name for UI (e.g., `"Kimi"`) |
| `key_env_var()` | `None` | env var name for the API key (`"KIMI_API_KEY"`) |
| `deliberation_style()` | `None` | `"parallel"` or `"sequential"` hint for council routing |

Override the ones your backend actually has data for. Returning `None` (the default) tells the framework "fall back to a sensible default" — that's a documented and supported state.

### Embeddings

Embeddings use one common adapter boundary shape: `list[float]` for a
successful embedding and `None` when the provider cannot embed. Batch calls
return one item per input text. Declare `supports_embeddings=True`,
`embedding_model`, and `embedding_dim` from `ProviderCapabilities` when your
route has a stable default embedding model.

The vectors are not semantically interchangeable across providers or embedding
models. Kestrel's storage code sizes vector queries from the returned list, but
operators who switch from one embedding model to another must re-embed existing
rows or widen/recreate provider-specific vector columns. Providers with no
embedding API, such as Anthropic, should leave the default `aembed()` behavior;
Sovereign will degrade to keyword/BM25/LIKE fallback instead of silently using
an unrelated global Ollama service.

---

## Minimum viable plugin

A working plugin for an OpenAI-compatible backend is three files.

### `your_provider/adapter.py`

```python
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
import openai

from kestrel_sdk.llm import LLMAdapter, LLMResponse, ToolCall


class KimiAdapter(LLMAdapter):
    """OpenAI-compatible adapter for Moonshot's Kimi K2."""

    async def get_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        extra = {}
        if tools:
            extra["tools"] = tools
            extra["tool_choice"] = "auto"

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            **extra,
            **{k: v for k, v in kwargs.items()
               if k in ("max_tokens", "temperature", "top_p")},
        )
        msg = resp.choices[0].message
        usage = resp.usage

        tool_calls = None
        if msg.tool_calls:
            import json
            tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    # Malformed-JSON sentinel — see "Edge cases" below.
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    # Recommended metadata.
    def substrate_type(self) -> Optional[str]:
        return "kimi"

    def display_name(self) -> Optional[str]:
        return "Kimi (Moonshot)"

    def key_env_var(self) -> Optional[str]:
        return "KIMI_API_KEY"
```

### `your_provider/pyproject.toml`

```toml
[project]
name = "kestrel-llm-kimi"
version = "0.1.0"
dependencies = [
    "kestrel-sovereign-sdk>=0.8,<1",
    "openai>=1.40.0",  # Kimi speaks OpenAI-compat
]

[project.entry-points."kestrel_sovereign.llm_providers"]
kimi = "your_provider.adapter:KimiAdapter"
```

### User-side `kestrel.toml`

The user installing your plugin adds:

```toml
[llm.vendors.kimi.routes.api]
adapter = "KimiAdapter"
api_key_env = "KIMI_API_KEY"
base_url = "https://api.moonshot.cn/v1"
model = "auto"
```

That's it. The framework's [provider registry](../../../kestrel_sovereign/llm/provider_registry.py) finds the entry-point, instantiates `KimiAdapter`, builds an `openai.AsyncOpenAI` client pointed at the configured `base_url`, and the new vendor lights up in route discovery.

---

## Streaming with tools

If your backend can stream tool calls, implement `get_streaming_response_with_tools`. The contract is a tagged union:

```python
async def get_streaming_response_with_tools(
    self, client, model, messages, tools=None, response_format=None, **kwargs
) -> AsyncIterator[Union[str, ToolCallStarted, LLMResponse]]:
    yield "leading text "
    yield ToolCallStarted(index=0, id="call_abc", name="lookup")
    yield LLMResponse(
        content="leading text ",
        tool_calls=[ToolCall(id="call_abc", name="lookup", arguments={"q": "x"})],
    )
```

### Per-provider emission rules for `ToolCallStarted`

This is the load-bearing piece for the constitutional honesty layer (issues [#1042](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1042) layer 2 / [#1045](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1045)). Downstream the marker drives an in-band revise sentinel on the chat stream, a parallel SSE `"revising"` event (a backup signal the chat client also subscribes to, available to any other subscriber that wants it), and a deterministic narration check in the response audit hook. The full pipeline + the consumer guarantees are documented in [HONESTY_LAYER.md](HONESTY_LAYER.md). **Get the timing right or the consumer-side guarantees fall over.**

| Provider family | Fire on | `id` at emit | `name` at emit |
|---|---|---|---|
| OpenAI / OpenAI-compat | first non-null `delta.tool_calls` fragment | MAY be `None` (typical first delta carries only `index`) | MAY be `None` |
| Anthropic | `content_block_start` with `type="tool_use"` | populated | populated |
| Google / Vertex (Gemini) | first `parts` entry with `functionCall` | MAY be `None` (Gemini sometimes omits) | populated |
| Ollama | first non-null `tool_calls` field | populated | populated |
| Other | first signal that a tool call is coming, before its arguments finish accumulating | per your backend | per your backend |

**Index is provider-native, NOT positional.** Anthropic's `content_block_index` may be sparse (block 0 = text, block 1 = tool_use, block 2 = text, block 3 = tool_use → markers fire with indices `1` and `3`, not `0` and `1`). Codex's `output_index` likewise. OpenAI's delta-tool-call-index is positional by accident only.

The contract is on **stream order**, not literal index value: the order of distinct `index` values across the marker stream defines the order of the corresponding entries in the final `LLMResponse.tool_calls`. Consumers iterate markers in stream order; `tool_calls[marker.index]` is **not** the contract and would misdispatch on sparse-index providers.

`SDK_LLM_CONTRACT_VERSION = 2` pins this clarification.

### Ordering invariants

1. `ToolCallStarted` events with distinct `index` values are yielded in the order their corresponding entries appear in `LLMResponse.tool_calls`.
2. Text chunks may interleave with `ToolCallStarted` events. Anthropic mixes text and tool blocks; OpenAI may emit a leading text segment before tool deltas. Consumers handle both text-before-tool and text-during-tool.
3. The terminal `LLMResponse` is yielded after all text and `ToolCallStarted` events, exactly once for tool-call responses. Pure-text streams may terminate without one.

### Edge cases

- **Malformed JSON arguments at end-of-stream.** Yield the partial under `{"_raw": "<accumulated>"}` rather than raising. The framework reports the error to the model as a tool result so the turn doesn't crash.
- **Multiple concurrent tool calls.** Provider streams may interleave deltas across tool indices. Accumulate per-index; emit one `ToolCallStarted` per distinct index, in arrival order.
- **Empty tool_calls list.** `LLMResponse(tool_calls=[])` is treated identically to `LLMResponse(tool_calls=None)` by the dataclass's `has_tool_calls` property. Both are valid.

---

## Adapter metadata

The framework consults adapter metadata to give plugins first-class participation in features beyond the LLM call path itself:

- `cost_per_1m_tokens()` is read by the council's deliberation cost accounting. Returning `None` means "treat as unknown"; the framework falls back to a conservative paid-API default rather than guessing your backend is cheap.
- `substrate_type()` is read by identity export for the `SubstrateType` lookup. Use the family identifier (`"claude"`, `"gpt"`, `"gemini"`, `"llama"`), not your vendor name. For aggregators (you proxy multiple substrates), return `None` — the framework reads the per-model id when it needs specifics.
- `display_name()` and `key_env_var()` show up in the service-keys UI and audit logs. Override when your brand name doesn't match your package name (`"OpenRouter"` vs `"openrouter"`).
- `deliberation_style()` returns `"parallel"` (fast/cheap, good for breadth-first deliberation rounds) or `"sequential"` (slower/careful, good for single-pass) or `None` (no preference). Hint, not constraint.

None of these are required. Plugins that don't implement them work fine; the framework defaults are conservative.

---

## Conformance testing

The SDK ships a set of contract-validation helpers in `kestrel_sdk.testing`. Plugin authors use them from their own test suite to verify the adapter conforms to the streaming-with-tools contract without having to hand-roll all the rules.

```python
import asyncio
import pytest
from kestrel_sdk.testing import drain_streaming_with_tools

@pytest.mark.asyncio
async def test_kimi_emits_tool_call_started():
    adapter = KimiAdapter()
    mock_client = build_kimi_mock_client(scenario="single_tool_call")  # plugin author's job
    stream = adapter.get_streaming_response_with_tools(
        client=mock_client,
        model="kimi-k2-turbo",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    result = await drain_streaming_with_tools(stream)
    # All contract assertions ran inside drain_*().
    # Now validate the plugin-specific outcome.
    assert len(result.tool_starts) == 1
    assert result.tool_starts[0].name == "lookup"
    assert result.final_response.tool_calls[0].arguments == {"q": "hi"}
```

`drain_streaming_with_tools` validates as it drains: every yielded item is `str | ToolCallStarted | LLMResponse`; one marker per distinct `index`; terminal `LLMResponse` after all markers; stream order matches `tool_calls` order; etc. See [the helper module][testing-module] for the full list.

The intentional split: **plugin authors own the mock provider client** (each backend has its own event shape an abstraction would leak), and **the SDK owns the contract assertions** on the resulting tagged-union stream. The Wave 3 mixin retrospective concluded the structural mismatch between provider event shapes is real and shouldn't be papered over.

[testing-module]: https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk/blob/main/kestrel_sdk/testing/conformance.py

### Standalone helpers

For more granular tests:

- `assert_response_contract(response: LLMResponse)` — validates a single `LLMResponse` shape (tool_call ids/names are strings, arguments is a dict, etc.)
- `assert_tool_call_started_contract(marker: ToolCallStarted)` — validates a single `ToolCallStarted` (index is int-not-bool, id/name are str-or-None never empty string)
- `drain_streaming_text_only(stream)` — drain a stream that the test scenario asserts is text-only; reject any non-string item.

---

## What the framework does for you

After your plugin is `pip install`'d, the framework:

1. **Discovers it** via the `kestrel_sovereign.llm_providers` entry point at startup. The plugin loader validates `issubclass(YourAdapter, kestrel_sdk.llm.LLMAdapter)` against the SDK base — your adapter does **not** need to inherit from any kestrel-sovereign symbol.
2. **Instantiates it** when a route under `[llm.vendors.<your_name>.routes.<route>]` is initialized. If your adapter exposes a `create_provider(config)` classmethod, the framework calls it; otherwise it builds a default OpenAI-compatible client from `base_url` + `api_key_env` and passes that as the `client` kwarg to `get_response`.
3. **Routes traffic** through the standard `LLMService.generate()` / `stream_with_tool_detection()` paths. Your adapter sees the same shapes the in-tree adapters see.
4. **Pulls metadata** from your `cost_per_1m_tokens()`, `substrate_type()`, `display_name()`, `key_env_var()`, `deliberation_style()` for the council, identity export, and UI surfaces.
5. **Discovers models** via your `list_models(client)`. The framework caches results and surfaces them in the model dropdown.

You can ship a working plugin without implementing any of `list_models`, `get_streaming_response`, or `get_streaming_response_with_tools` — the framework gates each capability and falls back gracefully. The minimum viable plugin (just `get_response`) is fully conforming.

---

## Reference: in-tree adapters as models

The eight in-tree adapters live at [`kestrel_sovereign/llm/`][in-tree-llm]. Read them when designing your plugin:

| In-tree adapter | What's worth copying |
|---|---|
| `openai_adapter.py` | OpenAI-compat backends, streaming with delta-indexed tool calls, structured output via `response_format` |
| `anthropic_adapter.py` | Typed-event streaming, content-block-aware tool detection, prompt cache markers |
| `ollama_adapter.py` | Local backends, no API key, capability detection from model template |
| `openrouter_adapter.py` | Aggregators (substrate=None), per-model pricing on `ModelInfo` |
| `vertex_adapter.py` | Cloud auth via Application Default Credentials (no env-var key) |
| `claude_max_adapter.py` | Plan-based auth (`auth_token` instead of `api_key`), inheriting from a vendor adapter |

[in-tree-llm]: https://github.com/KestrelSovereignAI/kestrel-sovereign/tree/main/kestrel_sovereign/llm

---

## Version pins

| SDK version | Adds |
|---|---|
| `0.5.0` | `LLMAdapter`, `LLMResponse`, `ToolCall`, `ModelInfo`, `ProviderInfo`, `BackendType` |
| `0.6.0` | `cost_per_1m_tokens`, `substrate_type`, `display_name`, `key_env_var`, `deliberation_style` |
| `0.7.0` | `ToolCallStarted`, optional `get_streaming_response_with_tools` |
| `0.8.0` | `kestrel_sdk.testing` conformance helpers; `SDK_LLM_CONTRACT_VERSION = 2` (clarifies `index` semantics) |
| `0.17.0` | `ProviderCapabilities`, `ProviderInfo.capabilities`, `SDK_LLM_CONTRACT_VERSION = 3` |
| `0.18.0` | Provider-owned embeddings: `aembed`, `aembed_batch`, embedding capability metadata, `SDK_LLM_CONTRACT_VERSION = 4` |

Pin `kestrel-sovereign-sdk >= 0.18,<1` to get the full surface this doc describes.
