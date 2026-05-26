# LLM Provider Capabilities

Kestrel tracks adapter-level capabilities separately from per-model
`ModelInfo`. Both live in `kestrel-sovereign-sdk`: `ModelInfo` describes
discovered models, while provider capabilities describe how an adapter can
speak to its upstream API.

Adapters may implement:

```python
def provider_capabilities(self) -> ProviderCapabilities | dict:
    ...
```

Adapters return `kestrel_sdk.llm.ProviderCapabilities`. During migration,
third-party provider packages may return the same shape as a plain dict; core
normalizes that into the SDK dataclass before putting it on `ProviderInfo`.

The core fields are:

- `supports_tools`: adapter can request tool/function calls.
- `supports_streaming`: adapter can stream assistant text.
- `supports_vision`: adapter can send image input.
- `supports_structured_output`: adapter can request schema-constrained output.
- `structured_output_mode`: `json_schema`, `json_object`, `provider_native`,
  `schema_format`, `tool_forced`, `none`, or `unknown`.
- `tool_streaming_mode`: `native_delta`, `nonstream_fallback`,
  `inline_executor`, `none`, or `unknown`.
- `vision_input_mode`: provider wire format for images.
- `model_dependent`: capability names that still vary by model or upstream
  route.
- `notes`: short operator-facing caveats.

Current high-level matrix:

| Provider | Tools | Streaming | Vision | Structured Output | Notes |
| --- | --- | --- | --- | --- | --- |
| OpenAI | yes | yes | model-dependent | `json_schema` | Native OpenAI request shapes. |
| OpenRouter | model-dependent | yes | model-dependent | model-dependent `json_schema` | Upstream model support is authoritative. |
| Anthropic / Claude Max | yes | yes | yes | `tool_forced` | Structured output uses a synthetic forced tool. |
| Google Gemini direct | yes | yes | yes | no | `response_format` is not wired into this adapter yet. |
| Vertex AI | model-dependent | yes | model-dependent | model-dependent `provider_native` | Uses Gemini/Vertex `response_schema`. |
| Ollama | model-dependent | yes | model-dependent | model-dependent `schema_format` | Depends on local model capabilities. |
| Codex app-server | yes | yes | no | no | Tools are dynamic app-server events. |
| Mock | no | yes | no | no | Demo text only. |
