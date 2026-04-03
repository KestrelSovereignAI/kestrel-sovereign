# Add a first-slice design for a `codex_provider` in `kestrel-sovereign`

## Problem

The ecosystem now has a working Codex execution lane in `kestrel-talon`, but `kestrel-sovereign` still treats OpenAI, Anthropic, Claude Max, Ollama, and others as first-class runtime providers while Codex remains external.

If Codex is going to become a first-class sovereign runtime option, the codebase needs a small design truth first rather than an ad hoc subprocess hack.

## Goal

Write the first-slice design for a `codex_provider` / `codex_adapter` path in the sovereign LLM runtime.

## Scope

- define what a `codex_provider` means in sovereign terms
- define how it differs from:
  - `openai`
  - `anthropic`
  - `claude_max`
- define the auth/session model assumptions
- define what the first implementation should and should not do
- identify where it plugs into provider registry, adapter layer, and model preference flow

## Suggested output

- `docs/architecture/CODEX_PROVIDER.md`

## Acceptance criteria

- a canonical design doc exists
- the doc names concrete files and integration points
- the first slice is explicitly scoped to text/runtime use, not all possible Codex features

## References

- `kestrel_sovereign/llm/provider_registry.py`
- `kestrel_sovereign/llm/openai_adapter.py`
- `kestrel_sovereign/llm/claude_max_adapter.py`
- `kestrel-talon` Codex backend work in sibling repo

## Talon note

Do not implement the provider in this issue. This is the architecture-setting first move.
