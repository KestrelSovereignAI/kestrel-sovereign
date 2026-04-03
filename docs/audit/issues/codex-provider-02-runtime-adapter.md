# Add a first-pass `codex_provider` runtime adapter to `kestrel-sovereign`

Parent: Codex provider design issue

## Problem

Sovereign has no first-class runtime provider path for Codex yet, even though Codex is now a real execution lane elsewhere in the ecosystem.

## Goal

Add a minimal but real `codex_provider` / `codex_adapter` path to the sovereign runtime.

## Scope

- add a Codex adapter module under `kestrel_sovereign/llm/`
- register it in `provider_registry.py`
- support config-driven provider initialization
- use the local Codex CLI as the first implementation surface
- support text generation path needed for normal agent runtime

## Non-goals

- vision/image review support
- full parity with every OpenAI/Anthropic feature
- solving all session persistence semantics on day one

## Acceptance criteria

- `provider_priority = ["codex", ...]` is a valid config path
- sovereign can initialize the Codex provider when the CLI/session is available
- the adapter can complete a basic text-generation smoke path
- failures degrade clearly when Codex CLI/auth is unavailable

## References

- `kestrel_sovereign/llm/provider_registry.py`
- `kestrel_sovereign/llm/adapter.py`
- `kestrel_sovereign/llm/service.py`
- `kestrel-talon` Codex backend implementation

## Talon note

This is the first-pass adapter issue. Favor a narrow, reliable text path over broad unsupported magic.
