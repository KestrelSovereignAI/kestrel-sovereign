# Add a first-pass `openai_plan` runtime adapter to `kestrel-sovereign`

Parent: OpenAI plan provider design issue

## Problem

Sovereign has no first-class runtime provider path for the OpenAI plan lane yet, even though that subscription-backed lane is now a real execution path elsewhere in the ecosystem.

## Goal

Add a minimal but real `openai_plan` runtime path to the sovereign runtime.

## Scope

- add the subscription-backed adapter module under `kestrel_sovereign/llm/`
- register it in `provider_registry.py`
- support config-driven provider initialization
- use the local OpenAI plan session as the first implementation surface
- support text generation path needed for normal agent runtime

## Non-goals

- vision/image review support
- full parity with every OpenAI/Anthropic feature
- solving all session persistence semantics on day one

## Acceptance criteria

- `provider_priority = ["openai_plan", ...]` is a valid config path
- sovereign can initialize the OpenAI plan provider when the session is available
- the adapter can complete a basic text-generation smoke path
- failures degrade clearly when subscription auth is unavailable

## References

- `kestrel_sovereign/llm/provider_registry.py`
- `kestrel_sovereign/llm/adapter.py`
- `kestrel_sovereign/llm/service.py`
- `kestrel-talon` OpenAI plan backend implementation

## Talon note

This is the first-pass adapter issue. Favor a narrow, reliable text path over broad unsupported magic.
