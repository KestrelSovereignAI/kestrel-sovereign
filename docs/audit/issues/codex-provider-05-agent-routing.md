## Problem

`kestrel-sovereign` can persist an agent-specific model preference, but the practical runtime story is still fuzzy when the target backend is not a normal API provider.

For Nellie specifically, we now have three distinct realities:

- `openai` via API key
- `claude_max` via OAuth token / Max subscription
- planned `codex` via local CLI/session

Without a clear routing contract, agent identity can say one thing while the runtime silently uses another.

## Goal

Make agent-specific provider and model routing explicit for backends like `claude_max` and `codex`.

## Scope

- define how persisted agent preference should map to runtime provider selection
- verify `provider=model` preferences survive startup and reload
- define what should happen when a preferred provider is unavailable
- document the difference between:
  - provider preference
  - model preference
  - fallback order
- add focused contract tests for agent-specific routing

## Acceptance criteria

- per-agent backend selection is explicit and test-covered
- unavailable providers fail honestly or fall back according to a documented rule
- Nellie can be pinned to `claude_max` or `codex` without ambiguity

## References

- `kestrel_sovereign/llm/service.py`
- `kestrel_sovereign/llm/provider_registry.py`
- `kestrel_sovereign/features/model/feature.py`
- `agent_data/nellie/`
