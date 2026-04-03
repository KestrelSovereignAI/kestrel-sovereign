## Problem

As more backends become first-class in sovereign, user-facing and operator-facing status surfaces risk becoming provider-shaped lies.

We already saw this drift in `kestrel-talon`, where status comments reported a generic Claude-shaped model field even when Codex was the real execution lane.

Sovereign should not repeat that mistake for agent state, diagnostics, or model-selection surfaces.

## Goal

Audit and correct sovereign status surfaces so they report the active provider/model truthfully for `openai`, `claude_max`, and `codex`.

## Scope

- identify user-facing or operator-facing status endpoints/messages that report provider/model
- ensure they reflect the active routed provider, not just config defaults
- add focused tests around at least one surface that previously could drift

## Acceptance criteria

- provider/model reporting is backend-honest
- no surface silently reports a default provider when another backend is active

## References

- `kestrel_sovereign/features/model/feature.py`
- `kestrel_sovereign/llm/service.py`
- `kestrel-talon` issue #9 for the analogous problem
