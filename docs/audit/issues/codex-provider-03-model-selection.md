# Integrate `codex_provider` with sovereign model selection and mandate flows

Parent: Codex provider runtime adapter

## Problem

Even if a Codex adapter exists, it will remain second-class unless model preference, routing, and mandate-aware selection can reason about it consistently.

## Goal

Make `codex_provider` a coherent participant in sovereign’s model selection path.

## Scope

- integrate Codex provider naming into model preference flow
- support explicit selections like `codex/gpt-5.4` if appropriate
- ensure current-model reporting and routing behave sensibly
- update any provider/mandate metadata that assumes only current provider families

## Acceptance criteria

- sovereign can persist and load a Codex-backed model preference
- current-model introspection behaves coherently
- routing logic does not treat Codex as an alien special case

## References

- `kestrel_sovereign/agent/model_preference.py`
- `kestrel_sovereign/features/model/feature.py`
- `kestrel_sovereign/llm/service.py`
- `kestrel_sovereign/llm/mandate.py`

## Talon note

Keep this issue about routing coherence, not adapter implementation.
