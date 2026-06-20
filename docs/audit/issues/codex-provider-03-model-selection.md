---
type: Issue Body
title: Integrate `openai_plan` with sovereign model selection and mandate flows
description: 'Parent: OpenAI plan provider runtime adapter'
resource: /docs/audit/issues/codex-provider-03-model-selection.md
tags:
- docs
- audit
- issue-body
timestamp: '2026-06-18T00:00:00Z'
status: snapshot
owner: documentation
canonical: false
generated: false
privacy: public
---

# Integrate `openai_plan` with sovereign model selection and mandate flows

Parent: OpenAI plan provider runtime adapter

## Problem

Even if an OpenAI plan adapter exists, it will remain second-class unless model preference, routing, and mandate-aware selection can reason about it consistently.

## Goal

Make `openai_plan` a coherent participant in sovereign’s model selection path.

## Scope

- integrate OpenAI plan provider naming into model preference flow
- support explicit selections like `openai_plan/gpt-5.4` if appropriate
- ensure current-model reporting and routing behave sensibly
- update any provider/mandate metadata that assumes only current provider families

## Acceptance criteria

- sovereign can persist and load an OpenAI-plan-backed model preference
- current-model introspection behaves coherently
- routing logic does not treat the subscription-backed provider as an alien special case

## References

- `kestrel_sovereign/agent/model_preference.py`
- `kestrel_sovereign/features/model/feature.py`
- `kestrel_sovereign/llm/service.py`
- `kestrel_sovereign/llm/mandate.py`

## Talon note

Keep this issue about routing coherence, not adapter implementation.
