---
type: Issue Body
title: Codex Provider 05 Agent Routing
description: '`kestrel-sovereign` can persist an agent-specific model preference,
  but the practical runtime story is still fuzzy when the target backend is not a
  normal API provider.'
resource: /docs/audit/issues/codex-provider-05-agent-routing.md
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

# Codex Provider 05 Agent Routing

## Problem

`kestrel-sovereign` can persist an agent-specific model preference, but the practical runtime story is still fuzzy when the target backend is not a normal API provider.

For Nellie specifically, we now have three distinct realities:

- `openai` via API key
- `claude_plan` via OAuth token / Claude plan
- `openai_plan` via local/session-backed OpenAI plan auth

Without a clear routing contract, agent identity can say one thing while the runtime silently uses another.

## Goal

Make agent-specific provider and model routing explicit for backends like `claude_plan` and `openai_plan`.

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
- Nellie can be pinned to `claude_plan` or `openai_plan` without ambiguity

## References

- `kestrel_sovereign/llm/service.py`
- `kestrel_sovereign/llm/provider_registry.py`
- `kestrel_sovereign/features/model/feature.py`
- `agent_data/nellie/`
