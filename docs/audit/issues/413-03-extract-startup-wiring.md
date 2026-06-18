---
type: Issue Body
title: Extract feature-specific startup wiring out of `KestrelAgent.initialize()`
description: 'Parent epic: #413 Depends on: module descriptor groundwork'
resource: /docs/audit/issues/413-03-extract-startup-wiring.md
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

# Extract feature-specific startup wiring out of `KestrelAgent.initialize()`

Parent epic: #413
Depends on: module descriptor groundwork

## Problem

`KestrelAgent.initialize()` is still a behavior sink. It handles storage/bootstrap concerns and also coordinates named feature behavior such as:

- `SpawnFeature` pre-exploration
- `SecurityFeature` tool registration
- reflection hook wiring
- default schedule setup
- direct named references for model and MCP features

This makes the composition root too smart and forces developers to understand too much at once.

## Goal

Shrink `KestrelAgent.initialize()` toward a true composition root plus generic lifecycle orchestration.

## Scope

- identify the feature-specific wiring currently hardcoded in startup
- move that wiring behind module lifecycle hooks or runtime bootstrap helpers
- reduce direct class-name checks in startup
- preserve current runtime behavior

## Suggested targets

- `kestrel_sovereign/kestrel_agent.py`
- new runtime bootstrap helper if needed
- tests covering startup order and preserved behavior

## Acceptance criteria

- startup no longer hardcodes multiple named features for normal lifecycle behavior
- feature-specific post-load logic is reachable through generic hooks or descriptor-driven registration
- existing behavior remains intact for spawn, security, reflection, and schedules

## References

- epic: [#413](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/413)
- heavy startup path: [`kestrel_sovereign/kestrel_agent.py`](kestrel_sovereign/kestrel_agent.py)
- dynamic tools: [`kestrel_sovereign/agent/tool_registry.py`](kestrel_sovereign/agent/tool_registry.py)

## Talon note

This is a seam issue, not a hero refactor. Favor small extractions with tests over a giant rewrite.
