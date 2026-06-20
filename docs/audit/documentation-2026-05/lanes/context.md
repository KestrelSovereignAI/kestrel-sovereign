---
type: Review Lane
title: Context
description: Review prompt for the Context lane of the May 2026 documentation audit.
resource: /docs/audit/documentation-2026-05/lanes/context.md
tags:
- audit
- documentation
- may-2026
- review-lane
timestamp: 2026-05-30 00:00:00+00:00
status: snapshot
owner: documentation-audit
canonical: false
generated: false
privacy: public
---


# Lane Brief: Context

Goal: reconcile documentation for prompt assembly, token budgets, context pruning, canonical history, rendered provider transport, retrieval insertion, and diagnostics.

Start with:

- `docs/architecture/CONTEXT_SYSTEM_DESIGN.md`
- `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md`
- `docs/architecture/LLM_SERVICE_ARCHITECTURE.md`
- `docs/generated/FEATURES_developer.md`
- `kestrel_sovereign/agent/context_manager.py`
- `kestrel_sovereign/agent/context_builder.py`
- `kestrel_sovereign/agent/token_budget.py`
- `kestrel_sovereign/endpoints/agent.py`

Check for:

- old history model claims
- missing canonical vs rendered transport distinction
- stale context-window or token budget descriptions
- docs that ignore feature-prompt caps or route caps
- retrieval behavior described without current gating/floor rules
- endpoint or CLI diagnostics not documented

Report to: `reports/context_report.md`

