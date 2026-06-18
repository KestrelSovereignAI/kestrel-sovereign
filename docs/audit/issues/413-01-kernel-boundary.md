---
type: Issue Body
title: Define the modular runtime kernel boundary for `kestrel-sovereign`
description: 'Parent epic: #413'
resource: /docs/audit/issues/413-01-kernel-boundary.md
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

# Define the modular runtime kernel boundary for `kestrel-sovereign`

Parent epic: #413

## Problem

`kestrel-sovereign` has feature discovery, feature base classes, and dynamic tool promotion, but it still lacks a crisp written boundary between:

- what the permanent runtime kernel owns
- what capability modules own

Without that boundary, new features will keep leaking host knowledge into core startup, state ownership, routing, and lifecycle wiring.

## Goal

Write the canonical modular-runtime boundary document and back it with a small contract test scaffold.

This issue is about architecture truth-setting, not large code movement.

## Scope

- add a short architecture doc for the modular runtime kernel
- define what belongs in permanent core vs capability modules
- define the minimum module contract concepts:
  - stable module id
  - provided capabilities
  - required capabilities
  - owned state
  - startup hooks
  - router contribution
  - export/import fragment ownership
- add initial tests or placeholders that make the intended contract legible

## Suggested outputs

- `docs/architecture/core/MODULAR_RUNTIME.md`
- `tests/unit/test_module_contracts.py`

## Acceptance criteria

- a canonical doc exists describing the runtime kernel boundary
- the doc is specific enough that future issues can implement against it
- the contract concepts above are named consistently
- there is at least one test file or scaffold asserting the new contract surface exists

## References

- epic: [#413](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/413)
- current feature framework: [`docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md`](docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md)
- feature discovery: [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py)
- feature base contract: [`kestrel_sovereign/features/base.py`](kestrel_sovereign/features/base.py)

## Talon note

Do not over-engineer the first pass. The goal is to make the future boundary explicit enough that later extraction work stops being vague.
