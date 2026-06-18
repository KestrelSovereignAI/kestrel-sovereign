---
type: Issue Body
title: Introduce stable module descriptors above feature discovery
description: 'Parent epic: #413 Depends on: kernel boundary doc'
resource: /docs/audit/issues/413-02-module-descriptors.md
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

# Introduce stable module descriptors above feature discovery

Parent epic: #413
Depends on: kernel boundary doc

## Problem

Current discovery loads modules by import scanning and class-name conventions. Core behavior also reaches for named features directly.

That makes refactors brittle and keeps the runtime coupled to Python class names rather than stable capability identities.

## Goal

Add a descriptor layer that lets the runtime reason about modules by stable id and declared contracts instead of class-name assumptions.

## Scope

- introduce `ModuleDescriptor` or equivalent metadata structure
- support stable module ids distinct from Python class names
- allow modules to declare:
  - provided capabilities
  - required capabilities
  - whether they contribute tools, routers, startup hooks, export fragments
- preserve compatibility with current feature discovery during migration
- add tests proving descriptor-based discovery works

## Non-goals

- removing the current feature discovery system in one pass
- rewriting every existing feature to a new interface immediately

## Acceptance criteria

- core can load module metadata by stable id
- module loading no longer depends purely on class name identity
- at least one real feature is discoverable through the descriptor path
- compatibility with existing feature loading is preserved

## References

- epic: [#413](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/413)
- discovery code: [`kestrel_sovereign/features/__init__.py`](kestrel_sovereign/features/__init__.py)
- base feature abstraction: [`kestrel_sovereign/features/base.py`](kestrel_sovereign/features/base.py)
- current feature inventory: [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md)

## Talon note

Optimize for migration safety. This issue succeeds if the runtime gains a stable metadata layer without losing the working system.
