---
type: Issue Body
title: Move toward capability-mounted routers and enforce modular seams with tests
description: 'Parent epic: #413 Depends on: descriptor groundwork and startup extraction'
resource: /docs/audit/issues/413-05-capability-routers-and-seam-tests.md
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

# Move toward capability-mounted routers and enforce modular seams with tests

Parent epic: #413
Depends on: descriptor groundwork and startup extraction

## Problem

`server.py` still statically assembles a broad always-on HTTP surface, and the codebase lacks strong regression tests against slipping back into monolithic coupling.

If modules are truly optional, router contribution and seam enforcement need to be testable.

## Goal

Make capability-owned routes more modular and add seam tests that fail when new core coupling is introduced.

## Scope

- support optional router contribution from modules where practical
- reduce static server knowledge of feature-owned routes
- add tests that assert:
  - reduced-capability boots still work
  - disabling a module removes its router cleanly
  - core does not grow new direct named-feature dependencies casually

## Suggested targets

- `server.py`
- `endpoints/`
- module/router registration helpers
- seam-focused tests under `tests/unit/` or `tests/integration/`

## Acceptance criteria

- at least one feature-owned router is mounted through a more modular registration path
- the app still boots coherently when that module is disabled
- new seam tests exist and would catch future regressions toward monolithic coupling

## References

- epic: [#413](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/413)
- server assembly: [`server.py`](server.py)
- maintained route inventory: [`KESTREL_FEATURES.md`](KESTREL_FEATURES.md)

## Talon note

This issue is partly architecture and partly guardrails. The tests are as important as the code movement.
