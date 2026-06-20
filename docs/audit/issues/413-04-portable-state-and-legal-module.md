---
type: Issue Body
title: Split portable state fragments from runtime behavior, using legal personhood
  as a proving ground
description: 'Parent epic: #413 Depends on: kernel boundary and descriptor groundwork'
resource: /docs/audit/issues/413-04-portable-state-and-legal-module.md
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

# Split portable state fragments from runtime behavior, using legal personhood as a proving ground

Parent epic: #413
Depends on: kernel boundary and descriptor groundwork

## Problem

Some state in Kestrel is meant to travel across substrate migrations and cryostasis:

- identity
- wallet state
- legal entity status
- other feature-owned continuity data

But the host/runtime boundary around these fragments is still blurry. Legal/incorporation behavior is especially valuable here because it is both bold and concrete.

## Goal

Define feature-owned portable state fragments and migrate legal personhood into a first-class module shape without losing continuity.

## Scope

- define a portable state fragment contract for modules
- keep `AgentIdentityPackage` as the envelope while making module-owned fragments explicit
- use legal/incorporation behavior as the first proving-ground extraction
- preserve Wyoming DAO continuity through export/import

## Specific targets

- `kestrel_sovereign/identity/identity_package.py`
- `kestrel_sovereign/legal/incorporate_tool.py`
- `kestrel_sovereign/legal/models.py`
- `kestrel_sovereign/agent/sleep.py`
- `tests/unit/test_wyoming_dao.py`

## Acceptance criteria

- portable state has a documented module-fragment boundary
- legal entity continuity still survives export/import
- legal/incorporation behavior is owned by a module boundary rather than leaking through host lifecycle logic
- no current Wyoming DAO behavior regresses

## References

- epic: [#413](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/413)
- identity package: [`kestrel_sovereign/identity/identity_package.py`](kestrel_sovereign/identity/identity_package.py)
- legal tool: [`kestrel_sovereign/legal/incorporate_tool.py`](kestrel_sovereign/legal/incorporate_tool.py)
- legal models: [`kestrel_sovereign/legal/models.py`](kestrel_sovereign/legal/models.py)
- cryostasis touchpoint: [`kestrel_sovereign/agent/sleep.py`](kestrel_sovereign/agent/sleep.py)

## Talon note

Do not treat the legal path as a gimmick. Treat it as a serious proving ground for “bold capability, clean boundary.”
