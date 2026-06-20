---
type: Issue Body
title: '`codex_provider` execution order'
description: 1. `#422` Design first slice for `codex_provider` 2. `#423` Add first-pass
  runtime adapter 3. `#424` Integrate model selection and mandate flows 4. `#425`
  Make agent-specific ba...
resource: /docs/audit/issues/codex-provider-execution-order.md
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

# `codex_provider` execution order

## Recommended order

1. `#422` Design first slice for `codex_provider`
2. `#423` Add first-pass runtime adapter
3. `#424` Integrate model selection and mandate flows
4. `#425` Make agent-specific backend routing explicit
5. `#421` Add smoke proof and docs
6. `#426` Add Nellie backend smoke proof
7. `#427` Audit status/model honesty across backend surfaces

## Why

- design first so the provider does not become a one-off subprocess hack
- adapter second so the runtime gets a real path quickly
- model selection third so the provider becomes first-class
- agent routing fourth so persisted preferences and runtime behavior line up
- proof/docs fifth so the path is demonstrable and teachable
- status honesty last so operator surfaces match the real backend state
