---
type: Audit Ledger
title: Feature Module Matrix
description: 'Discovery rule: [`kestrel_sovereign/features/__init__.py`](../../kestrel_sovereign/features/__init__.py)
  scans single-file modules, package `__init__.py`, and package `feature.py`.'
resource: /docs/audit/FEATURE_MODULE_MATRIX.md
tags:
- docs
- audit
- audit-ledger
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: public
---

# Feature Module Matrix

Discovery rule: [`kestrel_sovereign/features/__init__.py`](../../kestrel_sovereign/features/__init__.py) scans single-file modules, package `__init__.py`, and package `feature.py`.

## Discoverable modules

- `audit_anchor`
- `bootstrap`
- `bridge`
- `channels`
- `code_edit`
- `compute`
- `consent`
- `constitution`
- `context`
- `council`
- `delivery`
- `deploy`
- `gcp_compute`
- `github`
- `heartbeat`
- `identity`
- `keys`
- `mcp`
- `memory`
- `memory_agency`
- `model`
- `peers`
- `reflection`
- `runpod`
- `save`
- `scheduler`
- `security`
- `sovereignty`
- `state_of_mind`
- `strategic_memory`
- `tasks`
- `vastai`
- `visual_identity`
- `wallet`
- `web_search`
- `webhooks`
- `wellness`

## Reconciliation note

The old catalog described `28` feature plugins. The corrected discovery surface is `37` modules with `37` exported `Feature` subclasses; support packages that do not export a `Feature` subclass should not be counted as discoverable features.

Direct proof status for these modules now lives in [`FEATURE_PROOF_MATRIX.md`](./FEATURE_PROOF_MATRIX.md).
