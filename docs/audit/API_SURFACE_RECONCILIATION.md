---
type: Audit Ledger
title: API Surface Reconciliation
description: The legacy catalog under-described the live HTTP surface.
resource: /docs/audit/API_SURFACE_RECONCILIATION.md
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

# API Surface Reconciliation

The legacy catalog under-described the live HTTP surface.

Important examples of drift:

- OAuth auth routes were omitted
- `/health/detailed` was omitted
- multi-agent agent-management routes under `/api/agents` were omitted
- wallet, IPFS, and model-setting routes were under-described
- OpenAI-compatible endpoints need explicit contract coverage, not just presence in a catalog

The canonical fix is to inventory route families from `server.py` and `endpoints/` instead of relying on hand-maintained totals.
