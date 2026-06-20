---
type: Audit Ledger
title: Feature Audit Matrix
description: The audit is not complete until every maintained claim in `KESTREL_FEATURES.md`
  has executable proof or an explicit open gap.
resource: /docs/audit/FEATURE_AUDIT_MATRIX.md
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

# Feature Audit Matrix

| Domain | Canonical Source | Current Proof | Status |
|---|---|---|---|
| Foundation and constitution | `kestrel_sovereign/agent/constitution.py`, `kestrel_sovereign/data/KESTREL_CONSTITUTION.md` | existing unit coverage plus command-handler constitution contracts | Partial |
| Identity and continuity | `kestrel_sovereign/inception_service.py`, `kestrel_sovereign/identity/` | existing unit coverage, no full catalog proof map yet | Partial |
| Agent runtime and context | `kestrel_sovereign/kestrel_agent.py`, `kestrel_sovereign/agent/` | existing unit coverage plus direct runtime-route and consent-caller contracts, no full seam campaign yet | Partial |
| LLM routing and mandate | `kestrel_sovereign/llm/` | existing unit coverage, mandate drift history noted | Partial |
| Privacy, security, permissions | `kestrel_sovereign/privacy.py`, `endpoints/security.py`, `server.py` | auth decision-table proof added | Partial |
| Storage and memory | `kestrel_sovereign/storage.py`, `kestrel_sovereign/storage/`, `endpoints/database.py` | weak endpoint contract suite added | Partial |
| Feature module inventory | `kestrel_sovereign/features/__init__.py` | discovery reconciliation plus first-pass direct proof map added | Partial |
| Public HTTP surface | `server.py`, `endpoints/` | endpoint inventory and weak contract suite added | Partial |
| Audience-doc generation pipeline | `KESTREL_FEATURES.md`, `scripts/generate_feature_docs.py` | dry-run proof and guardrail tests added | Partial |

The audit is not complete until every maintained claim in `KESTREL_FEATURES.md` has executable proof or an explicit open gap.

Feature-level direct/indirect/gap status now lives in [`FEATURE_PROOF_MATRIX.md`](./FEATURE_PROOF_MATRIX.md).
