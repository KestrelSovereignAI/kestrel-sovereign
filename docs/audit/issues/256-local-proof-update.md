---
type: Issue Body
title: 256 Local Proof Update
description: Local audit proof is back in place after the power interruptions.
resource: /docs/audit/issues/256-local-proof-update.md
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

# 256 Local Proof Update

Local audit proof is back in place after the power interruptions.

Completed in the repo:

- rewrote `KESTREL_FEATURES.md` as the canonical maintained inventory
- added `docs/archive/KESTREL_FEATURES_legacy.md`
- added audit working papers:
  - `docs/audit/FEATURE_AUDIT_MATRIX.md`
  - `docs/audit/FEATURE_MODULE_MATRIX.md`
  - `docs/audit/API_ENDPOINT_MATRIX.md`
  - `docs/audit/AUTH_SURFACE_MATRIX.md`
  - reconciliation notes
- removed stale hardcoded count language from `scripts/generate_feature_docs.py`
- updated `README.md`, `docs/README.md`, and `scripts/convene_progress_review.py` to point at the canonical inventory
- fixed `/agent/stream` setup so client `HTTPException`s are preserved instead of rewritten to `500`
- added focused proof suites:
  - `tests/unit/test_auth_decision_table.py`
  - `tests/unit/test_endpoint_contract_suite.py`
  - `tests/unit/test_feature_doc_canonicality.py`
  - `tests/unit/test_generate_feature_docs.py`

Verification:

- `uv run pytest tests/unit/test_generate_feature_docs.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py -v`
- result: `19 passed`
- `uv run python scripts/generate_feature_docs.py --all --dry-run`
- result: dry-run succeeded for developer, user, and investor audiences

Remaining boundary:

- actual audience-doc generation is still environment-blocked until an LLM API key is present
