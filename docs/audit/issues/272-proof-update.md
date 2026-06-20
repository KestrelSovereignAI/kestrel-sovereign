---
type: Issue Body
title: 272 Proof Update
description: Auth decision-table proof is now in place.
resource: /docs/audit/issues/272-proof-update.md
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

# 272 Proof Update

Auth decision-table proof is now in place.

Added `tests/unit/test_auth_decision_table.py` covering:

- root-page browser behavior with and without required OAuth
- localhost-only bootstrap key behavior
- `/auth/me` semantic session requirement
- SSE query-param auth for `/agent/stream`
- rejection of query-param auth on non-SSE routes
- representative protected `/agent/*` route behavior
- multi-agent rewrite parity for `/api/agents/{name}/agent/info`

The suite exposed one real bug and the code is fixed:

- `endpoints/agent.py` now re-raises `HTTPException` during `/agent/stream` setup, instead of collapsing client `400` errors into `500`

Verification:

- `uv run pytest tests/unit/test_auth_decision_table.py -v`
- result: passed
