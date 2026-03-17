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
