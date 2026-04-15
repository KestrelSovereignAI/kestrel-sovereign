# Cross-Feature Seam Campaigns

This matrix tracks adversarial campaigns that cross feature boundaries. Component-local proof is tracked in `FEATURE_PROOF_MATRIX.md`; this file is for regressions that only appear when UI, auth, runtime, storage, LLM routing, tools, and deployment surfaces interact.

Status meanings:

- `Proven`: explicit scenarios, fixtures, invariants, and tests exist.
- `Partial`: some direct proof exists, but the campaign still has untested seams.
- `Planned`: campaign is defined but needs first-class proof.

| Campaign | Scenarios | Fixtures | Pass/Fail Invariants | Current Proof | Tracking | Status |
|---|---|---|---|---|---|---|
| Privacy transitions during active sessions | API and command privacy transitions; local-only mode entry/exit; voice/model side effects; isolated-session save | mocked agent, fake LLM service, command handler, privacy agent | all privacy transitions use async agent-level transition; no coroutine leaks; local-only modes force local providers; returning to cloud-allowed modes restores the resolved cloud route | `tests/unit/test_command_handler_privacy_contracts.py`, `tests/unit/test_command_handler_async_boundary_contracts.py`, `tests/unit/test_privacy_mode_model_restore_contracts.py` | #593 | Partial |
| Mandate and fallback routing across UI, API, command, and runtime | discovered default model ranking; explicit provider/model preference; UI endpoint rewrite; runtime provider fallback | model catalog fixtures, API client storage/fetch stubs, fake providers | no hardcoded current model names in generated docs; explicit preference wins; discovered newer provider models rank ahead of stale variants; browser rewrites do not route host-auth endpoints into agent space | `tests/unit/test_model_selection_contracts.py`, `tests/unit/test_generate_feature_docs.py`, `tests/unit/test_feature_doc_canonicality.py`, `tests/frontend/api_client.test.mjs` | #592 | Partial |
| Export/import with encryption, key rotation, and storage receipts | encrypted conversation preview; identity export/import; sovereignty export/import; key resolution | encrypted message rows, fake Fernet decryptor, identity/sovereignty fixtures | encrypted data is not silently downgraded to plaintext; failed decrypt remains visibly encrypted; export/import preserves identity and sovereignty metadata | `tests/unit/test_commands_conversations_endpoint_contracts.py`, `tests/integration/test_identity_export_import.py`, `tests/integration/test_sovereignty_e2e.py` | #595 | Partial |
| Permission bypass through MCP, A2A, tools, and commands | compute approval boundaries; A2A task state; MCP gateway/tool invocation; command-triggered privileged actions | fake task store, compute policy fixtures, MCP integration fixtures | privileged operations cannot bypass approval through alternate protocol paths; task status is auditable; command handlers preserve async approval boundaries | `tests/unit/test_security_feature.py`, `tests/unit/test_agent_runtime_endpoint_contracts.py`, `tests/integration/test_compute_security_integration.py`, `tests/integration/test_mcp_gateway.py` | #596 | Planned |
| Bootstrap/auth/session interactions in browser and SSE flows | localhost bootstrap; OAuth-required redirect; API key query auth limited to SSE; JWT expiry; stream retry after bootstrap refresh; host SSE EventSource routing | real FastAPI app with lifespan disabled, TestClient, browser API client fetch/storage stubs | bootstrap key stays localhost-only and disabled under OAuth-required mode; query-string API keys authenticate only SSE paths; JWT expiry clears platform tokens and redirects; API-key 401 refreshes once and retries both JSON and stream requests | `tests/unit/test_auth_decision_table.py`, `tests/unit/test_api_key_query_param.py`, `tests/unit/test_host_query_param_auth.py`, `tests/unit/test_sse_connection_limits.py`, `tests/frontend/api_client.test.mjs` | #255 | Proven |
| SQLite/PostgreSQL parity on storage, sync, and tasking | conversation/session queries; task store list/filter; sync webhook processing; DB table introspection | SQLite test storage, PostgreSQL integration fixture where available, fake task rows | storage APIs return the same semantic payload across backends; sync/webhook handlers do not block the event loop; task filters behave independently of backend | `tests/unit/test_agent_runtime_endpoint_contracts.py`, `tests/unit/test_commands_conversations_endpoint_contracts.py`, `docs/audit/SYNC_ASYNC_AUDIT.md` | #594 | Planned |
| Cloud/local drift for Ollama, RunPod, Vast.ai, GCP, Vertex, and Cloud Run | local provider discovery; cloud session lifecycle; Cloud Run deploy defaults; RunPod/Vast/GCP feature contracts | mocked cloud providers, profile fixtures, generated model cache | startup config is the pre-discovery source of truth; discovery updates cache without hardcoded model drift; cloud providers fail closed when credentials/runtime are missing; local training adapters do not block the event loop | `tests/unit/test_runpod_model_contracts.py`, `tests/unit/test_cloud_launcher_contracts.py`, `tests/unit/test_deploy_models.py`, `tests/unit/test_model_selection_contracts.py` | #597 | Partial |

## Campaign Rules

- Add a test before changing behavior unless the first change is documentation-only scenario definition.
- Prefer shared seam helpers over duplicated miniature middleware or fake clients when the real app/client can be exercised safely.
- Do not close a campaign because one happy path passes; each campaign needs adversarial inputs, negative cases, and explicit invariants.
- Track seam regressions separately from component-local issues so fixes do not disappear inside unrelated feature tickets.

## Next Proof Targets

1. Permission bypass through MCP, A2A, tools, and commands needs a direct adversarial suite that proves the same approval decision is enforced regardless of entrypoint.
2. SQLite/PostgreSQL parity needs a backend-switching fixture for conversation/session/tasking contracts, not just SQLite-local endpoint tests.
3. Privacy transitions need an active-stream/session test that proves runtime state cannot race model or storage privacy changes.
