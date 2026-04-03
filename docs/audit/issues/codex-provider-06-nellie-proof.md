## Problem

The sovereign `codex_provider` work should not stop at “provider initializes.” Nellie needs a believable proof path showing that an actual agent can be pinned to the backend and produce stable responses.

## Goal

Add a small smoke-proof path for Nellie backend selection across `claude_max` and `codex`.

## Scope

- define a smoke test or operator checklist for pinning Nellie to:
  - `claude_max`
  - `codex`
- verify the persisted preference matches the active provider/model path
- ensure failure modes are legible when local auth/session is missing
- capture what counts as “good enough proof” before broader rollout

## Acceptance criteria

- there is a concrete Nellie-specific proof path
- operator-facing docs say what was tested and what was not
- backend identity and runtime reality are compared explicitly

## Proof results

### What was tested (20 automated tests)

All tests in `tests/unit/test_nellie_backend_smoke.py`:

**Nellie pinned to claude_max (6 tests):**
- Preference round-trip: set → get returns `claude_max` / `claude-sonnet-4-6`
- Routing returns ONLY `claude_max`, not `anthropic` (even with same model name)
- `get_active_model_id()` agrees with persisted preference
- Backend identity (preference) matches runtime reality (routing output)
- `claude_max` and `anthropic` are never confused despite identical model IDs
- Persistence callback fires on pin

**Nellie pinned to codex (4 tests):**
- Preference round-trip: set → get returns `codex` / `gpt-5.4`
- Routing returns ONLY `codex`, not `openai`
- `get_active_model_id()` agrees with persisted preference
- Backend identity matches runtime reality

**Failure modes (5 tests):**
- Pinning to codex when it isn't initialized → `LLMProviderUnavailableError` mentioning `codex`
- Pinning to claude_max when it isn't initialized → `LLMProviderUnavailableError` mentioning `claude_max`
- Error messages list all available providers (operator can see what IS working)
- `CodexAdapter.get_response(client=None)` → `RuntimeError` with clear message
- `ProviderRegistry._initialize_claude_max()` without `ANTHROPIC_AUTH_TOKEN` → `ValueError` with setup instructions

**Backend switching (5 tests):**
- Switch claude_max → codex at runtime: routing follows
- Switch codex → claude_max at runtime: routing follows
- Clear preference: all providers become available again
- Identity stays consistent through all three backends (claude_max, codex, openai)

### What was NOT tested

- **Live API calls**: No actual LLM inference. These tests verify routing and preference mechanics, not that the remote API responds. Live proof requires valid auth tokens.
- **Database persistence**: The persistence callback is verified to fire, but actual database round-trip (write → restart → read) is not covered here. That's integration-test territory.
- **Streaming**: Streaming paths use the same routing, but are not separately exercised in this smoke proof.
- **Constitutional awareness**: Nellie's constitutional compliance is not tested here — that's the constitution-verifier's job.

### Operator checklist for manual verification

1. **claude_max**: Set `ANTHROPIC_AUTH_TOKEN` (from `claude login`), configure `claude_max` in `llm_config.toml`, pin Nellie with `!model-set claude_max/claude-sonnet-4-6`, send a message, verify response arrives.
2. **codex**: Set `CODEX_AUTH_TOKEN` (from `codex login`), configure `codex` in `llm_config.toml`, pin Nellie with `!model-set codex/gpt-5.4`, send a message, verify response arrives via Responses API.
3. **Failure check**: Remove the auth token and restart. Verify the provider fails to initialize with a clear error message, not a silent fallback.

### Good enough proof

This smoke proof establishes that the routing layer is honest: what the operator sets is what the system uses, and failures are visible. Live inference proof requires auth tokens and is left to the operator checklist above.

## References

- `agent_data/nellie/SOUL.md`
- `tests/unit/test_nellie_backend_smoke.py`
- `kestrel_sovereign/llm/provider_registry.py`
- `kestrel_sovereign/docs/audit/issues/codex-provider-04-smoke-proof.md`
