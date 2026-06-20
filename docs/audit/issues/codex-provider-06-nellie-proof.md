---
type: Issue Body
title: Codex Provider 06 Nellie Proof
description: The sovereign plan-provider work should not stop at “provider initializes.”
  Nellie needs a believable proof path showing that an actual agent can be pinned
  to the backend and pr...
resource: /docs/audit/issues/codex-provider-06-nellie-proof.md
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

# Codex Provider 06 Nellie Proof

## Problem

The sovereign plan-provider work should not stop at “provider initializes.” Nellie needs a believable proof path showing that an actual agent can be pinned to the backend and produce stable responses.

## Goal

Add a small smoke-proof path for Nellie backend selection across `claude_plan` and `openai_plan`.

## Scope

- define a smoke test or operator checklist for pinning Nellie to:
  - `claude_plan`
  - `openai_plan`
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

**Nellie pinned to claude_plan (6 tests):**
- Preference round-trip: set → get returns `claude_plan` / `claude-sonnet-4-6`
- Routing returns ONLY `claude_plan`, not `anthropic` (even with same model name)
- `get_active_model_id()` agrees with persisted preference
- Backend identity (preference) matches runtime reality (routing output)
- `claude_plan` and `anthropic` are never confused despite identical model IDs
- Persistence callback fires on pin

**Nellie pinned to openai_plan (4 tests):**
- Preference round-trip: set → get returns `openai_plan` / `gpt-5.4`
- Routing returns ONLY `openai_plan`, not `openai`
- `get_active_model_id()` agrees with persisted preference
- Backend identity matches runtime reality

**Failure modes (5 tests):**
- Pinning to openai_plan when it isn't initialized → `LLMProviderUnavailableError` mentioning `openai_plan`
- Pinning to claude_plan when it isn't initialized → `LLMProviderUnavailableError` mentioning `claude_plan`
- Error messages list all available providers (operator can see what IS working)
- `CodexAdapter.get_response(client=None)` → `RuntimeError` with clear message
- `ProviderRegistry._initialize_claude_plan()` without `ANTHROPIC_AUTH_TOKEN` → `ValueError` with setup instructions

**Backend switching (5 tests):**
- Switch claude_plan → openai_plan at runtime: routing follows
- Switch openai_plan → claude_plan at runtime: routing follows
- Clear preference: all providers become available again
- Identity stays consistent through all three backends (claude_plan, openai_plan, openai)

### What was NOT tested

- **Live API calls**: No actual LLM inference. These tests verify routing and preference mechanics, not that the remote API responds. Live proof requires valid auth tokens.
- **Database persistence**: The persistence callback is verified to fire, but actual database round-trip (write → restart → read) is not covered here. That's integration-test territory.
- **Streaming**: Streaming paths use the same routing, but are not separately exercised in this smoke proof.
- **Constitutional awareness**: Nellie's constitutional compliance is not tested here — that's the constitution-verifier's job.

### Operator checklist for manual verification

1. **claude_plan**: Set `ANTHROPIC_AUTH_TOKEN` (from `claude login`), configure `claude_plan` in `llm_config.toml`, pin Nellie with `!model-set claude_plan/claude-sonnet-4-6`, send a message, verify response arrives.
2. **openai_plan**: Set `CODEX_AUTH_TOKEN` (from `codex login`), configure `openai_plan` in `llm_config.toml`, pin Nellie with `!model-set openai_plan/gpt-5.4`, send a message, verify response arrives via Responses API.
3. **Failure check**: Remove the auth token and restart. Verify the provider fails to initialize with a clear error message, not a silent fallback.

### Good enough proof

This smoke proof establishes that the routing layer is honest: what the operator sets is what the system uses, and failures are visible. Live inference proof requires auth tokens and is left to the operator checklist above.

## References

- `agent_data/nellie/SOUL.md`
- `tests/unit/test_nellie_backend_smoke.py`
- `kestrel_sovereign/llm/provider_registry.py`
- `kestrel_sovereign/docs/audit/issues/codex-provider-04-smoke-proof.md`
