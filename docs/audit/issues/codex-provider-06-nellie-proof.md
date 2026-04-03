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

## References

- `agent_data/nellie/SOUL.md`
- `kestrel_sovereign/llm/provider_registry.py`
- `kestrel_sovereign/docs/audit/issues/codex-provider-04-smoke-proof.md`
