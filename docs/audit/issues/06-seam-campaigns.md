## Problem

Most expensive regressions do not live inside a single component. They appear when state crosses boundaries between UI, API, services, storage, LLM adapters, or background workers.

## Goal

Create dedicated seam campaigns that red-team cross-domain behavior rather than isolated modules.

## Campaigns

- privacy transitions during active sessions
- mandate and fallback routing across UI, API, command, and runtime
- export/import with encryption, key rotation, and storage receipts
- permission bypass attempts through MCP, A2A, tools, and commands
- bootstrap/auth/session interactions in browser and SSE flows
- SQLite/PostgreSQL parity on storage, sync, and tasking
- cloud/local drift for Ollama, RunPod, Vast.ai, GCP, Vertex, and Cloud Run

Canonical campaign tracking lives in `docs/audit/SEAM_CAMPAIGNS.md`.

Current campaign status:

- `Proven`: bootstrap/auth/session interactions in browser and SSE flows.
- `Partial`: privacy transitions (#593), mandate/fallback routing (#592), export/import/encryption (#595), cloud/local drift (#597).
- `Planned`: permission bypass through MCP/A2A/tools/commands (#596), SQLite/PostgreSQL storage/sync/tasking parity (#594).

## Required Proof

- integration and adversarial-first coverage
- e2e where user-visible
- load or concurrency tests where race conditions are plausible

## Exit Criteria

- each campaign has explicit scenarios, fixtures, and pass/fail invariants
- seam regressions are tracked independently from component-local tickets
